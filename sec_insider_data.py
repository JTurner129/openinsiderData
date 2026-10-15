"""Build the OpenInsider-style purchase dataset from SEC EDGAR bulk Form 4 data.

OpenInsider's screener tables are built from SEC Form 4 filings. This script
downloads the SEC's quarterly "Insider Transactions Data Sets" (the same
filings, as structured TSV files) and writes one row per filing in the same
columns openinsider_scraper.py produces, so the rest of the pipeline does not
care which source the rows came from.

It reads the same config.yaml keys as the scraper (output, scraping, filters,
cache, logging) plus the `sec` section.

Re-running is safe: each finished quarter is cached as Parquet under
cache.directory/sec and skipped next time, so an interrupted run resumes
where it stopped.
"""
import csv
import io
import json
import logging
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

LOG = logging.getLogger("sec_insider_data")

# Same column names and order as openinsider_scraper.py, then SEC identifiers.
SCRAPER_COLUMNS = ['transaction_date', 'trade_date', 'ticker', 'company_name',
                   'owner_name', 'Title', 'transaction_type', 'last_price',
                   'Qty', 'shares_held', 'Owned', 'Value']
EXTRA_COLUMNS = ['filing_date', 'accession_number', 'issuer_cik', 'owner_cik',
                 'n_reporting_owners', 'is_director', 'is_officer',
                 'is_ten_pct_owner', 'officer_title', 'source_quarter']

RELATIONSHIP_ABBR = {'Director': 'Dir', 'TenPercentOwner': '10%', 'Other': 'Other'}


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


class RateLimiter:
    """Spaces requests across all threads to at most `per_second`."""

    def __init__(self, per_second):
        self.min_interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            sleep_for = self.last + self.min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.last = time.monotonic()


class SecClient:
    def __init__(self, cfg):
        sec = cfg['sec']
        self.user_agent = os.environ.get('SEC_USER_AGENT', sec['user_agent'])
        self.timeout = cfg['scraping']['timeout']
        self.retries = cfg['scraping']['retry_attempts']
        self.delay = cfg['scraping'].get('request_delay', 1.0)
        self.limiter = RateLimiter(sec.get('max_requests_per_second', 5))
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent,
                                     'Accept-Encoding': 'gzip, deflate'})

    def get(self, url, stream=False):
        last_error = None
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=self.timeout, stream=stream)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                time.sleep(self.delay)
                return r
            except requests.RequestException as e:
                last_error = e
                wait = self.delay * (2 ** attempt) * 2
                LOG.warning(f"{url}: {e} (attempt {attempt + 1}); retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def list_quarters(client, page_url):
    """Return {'2006q1': absolute_zip_url, ...} from the SEC data sets page."""
    html = client.get(page_url).text
    links = re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html)
    quarters = {}
    for href, q in links:
        quarters[q] = href if href.startswith('http') else 'https://www.sec.gov' + href
    if not quarters:
        raise RuntimeError("No quarterly data set links found on the SEC page")
    return dict(sorted(quarters.items()))


def read_tsv(zf, name, usecols):
    with zf.open(name) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding='utf-8', errors='replace'),
                           sep='\t', dtype=str, usecols=usecols,
                           quoting=csv.QUOTE_NONE, on_bad_lines='warn',
                           keep_default_na=False)


def clean_ticker(raw):
    if not isinstance(raw, str):
        return None
    t = raw.strip().upper()
    t = re.split(r'[,;/\s]+', t)[0] if t else t   # "ABC, ABCD" -> "ABC"
    t = re.sub(r'^(NYSE|NASDAQ|AMEX|OTC|OTCBB|NYSEMKT|NYSEAMERICAN)[:.\-]', '', t)
    t = t.strip('."\'()[]')
    if t in ('', 'NONE', 'N/A', 'NA', 'NULL', '-', '--', 'NOTAPPLICABLE') or not re.search(r'[A-Z]', t):
        return None
    return t


def build_title(relationship, officer_title, other_text):
    parts = []
    rel = [r.strip() for r in (relationship or '').split(',') if r.strip()]
    for r in rel:
        if r in RELATIONSHIP_ABBR:
            parts.append(RELATIONSHIP_ABBR[r])
    if officer_title and officer_title.strip():
        parts.append(officer_title.strip())
    elif 'Officer' in rel:
        parts.append('Officer')
    if other_text and other_text.strip() and 'Other' in rel:
        parts[parts.index('Other')] = other_text.strip()
    return ', '.join(parts).replace('""', '"').strip('"').strip()


def process_quarter_zip(zip_path, quarter, cfg):
    """Turn one SEC quarterly zip into OpenInsider-style rows (one per filing)."""
    types = cfg['filters']['transaction_types'] or None
    form_types = set(cfg['sec'].get('form_types', ['4']))
    with zipfile.ZipFile(zip_path) as zf:
        sub = read_tsv(zf, 'SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE',
                                              'ISSUERCIK', 'ISSUERNAME', 'ISSUERTRADINGSYMBOL'])
        tr = read_tsv(zf, 'NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'NONDERIV_TRANS_SK', 'TRANS_DATE',
                                                 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
                                                 'TRANS_ACQUIRED_DISP_CD', 'SHRS_OWND_FOLWNG_TRANS'])
        own = read_tsv(zf, 'REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNERNAME',
                                                  'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE', 'RPTOWNER_TXT'])

    sub = sub[sub['DOCUMENT_TYPE'].isin(form_types)]
    if types:
        tr = tr[tr['TRANS_CODE'].isin(types)]
    # A purchase must be an acquisition; guards against miscoded rows.
    tr = tr[tr['TRANS_ACQUIRED_DISP_CD'] == 'A']
    tr = tr[tr['ACCESSION_NUMBER'].isin(sub['ACCESSION_NUMBER'])]
    if tr.empty:
        return pd.DataFrame(columns=SCRAPER_COLUMNS + EXTRA_COLUMNS)

    tr = tr.copy()
    tr['shares'] = pd.to_numeric(tr['TRANS_SHARES'], errors='coerce').fillna(0.0)
    tr['price'] = pd.to_numeric(tr['TRANS_PRICEPERSHARE'], errors='coerce')
    tr['value'] = tr['shares'] * tr['price'].fillna(0.0)
    tr['held_after'] = pd.to_numeric(tr['SHRS_OWND_FOLWNG_TRANS'], errors='coerce')
    tr['tdate'] = pd.to_datetime(tr['TRANS_DATE'], format='%d-%b-%Y', errors='coerce')
    tr['sk'] = pd.to_numeric(tr['NONDERIV_TRANS_SK'], errors='coerce')
    tr = tr.sort_values(['ACCESSION_NUMBER', 'sk'])

    # One row per filing, like OpenInsider: summed quantity and value,
    # value-weighted price, earliest trade date, holdings after the last line.
    g = tr.groupby('ACCESSION_NUMBER')
    agg = pd.DataFrame({
        'trade_date': g['tdate'].min(),
        'Qty': g['shares'].sum(),
        'Value': g['value'].sum(),
        'shares_held': g['held_after'].last(),
        'transaction_type': g['TRANS_CODE'].first(),
    })
    agg['last_price'] = (agg['Value'] / agg['Qty']).where(agg['Qty'] > 0)
    prior = agg['shares_held'] - agg['Qty']
    agg['Owned'] = (agg['Qty'] / prior).where(prior > 0)   # NaN = new position, like OpenInsider's "New"

    # First listed reporting owner is the primary filer; joint filers
    # (e.g. a person and their LLC) are one insider, not several.
    own = own.copy()
    own['owner_order'] = own.groupby('ACCESSION_NUMBER').cumcount()
    n_owners = own.groupby('ACCESSION_NUMBER').size().rename('n_reporting_owners')
    first = own[own['owner_order'] == 0].set_index('ACCESSION_NUMBER')
    rels = first['RPTOWNER_RELATIONSHIP'].fillna('')

    out = agg.join(sub.set_index('ACCESSION_NUMBER')).join(first).join(n_owners)
    out = out.reset_index().rename(columns={'ACCESSION_NUMBER': 'accession_number'})
    out['filing_date'] = pd.to_datetime(out['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    out['transaction_date'] = out['filing_date']   # the scraper stores the filing date here
    out['ticker'] = out['ISSUERTRADINGSYMBOL'].map(clean_ticker)
    out['company_name'] = out['ISSUERNAME'].str.strip()
    out['owner_name'] = out['RPTOWNERNAME'].str.strip()
    out['issuer_cik'] = pd.to_numeric(out['ISSUERCIK'], errors='coerce').astype('Int64')
    out['owner_cik'] = pd.to_numeric(out['RPTOWNERCIK'], errors='coerce').astype('Int64')
    rel = out['accession_number'].map(rels).fillna('')
    out['is_director'] = rel.str.contains('Director')
    out['is_officer'] = rel.str.contains('Officer')
    out['is_ten_pct_owner'] = rel.str.contains('TenPercentOwner')
    out['officer_title'] = out['RPTOWNER_TITLE'].fillna('').str.replace('""', '"').str.strip('" ')
    out['Title'] = [build_title(r, t, x) for r, t, x in
                    zip(rel, out['RPTOWNER_TITLE'].fillna(''), out['RPTOWNER_TXT'].fillna(''))]
    out['n_reporting_owners'] = out['n_reporting_owners'].fillna(0).astype(int)
    out['source_quarter'] = quarter
    return out[SCRAPER_COLUMNS + EXTRA_COLUMNS]


def apply_filters(df, cfg):
    f = cfg['filters']
    keep = df['Value'].abs() >= f['min_transaction_value']
    keep &= df['Qty'].abs() >= f['min_shares_traded']
    if f['exclude_companies']:
        keep &= ~df['ticker'].isin(f['exclude_companies'])
    if f['include_companies']:
        keep &= df['ticker'].isin(f['include_companies'])
    return df[keep]


def quarter_is_closed(q):
    year, qn = int(q[:4]), int(q[5])
    end_month = qn * 3
    now = datetime.now()
    return (now.year, now.month) > (year, end_month) and \
        (now - datetime(year, end_month, 1)).days > 45


def fetch_quarter(client, quarter, url, cache_dir, cfg):
    parquet_path = cache_dir / f"{quarter}.parquet"
    max_age_h = cfg['cache']['max_age']
    if cfg['cache']['enabled'] and parquet_path.exists():
        age_h = (time.time() - parquet_path.stat().st_mtime) / 3600
        if quarter_is_closed(quarter) or age_h < max_age_h:
            return quarter, pd.read_parquet(parquet_path), 'cached'

    zip_path = cache_dir / f"{quarter}.zip"
    tmp_path = zip_path.with_suffix('.zip.part')
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        r = client.get(url, stream=True)
        with open(tmp_path, 'wb') as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        if not zipfile.is_zipfile(tmp_path):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"{quarter}: download is not a valid zip")
        tmp_path.rename(zip_path)

    df = process_quarter_zip(zip_path, quarter, cfg)
    df.to_parquet(parquet_path, index=False)
    zip_path.unlink()   # the processed Parquet is all we need; saves ~10 MB per quarter
    return quarter, df, 'downloaded'


def main(config_path='config.yaml'):
    cfg = load_config(config_path)
    logging.basicConfig(level=getattr(logging, cfg['logging']['level'].upper()),
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler('sec_insider_data.log')])
    out_dir = Path(cfg['output']['directory'])
    cache_dir = Path(cfg['cache']['directory']) / 'sec'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = SecClient(cfg)
    quarters = list_quarters(client, cfg['sec']['datasets_page'])
    start = f"{cfg['scraping']['start_year']}q{(cfg['scraping']['start_month'] - 1) // 3 + 1}"
    wanted = {q: u for q, u in quarters.items() if q >= start}
    LOG.info(f"SEC lists {len(quarters)} quarters ({min(quarters)}..{max(quarters)}); "
             f"fetching {len(wanted)} from {start}")

    frames, failed = [], {}
    with ThreadPoolExecutor(max_workers=cfg['scraping']['max_workers']) as ex:
        futures = {ex.submit(fetch_quarter, client, q, u, cache_dir, cfg): q for q, u in wanted.items()}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                _, df, how = fut.result()
                frames.append(df)
                LOG.info(f"{q}: {len(df):,} purchase filings ({how})")
            except Exception as e:
                failed[q] = str(e)
                LOG.error(f"{q} failed: {e}")

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCRAPER_COLUMNS + EXTRA_COLUMNS)
    data = apply_filters(data, cfg)
    data = data.drop_duplicates('accession_number').sort_values(['filing_date', 'accession_number'])
    data = data.reset_index(drop=True)

    out_path = out_dir / cfg['output']['filename']
    if cfg['output']['format'].lower() == 'parquet':
        data.to_parquet(out_path, index=False)
    else:
        data.to_csv(out_path, index=False)
    data.to_csv(out_path.with_suffix('.csv.gz'), index=False)   # gzipped to stay under GitHub's file limit

    # Coverage report: every calendar month in range should have purchases.
    months = data['filing_date'].dt.to_period('M').value_counts().sort_index()
    full_range = pd.period_range(months.index.min(), months.index.max(), freq='M') if len(months) else []
    empty_months = [str(m) for m in full_range if m not in months.index]
    summary = {
        'source': 'SEC EDGAR Insider Transactions Data Sets (Form 4)',
        'quarters_available': [min(quarters), max(quarters)],
        'earliest_filing_date': str(data['filing_date'].min().date()) if len(data) else None,
        'latest_filing_date': str(data['filing_date'].max().date()) if len(data) else None,
        'rows': int(len(data)),
        'rows_without_ticker': int(data['ticker'].isna().sum()),
        'rows_per_year': {str(k): int(v) for k, v in data['filing_date'].dt.year.value_counts().sort_index().items()},
        'failed_quarters': failed,
        'empty_months': empty_months,
        'months_lowest_counts': {str(k): int(v) for k, v in months.nsmallest(5).items()},
    }
    with open(out_dir / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    LOG.info(json.dumps(summary, indent=2))
    if failed:
        LOG.error(f"{len(failed)} quarter(s) failed; re-run to retry them (finished quarters are cached).")
    return summary


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
