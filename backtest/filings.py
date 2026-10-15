"""Phase 6d: tests built from other SEC filings (insider sales, Form 4 footnotes, 8-Ks).

Rules were written down and committed before any of this data was looked at.
Each test is a yes/no filter on the cluster (>= 3 insiders) and top-executive
(>= $200K) screens, evaluated exactly like the Phase 6 filters: realistic
trading (next-open entry, buy-and-hold weights, estimated spread costs),
FF5+UMD+IWC alpha, chosen on 2006-2015 and checked on 2016+. A filter "holds"
only if it beats the unfiltered baseline in both periods with test t >= 1.96.
Holm correction is applied across the primary tests only.

Primary tests (the only ones that can feed the recommended method):
  P1 No insider selling: drop signals where any insider of the company filed
     an open-market sale (code S) worth >= $100K in the 90 days up to and
     including the signal date.
  P2 Discretionary buys only: drop purchase filings whose footnotes say the
     buy was not a free choice (10b5-1 plan, dividend reinvestment, employee
     stock purchase plan, IPO / public offering / private placement /
     subscription / rights offering allocation) or that tick the 10b5-1 box
     (forms since 2023). Keep a signal only if its remaining filings still
     meet the screen (>= 3 insiders and >= $200K for clusters; a top executive
     filing >= $200K for executive buys).
  P3 Not right after earnings: drop signals where the company filed an 8-K
     with Item 2.02 (results of operations) in the 30 days before the signal.

Exploratory tests (reported, never used to pick the method):
  E1 8-K Item 5.02 (officer or director change) in the prior 90 days (keep only these)
  E2 No 8-K financing items (1.01 material agreement, 2.03 new debt,
     3.02 unregistered equity sale) in the prior 30 days
  E3 No 8-K at all in the prior 30 days ("out of nowhere")
  E4 Most of the buying held directly (not via trusts, funds or family)
  E5 Net buyer: insider purchases in the 90 days exceed insider sales
  E6 Short interest >= 10% of shares outstanding at the last report before
     the signal (only if Compustat short interest data is available)
"""
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table
from .filters import evaluate, TRAIN, TEST

LOG = get_logger('filings')
EXTRA = CACHE / 'sec_extra'
SUBS = CACHE / 'sec_submissions'
SCREENS_TESTED = ['cluster3', 'exec200k']
HOLDS = {'6m': 126, '1y': 252}
SALE_MIN, SALE_DAYS = 100_000, 90

NON_DISCRETIONARY = re.compile(
    r'10b5-?1|10b-5-?1|dividend reinvest|\bdrip\b|employee stock purchase|\bespp\b|'
    r'initial public offering|\bipo\b|public offering|private placement|subscription agreement|'
    r'rights offering|underwrit', re.I)
NEGATED = re.compile(r'\bnot\b[^.]{0,40}(10b5|plan)', re.I)


# --- Quarterly data sets: sales, footnotes, ownership form --------------------
def _client(cfg):
    s = requests.Session()
    s.headers.update({'User-Agent': cfg['sec']['user_agent'], 'Accept-Encoding': 'gzip, deflate'})
    return s


def _get(session, url, tries=6):
    for i in range(tries):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f'HTTP {r.status_code}')
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = 5 * 2 ** i
            LOG.warning(f'{url}: {e}; retry in {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'failed: {url}')


def _tsv(zf, name, cols):
    with zf.open(name) as f:
        head = f.readline().decode('utf-8', 'replace').rstrip('\r\n').split('\t')
    use = [c for c in cols if c in head]
    with zf.open(name) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding='utf-8', errors='replace'), sep='\t', dtype=str,
                           usecols=use, quoting=3, on_bad_lines='skip', keep_default_na=False)


def process_zip(path, purchase_acc):
    with zipfile.ZipFile(path) as zf:
        sub = _tsv(zf, 'SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE', 'ISSUERCIK', 'AFF10B5ONE'])
        tr = _tsv(zf, 'NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
                                             'TRANS_ACQUIRED_DISP_CD', 'DIRECT_INDIRECT_OWNERSHIP'])
        own = _tsv(zf, 'REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNERCIK'])
        names = zf.namelist()
        fn = _tsv(zf, 'FOOTNOTES.tsv', ['ACCESSION_NUMBER', 'FOOTNOTE_TXT']) if 'FOOTNOTES.tsv' in names else \
            pd.DataFrame(columns=['ACCESSION_NUMBER', 'FOOTNOTE_TXT'])
    sub = sub[sub['DOCUMENT_TYPE'] == '4']
    tr = tr[tr['ACCESSION_NUMBER'].isin(sub['ACCESSION_NUMBER'])].copy()
    tr['value'] = pd.to_numeric(tr['TRANS_SHARES'], errors='coerce').fillna(0) * \
        pd.to_numeric(tr['TRANS_PRICEPERSHARE'], errors='coerce').fillna(0)

    # Open-market sales, one row per filing
    s = tr[(tr['TRANS_CODE'] == 'S') & (tr['TRANS_ACQUIRED_DISP_CD'] == 'D')]
    s = s.groupby('ACCESSION_NUMBER')['value'].sum().rename('sale_value').reset_index()
    first_owner = own.drop_duplicates('ACCESSION_NUMBER').set_index('ACCESSION_NUMBER')['RPTOWNERCIK']
    s = s.merge(sub[['ACCESSION_NUMBER', 'FILING_DATE', 'ISSUERCIK']], on='ACCESSION_NUMBER')
    s['owner_cik'] = s['ACCESSION_NUMBER'].map(first_owner)
    sales = pd.DataFrame({'accession_number': s['ACCESSION_NUMBER'],
                          'filing_date': pd.to_datetime(s['FILING_DATE'], format='%d-%b-%Y', errors='coerce'),
                          'issuer_cik': pd.to_numeric(s['ISSUERCIK'], errors='coerce'),
                          'owner_cik': pd.to_numeric(s['owner_cik'], errors='coerce'),
                          'sale_value': s['sale_value']})

    # Purchase filings: footnote flags, 10b5-1 box, share held directly
    p = tr[(tr['TRANS_CODE'] == 'P') & tr['ACCESSION_NUMBER'].isin(purchase_acc)]
    direct = (p.assign(d=p['value'] * (p['DIRECT_INDIRECT_OWNERSHIP'] == 'D'))
              .groupby('ACCESSION_NUMBER')[['d', 'value']].sum())
    direct_share = (direct['d'] / direct['value']).where(direct['value'] > 0)
    fn = fn[fn['ACCESSION_NUMBER'].isin(purchase_acc)]
    txt = fn.groupby('ACCESSION_NUMBER')['FOOTNOTE_TXT'].apply(lambda x: ' '.join(x))
    flag = txt.map(lambda t: bool(NON_DISCRETIONARY.search(t)) and not bool(NEGATED.search(t)) or
                   bool(re.search(r'dividend reinvest|employee stock purchase|public offering|private placement', t, re.I)))
    box = sub.set_index('ACCESSION_NUMBER')['AFF10B5ONE'] if 'AFF10B5ONE' in sub else pd.Series(dtype=str)
    acc = sorted(set(p['ACCESSION_NUMBER']))
    pf = pd.DataFrame({'accession_number': acc})
    pf['footnote_flag'] = pf['accession_number'].map(flag).fillna(False).astype(bool)
    pf['plan_box'] = pf['accession_number'].map(box).isin(['1', 'true', 'True', 'TRUE'])
    pf['direct_share'] = pf['accession_number'].map(direct_share)
    pf['non_discretionary'] = pf['footnote_flag'] | pf['plan_box']
    return sales, pf


def fetch_quarter_extras(cfg, quarter, url, purchase_acc, session):
    out = EXTRA / f'{quarter}.parquet'
    out_p = EXTRA / f'{quarter}_purch.parquet'
    if out.exists() and out_p.exists():
        return
    zp = EXTRA / f'{quarter}.zip'
    if not (zp.exists() and zipfile.is_zipfile(zp)):
        r = _get(session, url)
        zp.write_bytes(r.content)
    sales, pf = process_zip(zp, purchase_acc)
    sales.to_parquet(out, index=False)
    pf.to_parquet(out_p, index=False)
    zp.unlink()
    LOG.info(f'{quarter}: {len(sales):,} sale filings, {len(pf):,} purchase filings, '
             f'{int(pf["non_discretionary"].sum()):,} flagged non-discretionary')


def download_extras(cfg):
    import sys
    sys.path.insert(0, str(CACHE.parent))
    from sec_insider_data import SecClient, list_quarters
    EXTRA.mkdir(parents=True, exist_ok=True)
    quarters = list_quarters(SecClient(cfg), cfg['sec']['datasets_page'])
    purchase_acc = set(pd.read_parquet(DATA / 'insider_trades.parquet', columns=['accession_number'])['accession_number'])
    session = _client(cfg)
    for q, url in quarters.items():
        for attempt in range(3):
            try:
                fetch_quarter_extras(cfg, q, url, purchase_acc, session)
                break
            except Exception as e:   # one bad quarter should not stop the rest
                LOG.warning(f'{q}: {e} (attempt {attempt + 1})')
                (EXTRA / f'{q}.zip').unlink(missing_ok=True)
                time.sleep(30)
        time.sleep(1)
    sales = pd.concat([pd.read_parquet(p) for p in sorted(EXTRA.glob('*q?.parquet'))], ignore_index=True)
    purch = pd.concat([pd.read_parquet(p) for p in sorted(EXTRA.glob('*_purch.parquet'))], ignore_index=True)
    return sales.drop_duplicates('accession_number'), purch.drop_duplicates('accession_number')


# --- 8-K item codes from each company's filing history ------------------------
def _sub_json(session, cik):
    path = SUBS / f'{cik}.json'
    if path.exists():
        return json.loads(path.read_text())
    r = _get(session, f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json')
    if r is None:
        rows = []
    else:
        j = r.json()
        parts = [j['filings']['recent']]
        for f in j['filings'].get('files', []):
            rr = _get(session, f'https://data.sec.gov/submissions/{f["name"]}')
            if rr is not None:
                parts.append(rr.json())
            time.sleep(0.15)
        rows = []
        for p in parts:
            for form, d, items in zip(p.get('form', []), p.get('filingDate', []), p.get('items', [''] * len(p.get('form', [])))):
                if form in ('8-K', '8-K/A'):
                    rows.append([d, items or ''])
    path.write_text(json.dumps(rows))
    time.sleep(0.15)
    return rows


def download_8k(cfg, ciks):
    SUBS.mkdir(parents=True, exist_ok=True)
    session = _client(cfg)
    out = []
    todo = sorted(set(int(c) for c in ciks))
    with ThreadPoolExecutor(4) as ex:
        for i, (cik, rows) in enumerate(zip(todo, ex.map(lambda c: _sub_json(session, c), todo))):
            out += [(cik, d, it) for d, it in rows]
            if i % 500 == 0:
                LOG.info(f'8-K histories: {i:,}/{len(todo):,} companies')
    k = pd.DataFrame(out, columns=['issuer_cik', 'filing_date', 'items'])
    k['filing_date'] = pd.to_datetime(k['filing_date'], errors='coerce')
    return k.dropna(subset=['filing_date'])


# --- Features per signal ------------------------------------------------------
def _window_any(sig, events, days, inclusive_end):
    """For each signal, does the issuer have an event in (signal-days, signal] (or [.., signal) if not inclusive)?"""
    ev = events.groupby('issuer_cik')['filing_date'].apply(lambda s: np.sort(s.values.astype('datetime64[D]')))
    res = np.zeros(len(sig), bool)
    for i, (cik, d) in enumerate(zip(sig['issuer_cik'].values, sig['signal_date'].values.astype('datetime64[D]'))):
        arr = ev.get(cik)
        if arr is None:
            continue
        lo = np.searchsorted(arr, d - np.timedelta64(days, 'D'), 'right')
        hi = np.searchsorted(arr, d, 'right' if inclusive_end else 'left')
        res[i] = hi > lo
    return res


def _window_sum(sig, events, col, days):
    g = {c: (np.sort(x['filing_date'].values.astype('datetime64[D]')),
             x.sort_values('filing_date')[col].values) for c, x in events.groupby('issuer_cik')}
    res = np.zeros(len(sig))
    for i, (cik, d) in enumerate(zip(sig['issuer_cik'].values, sig['signal_date'].values.astype('datetime64[D]'))):
        if cik not in g:
            continue
        arr, v = g[cik]
        lo = np.searchsorted(arr, d - np.timedelta64(days, 'D'), 'right')
        hi = np.searchsorted(arr, d, 'right')
        res[i] = v[lo:hi].sum()
    return res


def build(cfg, F, sales, purch, k8, shortint=None):
    bt = cfg['backtest']
    variants = pd.read_parquet(DATA / 'signals_all_variants.parquet',
                               columns=['screen', 'dedupe_days', 'signal_date', 'issuer_cik', 'accessions'])
    variants = variants[variants['dedupe_days'] == bt['dedupe_days']]
    F = F.merge(variants.drop(columns='dedupe_days'), on=['screen', 'signal_date', 'issuer_cik'], how='left')
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet',
                         columns=['accession_number', 'insider_id', 'value_clean', 'is_top_exec'])
    pc = pc.merge(purch, on='accession_number', how='left')
    pc['non_discretionary'] = pc['non_discretionary'].fillna(False).astype(bool)
    pc = pc.set_index('accession_number')

    # P2: does the signal still qualify with only discretionary filings?
    keep, nd_share, direct = [], [], []
    for scr, accs in zip(F['screen'], F['accessions'].fillna('')):
        ids = [a for a in accs.split(',') if a in pc.index]
        w = pc.loc[ids]
        d = w[~w['non_discretionary']]
        if scr.startswith('cluster'):
            ok = d['insider_id'].nunique() >= 3 and d['value_clean'].sum() >= bt['cluster_min_value']
        else:
            ok = bool(((d['is_top_exec']) & (d['value_clean'] >= 200_000)).any())
        keep.append(bool(ok) if len(w) else True)
        nd_share.append(w['non_discretionary'].mean() if len(w) else np.nan)
        v = w['value_clean'].fillna(0)
        direct.append((w['direct_share'].fillna(1) * v).sum() / v.sum() if v.sum() > 0 else np.nan)
    F['discretionary_qualifies'] = keep
    F['non_discretionary_share'] = nd_share
    F['direct_share'] = direct

    big_sales = sales[sales['sale_value'] >= SALE_MIN]
    F['insider_sale_90d'] = _window_any(F, big_sales, SALE_DAYS, True)
    F['sales_90d'] = _window_sum(F, sales, 'sale_value', SALE_DAYS)
    pv = pc.reset_index().merge(pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['accession_number', 'issuer_cik', 'filing_date']),
                                on='accession_number')
    F['buys_90d'] = _window_sum(F, pv[['issuer_cik', 'filing_date', 'value_clean']].fillna(0), 'value_clean', SALE_DAYS)

    def items(pat):
        return k8[k8['items'].str.contains(pat, regex=True)]
    F['earnings_8k_30d'] = _window_any(F, items(r'2\.02'), 30, False)
    F['exec_change_8k_90d'] = _window_any(F, items(r'5\.02'), 90, False)
    F['financing_8k_30d'] = _window_any(F, items(r'1\.01|2\.03|3\.02'), 30, False)
    F['any_8k_30d'] = _window_any(F, k8, 30, False)
    F['has_8k_history'] = F['issuer_cik'].isin(k8['issuer_cik'])

    if shortint is not None:
        F = F.merge(shortint, on=['screen', 'signal_date', 'issuer_cik'], how='left')
    return F


def short_interest_feature(cfg, F):
    """Short interest / shares outstanding at the last report on or before the signal (Compustat only)."""
    from pathlib import Path
    from .common import ROOT
    p = (ROOT / cfg['backtest'].get('wrds_dir', '../insider-wrds-data') / 'compustat_shortint.parquet').resolve()
    if not p.exists() or 'price_ticker' not in F or not F['price_ticker'].astype(str).str.startswith('CS_').any():
        return None
    si = pd.read_parquet(p)
    si['sec'] = 'CS_' + si['gvkey'].astype(str) + '_' + si['iid'].astype(str)
    si['datadate'] = pd.to_datetime(si['datadate'])
    si = si.dropna(subset=['shortintadj']).sort_values('datadate')
    from .prices import load_price
    rows = []
    for sec, g in F.groupby('price_ticker'):
        s = si[si['sec'] == sec]
        if s.empty:
            continue
        try:
            px = load_price(sec)
        except Exception:
            continue
        if 'shares_out' not in px:
            continue
        for _, r in g.iterrows():
            prior = s[s['datadate'] <= r['signal_date']]
            if prior.empty or (r['signal_date'] - prior['datadate'].iloc[-1]).days > 45:
                continue
            # shortintadj is split-adjusted to today's basis, so compare with raw shares x adjustment
            sh = px['shares_out'].asof(prior['datadate'].iloc[-1])
            raw = prior['shortint'].iloc[-1]
            if sh and sh > 0 and pd.notna(raw):
                rows.append({'screen': r['screen'], 'signal_date': r['signal_date'], 'issuer_cik': r['issuer_cik'],
                             'short_interest_pct': raw / sh})
    return pd.DataFrame(rows) if rows else None


def tests(F):
    return [
        ('Primary', 'P1 No insider selling ≥$100K in prior 90 days', lambda t: ~t['insider_sale_90d']),
        ('Primary', 'P2 Discretionary buys only (no plans, DRIP, offerings)', lambda t: t['discretionary_qualifies']),
        ('Primary', 'P3 No earnings 8-K in prior 30 days', lambda t: t['has_8k_history'] & ~t['earnings_8k_30d']),
        ('Exploratory', 'E1 Officer/director change 8-K in prior 90 days', lambda t: t['exec_change_8k_90d']),
        ('Exploratory', 'E2 No financing 8-K in prior 30 days', lambda t: t['has_8k_history'] & ~t['financing_8k_30d']),
        ('Exploratory', 'E3 No 8-K at all in prior 30 days', lambda t: t['has_8k_history'] & ~t['any_8k_30d']),
        ('Exploratory', 'E4 Mostly held directly (≥50% of value)', lambda t: t['direct_share'] >= 0.5),
        ('Exploratory', 'E5 Net buyers (buys > sales over 90 days)', lambda t: t['buys_90d'] > t['sales_90d']),
        ('Exploratory', 'E6 Short interest ≥10% of shares', lambda t: t.get('short_interest_pct', pd.Series(np.nan, t.index)) >= 0.10),
        ('Exploratory', 'E6b Short interest <10% of shares', lambda t: t.get('short_interest_pct', pd.Series(np.nan, t.index)) < 0.10),
    ]


def run(cfg=None):
    cfg = cfg or load_config()
    from .robust import _setup
    factors, cal, market, rf, scorer, F = _setup(cfg)
    F = F[F['screen'].isin(SCREENS_TESTED)].copy()
    sales, purch = download_extras(cfg)
    k8 = download_8k(cfg, F['issuer_cik'].unique())
    F = build(cfg, F, sales, purch, k8, short_interest_feature(cfg, F))
    save_table(F.drop(columns=['accessions']), DATA / 'signal_filing_features', gzip_csv=True)

    rows = []
    for screen in SCREENS_TESTED:
        base = F[F['screen'] == screen]
        for hname, hd in HOLDS.items():
            b = evaluate(base, market, factors, hd)
            rows.append({'screen': screen, 'hold': hname, 'group': 'Baseline', 'filter': 'No filter (realistic)',
                         'share kept': 1.0, **b})
            for group, name, fn in tests(F):
                mask = fn(base).fillna(False).astype(bool)
                t = base[mask]
                if len(t) < 30:
                    continue
                rows.append({'screen': screen, 'hold': hname, 'group': group, 'filter': name,
                             'share kept': mask.mean(), **evaluate(t, market, factors, hd)})
            LOG.info(f'filings: {screen} {hname} done')
    d = pd.DataFrame(rows)
    base = d[d['group'] == 'Baseline'][['screen', 'hold'] + [f'{l} alpha' for l in ('full', 'train', 'test', '2020+')]]
    d = d.merge(base, on=['screen', 'hold'], suffixes=('', ' baseline'))
    for lab in ('full', 'train', 'test', '2020+'):
        d[f'{lab} vs baseline'] = d[f'{lab} alpha'] - d[f'{lab} alpha baseline']
    d = d.drop(columns=[f'{l} alpha baseline' for l in ('full', 'train', 'test', '2020+')])
    prim = d['group'] == 'Primary'
    p = d.loc[prim, 'full p'].fillna(1.0).values
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[order]))
    holm = np.empty(m)
    holm[order] = adj
    d.loc[prim, 'holm p'] = holm
    d['holds out of sample'] = (d['train vs baseline'] > 0) & (d['test vs baseline'] > 0) & (d['test t'] >= 1.96)
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / 'filing_tests.csv', index=False)
    LOG.info('filings: done')
    return d


def prefetch(cfg=None):
    """Download the SEC extras ahead of the backtest; needs only the purchase data and signals."""
    cfg = cfg or load_config()
    download_extras(cfg)
    sig = pd.read_parquet(DATA / 'signals.parquet', columns=['screen', 'issuer_cik'])
    download_8k(cfg, sig.loc[sig['screen'].isin(SCREENS_TESTED), 'issuer_cik'].unique())
    LOG.info('filings: prefetch done')


if __name__ == '__main__':
    run()
