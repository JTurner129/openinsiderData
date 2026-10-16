"""Phase 9: "Lazy Prices" text-change signal from 10-K and 10-Q filings (Cohen, Malloy & Nguyen 2020).

Rules fixed before any filing was downloaded:

  Filings: 10-K (and 10-K405 / 10-KSB) and 10-Q (and 10-QSB), originals only, 2004 onward, for every
  company with a US-headquartered Compustat record. Only the main document is read (the first <DOCUMENT> in the
  submission), streamed and discarded after scoring.

  Text: HTML stripped, lowercased, words of 2+ letters. Each filing is compared with the same company's
  previous comparable filing: a 10-K with its previous 10-K (filed 300-430 days earlier), a 10-Q with
  the 10-Q filed 300-430 days earlier (same fiscal quarter last year). Scores: cosine similarity of
  word counts, and Jaccard similarity of word sets. The same scores are computed for the Risk Factors
  section (Item 1A) when it can be found in both filings.

  Test (run once the scores exist): at each month-end, rank stocks by the cosine similarity of their
  most recent filing scored in the prior 3 months; quintiles, equal weight, held one month; universe =
  Compustat common stocks with price >= $1 and >= $100K a day traded. Main result = top quintile
  (least changed) minus bottom quintile (most changed), FF5+UMD alpha, all years and 2016+; and the
  long-only top quintile against SPY. Same for the Risk Factors similarity.

Downloader:  python -c "from backtest import text_changes as t; t.download()"
"""
import html
import io
import json
import re
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table

LOG = get_logger('text')
OUT = CACHE / 'text_changes'
FORMS = {'10-K': '10-K', '10-K405': '10-K', '10-KSB': '10-K', '10-Q': '10-Q', '10-QSB': '10-Q'}
YEARS = range(2004, 2027)
MAX_BYTES = 40_000_000
HASH_DIM = 1024
WORD = re.compile(r'[a-z]{2,}')
TAG = re.compile(r'<[^>]+>')
RF_START = re.compile(r'item\s*1a\.?\s*[\-–—:.]?\s*risk\s+factors', re.I)
RF_END = re.compile(r'item\s*(1b|2|3)\.?\s*[\-–—:.]?\s*(unresolved|properties|legal|unregistered|defaults|quantitative|management)', re.I)


class Limiter:
    def __init__(self, per_s):
        self.gap, self.t, self.lock = 1.0 / per_s, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            d = self.t + self.gap - time.monotonic()
            if d > 0:
                time.sleep(d)
            self.t = time.monotonic()


def filing_list(cfg):
    """All 10-K / 10-Q filings from EDGAR's quarterly form indexes, limited to US Compustat CIKs."""
    path = OUT / '_filings.parquet'
    if path.exists():
        return pd.read_parquet(path)
    from .compustat import wrds_dir
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['cik', 'loc'])
    ciks = set(pd.to_numeric(co.loc[co['loc'] == 'USA', 'cik'], errors='coerce').dropna().astype(int))
    s = requests.Session()
    s.headers['User-Agent'] = cfg['sec']['user_agent']
    rows = []
    for y in YEARS:
        for q in range(1, 5):
            r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                form = line[:12].strip()
                if form not in FORMS:
                    continue
                parts = line.split()
                cik, date, fname = int(parts[-3]), parts[-2], parts[-1]
                if cik in ciks:
                    rows.append((cik, FORMS[form], date, fname))
            time.sleep(0.2)
        LOG.info(f'filing index {y}: {len(rows):,} filings so far')
    df = pd.DataFrame(rows, columns=['cik', 'form', 'filing_date', 'path'])
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df = df.drop_duplicates('path').sort_values(['cik', 'filing_date'])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main_document(session, limiter, path):
    for attempt in range(5):
        limiter.wait()
        try:
            r = session.get('https://www.sec.gov/Archives/' + path, stream=True, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            buf = io.BytesIO()
            for ch in r.iter_content(1 << 16):
                buf.write(ch)
                if buf.tell() > MAX_BYTES or b'</DOCUMENT>' in ch or b'</document>' in ch:
                    break
            r.close()
            raw = buf.getvalue()
            end = raw.find(b'</DOCUMENT>')
            return raw[:end if end > 0 else None].decode('utf-8', 'replace')
        except requests.RequestException:
            time.sleep(5 * 2 ** attempt)
    return None


def clean_text(doc):
    doc = re.sub(r'(?is)<(script|style|ix:header)[^>]*>.*?</\1>', ' ', doc)
    return html.unescape(TAG.sub(' ', doc)).lower()


def risk_section(text):
    best = ''
    for m in RF_START.finditer(text):
        e = RF_END.search(text, m.end())
        chunk = text[m.end():e.start() if e else m.end() + 400_000]
        if len(chunk) > len(best):
            best = chunk
    return best if len(best) > 2000 else ''


def stats(words):
    c = Counter(words)
    v = np.zeros(HASH_DIM, np.float32)
    for w, n in c.items():
        v[zlib.crc32(w.encode()) % HASH_DIM] += n
    nv = np.linalg.norm(v)
    return c, (v / nv if nv else v).astype(np.float16)


def cosine(a, b):
    if not a or not b:
        return np.nan
    num = sum(n * b.get(w, 0) for w, n in a.items())
    den = np.sqrt(sum(n * n for n in a.values())) * np.sqrt(sum(n * n for n in b.values()))
    return num / den if den else np.nan


def jaccard(a, b):
    if not a or not b:
        return np.nan
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def score_company(cik, g, session, limiter):
    """Score every filing of one company against its prior comparable filing."""
    out = []
    hist = []      # (form, date, counts, risk_counts)
    for _, f in g.iterrows():
        doc = main_document(session, limiter, f['path'])
        if doc is None:
            continue
        text = clean_text(doc)
        words = WORD.findall(text)
        if len(words) < 500:
            continue
        c, vec = stats(words)
        rs = risk_section(text)
        rc = Counter(WORD.findall(rs)) if rs else Counter()
        prior = [h for h in hist if h[0] == f['form'] and 300 <= (f['filing_date'] - h[1]).days <= 430]
        p = prior[-1] if prior else None
        out.append({'cik': cik, 'form': f['form'], 'filing_date': f['filing_date'], 'path': f['path'],
                    'n_words': len(words), 'rf_words': sum(rc.values()),
                    'prior_date': p[1] if p else pd.NaT,
                    'sim_cosine': cosine(c, p[2]) if p else np.nan,
                    'sim_jaccard': jaccard(c, p[2]) if p else np.nan,
                    'rf_sim_cosine': cosine(rc, p[3]) if p and rc and p[3] else np.nan,
                    'rf_sim_jaccard': jaccard(rc, p[3]) if p and rc and p[3] else np.nan,
                    'rf_growth': (sum(rc.values()) / sum(p[3].values()) - 1) if p and rc and p[3] else np.nan,
                    'vec': vec.tobytes()})
        hist.append((f['form'], f['filing_date'], c, rc))
        hist = [h for h in hist if (f['filing_date'] - h[1]).days <= 450]
    return out


def download(cfg=None, workers=8, per_second=8):
    cfg = cfg or load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    fl = filing_list(cfg)
    done_path = OUT / '_done.json'
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    todo = [(c, g) for c, g in fl.groupby('cik') if c not in done]
    LOG.info(f'text: {len(fl):,} filings from {fl["cik"].nunique():,} companies; {len(todo):,} companies to do')
    limiter = Limiter(per_second)
    session = requests.Session()
    session.headers.update({'User-Agent': cfg['sec']['user_agent'], 'Accept-Encoding': 'gzip, deflate'})
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount('https://', adapter)
    buffer, lock = [], threading.Lock()
    shard = len(list(OUT.glob('scores_*.parquet')))
    t0, n_filings = time.time(), 0

    def flush():
        nonlocal buffer, shard
        if buffer:
            pd.DataFrame(buffer).to_parquet(OUT / f'scores_{shard:05d}.parquet', index=False)
            shard += 1
            buffer = []
        done_path.write_text(json.dumps(sorted(done)))

    def work(item):
        cik, g = item
        try:
            return cik, score_company(cik, g, session, limiter), len(g)
        except Exception as e:      # one bad company should not stop the run
            LOG.warning(f'text: CIK {cik} failed: {e}')
            return cik, [], len(g)

    with ThreadPoolExecutor(workers) as ex:
        for i, (cik, rows, n) in enumerate(ex.map(work, todo)):
            with lock:
                buffer.extend(rows)
                done.add(int(cik))
                n_filings += n
                if len(buffer) >= 2000:
                    flush()
            if i % 100 == 0:
                rate = n_filings / max(time.time() - t0, 1)
                left = (len(fl) - fl['cik'].isin(done).sum()) / max(rate, 1e-9) / 3600
                LOG.info(f'text: {len(done):,} companies done, {n_filings:,} filings this run, '
                         f'{rate:.1f}/s, about {left:.1f} h left')
    flush()
    LOG.info('text: download done')


def load_scores():
    parts = [pd.read_parquet(p, columns=['cik', 'form', 'filing_date', 'n_words', 'rf_words', 'sim_cosine',
                                         'sim_jaccard', 'rf_sim_cosine', 'rf_sim_jaccard', 'rf_growth'])
             for p in sorted(OUT.glob('scores_*.parquet'))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


if __name__ == '__main__':
    download()


def test(cfg=None):
    """Lazy Prices test exactly as in the module docstring. Returns are gross of trading costs (the
    pre-registered rules name none); the top quintile's monthly turnover is reported so costs can be judged."""
    import numpy as np
    import statsmodels.api as sm
    from .common import RESULTS, load_config
    from .compustat import wrds_dir
    from .prices import load_price
    cfg = cfg or load_config()
    S = load_scores()
    S['filing_date'] = pd.to_datetime(S['filing_date'])
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    S = S.merge(co.dropna(), on='cik')
    panel = pd.read_parquet(CACHE / 'cs_monthly_panel.parquet', columns=['sec', 'month_end', 'price', 'adv20', 'gvkey'])
    rets = pd.read_parquet(CACHE / 'cs_monthly_returns.parquet', columns=['sec', 'month_end', 'ret_next'])
    panel = panel.merge(rets, on=['sec', 'month_end'])
    panel['gvkey'] = panel['gvkey'].astype(str)
    panel['month_end'] = pd.to_datetime(panel['month_end'])
    panel = panel[(panel['price'] >= 1) & (panel['adv20'] >= 100_000)].dropna(subset=['ret_next'])
    panel = panel.sort_values('adv20').drop_duplicates(['gvkey', 'month_end'], keep='last')
    F = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')
    Fm = (1 + F).resample('ME').prod() - 1
    Fm.index = Fm.index.to_period('M')
    spy = load_price('SPY')['adj_close'].resample('ME').last().pct_change()
    spy.index = spy.index.to_period('M')
    out, monthly = [], {}
    for score in ('sim_cosine', 'rf_sim_cosine'):
        s = S.dropna(subset=[score]).sort_values('filing_date')[['gvkey', 'filing_date', score]]
        rows, prev = [], set()
        for me, g in panel.groupby('month_end'):
            w = s[(s['filing_date'] <= me) & (s['filing_date'] > me - pd.DateOffset(months=3))]
            w = w.drop_duplicates('gvkey', keep='last')
            x = g.merge(w, on='gvkey')
            if len(x) < 50:
                continue
            x['q'] = pd.qcut(x[score].rank(method='first'), 5, labels=False) + 1
            r = x.groupby('q')['ret_next'].mean()
            top = set(x.loc[x['q'] == 5, 'sec'])
            rows.append({'month': me.to_period('M') + 1, **{f'Q{int(k)}': v for k, v in r.items()}, 'n': len(x),
                         'Q5 turnover': 1 - len(top & prev) / max(len(top), 1)})
            prev = top
        Q = pd.DataFrame(rows).set_index('month')
        Q['Q5-Q1'] = Q['Q5'] - Q['Q1']
        monthly[score] = Q
        for lab, a, b in (('all', None, None), ('2016+', '2016-01', None), ('2005–2015', None, '2015-12')):
            q = Q.loc[a:b]
            for col in ('Q1', 'Q5', 'Q5-Q1'):
                x = q[col].dropna()
                f = Fm.reindex(x.index).dropna()
                x = x.reindex(f.index)
                y = x - (f['RF'] if col != 'Q5-Q1' else 0)
                res = sm.OLS(y, sm.add_constant(f[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']])).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
                sp = spy.reindex(x.index)
                cagr = (1 + x).prod() ** (12 / len(x)) - 1 if col != 'Q5-Q1' else np.nan
                out.append({'score': score, 'period': lab, 'portfolio': col, 'months': len(x),
                            'avg monthly': x.mean(), 'CAGR (gross)': cagr,
                            'SPY CAGR': (1 + sp.dropna()).prod() ** (12 / sp.notna().sum()) - 1 if col != 'Q5-Q1' else np.nan,
                            'alpha/yr': res.params['const'] * 12, 'alpha t': res.tvalues['const'],
                            'Q5 monthly turnover': q['Q5 turnover'].mean(), 'avg stocks': q['n'].mean()})
        LOG.info(f'text test: {score} done')
    R = pd.DataFrame(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'text_changes.csv', index=False)
    pd.concat(monthly, names=['score']).to_csv(RESULTS / 'text_changes_quintiles_monthly.csv')
    LOG.info('text test: done')
    return R
