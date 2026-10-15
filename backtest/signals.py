"""Phase 2: rebuild the OpenInsider screens from the purchase dataset.

Signal date is always the FILING date (when the public learns of the buy),
never the trade date.
"""
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, SCREENS, get_logger, load_config, save_table

LOG = get_logger('signals')

# --- Title parsing -----------------------------------------------------------
# Matches CEO / CFO / Chairman of the Board in messy free-text titles such as
# "Pres, CEO", "COB, Dir", "Chairman & Chief Executive Officer", "EVP and CFO".
# Excludes vice/deputy chairs and "former" roles.
_CEO = r"\bC\.?\s?E\.?\s?O\b|\bchief\s+exec(?:utive)?\b"
_CFO = r"\bC\.?\s?F\.?\s?O\b|\bchief\s+financial\b|\bprincipal\s+financial\s+officer\b"
_COB = r"\bCOB\b|(?<!vice\s)(?<!vice-)(?<!vice)(?<!deputy\s)\bchair(?:man|woman|person)?\b(?!\s*emerit)"
EXEC_TITLE_RE = re.compile(f"(?:{_CEO})|(?:{_CFO})|(?:{_COB})", re.IGNORECASE)
_FORMER_RE = re.compile(r"\b(former|fmr|retired)\b|\bex-", re.IGNORECASE)


def is_top_exec(title):
    """True if the title names a CEO, CFO or Chairman of the Board."""
    if not isinstance(title, str) or not title.strip():
        return False
    t = re.sub(r"\s+", " ", title)
    if _FORMER_RE.search(t):
        return False
    return bool(EXEC_TITLE_RE.search(t))


# --- Fund / trust detection --------------------------------------------------
FUND_NAME_RE = re.compile(r"\b(funds?|etfs?|trust|closed[- ]end|municipal|muni)\b", re.IGNORECASE)
BANK_SIC = range(6000, 6200)   # banks keep their name even if it says "Trust"


def fetch_company_info(ciks, cfg):
    """SIC code and name per issuer CIK from SEC EDGAR submissions JSON (cached)."""
    path = CACHE / 'sec_company_info.json'
    info = json.loads(path.read_text()) if path.exists() else {}
    todo = [int(c) for c in ciks if str(int(c)) not in info]
    LOG.info(f"Company info: {len(ciks) - len(todo)} cached, {len(todo)} to fetch from SEC")
    session = requests.Session()
    session.headers.update({'User-Agent': cfg['sec']['user_agent'], 'Accept-Encoding': 'gzip, deflate'})
    interval = 1.0 / cfg['sec'].get('max_requests_per_second', 5)
    for i, cik in enumerate(todo):
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        rec = {'sic': None, 'sic_description': None, 'sec_name': None, 'tickers': []}
        for attempt in range(4):
            time.sleep(interval * (2 ** attempt))
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 404:
                    break
                r.raise_for_status()
                j = r.json()
                rec = {'sic': int(j['sic']) if str(j.get('sic') or '').isdigit() else None,
                       'sic_description': j.get('sicDescription'), 'sec_name': j.get('name'),
                       'tickers': j.get('tickers') or []}
                break
            except (requests.RequestException, ValueError) as e:
                LOG.warning(f"CIK {cik}: {e} (attempt {attempt + 1})")
        info[str(cik)] = rec
        if (i + 1) % 200 == 0:
            path.write_text(json.dumps(info))
            LOG.info(f"  fetched {i + 1}/{len(todo)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return info


def fund_exclusion_reason(name, sec_name, sic, sic_range):
    if sic is not None and sic_range[0] <= sic <= sic_range[1]:
        return f'SIC {sic}'
    # Only the name on the filings counts: the SEC's current name can reflect a
    # later change (e.g. an operating company that became a liquidating trust).
    bank = sic is not None and sic in BANK_SIC
    if isinstance(name, str) and FUND_NAME_RE.search(name) and not bank:
        return 'name'
    return None


# --- Value cleaning ----------------------------------------------------------
def clean_values(df, max_value=1e9, ref_window_days=30, ratio=10.0):
    """Repair obvious price typos and drop impossible values.

    A filing whose price is more than 10x away from the median price of the
    same issuer's other purchase filings within +/-30 days gets its value
    recomputed at that median price. Values above $1B after that are treated
    as data-entry errors and excluded from dollar totals.
    """
    df = df.sort_values(['issuer_cik', 'filing_date']).copy()
    df['value_clean'] = df['Value'].astype(float)
    df['value_flag'] = ''
    ref = np.full(len(df), np.nan)
    days = df['filing_date'].values.astype('datetime64[D]').astype(np.int64)
    price = df['last_price'].values.astype(float)
    cik = df['issuer_cik'].values
    starts = np.r_[0, np.flatnonzero(cik[1:] != cik[:-1]) + 1, len(df)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        d, p = days[a:b], price[a:b]
        lo = np.searchsorted(d, d - ref_window_days, 'left')
        hi = np.searchsorted(d, d + ref_window_days, 'right')
        for k in range(b - a):
            others = np.r_[p[lo[k]:k], p[k + 1:hi[k]]]
            others = others[np.isfinite(others) & (others > 0)]
            if len(others):
                ref[a + k] = np.median(others)
    r = price / ref
    bad = np.isfinite(r) & ((r > ratio) | (r < 1 / ratio))
    df.loc[bad, 'value_clean'] = df.loc[bad, 'Qty'].values * ref[bad]
    df.loc[bad, 'value_flag'] = 'price_corrected'
    huge = df['value_clean'] > max_value
    df.loc[huge, 'value_clean'] = np.nan
    df.loc[huge, 'value_flag'] = 'value_excluded'
    LOG.info(f"Value cleaning: {bad.sum():,} prices corrected, {huge.sum():,} values > ${max_value:,.0f} excluded")
    return df


# --- Screens -----------------------------------------------------------------
def dedupe(cands, days):
    """Keep the first signal per issuer, then ignore that issuer for `days`."""
    if days <= 0 or cands.empty:
        return cands
    keep = []
    for _, g in cands.sort_values('signal_date').groupby('issuer_cik', sort=False):
        last = None
        for idx, d in zip(g.index, g['signal_date']):
            if last is None or (d - last).days > days:
                keep.append(idx)
                last = d
    return cands.loc[keep].sort_values(['signal_date', 'issuer_cik'])


def _signal_record(issuer, d, w):
    w = w.sort_values('filing_date')
    tickers = w['ticker'].dropna()
    qty = w['Qty'].sum()
    val = w['value_clean'].sum()
    return {
        'signal_date': d, 'issuer_cik': issuer,
        'ticker': tickers.iloc[-1] if len(tickers) else None,
        'company_name': w['company_name'].iloc[-1],
        'n_insiders': w['insider_id'].nunique(),
        'total_value': val, 'n_filings': len(w),
        'insiders': '; '.join(pd.unique(w['owner_name'].astype(str)))[:500],
        'titles': '; '.join(pd.unique(w['Title'].astype(str)))[:300],
        'first_trade_date': w['trade_date'].min(), 'last_trade_date': w['trade_date'].max(),
        'avg_buy_price': val / qty if qty > 0 else np.nan,
        'max_filing_lag_days': (w['filing_date'] - w['trade_date']).dt.days.max(),
        'accessions': ','.join(w['accession_number'])[:2000],
    }


def cluster_candidates(df, window_days, min_value, min_insiders):
    """Days a company first reaches >= min_insiders distinct insiders and
    >= min_value total within a rolling `window_days` filing-date window."""
    rows = []
    df = df.sort_values(['issuer_cik', 'filing_date'])
    ins_code = pd.factorize(df['insider_id'])[0]
    days = df['filing_date'].values.astype('datetime64[D]').astype(np.int64)
    vals = df['value_clean'].fillna(0.0).values
    cik = df['issuer_cik'].values
    starts = np.r_[0, np.flatnonzero(cik[1:] != cik[:-1]) + 1, len(df)]
    for a, b in zip(starts[:-1], starts[1:]):
        if len(np.unique(ins_code[a:b])) < min_insiders:
            continue
        d, ins, v = days[a:b], ins_code[a:b], vals[a:b]
        csum = np.r_[0.0, np.cumsum(v)]
        prev_true = None
        for day in np.unique(d):
            lo = np.searchsorted(d, day - (window_days - 1), 'left')
            hi = np.searchsorted(d, day, 'right')
            if csum[hi] - csum[lo] < min_value or len(np.unique(ins[lo:hi])) < min_insiders:
                continue
            # Fire on the first day the condition holds; a new cluster can
            # fire again only after the condition has lapsed for a full window.
            if prev_true is None or day - prev_true > window_days:
                w = df.iloc[a + lo:a + hi]
                rows.append(_signal_record(cik[a], pd.Timestamp(int(day), unit='D'), w))
            prev_true = day
    return pd.DataFrame(rows)


def exec_candidates(df, min_value):
    """Single purchase filings by a CEO/CFO/COB worth >= min_value, one per issuer-day."""
    x = df[df['is_top_exec'] & (df['value_clean'] >= min_value)]
    rows = [_signal_record(issuer, pd.Timestamp(d), w) for (issuer, d), w in x.groupby(['issuer_cik', 'filing_date'])]
    return pd.DataFrame(rows)


def prepare_purchases(cfg):
    path = DATA / cfg['output']['filename']
    df = pd.read_parquet(path)
    LOG.info(f"Loaded {len(df):,} purchase filings from {path}")
    df = df[df['transaction_type'] == 'P'].copy()
    df = df.dropna(subset=['filing_date', 'issuer_cik'])
    df['insider_id'] = df['owner_cik'].astype('string').fillna('name:' + df['owner_name'].astype(str))
    df['is_top_exec'] = df['Title'].map(is_top_exec)
    return clean_values(df)


def build_signals(cfg, dedupe_variants=None):
    bt = cfg['backtest']
    df = prepare_purchases(cfg)
    df.to_parquet(DATA / 'purchases_clean.parquet', index=False)

    # Cluster candidates at the loosest threshold, then fund/trust filter by issuer.
    window, min_value = bt['cluster_window_days'], bt['cluster_min_value']
    cands = {}
    base3 = cluster_candidates(df, window, min_value, min(bt['cluster_min_insiders']))
    info = fetch_company_info(sorted(base3['issuer_cik'].unique()), cfg)
    reasons = {}
    for cik in base3['issuer_cik'].unique():
        rec = info.get(str(int(cik)), {})
        name = base3.loc[base3['issuer_cik'] == cik, 'company_name'].iloc[-1]
        reasons[cik] = fund_exclusion_reason(name, rec.get('sec_name'), rec.get('sic'), bt['fund_sic_range'])
    excluded = {c: r for c, r in reasons.items() if r}
    excl_df = pd.DataFrame([{'issuer_cik': c, 'company_name': base3.loc[base3['issuer_cik'] == c, 'company_name'].iloc[-1],
                             'sic': info.get(str(int(c)), {}).get('sic'),
                             'sic_description': info.get(str(int(c)), {}).get('sic_description'),
                             'reason': r} for c, r in excluded.items()])
    if not excl_df.empty:
        excl_df['signals_dropped_3plus'] = excl_df['issuer_cik'].map(base3['issuer_cik'].value_counts())
    save_table(excl_df, DATA / 'excluded_funds_trusts')
    n_sig_dropped = int(base3['issuer_cik'].isin(excluded).sum())
    LOG.info(f"Fund/trust filter: dropped {len(excluded):,} companies "
             f"({sum(1 for r in excluded.values() if r.startswith('SIC')):,} by SIC {bt['fund_sic_range']}, "
             f"{sum(1 for r in excluded.values() if r == 'name'):,} by name) = {n_sig_dropped:,} raw cluster signals")
    df_clusters = df[~df['issuer_cik'].isin(excluded)]

    for key, (label, kind, param) in SCREENS.items():
        if kind == 'cluster':
            c = base3 if param == min(bt['cluster_min_insiders']) else \
                cluster_candidates(df_clusters, window, min_value, param)
            c = c[~c['issuer_cik'].isin(excluded)]
        else:
            c = exec_candidates(df, param)
        c = c.copy()
        c['screen'] = key
        cands[key] = c
        LOG.info(f"{key}: {len(c):,} raw candidates")

    variants = dedupe_variants or bt['dedupe_variants']
    out = []
    for key, c in cands.items():
        for dd in sorted(set(variants) | {bt['dedupe_days']}):
            s = dedupe(c, dd).copy()
            s['dedupe_days'] = dd
            out.append(s)
    signals = pd.concat(out, ignore_index=True)
    signals['signal_year'] = signals['signal_date'].dt.year

    base = signals[signals['dedupe_days'] == bt['dedupe_days']]
    counts = base.pivot_table(index='signal_year', columns='screen', values='issuer_cik',
                              aggfunc='count', fill_value=0)[list(SCREENS)]
    LOG.info(f"Signals per screen per year (dedupe {bt['dedupe_days']}d):\n{counts}")
    meta = {'fund_companies_dropped': len(excluded), 'fund_raw_signals_dropped': n_sig_dropped,
            'fund_dropped_by_sic': sum(1 for r in excluded.values() if r.startswith('SIC')),
            'fund_dropped_by_name': sum(1 for r in excluded.values() if r == 'name'),
            'prices_corrected': int((df['value_flag'] == 'price_corrected').sum()),
            'values_excluded': int((df['value_flag'] == 'value_excluded').sum()),
            'top_exec_filings': int(df['is_top_exec'].sum())}
    (DATA / 'signals_meta.json').write_text(json.dumps(meta, indent=2))
    save_table(signals, DATA / 'signals_all_variants')
    save_table(base.drop(columns=['accessions']), DATA / 'signals')
    save_table(counts, DATA / 'signal_counts_by_year', index=True)
    return signals, meta


if __name__ == '__main__':
    build_signals(load_config())
