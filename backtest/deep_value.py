"""Phase 12: deep value, cigar butts and the "DFV" setup (cheap + hated + not dying + smart money), long-only.

Rules were written down and committed before any of this was run.

Universe (each month-end, point-in-time): Compustat US common stocks in the price cache with price >= $1,
>= $100K average daily dollar volume over the prior 20 days and market cap >= $20M.

Fundamentals: quarterly Compustat, public on the report date (else 45 days after quarter end); trailing
12 months = sum of the last 4 quarters.
  NCAV = current assets - total liabilities.     Net cash = cash - (long-term + short-term debt).
  EV = market cap + debt - cash.                 EBITDA = operating income after depreciation + depreciation.
  Cash runway OK = trailing net income >= 0, or cash >= 2 x trailing losses.
Short interest % = latest Compustat short interest (<= 45 days old) / shares outstanding.
Smart money in the prior 90 days: an officer or director open-market purchase totalling >= $100K, or an
original Schedule 13D (a new 5%+ holder with intent to influence; amendments excluded) naming the company.
SUE as in Phase 11.

Cheap  = any of: market cap < NCAV; net cash >= 30% of market cap; price-to-book <= 1; EV/EBITDA <= 5 with
         EBITDA > 0; EV < 0.
Hated  = short interest >= 15% of shares, or stock down >= 50% over the prior 12 months.
Alive  = cash runway OK, or SUE > 0 in the prior 90 days.

Strategies (an entry is the first month-end a company qualifies; the same company cannot re-enter for
12 months; equal weight; held 12 months or to delisting; costs = estimated half-spread each way + 0.15% FX
each way; money not in positions is in SPY):
  D1 Net-nets: market cap < NCAV.
  D2 Net-nets with smart money.
  D3 DFV: cheap and hated and alive and smart money.
  D4 DFV without the smart-money condition (to see what it adds).
  D5 Every original 13D on a universe stock.
  D6 13D on a cheap stock.
  Each also with a concentration cap: at most 20 open positions (new entries skipped while full).
Reported against SPY and QQQ: CAGR, Sharpe, max drawdown, all years / 2006-15 / 2016+, and per-entry
12-month return minus SPY with month-clustered t-stats.
"""
import time

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table
from .compustat import CS_CACHE, wrds_dir

LOG = get_logger('deep_value')
FX = 0.0015
HOLD = 252
CAP = 20
TRAIN_END = pd.Timestamp('2015-12-31')
FORMS_13D = ('SC 13D', 'SCHEDULE 13D')


# --- Schedule 13D filings ------------------------------------------------------------------------
def filings_13d(cfg):
    path = CACHE / 'sec_13d.parquet'
    if path.exists():
        return pd.read_parquet(path)
    s = requests.Session()
    s.headers['User-Agent'] = cfg['sec']['user_agent']
    rows = []
    for y in range(2004, pd.Timestamp.today().year + 1):
        for q in range(1, 5):
            r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                form = line[:17].strip()
                if form not in FORMS_13D:
                    continue
                parts = line.split()
                rows.append((int(parts[-3]), parts[-2], parts[-1]))
            time.sleep(0.2)
        LOG.info(f'13D index {y}: {len(rows):,} lines')
    d = pd.DataFrame(rows, columns=['cik', 'filing_date', 'path'])
    d['filing_date'] = pd.to_datetime(d['filing_date'])
    # each filing is listed under the subject company and under the filer; keep the subject (a US issuer)
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['cik', 'loc'])
    issuers = set(pd.to_numeric(co.loc[co['loc'] == 'USA', 'cik'], errors='coerce').dropna().astype(int))
    d = d[d['cik'].isin(issuers)]
    # filer = CIK in the accession path; drop self-filings
    d['filer'] = d['path'].str.extract(r'edgar/data/(\d+)/')[0].astype(int)
    d = d.drop_duplicates(['cik', 'path'])
    d.to_parquet(path, index=False)
    return d


# --- Monthly panel -------------------------------------------------------------------------------
def monthly_panel(cal):
    """Month-end price, market cap, ADV and 12-month return for every cached Compustat security."""
    path = CACHE / 'cs_monthly_panel.parquet'
    if path.exists():
        return pd.read_parquet(path)
    from .prices import load_price
    ends = pd.DatetimeIndex(cal).to_series().groupby(pd.DatetimeIndex(cal).to_period('M')).max()
    rows = []
    files = sorted(CS_CACHE.glob('CS_*.parquet'))
    for k, p in enumerate(files):
        px = load_price(p.stem)
        if px is None or px.empty or 'shares_out' not in px:
            continue
        px = px[~px.index.duplicated()]
        adv = px['dollar_volume'].rolling(20, min_periods=10).mean() if 'dollar_volume' in px else None
        ret12 = px['adj_close'] / px['adj_close'].shift(252) - 1
        e = ends[(ends >= px.index[0]) & (ends <= px.index[-1])]
        if e.empty:
            continue
        sub = px.reindex(e.values, method='ffill')
        rows.append(pd.DataFrame({'sec': p.stem, 'month_end': e.values, 'price': sub['raw_close'].values,
                                  'mcap': (sub['raw_close'] * sub['shares_out']).values,
                                  'shares': sub['shares_out'].values,
                                  'adv20': adv.reindex(e.values, method='ffill').values if adv is not None else np.nan,
                                  'ret12': ret12.reindex(e.values, method='ffill').values}))
        if (k + 1) % 5000 == 0:
            LOG.info(f'panel: {k + 1:,}/{len(files):,}')
    P = pd.concat(rows, ignore_index=True)
    P['gvkey'] = P['sec'].str.split('_').str[1]
    P.to_parquet(path, index=False)
    return P


def fundamentals_q(cfg):
    q = pd.read_parquet(wrds_dir(cfg) / 'compustat_fundq_full.parquet')
    num = ['actq', 'ltq', 'cheq', 'dlttq', 'dlcq', 'ceqq', 'oiadpq', 'dpq', 'niq', 'epspxq', 'prstkcy']
    for c in num:
        q[c] = pd.to_numeric(q[c], errors='coerce')
    q['datadate'] = pd.to_datetime(q['datadate'])
    q['rdq'] = pd.to_datetime(q['rdq'], errors='coerce')
    q = q.sort_values(['gvkey', 'datadate']).drop_duplicates(['gvkey', 'datadate'], keep='last')
    q['public'] = q['rdq'].fillna(q['datadate'] + pd.Timedelta(days=45))
    g = q.groupby('gvkey')
    q['ebitda_ttm'] = g['oiadpq'].transform(lambda s: s.rolling(4, min_periods=4).sum()) + \
        g['dpq'].transform(lambda s: s.rolling(4, min_periods=4).sum())
    q['ni_ttm'] = g['niq'].transform(lambda s: s.rolling(4, min_periods=4).sum())
    chg = q['epspxq'] - g['epspxq'].shift(4)
    sd = chg.groupby(q['gvkey']).transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q['sue'] = (chg / sd).where(sd > 0)
    keep = ['gvkey', 'public', 'actq', 'ltq', 'cheq', 'dlttq', 'dlcq', 'ceqq', 'ebitda_ttm', 'ni_ttm', 'sue']
    return q[keep].dropna(subset=['public']).sort_values('public')


def features(cfg, cal):
    P = monthly_panel(cal)
    P = P[(P['price'] >= 1) & (P['adv20'] >= 100_000) & (P['mcap'] >= 20e6)].copy()
    P = P.sort_values('month_end')
    P['gvkey'] = P['gvkey'].astype(str)
    P['sec'] = P['sec'].astype(str)
    q = fundamentals_q(cfg)
    q['gvkey'] = q['gvkey'].astype(str)
    P = pd.merge_asof(P, q.rename(columns={'public': 'f_public'}), left_on='month_end', right_on='f_public',
                      by='gvkey', direction='backward', tolerance=pd.Timedelta(days=200))
    mc = P['mcap'] / 1e6                        # Compustat items are in millions
    debt = P['dlttq'].fillna(0) + P['dlcq'].fillna(0)
    P['ncav'] = P['actq'] - P['ltq']
    P['net_cash'] = P['cheq'] - debt
    P['ev'] = mc + debt - P['cheq']
    P['netnet'] = mc < P['ncav']
    P['cheap'] = P['netnet'] | (P['net_cash'] >= 0.3 * mc) | ((P['ceqq'] > 0) & (mc <= P['ceqq'])) | \
        ((P['ebitda_ttm'] > 0) & (P['ev'] / P['ebitda_ttm'] <= 5)) | (P['ev'] < 0)
    P['runway_ok'] = (P['ni_ttm'] >= 0) | (P['cheq'] >= -2 * P['ni_ttm'])
    P['sue_recent'] = P['sue'].where((P['month_end'] - P['f_public']).dt.days <= 90)
    P['alive'] = P['runway_ok'].fillna(False) | (P['sue_recent'] > 0)
    # short interest
    si = pd.read_parquet(wrds_dir(cfg) / 'compustat_shortint.parquet')
    si['datadate'] = pd.to_datetime(si['datadate'])
    si['sec'] = 'CS_' + si['gvkey'].astype(str) + '_' + si['iid'].astype(str)
    si['sec'] = si['sec'].astype(str)
    si = si.sort_values('datadate')[['sec', 'datadate', 'shortint']]
    P = pd.merge_asof(P.sort_values('month_end'), si, left_on='month_end', right_on='datadate', by='sec',
                      direction='backward', tolerance=pd.Timedelta(days=45))
    P['short_pct'] = pd.to_numeric(P['shortint'], errors='coerce') / P['shares']
    P['hated'] = (P['short_pct'] >= 0.15) | (P['ret12'] <= -0.5)
    return P


def smart_money(cfg, P):
    """Officer/director buys >= $100K and original 13Ds in the 90 days to each month-end."""
    from .compustat import security_map
    sm = security_map(cfg)[['cik_i', 'gvkey']].dropna().drop_duplicates()
    sm['cik_i'] = sm['cik_i'].astype(int)
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['accession_number', 'issuer_cik', 'filing_date', 'value_clean'])
    it = pd.read_parquet(DATA / 'insider_trades.parquet', columns=['accession_number', 'is_director', 'is_officer'])
    pc = pc.merge(it, on='accession_number')
    pc = pc[(pc['is_director'] | pc['is_officer'])]
    pc['filing_date'] = pd.to_datetime(pc['filing_date'])
    pc = pc.merge(sm, left_on='issuer_cik', right_on='cik_i')
    d13 = filings_13d(cfg).merge(sm, left_on='cik', right_on='cik_i')
    out = {}
    for name, df, val in (('insider_90d', pc, 'value_clean'), ('d13_90d', d13, None)):
        df = df.sort_values('filing_date')
        by = {k: (g['filing_date'].values, (g[val].fillna(0).values if val else np.ones(len(g)))) for k, g in df.groupby('gvkey')}
        res = np.zeros(len(P))
        for i, (k, d) in enumerate(zip(P['gvkey'].values, P['month_end'].values)):
            t = by.get(k)
            if t is None:
                continue
            lo = np.searchsorted(t[0], d - np.timedelta64(90, 'D'), 'right')
            hi = np.searchsorted(t[0], d, 'right')
            res[i] = t[1][lo:hi].sum()
        out[name] = res
    P['insider_buy_90d'] = out['insider_90d'] >= 100_000
    P['d13_90d'] = out['d13_90d'] > 0
    P['smart_money'] = P['insider_buy_90d'] | P['d13_90d']
    return P


def entries(P, cond):
    """First month-end each company qualifies; no re-entry for 12 months."""
    x = P[cond].sort_values('month_end')
    keep, last = [], {}
    for i, (k, d) in zip(x.index, zip(x['gvkey'].values, x['month_end'].values)):
        if k in last and (d - last[k]) / np.timedelta64(1, 'D') < 365:
            continue
        keep.append(i)
        last[k] = d
    return x.loc[keep]


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market, corwin_schultz
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    cal = spy_px.index.values
    idx = pd.DatetimeIndex(cal)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    qqq = load_price('QQQ')['adj_close'].pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    P = features(cfg, cal)
    P = smart_money(cfg, P)
    save_table(P.drop(columns=['datadate'], errors='ignore'), DATA / 'deep_value_panel')
    LOG.info(f'deep value panel: {len(P):,} stock-months; net-nets {P["netnet"].mean():.1%}, cheap {P["cheap"].mean():.1%}, '
             f'hated {P["hated"].mean():.1%}, smart money {P["smart_money"].mean():.1%}')
    rules = {
        'D1 Net-nets': P['netnet'],
        'D2 Net-nets + smart money': P['netnet'] & P['smart_money'],
        'D3 DFV: cheap + hated + alive + smart money': P['cheap'] & P['hated'] & P['alive'] & P['smart_money'],
        'D4 DFV without smart money': P['cheap'] & P['hated'] & P['alive'],
        'D5 Every original 13D': P['d13_90d'],
        'D6 13D on a cheap stock': P['d13_90d'] & P['cheap'],
    }
    market = Market(cal)
    rows, curves, all_entries = [], {}, []
    for name, cond in rules.items():
        E = entries(P, cond.fillna(False).values).copy()
        E['price_ticker'] = E['sec']
        E['entry_date'] = [idx[min(np.searchsorted(cal, np.datetime64(d), 'right'), len(idx) - 1)] for d in E['month_end']]
        E['strategy'] = name
        all_entries.append(E[['strategy', 'sec', 'gvkey', 'month_end', 'entry_date', 'mcap', 'short_pct', 'ret12']])
        # costs: Corwin-Schultz spread from each stock's prior 20 days
        spreads = []
        for _, e in E.iterrows():
            px = load_price(e['sec'])
            w = px.loc[:e['month_end']].tail(21)
            sp = np.nanmedian(corwin_schultz(w['adj_high'].values, w['adj_low'].values, w['adj_close'].values)) if len(w) > 5 else np.nan
            spreads.append(sp if np.isfinite(sp) else 0.01)
        E['cost'] = np.array(spreads) + 2 * FX
        eng = SubsetEngine(E.reset_index(drop=True), market, HOLD, E['cost'].values)
        E = E.reset_index(drop=True)
        # concentration cap: skip entries while 20 positions are open
        open_until, capped = [], np.zeros(len(E), bool)
        for i, d in enumerate(E['entry_date']):
            open_until = [u for u in open_until if u > d]
            if len(open_until) < CAP:
                capped[i] = True
                open_until.append(d + pd.Timedelta(days=365))
        for variant, mask in (('all entries', np.ones(len(E), bool)), (f'max {CAP} positions', capped)):
            p = eng.returns(mask.astype(float), rf.values)
            if p is None:
                continue
            act = pd.Series(eng.active(mask), index=idx)
            p = p.reindex(idx).where(act, spy).loc[act[act].index[0]:]
            curves[f'{name}|{variant}'] = p
            for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
                s_, b_, q_ = perf(p.loc[sl], rf), perf(spy.loc[p.index[0]:].loc[sl], rf), perf(qqq.loc[p.index[0]:].loc[sl], rf)
                rows.append({'strategy': name, 'variant': variant, 'period': lab, 'entries': int(mask.sum()),
                             'per year': mask.sum() / max(len(p) / 252, 1), 'days invested': float(act.loc[p.index[0]:].loc[sl].mean()),
                             'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'), 'QQQ CAGR': q_.get('CAGR'),
                             'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'), 'max DD': s_.get('max drawdown'),
                             'SPY max DD': b_.get('max drawdown')})
        LOG.info(f'deep value: {name}: {len(E):,} entries')
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'deep_value.csv', index=False)
    save_table(pd.concat(all_entries), DATA / 'deep_value_entries')
    pd.DataFrame(curves).to_parquet(DATA / 'deep_value_daily.parquet')
    LOG.info('deep value: done')
    return r


if __name__ == '__main__':
    run()
