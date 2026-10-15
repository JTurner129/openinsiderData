"""Phase 23: out-of-sample holdout, 1994-2004 — the spin-off rules, unchanged, on years never used before.

Rules were written down and committed before the 1990-2004 data was looked at (it arrived after Phases 19, 21
and 22 were finished; nothing below was tuned on it).

Data: Compustat daily prices 1990-2004 for every US-dollar common stock of a US company (WRDS, holdout pull),
built into a separate cache (CS_CACHE_NAME=prices_cs_full); SEC EDGAR full-index for original Form 10-12B
filings 1993-2004; SPY from Yahoo (from 1993); Fama-French 5 factors + momentum, daily, from Kenneth French.
Spin-off events: EXACTLY the Phase 19 F3 rules — an original Form 10-12B whose company's first Compustat trading
day falls within a year after the filing; enter 21 trading days after that first day; hold 12 months (252 trading
days); tradeable at entry: price >= $1 and >= $100K average daily dollar volume over the prior 20 days (same
nominal threshold, not inflation-adjusted); costs Corwin-Schultz half-spread each way + 0.15% FX each way.
Holdout events: entry dates 1994-01-01 to 2004-12-31. Days with nothing held earn SPY.
Reported for the holdout (1994-2004, also 1994-99 and 2000-04): spin-off book alone, P1 = 70% SPY + 30%
spin-off book (monthly rebalance, 0.15% per side on the amount moved), and SPY: CAGR, Sharpe, max drawdown.
Per-trade 12-month excess over SPY (stock return x (1 - cost) minus SPY, as in Phase 19): mean, median, share positive, month-clustered t. Book alpha vs FF5 + UMD
(monthly, Newey-West 12 lags). Robustness: costs x2; without the 5 best spin-offs.
Pooled 1994-2026 (holdout + Phase 21 events): per-trade mean, month-clustered t, and FF5+UMD alpha with t.
Pre-written pass condition for the holdout: per-trade mean excess over SPY > 0 AND the spin-off book beats SPY
on CAGR over 1994-2004 AND P1 beats SPY on CAGR over 1994-2004. "Statistically established" additionally needs
the pooled month-clustered t >= 2.5 or pooled alpha t >= 2.5.
"""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table

LOG = get_logger('holdout')
FX = 0.0015
SIDE = 0.0015
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'


def form10_early(cfg):
    path = CACHE / 'sec_form10_1993_2004.parquet'
    if path.exists():
        return pd.read_parquet(path)
    s = requests.Session()
    s.headers['User-Agent'] = cfg['sec']['user_agent']
    rows = []
    for y in range(1993, 2005):
        for q in range(1, 5):
            r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
            if r.status_code != 200:
                LOG.warning(f'holdout: EDGAR index {y} Q{q}: HTTP {r.status_code}')
                continue
            for line in r.text.splitlines():
                if line[:17].strip() == '10-12B':
                    p = line.split()
                    rows.append((int(p[-3]), p[-2]))
            time.sleep(0.2)
    f = pd.DataFrame(rows, columns=['cik', 'filing_date'])
    f['filing_date'] = pd.to_datetime(f['filing_date'])
    f = f.sort_values('filing_date').drop_duplicates('cik')
    f.to_parquet(path, index=False)
    return f


def spy_long():
    path = CACHE / 'spy_1993.parquet'
    if path.exists():
        return pd.read_parquet(path)['adj_close']
    import yfinance as yf
    d = yf.download('SPY', start='1993-01-01', end='2006-12-31', auto_adjust=False, progress=False)
    s = d['Adj Close']
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.rename('adj_close').to_frame().to_parquet(path)
    return s


def french_daily():
    path = CACHE / 'ff5_umd_daily_full.parquet'
    if path.exists():
        return pd.read_parquet(path)
    from .industry_momentum import _first_block
    f5 = _first_block(FRENCH + 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip', daily=True)
    mo = _first_block(FRENCH + 'F-F_Momentum_Factor_daily_CSV.zip', daily=True)
    mo.columns = ['UMD']
    f = f5.join(mo, how='inner')
    f.to_parquet(path)
    return f


def clustered_t(ex, months):
    ex = pd.Series(ex)
    g = (ex - ex.mean()).groupby(pd.Series(months, index=ex.index)).sum()
    return ex.mean() / (np.sqrt((g ** 2).sum()) / len(ex))


def alpha(book, F):
    import statsmodels.api as sm
    m = (1 + book).resample('ME').prod() - 1
    Fm = (1 + F).resample('ME').prod() - 1
    d = pd.concat([m.rename('p'), Fm], axis=1).dropna()
    X = sm.add_constant(d[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']])
    r = sm.OLS(d['p'] - d['RF'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    return r.params['const'] * 12, r.tvalues['const'], len(d)


def perf(r, rf):
    from .strategies import perf as p
    return p(r, rf)


def run(cfg=None):
    import os
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    from .combos import SubsetEngine
    from .compustat import CS_CACHE, wrds_dir
    from .filters import Market, corwin_schultz
    from .forced_flows import next_day, tradeable
    from .prices import load_price
    from .spinoff_core import blend
    spy_px = spy_long()
    F = french_daily()
    cal = spy_px.loc['1993-02-01':'2006-12-31'].index.values
    idx = pd.DatetimeIndex(cal)
    spy = spy_px.pct_change().reindex(idx).fillna(0.0)
    rf = F['RF'].reindex(idx).fillna(0.0)
    f10 = form10_early(cfg)
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    f = f10.merge(co.dropna(), on='cik')
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    meta['first_date'] = pd.to_datetime(meta['first_date'])
    f = f.merge(meta[['gvkey', 'sec', 'first_date']], on='gvkey')
    f = f[(f['first_date'] >= f['filing_date']) & (f['first_date'] <= f['filing_date'] + pd.Timedelta(days=365))]
    rows = []
    for _, r in f.iterrows():
        d = next_day(idx, r['first_date'], 21)
        if d is not None:
            rows.append({'price_ticker': 'CS_' + str(r['sec']), 'gvkey': r['gvkey'], 'entry_date': d, 'event': r['first_date']})
    E = pd.DataFrame(rows).drop_duplicates('gvkey')
    E = E[(E['entry_date'] >= '1994-01-01') & (E['entry_date'] <= '2004-12-31')].reset_index(drop=True)
    LOG.info(f'holdout: {len(f10):,} Form 10-12B filers 1993-2004, {len(E)} spin-off events with entries 1994-2004')
    keep, costs = [], []
    for i, e in E.iterrows():
        ok, w = tradeable(e['price_ticker'], e['entry_date'])
        if not ok:
            continue
        sp = np.nanmedian(corwin_schultz(w['adj_high'].values, w['adj_low'].values, w['adj_close'].values))
        keep.append(i)
        costs.append((sp if np.isfinite(sp) else 0.01) + 2 * FX)
    E = E.loc[keep].reset_index(drop=True)
    E['cost'] = costs
    ex, mon, ret12 = [], [], []
    for _, e in E.iterrows():
        a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
        b = spy_px.loc[e['entry_date']:].iloc[:253]
        r12 = a.iloc[-1] / a.iloc[0] - 1 if len(a) > 1 else np.nan
        ret12.append(r12)
        ex.append(r12 * (1 - e['cost']) - (b.iloc[-1] / b.iloc[0] - 1) if len(a) > 1 else np.nan)
        mon.append(str(pd.Timestamp(e['entry_date']).to_period('M')))
    E['ret12'], E['excess12'], E['month'] = ret12, ex, mon
    LOG.info(f'holdout: {len(E)} tradeable spin-offs')
    market = Market(cal)

    def book(ev, cm=1.0):
        ev = ev.reset_index(drop=True)
        eng = SubsetEngine(ev, market, 252, ev['cost'].values * cm)
        mask = np.ones(len(ev))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy)
        return p.loc[act[act].index[0]:'2005-12-31']

    top = E.sort_values('ret12', ascending=False)
    books = {'spin-off book alone': book(E), 'spin-off book, costs x2': book(E, 2),
             'spin-off book without 5 best': book(E.drop(top.index[:5]))}
    P = dict(books)
    P['P1 70% SPY + 30% spin-offs'] = blend(books['spin-off book alone'], spy.loc[books['spin-off book alone'].index], 0.30)
    P['P1 costs x2'] = blend(books['spin-off book, costs x2'], spy.loc[books['spin-off book, costs x2'].index], 0.30, 2 * SIDE)
    P['P1 without 5 best'] = blend(books['spin-off book without 5 best'], spy.loc[books['spin-off book without 5 best'].index], 0.30)
    P['SPY'] = spy.loc[books['spin-off book alone'].index]
    out = []
    for name, p in P.items():
        for lab, sl in (('1994–2004', slice('1994-01-01', '2004-12-31')), ('1994–99', slice('1994-01-01', '1999-12-31')),
                        ('2000–04', slice('2000-01-01', '2004-12-31'))):
            s_, b_ = perf(p.loc[sl], rf), perf(spy.loc[sl], rf)
            out.append({'portfolio': name, 'period': lab, 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'),
                        'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'), 'max DD': s_.get('max drawdown'),
                        'SPY max DD': b_.get('max drawdown')})
    R = pd.DataFrame(out)
    exs = E['excess12'].dropna()
    t_h = clustered_t(exs.values, E.loc[exs.index, 'month'].values)
    a_h, ta_h, n_h = alpha(books['spin-off book alone'].loc[:'2004-12-31'], F)
    # pooled with the 2006+ events (Phase 21), evaluated on the standard cache's results
    late = pd.read_parquet(DATA / 'spinoff_events.parquet')
    late_ex = []
    from .prices import PRICE_CACHE, yahoo_symbol
    spy_new = pd.read_parquet(PRICE_CACHE / f'{yahoo_symbol("SPY")}.parquet')['adj_close']
    for _, e in late.iterrows():
        b = spy_new.loc[e['entry_date']:].iloc[:253]
        late_ex.append(e['ret12'] * (1 - e['cost']) - (b.iloc[-1] / b.iloc[0] - 1) if pd.notna(e['ret12']) and len(b) > 1 else np.nan)
    late['excess12'] = late_ex
    late['month'] = pd.to_datetime(late['entry_date']).dt.to_period('M').astype(str)
    pool = pd.concat([E[['excess12', 'month']], late[['excess12', 'month']]]).dropna()
    t_p = clustered_t(pool['excess12'].values, pool['month'].values)
    late_book = pd.read_parquet(DATA / 'spinoff_core_daily.parquet')['spin-off book alone (100%)']
    late_book.index = pd.to_datetime(late_book.index)
    pooled_book = pd.concat([books['spin-off book alone'].loc[:late_book.index[0] - pd.Timedelta(days=1)], late_book])
    a_p, ta_p, n_p = alpha(pooled_book, F)
    g = lambda n, k: R.loc[(R['portfolio'] == n) & (R['period'] == '1994–2004'), k].iloc[0]
    passed = bool(exs.mean() > 0 and g('spin-off book alone', 'CAGR') > g('SPY', 'CAGR')
                  and g('P1 70% SPY + 30% spin-offs', 'CAGR') > g('SPY', 'CAGR'))
    established = bool(passed and (t_p >= 2.5 or ta_p >= 2.5))
    summary = pd.DataFrame([
        {'sample': 'holdout 1994–2004', 'spin-offs': len(exs), 'mean excess per trade': exs.mean(), 'median': exs.median(),
         'share beating SPY': (exs > 0).mean(), 'clustered t': t_h, 'FF5+UMD alpha/yr': a_h, 'alpha t': ta_h, 'months': n_h},
        {'sample': 'Phase 21 2005–2026', 'spin-offs': late['excess12'].notna().sum(), 'mean excess per trade': late['excess12'].mean(),
         'median': late['excess12'].median(), 'share beating SPY': (late['excess12'].dropna() > 0).mean(),
         'clustered t': clustered_t(late['excess12'].dropna().values, late.loc[late['excess12'].notna(), 'month'].values)},
        {'sample': 'pooled 1994–2026', 'spin-offs': len(pool), 'mean excess per trade': pool['excess12'].mean(),
         'median': pool['excess12'].median(), 'share beating SPY': (pool['excess12'] > 0).mean(), 'clustered t': t_p,
         'FF5+UMD alpha/yr': a_p, 'alpha t': ta_p, 'months': n_p}])
    summary['holdout pass (pre-written)'] = passed
    summary['statistically established (pre-written)'] = established
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'holdout_spinoffs.csv', index=False)
    summary.to_csv(RESULTS / 'holdout_spinoffs_stats.csv', index=False)
    save_table(E, DATA / 'holdout_spinoff_events')
    pd.DataFrame(P).to_parquet(DATA / 'holdout_spinoffs_daily.parquet')
    LOG.info(f'holdout: done; pass = {passed}, established = {established}')
    return R, summary


if __name__ == '__main__':
    run()
