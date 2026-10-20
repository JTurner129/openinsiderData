"""Phase 36: are spin-offs with NO analyst coverage at the buy date (most neglected) better?

Rules were written down and committed before this was run (the counts were inspected only for coverage:
339 of 363 numeric, 136 zeros, median 1).

Data: LSEG Workspace TR.NumberOfAnalysts on each US spin-off's buy date (eikon/analyst_counts_DONE.xlsx), matched
to the Phase 21 + 23 events by gvkey and buy date. Returns: the same 12-month excess over SPY after costs.
Note: many zeros will reflect the post-listing quiet period (brokers that helped with the spin-off cannot publish
for a few weeks) as well as true neglect; both are "nobody is telling investors about it yet".
Groups: "uncovered" = 0 analysts; "covered" = 1 or more. Secondary (reported, not decisive): 0-1 vs 4+.
Reported: n, mean, median, share > 0 per group, difference with a two-sample month-clustered t, by period
(1994-2004, 2005-2026, all), plus stand-alone books (CAGR, Sharpe, max DD, avg positions).
Pre-written bar: uncovered beats covered by >= 5 points per trade with t >= 2 over all years AND positive in
both periods AND the uncovered book holds >= 6 positions on average.
"""
import os

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger

LOG = get_logger('analyst_neglect')


def run(xlsx='eikon/analyst_counts_DONE.xlsx'):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    import statsmodels.formula.api as smf
    from .combos import SubsetEngine
    from .filters import Market
    from .prices import load_price
    from .spinoff_grid import spy_all
    from .strategies import perf
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy_r = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    A = pd.read_excel(xlsx)
    A.columns = ['ISIN', 'tic', 'date', 'gvkey', 'analysts']
    A['gvkey'] = A['gvkey'].astype(str).str.zfill(6)
    A['date'] = pd.to_datetime(A['date'])
    A['analysts'] = pd.to_numeric(A['analysts'], errors='coerce')
    cols = ['price_ticker', 'gvkey', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(period='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(period='2005–2026')], ignore_index=True)
    E['gvkey'] = E['gvkey'].astype(str).str.zfill(6)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    E = E.merge(A[['gvkey', 'date', 'analysts']], left_on=['gvkey', 'entry_date'], right_on=['gvkey', 'date'], how='left')
    E = E.dropna(subset=['analysts']).reset_index(drop=True)
    ex = []
    for _, e in E.iterrows():
        a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
        s = px.loc[e['entry_date']:].iloc[:253]
        ex.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost']) - (s.iloc[-1] / s.iloc[0] - 1) if len(a) > 1 else np.nan)
    E['ex'] = ex
    E['month'] = E['entry_date'].dt.to_period('M').astype(str)
    market = Market(idx.values)

    def tdiff(f, s):
        d = pd.concat([f.assign(grp=1), s.assign(grp=0)]).dropna(subset=['ex'])
        if d['grp'].nunique() < 2 or len(d) < 10:
            return np.nan
        return smf.ols('ex ~ grp', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['month'])[0]}).tvalues['grp']

    def book(ev):
        ev = ev.reset_index(drop=True)
        eng = SubsetEngine(ev, market, 252, ev['cost'].values)
        mask = np.ones(len(ev))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r).loc[act[act].index[0]:]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
        s = perf(p, rf)
        return {'CAGR': s.get('CAGR'), 'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown'),
                'avg positions': float(cnt[cnt > 0].mean())}

    rows = []
    for lab in ('1994–2004', '2005–2026', 'all'):
        x = E if lab == 'all' else E[E['period'] == lab]
        u, c = x[x['analysts'] == 0], x[x['analysts'] >= 1]
        lo, hi = x[x['analysts'] <= 1], x[x['analysts'] >= 4]
        r = {'period': lab, 'uncovered n': len(u), 'uncovered mean': u['ex'].mean(), 'uncovered median': u['ex'].median(),
             'covered n': len(c), 'covered mean': c['ex'].mean(), 'covered median': c['ex'].median(),
             'difference': u['ex'].mean() - c['ex'].mean(), 't (difference)': tdiff(u, c),
             '0-1 analysts mean': lo['ex'].mean(), '4+ analysts mean': hi['ex'].mean(), '4+ n': len(hi)}
        if lab == 'all':
            for k, v in book(u).items():
                r['uncovered book ' + k] = v
            for k, v in book(c).items():
                r['covered book ' + k] = v
        rows.append(r)
    R = pd.DataFrame(rows)
    g = R.set_index('period')
    R['adopted (pre-written)'] = bool(g.loc['all', 'difference'] >= 0.05 and g.loc['all', 't (difference)'] >= 2 and
                                      g.loc['1994–2004', 'difference'] > 0 and g.loc['2005–2026', 'difference'] > 0 and
                                      g.loc['all', 'uncovered book avg positions'] >= 6)
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'analyst_neglect.csv', index=False)
    LOG.info('analyst_neglect: done')
    return R
