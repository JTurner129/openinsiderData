"""Phase 22 (Phase C): a small, fixed-weight combined portfolio of the ingredients that passed on their own.

Rules were written down and committed before this was run. No weights are searched or tuned.

Ingredients (daily return series already produced by earlier phases, same costs: half-spread each way +
0.15% FX each way; days with nothing held earn SPY):
  core     QQQ (C1, C3) or SPY (C2)
  SPIN     spin-off book (Phase 21: every Form 10-12B spin-off, enter 21 trading days after first trade,
           hold 12 months, equal weight)
  INS      S&P 500 officer/director open-market buy >= $100K + positive earnings surprise, hold 6 months
           (Phase 11, L2)
Portfolios, rebalanced to target on the first trading day of each month, 0.15% per side on the amount moved:
  C1 (MAIN) 60% QQQ + 20% SPIN + 20% INS
  C2        60% SPY + 20% SPIN + 20% INS
  C3        50% QQQ + 25% SPIN + 25% INS
  C4 (diagnostic) C1 with 10% of the QQQ moved to thrift conversions (Phase 14, 1-year hold)
Stress tests on C1:
  S1 extra 1.0% a year cost drag (all ingredients already passed or were reported at costs x2 on their own)
  S2 SPIN rebuilt without its 5 best spin-offs (by 12-month return)
  S3 without the single best calendar year of C1's excess over QQQ (the year is dropped from both)
Reported for all years (from the first day all ingredients exist), 2006-15 and 2016+: CAGR, Sharpe, max
drawdown, worst calendar year, and the same for SPY and QQQ; plus yearly returns.
Pre-written bar for "use it": C1 beats QQQ on CAGR in BOTH 2006-15 and 2016+, has a higher Sharpe than QQQ
over all years, and still beats QQQ on CAGR (all years) under S1, S2 and S3.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config
from .spinoff_core import SIDE

LOG = get_logger('combined')
PERIODS = (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None)))


def mix(parts, weights, side=SIDE):
    """Monthly rebalanced mix of several daily return series (aligned, no gaps)."""
    R = pd.concat(parts, axis=1).dropna()
    w0 = np.asarray(weights, float)
    w, month, out = w0.copy(), None, []
    for d, row in zip(R.index, R.values):
        cost = 0.0
        if d.month != month:
            if month is not None:
                cost = np.abs(w - w0).sum() * side
            w, month = w0.copy(), d.month
        r = float(w @ row) - cost
        out.append(r)
        w = w * (1 + row) / (1 + r + cost)
    return pd.Series(out, index=R.index)


def spin_book(drop_best=0):
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market
    from .prices import load_price
    cfg = load_config()
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    E = pd.read_parquet(DATA / 'spinoff_events.parquet')
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    if drop_best:
        E = E.drop(E.sort_values('ret12', ascending=False).index[:drop_best])
    E = E.reset_index(drop=True)
    eng = SubsetEngine(E, Market(spy_px.index.values), 252, E['cost'].values)
    mask = np.ones(len(E))
    act = pd.Series(eng.active(mask), index=idx)
    p = eng.returns(mask, rf.values).reindex(idx).where(act, spy)
    return p.loc[act[act].index[0]:]


def stats(r, rf):
    from .strategies import perf
    s = perf(r, rf)
    yr = (1 + r).groupby(r.index.year).prod() - 1
    s['worst year'] = yr.min() if len(yr) else np.nan
    return s


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    qqq = load_price('QQQ')['adj_close'].pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    spin = spin_book()
    spin5 = spin_book(5)
    ins = pd.read_parquet(DATA / 'largecap_daily.parquet')['L2 S&P 500 + positive earnings surprise|6m']
    thrift = pd.read_parquet(DATA / 'niche_daily.parquet')['T1 Thrift conversions|1y']
    ins.index = pd.to_datetime(ins.index)
    thrift.index = pd.to_datetime(thrift.index)
    start = max(spin.index[0], ins.dropna().index[0])
    sl = slice(start, None)
    P = {
        'C1 (MAIN) 60% QQQ + 20% spin-offs + 20% insiders': mix([qqq[sl], spin[sl], ins[sl]], [.6, .2, .2]),
        'C2 60% SPY + 20% spin-offs + 20% insiders': mix([spy[sl], spin[sl], ins[sl]], [.6, .2, .2]),
        'C3 50% QQQ + 25% spin-offs + 25% insiders': mix([qqq[sl], spin[sl], ins[sl]], [.5, .25, .25]),
        'C4 50% QQQ + 20% spin-offs + 20% insiders + 10% thrifts':
            mix([qqq[sl], spin[sl], ins[sl], thrift.reindex(idx)[sl].fillna(spy[sl])], [.5, .2, .2, .1]),
    }
    M = 'C1 (MAIN) 60% QQQ + 20% spin-offs + 20% insiders'
    P['S1 C1 with extra 1%/yr cost'] = P[M] - 0.01 / 252
    P['S2 C1 without 5 best spin-offs'] = mix([qqq[sl], spin5[sl], ins[sl]], [.6, .2, .2])
    yex = (1 + P[M]).groupby(P[M].index.year).prod() - (1 + qqq.loc[P[M].index]).groupby(P[M].index.year).prod()
    best = int(yex.idxmax())
    keep = P[M].index.year != best
    P[f'S3 C1 without best year ({best})'] = P[M][keep]
    P['SPY'] = spy.loc[P[M].index]
    P['QQQ'] = qqq.loc[P[M].index]
    rows = []
    for name, p in P.items():
        for lab, s in PERIODS:
            st = stats(p.loc[s], rf)
            qs = stats(qqq.loc[p.index].loc[s], rf)
            rows.append({'portfolio': name, 'period': lab, 'CAGR': st.get('CAGR'), 'QQQ CAGR (same days)': qs.get('CAGR'),
                         'Sharpe': st.get('Sharpe'), 'max DD': st.get('max drawdown'), 'worst year': st.get('worst year')})
    r = pd.DataFrame(rows)
    g = lambda n, p, k: r.loc[(r['portfolio'] == n) & (r['period'] == p), k].iloc[0]
    S3 = f'S3 C1 without best year ({best})'
    bar = all(g(M, p, 'CAGR') > g('QQQ', p, 'CAGR') for p in ('2006–15', '2016+')) \
        and g(M, 'all', 'Sharpe') > g('QQQ', 'all', 'Sharpe') \
        and all(g(n, 'all', 'CAGR') > g(n, 'all', 'QQQ CAGR (same days)')
                for n in ('S1 C1 with extra 1%/yr cost', 'S2 C1 without 5 best spin-offs', S3))
    r['pre-written bar met (C1)'] = bar
    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in P.items() if not k.startswith('S3')})
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'combined.csv', index=False)
    yearly.to_csv(RESULTS / 'combined_yearly.csv')
    pd.DataFrame({k: v for k, v in P.items() if not k.startswith('S3')}).to_parquet(DATA / 'combined_daily.parquet')
    LOG.info(f'combined: done; pre-written bar met = {bar}')
    return r, bar


if __name__ == '__main__':
    run()
