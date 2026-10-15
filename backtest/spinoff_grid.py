"""Phase 24: is "buy 21 trading days after listing, hold 12 months" a sensible choice? A sensitivity map.

Rules were written down and committed before this was run.

Purpose: check that the result does not hinge on the exact entry delay or holding period, NOT to pick the
best cell (picking the best of 25 cells on the same data would overfit).
Events: the tradeable spin-offs already defined (Phase 23 holdout list, entries 1994-2004; Phase 21 list,
entries 2005-2026), kept fixed across cells so only timing changes. Same costs (each event's estimated spread +
0.15% FX each way). Prices from the 1990+ Compustat cache (CS_CACHE_NAME=prices_cs_full); SPY from Yahoo.
Grid: entry delay after the first trading day = 0, 10, 21, 42, 63 trading days (0 = next day's open after the
first close); holding period = 6, 9, 12, 18, 24 months (126, 189, 252, 378, 504 trading days).
Equal weight per entry, buy and hold; days with nothing held earn SPY.
Reported per cell and period (1994-2004 holdout, 2005-2026): spin-off book CAGR and Sharpe, SPY over the same
days, and average positions held.
Decision rule (pre-written): keep the current rule (21 days, 12 months) unless another cell beats it on BOTH
CAGR and Sharpe in BOTH periods, with Sharpe higher by at least 0.10 in each. Even then the change is flagged as
data-mined (25 cells tried) and the current rule stays the default until confirmed on new data.
"""
import os

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('spinoff_grid')
DELAYS = (0, 10, 21, 42, 63)
HOLDS = {'6m': 126, '9m': 189, '12m': 252, '18m': 378, '24m': 504}
PERIODS = {'1994–2004': ('1994-01-01', '2004-12-31'), '2005–2026': ('2005-01-01', None)}


def spy_all():
    path = CACHE / 'spy_1993_2026.parquet'
    if path.exists():
        return pd.read_parquet(path)['adj_close']
    import yfinance as yf
    d = yf.download('SPY', start='1993-01-01', auto_adjust=False, progress=False)['Adj Close']
    d = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d.rename('adj_close').to_frame().to_parquet(path)
    return d


def run(cfg=None):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    from .combos import SubsetEngine
    from .filters import Market
    from .forced_flows import next_day
    from .strategies import perf
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy = px.pct_change().reindex(idx).fillna(0.0)
    F = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')
    rf = F['RF'].reindex(idx).fillna(0.0)
    market = Market(idx.values)
    cols = ['price_ticker', 'gvkey', 'event', 'cost']
    H = pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(sample='1994–2004')
    L = pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(sample='2005–2026')
    E0 = pd.concat([H, L], ignore_index=True)
    E0['event'] = pd.to_datetime(E0['event'])
    rows = []
    for d in DELAYS:
        E = E0.copy()
        E['entry_date'] = [next_day(idx, ev, d + 1 if d == 0 else d) for ev in E['event']]
        E = E.dropna(subset=['entry_date']).reset_index(drop=True)
        for hn, hd in HOLDS.items():
            for per, (a, b) in PERIODS.items():
                ev = E[E['sample'] == per].reset_index(drop=True)
                eng = SubsetEngine(ev, market, hd, ev['cost'].values)
                mask = np.ones(len(ev))
                act = pd.Series(eng.active(mask), index=idx)
                p = eng.returns(mask, rf.values).reindex(idx).where(act, spy)
                cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx)
                start = max(pd.Timestamp(a), act[act].index[0])
                p, c = p.loc[start:b], cnt.loc[start:b]
                s, m = perf(p, rf), perf(spy.loc[p.index], rf)
                rows.append({'entry delay (days)': d, 'hold': hn, 'period': per, 'CAGR': s.get('CAGR'),
                             'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown'), 'SPY CAGR': m.get('CAGR'),
                             'SPY Sharpe': m.get('Sharpe'), 'avg positions': float(c[c > 0].mean())})
            LOG.info(f'grid: delay {d}, hold {hn} done')
    r = pd.DataFrame(rows)
    base = r[(r['entry delay (days)'] == 21) & (r['hold'] == '12m')].set_index('period')
    better = []
    for (d, h), g in r.groupby(['entry delay (days)', 'hold']):
        g = g.set_index('period')
        if (d, h) != (21, '12m') and all(g.loc[p, 'CAGR'] > base.loc[p, 'CAGR'] and g.loc[p, 'Sharpe'] >= base.loc[p, 'Sharpe'] + 0.10
                                         for p in PERIODS):
            better.append(f'{d}d/{h}')
    r['cells beating current rule in both periods (data-mined)'] = ', '.join(better) or 'none'
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'spinoff_grid.csv', index=False)
    LOG.info(f'grid: done; cells beating the current rule in both periods: {better or "none"}')
    return r


if __name__ == '__main__':
    run()
