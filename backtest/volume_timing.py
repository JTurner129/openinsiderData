"""Phase 34: buy each spin-off when its forced selling has dried up (volume signal) instead of on day 21.

Rules were written down and committed before this was run.

Events: Phase 23 (1994-2004) + Phase 21 (2005-2026) spin-offs, Compustat prices (1990+ cache), same costs.
Signal: V0 = average daily share volume over the first 5 trading days. From trading day 10 on, the first day whose
10-day average volume is <= V0 / 3 is the signal day; buy at the next day's open. If no signal by trading day 63,
buy on day 63. Hold 252 trading days. Benchmark rule: the existing day-21 entry, same events.
Reported: per-trade 12-month excess over SPY for both rules; paired difference (signal minus day-21) with
month-clustered t; average signal day; stand-alone books (CAGR, Sharpe, max DD) for 1994-2004, 2005-2026, all.
Pre-written bar: paired mean difference > 0 with t >= 2 over all years AND the signal book's CAGR beats the
day-21 book in BOTH periods. Otherwise day 21 stays.
"""
import os

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger

LOG = get_logger('volume_timing')


def run():
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    from .combos import SubsetEngine
    from .filters import Market
    from .prices import load_price
    from .spinoff_grid import spy_all
    from .strategies import perf
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy_r = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    cols = ['price_ticker', 'gvkey', 'event', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(period='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(period='2005–2026')], ignore_index=True)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    sig_dates, sig_day = [], []
    for _, e in E.iterrows():
        p = load_price(e['price_ticker'])
        h = p.loc[pd.Timestamp(e['event']):]
        v = h['volume']
        d = None
        if len(h) > 64 and v.iloc[:5].notna().sum() >= 3:
            v0 = v.iloc[:5].mean()
            ma = v.rolling(10, min_periods=7).mean()
            for t in range(10, 63):
                if pd.notna(ma.iloc[t]) and ma.iloc[t] <= v0 / 3:
                    d = t + 1
                    break
            d = d if d is not None else 63
            sig_dates.append(h.index[d])
            sig_day.append(d)
        else:
            sig_dates.append(pd.NaT)
            sig_day.append(np.nan)
    E['signal_date'] = sig_dates
    E['signal_day'] = sig_day
    E = E.dropna(subset=['signal_date']).reset_index(drop=True)

    def excess(date_col):
        out = []
        for _, e in E.iterrows():
            a = load_price(e['price_ticker'])['adj_close'].loc[e[date_col]:].iloc[:253]
            s = px.loc[e[date_col]:].iloc[:253]
            out.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost']) - (s.iloc[-1] / s.iloc[0] - 1) if len(a) > 1 else np.nan)
        return np.array(out)

    E['ex21'] = excess('entry_date')
    E['exsig'] = excess('signal_date')
    E['diff'] = E['exsig'] - E['ex21']
    E['month'] = E['entry_date'].dt.to_period('M').astype(str)
    market = Market(idx.values)

    def cl(x, m):
        x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
        m = pd.Series(np.asarray(m)).reset_index(drop=True)
        ok = x.notna()
        x, m = x[ok], m[ok]
        g = (x - x.mean()).groupby(m).sum()
        return x.mean() / (np.sqrt((g ** 2).sum()) / len(x))

    def book(ev, col, a, b):
        ev = ev.reset_index(drop=True).rename(columns={col: 'entry'})
        ev = ev.drop(columns=['entry_date'], errors='ignore').rename(columns={'entry': 'entry_date'})
        eng = SubsetEngine(ev, market, 252, ev['cost'].values)
        mask = np.ones(len(ev))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r)
        p = p.loc[max(pd.Timestamp(a), act[act].index[0]):b]
        s = perf(p, rf)
        return s.get('CAGR'), s.get('Sharpe'), s.get('max drawdown')

    rows = []
    for lab, a, b in (('1994–2004', '1994-01-01', '2004-12-31'), ('2005–2026', '2005-01-01', None), ('all', '1994-01-01', None)):
        x = E if lab == 'all' else E[E['period'] == lab]
        b21 = book(x, 'entry_date', a, b)
        bsg = book(x, 'signal_date', a, b)
        rows.append({'period': lab, 'n': len(x), 'avg signal day': x['signal_day'].mean(),
                     'share hitting 63-day cap': (x['signal_day'] == 63).mean(),
                     'day-21 mean excess': x['ex21'].mean(), 'signal mean excess': x['exsig'].mean(),
                     'day-21 median': x['ex21'].median(), 'signal median': x['exsig'].median(),
                     'paired difference': x['diff'].mean(), 't (paired)': cl(x['diff'], x['month']),
                     'day-21 book CAGR': b21[0], 'signal book CAGR': bsg[0], 'day-21 Sharpe': b21[1],
                     'signal Sharpe': bsg[1], 'day-21 max DD': b21[2], 'signal max DD': bsg[2]})
    R = pd.DataFrame(rows)
    g = R.set_index('period')
    R['adopted (pre-written)'] = bool(g.loc['all', 'paired difference'] > 0 and g.loc['all', 't (paired)'] >= 2 and
                                      all(g.loc[p, 'signal book CAGR'] > g.loc[p, 'day-21 book CAGR'] for p in ('1994–2004', '2005–2026')))
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'volume_timing.csv', index=False)
    LOG.info('volume_timing: done')
    return R
