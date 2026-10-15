"""Phase 21: index-fund core + spin-off slice (follow-up to Phase 19, F3).

Rules were written down and committed before this was run.

Why: Phase 19 found spin-offs (entered 21 trading days after first trade, held 12 months) at 22.7% CAGR vs SPY
11.2%, but driven by a few huge winners (without the top 10: 10.1%). A 100% spin-off book is too lumpy to hold;
a fixed slice next to an index fund keeps the upside while a few years without a big winner cost little.

Spin-off book: exactly Phase 19 F3 events (same tradeable filter and costs: half-spread each way + 0.15% FX
each way), equal weight at entry, buy-and-hold; days with no position held earn the core index.
Portfolios (rebalanced to target on the first trading day of each month; 0.15% cost per side on the amount
moved, charged as a return drag):
  P1 (MAIN) 70% SPY + 30% spin-off book
  P2 80% SPY + 20%;   P3 50% SPY + 50%
  P4 70% QQQ + 30% spin-off book (QQQ also funds idle days)
Robustness, all on P1:
  R1 costs x2 (spread and FX doubled, rebalancing cost doubled)
  R2 drop the 5 best spin-offs by holding-period return (and R2b drop the 10 best)
  R3 enter 63 trading days after first trade instead of 21
  R4 hold 6 months instead of 12;  R5 hold 24 months
  R6 only spin-offs with market cap >= $1B at entry;  R7 only < $1B (diagnostic)
Reported for all years, 2006-15 and 2016+: CAGR, Sharpe, max drawdown, against SPY and QQQ; plus the number
of positions held on an average day and the share of P1's excess return over SPY coming from the best 5 names.
Pre-written bar for "use it": P1 beats SPY on CAGR and Sharpe in BOTH 2006-15 and 2016+, and still beats SPY
on CAGR (all years) under R1 and under R2. QQQ is reported but not part of the bar.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config, save_table
from .compustat import CS_CACHE

LOG = get_logger('spinoff_core')
SIDE = 0.0015
PERIODS = (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None)))


def blend(book, core, w, side=SIDE):
    """Monthly rebalanced mix: w in book, 1-w in core. Drift within the month; trading cost at each rebalance."""
    out, wb = [], w
    month = None
    for d, rb, rc in zip(book.index, book.values, core.values):
        cost = 0.0
        if d.month != month:
            cost = abs(wb - w) * 2 * side if month is not None else 0.0
            wb, month = w, d.month
        r = wb * rb + (1 - wb) * rc - cost
        out.append(r)
        wb = wb * (1 + rb) / (1 + r + cost) if (1 + r + cost) else w
    return pd.Series(out, index=book.index)


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market
    from .forced_flows import next_day
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    cal = spy_px.index.values
    idx = pd.DatetimeIndex(cal)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    qqq = load_price('QQQ')['adj_close'].pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    market = Market(cal)
    E = pd.read_parquet(DATA / 'forced_flow_events.parquet')
    E = E[E['event_type'] == 'F3 Spin-offs'].reset_index(drop=True)
    E['entry_date'] = pd.to_datetime(E['entry_date'])

    # holding-period return per spin-off (12 months), for R2 and the concentration diagnostic
    def hold_ret(e, n=252):
        a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:n + 1]
        return a.iloc[-1] / a.iloc[0] - 1 if len(a) > 1 else np.nan
    E['ret12'] = [hold_ret(e) for _, e in E.iterrows()]
    mcap = []
    for _, e in E.iterrows():
        px = load_price(e['price_ticker'])
        w = px.loc[:e['entry_date']].tail(1) if px is not None else None
        mc = np.nan
        if w is not None and len(w) and 'shares_out' in w:
            mc = float(w['raw_close'].iloc[0] * w['shares_out'].iloc[0])
        mcap.append(mc)
    E['mcap'] = mcap
    LOG.info(f'spinoff_core: {len(E)} spin-offs, market cap known for {E["mcap"].notna().sum()}')

    def book(events, hold=252, cost_mult=1.0, core=spy):
        eng = SubsetEngine(events, market, hold, events['cost'].values * cost_mult)
        mask = np.ones(len(events))
        p = eng.returns(mask, rf.values)
        act = pd.Series(eng.active(mask), index=idx)
        p = p.reindex(idx).where(act, core)
        start = act[act].index[0]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx)
        return p.loc[start:], cnt.loc[start:]

    late = E.copy()
    late['entry_date'] = [next_day(idx, d, 42) for d in late['entry_date']]
    late = late.dropna(subset=['entry_date'])
    top = E.sort_values('ret12', ascending=False)
    variants = {
        'P1 (MAIN) 70% SPY + 30% spin-offs': (E, 252, 1, spy, 0.30, 1),
        'P2 80% SPY + 20% spin-offs': (E, 252, 1, spy, 0.20, 1),
        'P3 50% SPY + 50% spin-offs': (E, 252, 1, spy, 0.50, 1),
        'P4 70% QQQ + 30% spin-offs': (E, 252, 1, qqq, 0.30, 1),
        'R1 P1 costs x2': (E, 252, 2, spy, 0.30, 2),
        'R2 P1 without 5 best spin-offs': (E.drop(top.index[:5]), 252, 1, spy, 0.30, 1),
        'R2b P1 without 10 best spin-offs': (E.drop(top.index[:10]), 252, 1, spy, 0.30, 1),
        'R3 P1 enter 63 days after first trade': (late, 252, 1, spy, 0.30, 1),
        'R4 P1 hold 6 months': (E, 126, 1, spy, 0.30, 1),
        'R5 P1 hold 24 months': (E, 504, 1, spy, 0.30, 1),
        'R6 P1 only spin-offs >= $1B': (E[E['mcap'] >= 1e9], 252, 1, spy, 0.30, 1),
        'R7 P1 only spin-offs < $1B': (E[E['mcap'] < 1e9], 252, 1, spy, 0.30, 1),
        'spin-off book alone (100%)': (E, 252, 1, spy, 1.0, 1),
    }
    rows, curves = [], {}
    for name, (ev, hold, cm, core, w, rm) in variants.items():
        ev = ev.reset_index(drop=True)
        if len(ev) < 20:
            LOG.warning(f'spinoff_core: {name}: only {len(ev)} events, skipped')
            continue
        b, cnt = book(ev, hold, cm, core)
        c = core.loc[b.index[0]:]
        p = blend(b, c, w, SIDE * rm) if w < 1 else b
        curves[name] = p
        for lab, sl in PERIODS:
            s_, b_, q_ = perf(p.loc[sl], rf), perf(spy.loc[p.index[0]:].loc[sl], rf), perf(qqq.loc[p.index[0]:].loc[sl], rf)
            rows.append({'portfolio': name, 'period': lab, 'spin-offs': len(ev),
                         'avg positions held': float(cnt.loc[sl][cnt.loc[sl] > 0].mean()),
                         'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'), 'QQQ CAGR': q_.get('CAGR'),
                         'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'), 'QQQ Sharpe': q_.get('Sharpe'),
                         'max DD': s_.get('max drawdown'), 'SPY max DD': b_.get('max drawdown'),
                         'QQQ max DD': q_.get('max drawdown')})
        LOG.info(f'spinoff_core: {name} done')
    r = pd.DataFrame(rows)
    g = lambda n, p, k: r.loc[(r['portfolio'] == n) & (r['period'] == p), k].iloc[0]
    M = 'P1 (MAIN) 70% SPY + 30% spin-offs'
    bar = all(g(M, p, 'CAGR') > g(M, p, 'SPY CAGR') and g(M, p, 'Sharpe') > g(M, p, 'SPY Sharpe') for p in ('2006–15', '2016+')) \
        and g('R1 P1 costs x2', 'all', 'CAGR') > g('R1 P1 costs x2', 'all', 'SPY CAGR') \
        and g('R2 P1 without 5 best spin-offs', 'all', 'CAGR') > g('R2 P1 without 5 best spin-offs', 'all', 'SPY CAGR')
    r['pre-written bar met (P1)'] = bar
    ex_all = g(M, 'all', 'CAGR') - g(M, 'all', 'SPY CAGR')
    r['share of P1 excess CAGR from best 5'] = (g(M, 'all', 'CAGR') - g('R2 P1 without 5 best spin-offs', 'all', 'CAGR')) / ex_all \
        if ex_all else np.nan
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'spinoff_core.csv', index=False)
    pd.DataFrame(curves).to_parquet(DATA / 'spinoff_core_daily.parquet')
    save_table(E, DATA / 'spinoff_events')
    LOG.info(f'spinoff_core: done; pre-written bar met = {bar}')
    return r, bar


if __name__ == '__main__':
    run()
