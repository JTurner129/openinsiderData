"""Phase 16: an adapting strategy — rotate monthly into whichever strategies have been working lately
("strategy momentum", Ehsani & Linnainmaa 2022). Long-only.

Rules were written down and committed before this was run.

Library: EVERY long-only strategy already tested whose daily return series exists (failures included), plus
SPY, QQQ, RSP (equal-weight S&P 500) and IWM (small caps). Strategy series already include their trading
costs; switching between strategies costs an extra 0.3% of the amount moved (spread + FX on the turnover).

Main rule (fixed in advance): at each month-end, rank every series by its total return over the prior
12 months (only data up to that day); hold the top 3 in equal weight for the next month. A series needs
12 months of history to be eligible.
Robustness (reported, not used to choose): lookback 6 / 24 months; top 1 / 5; ranking by Sharpe instead of
return.
Reported against SPY and QQQ: CAGR, Sharpe, max drawdown, all years / 2006-15 / 2016+, average holdings
turnover, and which series were held most.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config

LOG = get_logger('rotation')
SWITCH_COST = 0.003
SOURCES = ['largecap_daily', 'nasdaq_daily', 'long_only_daily', 'deep_value_daily', 'niche_daily']


def library(idx):
    from .prices import download_prices, load_price
    lib = {}
    for f in SOURCES:
        p = DATA / f'{f}.parquet'
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        for c in d.columns:
            if c in ('QQQ', 'Equal-weight Nasdaq 100'):
                continue
            s = d[c].reindex(idx)
            if s.notna().sum() > 500:
                lib[f'{f.replace("_daily", "")}: {c}'] = s
    missing = [t for t in ('SPY', 'QQQ', 'RSP', 'IWM') if load_price(t) is None]
    if missing:
        download_prices(missing, refresh_missing=True)
    for t in ('SPY', 'QQQ', 'RSP', 'IWM'):
        lib[t] = load_price(t)['adj_close'].pct_change().reindex(idx)
    return pd.DataFrame(lib)


def rotate(R, lookback, top, by='return'):
    months = R.index.to_period('M')
    ends = R.index.to_series().groupby(months).max()
    out = pd.Series(0.0, index=R.index)
    held, w_prev, turnover = {}, pd.Series(dtype=float), []
    for k in range(len(ends) - 1):
        d0, d1 = ends.iloc[k], ends.iloc[k + 1]
        hist = R.loc[:d0].iloc[-21 * lookback:]
        ok = hist.notna().sum() >= 21 * lookback * 0.95
        if ok.sum() < top:
            continue
        h = hist.loc[:, ok].fillna(0.0)
        if by == 'sharpe':
            score = h.mean() / h.std().replace(0, np.nan)
        else:
            score = (1 + h).prod() - 1
        pick = score.sort_values(ascending=False).index[:top]
        w = pd.Series(1.0 / top, index=pick)
        moved = w.subtract(w_prev, fill_value=0).abs().sum() / 2 if len(w_prev) else 1.0
        turnover.append(moved)
        nxt = R.loc[d0:d1].iloc[1:][pick].fillna(0.0)
        # buy-and-hold within the month: weights drift with returns
        v = (1 + nxt).cumprod() * w
        port = v.sum(axis=1)
        r = port / port.shift(1).fillna(1.0) - 1
        r.iloc[0] -= moved * SWITCH_COST
        out.loc[r.index] = r.values
        for c in pick:
            held[c] = held.get(c, 0) + 1
        w_prev = w
    first = out.ne(0).idxmax()
    return out.loc[first:], held, float(np.mean(turnover)) if turnover else np.nan


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    from .strategies import perf
    idx = load_price('SPY').index
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    R = library(idx)
    LOG.info(f'rotation: library of {R.shape[1]} series')
    spy, qqq = R['SPY'], R['QQQ']
    rows, curves, helds = [], {}, {}
    variants = [('MAIN: 12m return, top 3', 12, 3, 'return')] + \
        [(f'12m return, top {k}', 12, k, 'return') for k in (1, 5)] + \
        [(f'{m}m return, top 3', m, 3, 'return') for m in (6, 24)] + [('12m Sharpe, top 3', 12, 3, 'sharpe')]
    for name, lb, top, by in variants:
        p, held, to = rotate(R, lb, top, by)
        curves[name] = p
        helds[name] = held
        for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s_ = perf(p.loc[sl], rf)
            b_ = perf(spy.loc[p.index[0]:].loc[sl].dropna(), rf)
            q_ = perf(qqq.loc[p.index[0]:].loc[sl].dropna(), rf)
            rows.append({'rule': name, 'period': lab, 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'),
                         'QQQ CAGR': q_.get('CAGR'), 'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'),
                         'QQQ Sharpe': q_.get('Sharpe'), 'max DD': s_.get('max drawdown'),
                         'SPY max DD': b_.get('max drawdown'), 'avg monthly turnover': to})
        LOG.info(f'rotation: {name} done')
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'rotation.csv', index=False)
    pd.Series(helds['MAIN: 12m return, top 3']).sort_values(ascending=False).rename('months held').to_csv(
        RESULTS / 'rotation_holdings.csv')
    pd.DataFrame(curves).to_parquet(DATA / 'rotation_daily.parquet')
    LOG.info('rotation: done')
    return r


if __name__ == '__main__':
    run()
