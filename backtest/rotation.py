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
Amendment 1 (written after seeing the first run, committed before re-running): the 12-month-Sharpe variant
did best but leaned on a tiny strategy (biotech below cash + insiders, ~9 stocks: +372% in 2009). New main rule
for a second, confirmatory run: rank by 12-month Sharpe, hold the top 3, monthly; and the library is limited
to strategies that hold at least 15 stocks on average (index funds always eligible). Robustness: top 1 / 5,
lookback 6 / 24 months, and the same rule with every year's single best calendar year removed.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config

LOG = get_logger('rotation')
SWITCH_COST = 0.003
SOURCES = ['largecap_daily', 'nasdaq_daily', 'long_only_daily', 'deep_value_daily', 'niche_daily']


MIN_HOLDINGS = 15


def avg_holdings():
    """Average number of open positions per strategy series, from the result tables that report it."""
    from .common import RESULTS
    out = {}
    for f, col_name, col_n in (('niche.csv', 'niche', 'avg open'),):
        p = RESULTS / f
        if p.exists():
            d = pd.read_csv(p)
            for _, r in d[d['period'] == 'all'].iterrows():
                out[f"niche: {r[col_name]}|{r['hold']}"] = r[col_n]
    return out


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


def holdings_filter(R, cfg):
    """Keep series whose portfolios hold >= MIN_HOLDINGS stocks on average (measured from the trade logs)."""
    from .combos import SubsetEngine
    counts = avg_holdings()
    keep = []
    for c in R.columns:
        if c in ('SPY', 'QQQ', 'RSP', 'IWM'):
            keep.append(c)
            continue
        n = counts.get(c)
        if n is None:
            n = estimate_holdings(c)
        if n is not None and n >= MIN_HOLDINGS:
            keep.append(c)
        LOG.info(f'rotation: {c}: average holdings {n}')
    return R[keep]


def estimate_holdings(name):
    """Average open positions from each strategy's entry list and hold length."""
    import re as _re
    hold = 126 if name.endswith('|6m') else 252 if name.endswith('|1y') else 252
    try:
        if name.startswith('deep_value:'):
            E = pd.read_parquet(DATA / 'deep_value_entries.parquet')
            strat = name.split(': ', 1)[1].split('|')[0]
            if 'max 20' in name:
                return 20.0 * 0.9
            d = pd.to_datetime(E.loc[E['strategy'] == strat, 'entry_date'])
        elif name.startswith('largecap:') or name.startswith('nasdaq:'):
            S = pd.read_parquet(DATA / 'largecap_signals_any.parquet') if 'insiders' not in name else \
                pd.read_parquet(DATA / 'largecap_signals_cluster.parquet')
            from .phase_b import members_on
            cfg = load_config()
            idxid = ['000003'] if name.startswith('largecap:') else ['000208']
            m = members_on(list(zip(S['gvkey'], pd.to_datetime(S['signal_date']))), idxid, cfg)
            if 'surprise' in name:
                m = m & (S['sue_recent'] > 0).values
            d = pd.to_datetime(S.loc[m, 'entry_date'])
        elif name.startswith('long_only:'):
            F = pd.read_parquet(DATA / 'signal_features.parquet')
            fe = pd.read_parquet(DATA / 'signal_fundamentals.parquet')
            F = F[F['screen'] == 'cluster3'].merge(fe.drop(columns=['price_ticker']), on=['screen', 'signal_date', 'issuer_cik'])
            bd = (F['ret_prior_6m'] <= -0.5) & (F['adv20'] >= 1e5)
            rules = {'All clusters': F.index == F.index, 'Clusters + positive surprise': F['sue_recent'] > 0,
                     'Clusters + F-score<=3': F['fscore'] <= 3, 'Big-drop tradeable': bd,
                     'Big-drop + positive surprise': bd & (F['sue_recent'] > 0), 'Big-drop + F-score<=3': bd & (F['fscore'] <= 3),
                     'Big-drop + leverage<20%': bd & (F['leverage'] < 0.2)}
            key = name.split(': ', 1)[1].split('|')[0]
            d = pd.to_datetime(F.loc[rules[key], 'entry_date'])
        else:
            return None
    except Exception as e:
        LOG.warning(f'rotation: holdings for {name}: {e}')
        return None
    if len(d) == 0:
        return 0.0
    days = pd.bdate_range(d.min(), d.max() + pd.Timedelta(days=int(hold * 1.45)))
    starts = np.searchsorted(days.values, np.sort(d.values))
    cnt = np.zeros(len(days) + hold + 1)
    np.add.at(cnt, starts, 1)
    np.add.at(cnt, np.minimum(starts + hold, len(cnt) - 1), -1)
    open_ = np.cumsum(cnt)[:len(days)]
    return float(open_[open_ > 0].mean())


def run_confirmatory(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    from .strategies import perf
    idx = load_price('SPY').index
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    R = holdings_filter(library(idx), cfg)
    LOG.info(f'rotation (confirmatory): {R.shape[1]} series with >= {MIN_HOLDINGS} holdings')
    spy, qqq = R['SPY'], R['QQQ']
    rows, curves = [], {}
    variants = [('CONFIRM MAIN: 12m Sharpe, top 3', 12, 3, 'sharpe')] + \
        [(f'12m Sharpe, top {k}', 12, k, 'sharpe') for k in (1, 5)] + \
        [(f'{m}m Sharpe, top 3', m, 3, 'sharpe') for m in (6, 24)]
    for name, lb, top, by in variants:
        p, held, to = rotate(R, lb, top, by)
        curves[name] = p
        yr = (1 + p).groupby(p.index.year).prod()
        best = yr.idxmax()
        for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s_ = perf(p.loc[sl], rf)
            b_ = perf(spy.loc[p.index[0]:].loc[sl].dropna(), rf)
            q_ = perf(qqq.loc[p.index[0]:].loc[sl].dropna(), rf)
            rows.append({'rule': name, 'period': lab, 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'),
                         'QQQ CAGR': q_.get('CAGR'), 'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'),
                         'max DD': s_.get('max drawdown'), 'best year': int(best), 'best year return': yr.max() - 1,
                         'CAGR without best year': yr.drop(best).prod() ** (1 / (len(yr) - 1)) - 1,
                         'most held': ', '.join(k for k, _ in sorted(held.items(), key=lambda x: -x[1])[:3])})
        LOG.info(f'rotation (confirmatory): {name} done')
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / 'rotation_confirmatory.csv', index=False)
    pd.DataFrame(curves).to_parquet(DATA / 'rotation_confirmatory_daily.parquet')
    LOG.info('rotation (confirmatory): done')
    return r


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
