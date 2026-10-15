"""Phase 8: ways to turn the insider findings into something that could beat SPY.

Rules were written down and committed before any of these were run.

A1  Market timing from aggregate insider buying.
    Breadth = number of distinct companies (funds and trusts excluded) with an open-market purchase
    of at least $10K filed in the trailing 30 calendar days. Each day it is ranked against the prior
    504 trading days (only data known that day). A signal fires when breadth is in the top 10% of
    that history. Also tested: buy/sell breadth ratio (companies with purchases divided by companies
    with purchases or open-market sales) with the same top-10% rule.
    After a signal the "on" window lasts 126 trading days (extended by later signals).
    Portfolios, all rebalanced daily, cash earns the risk-free rate:
      T0  100% SPY
      T1  100% SPY, 150% SPY while on (the extra 50% borrowed at risk-free + 0.5%)
      T2  60% SPY / 40% cash, 100% SPY while on
      T3  100% SPY while on, 100% cash otherwise
    Evaluation starts once 504 days of history exist (January 2008).

A2  SPY plus a slice of the tradeable big-drop strategy (clusters >= 3 insiders, stock down >= 50%
    in 6 months, >= $100K a day traded, 6-month holds; the matched test's calendar series).
    Weights 10%, 20%, 30%, rebalanced monthly. When the slice has no positions its weight sits in SPY.

A3  Big-drop signals hedged by shorting their 5 matched look-alikes, after an estimated borrow fee
    from each look-alike's short interest at entry (shares short / shares outstanding):
    < 5%: 0.5% a year; 5-10%: 3%; 10-20%: 10%; >= 20%: 25%; no data: 10%.
    Reported per trade (6-month hold) with month-clustered t-stats, all years and 2016+.

A5  (added later, rules fixed before running) SPY plus the hedged big-drop book as an overlay:
    100% SPY + k x (tradeable big-drop signals minus their matched look-alikes, the matched test's daily
    calendar series), k = 0.5 and 1.0, minus a borrow fee of 5.4% a year on the short side (the A3
    average of 2.7% per 6 months) while the book holds positions. Shorts are assumed to fund the longs.
    Reported against SPY: CAGR, volatility, Sharpe, max drawdown, all years / to 2015 / 2016+.

A4  Entry month: the same big-drop per-trade edge for entries from 15 November to 31 December
    (tax-loss selling season) against all other months.
"""
import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table

LOG = get_logger('strategies')
LOOKBACK, TOP, ON_DAYS, WINDOW_D, MIN_BUY = 504, 0.90, 126, 30, 10_000
TRAIN_END = pd.Timestamp('2015-12-31')


def perf(r, rf):
    r = r.dropna()
    if len(r) < 20:
        return {}
    w = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    yrs = len(r) / 252
    return {'CAGR': w.iloc[-1] ** (1 / yrs) - 1, 'volatility': r.std() * np.sqrt(252),
            'Sharpe': ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
            'max drawdown': (w / w.cummax() - 1).min(), 'growth of $1': w.iloc[-1]}


def breadth(cal):
    """Daily breadth series (buys, sells) known at each close."""
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['issuer_cik', 'filing_date', 'value_clean'])
    funds = pd.read_parquet(DATA / 'excluded_funds_trusts.parquet')['issuer_cik'] if \
        (DATA / 'excluded_funds_trusts.parquet').exists() else pd.Series(dtype=float)
    pc = pc[(pc['value_clean'] >= MIN_BUY) & ~pc['issuer_cik'].isin(funds)]
    sales = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / 'sec_extra').glob('*q?.parquet'))])
    sales = sales[(sales['sale_value'] >= MIN_BUY) & ~sales['issuer_cik'].isin(funds)]
    out = {}
    for name, df in (('buy', pc), ('sell', sales)):
        d = df[['issuer_cik', 'filing_date']].dropna().drop_duplicates()
        d['filing_date'] = pd.to_datetime(d['filing_date']).dt.normalize()
        days = pd.date_range(d['filing_date'].min(), cal[-1], freq='D')
        # distinct companies in the trailing 30-day window, swept day by day
        by_day = d.groupby('filing_date')['issuer_cik'].apply(set).reindex(days)
        counts = []
        from collections import Counter, deque
        live, q = Counter(), deque()
        for day, s in by_day.items():
            s = s if isinstance(s, set) else set()
            q.append((day, s))
            for c in s:
                live[c] += 1
            while q and q[0][0] <= day - pd.Timedelta(days=WINDOW_D):
                _, old = q.popleft()
                for c in old:
                    live[c] -= 1
                    if live[c] == 0:
                        del live[c]
            counts.append(len(live))
        out[name] = pd.Series(counts, index=days)
    b = pd.DataFrame(out).reindex(pd.DatetimeIndex(cal), method='ffill')
    b['ratio'] = b['buy'] / (b['buy'] + b['sell'])
    return b


def rank_signal(x):
    """Percentile of today's value within the prior LOOKBACK days (today excluded), then the on-window."""
    vals = x.values
    pct = np.full(len(vals), np.nan)
    for i in range(LOOKBACK, len(vals)):
        hist = vals[i - LOOKBACK:i]
        pct[i] = (hist < vals[i]).mean()
    fire = pct >= TOP
    on = np.zeros(len(vals), bool)
    last = -10 ** 9
    for i in range(len(vals)):
        if fire[i]:
            last = i
        on[i] = i - last < ON_DAYS and last >= 0
    # trade on the next day: today's close is when the signal is known
    on = np.r_[False, on[:-1]]
    return pd.Series(pct, index=x.index), pd.Series(fire, index=x.index), pd.Series(on, index=x.index)


def timing(cal, spy, rf):
    b = breadth(cal)
    rows, curves, sig_tbl = [], {}, []
    start = b.index[LOOKBACK]
    for name in ('buy', 'ratio'):
        pct, fire, on = rank_signal(b[name])
        on = on.loc[start:]
        s, r = spy.loc[start:], rf.loc[start:]
        strat = {
            'T0 100% SPY': s,
            'T1 SPY, 150% while on': s + on * 0.5 * (s - r - 0.005 / 252),
            'T2 60% SPY, 100% while on': np.where(on, s, 0.6 * s + 0.4 * r),
            'T3 SPY while on, else cash': np.where(on, s, r),
        }
        for k, v in strat.items():
            v = pd.Series(np.asarray(v, float), index=s.index)
            curves[f'{name}|{k}'] = v
            for lab, sl in (('full', slice(None)), ('to 2015', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
                rows.append({'signal': 'Buy breadth' if name == 'buy' else 'Buy/sell ratio', 'portfolio': k,
                             'period': lab, 'days on': float(on.loc[sl].mean()), **perf(v.loc[sl], rf)})
        # episodes: first day of each new on-window and SPY's next 126-day return
        starts = on & ~on.shift(1, fill_value=False)
        for d in on.index[starts.values]:
            i = spy.index.get_loc(d)
            fwd = (1 + spy.iloc[i:i + 126]).prod() - 1
            sig_tbl.append({'signal': name, 'on from': d.date(), 'SPY next 6 months': fwd,
                            'breadth': b.loc[d, name]})
    save_table(b.assign(date=b.index), DATA / 'insider_breadth')
    return pd.DataFrame(rows), pd.DataFrame(sig_tbl), curves


def sleeve(spy, rf):
    d = pd.read_parquet(DATA / 'matched_daily.parquet')
    sl = d['drop50liq|6m|signal'].reindex(spy.index)
    active = sl.notna()
    sl = sl.where(active, spy)
    rows, curves = [], {}
    month = spy.index.to_period('M')
    for w in (0.0, 0.1, 0.2, 0.3):
        # monthly rebalance: weights drift within the month
        out = []
        for _, idx in spy.groupby(month).groups.items():
            a, b_ = spy.loc[idx], sl.loc[idx]
            va, vb = (1 - w) * (1 + a).cumprod(), w * (1 + b_).cumprod()
            v = va + vb
            out.append(v / v.shift(1, fill_value=1.0) - 1)
        p = pd.concat(out).sort_index()
        p = p.loc[sl.index[active][0]:]
        name = f'{int(w * 100)}% big-drop slice'
        curves[name] = p
        for lab, s_ in (('full', slice(None)), ('to 2015', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            rows.append({'portfolio': name, 'period': lab, **perf(p.loc[s_], rf)})
    return pd.DataFrame(rows), curves


def borrow_fee(si):
    return np.select([si < 0.05, si < 0.10, si < 0.20, si >= 0.20], [0.005, 0.03, 0.10, 0.25], default=0.10)


def hedged(cfg):
    from .common import ROOT
    P = pd.read_parquet(DATA / 'matched_controls.parquet')
    F = pd.read_parquet(DATA / 'signal_features.parquet', columns=['screen', 'signal_date', 'issuer_cik', 'adv20'])
    P = P.merge(F, on=['screen', 'signal_date', 'issuer_cik'], how='left')
    P = P[(P['screen'] == 'cluster3') & (P['ret_prior_6m'] <= -0.5) & (P['adv20'] >= 100_000) & P['complete_6m']].copy()
    si_path = (ROOT / cfg['backtest'].get('wrds_dir', '../insider-wrds-data') / 'compustat_shortint.parquet').resolve()
    si = pd.read_parquet(si_path)
    si['sec'] = 'CS_' + si['gvkey'].astype(str) + '_' + si['iid'].astype(str)
    si['datadate'] = pd.to_datetime(si['datadate'])
    si = si.sort_values('datadate')
    by_sec = {k: g for k, g in si.groupby('sec')}
    shares = {}

    def short_pct(sec, d):
        g = by_sec.get(sec)
        if g is None:
            return np.nan
        g = g[(g['datadate'] <= d) & (g['datadate'] > d - pd.Timedelta(days=45))]
        if g.empty:
            return np.nan
        if sec not in shares:
            from .prices import load_price
            px = load_price(sec)
            shares[sec] = px['shares_out'] if px is not None and 'shares_out' in px else None
        so = shares[sec]
        if so is None:
            return np.nan
        so = so.loc[:d].dropna()
        return g['shortint'].iloc[-1] / so.iloc[-1] if len(so) and so.iloc[-1] > 0 else np.nan

    fees, pcts = [], []
    for _, r in P.iterrows():
        d = pd.Timestamp(r['entry_date'])
        sp = [short_pct(c, d) for c in r['controls'].split(',')]
        f = borrow_fee(np.array(sp, float))
        fees.append(float(np.mean(f)) * 0.5)          # half a year of fees
        pcts.append(np.nanmean(sp) if np.isfinite(sp).any() else np.nan)
    P['borrow_cost_6m'] = fees
    P['controls_short_pct'] = pcts
    P['edge'] = P['signal_6m'] - P['control_6m']
    P['edge_after_borrow'] = P['edge'] - P['borrow_cost_6m']
    P['month'] = pd.to_datetime(P['entry_date']).dt.to_period('M').astype(str)
    P['late'] = pd.to_datetime(P['signal_date']) > TRAIN_END
    em = pd.to_datetime(P['entry_date'])
    P['tax_season'] = ((em.dt.month == 11) & (em.dt.day >= 15)) | (em.dt.month == 12)

    def ct(x, col):
        if len(x) < 10:
            return np.nan, np.nan, len(x)
        mu = x[col].mean()
        g = (x[col] - mu).groupby(x['month']).sum()
        se = np.sqrt((g ** 2).sum()) / len(x)
        return mu, mu / se, len(x)

    rows = []
    for lab, sel in (('all years', P.index == P.index), ('2006–15', ~P['late']), ('2016+', P['late'])):
        x = P[sel]
        for col, name in (('edge', 'Edge before borrow'), ('edge_after_borrow', 'Edge after estimated borrow fee')):
            mu, t, n = ct(x, col)
            rows.append({'test': 'A3 hedged', 'period': lab, 'measure': name, 'per 6-month trade': mu, 't': t, 'trades': n,
                         'avg borrow cost': x['borrow_cost_6m'].mean()})
    for lab, sel in (('Nov 15 – Dec 31 entries', P['tax_season']), ('Other entries', ~P['tax_season'])):
        x = P[sel]
        mu, t, n = ct(x, 'edge')
        rows.append({'test': 'A4 entry month', 'period': lab, 'measure': 'Edge vs look-alikes', 'per 6-month trade': mu,
                     't': t, 'trades': n, 'signal return': x['signal_6m'].mean()})
    save_table(P.drop(columns=['controls']), DATA / 'bigdrop_hedged_trades')
    return pd.DataFrame(rows)


def overlay(spy, rf):
    d = pd.read_parquet(DATA / 'matched_daily.parquet')
    sig, ctl = d['drop50liq|6m|signal'], d['drop50liq|6m|control']
    ls = (sig - ctl).reindex(spy.index)
    active = ls.notna()
    ls = ls.fillna(0.0)
    start = ls.index[active.values][0]
    rows, curves = [], {}
    for k in (0.0, 0.5, 1.0):
        p = (spy + k * (ls - active * 0.054 / 252)).loc[start:]
        name = 'SPY only' if k == 0 else f'SPY + {k:g}x hedged big-drop book'
        curves[name] = p
        for lab, sl in (('full', slice(None)), ('to 2015', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            rows.append({'portfolio': name, 'period': lab, **perf(p.loc[sl], rf)})
    return pd.DataFrame(rows), curves


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    factors = load_factors(cfg)
    spy = load_price('SPY')['adj_close'].pct_change().dropna()
    rf = factors['RF'].reindex(spy.index).ffill().fillna(0.0)
    cal = spy.index.values
    t_rows, episodes, t_curves = timing(cal, spy, rf)
    s_rows, s_curves = sleeve(spy, rf)
    h_rows = hedged(cfg)
    o_rows, o_curves = overlay(spy, rf)
    o_rows.to_csv(RESULTS / 'overlay.csv', index=False)
    s_curves.update({f'overlay|{k}': v for k, v in o_curves.items()})
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_rows.to_csv(RESULTS / 'timing.csv', index=False)
    episodes.to_csv(RESULTS / 'timing_episodes.csv', index=False)
    s_rows.to_csv(RESULTS / 'sleeve.csv', index=False)
    h_rows.to_csv(RESULTS / 'hedged_borrow.csv', index=False)
    pd.DataFrame({**t_curves, **{f'sleeve|{k}': v for k, v in s_curves.items()}}).to_parquet(DATA / 'strategy_daily.parquet')
    LOG.info('strategies: done')
    return t_rows, episodes, s_rows, h_rows


if __name__ == '__main__':
    run()
