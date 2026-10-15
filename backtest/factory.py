"""Phase 17: a strategy to find the strategy. Generate ~1,800 rule-based strategies automatically, select
among them using only past data, and test the SELECTION PROCESS out of sample. Long-only.

Rules were written down and committed before this was run.

Universe and data: the monthly deep-value panel (US common stocks, price >= $1, >= $100K a day traded,
market cap >= $20M; all features point-in-time at each month-end). Next-month total return from Compustat
prices, delistings included.

Building blocks (true/false for each stock at each month-end):
  insider: officer/director purchases >= $100K in prior 90 days     activist: original 13D in prior 90 days
  surprise_pos: SUE > 0 reported in prior 90 days                     f_high: F-score >= 7     f_low: F-score <= 3
  cheap: any deep-value test (Phase 12)    netnet    pb_low: price-to-book <= 1    ev_low: 0 < EV/EBITDA <= 8
  hated: short interest >= 15% or 12-month return <= -50%             shorted: short interest >= 10%
  mom_high: 12-1 month return in the top 30% that month                mom_low: bottom 30%
  rev: prior 1-month return in the bottom 20%                          large: 500 largest by market cap
  small: market cap < $300M     net_cash: cash > debt                  alive: Phase 12 definition
  bank: SIC 6020-6029, 6035, 6036   tech: SIC 3570-3579, 3670-3679, 7370-7379
  health: SIC 2830-2836, 3840-3851, 8000-8099   new_thrift: SIC 6035/6036 listed < 3 years
Strategies: every single block, every pair and every triple (AND), equal weight, rebalanced monthly. If fewer
than 10 stocks qualify that month, that strategy holds SPY instead. Costs: turnover x (1.0% round trip for
market cap < $300M, 0.4% otherwise, + 0.3% FX).

Selection (walk-forward, decided only with data up to each December):
  MAIN: each January from 2009, rank strategies by Sharpe ratio of the prior 36 months of net monthly returns
  (strategies need >= 30 of those months with >= 10 stocks); hold the top 10 in equal weight for 12 months.
  Robustness: top 1 / 20; lookback 24 / 60 months; rank by return instead of Sharpe.
Overfitting check: Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2017) by
combinatorially symmetric cross-validation on 2009-2026 monthly returns, 16 blocks.
Reported against SPY and QQQ: CAGR, Sharpe, max drawdown, 2009-15 / 2016+, and the strategies picked.
"""
import itertools

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table
from .compustat import CS_CACHE, wrds_dir

LOG = get_logger('factory')
MIN_NAMES = 10
START = pd.Timestamp('2009-01-01')


def monthly_returns(cal):
    path = CACHE / 'cs_monthly_returns.parquet'
    if path.exists():
        return pd.read_parquet(path)
    from .prices import load_price
    idx = pd.DatetimeIndex(cal)
    ends = idx.to_series().groupby(idx.to_period('M')).max()
    rows = []
    for k, p in enumerate(sorted(CS_CACHE.glob('CS_*.parquet'))):
        px = load_price(p.stem)
        if px is None or px.empty:
            continue
        a = px['adj_close'][~px.index.duplicated()]
        e = ends[(ends >= a.index[0])]
        m = a.reindex(a.index.union(e.values)).ffill().reindex(e.values)
        last = a.index[-1]
        nxt = m.shift(-1) / m - 1
        # a stock that stops trading inside the next month: return to its last (delisting) price
        nxt = nxt.where(~((e.values < last) & (e.shift(-1).values > last)), a.iloc[-1] / m - 1)
        r1 = m / m.shift(1) - 1
        r12_1 = m.shift(1) / m.shift(12) - 1
        rows.append(pd.DataFrame({'sec': p.stem, 'month_end': e.values, 'ret_next': nxt.values,
                                  'ret_1m': r1.values, 'ret_12_1': r12_1.values}))
        if (k + 1) % 5000 == 0:
            LOG.info(f'monthly returns: {k + 1:,}')
    M = pd.concat(rows, ignore_index=True)
    M = M[M['month_end'] <= idx[-1]]
    M.to_parquet(path, index=False)
    return M


def build_blocks(cfg, cal):
    from .phase_b import fscore_table
    P = pd.read_parquet(DATA / 'deep_value_panel.parquet')
    P['month_end'] = pd.to_datetime(P['month_end']).astype('datetime64[ns]')
    P['gvkey'] = P['gvkey'].astype(str)
    M = monthly_returns(cal)
    M['month_end'] = pd.to_datetime(M['month_end']).astype('datetime64[ns]')
    P = P.merge(M, on=['sec', 'month_end'], how='left')
    f = fscore_table(cfg)
    f['gvkey'] = f['gvkey'].astype(str)
    f['public'] = f['public'].astype('datetime64[ns]')
    P = pd.merge_asof(P.sort_values('month_end'), f.sort_values('public')[['gvkey', 'public', 'fscore']],
                      left_on='month_end', right_on='public', by='gvkey', direction='backward',
                      tolerance=pd.Timedelta(days=550))
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'sic'])
    co['gvkey'] = co['gvkey'].astype(str)
    co['sic'] = pd.to_numeric(co['sic'], errors='coerce')
    P = P.merge(co, on='gvkey', how='left')
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['sec'] = 'CS_' + meta['sec'].astype(str)
    P = P.merge(meta[['sec', 'first_date']], on='sec', how='left')
    mc = P['mcap'] / 1e6
    g = P.groupby('month_end')
    mom_rank = g['ret_12_1'].rank(pct=True)
    rev_rank = g['ret_1m'].rank(pct=True)
    size_rank = g['mcap'].rank(ascending=False)
    debt = P['dlttq'].fillna(0) + P['dlcq'].fillna(0)
    B = pd.DataFrame({
        'insider': P['insider_buy_90d'].fillna(False).astype(bool),
        'activist': P['d13_90d'].fillna(False).astype(bool),
        'surprise_pos': P['sue_recent'] > 0,
        'f_high': P['fscore'] >= 7, 'f_low': P['fscore'] <= 3,
        'cheap': P['cheap'].fillna(False).astype(bool), 'netnet': P['netnet'].fillna(False).astype(bool),
        'pb_low': (P['ceqq'] > 0) & (mc <= P['ceqq']),
        'ev_low': (P['ebitda_ttm'] > 0) & (P['ev'] / P['ebitda_ttm'] <= 8),
        'hated': P['hated'].fillna(False).astype(bool), 'shorted': P['short_pct'] >= 0.10,
        'mom_high': mom_rank >= 0.7, 'mom_low': mom_rank <= 0.3, 'rev': rev_rank <= 0.2,
        'large': size_rank <= 500, 'small': mc < 300, 'net_cash': P['cheq'] > debt,
        'alive': P['alive'].fillna(False).astype(bool),
        'bank': P['sic'].between(6020, 6029) | P['sic'].isin([6035, 6036]),
        'tech': P['sic'].between(3570, 3579) | P['sic'].between(3670, 3679) | P['sic'].between(7370, 7379),
        'health': P['sic'].between(2830, 2836) | P['sic'].between(3840, 3851) | P['sic'].between(8000, 8099),
        'new_thrift': P['sic'].isin([6035, 6036]) & ((P['month_end'] - pd.to_datetime(P['first_date'])).dt.days < 3 * 365),
    }).fillna(False).astype(bool)
    keep = P[['sec', 'month_end', 'mcap', 'ret_next']].copy()
    return keep, B


def strategy_returns(keep, B):
    names = list(B.columns)
    combos = [(a,) for a in names] + list(itertools.combinations(names, 2)) + list(itertools.combinations(names, 3))
    months = np.sort(keep['month_end'].unique())
    X = B.values
    ret = keep['ret_next'].values
    small = (keep['mcap'].values < 300e6)
    cost_rt = np.where(small, 0.010, 0.004) + 0.003
    sec_codes = pd.factorize(keep['sec'])[0]
    by_month = {m: np.flatnonzero(keep['month_end'].values == m) for m in months}
    col = {n: i for i, n in enumerate(names)}
    out = np.full((len(months), len(combos)), np.nan)
    count = np.zeros((len(months), len(combos)), dtype=np.int32)
    prev = [None] * len(combos)
    for t, m in enumerate(months):
        rows = by_month[m]
        ok = np.isfinite(ret[rows])
        rows = rows[ok]
        Xm = X[rows]
        rm, cm, sm = ret[rows], cost_rt[rows], sec_codes[rows]
        for j, c in enumerate(combos):
            mask = Xm[:, col[c[0]]].copy()
            for k in c[1:]:
                mask &= Xm[:, col[k]]
            n = int(mask.sum())
            count[t, j] = n
            if n < MIN_NAMES:
                prev[j] = None
                continue
            held = sm[mask]
            if prev[j] is None:
                turnover = 1.0
            else:
                turnover = 1.0 - np.isin(held, prev[j]).mean()
            out[t, j] = rm[mask].mean() - turnover * cm[mask].mean()
            prev[j] = held
        if t % 24 == 0:
            LOG.info(f'factory: month {t + 1}/{len(months)}')
    names_c = [' & '.join(c) for c in combos]
    # returns are for the month AFTER each month-end
    nxt = pd.DatetimeIndex(months) + pd.offsets.MonthEnd(1)
    return pd.DataFrame(out, index=nxt, columns=names_c), pd.DataFrame(count, index=nxt, columns=names_c)


def select(R, N, spy_m, rf_m, lookback, top, by):
    """Each January, pick the top strategies by trailing performance; hold for 12 months."""
    filled = R.where(N >= MIN_NAMES, spy_m.reindex(R.index), axis=0)
    out, picks = pd.Series(np.nan, index=R.index), {}
    years = sorted({d.year for d in R.index if d >= START})
    for y in years:
        hist = filled.loc[:pd.Timestamp(f'{y - 1}-12-31')].iloc[-lookback:]
        good = (N.loc[hist.index] >= MIN_NAMES).sum() >= int(lookback * 30 / 36)
        h = hist.loc[:, good]
        if h.shape[1] < top or len(hist) < lookback:
            continue
        ex = h.sub(rf_m.reindex(h.index).values, axis=0)
        score = ex.mean() / ex.std() if by == 'sharpe' else (1 + h).prod()
        pick = score.sort_values(ascending=False).index[:top]
        picks[y] = list(pick)
        fut = filled.loc[pd.Timestamp(f'{y}-01-01'):pd.Timestamp(f'{y}-12-31'), pick]
        # buy-and-hold across the year (monthly drift), equal weight at January
        v = (1 + fut.fillna(0)).cumprod().mean(axis=1)
        out.loc[fut.index] = (v / v.shift(1).fillna(1.0) - 1).values
    return out.dropna(), picks


def pbo(R, N, spy_m, rf_m, blocks=16):
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation."""
    X = R.where(N >= MIN_NAMES, spy_m.reindex(R.index), axis=0).loc[START:]
    X = X.loc[:, X.notna().mean() > 0.9].fillna(0.0)
    ex = X.sub(rf_m.reindex(X.index).values, axis=0)
    parts = np.array_split(np.arange(len(ex)), blocks)
    lam = []
    for train in itertools.combinations(range(blocks), blocks // 2):
        tr = np.concatenate([parts[i] for i in train])
        te = np.concatenate([parts[i] for i in range(blocks) if i not in train])
        s_tr = ex.iloc[tr].mean() / ex.iloc[tr].std()
        s_te = ex.iloc[te].mean() / ex.iloc[te].std()
        best = s_tr.idxmax()
        rank = (s_te < s_te[best]).mean()
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lam.append(np.log(rank / (1 - rank)))
    lam = np.array(lam)
    return float((lam <= 0).mean()), len(lam)


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    cal = spy_px.index.values
    if (DATA / 'factory_strategy_returns.parquet').exists() and (DATA / 'factory_strategy_counts.parquet').exists():
        R = pd.read_parquet(DATA / 'factory_strategy_returns.parquet')
        N = pd.read_parquet(DATA / 'factory_strategy_counts.parquet')
    else:
        keep, B = build_blocks(cfg, cal)
        LOG.info(f'factory: {len(keep):,} stock-months, {B.shape[1]} building blocks; block frequencies: ' +
                 ', '.join(f'{c} {B[c].mean():.1%}' for c in B.columns))
        R, N = strategy_returns(keep, B)
        R.to_parquet(DATA / 'factory_strategy_returns.parquet')
        N.to_parquet(DATA / 'factory_strategy_counts.parquet')
    me = lambda s: (1 + s).groupby(s.index.to_period('M')).prod() - 1
    spy_m = me(spy_px['adj_close'].pct_change().dropna())
    spy_m.index = spy_m.index.to_timestamp('M')
    qqq_m = me(load_price('QQQ')['adj_close'].pct_change().dropna())
    qqq_m.index = qqq_m.index.to_timestamp('M')
    rf_d = load_factors(cfg)['RF']
    rf_m = me(rf_d)
    rf_m.index = rf_m.index.to_timestamp('M')
    R.index = R.index.to_period('M').to_timestamp('M')
    N.index = R.index

    def mperf(s, lab_sl):
        s = s.loc[lab_sl].dropna()
        if len(s) < 12:
            return {}
        w = (1 + s).cumprod()
        ex = s - rf_m.reindex(s.index).fillna(0)
        return {'CAGR': w.iloc[-1] ** (12 / len(s)) - 1, 'Sharpe': ex.mean() / ex.std() * np.sqrt(12),
                'max DD': (w / w.cummax() - 1).min()}

    rows, curves, all_picks = [], {}, {}
    variants = [('MAIN: 36m Sharpe, top 10', 36, 10, 'sharpe'), ('36m Sharpe, top 1', 36, 1, 'sharpe'),
                ('36m Sharpe, top 20', 36, 20, 'sharpe'), ('24m Sharpe, top 10', 24, 10, 'sharpe'),
                ('60m Sharpe, top 10', 60, 10, 'sharpe'), ('36m return, top 10', 36, 10, 'return')]
    for name, lb, top, by in variants:
        p, picks = select(R, N, spy_m, rf_m, lb, top, by)
        curves[name] = p
        all_picks[name] = picks
        for lab, sl in (('all', slice(None)), ('2009–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s_ = mperf(p, sl)
            b_ = mperf(spy_m.loc[p.index[0]:], sl)
            q_ = mperf(qqq_m.loc[p.index[0]:], sl)
            rows.append({'rule': name, 'period': lab, 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'),
                         'QQQ CAGR': q_.get('CAGR'), 'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'),
                         'QQQ Sharpe': q_.get('Sharpe'), 'max DD': s_.get('max DD'), 'SPY max DD': b_.get('max DD')})
        LOG.info(f'factory: {name} done')
    prob, n = pbo(R, N, spy_m, rf_m)
    r = pd.DataFrame(rows)
    r['PBO (all strategies)'] = prob
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'factory.csv', index=False)
    pd.DataFrame([{'year': y, 'picks': '; '.join(v)} for y, v in all_picks['MAIN: 36m Sharpe, top 10'].items()]).to_csv(
        RESULTS / 'factory_picks.csv', index=False)
    pd.DataFrame(curves).to_parquet(DATA / 'factory_selected_monthly.parquet')
    LOG.info(f'factory: PBO {prob:.2f} over {n} splits; done')
    return r


if __name__ == '__main__':
    run()
