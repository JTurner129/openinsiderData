"""Phase 6b: combination search across filters, with an honest out-of-sample test.

Setup (the practical one): zero fees, full estimated spread paid, buy-and-hold
weights, entry at the next open.

1. Exhaustive search: every single filter and every pair, plus triples among
   the 20 best singles, scored on 2006-2015 signals only and checked on
   2016-2026 signals only.
2. Walk-forward: each year from 2011, pick the best combination (by t-stat)
   using only earlier signals, trade it that year, and stitch the years
   together. This is what "search for the best combination" would really
   have earned.

Each signal's daily contribution to a buy-and-hold portfolio is precomputed
as a sparse row, so a portfolio for any subset is one matrix product.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import load_factors
from .common import DATA, RESULTS, get_logger, load_config, save_table
from .filters import Market, filter_definitions
from .prices import load_price

LOG = get_logger('combos')
HOLDS = {'1m': 21, '3m': 63, '6m': 126, '1y': 252}
SCREENS_SEARCHED = ['cluster3', 'exec200k']
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'IWC-RF']
TRAIN_END, TEST_START = pd.Timestamp('2015-12-31'), pd.Timestamp('2016-01-01')
MIN_SIGNALS = 100        # per evaluation period
N_TRIPLE_BASE = 20
WF_START = 2011


class SubsetEngine:
    """Sparse per-signal (value x return, value) rows for buy-and-hold portfolios."""

    def __init__(self, trades, market, hold, cost):
        cal = market.cal
        self.cal = pd.DatetimeIndex(cal)
        rows, cols, num, den = [], [], [], []
        costs = np.broadcast_to(np.asarray(cost, dtype=float), (len(trades),))
        for k, ((_, t), c) in enumerate(zip(trades.iterrows(), costs)):
            s = market.get(t['price_ticker'])
            if s is None or pd.isna(t['entry_date']):
                continue
            m = int(np.searchsorted(cal, np.datetime64(t['entry_date']), 'left'))
            end = min(m + hold, s['last'], len(cal) - 1)
            if end < m:
                continue
            r = s['ret'][m:end + 1].copy()
            r[0] = s['intraday'][m]                       # bought at the open
            r[0] = (1 + r[0]) * (1 - c / 2) - 1           # half the spread on entry
            r[-1] = (1 + r[-1]) * (1 - c / 2) - 1         # half on exit
            v = np.concatenate([[1.0], np.cumprod(1 + r)[:-1]])
            idx = np.arange(m, end + 1)
            rows.append(np.full(len(idx), k))
            cols.append(idx)
            num.append(v * r)
            den.append(v)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        shape = (len(trades), len(cal))
        self.num = sparse.csr_matrix((np.concatenate(num), (rows, cols)), shape=shape)
        self.den = sparse.csr_matrix((np.concatenate(den), (rows, cols)), shape=shape)

    def returns(self, mask, rf):
        w = np.asarray(mask, dtype=float)
        n = self.num.T @ w
        d = self.den.T @ w
        active = d > 0
        if not active.any():
            return None
        first = np.argmax(active)
        last = len(active) - 1 - np.argmax(active[::-1])
        r = np.where(active, n / np.where(active, d, 1), rf)
        return pd.Series(r[first:last + 1], index=self.cal[first:last + 1])


def hac_alpha(y, X, lags=5):
    """OLS intercept and Newey-West t-stat (fast numpy version)."""
    X = np.column_stack([np.ones(len(y)), X])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    e = y - X @ b
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        g = xe[l:].T @ xe[:-l]
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(cov[0, 0])
    return b[0] * 252, b[0] / se if se > 0 else np.nan


class Scorer:
    def __init__(self, factors):
        self.f = factors.dropna(subset=FACTORS + ['RF'])

    def score(self, port):
        if port is None:
            return np.nan, np.nan, 0
        d = self.f.reindex(port.index).dropna()
        if len(d) < 120:
            return np.nan, np.nan, len(d)
        y = port.reindex(d.index).values - d['RF'].values
        a, t = hac_alpha(y, d[FACTORS].values)
        return a, t, len(d)


def filter_masks(base):
    masks = {'No filter': np.ones(len(base), dtype=bool)}
    for _, name, fn in filter_definitions(base):
        m = fn(base).fillna(False).astype(bool).values
        if m.sum() >= 30:
            masks[name] = m
    return masks


def search(base, engine, scorer, rf, masks, train, test):
    """Score every single, every pair, and triples among the best singles."""
    names = [n for n in masks if n != 'No filter']
    results = []

    def evaluate(combo):
        m = np.logical_and.reduce([masks[c] for c in combo]) if combo else masks['No filter']
        ntr, nte = int((m & train).sum()), int((m & test).sum())
        if ntr < MIN_SIGNALS or nte < 30:
            return None
        a_tr, t_tr, _ = scorer.score(engine.returns(m & train, rf))
        a_te, t_te, _ = scorer.score(engine.returns(m & test, rf))
        return {'combo': ' + '.join(combo) if combo else 'No filter', 'n_filters': len(combo),
                'train_signals': ntr, 'test_signals': nte, 'train alpha': a_tr, 'train t': t_tr,
                'test alpha': a_te, 'test t': t_te}

    for combo in [()] + [(n,) for n in names] + list(combinations(names, 2)):
        r = evaluate(combo)
        if r:
            results.append(r)
    singles = pd.DataFrame([r for r in results if r['n_filters'] == 1])
    if len(singles):
        top = singles.nlargest(N_TRIPLE_BASE, 'train t')['combo'].tolist()
        for combo in combinations(top, 3):
            r = evaluate(combo)
            if r:
                results.append(r)
    return pd.DataFrame(results)


def walk_forward(base, engine, scorer, rf, masks, years, n_pair_base=15):
    """Each year, using only earlier signals: score every single filter, then pairs among the
    top singles; pick the best by t-stat (no filter is a candidate too) and trade it that year."""
    names = [n for n in masks if n != 'No filter']
    sig_year = base['signal_date'].dt.year.values
    oos_mask = np.zeros(len(base), dtype=bool)
    picks = []

    def past_score(combo, past, y):
        m = np.logical_and.reduce([masks[c] for c in combo]) if combo else masks['No filter']
        mp = m & past
        if mp.sum() < MIN_SIGNALS:
            return np.nan, np.nan
        port = engine.returns(mp, rf)
        if port is None:
            return np.nan, np.nan
        a, t, _ = scorer.score(port.loc[:pd.Timestamp(f'{y - 1}-12-31')])
        return a, t

    for y in years:
        past = sig_year < y
        scored = {(): past_score((), past, y)}
        singles = {(n,): past_score((n,), past, y) for n in names}
        scored.update(singles)
        ok = sorted([k for k, v in singles.items() if np.isfinite(v[1])], key=lambda k: -singles[k][1])[:n_pair_base]
        for c1, c2 in combinations([k[0] for k in ok], 2):
            scored[(c1, c2)] = past_score((c1, c2), past, y)
        valid = {k: v for k, v in scored.items() if np.isfinite(v[1])}
        if not valid:
            continue
        best = max(valid, key=lambda k: valid[k][1])
        m = np.logical_and.reduce([masks[c] for c in best]) if best else masks['No filter']
        now = m & (sig_year == y)
        oos_mask |= now
        picks.append({'year': y, 'chosen combo': ' + '.join(best) if best else 'No filter',
                      'in-sample alpha': valid[best][0], 'in-sample t': valid[best][1], 'signals traded': int(now.sum())})
    return oos_mask, pd.DataFrame(picks)


def run(cfg=None):
    cfg = cfg or load_config()
    factors = load_factors(cfg).copy()
    factors['IWC-RF'] = load_price('IWC')['adj_close'].pct_change().reindex(factors.index) - factors['RF']
    rf = None
    cal = load_price('SPY').index.values
    market = Market(cal)
    rf_series = factors['RF'].reindex(pd.DatetimeIndex(cal)).ffill().fillna(0.0).values
    scorer = Scorer(factors)
    F = pd.read_parquet(DATA / 'signal_features.parquet')
    F = F[F['price_found']].copy()
    for c in ('signal_date', 'entry_date'):
        F[c] = pd.to_datetime(F[c])

    all_search, all_picks, wf_rows, wf_daily = [], [], [], {}
    for screen in SCREENS_SEARCHED:
        base = F[F['screen'] == screen].reset_index(drop=True)
        masks = filter_masks(base)
        train = (base['signal_date'] <= TRAIN_END).values
        test = (base['signal_date'] >= TEST_START).values
        years = list(range(WF_START, int(base['signal_date'].dt.year.max()) + 1))
        for hname, hd in HOLDS.items():
            engine = SubsetEngine(base, market, hd, base['spread_est'].values)
            res = search(base, engine, scorer, rf_series, masks, train, test)
            res.insert(0, 'hold', hname)
            res.insert(0, 'screen', screen)
            all_search.append(res)
            # Walk-forward
            oos_mask, picks = walk_forward(base, engine, scorer, rf_series, masks, years)
            picks.insert(0, 'hold', hname)
            picks.insert(0, 'screen', screen)
            all_picks.append(picks)
            oos = engine.returns(oos_mask, rf_series)
            nof = engine.returns(masks['No filter'] & (base['signal_date'].dt.year >= WF_START).values, rf_series)
            row = {'screen': screen, 'hold': hname}
            for lab, p in (('walk-forward', oos), ('no filter', nof)):
                p = p.loc[f'{WF_START}-01-01':] if p is not None else None
                a, t, n = scorer.score(p)
                row[f'{lab} alpha'], row[f'{lab} t'] = a, t
                row[f'{lab} CAGR'] = (1 + p).prod() ** (252 / len(p)) - 1 if p is not None and len(p) else np.nan
                p20 = p.loc['2020-01-01':] if p is not None else None
                a20, t20, _ = scorer.score(p20)
                row[f'{lab} alpha 2020+'], row[f'{lab} t 2020+'] = a20, t20
                wf_daily[f'{screen}|{hname}|{lab}'] = p
            row['walk-forward signals'] = int(oos_mask.sum())
            wf_rows.append(row)
            LOG.info(f"combos: {screen} {hname}: {len(res):,} combinations scored; walk-forward alpha "
                     f"{row['walk-forward alpha']:.1%} (t {row['walk-forward t']:.2f}) vs no filter "
                     f"{row['no filter alpha']:.1%}")

    search_df = pd.concat(all_search, ignore_index=True)
    picks_df = pd.concat(all_picks, ignore_index=True)
    wf_df = pd.DataFrame(wf_rows)
    # How well does the 2006-2015 ranking predict 2016-2026?
    summ = []
    for (s, h), g in search_df.groupby(['screen', 'hold']):
        g = g.dropna(subset=['train t', 'test alpha'])
        base_row = g[g['combo'] == 'No filter']
        top = g.nlargest(20, 'train t')
        summ.append({'screen': s, 'hold': h, 'combinations': len(g),
                     'rank correlation (train t vs test alpha)': g['train t'].corr(g['test alpha'], method='spearman'),
                     'no-filter test alpha': base_row['test alpha'].iloc[0] if len(base_row) else np.nan,
                     'top-20 by train t: median test alpha': top['test alpha'].median(),
                     'top-20 by train t: share beating no filter in test': (top['test alpha'] > base_row['test alpha'].iloc[0]).mean() if len(base_row) else np.nan,
                     'all combos: median test alpha': g['test alpha'].median(),
                     'best train combo': top['combo'].iloc[0], 'its train alpha': top['train alpha'].iloc[0],
                     'its train t': top['train t'].iloc[0], 'its test alpha': top['test alpha'].iloc[0],
                     'its test t': top['test t'].iloc[0]})
    summ_df = pd.DataFrame(summ)

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_table(search_df, DATA / 'combo_search_all', gzip_csv=True)
    search_df.sort_values(['screen', 'hold', 'train t'], ascending=[True, True, False]) \
        .groupby(['screen', 'hold']).head(25).to_csv(RESULTS / 'combo_top_by_train.csv', index=False)
    summ_df.to_csv(RESULTS / 'combo_summary.csv', index=False)
    wf_df.to_csv(RESULTS / 'combo_walkforward.csv', index=False)
    picks_df.to_csv(RESULTS / 'combo_walkforward_picks.csv', index=False)
    pd.DataFrame({k: v for k, v in wf_daily.items() if v is not None}).to_parquet(DATA / 'combo_walkforward_daily.parquet')
    LOG.info('combos: done')
    return search_df, summ_df, wf_df, picks_df


if __name__ == '__main__':
    run()
