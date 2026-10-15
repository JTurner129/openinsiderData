"""Phase 6e: matched comparison group. Does insider buying add anything beyond what the stock looks like?

Rules fixed before running:
  For every cluster (>= 3 insiders) and top-executive (>= $200K) signal, on its
  entry day pick the 5 stocks in the full Compustat universe that look most
  alike the day before entry:
    - log market cap
    - return over the prior 126 trading days
    - log price-to-book (only when the signal stock has positive book equity;
      candidates then need positive book equity too)
  Distance is the sum of absolute differences, each scaled by that day's
  cross-sectional standard deviation. Candidates must trade on the entry day
  and have no insider purchase filing within 182 days either side of the entry.

  Signal and controls are treated identically: entry at the next open, exit
  after 126 or 252 trading days (or at delisting, with the assumed delisting
  return), half the estimated spread paid on entry and on exit.

  Reported: mean per-trade return of signal minus its control basket, with
  t-stats clustered by entry month, for 2006-2015, 2016+ and all years; and a
  calendar-time long-signal / short-controls portfolio with FF5+UMD+IWC alpha.
  Same comparisons for two subsets fixed in advance: stock down >= 50% over
  the prior 6 months, and price-to-book <= 1.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config, save_table
from .prices import load_price

LOG = get_logger('matched')
HOLDS = {'6m': 126, '1y': 252}
K = 5
LOOK = 126
EXCLUDE_DAYS = 182
SCREENS = ['cluster3', 'exec200k']
TRAIN_END = pd.Timestamp('2015-12-31')
LIQ_ADV = 100_000      # the $100K/day floor from the realistic-trading tests


def universe(cal):
    from .compustat import CS_CACHE, fundamentals
    from .filters import corwin_schultz
    tickers = sorted(p.stem for p in CS_CACHE.glob('CS_*.parquet'))
    T, N = len(cal), len(tickers)
    idx = pd.DatetimeIndex(cal)
    C = np.full((T, N), np.nan, np.float32)
    R = np.zeros((T, N), np.float32)
    I = np.zeros((T, N), np.float32)
    S = np.full((T, N), np.nan, np.float32)
    M = np.full((T, N), np.nan, np.float32)
    B = np.full((T, N), np.nan, np.float32)
    last = np.full(N, -1, np.int32)
    book = {g: x for g, x in fundamentals().groupby('gvkey')}
    for j, t in enumerate(tickers):
        px = load_price(t)
        if px is None or px.empty:
            continue
        px = px.dropna(subset=['adj_close'])
        close = px['adj_close'].reindex(idx)
        has = close.notna().values
        if not has.any():
            continue
        C[:, j] = close.values
        R[:, j] = (close / px['adj_close'].shift(1).reindex(idx) - 1).fillna(0.0).values
        opn = px['adj_open'].where(px['adj_open'] > 0, px['adj_close'])
        I[:, j] = (px['adj_close'] / opn - 1).reindex(idx).fillna(0.0).values
        S[:, j] = pd.Series(corwin_schultz(px['adj_high'].values, px['adj_low'].values, px['adj_close'].values),
                            index=px.index).reindex(idx).values
        if 'shares_out' in px and 'raw_close' in px:
            M[:, j] = (px['raw_close'] * px['shares_out']).reindex(idx).values
        g = t.split('_')[1]
        if g in book:
            b = book[g].drop_duplicates('public', keep='last').set_index('public')['book_equity']
            B[:, j] = b.reindex(idx.union(b.index)).ffill(limit=350).reindex(idx).values   # stale after ~500 days
        last[j] = int(np.flatnonzero(has)[-1])
        if (j + 1) % 5000 == 0:
            LOG.info(f'  universe: loaded {j + 1:,}/{N:,}')
    return tickers, C, R, I, S, M, B, last


def path(R, I, S, last, m, js, hd):
    """Daily returns (rows) for positions entered at the open of day m, costs included."""
    T = R.shape[0]
    end = min(m + hd, T - 1)
    L = end - m + 1
    r = R[m:end + 1, js].astype(np.float64).copy()
    r[0] = I[m, js]
    r[np.arange(m, end + 1)[:, None] > last[js][None, :]] = 0.0
    sp = np.nanmedian(S[max(0, m - 20):m, js], axis=0) if m > 0 else np.full(len(js), np.nan)
    cost = np.maximum(np.where(np.isfinite(sp), sp, 0.01), 0.0001)
    r[0] = (1 + r[0]) * (1 - cost / 2) - 1
    ex = np.clip(np.minimum(last[js], end) - m, 0, L - 1)
    r[ex, np.arange(len(js))] = (1 + r[ex, np.arange(len(js))]) * (1 - cost / 2) - 1
    return r


def clustered_t(diff, month):
    d = pd.DataFrame({'d': diff, 'm': month}).dropna()
    if len(d) < 10:
        return np.nan, np.nan, len(d)
    mu = d['d'].mean()
    g = (d['d'] - mu).groupby(d['m']).sum()
    se = np.sqrt((g ** 2).sum()) / len(d)
    return mu, mu / se if se > 0 else np.nan, len(d)


def run(cfg=None):
    cfg = cfg or load_config()
    from .robust import _setup
    from .compustat import security_map
    factors, cal, market, rf, scorer, F = _setup(cfg)
    F = F[F['screen'].isin(SCREENS) & F['price_ticker'].astype(str).str.startswith('CS_')].reset_index(drop=True)
    tickers, C, R, I, S, M, B, last = universe(cal)
    col = {t: j for j, t in enumerate(tickers)}
    T = len(cal)
    LOG.info(f'matched: universe {len(tickers):,} securities, {len(F):,} signals')

    # Any insider purchase filing marks a security as "insider-touched" for +/- 182 days
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['issuer_cik', 'filing_date'])
    sm = security_map(cfg)[['cik_i', 'sec']].dropna()
    sm['sec'] = 'CS_' + sm['sec']
    pc = pc.merge(sm, left_on='issuer_cik', right_on='cik_i')
    pc = pc[pc['sec'].isin(col)]
    p_col = pc['sec'].map(col).values
    p_day = np.searchsorted(cal, pc['filing_date'].values.astype('datetime64[ns]'))

    order = np.argsort(p_day)
    p_day, p_col = p_day[order], p_col[order]
    with np.errstate(all='ignore'):
        logM = np.log(np.where(M > 0, M, np.nan))

    rows = []
    BOOKS = ('all', 'drop50', 'drop50liq', 'pb1')
    acc = {(b, h, side): (np.zeros(T), np.zeros(T)) for b in BOOKS for h in HOLDS for side in ('signal', 'control')}
    by_day = F.groupby(F['entry_date'].map(lambda d: int(np.searchsorted(cal, np.datetime64(d), 'left'))))
    for n, (m, g) in enumerate(by_day):
        if m - 1 - LOOK < 0 or m >= T:
            continue
        c0, c1 = C[m - 1 - LOOK], C[m - 1]
        with np.errstate(all='ignore'):
            ret6 = c1 / c0 - 1
            lm = logM[m - 1]
            pb = np.where(B[m - 1] > 0, np.log(np.exp(lm) / (B[m - 1])), np.nan)
        lo, hi = np.searchsorted(p_day, m - EXCLUDE_DAYS * 252 // 365), np.searchsorted(p_day, m + EXCLUDE_DAYS * 252 // 365, 'right')
        touched = np.zeros(len(tickers), bool)
        touched[p_col[lo:hi]] = True
        base_ok = np.isfinite(ret6) & np.isfinite(lm) & np.isfinite(C[m]) & ~touched
        sd_r, sd_m = np.nanstd(ret6[base_ok]), np.nanstd(lm[base_ok])
        sd_p = np.nanstd(pb[base_ok & np.isfinite(pb)])
        for _, s in g.iterrows():
            j0 = col.get(s['price_ticker'])
            if j0 is None or not (np.isfinite(ret6[j0]) and np.isfinite(lm[j0]) and np.isfinite(C[m, j0])):
                continue
            ok = base_ok.copy()
            ok[j0] = False
            dist = np.abs(ret6 - ret6[j0]) / sd_r + np.abs(lm - lm[j0]) / sd_m
            if np.isfinite(pb[j0]):
                ok &= np.isfinite(pb)
                dist = dist + np.abs(pb - pb[j0]) / sd_p
            cand = np.flatnonzero(ok)
            if len(cand) < K:
                continue
            js = cand[np.argpartition(dist[cand], K)[:K]]
            rec = {'screen': s['screen'], 'signal_date': s['signal_date'], 'entry_date': cal[m],
                   'issuer_cik': s['issuer_cik'], 'price_ticker': s['price_ticker'],
                   'ret_prior_6m': float(ret6[j0]), 'log_mcap': float(lm[j0]),
                   'price_to_book': float(np.exp(pb[j0])) if np.isfinite(pb[j0]) else np.nan,
                   'match_distance': float(np.mean(dist[js])), 'controls': ','.join(tickers[k] for k in js)}
            for h, hd in HOLDS.items():
                rs = path(R, I, S, last, m, np.array([j0]), hd)[:, 0]
                rc = path(R, I, S, last, m, js, hd)
                rec[f'signal_{h}'] = np.prod(1 + rs) - 1
                rec[f'control_{h}'] = np.mean(np.prod(1 + rc, axis=0) - 1)
                rec[f'complete_{h}'] = m + hd < T
                # Calendar-time books (cluster screen only): each position's own value path
                if s['screen'] != 'cluster3':
                    continue
                end = m + len(rs)
                books = ['all'] + (['drop50'] if ret6[j0] <= -0.5 else []) + \
                    (['drop50liq'] if ret6[j0] <= -0.5 and s.get('adv20', 0) >= LIQ_ADV else []) + \
                    (['pb1'] if np.isfinite(pb[j0]) and np.exp(pb[j0]) <= 1 else [])
                for side, r in (('signal', rs[:, None]), ('control', rc)):
                    v = np.vstack([np.ones((1, r.shape[1])), np.cumprod(1 + r, axis=0)[:-1]])
                    for bk in books:
                        num, den = acc[(bk, h, side)]
                        num[m:end] += (v * r).sum(axis=1) / r.shape[1]
                        den[m:end] += v.sum(axis=1) / r.shape[1]
            rows.append(rec)
        if n % 500 == 0:
            LOG.info(f'matched: {n:,}/{len(by_day):,} entry days, {len(rows):,} signals matched')
    P = pd.DataFrame(rows)
    save_table(P, DATA / 'matched_controls', gzip_csv=True)

    idx = pd.DatetimeIndex(cal)

    def series(num, den):
        act = den > 0
        return pd.Series(np.where(act, num / np.where(act, den, 1), rf), index=idx)

    out = []
    P['month'] = pd.to_datetime(P['entry_date']).dt.to_period('M').astype(str)
    subsets = {'All signals': lambda x: np.ones(len(x), bool),
               'Stock down ≥ 50% in prior 6 months': lambda x: x['ret_prior_6m'] <= -0.5,
               'Price-to-book ≤ 1': lambda x: x['price_to_book'] <= 1}
    for screen in SCREENS:
        for sub, fn in subsets.items():
            for h in HOLDS:
                x = P[(P['screen'] == screen) & P[f'complete_{h}']]
                x = x[np.asarray(fn(x), bool)]
                r = {'screen': screen, 'subset': sub, 'hold': h, 'signals': len(x),
                     'signal mean': x[f'signal_{h}'].mean(), 'control mean': x[f'control_{h}'].mean(),
                     'signal median': x[f'signal_{h}'].median(), 'control median': x[f'control_{h}'].median()}
                for lab, sel in (('full', np.ones(len(x), bool)),
                                 ('2006–15', pd.to_datetime(x['signal_date']) <= TRAIN_END),
                                 ('2016+', pd.to_datetime(x['signal_date']) > TRAIN_END)):
                    y = x[np.asarray(sel, bool)]
                    mu, t, nn = clustered_t((y[f'signal_{h}'] - y[f'control_{h}']).values, y['month'].values)
                    r[f'{lab} edge'], r[f'{lab} t'], r[f'{lab} n'] = mu, t, nn
                out.append(r)
    # Calendar-time long cluster signals / short their matched controls
    ct, daily = [], {}
    names = {'all': 'All cluster signals', 'drop50': 'Clusters after a ≥50% drop',
             'drop50liq': 'Clusters after a ≥50% drop, ≥$100K/day traded', 'pb1': 'Clusters at price-to-book ≤ 1'}
    from .robust import perf
    for bk in BOOKS:
        for h in HOLDS:
            ps, pcn = series(*acc[(bk, h, 'signal')]), series(*acc[(bk, h, 'control')])
            act = acc[(bk, h, 'signal')][1] > 0
            rfs = pd.Series(rf, index=idx)
            ls = (ps - pcn + rfs)[act]
            daily[f'{bk}|{h}|signal'], daily[f'{bk}|{h}|control'] = ps[act], pcn[act]
            for lab, sl in (('full', slice(None)), ('2016+', slice('2016-01-01', None))):
                r = {'book': names[bk], 'hold': h, 'period': lab}
                for side, ser in (('long signals', ps[act]), ('long matched', pcn[act]), ('long-short', ls)):
                    a, t, _ = scorer.score(ser.loc[sl])
                    r[f'{side} alpha'], r[f'{side} t'] = a, t
                r['long signals CAGR'] = perf(ps[act].loc[sl], factors['RF']).get('CAGR')
                r['long matched CAGR'] = perf(pcn[act].loc[sl], factors['RF']).get('CAGR')
                ct.append(r)
    pd.DataFrame(daily).to_parquet(DATA / 'matched_daily.parquet')
    res = pd.DataFrame(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / 'matched_controls.csv', index=False)
    pd.DataFrame(ct).to_csv(RESULTS / 'matched_controls_calendar.csv', index=False)
    LOG.info('matched: done')
    return res


if __name__ == '__main__':
    run()
