"""Phase 6c: stress tests for the big-drop cluster strategy, and insider track record.

Rules were fixed before looking at results.

Stress tests (clusters >= 3 insiders, stock down >= 50% over the prior 126
trading days; 6-month and 1-year holds; zero fees, full estimated spread,
buy-and-hold weights, next-open entry):
  - leave out crash episodes (2008-09, 2020, 2022) one at a time and together
  - block bootstrap of signals by month, quarter and year (1,000 draws)
  - dollar-for-dollar hedge with IWC and IWM
  - control group: other stocks in the price data that also fell >= 50% over
    the prior 126 trading days on the same entry day, with no cluster-buy
    signal within 6 months either side, bought and held the same way

Insider track record: an insider is "proven" at a signal date if they have at
least 2 earlier purchases whose 6-month outcome was already known (filed at
least 126 trading days + 1 week before), with an average 6-month return above
SPY over the same dates. Tests: cluster includes a proven insider; cluster
includes an insider with a losing record; top vs bottom third by cluster
average record; executive buys by a proven executive.
"""
import numpy as np
import pandas as pd

from .analysis import load_factors
from .combos import Scorer, SubsetEngine
from .common import DATA, RESULTS, SCREENS, get_logger, load_config, save_table
from .filters import Market
from .prices import PRICE_CACHE, load_price, yahoo_symbol

LOG = get_logger('robust')
HOLDS = {'6m': 126, '1y': 252}
DROP_LOOKBACK, DROP_LEVEL = 126, -0.50
EPISODES = {'2008–09 crisis': (2008, 2009), '2020 COVID crash': (2020, 2020), '2022 bear market': (2022, 2022)}
N_BOOT = 1000
TRACK_H = 126
TRACK_MIN = 2
TEST_START = pd.Timestamp('2016-01-01')


def _setup(cfg):
    factors = load_factors(cfg).copy()
    factors['IWC-RF'] = load_price('IWC')['adj_close'].pct_change().reindex(factors.index) - factors['RF']
    cal = load_price('SPY').index.values
    market = Market(cal)
    rf = factors['RF'].reindex(pd.DatetimeIndex(cal)).ffill().fillna(0.0).values
    F = pd.read_parquet(DATA / 'signal_features.parquet')
    F = F[F['price_found']].copy()
    for c in ('signal_date', 'entry_date'):
        F[c] = pd.to_datetime(F[c])
    return factors, cal, market, rf, Scorer(factors), F


def perf(port, rf_series):
    if port is None or len(port) < 20:
        return {}
    w = (1 + port).cumprod()
    ex = port - rf_series.reindex(port.index).ffill().fillna(0.0)
    return {'CAGR': w.iloc[-1] ** (252 / len(port)) - 1, 'volatility': port.std() * np.sqrt(252),
            'Sharpe': ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
            'max drawdown': (w / w.cummax() - 1).min()}


# --- 2. Stress tests -------------------------------------------------------------------------
def stress_tests(F, market, scorer, rf, factors):
    base = F[F['screen'] == 'cluster3'].reset_index(drop=True)
    drop = (base['ret_prior_6m'] <= DROP_LEVEL).values
    year = base['signal_date'].dt.year.values
    rf_series = factors['RF']
    rows, boot_rows, hedge_rows, curves = [], [], [], {}
    rng = np.random.default_rng(20261015)
    for h, hd in HOLDS.items():
        eng = SubsetEngine(base, market, hd, base['spread_est'].values)

        def score(mask, label, group):
            p = eng.returns(mask, rf)
            r = {'hold': h, 'group': group, 'variant': label, 'signals': int(np.sum(mask > 0))}
            for lab, sl in (('full', slice(None)), ('2016+', slice('2016-01-01', None)), ('2020+', slice('2020-01-01', None))):
                a, t, _ = scorer.score(p.loc[sl] if p is not None else None)
                r[f'{lab} alpha'], r[f'{lab} t'] = a, t
            r.update(perf(p, rf_series))
            return r, p

        r, p_drop = score(drop, 'Big-drop strategy (all years)', 'Baseline')
        rows.append(r)
        curves[f'big-drop|{h}'] = p_drop
        r, p_all = score(np.ones(len(base), bool), 'All clusters ≥3 (no drop filter)', 'Baseline')
        rows.append(r)
        curves[f'all clusters|{h}'] = p_all
        excl_all = drop.copy()
        for name, (y0, y1) in EPISODES.items():
            m = drop & ~((year >= y0) & (year <= y1))
            excl_all &= ~((year >= y0) & (year <= y1))
            rows.append(score(m, f'Without {name}', 'Leave out crashes')[0])
            rows.append(score(drop & (year >= y0) & (year <= y1), f'Only {name}', 'Leave out crashes')[0])
        rows.append(score(excl_all, 'Without all three crashes', 'Leave out crashes')[0])

        # Block bootstrap: resample calendar blocks of signals with replacement
        for block, key in (('month', base['signal_date'].dt.to_period('M')), ('quarter', base['signal_date'].dt.to_period('Q')),
                           ('year', base['signal_date'].dt.year)):
            key = key.astype(str).values
            blocks = np.unique(key[drop])
            idx_by_block = {b: np.flatnonzero(drop & (key == b)) for b in blocks}
            alphas = []
            for _ in range(N_BOOT):
                pick = rng.choice(blocks, size=len(blocks), replace=True)
                w = np.zeros(len(base))
                for b in pick:
                    w[idx_by_block[b]] += 1
                a, _, _ = scorer.score(eng.returns(w, rf))
                alphas.append(a)
            alphas = np.array(alphas)
            alphas = alphas[np.isfinite(alphas)]
            boot_rows.append({'hold': h, 'block': block, 'blocks': len(blocks), 'draws': len(alphas),
                              'median alpha': np.median(alphas), 'p5 alpha': np.percentile(alphas, 5),
                              'p2.5 alpha': np.percentile(alphas, 2.5), 'p97.5 alpha': np.percentile(alphas, 97.5),
                              'share alpha ≤ 0': (alphas <= 0).mean()})
        LOG.info(f"stress: {h} bootstrap done")

        # Hedged with IWC / IWM, 1:1 dollars, rebalanced daily
        for etf in ('IWC', 'IWM'):
            e = load_price(etf)['adj_close'].pct_change().reindex(p_drop.index).fillna(0.0)
            hedged = p_drop - e + rf_series.reindex(p_drop.index).ffill().fillna(0.0)   # short proceeds earn cash
            st = perf(hedged, rf_series)
            a, t, _ = scorer.score(hedged)
            hedge_rows.append({'hold': h, 'hedge': f'Short {etf} 1:1', **st, 'alpha': a, 't': t,
                               '2016+ CAGR': perf(hedged.loc['2016':], rf_series).get('CAGR')})
            curves[f'big-drop hedged {etf}|{h}'] = hedged
        a, t, _ = scorer.score(p_drop)
        hedge_rows.append({'hold': h, 'hedge': 'Unhedged', **perf(p_drop, rf_series), 'alpha': a, 't': t,
                           '2016+ CAGR': perf(p_drop.loc['2016':], rf_series).get('CAGR')})
    return pd.DataFrame(rows), pd.DataFrame(boot_rows), pd.DataFrame(hedge_rows), curves


def control_group(F, market, scorer, rf, factors):
    """Stocks that crashed just as much on the same day but had no insider cluster."""
    cal = market.cal
    base = F[F['screen'] == 'cluster3'].reset_index(drop=True)
    sig = base[base['ret_prior_6m'] <= DROP_LEVEL]
    cfg = load_config()
    if cfg['backtest'].get('price_source', 'yahoo') == 'compustat':
        from .compustat import CS_CACHE
        # Every Compustat common stock. The extract's exchange code is the CURRENT one, so no
        # exchange filter here (it would drop companies that later fell to OTC, mostly failures).
        tickers = sorted(p.stem for p in CS_CACHE.glob('CS_*.parquet'))
    else:
        tickers = sorted(p.stem for p in PRICE_CACHE.glob('*.parquet'))
        tickers = [t for t in tickers if t not in ('SPY', 'IWM', 'IWC')]
    closes = {}
    for t in tickers:
        s = market.get(t)
        if s is not None:
            closes[t] = s
    LOG.info(f"control: {len(closes):,} tickers in the price universe")
    # Tickers with any cluster-buy signal: excluded within +/- 6 months of it
    cl = F[F['screen'] == 'cluster3'][['price_ticker', 'signal_date']].dropna()
    cl['price_ticker'] = cl['price_ticker'].map(lambda t: t if str(t).startswith('CS_') else yahoo_symbol(t))
    near = {t: g['signal_date'].values for t, g in cl.groupby('price_ticker')}
    rows = []
    for _, t in sig.iterrows():
        m = int(np.searchsorted(cal, np.datetime64(t['entry_date']), 'left'))
        if m - 1 - DROP_LOOKBACK < 0:
            continue
        for tk, s in closes.items():
            c0, c1 = s['close'][m - 1 - DROP_LOOKBACK], s['close'][m - 1]
            if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0) or c1 / c0 - 1 > DROP_LEVEL:
                continue
            if not s['has'][m]:
                continue
            d = near.get(tk)
            if d is not None and (np.abs((d - np.datetime64(t['entry_date'])).astype('timedelta64[D]').astype(int)) <= 182).any():
                continue
            sp = np.nanmedian(s['spread'][max(0, m - 20):m])
            rows.append({'price_ticker': tk, 'entry_date': t['entry_date'], 'signal_date': t['signal_date'],
                         'spread_est': max(sp if np.isfinite(sp) else 0.01, 0.0001), 'matched_to': t['price_ticker']})
    ctrl = pd.DataFrame(rows).drop_duplicates(['price_ticker', 'entry_date'])
    # Weight so each signal date's control group counts as much as one strategy position
    ctrl['w'] = 1.0 / ctrl.groupby('entry_date')['price_ticker'].transform('count')
    LOG.info(f"control: {len(ctrl):,} control positions for {sig['entry_date'].nunique():,} entry dates")
    out, curves = [], {}
    for h, hd in HOLDS.items():
        eng_c = SubsetEngine(ctrl, market, hd, ctrl['spread_est'].values)
        eng_s = SubsetEngine(sig.reset_index(drop=True), market, hd, sig['spread_est'].values)
        for label, eng, w in (('Big-drop clusters (insiders buying)', eng_s, np.ones(len(sig))),
                              ('Control: same-day ≥50% drops, no insider cluster (equal weight per date)', eng_c, ctrl['w'].values),
                              ('Control: same-day ≥50% drops, no insider cluster (every stock equal)', eng_c, np.ones(len(ctrl)))):
            p = eng.returns(w, rf)
            r = {'hold': h, 'portfolio': label, 'positions': int(len(w))}
            for lab, sl in (('full', slice(None)), ('2016+', slice('2016-01-01', None))):
                a, t, _ = scorer.score(p.loc[sl])
                r[f'{lab} alpha'], r[f'{lab} t'] = a, t
            r.update(perf(p, factors['RF']))
            out.append(r)
            curves[f'{label}|{h}'] = p
    return pd.DataFrame(out), ctrl, curves


# --- 3. Insider track record ------------------------------------------------------------------
def purchase_outcomes(market):
    """6-month return vs SPY for every purchase filing we can price (entry next open after filing)."""
    cal = market.cal
    p = pd.read_parquet(DATA / 'insider_trades.parquet')
    p = p[(p['transaction_type'] == 'P') & p['ticker'].notna()].copy()
    p['insider_id'] = p['owner_cik'].astype('string').fillna('name:' + p['owner_name'].astype(str))
    spy = market.get('SPY')
    out = np.full(len(p), np.nan)
    known = np.full(len(p), np.datetime64('NaT'), dtype='datetime64[ns]')
    fd = p['filing_date'].values
    for tk, idx in p.groupby('ticker').indices.items():
        s = market.get(tk)
        if s is None:
            continue
        ms = np.searchsorted(cal, fd[idx], 'right')       # first trading day after filing
        for k, m in zip(idx, ms):
            e = m + TRACK_H
            if e >= len(cal) or not s['has'][m]:
                continue
            r = (1 + s['intraday'][m]) * np.prod(1 + s['ret'][m + 1:e + 1]) - 1
            b = (1 + spy['intraday'][m]) * np.prod(1 + spy['ret'][m + 1:e + 1]) - 1
            out[k] = r - b
            known[k] = cal[e]
    p['excess_6m'] = out
    p['outcome_known'] = known
    LOG.info(f"track record: {np.isfinite(out).sum():,} of {len(p):,} purchases have a 6-month outcome")
    return p[['accession_number', 'insider_id', 'filing_date', 'excess_6m', 'outcome_known']]


def track_features(F, outcomes):
    """Per signal: each participating insider's record from purchases whose outcome was known before the signal."""
    sig = pd.read_parquet(DATA / 'signals_all_variants.parquet')
    key = ['screen', 'dedupe_days', 'signal_date', 'issuer_cik']
    sig['signal_date'] = pd.to_datetime(sig['signal_date'])
    F = F.merge(sig[key + ['accessions']], on=key, how='left', suffixes=('', '_s'))
    acc_ins = outcomes.set_index('accession_number')['insider_id']
    by_ins = {i: g.sort_values('outcome_known') for i, g in outcomes.dropna(subset=['excess_6m']).groupby('insider_id')}
    proven, loser, mean_rec, n_with = [], [], [], []
    for _, t in F.iterrows():
        insiders = {acc_ins.get(a) for a in str(t['accessions'] or '').split(',')} - {None}
        recs = []
        for i in insiders:
            g = by_ins.get(i)
            if g is None:
                continue
            past = g[g['outcome_known'] < t['signal_date']]
            if len(past) >= TRACK_MIN:
                recs.append(past['excess_6m'].mean())
        proven.append(any(r > 0 for r in recs))
        loser.append(any(r <= 0 for r in recs))
        mean_rec.append(np.mean(recs) if recs else np.nan)
        n_with.append(len(recs))
    F['has_proven_insider'] = proven
    F['has_losing_insider'] = loser
    F['mean_track_record'] = mean_rec
    F['insiders_with_record'] = n_with
    return F.drop(columns=['accessions'])


def track_tests(F, market, scorer, rf):
    rows = []
    for screen in ('cluster3', 'exec200k'):
        base = F[F['screen'] == screen].reset_index(drop=True)
        rec = base['mean_track_record']
        terc = rec.quantile([1 / 3, 2 / 3]).values if rec.notna().sum() > 30 else [np.nan, np.nan]
        tests = {
            'All signals': np.ones(len(base), bool),
            'Has ≥1 insider with a record': (base['insiders_with_record'] > 0).values,
            'Includes a proven insider': base['has_proven_insider'].values,
            'No proven insider': ~base['has_proven_insider'].values,
            'Includes an insider with a losing record': base['has_losing_insider'].values,
            'Proven insider and no loser': (base['has_proven_insider'] & ~base['has_losing_insider']).values,
            'Top third by average record': (rec >= terc[1]).fillna(False).values,
            'Bottom third by average record': (rec <= terc[0]).fillna(False).values,
            'Proven insider + stock down ≥50% in 6m': (base['has_proven_insider'] & (base['ret_prior_6m'] <= DROP_LEVEL)).values,
        }
        test_period = (base['signal_date'] >= TEST_START).values
        for h, hd in (('3m', 63), ('6m', 126), ('1y', 252)):
            eng = SubsetEngine(base, market, hd, base['spread_est'].values)
            for name, m in tests.items():
                if m.sum() < 30:
                    continue
                p = eng.returns(m, rf)
                r = {'screen': screen, 'hold': h, 'test': name, 'signals': int(m.sum()),
                     'signals 2016+': int((m & test_period).sum())}
                for lab, mm in (('full', m), ('pre-2016 signals', m & ~test_period), ('2016+ signals', m & test_period)):
                    a, t, _ = scorer.score(eng.returns(mm, rf)) if mm.sum() >= 20 else (np.nan, np.nan, 0)
                    r[f'{lab} alpha'], r[f'{lab} t'] = a, t
                rows.append(r)
        LOG.info(f"track record: {screen} done")
    return pd.DataFrame(rows)


def run(cfg=None):
    cfg = cfg or load_config()
    factors, cal, market, rf, scorer, F = _setup(cfg)
    stress, boot, hedge, curves = stress_tests(F, market, scorer, rf, factors)
    ctrl_tbl, ctrl, ctrl_curves = control_group(F, market, scorer, rf, factors)
    outcomes = purchase_outcomes(market)
    T = track_features(F[F['screen'].isin(['cluster3', 'exec200k'])].copy(), outcomes)
    track = track_tests(T, market, scorer, rf)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stress.to_csv(RESULTS / 'bigdrop_stress.csv', index=False)
    boot.to_csv(RESULTS / 'bigdrop_bootstrap.csv', index=False)
    hedge.to_csv(RESULTS / 'bigdrop_hedged.csv', index=False)
    ctrl_tbl.to_csv(RESULTS / 'bigdrop_control.csv', index=False)
    track.to_csv(RESULTS / 'track_record.csv', index=False)
    save_table(T[['screen', 'signal_date', 'issuer_cik', 'ticker', 'has_proven_insider', 'has_losing_insider',
                  'mean_track_record', 'insiders_with_record']], DATA / 'signal_track_record')
    save_table(ctrl, DATA / 'bigdrop_control_positions')
    pd.DataFrame({k: v for k, v in {**curves, **ctrl_curves}.items() if v is not None}).to_parquet(DATA / 'bigdrop_daily.parquet')
    LOG.info('robust: done')
    return stress, boot, hedge, ctrl_tbl, track


if __name__ == '__main__':
    run()
