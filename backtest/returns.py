"""Phase 4: per-signal returns at fixed trading-day horizons."""
import json

import numpy as np
import pandas as pd

from .common import DATA, get_logger, load_config, save_table
from .prices import load_price

LOG = get_logger('returns')
MAX_ENTRY_GAP_DAYS = 7   # stock must trade within a week of the planned entry day


def _bench(cfg):
    out = {}
    for b in cfg['backtest']['benchmarks']:
        out[b] = load_price(b)
    return out


def compute_returns(signals, cfg, entry_delay=1):
    bt = cfg['backtest']
    horizons = bt['horizons']
    cost = bt['round_trip_cost']
    tol = bt['price_mismatch_tolerance']
    bench = _bench(cfg)
    cal = bench['SPY'].index.values            # master trading calendar
    last_idx = len(cal) - 1
    b_open = {b: df['adj_open'].reindex(cal).values for b, df in bench.items()}
    b_close = {b: df['adj_close'].reindex(cal).values for b, df in bench.items()}

    rows = []
    for ticker, g in signals.groupby(signals['ticker'].fillna('__none__'), sort=False):
        px = None if ticker == '__none__' else load_price(ticker)
        if px is not None:
            px = px.dropna(subset=['adj_close'])
            sd = px.index.values
            s_open, s_close, s_raw = px['adj_open'].values, px['adj_close'].values, px['raw_close'].values
        for idx, sig in g.iterrows():
            r = {'row_id': idx, 'entry_delay': entry_delay}
            m = int(np.searchsorted(cal, np.datetime64(sig['signal_date']), 'right')) + entry_delay - 1
            if ticker == '__none__':
                r['price_status'] = 'no_ticker'
            elif px is None or px.empty:
                r['price_status'] = 'ticker_not_found'
            elif m > last_idx:
                r['price_status'] = 'not_yet_entered'
            else:
                entry_date = cal[m]
                k = int(np.searchsorted(sd, entry_date, 'left'))
                if k >= len(sd) or (sd[k] - entry_date) > np.timedelta64(MAX_ENTRY_GAP_DAYS, 'D'):
                    r['price_status'] = 'no_price_at_entry'
                else:
                    entry_px = s_open[k]
                    used_close = not (np.isfinite(entry_px) and entry_px > 0)
                    if used_close:
                        entry_px = s_close[k]
                    raw_entry = s_raw[k] * (entry_px / s_close[k]) if s_close[k] else np.nan
                    ref = sig.get('avg_buy_price')
                    ratio = raw_entry / ref if ref and np.isfinite(ref) and ref > 0 else np.nan
                    r.update({'entry_date': pd.Timestamp(sd[k]), 'entry_price_adj': entry_px,
                              'entry_price_raw': raw_entry, 'entry_used_close': used_close,
                              'market_vs_insider_price': ratio})
                    if np.isfinite(ratio) and (ratio > tol or ratio < 1 / tol):
                        r['price_status'] = 'price_mismatch'
                    else:
                        r['price_status'] = 'ok'
                        m = int(np.searchsorted(cal, sd[k], 'left'))
                        for name, h in horizons.items():
                            e = m + h
                            if e > last_idx:
                                continue          # horizon not yet elapsed
                            exit_date = cal[e]
                            j = int(np.searchsorted(sd, exit_date, 'right')) - 1
                            truncated = (exit_date - sd[j]) > np.timedelta64(MAX_ENTRY_GAP_DAYS, 'D')
                            ret = s_close[j] / entry_px - 1
                            r[f'ret_{name}'] = ret
                            r[f'truncated_{name}'] = bool(truncated)
                            for b in bench:
                                br = b_close[b][e] / b_open[b][m] - 1
                                r[f'{b.lower()}_{name}'] = br
                                r[f'xs_{b.lower()}_{name}'] = ret - br
                            net = (1 + ret) * (1 - cost) - 1
                            r[f'net_{name}'] = net
                            r[f'net_xs_spy_{name}'] = net - r[f'spy_{name}']
            rows.append(r)
    res = pd.DataFrame(rows).set_index('row_id')
    out = signals.join(res)
    out['price_found'] = out['price_status'] == 'ok'
    names = list(horizons)
    out['horizons_elapsed'] = out[[f'ret_{n}' for n in names if f'ret_{n}' in out]].notna().sum(axis=1)
    out['partial_horizons'] = out['price_found'] & (out['horizons_elapsed'] < len(names))
    for n in names:
        for c in [f'ret_{n}', f'xs_spy_{n}', f'xs_iwm_{n}', f'net_{n}', f'net_xs_spy_{n}', f'spy_{n}', f'iwm_{n}']:
            if c not in out:
                out[c] = np.nan
    return out


FALLBACK_STATUSES = {'no_ticker', 'ticker_not_found', 'no_price_at_entry', 'price_mismatch'}


def compute_with_fallback(signals, cfg, entry_delay=1):
    """Price each signal with its filed ticker; if that fails, try the
    company's current SEC tickers (ticker changes) in order. The same entry
    and price-match checks apply, so a recycled ticker is still rejected."""
    out = compute_returns(signals, cfg, entry_delay)
    out['price_ticker'] = out['ticker'].where(out['price_found'])
    out['ticker_source'] = np.where(out['price_found'], 'filing', None)
    alt_path = DATA / 'alternate_tickers.json'
    alts = {int(k): v for k, v in json.loads(alt_path.read_text()).items()} if alt_path.exists() else {}
    for k in range(max((len(v) for v in alts.values()), default=0)):
        need = out[~out['price_found'] & out['price_status'].isin(FALLBACK_STATUSES) &
                   out['issuer_cik'].map(lambda c: len(alts.get(int(c), [])) > k)]
        if need.empty:
            break
        sub = signals.loc[need.index].copy()
        sub['ticker'] = sub['issuer_cik'].map(lambda c: alts[int(c)][k])
        rr = compute_returns(sub, cfg, entry_delay)
        ok = rr.index[rr['price_found']]
        if len(ok):
            filed = out.loc[ok, 'ticker']
            for c in rr.columns:
                out.loc[ok, c] = rr.loc[ok, c]
            out.loc[ok, 'ticker'] = filed
            out.loc[ok, 'price_ticker'] = rr.loc[ok, 'ticker']
            out.loc[ok, 'ticker_source'] = 'sec_current'
        LOG.info(f"  fallback round {k + 1}: {len(need):,} unpriced signals tried, {len(ok):,} recovered via current SEC ticker")
    return out


def run(cfg=None):
    cfg = cfg or load_config()
    bt = cfg['backtest']
    signals = pd.read_parquet(DATA / 'signals_all_variants.parquet')
    compustat = bt.get('price_source', 'yahoo') == 'compustat'
    if compustat:
        # Price each signal by its Compustat security (matched by CIK), keeping the filed ticker for display
        ids = pd.read_parquet(DATA / 'signal_price_ids.parquet')
        signals = signals.merge(ids, on=['screen', 'dedupe_days', 'signal_date', 'issuer_cik'], how='left')
        signals['filed_ticker'] = signals['ticker']
        signals['ticker'] = signals['price_id']
    frames = []
    for delay in bt['entry_delay_variants']:
        sub = signals if delay == bt['entry_delay_days'] else signals[signals['dedupe_days'] == bt['dedupe_days']]
        if compustat:
            r = compute_returns(sub, cfg, delay)
            r['price_ticker'] = r['ticker'].where(r['price_found'])
            r['ticker_source'] = np.where(r['price_found'], 'compustat', None)
            r['ticker'] = r['filed_ticker']
            r['price_status'] = r['price_status'].replace({'no_ticker': 'not_in_compustat'})
        else:
            r = compute_with_fallback(sub, cfg, delay)
        frames.append(r)
        LOG.info(f"entry delay {delay}: {len(r):,} signal rows, {r['price_found'].mean():.1%} priced")
    allr = pd.concat(frames, ignore_index=True)
    save_table(allr.drop(columns=['accessions'], errors='ignore'), DATA / 'trade_returns_all_variants', gzip_csv=True)
    base = allr[(allr['dedupe_days'] == bt['dedupe_days']) & (allr['entry_delay'] == bt['entry_delay_days'])]
    save_table(base.drop(columns=['accessions'], errors='ignore'), DATA / 'trade_returns')
    return allr


if __name__ == '__main__':
    run()
