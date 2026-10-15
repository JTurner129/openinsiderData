"""Phase 3: adjusted daily prices from yfinance, cached per ticker."""
import json
import logging
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .common import CACHE, DATA, get_logger, load_config

LOG = get_logger('prices')
PRICE_CACHE = CACHE / 'prices'
START = '2005-06-01'
BATCH = 40


def yahoo_symbol(ticker):
    t = re.sub(r'\.(PK|OB|OTC)$', '', ticker)    # OTC suffixes some filers add
    return t.replace('.', '-').replace('/', '-')


def _from_download(raw, sym):
    try:
        df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
    except KeyError:
        return None
    df = df.dropna(subset=['Close'])
    if df.empty:
        return None
    if 'Stock Splits' not in df:
        df['Stock Splits'] = 0.0
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Stock Splits']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _finalize(df):
    """Adjusted open/high/low/close, split-unadjusted (as-traded) close and dollar volume."""
    out = pd.DataFrame(index=df.index)
    ratio = (df['Adj Close'] / df['Close']).replace([np.inf, -np.inf], np.nan)
    out['adj_close'] = df['Adj Close']
    out['adj_open'] = df['Open'] * ratio
    out['adj_high'] = df['High'] * ratio
    out['adj_low'] = df['Low'] * ratio
    # Split-adjusted close x split-adjusted volume = dollars actually traded that day.
    out['dollar_volume'] = df['Close'] * df['Volume']
    # yfinance 'Close' is split-adjusted; undo splits that happened after each date.
    splits = df['Stock Splits'].fillna(0.0).replace(0.0, 1.0)
    later = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out['raw_close'] = df['Close'] * later
    out['volume'] = df['Volume']
    return out.astype('float64')


class _YahooErrors(logging.Handler):
    """Collects yfinance's error log lines so throttling can be told apart
    from tickers Yahoo genuinely does not have."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def throttled(self):
        return any(re.search(r'rate.?limit|too many requests', l, re.IGNORECASE) for l in self.lines)


def download_prices(tickers, refresh_missing=False, batch=BATCH, pause=1.5, upgrade=False):
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    missing_path = PRICE_CACHE / '_missing.json'
    missing = json.loads(missing_path.read_text()) if missing_path.exists() else {}
    todo = []
    for t in tickers:
        path = PRICE_CACHE / f'{yahoo_symbol(t)}.parquet'
        if path.exists():
            # upgrade=True re-downloads files cached before high/low/dollar volume were stored
            if not (upgrade and 'adj_high' not in pd.read_parquet(path).columns):
                continue
        if t in missing and not refresh_missing:
            continue
        todo.append(t)
    LOG.info(f"Prices: {len(tickers) - len(todo)} tickers cached/known-missing, {len(todo)} to download")
    errors = _YahooErrors()
    logging.getLogger('yfinance').addHandler(errors)
    i, backoff = 0, 30
    while i < len(todo):
        chunk = todo[i:i + batch]
        syms = [yahoo_symbol(t) for t in chunk]
        errors.lines.clear()
        try:
            raw = yf.download(syms, start=START, auto_adjust=False, actions=True, group_by='ticker',
                              threads=4, progress=False, timeout=30)
        except Exception as e:   # yfinance raises assorted errors on throttling
            errors.lines.append(f'rate limit? {e}')
            raw = None
        if errors.throttled():
            # Throttled: nothing in this batch can be trusted as "missing". Back off and retry it.
            LOG.warning(f"Yahoo rate limit at {i}/{len(todo)}; sleeping {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
        backoff = 30
        for t, s in zip(chunk, syms):
            df = _from_download(raw, s) if raw is not None and not raw.empty else None
            if df is None or len(df) < 2:
                missing[t] = 'no data from Yahoo'
                continue
            missing.pop(t, None)
            _finalize(df).to_parquet(PRICE_CACHE / f'{s}.parquet')
        missing_path.write_text(json.dumps(missing))
        i += batch
        LOG.info(f"  {min(i, len(todo))}/{len(todo)} downloaded")
        time.sleep(pause)   # be polite to Yahoo
    logging.getLogger('yfinance').removeHandler(errors)
    return missing


SPIKE = np.log(5.0)   # a one-day move of 5x or more, reversed by 5x or more the next day, is a bad print
LEVEL = np.log(10.0)  # a one-day rise of 10x or more that persists is an unrecorded reverse split


def clean_bad_prints(df):
    """Remove bad prints from Yahoo data.

    - Days with a zero or negative adjusted close (Yahoo rounds heavily
      adjusted penny-stock history to 0) are dropped as "no data".
    - A close that moves >= 5x up or down and >= 5x the opposite way the next
      day (a V-shaped spike, e.g. $0.70 -> $0.0001 -> $0.31) is replaced by
      the prior close.
    - Opens more than 5x away from the same day's close are replaced by it.
    - A one-day RISE of >= 10x that persists is treated as a reverse split
      Yahoo did not adjust for: earlier history is rescaled so that day's
      return is zero. Only rises are corrected, never drops, so real crashes
      always count against the strategy (a genuine 10x one-day gain would be
      lost, which errs on the conservative side).
    Smaller genuine collapses or rallies are left alone."""
    zero = int((~(df['adj_close'] > 0)).sum())
    df = df[df['adj_close'] > 0].copy()
    close = df['adj_close'].values.astype(float)
    spiked = np.zeros(len(close), dtype=bool)
    for _ in range(3):   # repeat for back-to-back bad prints
        d = np.diff(np.log(close))
        bad = np.zeros(len(close), dtype=bool)
        if len(close) > 2:
            bad[1:-1] = (np.abs(d[:-1]) >= SPIKE) & (np.abs(d[1:]) >= SPIKE) & (np.sign(d[:-1]) != np.sign(d[1:]))
        if not bad.any():
            break
        spiked |= bad
        close = np.where(bad, np.roll(close, 1), close)
    op = df['adj_open'].values.astype(float)
    d = np.diff(np.log(close))
    shifts = np.flatnonzero(d >= LEVEL) + 1
    scale = np.ones(len(close))
    for k in shifts:
        scale[:k] *= close[k] / close[k - 1]
    close = close * scale
    op = op * scale
    if spiked.any() or len(shifts):
        df['raw_close'] = df['raw_close'] * np.where(spiked, close / df['adj_close'].values / scale, 1.0)
        df['adj_close'] = close
        df['adj_open'] = op
        for c in ('adj_high', 'adj_low'):
            if c in df:
                # spiked days carry no usable range; rescaled history moves with the close
                df[c] = np.where(spiked, close, df[c].values * scale)
    with np.errstate(divide='ignore', invalid='ignore'):
        bad_open = ~(op > 0) | (np.abs(np.log(op / close)) >= SPIKE)
    df.loc[bad_open, 'adj_open'] = df.loc[bad_open, 'adj_close']
    if 'adj_high' in df:
        # keep the day's range consistent with the (possibly repaired) open and close
        df['adj_high'] = np.fmax(df['adj_high'], np.fmax(df['adj_open'], df['adj_close']))
        df['adj_low'] = np.fmin(df['adj_low'], np.fmin(df['adj_open'], df['adj_close']))
    df.attrs.update(zero_days_dropped=zero, spikes_fixed=int(spiked.sum()), opens_fixed=int(bad_open.sum()),
                    level_shifts_fixed=int(len(shifts)))
    return df


def load_price(ticker, clean=True):
    p = PRICE_CACHE / f'{yahoo_symbol(ticker)}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return clean_bad_prints(df) if clean and len(df) > 2 else df


def build_price_panel(tickers):
    frames = []
    for t in tickers:
        df = load_price(t)
        if df is not None:
            df = df.copy()
            df['ticker'] = t
            frames.append(df.rename_axis('date').reset_index())
    panel = pd.concat(frames, ignore_index=True)
    for c in ['adj_open', 'adj_close', 'raw_close', 'volume']:
        panel[c] = panel[c].astype('float32')
    return panel


def sec_current_tickers(cfg):
    """issuer CIK -> list of the company's current tickers, from the SEC.

    Used when the ticker on the filing no longer works on Yahoo (most often
    because the company changed its symbol after the filing)."""
    import requests
    path = CACHE / 'sec_company_tickers.json'
    if not path.exists():
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                         headers={'User-Agent': cfg['sec']['user_agent']}, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    out = {}
    for rec in json.loads(path.read_text()).values():
        out.setdefault(int(rec['cik_str']), []).append(rec['ticker'].upper())
    # Company info fetched in Phase 2 (submissions JSON) also lists tickers.
    info_path = CACHE / 'sec_company_info.json'
    if info_path.exists():
        for cik, rec in json.loads(info_path.read_text()).items():
            for t in rec.get('tickers') or []:
                if t.upper() not in out.setdefault(int(cik), []):
                    out[int(cik)].append(t.upper())
    return out


def alternate_tickers(signals, cfg):
    """Per issuer CIK, the SEC's current tickers that differ from the filed ones."""
    current = sec_current_tickers(cfg)
    filed = signals.groupby('issuer_cik')['ticker'].agg(lambda x: set(x.dropna()))
    alts = {}
    for cik, tickers in filed.items():
        extra = [t for t in current.get(int(cik), []) if t not in tickers]
        if extra:
            alts[int(cik)] = extra
    path = DATA / 'alternate_tickers.json'
    path.write_text(json.dumps({str(k): v for k, v in alts.items()}, indent=1))
    return alts


def run(cfg=None, refresh_missing=False):
    cfg = cfg or load_config()
    signals = pd.read_parquet(DATA / 'signals_all_variants.parquet')
    tickers = sorted(signals['ticker'].dropna().unique().tolist())
    alts = alternate_tickers(signals, cfg)
    alt_tickers = sorted({t for v in alts.values() for t in v} - set(tickers))
    LOG.info(f"{len(alts):,} companies have a different current SEC ticker; {len(alt_tickers):,} extra tickers to try")
    tickers = tickers + alt_tickers
    bench = cfg['backtest']['benchmarks']
    missing = download_prices(bench + tickers, refresh_missing)
    for b in bench:
        if b in missing:
            raise RuntimeError(f"Benchmark {b} prices unavailable")
    panel = build_price_panel(bench + tickers)
    stats = {'zero_days_dropped': 0, 'spikes_fixed': 0, 'opens_fixed': 0, 'level_shifts_fixed': 0, 'tickers_with_spikes': 0}
    for t in panel['ticker'].unique():
        a = load_price(t).attrs
        for k in ('zero_days_dropped', 'spikes_fixed', 'opens_fixed', 'level_shifts_fixed'):
            stats[k] += a.get(k, 0)
        stats['tickers_with_spikes'] += a.get('spikes_fixed', 0) > 0
    stats.update(tickers_requested=len(bench + tickers), tickers_found=int(panel['ticker'].nunique()))
    (DATA / 'price_cleaning.json').write_text(json.dumps(stats, indent=2))
    LOG.info(f"Bad-print cleaning (applied on load): {stats}")
    panel.to_parquet(DATA / 'prices.parquet', index=False)
    panel.to_csv(DATA / 'prices.csv.gz', index=False, float_format='%.4f')
    LOG.info(f"Price panel: {panel['ticker'].nunique():,} tickers, {len(panel):,} rows; "
             f"{len([t for t in tickers if t in missing]):,} of {len(tickers):,} tickers missing from Yahoo")
    return panel


if __name__ == '__main__':
    run()
