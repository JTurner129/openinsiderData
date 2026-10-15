"""Phase 3: adjusted daily prices from yfinance, cached per ticker."""
import json
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
    df = df[['Open', 'Close', 'Adj Close', 'Volume', 'Stock Splits']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _finalize(df):
    """Add adjusted open and split-unadjusted (as-traded) close."""
    out = pd.DataFrame(index=df.index)
    ratio = (df['Adj Close'] / df['Close']).replace([np.inf, -np.inf], np.nan)
    out['adj_close'] = df['Adj Close']
    out['adj_open'] = df['Open'] * ratio
    # yfinance 'Close' is split-adjusted; undo splits that happened after each date.
    splits = df['Stock Splits'].fillna(0.0).replace(0.0, 1.0)
    later = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out['raw_close'] = df['Close'] * later
    out['volume'] = df['Volume']
    return out.astype('float64')


def download_prices(tickers, refresh_missing=False):
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    missing_path = PRICE_CACHE / '_missing.json'
    missing = json.loads(missing_path.read_text()) if missing_path.exists() else {}
    todo = []
    for t in tickers:
        if (PRICE_CACHE / f'{yahoo_symbol(t)}.parquet').exists():
            continue
        if t in missing and not refresh_missing:
            continue
        todo.append(t)
    LOG.info(f"Prices: {len(tickers) - len(todo)} tickers cached/known-missing, {len(todo)} to download")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        syms = [yahoo_symbol(t) for t in batch]
        for attempt in range(3):
            try:
                raw = yf.download(syms, start=START, auto_adjust=False, actions=True, group_by='ticker',
                                  threads=4, progress=False, timeout=30)
                break
            except Exception as e:   # yfinance raises assorted errors on throttling
                LOG.warning(f"batch {i // BATCH}: {e}; retrying")
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for t, s in zip(batch, syms):
            df = _from_download(raw, s) if raw is not None and not raw.empty else None
            if df is None or len(df) < 2:
                missing[t] = 'no data from Yahoo'
                continue
            _finalize(df).to_parquet(PRICE_CACHE / f'{s}.parquet')
        missing_path.write_text(json.dumps(missing))
        LOG.info(f"  {min(i + BATCH, len(todo))}/{len(todo)} downloaded")
        time.sleep(1.5)   # be polite to Yahoo
    return missing


def load_price(ticker):
    p = PRICE_CACHE / f'{yahoo_symbol(ticker)}.parquet'
    return pd.read_parquet(p) if p.exists() else None


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
    panel.to_parquet(DATA / 'prices.parquet', index=False)
    panel.to_csv(DATA / 'prices.csv.gz', index=False, float_format='%.4f')
    LOG.info(f"Price panel: {panel['ticker'].nunique():,} tickers, {len(panel):,} rows; "
             f"{len([t for t in tickers if t in missing]):,} of {len(tickers):,} tickers missing from Yahoo")
    return panel


if __name__ == '__main__':
    run()
