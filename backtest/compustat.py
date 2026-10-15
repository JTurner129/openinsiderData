"""Compustat Security Daily (from WRDS) as the stock price source.

Compustat keeps delisted and bankrupt companies, so it removes the
survivorship bias of Yahoo prices. Each security is written to
.cache/prices_cs/CS_<gvkey>_<iid>.parquet in the same columns as the Yahoo
cache, and signals are matched to securities by SEC CIK (comp.company.cik).

Delisting: Compustat stops at the last traded price and has no delisting
return. For securities deleted for performance reasons, load_price appends one
synthetic day with an assumed delisting return (Shumway 1997; Shumway and
Warther 1999): -30% on NYSE/AMEX, -55% elsewhere. The mode is set by
`backtest.delisting_mode` in config.yaml (or the DELIST_MODE environment
variable): 'shumway' (default), 'minus100' (stress test) or 'none'.

The WRDS files are licensed and live in a separate private repository
(`backtest.wrds_dir`); nothing derived from them is committed here except
aggregate results.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import CACHE, DATA, ROOT, get_logger, load_config

LOG = get_logger('compustat')
CS_CACHE = CACHE / 'prices_cs'
LISTED = (11, 12, 14)                      # NYSE, NYSE American, Nasdaq
NYSE_AMEX = (11, 12)
# Compustat deletion reasons treated as performance-related (no deal price at the end)
PERFORMANCE_DELIST = {'02', '03', '07', '10'}   # bankruptcy, liquidation, other, no longer files with SEC
SHUMWAY = {'nyse_amex': -0.30, 'other': -0.55}


def wrds_dir(cfg=None):
    cfg = cfg or load_config()
    d = Path(cfg['backtest'].get('wrds_dir', '../insider-wrds-data'))
    return d if d.is_absolute() else (ROOT / d).resolve()


def cs_id(gvkey, iid):
    return f'CS_{gvkey}_{iid}'


def build_cache(cfg=None, force=False):
    """Split the quarterly Compustat daily files into one Parquet file per security."""
    cfg = cfg or load_config()
    src = wrds_dir(cfg)
    marker = CS_CACHE / '_securities.parquet'
    if marker.exists() and not force:
        LOG.info(f'Compustat cache already built ({marker})')
        return pd.read_parquet(marker)
    files = sorted(src.glob('compustat_daily_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No compustat_daily_*.parquet files in {src}')
    cols = ['gvkey', 'iid', 'datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'exchg']
    num = ['prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'cshoc', 'ajexdi', 'trfd']
    # Pass 1: re-bucket every quarterly file by security, so pass 2 can hold one bucket in memory
    NB = 64
    tmp = CACHE / 'cs_buckets'
    if tmp.exists():
        for f in tmp.glob('*'):
            f.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(files):
        t = pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]).to_pandas()
        t['prccd'] = pd.to_numeric(t['prccd'], errors='coerce')
        t = t[t['prccd'] > 0]
        t['sec'] = t['gvkey'].astype(str) + '_' + t['iid'].astype(str)
        t['date'] = pd.to_datetime(t['datadate'])
        for c in num:
            t[c] = pd.to_numeric(t[c], errors='coerce').astype('float64')
        t['exchg'] = pd.to_numeric(t['exchg'], errors='coerce').astype('float32')
        t['bucket'] = pd.util.hash_array(t['sec'].to_numpy(dtype=object)) % NB
        for bkt, g in t.groupby('bucket'):
            g[['sec', 'date'] + num + ['exchg']].to_parquet(tmp / f'b{bkt:02d}_{k:03d}.parquet', index=False)
        if (k + 1) % 20 == 0:
            LOG.info(f'  bucketed {k + 1}/{len(files)} files')

    # Pass 2: one bucket at a time, one file per security
    CS_CACHE.mkdir(parents=True, exist_ok=True)
    for old in CS_CACHE.glob('CS_*.parquet'):
        old.unlink()
    meta = []
    total = 0
    for bkt in range(NB):
        parts = sorted(tmp.glob(f'b{bkt:02d}_*.parquet'))
        if not parts:
            continue
        d = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        d = d.sort_values(['sec', 'date']).drop_duplicates(['sec', 'date'])
        total += len(d)
        for sec, g in d.groupby('sec', sort=False):
            g = g.set_index('date')
            aj = g['ajexdi'].replace(0, np.nan).ffill().bfill().fillna(1.0)
            tr = g['trfd'].replace(0, np.nan).ffill().bfill().fillna(1.0)
            f = tr / aj                                     # split- and dividend-adjusted factor
            out = pd.DataFrame({
                'adj_close': g['prccd'] * f,
                'adj_open': g['prcod'] * f,
                'adj_high': g['prchd'] * f,
                'adj_low': g['prcld'] * f,
                'dollar_volume': g['prccd'] * g['cshtrd'],
                'raw_close': g['prccd'],                    # Compustat prices are as traded
                'volume': g['cshtrd'],
                'shares_out': g['cshoc'],
            }, index=g.index)
            out.index.name = 'Date'
            out.astype('float64').to_parquet(CS_CACHE / f'CS_{sec}.parquet')
            ex = g['exchg'].dropna()
            meta.append({'sec': sec, 'first_date': g.index.min(), 'last_date': g.index.max(), 'rows': len(g),
                         'last_exchg': ex.iloc[-1] if len(ex) else np.nan,
                         'main_exchg': ex.mode().iloc[0] if len(ex) else np.nan,
                         'median_dollar_volume': float((g['prccd'] * g['cshtrd']).median())})
        for x in parts:
            x.unlink()
    LOG.info(f'Compustat daily: {total:,} rows, {len(meta):,} securities')
    meta = pd.DataFrame(meta)
    meta[['gvkey', 'iid']] = meta['sec'].str.split('_', n=1, expand=True)
    _write_delistings(meta, src)
    meta.to_parquet(marker, index=False)
    LOG.info(f'Wrote {len(meta):,} security files to {CS_CACHE}')
    return meta


def _write_delistings(meta, src):
    """One row per security that ended for a performance reason, with assumed delisting returns."""
    se = pd.read_parquet(src / 'compustat_security.parquet')
    co = pd.read_parquet(src / 'compustat_company.parquet')
    se['sec'] = se['gvkey'].astype(str) + '_' + se['iid'].astype(str)
    m = meta.merge(se[['sec', 'dldtei', 'dlrsni']], on='sec', how='left')
    m = m.merge(co[['gvkey', 'dldte', 'dlrsn']], on='gvkey', how='left')
    m['del_date'] = pd.to_datetime(m['dldtei'].fillna(m['dldte']), errors='coerce')
    m['reason'] = m['dlrsni'].fillna(m['dlrsn']).astype('string')
    # Only when the price history actually runs up to the deletion (otherwise it kept trading elsewhere)
    ends_at_deletion = m['del_date'].notna() & (m['last_date'] >= m['del_date'] - pd.Timedelta(days=120))
    perf = m['reason'].isin(PERFORMANCE_DELIST) & ends_at_deletion
    m['delist_shumway'] = np.where(m['last_exchg'].isin(NYSE_AMEX), SHUMWAY['nyse_amex'], SHUMWAY['other'])
    out = m.loc[perf, ['sec', 'last_date', 'del_date', 'reason', 'last_exchg', 'delist_shumway']]
    out.to_parquet(CS_CACHE / '_delistings.parquet', index=False)
    LOG.info(f'Delistings: {len(out):,} securities ended for performance reasons '
             f'({out["reason"].value_counts().to_dict()})')


_DELIST = None


def delisting_mode(cfg=None):
    mode = os.environ.get('DELIST_MODE')
    if mode:
        return mode
    cfg = cfg or load_config()
    return cfg['backtest'].get('delisting_mode', 'shumway')


def _delistings():
    global _DELIST
    if _DELIST is None:
        p = CS_CACHE / '_delistings.parquet'
        _DELIST = pd.read_parquet(p).set_index('sec') if p.exists() else pd.DataFrame()
    return _DELIST


def load_cs(ticker):
    """Raw cached series for CS_<gvkey>_<iid>, plus the synthetic delisting day."""
    p = CS_CACHE / f'{ticker}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    sec = ticker[3:]
    dl = _delistings()
    mode = delisting_mode()
    if mode != 'none' and len(dl) and sec in dl.index and len(df):
        r = dl.loc[sec]
        ret = -1.0 if mode == 'minus100' else float(r['delist_shumway'])
        last = df.iloc[-1]
        f = max(1 + ret, 1e-4)                  # a zero price would be dropped as a bad print
        day = df.index[-1] + pd.offsets.BDay(1)
        extra = pd.DataFrame({c: [last[c] * f] for c in ('adj_close', 'raw_close')}, index=[day])
        for c in ('adj_open', 'adj_high', 'adj_low'):
            extra[c] = last['adj_close'] * f
        extra['dollar_volume'] = 0.0
        extra['volume'] = 0.0
        extra['shares_out'] = last.get('shares_out', np.nan)
        extra.index.name = df.index.name
        df = pd.concat([df, extra[df.columns]])
        df.attrs['delisting_return'] = ret
    return df


# --- Matching signals to securities --------------------------------------------------------
def _base_tic(t):
    return re.sub(r'\..*$', '', str(t)).upper() if isinstance(t, str) else None


def security_map(cfg=None):
    """CIK -> candidate securities with their price coverage."""
    src = wrds_dir(cfg)
    co = pd.read_parquet(src / 'compustat_company.parquet')
    se = pd.read_parquet(src / 'compustat_security.parquet')
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    co['cik_i'] = pd.to_numeric(co['cik'], errors='coerce')
    se = se[se['tpci'].astype(str) == '0']
    se['sec'] = se['gvkey'].astype(str) + '_' + se['iid'].astype(str)
    se['tic_base'] = se['tic'].map(_base_tic)
    m = co.dropna(subset=['cik_i'])[['cik_i', 'gvkey', 'conm']].merge(se[['gvkey', 'sec', 'tic_base']], on='gvkey')
    return m.merge(meta[['sec', 'first_date', 'last_date', 'median_dollar_volume', 'main_exchg']], on='sec')


def match_signals(signals, cfg=None):
    """price_id for each signal: same company (CIK), trading on the signal date; ticker match first, then most traded."""
    smap = security_map(cfg)
    by_cik = {int(c): g for c, g in smap.groupby('cik_i')}
    ids = []
    for _, s in signals.iterrows():
        g = by_cik.get(int(s['issuer_cik']))
        pick = None
        if g is not None:
            d = s['signal_date']
            live = g[(g['first_date'] <= d + pd.Timedelta(days=7)) & (g['last_date'] >= d)]
            if len(live):
                tk = _base_tic(s.get('ticker'))
                same = live[live['tic_base'] == tk] if tk else live.iloc[0:0]
                pick = (same if len(same) else live).sort_values('median_dollar_volume', ascending=False)['sec'].iloc[0]
        ids.append(cs_id(*pick.split('_', 1)) if pick else None)
    return pd.Series(ids, index=signals.index, name='price_id')


def fundamentals(cfg=None):
    """Quarterly book equity per gvkey with the date it became public (report date, else +95 days)."""
    src = wrds_dir(cfg)
    f = pd.read_parquet(src / 'compustat_fundq.parquet')
    f['datadate'] = pd.to_datetime(f['datadate'])
    f['rdq'] = pd.to_datetime(f['rdq'], errors='coerce')
    f['public'] = f['rdq'].fillna(f['datadate'] + pd.Timedelta(days=95))
    book = pd.to_numeric(f['ceqq'], errors='coerce').fillna(pd.to_numeric(f['seqq'], errors='coerce'))
    f['book_equity'] = book * 1e6                  # Compustat reports in millions
    return f.dropna(subset=['book_equity'])[['gvkey', 'public', 'book_equity']].sort_values(['gvkey', 'public'])
