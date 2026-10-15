"""Phase 20: react fast to trends — industry momentum on 100 years of data (Moskowitz & Grinblatt 1999).

Rules were written down and committed before any data was downloaded.

Data (free): Kenneth French 49 Industry Portfolios, value-weighted, monthly (1926+) and daily (1926+);
Fama-French market and risk-free rate.
Rules: rank the 49 industries by total return over the lookback (skipping nothing), hold the top 5 in equal
weight until the next rebalance. Industries need a full lookback of data.
  Monthly rebalance, lookbacks 1, 3, 6, 12 months (MAIN: 6 months).
  Fast version: weekly rebalance (every Friday close), lookbacks 4 and 13 weeks, from daily data.
Costs: 0.15% per side on the share of the portfolio that changes (ETF spreads + FX in an ISA), i.e. 0.3% per
full round trip of the turnover.
Scored out of sample on 1990-2007 and 2008-2026 (and 1927-1989 for context): CAGR, Sharpe, max drawdown,
against the market (Mkt-RF + RF) and, from 1999, QQQ. Also the worst 3-month loss, to show crash risk.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import requests

from .common import CACHE, RESULTS, get_logger

LOG = get_logger('industry')
BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
SIDE_COST = 0.0015


def _first_block(url, daily=False):
    raw = requests.get(url, timeout=180).content
    z = zipfile.ZipFile(io.BytesIO(raw))
    lines = z.read(z.namelist()[0]).decode('latin-1').splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith(',') and len(l.split(',')) >= 4)
    cols = [c.strip() for c in lines[hdr].split(',')[1:]]
    rows = []
    n = 8 if daily else 6
    for l in lines[hdr + 1:]:
        p = [x.strip() for x in l.split(',')]
        if not p[0].isdigit() or len(p[0]) != n:
            break
        rows.append([p[0]] + [float(x) for x in p[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=['date'] + cols)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d' if daily else '%Y%m')
    if not daily:
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)
    df = df.set_index('date').replace([-99.99, -999], np.nan) / 100
    return df


def load():
    p = CACHE / 'industry49.parquet'
    if p.exists():
        return pd.read_parquet(CACHE / 'industry49.parquet'), pd.read_parquet(CACHE / 'industry49_daily.parquet'), \
            pd.read_parquet(CACHE / 'ff_monthly.parquet'), pd.read_parquet(CACHE / 'ff_daily_full.parquet')
    m = _first_block(BASE + '49_Industry_Portfolios_CSV.zip')
    d = _first_block(BASE + '49_Industry_Portfolios_daily_CSV.zip', daily=True)
    fm = _first_block(BASE + 'F-F_Research_Data_Factors_CSV.zip')
    fd = _first_block(BASE + 'F-F_Research_Data_Factors_daily_CSV.zip', daily=True)
    m.to_parquet(p)
    d.to_parquet(CACHE / 'industry49_daily.parquet')
    fm.to_parquet(CACHE / 'ff_monthly.parquet')
    fd.to_parquet(CACHE / 'ff_daily_full.parquet')
    return m, d, fm, fd


def momentum(R, lookback, top=5):
    """R: period returns (rows = periods). Rank on the prior `lookback` periods, hold the top `top` next period."""
    past = (1 + R).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1
    out, prev_w = [], None
    for t in range(lookback, len(R) - 1):
        score = past.iloc[t].dropna()
        if len(score) < top:
            out.append((R.index[t + 1], np.nan))
            continue
        pick = score.sort_values(ascending=False).index[:top]
        w = pd.Series(1.0 / top, index=pick)
        turnover = 1.0 if prev_w is None else w.subtract(prev_w, fill_value=0).abs().sum() / 2
        ret = R.iloc[t + 1][pick].mean() - turnover * 2 * SIDE_COST
        out.append((R.index[t + 1], ret))
        prev_w = w
    return pd.Series(dict(out)).dropna()


def stats(r, rf, per_year):
    r = r.dropna()
    if len(r) < per_year:
        return {}
    w = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    roll = (1 + r).rolling(max(1, per_year // 4)).apply(np.prod, raw=True) - 1
    return {'CAGR': w.iloc[-1] ** (per_year / len(r)) - 1, 'Sharpe': ex.mean() / ex.std() * np.sqrt(per_year),
            'max DD': (w / w.cummax() - 1).min(), 'worst 3 months': roll.min()}


def run():
    from .prices import load_price
    m, d, fm, fd = load()
    mkt_m = fm['Mkt-RF'] + fm['RF']
    wk = (1 + d).resample('W-FRI').prod() - 1
    wk = wk.where(d.resample('W-FRI').count() > 0)
    fdw = (1 + fd).resample('W-FRI').prod() - 1
    mkt_w = fdw['Mkt-RF'] + fdw['RF']
    q = load_price('QQQ')
    qm = q['adj_close'].resample('ME').last().pct_change() if q is not None else None
    qw = q['adj_close'].resample('W-FRI').last().pct_change() if q is not None else None
    rows, curves = [], {}
    variants = [(f'monthly, {lb}m lookback' + (' (MAIN)' if lb == 6 else ''), m, lb, 12, fm['RF'], mkt_m, qm) for lb in (1, 3, 6, 12)] + \
        [(f'weekly, {lb}w lookback', wk, lb, 52, fdw['RF'], mkt_w, qw) for lb in (4, 13)]
    for name, R, lb, py, rf, mkt, qq in variants:
        s = momentum(R, lb)
        curves[name] = s
        for lab, a, b in (('1927–1989', '1927-01-01', '1989-12-31'), ('1990–2007', '1990-01-01', '2007-12-31'),
                          ('2008–2026', '2008-01-01', None), ('1990–2026', '1990-01-01', None)):
            st = stats(s.loc[a:b], rf, py)
            mk = stats(mkt.loc[a:b], rf, py)
            qs = stats(qq.loc[a:b], rf, py) if qq is not None else {}
            rows.append({'rule': name, 'period': lab, 'CAGR': st.get('CAGR'), 'market CAGR': mk.get('CAGR'),
                         'QQQ CAGR': qs.get('CAGR'), 'Sharpe': st.get('Sharpe'), 'market Sharpe': mk.get('Sharpe'),
                         'QQQ Sharpe': qs.get('Sharpe'), 'max DD': st.get('max DD'), 'market max DD': mk.get('max DD'),
                         'worst 3 months': st.get('worst 3 months')})
        LOG.info(f'industry: {name} done')
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'industry_momentum.csv', index=False)
    pd.DataFrame({k: v for k, v in curves.items() if 'monthly' in k}).to_parquet(CACHE / 'industry_momentum_monthly.parquet')
    LOG.info('industry: done')
    return r


if __name__ == '__main__':
    run()
