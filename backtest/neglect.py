"""Phase 28: "neglect" signals — index orphans among spin-offs, and earnings surprises nobody is watching.

Rules were written down and committed before this was run.

A. Spin-off index orphans (both periods: 1994-2004 holdout events and 2005-2026 events; same entry, hold,
   costs as Phases 21/23). Group by membership on the ENTRY date of any of S&P 500 (000003), S&P MidCap 400
   (024248) or S&P SmallCap 600 (030824), from Compustat index constituents:
     orphans = in none of them (no index fund has to own it; maximum forced selling)
     members = in at least one (index funds had to buy it)
   Reported per period and group: per-trade 12-month excess over SPY (mean, median, share > 0, month-clustered
   t), the difference orphans - members, and stand-alone books (CAGR, Sharpe, max drawdown, average positions).
   Adoption: an orphans-only rule is adopted only if the orphan book beats the unfiltered book on CAGR AND
   Sharpe in BOTH periods with >= 6 positions on average in both. Otherwise diagnostic only.
B. Earnings surprise x Wikipedia attention (2016-07 to 2026, the companies mapped in Phase 10).
   Events: quarterly reports with a report date (rdq) and SUE > 0 (Phase 11 definition). Attention = the
   company's abnormal attention (Phase 10 definition) in the calendar month BEFORE the report month.
   Entry at the open two trading days after rdq; hold 63 trading days; price >= $1 and >= $100K average daily
   dollar volume at entry; cost 0.8% round trip (0.5% spread for these larger companies + 0.3% FX).
   Groups: low attention (<= 0) vs high attention (> 0). Reported: per-trade excess over SPY (mean, median,
   share > 0, clustered t), difference low - high with clustered t, and stand-alone books for 2016-2020 and 2021+.
   Adoption: low-attention events beat high-attention by >= 2 points per trade with clustered t >= 2 on the
   difference AND the low-attention book beats SPY on CAGR in both 2016-2020 and 2021+. Otherwise Wikipedia
   attention is dropped for good.
"""
import os

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('neglect')
SP1500 = ('000003', '024248', '030824')


def clustered(ex, months):
    ex = pd.Series(np.asarray(ex, float)).reset_index(drop=True)
    m = pd.Series(np.asarray(months)).reset_index(drop=True)
    ok = ex.notna()
    ex, m = ex[ok], m[ok]
    if len(ex) < 5:
        return ex.mean(), np.nan, len(ex)
    g = (ex - ex.mean()).groupby(m).sum()
    return ex.mean(), ex.mean() / (np.sqrt((g ** 2).sum()) / len(ex)), len(ex)


def _book(df, market, idx, rf, spy_r, hold, start=None, end=None):
    from .combos import SubsetEngine
    from .strategies import perf
    df = df.reset_index(drop=True)
    if len(df) < 5:
        return {}
    eng = SubsetEngine(df, market, hold, df['cost'].values)
    mask = np.ones(len(df))
    act = pd.Series(eng.active(mask), index=idx)
    p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r)
    s0 = act[act].index[0] if start is None else max(pd.Timestamp(start), act[act].index[0])
    p = p.loc[s0:end]
    cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
    s, b = perf(p, rf), perf(spy_r.loc[p.index], rf)
    return {'book CAGR': s.get('CAGR'), 'book Sharpe': s.get('Sharpe'), 'book max DD': s.get('max drawdown'),
            'avg positions': float(cnt[cnt > 0].mean()), 'SPY CAGR (same days)': b.get('CAGR'),
            'SPY Sharpe': b.get('Sharpe')}


def part_a(cfg):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    from .filters import Market
    from .phase_b import members_on
    from .prices import load_price
    from .spinoff_grid import spy_all
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy_r = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    market = Market(idx.values)
    cols = ['price_ticker', 'gvkey', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(period='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(period='2005–2026')], ignore_index=True)
    E['gvkey'] = E['gvkey'].astype(str).str.zfill(6)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    E['member'] = members_on(list(zip(E['gvkey'], E['entry_date'])), SP1500, cfg)
    ex = []
    for _, e in E.iterrows():
        a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
        s = px.loc[e['entry_date']:].iloc[:253]
        ex.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost']) - (s.iloc[-1] / s.iloc[0] - 1) if len(a) > 1 else np.nan)
    E['ex'] = ex
    E['month'] = E['entry_date'].dt.to_period('M').astype(str)
    rows = []
    for per, (a, b) in (('1994–2004', ('1994-01-01', '2004-12-31')), ('2005–2026', ('2005-01-01', None))):
        P = E[E['period'] == per]
        for grp, m in (('all', np.ones(len(P), bool)), ('orphans (no S&P 1500 index)', ~P['member'].values),
                       ('index members', P['member'].values)):
            x = P[m]
            mean, t, n = clustered(x['ex'], x['month'])
            rows.append({'part': 'A spin-offs', 'period': per, 'group': grp, 'n': n, 'mean excess': mean,
                         'median': x['ex'].median(), 'share > 0': (x['ex'] > 0).mean(), 't': t,
                         **_book(x, market, idx, rf, spy_r, 252, a, b)})
        o, mbr = P[~P['member']], P[P['member']]
        d_mean, d_t, _ = clustered(o['ex'] - mbr['ex'].mean(), o['month'])
        rows.append({'part': 'A spin-offs', 'period': per, 'group': 'difference orphans - members',
                     'mean excess': o['ex'].mean() - mbr['ex'].mean(), 't': d_t})
    R = pd.DataFrame(rows)
    ok = True
    for per in ('1994–2004', '2005–2026'):
        g = R[(R['period'] == per)].set_index('group')
        o, al = g.loc['orphans (no S&P 1500 index)'], g.loc['all']
        ok &= bool(o['book CAGR'] > al['book CAGR'] and o['book Sharpe'] > al['book Sharpe'] and o['avg positions'] >= 6)
    R['adopted (pre-written rule)'] = ok
    return R


def part_b(cfg):
    from .compustat import wrds_dir
    from .filters import Market
    from .forced_flows import next_day
    from .phase_b import quarterly_table
    from .prices import load_price
    from .analysis import load_factors
    from .wiki_attention import abnormal_attention
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close']
    spy_r = spy.pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    market = Market(idx.values)
    ab = abnormal_attention()
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    ab = ab.merge(co.dropna(), on='cik')
    q = quarterly_table(cfg).rename(columns={'public': 'rdq'})
    q['gvkey'] = q['gvkey'].astype(str)
    q['rdq'] = pd.to_datetime(q['rdq'])
    q = q[(q['sue'] > 0) & (q['rdq'] >= '2016-07-01') & q['gvkey'].isin(set(ab['gvkey']))].copy()
    q['month'] = q['rdq'].dt.to_period('M') - 1
    q = q.merge(ab[['gvkey', 'month', 'abnormal']], on=['gvkey', 'month'])
    panel = pd.read_parquet(CACHE / 'cs_monthly_panel.parquet', columns=['sec', 'gvkey', 'month_end', 'adv20'])
    panel['gvkey'] = panel['gvkey'].astype(str)
    sec = panel.sort_values('adv20').drop_duplicates('gvkey', keep='last').set_index('gvkey')['sec']
    q['price_ticker'] = q['gvkey'].map(sec)
    q = q.dropna(subset=['price_ticker'])
    q['entry_date'] = [next_day(idx, d, 2) for d in q['rdq']]
    q = q.dropna(subset=['entry_date'])
    keep, ex = [], []
    for i, e in q.iterrows():
        px = load_price(e['price_ticker'])
        if px is None:
            continue
        w = px.loc[:e['entry_date']].iloc[:-1].tail(20)
        if len(w) < 10 or w['raw_close'].iloc[-1] < 1 or w['dollar_volume'].mean() < 100_000:
            continue
        a = px['adj_close'].loc[e['entry_date']:].iloc[:64]
        s = spy.loc[e['entry_date']:].iloc[:64]
        if len(a) < 2:
            continue
        keep.append(i)
        ex.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - 0.008) - (s.iloc[-1] / s.iloc[0] - 1))
    q = q.loc[keep].copy()
    q['ex'] = ex
    q['cost'] = 0.008
    q['emonth'] = pd.to_datetime(q['entry_date']).dt.to_period('M').astype(str)
    q = q.drop_duplicates(['price_ticker', 'entry_date']).reset_index(drop=True)
    rows = []
    lo, hi = q[q['abnormal'] <= 0], q[q['abnormal'] > 0]
    for grp, x in (('positive surprise, all', q), ('positive surprise, low attention', lo), ('positive surprise, high attention', hi)):
        mean, t, n = clustered(x['ex'], x['emonth'])
        row = {'part': 'B surprise x attention', 'period': '2016–2026', 'group': grp, 'n': n, 'mean excess': mean,
               'median': x['ex'].median(), 'share > 0': (x['ex'] > 0).mean(), 't': t}
        rows.append(row)
        for per, (a, b) in (('2016–2020', ('2016-07-01', '2020-12-31')), ('2021+', ('2021-01-01', None))):
            rows.append({'part': 'B surprise x attention', 'period': per, 'group': grp + ' (book)',
                         **_book(x, market, idx, rf, spy_r, 63, a, b)})
    d_mean, d_t, _ = clustered(lo['ex'] - hi['ex'].mean(), lo['emonth'])
    rows.append({'part': 'B surprise x attention', 'period': '2016–2026', 'group': 'difference low - high',
                 'mean excess': lo['ex'].mean() - hi['ex'].mean(), 't': d_t})
    R = pd.DataFrame(rows)
    g = R.set_index(['period', 'group'])
    ok = bool(lo['ex'].mean() - hi['ex'].mean() >= 0.02 and d_t >= 2 and all(
        g.loc[(p, 'positive surprise, low attention (book)'), 'book CAGR'] >
        g.loc[(p, 'positive surprise, low attention (book)'), 'SPY CAGR (same days)'] for p in ('2016–2020', '2021+')))
    R['adopted (pre-written rule)'] = ok
    return R


def run(cfg=None):
    cfg = cfg or load_config()
    R = pd.concat([part_a(cfg), part_b(cfg)], ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'neglect.csv', index=False)
    LOG.info('neglect: done')
    return R


if __name__ == '__main__':
    run()
