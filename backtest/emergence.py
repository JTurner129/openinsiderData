"""Phase 30: companies emerging from bankruptcy (post-reorganisation equity) — the spin-off logic again:
new shares land with creditors and old holders who never chose them and dump them.

Rules were written down and committed before any of this data was downloaded.

Events (2005-2026; SEC 8-K item codes exist from August 2004):
  1. Bankruptcy: the company (CIK) filed an 8-K reporting Item 1.03 (Bankruptcy or Receivership), from the
     SEC submissions data (data.sec.gov).
  2. Emergence listing: the same CIK filed a Form 8-A12B (registration of a class of securities on a national
     exchange) between 60 days and 3 years after its first Item 1.03 8-K in that episode.
  3. Price: the company's Compustat security with the most trading (US common stock), daily prices.
Entry: at the open 21 trading days after the 8-A12B filing date (after the initial dumping, as for spin-offs);
tradeable at entry: price >= $1 and >= $100K average daily dollar volume over the prior 20 days. Hold 12 months
(252 trading days). Costs: Corwin-Schultz half-spread each way + 0.15% FX each way. One event per company per
episode (the first qualifying 8-A12B). Days with nothing held earn SPY.
Reported: per-trade 12-month excess over SPY (mean, median, share > 0, month-clustered t); stand-alone book
(CAGR, Sharpe, max drawdown, average positions) for 2005-2015 and 2016+ against SPY; the same with the 5 best
trades removed; and 70% SPY + 30% emergence book.
Pre-written bar: per-trade mean excess > 0 with clustered t >= 2, AND the book beats SPY on CAGR in both
2005-2015 and 2016+, AND it still beats SPY over all years without the 5 best trades. Only then is it a
candidate to sit next to spin-offs (a second fixed-weight slice, tested separately).
"""
import time

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table

LOG = get_logger('emergence')
SUBS = CACHE / 'sec_submissions'
FX = 0.0015


def form_index(cfg, form='8-A12B', years=range(2004, 2027)):
    path = CACHE / f'sec_{form.replace("-", "").lower()}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    s = requests.Session()
    s.headers['User-Agent'] = cfg['sec']['user_agent']
    rows = []
    for y in years:
        for q in range(1, 5):
            r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if line[:17].strip() == form:
                    p = line.split()
                    rows.append((int(p[-3]), p[-2]))
            time.sleep(0.2)
    f = pd.DataFrame(rows, columns=['cik', 'filing_date'])
    f['filing_date'] = pd.to_datetime(f['filing_date'])
    f.to_parquet(path, index=False)
    return f


def bankruptcy_8k(cfg, ciks):
    """Dates of 8-Ks reporting Item 1.03 for the given CIKs (reuses the Phase 6d 8-K cache and downloader)."""
    from .filings import _client, _sub_json
    from concurrent.futures import ThreadPoolExecutor
    SUBS.mkdir(parents=True, exist_ok=True)
    session = _client(cfg)
    todo = sorted(set(int(c) for c in ciks))
    out = []
    with ThreadPoolExecutor(4) as ex:
        for i, (cik, rows) in enumerate(zip(todo, ex.map(lambda c: _sub_json(session, c), todo))):
            out += [(cik, pd.Timestamp(d)) for d, it in rows if '1.03' in (it or '')]
            if (i + 1) % 500 == 0:
                LOG.info(f'emergence: 8-K histories {i + 1}/{len(todo)}')
    return pd.DataFrame(out, columns=['cik', 'bk_date']).drop_duplicates()


def events(cfg):
    from .compustat import wrds_dir
    from .forced_flows import next_day
    from .prices import load_price
    a = form_index(cfg)
    bk = bankruptcy_8k(cfg, a['cik'].unique())
    LOG.info(f'emergence: {len(a):,} 8-A12B filings, {bk["cik"].nunique():,} of those companies filed an Item 1.03 8-K')
    m = a.merge(bk, on='cik')
    m = m[(m['filing_date'] >= m['bk_date'] + pd.Timedelta(days=60)) & (m['filing_date'] <= m['bk_date'] + pd.Timedelta(days=3 * 365))]
    m = m.sort_values('filing_date').drop_duplicates(['cik', 'bk_date']).drop_duplicates(['cik', 'filing_date'])
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    from .compustat import CS_CACHE
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    sec = meta.sort_values('median_dollar_volume', ascending=False).drop_duplicates('gvkey').set_index('gvkey')['sec']
    m = m.merge(co.dropna(), on='cik')
    m['price_ticker'] = 'CS_' + m['gvkey'].map(sec).astype(str)
    m = m[m['gvkey'].isin(sec.index)]
    spy = load_price('SPY')
    idx = pd.DatetimeIndex(spy.index.values)
    m['entry_date'] = [next_day(idx, d, 21) for d in m['filing_date']]
    return m.dropna(subset=['entry_date']).drop_duplicates(['gvkey', 'bk_date']).reset_index(drop=True)


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market, corwin_schultz
    from .forced_flows import tradeable
    from .prices import load_price
    from .spinoff_core import blend
    from .strategies import perf
    E = events(cfg)
    keep, costs = [], []
    for i, e in E.iterrows():
        ok, w = tradeable(e['price_ticker'], e['entry_date'])
        if not ok:
            continue
        sp = np.nanmedian(corwin_schultz(w['adj_high'].values, w['adj_low'].values, w['adj_close'].values))
        keep.append(i)
        costs.append((sp if np.isfinite(sp) else 0.01) + 2 * FX)
    E = E.loc[keep].reset_index(drop=True)
    E['cost'] = costs
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close']
    spy_r = spy.pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    ex = []
    for _, e in E.iterrows():
        a_ = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
        s_ = spy.loc[e['entry_date']:].iloc[:253]
        ex.append((a_.iloc[-1] / a_.iloc[0] - 1) * (1 - e['cost']) - (s_.iloc[-1] / s_.iloc[0] - 1) if len(a_) > 1 else np.nan)
    E['ex'] = ex
    E['month'] = pd.to_datetime(E['entry_date']).dt.to_period('M').astype(str)
    LOG.info(f'emergence: {len(E)} tradeable events')
    market = Market(spy_px.index.values)

    def book(ev):
        ev = ev.reset_index(drop=True)
        eng = SubsetEngine(ev, market, 252, ev['cost'].values)
        mask = np.ones(len(ev))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r).loc[act[act].index[0]:]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
        return p, cnt

    def clustered(x, months):
        x = pd.Series(np.asarray(x, float))
        g = (x - x.mean()).groupby(pd.Series(np.asarray(months))).sum()
        return x.mean() / (np.sqrt((g ** 2).sum()) / len(x))

    ok_ex = E['ex'].notna()
    t_all = clustered(E.loc[ok_ex, 'ex'], E.loc[ok_ex, 'month'])
    top5 = E.sort_values('ex', ascending=False).index[:5]
    rows = []
    curves = {}
    for name, ev in (('emergence book', E), ('emergence book without 5 best', E.drop(top5))):
        p, cnt = book(ev)
        curves[name] = p
        for lab, sl in (('all', slice(None)), ('2005–2015', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s, b = perf(p.loc[sl], rf), perf(spy_r.loc[p.index].loc[sl], rf)
            c = cnt.loc[sl]
            rows.append({'portfolio': name, 'period': lab, 'CAGR': s.get('CAGR'), 'SPY CAGR': b.get('CAGR'),
                         'Sharpe': s.get('Sharpe'), 'SPY Sharpe': b.get('Sharpe'), 'max DD': s.get('max drawdown'),
                         'avg positions': float(c[c > 0].mean()) if (c > 0).any() else 0})
    mix = blend(curves['emergence book'], spy_r.loc[curves['emergence book'].index], 0.30)
    for lab, sl in (('all', slice(None)), ('2005–2015', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
        s, b = perf(mix.loc[sl], rf), perf(spy_r.loc[mix.index].loc[sl], rf)
        rows.append({'portfolio': '70% SPY + 30% emergence', 'period': lab, 'CAGR': s.get('CAGR'), 'SPY CAGR': b.get('CAGR'),
                     'Sharpe': s.get('Sharpe'), 'SPY Sharpe': b.get('Sharpe'), 'max DD': s.get('max drawdown')})
    R = pd.DataFrame(rows)
    R['events'] = len(E)
    R['per-trade mean excess'] = E['ex'].mean()
    R['per-trade median excess'] = E['ex'].median()
    R['share beating SPY'] = (E['ex'] > 0).mean()
    R['clustered t'] = t_all
    g = lambda n, p, k: R.loc[(R['portfolio'] == n) & (R['period'] == p), k].iloc[0]
    bar = bool(E['ex'].mean() > 0 and t_all >= 2
               and all(g('emergence book', p, 'CAGR') > g('emergence book', p, 'SPY CAGR') for p in ('2005–2015', '2016+'))
               and g('emergence book without 5 best', 'all', 'CAGR') > g('emergence book without 5 best', 'all', 'SPY CAGR'))
    R['pre-written bar met'] = bar
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'emergence.csv', index=False)
    save_table(E, DATA / 'emergence_events')
    pd.DataFrame(curves).to_parquet(DATA / 'emergence_daily.parquet')
    LOG.info(f'emergence: done; bar met = {bar}')
    return R


if __name__ == '__main__':
    run()
