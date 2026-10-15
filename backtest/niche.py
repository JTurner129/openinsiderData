"""Phase 14: niches with clear economic drivers where small investors have an edge. Long-only.

Rules were written down and committed before any of this was run. Costs: estimated half-spread each way
+ 0.15% FX each way. Money not in positions is in SPY. Equal weight per entry.

T1 Thrift conversions. Savings institutions (SIC 6035/6036) whose shares first trade on or after 2005-03-01
   (Compustat IPO date, else first trading day in our data). Buy 21 trading days after the first trading
   day (after the IPO pop settles), if price >= $5 and average daily dollar volume >= $50K. Hold 3 years
   (756 trading days) or until delisting (takeovers are captured at the deal price). Also a 1-year hold.
   Benchmarks: SPY, and KRE (regional bank ETF) for context.
T2 Cheap small banks with insiders buying. Banks and thrifts (SIC 6020-6029, 6035, 6036) in the monthly
   deep-value universe (price >= $1, >= $100K a day traded, market cap >= $20M) with market cap <= book
   equity and an officer/director purchase totalling >= $100K in the prior 90 days. Hold 12 months.
T3 Biotech below cash with insiders buying. SIC 2834, 2836, 8731 in the same universe with EV < 0
   (cash > market cap + debt) and an officer/director purchase >= $100K in the prior 90 days. Hold 12 months.
   Also T3b: EV < 0 without the insider condition.
Exploratory: where the insider big-drop edge is strongest, by SIC division (matched per-trade edge).
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config, save_table
from .compustat import CS_CACHE, wrds_dir

LOG = get_logger('niche')
FX = 0.0015


def sic_map(cfg):
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'sic', 'ipodate', 'conm'])
    co['sic'] = pd.to_numeric(co['sic'], errors='coerce')
    co['ipodate'] = pd.to_datetime(co['ipodate'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    return co


def thrift_entries(cfg, cal):
    from .prices import load_price
    co = sic_map(cfg)
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    th = meta.merge(co[co['sic'].isin([6035, 6036])], on='gvkey')
    th['first_date'] = pd.to_datetime(th['first_date'])
    th['start'] = th['ipodate'].where(th['ipodate'].notna() & (th['ipodate'] >= pd.Timestamp('2005-01-01')), th['first_date'])
    th = th[th['start'] >= pd.Timestamp('2005-03-01')]
    rows = []
    idx = pd.DatetimeIndex(cal)
    for _, t in th.iterrows():
        sec = 'CS_' + str(t['sec'])
        px = load_price(sec)
        if px is None or len(px) < 30:
            continue
        px = px.loc[t['start']:]
        if len(px) < 25:
            continue
        d = px.index[21]
        price = px['raw_close'].iloc[21]
        adv = px['dollar_volume'].iloc[2:22].mean() if 'dollar_volume' in px else np.nan
        if price >= 5 and adv >= 50_000:
            rows.append({'price_ticker': sec, 'gvkey': t['gvkey'], 'name': t['conm'], 'first_trade': px.index[0],
                         'entry_date': idx[min(np.searchsorted(idx.values, np.datetime64(d), 'right'), len(idx) - 1)],
                         'price': price, 'adv': adv})
    return pd.DataFrame(rows)


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market, corwin_schultz
    from .prices import download_prices, load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    cal = spy_px.index.values
    idx = pd.DatetimeIndex(cal)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    if load_price('KRE') is None:
        download_prices(['KRE'], refresh_missing=True)
    kre = load_price('KRE')['adj_close'].pct_change().reindex(idx) if load_price('KRE') is not None else None
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    market = Market(cal)
    co = sic_map(cfg)

    # T2/T3 from the deep-value monthly panel
    P = pd.read_parquet(DATA / 'deep_value_panel.parquet')
    P['month_end'] = pd.to_datetime(P['month_end'])
    P['gvkey'] = P['gvkey'].astype(str)
    P = P.merge(co[['gvkey', 'sic']], on='gvkey', how='left')
    bank = P['sic'].between(6020, 6029) | P['sic'].isin([6035, 6036])
    bio = P['sic'].isin([2834, 2836, 8731])
    mc = P['mcap'] / 1e6
    from .deep_value import entries
    sets = {}
    sets['T1 Thrift conversions'] = thrift_entries(cfg, cal)
    for name, cond in (('T2 Cheap banks + insiders buying', bank & (P['ceqq'] > 0) & (mc <= P['ceqq']) & P['insider_buy_90d']),
                       ('T3 Biotech below cash + insiders buying', bio & (P['ev'] < 0) & P['insider_buy_90d']),
                       ('T3b Biotech below cash', bio & (P['ev'] < 0))):
        E = entries(P, cond.fillna(False).values).copy()
        E['price_ticker'] = E['sec']
        E['entry_date'] = [idx[min(np.searchsorted(cal, np.datetime64(d), 'right'), len(idx) - 1)] for d in E['month_end']]
        sets[name] = E
    rows, curves, saved = [], {}, []
    for name, E in sets.items():
        E = E.reset_index(drop=True)
        if E.empty:
            continue
        costs = []
        for _, e in E.iterrows():
            px = load_price(e['price_ticker'])
            w = px.loc[:e['entry_date']].tail(21)
            sp = np.nanmedian(corwin_schultz(w['adj_high'].values, w['adj_low'].values, w['adj_close'].values)) if len(w) > 5 else np.nan
            costs.append((sp if np.isfinite(sp) else 0.01) + 2 * FX)
        holds = (('3y', 756), ('1y', 252)) if name.startswith('T1') else (('1y', 252),)
        for hn, hd in holds:
            eng = SubsetEngine(E, market, hd, np.array(costs))
            mask = np.ones(len(E))
            p = eng.returns(mask, rf.values)
            act = pd.Series(eng.active(mask), index=idx)
            p = p.reindex(idx).where(act, spy).loc[act[act].index[0]:]
            curves[f'{name}|{hn}'] = p
            for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
                s_, b_ = perf(p.loc[sl], rf), perf(spy.loc[p.index[0]:].loc[sl], rf)
                k_ = perf(kre.loc[p.index[0]:].loc[sl].dropna(), rf) if kre is not None else {}
                rows.append({'niche': name, 'hold': hn, 'period': lab, 'entries': len(E),
                             'avg open': float(np.asarray(eng.den.T @ mask).ravel()[act.values].mean()) if act.any() else 0,
                             'days invested': float(act.loc[p.index[0]:].loc[sl].mean()),
                             'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'), 'KRE CAGR': k_.get('CAGR'),
                             'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'),
                             'max DD': s_.get('max drawdown'), 'SPY max DD': b_.get('max drawdown')})
        E = E.assign(niche=name)
        saved.append(E[[c for c in ('niche', 'price_ticker', 'gvkey', 'entry_date', 'name') if c in E]])
        LOG.info(f'niche: {name}: {len(E):,} entries')
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'niche.csv', index=False)
    save_table(pd.concat(saved), DATA / 'niche_entries')
    pd.DataFrame(curves).to_parquet(DATA / 'niche_daily.parquet')

    # exploratory: big-drop matched edge by SIC division
    M = pd.read_parquet(DATA / 'matched_controls.parquet')
    M = M[(M['screen'] == 'cluster3') & (M['ret_prior_6m'] <= -0.5) & M['complete_6m']].copy()
    M['gvkey'] = M['price_ticker'].astype(str).str.split('_').str[1]
    M = M.merge(co[['gvkey', 'sic']], on='gvkey', how='left')
    div = pd.cut(M['sic'], [0, 999, 1499, 1799, 3999, 4999, 5199, 5999, 6799, 8999, 9999],
                 labels=['Agriculture', 'Mining/energy', 'Construction', 'Manufacturing', 'Transport/utilities',
                         'Wholesale', 'Retail', 'Finance', 'Services', 'Public admin'])
    M['division'] = div
    M['edge'] = M['signal_6m'] - M['control_6m']
    by = M.groupby('division', observed=True).agg(trades=('edge', 'size'), edge=('edge', 'mean'),
                                                   median_edge=('edge', 'median'), signal_ret=('signal_6m', 'mean'))
    by.to_csv(RESULTS / 'niche_edge_by_sector.csv')
    LOG.info('niche: done')
    return r, by


if __name__ == '__main__':
    run()
