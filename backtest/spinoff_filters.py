"""Phase 25: can a simple filter or a second ingredient improve the spin-off slice?

Rules were written down and committed before this was run.

Events, costs, prices, entry (21 trading days after first trade) and hold (12 months) exactly as Phases 23-24.
Filters, each known at the entry date:
  A "dumped": the stock's return from its first close to the close before entry is below 0 (index-fund and
    parent-shareholder selling pushed it down) vs >= 0. Story: forced selling creates the bargain.
  B "cheap": book-to-market >= 0.5, using the latest quarterly report with a report date before entry
    (book equity ceqq / (price x shares)). Spin-offs without a report before entry are excluded from both
    sides of B. Coverage is reported; the holdout has reports for only about half the events.
  C "small": market cap at entry below the median of the PRIOR spin-offs (expanding, point in time; the first
    10 events use $1B).
Each filter's two halves are reported as stand-alone books (CAGR, Sharpe, max drawdown, average positions)
for 1994-2004 and 2005-2026, against the unfiltered book.
Adoption rule (pre-written): a filter is adopted only if its kept half beats the unfiltered book on BOTH CAGR
and Sharpe in BOTH periods, and still holds >= 6 positions on average in both periods.
Portfolio mixes (2005-2026 only, the period where the other ingredients exist; monthly rebalance, 0.15% per
side on the amount moved):
  M0 70% QQQ + 30% spin-offs (reference)
  M1 70% QQQ + 15% spin-offs + 15% thrift conversions (Phase 14, 1-year hold)
  M2 60% QQQ + 30% spin-offs + 10% thrift conversions
  M3 60% QQQ + 25% spin-offs + 15% S&P 500 insider buy + positive surprise (Phase 11 L2)
A mix is called better only if it beats M0 on Sharpe AND on CAGR in both 2006-15 and 2016+.
"""
import os

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('spinoff_filters')
PERIODS = {'1994–2004': ('1994-01-01', '2004-12-31'), '2005–2026': ('2005-01-01', None)}


def run(cfg=None):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    from .combined import mix
    from .combos import SubsetEngine
    from .compustat import wrds_dir
    from .filters import Market
    from .prices import load_price
    from .spinoff_grid import spy_all
    from .strategies import perf
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    market = Market(idx.values)
    cols = ['price_ticker', 'gvkey', 'event', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(sample='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(sample='2005–2026')], ignore_index=True)
    E['gvkey'] = E['gvkey'].astype(str)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    E = E.sort_values('entry_date').reset_index(drop=True)
    # A: return from first close to the close before entry; C: market cap at entry
    run_up, mcap = [], []
    for _, e in E.iterrows():
        p = load_price(e['price_ticker'])
        w = p.loc[:e['entry_date']].iloc[:-1]
        run_up.append(w['adj_close'].iloc[-1] / w['adj_close'].iloc[0] - 1 if len(w) > 1 else np.nan)
        mcap.append(float(w['raw_close'].iloc[-1] * w['shares_out'].iloc[-1]) if len(w) else np.nan)
    E['run_up'], E['mcap'] = run_up, mcap
    # B: book-to-market from the latest report before entry
    q = pd.concat([pd.read_parquet(wrds_dir(cfg) / f, columns=['gvkey', 'rdq', 'ceqq'])
                   for f in ('compustat_fundq_full.parquet', 'compustat_fundq_full_pre2003.parquet')])
    q['gvkey'] = q['gvkey'].astype(str)
    q['rdq'] = pd.to_datetime(q['rdq'])
    q = q.dropna(subset=['rdq', 'ceqq']).sort_values('rdq')
    bm = []
    for _, e in E.iterrows():
        g = q[(q['gvkey'] == e['gvkey']) & (q['rdq'] < e['entry_date'])]
        bm.append(g['ceqq'].iloc[-1] * 1e6 / e['mcap'] if len(g) and e['mcap'] > 0 else np.nan)
    E['bm'] = bm
    med = E['mcap'].expanding().median().shift(1)
    med.iloc[:10] = 1e9
    E['small'] = E['mcap'] < med
    halves = {
        'all (unfiltered)': np.ones(len(E), bool),
        'A dumped (fell before entry)': (E['run_up'] < 0).values,
        'A not dumped': (E['run_up'] >= 0).values,
        'B cheap (B/M >= 0.5)': (E['bm'] >= 0.5).values,
        'B not cheap (B/M < 0.5)': (E['bm'] < 0.5).values,
        'C small (below prior median)': E['small'].values,
        'C large': (~E['small']).values,
    }
    rows, books = [], {}
    for per, (a, b) in PERIODS.items():
        ev = E[E['sample'] == per].reset_index(drop=True)
        eng = SubsetEngine(ev, market, 252, ev['cost'].values)
        sub = E['sample'].values == per
        for name, m in halves.items():
            mask = m[sub].astype(float)
            if mask.sum() < 10:
                continue
            act = pd.Series(eng.active(mask), index=idx)
            p = eng.returns(mask, rf.values).reindex(idx).where(act, spy)
            start = max(pd.Timestamp(a), act[act].index[0])
            p = p.loc[start:b]
            cnt = pd.Series(np.asarray((eng.den[np.flatnonzero(mask)] > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
            s = perf(p, rf)
            books[(per, name)] = p
            rows.append({'period': per, 'book': name, 'spin-offs': int(mask.sum()), 'CAGR': s.get('CAGR'),
                         'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown'),
                         'avg positions': float(cnt[cnt > 0].mean()), 'SPY CAGR': perf(spy.loc[p.index], rf).get('CAGR')})
    R = pd.DataFrame(rows)
    adopted = []
    for name in halves:
        if name.startswith('all'):
            continue
        ok = True
        for per in PERIODS:
            x = R[(R['period'] == per) & (R['book'] == name)]
            base = R[(R['period'] == per) & (R['book'] == 'all (unfiltered)')].iloc[0]
            if x.empty or not (x['CAGR'].iloc[0] > base['CAGR'] and x['Sharpe'].iloc[0] > base['Sharpe']
                               and x['avg positions'].iloc[0] >= 6):
                ok = False
        if ok:
            adopted.append(name)
    R['filters adopted (pre-written rule)'] = ', '.join(adopted) or 'none'
    R['B coverage'] = R['period'].map(E.groupby('sample')['bm'].apply(lambda s: s.notna().mean()))
    # portfolio mixes, 2005-2026
    qqq = load_price('QQQ')['adj_close'].pct_change()
    qqq.index = pd.to_datetime(qqq.index)
    spin = books[('2005–2026', 'all (unfiltered)')]
    thrift = pd.read_parquet(DATA / 'niche_daily.parquet')['T1 Thrift conversions|1y']
    ins = pd.read_parquet(DATA / 'largecap_daily.parquet')['L2 S&P 500 + positive earnings surprise|6m']
    for s_ in (thrift, ins):
        s_.index = pd.to_datetime(s_.index)
    common = spin.index.intersection(qqq.dropna().index).intersection(ins.dropna().index)
    Q, SP = qqq.reindex(common).fillna(0), spin.reindex(common)
    TH, IN = thrift.reindex(common).fillna(spy.reindex(common)), ins.reindex(common)
    M = {'M0 70% QQQ + 30% spin-offs': mix([Q, SP], [.7, .3]),
         'M1 70% QQQ + 15% spin-offs + 15% thrifts': mix([Q, SP, TH], [.7, .15, .15]),
         'M2 60% QQQ + 30% spin-offs + 10% thrifts': mix([Q, SP, TH], [.6, .3, .1]),
         'M3 60% QQQ + 25% spin-offs + 15% insiders': mix([Q, SP, IN], [.6, .25, .15]),
         'QQQ': Q}
    mrows = []
    for name, p in M.items():
        for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s = perf(p.loc[sl], rf)
            mrows.append({'mix': name, 'period': lab, 'CAGR': s.get('CAGR'), 'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown')})
    MR = pd.DataFrame(mrows)
    g = lambda n, p, k: MR.loc[(MR['mix'] == n) & (MR['period'] == p), k].iloc[0]
    MR['better than M0 (pre-written rule)'] = MR['mix'].map(lambda n: n.startswith('M') and n[:2] != 'M0' and all(
        g(n, p, 'CAGR') > g('M0 70% QQQ + 30% spin-offs', p, 'CAGR') and g(n, p, 'Sharpe') > g('M0 70% QQQ + 30% spin-offs', p, 'Sharpe')
        for p in ('2006–15', '2016+')))
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'spinoff_filters.csv', index=False)
    MR.to_csv(RESULTS / 'spinoff_mixes.csv', index=False)
    LOG.info(f'filters: done; adopted: {adopted or "none"}')
    return R, MR


if __name__ == '__main__':
    run()
