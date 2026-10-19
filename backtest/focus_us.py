"""Phase 32: does "spin-off in a different industry from its parent" (focus-increasing) beat same-industry
spin-offs in OUR US sample (Phase 21 + Phase 23 events, Compustat prices, 1994-2026)?

Rules were written down and committed before this was run. Motivation: Phase 31b found +16.7 points per trade
(t 2.66) on the SDC US list with Datastream prices; this is a partly independent check (different event list,
different prices) and adds the 1994-2004 holdout.

Matching: each of our spin-off events (gvkey, first trading day) is matched to an SDC US spin-off (LSEG Deals
export, column 15 nation = United States) whose effective date is within -60/+120 days of our first trading day
and whose target name shares the first significant word (after removing Inc/Corp/Co/Holdings/Group/Ltd/The and
punctuation) with the Compustat company name. Unmatched events are reported, not guessed.
Industry: SIC descriptions from the SDC SIC export (target column 20, immediate parent column 21).
Focus-increasing = descriptions differ. Same returns as Phases 21/23 (12-month excess over SPY after costs).
Reported: n, mean, median, share > 0 for focus vs same, by period (1994-2004, 2005-2026, all), difference and
month-clustered t; plus books (each group alone): CAGR, Sharpe, max DD, avg positions.
Pre-written bar for using it as a rule: focus beats same-industry by >= 5 points per trade with clustered t >= 2
over all years AND the difference is positive in both periods.
"""
import os
import re

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('focus_us')
STOP = {'inc', 'corp', 'co', 'holdings', 'holding', 'group', 'ltd', 'the', 'company', 'corporation', 'plc', 'llc',
        'international', 'intl', 'new', 'spinco', 'of', 'and', '&'}


def key(name):
    w = [x for x in re.sub(r'[^a-z0-9 ]', ' ', str(name).lower()).split() if x not in STOP]
    return w[0] if w else ''


def run(cfg=None, sdc_xlsx='eikon/sdc_spinoffs_2026-10-19.xlsx', sic_xlsx='eikon/sdc_sic_2026-10-19.xlsx'):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    from .combos import SubsetEngine
    from .compustat import wrds_dir
    from .filters import Market
    from .prices import load_price
    from .spinoff_grid import spy_all
    from .strategies import perf
    a = pd.read_excel(sdc_xlsx, header=None)
    sdc = pd.DataFrame({'deal': a[0], 'name': a[10].astype(str) + ' ' + a[9].astype(str), 'nation': a[15],
                        'eff': pd.to_datetime(a[3].astype(str).str[4:15], format='%b %d %Y', errors='coerce')})
    b = pd.read_excel(sic_xlsx, header=None)[[0, 20, 21]].rename(columns={0: 'deal', 20: 'sic_t', 21: 'sic_p'})
    sdc = sdc[sdc['nation'] == 'United States'].merge(b, on='deal')
    cols = ['price_ticker', 'gvkey', 'event', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(period='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(period='2005–2026')], ignore_index=True)
    E['gvkey'] = E['gvkey'].astype(str)
    E['event'] = pd.to_datetime(E['event'])
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'conm'])
    co['gvkey'] = co['gvkey'].astype(str)
    E = E.merge(co, on='gvkey', how='left')
    E['k'] = E['conm'].map(key)
    match = []
    for i, e in E.iterrows():
        c = sdc[(sdc['eff'] >= e['event'] - pd.Timedelta(days=60)) & (sdc['eff'] <= e['event'] + pd.Timedelta(days=120))]
        c = c[c['name'].map(lambda s: e['k'] != '' and e['k'] in re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split())]
        match.append((c.iloc[0]['sic_t'], c.iloc[0]['sic_p']) if len(c) else (np.nan, np.nan))
    E[['sic_t', 'sic_p']] = pd.DataFrame(match, index=E.index)
    E['known'] = E['sic_t'].notna() & E['sic_p'].notna()
    E['focus'] = E['known'] & (E['sic_t'] != E['sic_p'])
    LOG.info(f'focus_us: matched {E["known"].sum()} of {len(E)} events to SDC parent industry')
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy_r = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    ex = []
    for _, e in E.iterrows():
        p = load_price(e['price_ticker'])
        a_ = p['adj_close'].loc[e['entry_date']:].iloc[:253] if p is not None else pd.Series(dtype=float)
        s_ = px.loc[e['entry_date']:].iloc[:253]
        ex.append((a_.iloc[-1] / a_.iloc[0] - 1) * (1 - e['cost']) - (s_.iloc[-1] / s_.iloc[0] - 1) if len(a_) > 1 else np.nan)
    E['ex'] = ex
    E['month'] = E['entry_date'].dt.to_period('M').astype(str)
    market = Market(idx.values)

    def cl(x, m):
        x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
        m = pd.Series(np.asarray(m)).reset_index(drop=True)
        ok = x.notna()
        x, m = x[ok], m[ok]
        if len(x) < 5:
            return np.nan
        g = (x - x.mean()).groupby(m).sum()
        return x.mean() / (np.sqrt((g ** 2).sum()) / len(x))

    def book(ev):
        ev = ev.reset_index(drop=True)
        eng = SubsetEngine(ev, market, 252, ev['cost'].values)
        mask = np.ones(len(ev))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r).loc[act[act].index[0]:]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
        s = perf(p, rf)
        return {'book CAGR': s.get('CAGR'), 'book Sharpe': s.get('Sharpe'), 'book max DD': s.get('max drawdown'),
                'avg positions': float(cnt[cnt > 0].mean()), 'SPY CAGR': perf(spy_r.loc[p.index], rf).get('CAGR')}

    rows = []
    K = E[E['known']]
    for per in ('1994–2004', '2005–2026', 'all'):
        x = K if per == 'all' else K[K['period'] == per]
        f, s = x[x['focus']], x[~x['focus']]
        r = {'period': per, 'matched': len(x), 'focus n': len(f), 'focus mean': f['ex'].mean(), 'focus median': f['ex'].median(),
             'focus share>0': (f['ex'] > 0).mean(), 'same n': len(s), 'same mean': s['ex'].mean(), 'same median': s['ex'].median(),
             'same share>0': (s['ex'] > 0).mean(), 'difference': f['ex'].mean() - s['ex'].mean(),
             't (difference)': cl(f['ex'] - s['ex'].mean(), f['month'])}
        if per == 'all':
            r.update({('focus ' + k): v for k, v in book(f).items()})
            r.update({('same ' + k): v for k, v in book(s).items()})
        rows.append(r)
    R = pd.DataFrame(rows)
    g = R.set_index('period')
    R['adopted (pre-written)'] = bool(g.loc['all', 'difference'] >= 0.05 and g.loc['all', 't (difference)'] >= 2
                                      and g.loc['1994–2004', 'difference'] > 0 and g.loc['2005–2026', 'difference'] > 0)
    R['unmatched events'] = int((~E['known']).sum())
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'focus_us.csv', index=False)
    E.to_csv(RESULTS / 'focus_us_events.csv', index=False)
    LOG.info('focus_us: done')
    return R
