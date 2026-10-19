"""Phase 33: two last spin-off ideas, US only, Compustat prices 1994-2026.

Rules were written down and committed before this was run.

A. Buy the PARENT after a spin-off. Events: LSEG Deals (SDC) US spin-offs (nation United States) with effective
   date 1994-01-01 to 2025-10-31. The parent (SDC ultimate parent name, column 17: the listed group; immediate parents are often unlisted subsidiaries) is matched to a Compustat
   company by normalised name (lower case, punctuation and Inc/Corp/Co/Holdings/Group/Ltd/The removed):
   exact normalised match, or difflib similarity >= 0.92, among companies whose main security trades on the
   effective date. Unmatched parents are dropped (reported). One event per parent per year.
   Entry: open 21 trading days after the SDC effective date; hold 12 months; tradeable price >= $1 and
   >= $100K average daily dollar volume; same costs as Phase 19. Reported: per-trade 12-month excess over SPY
   (mean, median, share > 0, month-clustered t), stand-alone book (CAGR, Sharpe, max DD, avg positions) for
   1994-2004, 2005-2026 and all, against SPY.
   Bar: per-trade mean > 0 with t >= 2 AND book beats SPY on CAGR in both periods.
B. Relative size. For our spin-off events (Phases 21+23) that match an SDC deal (as in Phase 32) AND whose parent
   matches a Compustat company (rule A), relative size = spin-off market cap at entry / parent market cap on the
   same day. Split at the median: "small relative" vs "large relative".
   Reported: per-trade excess over SPY for each half, difference with a two-sample month-clustered t, by period.
   Bar: small-relative beats large-relative by >= 5 points per trade with t >= 2 over all years AND the
   difference is positive in both periods.
"""
import difflib
import os
import re

import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('parent_size')
STOP = {'inc', 'corp', 'co', 'holdings', 'holding', 'group', 'ltd', 'the', 'company', 'corporation', 'plc', 'llc',
        'cos', 'companies', 'incorporated', 'limited', 'lp', 'trust'}


def norm(s):
    return ' '.join(w for w in re.sub(r'[^a-z0-9 ]', ' ', str(s).lower()).split() if w not in STOP)


def two_sample_t(f, s, col):
    import statsmodels.formula.api as smf
    d = pd.concat([f.assign(grp=1), s.assign(grp=0)]).dropna(subset=[col])
    if d['grp'].nunique() < 2 or len(d) < 10:
        return np.nan
    r = smf.ols(f'{col} ~ grp', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['month'])[0]})
    return r.tvalues['grp']


def clustered(x, months):
    x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
    m = pd.Series(np.asarray(months)).reset_index(drop=True)
    ok = x.notna()
    x, m = x[ok], m[ok]
    g = (x - x.mean()).groupby(m).sum()
    return x.mean() / (np.sqrt((g ** 2).sum()) / len(x)) if len(x) > 5 else np.nan


def run(cfg=None, sdc_xlsx='eikon/sdc_spinoffs_2026-10-19.xlsx'):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    from .combos import SubsetEngine
    from .compustat import CS_CACHE, wrds_dir
    from .filters import Market, corwin_schultz
    from .forced_flows import next_day, tradeable
    from .prices import load_price
    from .spinoff_grid import spy_all
    from .strategies import perf
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    spy_r = px.pct_change().reindex(idx).fillna(0.0)
    rf = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')['RF'].reindex(idx).fillna(0.0)
    a = pd.read_excel(sdc_xlsx, header=None)
    sdc = pd.DataFrame({'deal': a[0], 'target': a[10], 'parent': a[17], 'nation': a[15],
                        'eff': pd.to_datetime(a[3].astype(str).str[4:15], format='%b %d %Y', errors='coerce')})
    sdc = sdc[(sdc['nation'] == 'United States') & (sdc['eff'] >= '1994-01-01') & (sdc['eff'] <= '2025-10-31')]
    sdc = sdc[sdc['parent'].astype(str).str.lower() != sdc['target'].astype(str).str.lower()]
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'conm'])
    co['gvkey'] = co['gvkey'].astype(str)
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    meta['first_date'] = pd.to_datetime(meta['first_date'])
    meta['last_date'] = pd.to_datetime(meta['last_date'])
    main = meta.sort_values('median_dollar_volume', ascending=False).drop_duplicates('gvkey')
    co = co.merge(main[['gvkey', 'sec', 'first_date', 'last_date']], on='gvkey')
    co['n'] = co['conm'].map(norm)
    by_name = {k: g for k, g in co.groupby('n')}
    names = list(by_name)
    picks = []
    for _, r in sdc.iterrows():
        n = norm(r['parent'])
        cands = by_name.get(n)
        if cands is None:
            close = difflib.get_close_matches(n, names, n=1, cutoff=0.92)
            cands = by_name.get(close[0]) if close else None
        if cands is None:
            picks.append(None)
            continue
        live = cands[(cands['first_date'] <= r['eff']) & (cands['last_date'] >= r['eff'] + pd.Timedelta(days=40))]
        picks.append(('CS_' + str(live.iloc[0]['sec']), live.iloc[0]['gvkey']) if len(live) else None)
    sdc['pick'] = picks
    LOG.info(f'parent_size: {len(sdc)} US SDC spin-offs, parent matched for {sdc["pick"].notna().sum()}')
    P = sdc.dropna(subset=['pick']).copy()
    P['price_ticker'] = P['pick'].map(lambda x: x[0])
    P['pgvkey'] = P['pick'].map(lambda x: x[1])
    P['entry_date'] = [next_day(idx, d, 21) for d in P['eff']]
    P = P.dropna(subset=['entry_date'])
    P['yr'] = P['entry_date'].dt.year
    P = P.drop_duplicates(['pgvkey', 'yr']).reset_index(drop=True)
    keep, costs = [], []
    for i, e in P.iterrows():
        ok, w = tradeable(e['price_ticker'], e['entry_date'])
        if not ok:
            continue
        sp = np.nanmedian(corwin_schultz(w['adj_high'].values, w['adj_low'].values, w['adj_close'].values))
        keep.append(i)
        costs.append((sp if np.isfinite(sp) else 0.01) + 0.003)
    P = P.loc[keep].reset_index(drop=True)
    P['cost'] = costs
    ex = []
    for _, e in P.iterrows():
        a_ = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
        s_ = px.loc[e['entry_date']:].iloc[:253]
        ex.append((a_.iloc[-1] / a_.iloc[0] - 1) * (1 - e['cost']) - (s_.iloc[-1] / s_.iloc[0] - 1) if len(a_) > 1 else np.nan)
    P['ex'] = ex
    P['month'] = P['entry_date'].dt.to_period('M').astype(str)
    market = Market(idx.values)
    rows = []
    for lab, a0, b0 in (('1994–2004', '1994-01-01', '2004-12-31'), ('2005–2026', '2005-01-01', None), ('all', '1994-01-01', None)):
        x = P[(P['entry_date'] >= a0) & ((P['entry_date'] <= b0) if b0 else True)].reset_index(drop=True)
        eng = SubsetEngine(x, market, 252, x['cost'].values)
        mask = np.ones(len(x))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r).loc[act[act].index[0]:b0]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
        s, b = perf(p, rf), perf(spy_r.loc[p.index], rf)
        rows.append({'test': 'A parents', 'period': lab, 'n': len(x), 'mean excess': x['ex'].mean(), 'median': x['ex'].median(),
                     'share > 0': (x['ex'] > 0).mean(), 't': clustered(x['ex'], x['month']), 'book CAGR': s.get('CAGR'),
                     'SPY CAGR': b.get('CAGR'), 'book Sharpe': s.get('Sharpe'), 'SPY Sharpe': b.get('Sharpe'),
                     'book max DD': s.get('max drawdown'), 'avg positions': float(cnt[cnt > 0].mean())})
    RA = pd.DataFrame(rows)
    g = RA.set_index('period')
    RA['bar met (A)'] = bool(g.loc['all', 'mean excess'] > 0 and g.loc['all', 't'] >= 2 and
                             all(g.loc[p, 'book CAGR'] > g.loc[p, 'SPY CAGR'] for p in ('1994–2004', '2005–2026')))
    # B: relative size, using the Phase 32 matches (focus_us_events.csv holds our events matched to SDC)
    F = pd.read_csv(RESULTS / 'focus_us_events.csv', parse_dates=['event', 'entry_date'])
    F['k'] = F['conm'].map(norm)
    sd = sdc.dropna(subset=['pick']).copy()
    sd['tk'] = sd['target'].map(norm)
    rel = []
    for _, e in F.iterrows():
        c = sd[(sd['eff'] >= e['event'] - pd.Timedelta(days=60)) & (sd['eff'] <= e['event'] + pd.Timedelta(days=120))]
        c = c[c['tk'].map(lambda t: bool(e['k']) and e['k'].split()[0] in t.split())]
        if not len(c):
            rel.append(np.nan)
            continue
        pt = c.iloc[0]['pick'][0]
        sp_, pp_ = load_price(e['price_ticker']), load_price(pt)
        try:
            ms = float(sp_.loc[:e['entry_date']].iloc[-1]['raw_close'] * sp_.loc[:e['entry_date']].iloc[-1]['shares_out'])
            mp = float(pp_.loc[:e['entry_date']].iloc[-1]['raw_close'] * pp_.loc[:e['entry_date']].iloc[-1]['shares_out'])
            rel.append(ms / mp if mp > 0 else np.nan)
        except Exception:
            rel.append(np.nan)
    F['rel'] = rel
    K = F.dropna(subset=['rel', 'ex'])
    med = K['rel'].median()
    rowsb = []
    for lab in ('1994–2004', '2005–2026', 'all'):
        x = K if lab == 'all' else K[K['period'] == lab]
        sm, lg = x[x['rel'] <= med], x[x['rel'] > med]
        rowsb.append({'test': 'B relative size', 'period': lab, 'n': len(x), 'median relative size': med,
                      'small-relative n': len(sm), 'small mean': sm['ex'].mean(), 'small median': sm['ex'].median(),
                      'large-relative n': len(lg), 'large mean': lg['ex'].mean(), 'large median': lg['ex'].median(),
                      'difference': sm['ex'].mean() - lg['ex'].mean(), 't (difference)': two_sample_t(sm, lg, 'ex')})
    RB = pd.DataFrame(rowsb)
    gb = RB.set_index('period')
    RB['bar met (B)'] = bool(gb.loc['all', 'difference'] >= 0.05 and gb.loc['all', 't (difference)'] >= 2 and
                             gb.loc['1994–2004', 'difference'] > 0 and gb.loc['2005–2026', 'difference'] > 0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    RA.to_csv(RESULTS / 'parent_after_spinoff.csv', index=False)
    RB.to_csv(RESULTS / 'spinoff_relative_size.csv', index=False)
    P.to_csv(RESULTS / 'parent_after_spinoff_trades.csv', index=False)
    LOG.info('parent_size: done')
    return RA, RB
