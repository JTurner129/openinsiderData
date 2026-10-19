"""Phase 35: does a spin-off's FIRST quarterly earnings report tell us whether to keep it?

Rules were written down and committed before this was run.

Events: Phase 23 + Phase 21 US spin-offs (entry 21 trading days after first trade, 12-month hold, same costs).
First report: the earliest Compustat quarterly record (fundq, 1987+) for the spin-off's gvkey with a report date
(rdq) after the entry date and at least 30 trading days before the planned exit. "Rose" if EPS before extras
(epspxq) is above the same fiscal quarter a year earlier (Compustat often carries pre-spin history from the Form 10);
"fell" if below; reports with no prior-year comparison are excluded and counted.
Return measured from the open two trading days after rdq to the planned exit (minus half the cost, as a sale at
the planned exit), excess over SPY over the same days.
Rule being tested: sell early when the first report "fell".
Reported: n, mean, median, share > 0 of the post-report excess for "rose" and "fell", difference with a two-sample
month-clustered t, by period (1994-2004, 2005-2026, all); coverage.
Pre-written bar: rose minus fell >= 5 points with t >= 2 over all years AND positive in both periods, AND the fell
group's post-report mean excess < 0. Otherwise the plan stays "hold every spin-off 12 months".
"""
import os

import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config

LOG = get_logger('first_earnings')


def run(cfg=None):
    assert os.environ.get('CS_CACHE_NAME') == 'prices_cs_full', 'run with CS_CACHE_NAME=prices_cs_full'
    cfg = cfg or load_config()
    import statsmodels.formula.api as smf
    from .compustat import wrds_dir
    from .forced_flows import next_day
    from .prices import load_price
    from .spinoff_grid import spy_all
    px = spy_all()
    idx = pd.DatetimeIndex(px.index.values)
    q = pd.concat([pd.read_parquet(wrds_dir(cfg) / f, columns=['gvkey', 'datadate', 'rdq', 'fyearq', 'fqtr', 'epspxq'])
                   for f in ('compustat_fundq_full.parquet', 'compustat_fundq_full_pre2003.parquet')])
    q['gvkey'] = q['gvkey'].astype(str)
    q['rdq'] = pd.to_datetime(q['rdq'], errors='coerce')
    q['epspxq'] = pd.to_numeric(q['epspxq'], errors='coerce')
    q = q.drop_duplicates(['gvkey', 'fyearq', 'fqtr'])
    prev = q[['gvkey', 'fyearq', 'fqtr', 'epspxq']].copy()
    prev['fyearq'] = prev['fyearq'] + 1
    q = q.merge(prev.rename(columns={'epspxq': 'eps_ly'}), on=['gvkey', 'fyearq', 'fqtr'], how='left')
    by = {k: g.sort_values('rdq') for k, g in q.dropna(subset=['rdq']).groupby('gvkey')}
    cols = ['price_ticker', 'gvkey', 'entry_date', 'cost']
    E = pd.concat([pd.read_parquet(DATA / 'holdout_spinoff_events.parquet')[cols].assign(period='1994–2004'),
                   pd.read_parquet(DATA / 'spinoff_events.parquet')[cols].assign(period='2005–2026')], ignore_index=True)
    E['gvkey'] = E['gvkey'].astype(str).str.zfill(6)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    rows = []
    for _, e in E.iterrows():
        exit_d = next_day(idx, e['entry_date'], 252)
        g = by.get(e['gvkey'])
        if g is None or exit_d is None:
            rows.append({'status': 'no report'})
            continue
        g = g[(g['rdq'] > e['entry_date']) & (g['rdq'] < exit_d - pd.Timedelta(days=45))]
        if not len(g):
            rows.append({'status': 'no report'})
            continue
        r = g.iloc[0]
        if pd.isna(r['epspxq']) or pd.isna(r['eps_ly']) or r['epspxq'] == r['eps_ly']:
            rows.append({'status': 'no comparison'})
            continue
        d0 = next_day(idx, r['rdq'], 2)
        a = load_price(e['price_ticker'])['adj_close'].loc[d0:exit_d]
        s = px.loc[d0:exit_d]
        if len(a) < 20 or len(s) < 20:
            rows.append({'status': 'no prices'})
            continue
        ex = (a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost'] / 2) - (s.iloc[-1] / s.iloc[0] - 1)
        rows.append({'status': 'ok', 'period': e['period'], 'group': 'rose' if r['epspxq'] > r['eps_ly'] else 'fell',
                     'ex': ex, 'month': str(pd.Timestamp(d0).to_period('M'))})
    D = pd.DataFrame(rows)
    cov = D['status'].value_counts().to_dict()
    D = D[D['status'] == 'ok']
    out = []
    for lab in ('1994–2004', '2005–2026', 'all'):
        x = D if lab == 'all' else D[D['period'] == lab]
        rs, fl = x[x['group'] == 'rose'], x[x['group'] == 'fell']
        t = np.nan
        if len(rs) > 5 and len(fl) > 5:
            d = x.assign(grp=(x['group'] == 'rose').astype(int))
            t = smf.ols('ex ~ grp', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['month'])[0]}).tvalues['grp']
        out.append({'period': lab, 'rose n': len(rs), 'rose mean': rs['ex'].mean(), 'rose median': rs['ex'].median(),
                    'fell n': len(fl), 'fell mean': fl['ex'].mean(), 'fell median': fl['ex'].median(),
                    'difference': rs['ex'].mean() - fl['ex'].mean(), 't (difference)': t})
    R = pd.DataFrame(out)
    g = R.set_index('period')
    R['adopted (pre-written)'] = bool(g.loc['all', 'difference'] >= 0.05 and g.loc['all', 't (difference)'] >= 2 and
                                      g.loc['1994–2004', 'difference'] > 0 and g.loc['2005–2026', 'difference'] > 0 and
                                      g.loc['all', 'fell mean'] < 0)
    R['coverage'] = str(cov)
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'first_earnings.csv', index=False)
    LOG.info('first_earnings: done')
    return R
