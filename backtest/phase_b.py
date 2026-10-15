"""Phase 11: company fundamentals as filters on insider signals, judged LONG-ONLY (no shorting, no leverage).

Rules were written down and committed before any of this was run.

Costs: estimated half-spread on entry and exit (as before) plus 0.15% currency conversion each way
(Trading 212 GBP account). Entry at the next open after the signal; equal weight per signal.

Point-in-time data (nothing used before it was public):
  Annual (funda): available 90 days after the fiscal year end.
  Quarterly (fundq): available on the report date (rdq), else 45 days after the quarter end.
  Ratings: S&P long-term issuer rating as of the month before the signal.
  Index membership: S&P 500 / MidCap 400 / SmallCap 600 histories.

Features at each signal date:
  F-score (Piotroski 2000), last two annual reports, 9 points: ROA > 0; operating cash flow > 0;
    ROA up; cash flow > net income; long-term debt / assets down; current ratio up; no new shares
    (shares outstanding not up); gross margin up; asset turnover up.
  SUE: latest quarter's EPS (before extras) minus the same quarter a year earlier, divided by the standard
    deviation of that change over the prior 8 quarters (needs 4+); only if reported in the prior 90 days.
  Buyback yield: last fiscal year's share repurchases / market cap at the signal.
  Leverage: (long-term + short-term debt) / total assets, latest quarter.
  Rating: investment grade (BBB- or better), junk, or unrated.
  Index deletion: removed from the S&P 500, 400 or 600 in the prior 6 months.

Primary tests (Holm-corrected together) on two samples, (a) all cluster buys (>= 3 insiders, $200K) and
(b) tradeable big-drop cluster buys (stock down >= 50% in 6 months, >= $100K a day traded):
  P1 F-score >= 6        P2 SUE > 0 (positive surprise in the prior 90 days)
  P3 Buyback yield >= 1%
Measure: average 6-month and 12-month return minus SPY over the same dates, per trade, t-stats
clustered by entry month; calendar-time FF5+UMD alpha of the long-only portfolio; all years / 2006-15 / 2016+.
A filter "helps" if it beats the unfiltered sample in both periods.

Exploratory: F-score <= 3; SUE <= 0; leverage < 0.2; investment grade; junk; unrated; index deletion.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config, save_table
from .compustat import wrds_dir

LOG = get_logger('phase_b')
FX = 0.0015
TRAIN_END = pd.Timestamp('2015-12-31')
SP_INDEXES = ['000003', '024248', '030824']      # S&P 500, MidCap 400, SmallCap 600
RATING_ORDER = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B',
                'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D', 'SD']


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def fscore_table(cfg):
    a = pd.read_parquet(wrds_dir(cfg) / 'compustat_funda.parquet')
    a = num(a, ['at', 'ib', 'oancf', 'dltt', 'act', 'lct', 'csho', 'sale', 'cogs', 'prstkc'])
    a['datadate'] = pd.to_datetime(a['datadate'])
    a = a.sort_values(['gvkey', 'datadate']).drop_duplicates(['gvkey', 'datadate'], keep='last')
    g = a.groupby('gvkey')
    p = g.shift(1)
    at_avg = (a['at'] + p['at']) / 2
    roa = a['ib'] / p['at']
    roa_p = p['ib'] / g.shift(2)['at']
    cfo = a['oancf'] / p['at']
    lev, lev_p = a['dltt'] / at_avg, p['dltt'] / ((p['at'] + g.shift(2)['at']) / 2)
    cr, cr_p = a['act'] / a['lct'], p['act'] / p['lct']
    gm, gm_p = (a['sale'] - a['cogs']) / a['sale'], (p['sale'] - p['cogs']) / p['sale']
    tu, tu_p = a['sale'] / p['at'], p['sale'] / g.shift(2)['at']
    parts = [roa > 0, cfo > 0, roa > roa_p, cfo > roa, lev < lev_p, cr > cr_p, a['csho'] <= p['csho'],
             gm > gm_p, tu > tu_p]
    valid = roa.notna() & roa_p.notna() & cfo.notna()
    a['fscore'] = sum(x.fillna(False).astype(int) for x in parts).where(valid)
    a['buyback'] = a['prstkc']
    a['public'] = a['datadate'] + pd.Timedelta(days=90)
    return a[['gvkey', 'public', 'fscore', 'buyback']].dropna(subset=['public'])


def quarterly_table(cfg):
    q = pd.read_parquet(wrds_dir(cfg) / 'compustat_fundq_full.parquet')
    q = num(q, ['epspxq', 'atq', 'dlttq', 'dlcq', 'fqtr', 'fyearq'])
    q['datadate'] = pd.to_datetime(q['datadate'])
    q['rdq'] = pd.to_datetime(q['rdq'], errors='coerce')
    q = q.sort_values(['gvkey', 'datadate']).drop_duplicates(['gvkey', 'datadate'], keep='last')
    q['public'] = q['rdq'].fillna(q['datadate'] + pd.Timedelta(days=45))
    g = q.groupby('gvkey')
    chg = q['epspxq'] - g['epspxq'].shift(4)
    q['chg'] = chg
    sd = q.groupby('gvkey')['chg'].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q['sue'] = (chg / sd).where(sd > 0)
    q['leverage'] = (q['dlttq'].fillna(0) + q['dlcq'].fillna(0)) / q['atq']
    return q[['gvkey', 'public', 'sue', 'leverage']]


def asof(sig, table, cols, key='gvkey', max_age=None):
    """Latest table row per key with public <= signal date."""
    out = pd.DataFrame(index=sig.index, columns=cols + ['public'], dtype=float)
    t = table.dropna(subset=['public']).sort_values('public')
    by = {k: g for k, g in t.groupby(key)}
    for i, (k, d) in enumerate(zip(sig[key].values, sig['signal_date'].values)):
        g = by.get(k)
        if g is None:
            continue
        j = np.searchsorted(g['public'].values, d, 'right') - 1
        if j < 0:
            continue
        row = g.iloc[j]
        if max_age is not None and (pd.Timestamp(d) - row['public']).days > max_age:
            continue
        for c in cols:
            out.iat[i, out.columns.get_loc(c)] = row[c]
        out.iat[i, out.columns.get_loc('public')] = row['public'].value
    return out


def ratings(cfg):
    r = pd.read_parquet(wrds_dir(cfg) / 'compustat_ratings.parquet')
    r['datadate'] = pd.to_datetime(r['datadate'])
    r = r.dropna(subset=['splticrm'])
    rank = {k: i for i, k in enumerate(RATING_ORDER)}
    r['rank'] = r['splticrm'].map(rank)
    r['public'] = r['datadate'] + pd.Timedelta(days=31)
    return r[['gvkey', 'public', 'rank']]


def index_deletions(cfg):
    m = pd.read_parquet(wrds_dir(cfg) / 'compustat_index_members.parquet')
    m = m[m['gvkeyx'].isin(SP_INDEXES)].copy()
    m['thru'] = pd.to_datetime(m['thru'], errors='coerce')
    return m.dropna(subset=['thru'])[['gvkey', 'thru', 'gvkeyx']]


def features(cfg, S):
    S = S.copy()
    S['gvkey'] = S['price_ticker'].astype(str).str.split('_').str[1]
    f = fscore_table(cfg)
    q = quarterly_table(cfg)
    r = ratings(cfg)
    S[['fscore', 'buyback', 'f_public']] = asof(S, f, ['fscore', 'buyback'], max_age=550).values
    S[['sue', 'leverage', 'q_public']] = asof(S, q, ['sue', 'leverage'], max_age=200).values
    S['q_public'] = pd.to_datetime(S['q_public'])
    S['sue_recent'] = S['sue'].where((S['signal_date'] - S['q_public']).dt.days <= 90)
    S[['rating_rank', 'r_public']] = asof(S, r, ['rank'], max_age=400).values
    S['buyback_yield'] = S['buyback'] * 1e6 / S['market_cap']
    dele = index_deletions(cfg)
    by = {k: g['thru'].values for k, g in dele.groupby('gvkey')}
    S['index_deleted_6m'] = [bool(((by.get(k, np.array([], 'datetime64[ns]')) <= np.datetime64(d)) &
                                   (by.get(k, np.array([], 'datetime64[ns]')) > np.datetime64(d) - np.timedelta64(182, 'D'))).any())
                             for k, d in zip(S['gvkey'], S['signal_date'].values)]
    return S


FILTERS = [
    ('Primary', 'P1 F-score >= 6', lambda t: t['fscore'] >= 6),
    ('Primary', 'P2 Positive earnings surprise (SUE > 0, last 90 days)', lambda t: t['sue_recent'] > 0),
    ('Primary', 'P3 Buyback yield >= 1%', lambda t: t['buyback_yield'] >= 0.01),
    ('Exploratory', 'F-score <= 3', lambda t: t['fscore'] <= 3),
    ('Exploratory', 'SUE <= 0 (last 90 days)', lambda t: t['sue_recent'] <= 0),
    ('Exploratory', 'Leverage < 20% of assets', lambda t: t['leverage'] < 0.2),
    ('Exploratory', 'Investment grade (BBB- or better)', lambda t: t['rating_rank'] <= RATING_ORDER.index('BBB-')),
    ('Exploratory', 'Junk rated', lambda t: t['rating_rank'] > RATING_ORDER.index('BBB-')),
    ('Exploratory', 'Unrated', lambda t: t['rating_rank'].isna()),
    ('Exploratory', 'Removed from an S&P index in prior 6 months', lambda t: t['index_deleted_6m']),
]


def per_trade(t, h):
    """Return minus SPY per trade, after FX, with month-clustered t."""
    col, spy = f'ret_{h}', f'spy_{h}'
    x = t.dropna(subset=[col, spy])
    x = x[~x[f'truncated_{h}'].fillna(False).astype(bool)] if f'truncated_{h}' in x else x
    if len(x) < 20:
        return np.nan, np.nan, len(x)
    d = (1 + x[col]) * (1 - FX) ** 2 - 1 - x[spy]
    mu = d.mean()
    g = (d - mu).groupby(pd.to_datetime(x['entry_date']).dt.to_period('M')).sum()
    se = np.sqrt((g ** 2).sum()) / len(d)
    return mu, mu / se if se > 0 else np.nan, len(d)


def run(cfg=None):
    cfg = cfg or load_config()
    from .robust import _setup
    from .filters import evaluate
    factors, cal, market, rf, scorer, F = _setup(cfg)
    F = F[(F['screen'] == 'cluster3') & F['price_ticker'].astype(str).str.startswith('CS_')].copy()
    F['cost_rt'] = F['cost_rt'] + 2 * FX
    F = features(cfg, F)
    save_table(F[['screen', 'signal_date', 'issuer_cik', 'price_ticker', 'fscore', 'sue_recent', 'buyback_yield',
                  'leverage', 'rating_rank', 'index_deleted_6m']], DATA / 'signal_fundamentals')
    LOG.info('phase_b: coverage ' + ', '.join(f'{c} {F[c].notna().mean():.0%}' for c in
                                               ['fscore', 'sue_recent', 'buyback_yield', 'leverage', 'rating_rank']))
    samples = {'All cluster buys': F,
               'Big-drop, tradeable': F[(F['ret_prior_6m'] <= -0.5) & (F['adv20'] >= 100_000)]}
    rows = []
    for sname, base in samples.items():
        for group, name, fn in [('Baseline', 'No filter', lambda t: pd.Series(True, index=t.index))] + FILTERS:
            m = fn(base).fillna(False).astype(bool)
            t = base[m]
            if len(t) < 30:
                continue
            r = {'sample': sname, 'group': group, 'filter': name, 'signals': len(t), 'share kept': m.mean()}
            late = t['signal_date'] > TRAIN_END
            for h in ('6m', '1y'):
                for lab, sel in (('all', slice(None)), ('2006–15', ~late), ('2016+', late)):
                    mu, tt, n = per_trade(t[sel] if not isinstance(sel, slice) else t, h)
                    r[f'{h} vs SPY {lab}'], r[f'{h} t {lab}'] = mu, tt
            e = evaluate(t, market, factors, 126, entry='open', weights='buyhold')
            r['6m alpha'], r['6m alpha t'], r['6m alpha 2016+'] = e['full alpha'], e['full t'], e['test alpha']
            rows.append(r)
            LOG.info(f'phase_b: {sname} / {name} done')
    d = pd.DataFrame(rows)
    base = d[d['filter'] == 'No filter'].set_index('sample')
    for col in ('6m vs SPY 2006–15', '6m vs SPY 2016+', '1y vs SPY 2006–15', '1y vs SPY 2016+'):
        d[col + ' vs baseline'] = d[col] - d['sample'].map(base[col])
    d['helps in both periods (6m)'] = (d['6m vs SPY 2006–15 vs baseline'] > 0) & (d['6m vs SPY 2016+ vs baseline'] > 0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / 'phase_b_filters.csv', index=False)
    LOG.info('phase_b: done')
    return d


if __name__ == '__main__':
    run()
