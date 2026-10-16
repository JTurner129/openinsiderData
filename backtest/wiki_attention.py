"""Phase 10: investor attention from English Wikipedia page views (Da, Engelberg & Gao 2011 used Google search).

Rules fixed before any page-view data was downloaded:

  Companies: SEC CIKs linked to an English Wikipedia article through Wikidata property P5531, kept when
  the CIK is a US-headquartered Compustat company.
  Views: Wikimedia "pageview_complete" monthly dumps, user agents only (no bots), all access types summed,
  July 2015 onward.
  Abnormal attention at each month-end = log(1 + views in that month) - log(1 + median monthly views over
  the prior 12 months). Needs 12 months of history, so tests start in mid-2016.

  Test 1: at each month-end, quintiles of abnormal attention among mapped stocks with price >= $1 and
  >= $100K a day traded; equal weight; held one month. Report top minus bottom and each quintile against
  SPY, FF5+UMD alpha, for 2016-2020 and 2021+ separately.
  Test 2: cluster-buy and big-drop signals split by the company's abnormal attention in the month before
  the signal (above vs below zero), per-trade 6-month return and matched edge.

Downloader:  python -c "from backtest import wiki_attention as w; w.download()"
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CACHE, get_logger

LOG = get_logger('wiki')
OUT = CACHE / 'wiki'
UA = 'insider-backtest-research/1.0'
BASE = 'https://dumps.wikimedia.org/other/pageview_complete/monthly'


def months():
    m = pd.period_range('2015-07', pd.Timestamp.today().to_period('M') - 1, freq='M')
    return [p for p in m]


def parse_days(code):
    """'A12B3C40' -> {1: 12, 2: 3, 3: 40}: a letter per day of the month, then that day's views."""
    out, i = {}, 0
    while i < len(code):
        day = ord(code[i]) - 64
        j = i + 1
        while j < len(code) and code[j].isdigit():
            j += 1
        if j > i + 1:
            out[day] = out.get(day, 0) + int(code[i + 1:j])
        i = j
    return out


def month_file(p, titles):
    out = OUT / f'views_{p}.parquet'
    if out.exists():
        return p, 'cached'
    pat = OUT / '_patterns.txt'
    url = f'{BASE}/{p.year}/{p}/pageviews-{p.strftime("%Y%m")}-user.bz2'
    cmd = (f"curl -sS --retry 5 --retry-delay 20 -A '{UA}' '{url}' | bzip2 -dc | "
           f"grep -a -F -f '{pat}' || true")
    for attempt in range(3):
        res = subprocess.run(['bash', '-c', 'set -o pipefail; ' + cmd], capture_output=True, text=True)
        if res.stdout:
            break
        LOG.warning(f'wiki {p}: empty result (attempt {attempt + 1}): {res.stderr[:200]}')
        time.sleep(60)
    rows = []
    for line in res.stdout.splitlines():
        parts = line.split(' ')
        if len(parts) < 6 or parts[0] != 'en.wikipedia' or parts[1] not in titles:
            continue
        for d, v in parse_days(parts[5]).items():
            rows.append((parts[1], d, v))
    df = pd.DataFrame(rows, columns=['title', 'day', 'views']).groupby(['title', 'day'], as_index=False)['views'].sum()
    df['date'] = pd.to_datetime(dict(year=p.year, month=p.month, day=df['day']), errors='coerce')
    df.drop(columns='day').to_parquet(out, index=False)
    return p, f'{df["title"].nunique():,} articles'


def download(parallel=2):
    m = pd.read_parquet(OUT / 'cik_titles.parquet')
    titles = set(m['title'])
    (OUT / '_patterns.txt').write_text(''.join(f'en.wikipedia {t} \n' for t in sorted(titles)))
    todo = months()
    LOG.info(f'wiki: {len(titles):,} articles, {len(todo)} monthly dumps')
    t0 = time.time()
    with ThreadPoolExecutor(parallel) as ex:
        for i, (p, msg) in enumerate(ex.map(lambda p: month_file(p, titles), todo)):
            el = (time.time() - t0) / 3600
            LOG.info(f'wiki {p}: {msg} ({i + 1}/{len(todo)}, {el:.1f} h elapsed)')
    LOG.info('wiki: download done')


def daily_views():
    parts = [pd.read_parquet(p) for p in sorted(OUT.glob('views_*.parquet'))]
    v = pd.concat(parts, ignore_index=True)
    m = pd.read_parquet(OUT / 'cik_titles.parquet')
    return v.merge(m[['title', 'cik']], on='title')


if __name__ == '__main__':
    download()


def abnormal_attention():
    """Monthly abnormal attention per CIK (rules in the module docstring)."""
    v = daily_views()
    v['month'] = v['date'].dt.to_period('M')
    m = v.groupby(['cik', 'month'])['views'].sum().unstack('month').sort_index(axis=1)
    m = m.reindex(columns=pd.period_range(m.columns.min(), m.columns.max(), freq='M')).fillna(0)
    lv = np.log1p(m)
    med = np.log1p(m.T.rolling(12, min_periods=12).median().shift(1).T)
    ab = (lv - med).stack().rename('abnormal').reset_index()
    ab.columns = ['cik', 'month', 'abnormal']
    return ab.dropna()


def test(cfg=None):
    """Test 1 (quintiles, one-month holds) and Test 2 (insider signals split by attention), as pre-registered.
    Test 1 returns are gross of trading costs (the pre-registered rules name none); monthly turnover is
    reported so costs can be judged."""
    import statsmodels.api as sm
    from .common import DATA, RESULTS, load_config
    from .compustat import wrds_dir
    cfg = cfg or load_config()
    ab = abnormal_attention()
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik', 'loc'])
    co = co[co['loc'] == 'USA']
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    ab = ab.merge(co[['cik', 'gvkey']].dropna(), on='cik')
    panel = pd.read_parquet(CACHE / 'cs_monthly_panel.parquet', columns=['sec', 'month_end', 'price', 'adv20', 'gvkey'])
    rets = pd.read_parquet(CACHE / 'cs_monthly_returns.parquet', columns=['sec', 'month_end', 'ret_next'])
    panel = panel.merge(rets, on=['sec', 'month_end'])
    panel['gvkey'] = panel['gvkey'].astype(str)
    panel['month'] = pd.to_datetime(panel['month_end']).dt.to_period('M')
    panel = panel[(panel['price'] >= 1) & (panel['adv20'] >= 100_000)]
    panel = panel.sort_values('adv20').drop_duplicates(['gvkey', 'month'], keep='last')
    d = panel.merge(ab, on=['gvkey', 'month'])
    d = d.dropna(subset=['ret_next'])
    rows, prev = [], {}
    for mth, g in d.groupby('month'):
        if len(g) < 50:
            continue
        g = g.assign(q=pd.qcut(g['abnormal'].rank(method='first'), 5, labels=False) + 1)
        r = g.groupby('q')['ret_next'].mean()
        top = set(g.loc[g['q'] == 5, 'sec'])
        to = 1 - len(top & prev.get(5, set())) / max(len(top), 1)
        prev[5] = top
        rows.append({'month': mth, **{f'Q{int(k)}': v for k, v in r.items()}, 'n': len(g), 'Q5 turnover': to})
    Q = pd.DataFrame(rows).set_index('month')
    Q['Q5-Q1'] = Q['Q5'] - Q['Q1']
    F = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')
    Fm = ((1 + F).resample('ME').prod() - 1)
    Fm.index = Fm.index.to_period('M')
    # one-month holds formed at month t earn month t+1
    Q.index = Q.index + 1
    from .prices import load_price
    spy = load_price('SPY')['adj_close'].resample('ME').last().pct_change()
    spy.index = spy.index.to_period('M')
    out = []
    for lab, a, b in (('2016–2020', '2016-07', '2020-12'), ('2021+', '2021-01', None), ('all', '2016-07', None)):
        q = Q.loc[a:b]
        for col in ('Q1', 'Q3', 'Q5', 'Q5-Q1'):
            x = q[col].dropna()
            f = Fm.reindex(x.index).dropna()
            x = x.reindex(f.index)
            y = x - (f['RF'] if col != 'Q5-Q1' else 0)
            res = sm.OLS(y, sm.add_constant(f[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']])).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
            sp = spy.reindex(x.index)
            out.append({'test': 'T1 quintiles', 'period': lab, 'portfolio': col, 'months': len(x),
                        'avg monthly return': x.mean(), 'avg vs SPY': (x - sp).mean() if col != 'Q5-Q1' else np.nan,
                        'alpha/yr': res.params['const'] * 12, 'alpha t': res.tvalues['const'],
                        'Q5 monthly turnover': q['Q5 turnover'].mean()})
    # Test 2: insider signals split by attention in the month before the signal
    S = pd.read_parquet(DATA / 'signal_features.parquet',
                        columns=['issuer_cik', 'signal_date', 'screen', 'net_xs_spy_6m', 'ret_prior_6m'])
    S = S[S['screen'] == 'cluster3'].dropna(subset=['net_xs_spy_6m'])
    S['month'] = pd.to_datetime(S['signal_date']).dt.to_period('M') - 1
    S = S.merge(ab.rename(columns={'cik': 'issuer_cik'})[['issuer_cik', 'month', 'abnormal']], on=['issuer_cik', 'month'])
    for sub, mask in (('cluster buys', np.ones(len(S), bool)), ('big-drop clusters', (S['ret_prior_6m'] <= -0.5).values)):
        for side, m2 in (('attention above 0', S['abnormal'] > 0), ('attention at/below 0', S['abnormal'] <= 0)):
            x = S[mask & m2.values]
            if len(x) < 5:
                continue
            g = (x['net_xs_spy_6m'] - x['net_xs_spy_6m'].mean()).groupby(pd.to_datetime(x['signal_date']).dt.to_period('M')).sum()
            t = x['net_xs_spy_6m'].mean() / (np.sqrt((g ** 2).sum()) / len(x))
            out.append({'test': f'T2 {sub}', 'period': 'all', 'portfolio': side, 'months': len(x),
                        'avg monthly return': np.nan, 'avg vs SPY': x['net_xs_spy_6m'].mean(), 'alpha/yr': np.nan, 'alpha t': t})
    R = pd.DataFrame(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'wiki_attention.csv', index=False)
    Q.to_csv(RESULTS / 'wiki_attention_quintiles_monthly.csv')
    LOG.info('wiki: test done')
    return R
