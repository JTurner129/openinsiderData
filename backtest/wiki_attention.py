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
