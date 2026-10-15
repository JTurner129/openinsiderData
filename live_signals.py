"""Daily paper-trading scanner: new insider-buying signals from today's SEC Form 4 filings.

Run once a day after the SEC's evening index update (or the next morning):

    python live_signals.py              # scan the last 5 business days of filings
    python live_signals.py --days 30    # first run: build 30 days of history for the cluster window

It keeps every open-market purchase (code P) and sale (code S) it has seen in
live/form4_trades.parquet, applies the rules in config.yaml under `live:`, and
appends qualifying new signals to live/signals_log.csv with the date they were
first seen. That log is the paper-trading record: trades are taken at the next
open after a signal is first listed, and never edited afterwards.

Only SEC data is used, so this runs anywhere (no WRDS needed).
"""
import argparse
import re
import time
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / 'live'
STORE = LIVE / 'form4_trades.parquet'
LOG_CSV = LIVE / 'signals_log.csv'
TOP_EXEC = re.compile(r'\bCEO\b|chief executive|\bCFO\b|chief financial|chair(man|woman|person)?\b|\bCOB\b', re.I)
NON_DISCRETIONARY = re.compile(
    r'10b5-?1|dividend reinvest|\bdrip\b|employee stock purchase|\bespp\b|initial public offering|\bipo\b|'
    r'public offering|private placement|subscription agreement|rights offering|underwrit', re.I)


class Sec:
    def __init__(self, user_agent, per_second=5):
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': user_agent, 'Accept-Encoding': 'gzip, deflate'})
        self.gap = 1.0 / per_second
        self.t = 0.0

    def get(self, url):
        for i in range(5):
            time.sleep(max(0.0, self.t + self.gap - time.monotonic()))
            self.t = time.monotonic()
            try:
                r = self.s.get(url, timeout=60)
                if r.status_code == 404:
                    return None
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(str(r.status_code))
                r.raise_for_status()
                return r.text
            except requests.RequestException:
                time.sleep(5 * 2 ** i)
        return None


def form4_paths(sec, day):
    q = (day.month - 1) // 3 + 1
    idx = sec.get(f'https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx')
    if not idx:
        return []
    paths = {line.split()[-1] for line in idx.splitlines() if line.startswith('4 ') and line.split()[-1].endswith('.txt')}
    return sorted(paths)


def _t(node, path):
    x = node.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse(text, filed, path):
    m = re.search(r'<ownershipDocument>.*?</ownershipDocument>', text, re.S)
    if not m:
        return []
    doc = ET.fromstring(m.group(0))
    issuer_cik = _t(doc, 'issuer/issuerCik')
    ticker = (_t(doc, 'issuer/issuerTradingSymbol') or '').upper()
    name = _t(doc, 'issuer/issuerName')
    owner = doc.find('reportingOwner')
    owner_cik = _t(owner, 'reportingOwnerId/rptOwnerCik') if owner is not None else None
    owner_name = _t(owner, 'reportingOwnerId/rptOwnerName') if owner is not None else None
    rel = owner.find('reportingOwnerRelationship') if owner is not None else None
    title = ' '.join(filter(None, [_t(rel, 'officerTitle'), _t(rel, 'otherText')])) if rel is not None else ''
    is_dir = rel is not None and (_t(rel, 'isDirector') or '') in ('1', 'true')
    plan_box = (_t(doc, 'aff10b5One') or '') in ('1', 'true')
    notes = ' '.join(f.text or '' for f in doc.iter('footnote'))
    rows = []
    for tr in doc.iter('nonDerivativeTransaction'):
        code = _t(tr, 'transactionCoding/transactionCode')
        ad = _t(tr, 'transactionAmounts/transactionAcquiredDisposedCode/value')
        if code not in ('P', 'S'):
            continue
        sh = float(_t(tr, 'transactionAmounts/transactionShares/value') or 0)
        px = float(_t(tr, 'transactionAmounts/transactionPricePerShare/value') or 0)
        rows.append({'accession': path.rsplit('/', 1)[-1][:-4], 'filing_date': filed, 'issuer_cik': issuer_cik,
                     'ticker': ticker, 'company': name, 'owner_cik': owner_cik, 'owner': owner_name,
                     'title': title, 'is_director': is_dir, 'is_top_exec': bool(TOP_EXEC.search(title)),
                     'code': code, 'acq_disp': ad, 'trade_date': _t(tr, 'transactionDate/value'),
                     'shares': sh, 'price': px, 'value': sh * px,
                     'non_discretionary': plan_box or bool(NON_DISCRETIONARY.search(notes))})
    return rows


def scan(cfg, days):
    sec = Sec(cfg['sec']['user_agent'], cfg['sec'].get('max_requests_per_second', 5))
    store = pd.read_parquet(STORE) if STORE.exists() else pd.DataFrame()
    seen = set(store['accession']) if len(store) else set()
    new = []
    d = date.today()
    checked = 0
    while checked < days:
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        checked += 1
        paths = form4_paths(sec, d)
        todo = [p for p in paths if p.rsplit('/', 1)[-1][:-4] not in seen]
        print(f'{d}: {len(paths):,} Form 4s, {len(todo):,} new')
        for p in todo:
            text = sec.get(f'https://www.sec.gov/Archives/{p}')
            if text:
                try:
                    new += parse(text, pd.Timestamp(d), p)
                except ET.ParseError:
                    pass
    if new:
        store = pd.concat([store, pd.DataFrame(new)], ignore_index=True).drop_duplicates()
        LIVE.mkdir(exist_ok=True)
        store.to_parquet(STORE, index=False)
    return store


def signals(store, rules):
    p = store[(store['code'] == 'P') & (store['acq_disp'] == 'A') & (store['value'] > 0)].copy()
    if rules.get('discretionary_only', False):
        p = p[~p['non_discretionary']]
    p['filing_date'] = pd.to_datetime(p['filing_date'])
    p['insider'] = p['owner_cik'].fillna(p['owner'])
    s = store[(store['code'] == 'S') & (store['acq_disp'] == 'D')].copy()
    s['filing_date'] = pd.to_datetime(s['filing_date'])
    today = p['filing_date'].max()
    out = []
    win = rules['cluster_window_days']
    for cik, g in p.groupby('issuer_cik'):
        w = g[g['filing_date'] > today - pd.Timedelta(days=win)]
        if w['insider'].nunique() >= rules['cluster_min_insiders'] and w['value'].sum() >= rules['cluster_min_value']:
            out.append(('cluster', cik, w))
        ex = g[g['is_top_exec'] & (g['value'] >= rules['exec_min_value']) & (g['filing_date'] > today - pd.Timedelta(days=5))]
        if rules.get('include_exec', False) and len(ex):
            out.append(('exec', cik, ex))
    rows = []
    for kind, cik, w in out:
        sold = s[(s['issuer_cik'] == cik) & (s['filing_date'] > today - pd.Timedelta(days=90))]['value'].sum()
        if rules.get('exclude_insider_selling', False) and sold >= 100_000:
            continue
        rows.append({'first_seen': date.today().isoformat(), 'screen': kind, 'issuer_cik': cik,
                     'ticker': w['ticker'].iloc[-1], 'company': w['company'].iloc[-1],
                     'insiders': w['insider'].nunique(), 'total_value': round(w['value'].sum()),
                     'last_filing': w['filing_date'].max().date().isoformat(),
                     'insider_sales_90d': round(sold),
                     'names': '; '.join(pd.unique(w['owner'].astype(str)))[:200]})
    return pd.DataFrame(rows)


def add_prior_return(sig, lookback_days=182):
    """Prior ~6-month price change for each signal's ticker (Yahoo Finance via yfinance, if installed)."""
    try:
        import yfinance as yf
    except ImportError:
        print('yfinance not installed: pip install yfinance (needed for the crash filter)')
        sig['ret_prior_6m'] = np.nan
        return sig
    out = []
    for t in sig['ticker']:
        try:
            h = yf.Ticker(str(t).replace('.', '-')).history(period='1y', auto_adjust=True)['Close'].dropna()
            h = h[h.index >= h.index[-1] - pd.Timedelta(days=lookback_days)]
            out.append(h.iloc[-1] / h.iloc[0] - 1 if len(h) > 20 else np.nan)
        except Exception:
            out.append(np.nan)
    sig['ret_prior_6m'] = out
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=5)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    rules = cfg['live']
    store = scan(cfg, args.days)
    sig = signals(store, rules)
    if not sig.empty and rules.get('max_prior_6m_return') is not None:
        sig = add_prior_return(sig)
        miss = sig['ret_prior_6m'].isna()
        if miss.any():
            print('Could not price (check by hand): ' + ', '.join(sig.loc[miss, 'ticker'].astype(str)))
        sig = sig[sig['ret_prior_6m'] <= rules['max_prior_6m_return']]
    if sig.empty:
        print('No signals.')
        return
    log = pd.read_csv(LOG_CSV, dtype=str) if LOG_CSV.exists() else pd.DataFrame(columns=sig.columns)
    # A company re-enters the log only after `dedupe_days` since its last logged signal
    fresh = []
    for _, r in sig.iterrows():
        prev = log[(log['issuer_cik'].astype(str) == str(r['issuer_cik'])) & (log['screen'] == r['screen'])]
        if len(prev) and (pd.Timestamp(r['first_seen']) - pd.to_datetime(prev['first_seen']).max()).days <= rules['dedupe_days']:
            continue
        fresh.append(r)
    fresh = pd.DataFrame(fresh)
    if len(fresh):
        pd.concat([log, fresh.astype(str)], ignore_index=True).to_csv(LOG_CSV, index=False)
        print(f'{len(fresh)} NEW signal(s) — buy at the next open:')
        cols = ['screen', 'ticker', 'company', 'insiders', 'total_value'] + \
            (['ret_prior_6m'] if 'ret_prior_6m' in fresh else []) + ['insider_sales_90d']
        print(fresh[cols].to_string(index=False))
    else:
        print('No new signals (all already logged).')


if __name__ == '__main__':
    main()
