"""Phase 19: follow the forced sellers (and buyers). Niche events where someone must trade for non-economic
reasons, with and without insiders buying. Long-only.

Rules were written down and committed before any of this was run.

Costs: estimated half-spread each way + 0.15% FX each way. Tradeable at entry: price >= $1 and >= $100K average
daily dollar volume over the prior 20 days. Money not in positions is in SPY. Equal weight per entry.
Insider overlay ("+ insiders"): an officer or director open-market purchase totalling >= $50K filed in the
90 days before entry.

F1 IPO lockup expiry. Compustat IPO dates from 2004. Expiry = IPO + 180 calendar days. Enter 5 trading days
   after expiry (after the new supply has hit). Hold 6 months.
F2 Russell reconstitution (rebuilt from market caps). Each year, rank stocks in the monthly universe by market
   cap at the last trading day of May. "Fell into the Russell 2000 top": rank 1001-1300 this year but <= 1000 a
   year earlier (they become the largest Russell 2000 members, so index funds must buy). Enter on the first
   trading day of June; two holds: to the last trading day of June (reconstitution) and 12 months.
F3 Spin-offs. Companies that filed an original Form 10-12B, whose first Compustat trading day falls within a
   year after that filing. Enter 21 trading days after the first trading day (after the initial dumping). Hold
   12 months.
F4 Dividend initiations. First fiscal year with common dividends paid > 0 after at least 3 prior fiscal years
   with none. Enter when that annual report is public (fiscal year end + 90 days). Hold 12 months.
Reported: per-entry return minus SPY over the hold (month-clustered t), and the calendar-time portfolio's
CAGR, Sharpe and max drawdown against SPY and QQQ, all years / 2006-15 / 2016+.
"""
import time

import numpy as np
import pandas as pd
import requests

from .common import CACHE, DATA, RESULTS, get_logger, load_config, save_table
from .compustat import CS_CACHE, wrds_dir

LOG = get_logger('forced_flows')
FX = 0.0015


def next_day(idx, d, offset=0):
    i = int(np.searchsorted(idx.values, np.datetime64(d), 'left')) + offset
    return idx[i] if 0 <= i < len(idx) else None


def tradeable(sec, d):
    from .prices import load_price
    px = load_price(sec)
    if px is None:
        return False, np.nan
    w = px.loc[:d].tail(20)
    if len(w) < 10:
        return False, np.nan
    ok = (w['raw_close'].iloc[-1] >= 1) and (w['dollar_volume'].mean() >= 100_000 if 'dollar_volume' in w else False)
    return bool(ok), w


def lockups(cfg, idx, secmap):
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'ipodate'])
    co['gvkey'] = co['gvkey'].astype(str)
    co['ipodate'] = pd.to_datetime(co['ipodate'], errors='coerce')
    co = co[co['ipodate'] >= '2004-01-01']
    rows = []
    for _, c in co.iterrows():
        sec = secmap.get(c['gvkey'])
        if sec is None:
            continue
        exp = c['ipodate'] + pd.Timedelta(days=180)
        d = next_day(idx, exp, 5)
        if d is not None:
            rows.append({'price_ticker': sec, 'gvkey': c['gvkey'], 'entry_date': d, 'event': exp})
    return pd.DataFrame(rows)


def russell(cfg, idx):
    P = pd.read_parquet(DATA / 'deep_value_panel.parquet', columns=['sec', 'gvkey', 'month_end', 'mcap'])
    P['month_end'] = pd.to_datetime(P['month_end'])
    may = P[P['month_end'].dt.month == 5].copy()
    may['year'] = may['month_end'].dt.year
    may = may.sort_values('mcap', ascending=False).drop_duplicates(['year', 'gvkey'])
    may['rank'] = may.groupby('year')['mcap'].rank(ascending=False, method='first')
    prev = may[['gvkey', 'year', 'rank']].assign(year=may['year'] + 1).rename(columns={'rank': 'rank_prev'})
    m = may.merge(prev, on=['gvkey', 'year'], how='left')
    m = m[(m['rank'].between(1001, 1300)) & (m['rank_prev'] <= 1000)]
    rows = []
    for _, r in m.iterrows():
        d = next_day(idx, pd.Timestamp(f"{r['year']}-06-01"))
        june = idx[(idx.year == r['year']) & (idx.month == 6)]
        if d is not None and len(june):
            rows.append({'price_ticker': r['sec'], 'gvkey': str(r['gvkey']), 'entry_date': d, 'exit_june': june[-1]})
    return pd.DataFrame(rows)


def spinoffs(cfg, idx, secmap):
    path = CACHE / 'sec_form10.parquet'
    if path.exists():
        f10 = pd.read_parquet(path)
    else:
        s = requests.Session()
        s.headers['User-Agent'] = cfg['sec']['user_agent']
        rows = []
        for y in range(2004, pd.Timestamp.today().year + 1):
            for q in range(1, 5):
                r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
                if r.status_code != 200:
                    continue
                for line in r.text.splitlines():
                    if line[:17].strip() == '10-12B':
                        p = line.split()
                        rows.append((int(p[-3]), p[-2]))
                time.sleep(0.2)
        f10 = pd.DataFrame(rows, columns=['cik', 'filing_date'])
        f10['filing_date'] = pd.to_datetime(f10['filing_date'])
        f10 = f10.sort_values('filing_date').drop_duplicates('cik')
        f10.to_parquet(path, index=False)
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    f = f10.merge(co.dropna(), on='cik')
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    meta['first_date'] = pd.to_datetime(meta['first_date'])
    f = f.merge(meta[['gvkey', 'sec', 'first_date']], on='gvkey')
    f = f[(f['first_date'] >= f['filing_date']) & (f['first_date'] <= f['filing_date'] + pd.Timedelta(days=365))]
    rows = []
    for _, r in f.iterrows():
        d = next_day(idx, r['first_date'], 21)
        if d is not None:
            rows.append({'price_ticker': 'CS_' + str(r['sec']), 'gvkey': r['gvkey'], 'entry_date': d, 'event': r['first_date']})
    return pd.DataFrame(rows).drop_duplicates('gvkey')


def dividend_initiations(cfg, idx, secmap):
    a = pd.read_parquet(wrds_dir(cfg) / 'compustat_funda.parquet', columns=['gvkey', 'datadate', 'fyear', 'dvc'])
    a['gvkey'] = a['gvkey'].astype(str)
    a['dvc'] = pd.to_numeric(a['dvc'], errors='coerce')
    a['datadate'] = pd.to_datetime(a['datadate'])
    a = a.sort_values(['gvkey', 'datadate']).drop_duplicates(['gvkey', 'fyear'], keep='last')
    g = a.groupby('gvkey')
    prior_none = sum((g['dvc'].shift(k).fillna(-1) == 0).astype(int) for k in (1, 2, 3)) == 3
    init = a[(a['dvc'] > 0) & prior_none & (a['datadate'] >= '2005-01-01')]
    rows = []
    for _, r in init.iterrows():
        sec = secmap.get(r['gvkey'])
        d = next_day(idx, r['datadate'] + pd.Timedelta(days=90))
        if sec is not None and d is not None:
            rows.append({'price_ticker': sec, 'gvkey': r['gvkey'], 'entry_date': d, 'event': r['datadate']})
    return pd.DataFrame(rows)


def insider_flag(cfg, E):
    from .compustat import security_map
    sm = security_map(cfg)[['cik_i', 'gvkey']].dropna().drop_duplicates()
    sm['gvkey'] = sm['gvkey'].astype(str)
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['accession_number', 'issuer_cik', 'filing_date', 'value_clean'])
    it = pd.read_parquet(DATA / 'insider_trades.parquet', columns=['accession_number', 'is_director', 'is_officer'])
    pc = pc.merge(it, on='accession_number')
    pc = pc[pc['is_director'] | pc['is_officer']].merge(sm, left_on='issuer_cik', right_on='cik_i')
    pc['filing_date'] = pd.to_datetime(pc['filing_date'])
    by = {k: g.sort_values('filing_date') for k, g in pc.groupby('gvkey')}
    out = []
    for _, e in E.iterrows():
        g = by.get(e['gvkey'])
        if g is None:
            out.append(False)
            continue
        w = g[(g['filing_date'] < e['entry_date']) & (g['filing_date'] >= e['entry_date'] - pd.Timedelta(days=90))]
        out.append(bool(w['value_clean'].sum() >= 50_000))
    return np.array(out)


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market, corwin_schultz
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    cal = spy_px.index.values
    idx = pd.DatetimeIndex(cal)
    spy = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    qqq = load_price('QQQ')['adj_close'].pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    market = Market(cal)
    meta = pd.read_parquet(CS_CACHE / '_securities.parquet')
    meta['gvkey'] = meta['gvkey'].astype(str)
    secmap = meta.sort_values('median_dollar_volume', ascending=False).drop_duplicates('gvkey').set_index('gvkey')['sec']
    secmap = {k: 'CS_' + str(v) for k, v in secmap.items()}
    events = {'F1 IPO lockup expiry': (lockups(cfg, idx, secmap), (('6m', 126),)),
              'F2 Fell into Russell 2000 top': (russell(cfg, idx), (('June', 'june'), ('1y', 252))),
              'F3 Spin-offs': (spinoffs(cfg, idx, secmap), (('1y', 252),)),
              'F4 Dividend initiations': (dividend_initiations(cfg, idx, secmap), (('1y', 252),))}
    rows, curves, saved = [], {}, []
    for name, (E, holds) in events.items():
        if E.empty:
            LOG.warning(f'forced: {name}: no events')
            continue
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
        E['insiders'] = insider_flag(cfg, E)
        LOG.info(f'forced: {name}: {len(E):,} tradeable events, {int(E["insiders"].sum())} with insiders buying')
        saved.append(E.assign(event_type=name))
        for hn, hd in holds:
            if hd == 'june':
                hd_days = int(np.median([max(1, np.searchsorted(cal, np.datetime64(x)) - np.searchsorted(cal, np.datetime64(d)))
                                         for d, x in zip(E['entry_date'], E['exit_june'])]))
            else:
                hd_days = hd
            eng = SubsetEngine(E, market, hd_days, E['cost'].values)
            for variant, mask in (('all', np.ones(len(E))), ('+ insiders', E['insiders'].values.astype(float))):
                if mask.sum() < 20:
                    continue
                p = eng.returns(mask, rf.values)
                if p is None:
                    continue
                act = pd.Series(eng.active(mask), index=idx)
                p = p.reindex(idx).where(act, spy).loc[act[act].index[0]:]
                curves[f'{name}|{hn}|{variant}'] = p
                # per-entry excess over SPY
                ex, mon = [], []
                for (_, e), mm in zip(E.iterrows(), mask):
                    if not mm:
                        continue
                    a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:hd_days + 1]
                    b = spy_px['adj_close'].loc[e['entry_date']:].iloc[:hd_days + 1]
                    if len(a) > 1 and len(b) > hd_days * 0.9:
                        ex.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost']) - (b.iloc[-1] / b.iloc[0] - 1))
                        mon.append(str(pd.Timestamp(e['entry_date']).to_period('M')))
                ex = pd.Series(ex)
                if len(ex) > 10:
                    g = (ex - ex.mean()).groupby(pd.Series(mon)).sum()
                    t_ex = ex.mean() / (np.sqrt((g ** 2).sum()) / len(ex))
                else:
                    t_ex = np.nan
                for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
                    s_ = perf(p.loc[sl], rf)
                    b_ = perf(spy.loc[p.index[0]:].loc[sl], rf)
                    q_ = perf(qqq.loc[p.index[0]:].loc[sl], rf)
                    rows.append({'event': name, 'hold': hn, 'variant': variant, 'period': lab, 'entries': int(mask.sum()),
                                 'avg excess vs SPY per trade': ex.mean() if lab == 'all' else np.nan,
                                 't': t_ex if lab == 'all' else np.nan,
                                 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'), 'QQQ CAGR': q_.get('CAGR'),
                                 'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'),
                                 'max DD': s_.get('max drawdown'), 'SPY max DD': b_.get('max drawdown')})
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'forced_flows.csv', index=False)
    save_table(pd.concat(saved).drop(columns=['exit_june'], errors='ignore'), DATA / 'forced_flow_events')
    pd.DataFrame(curves).to_parquet(DATA / 'forced_flows_daily.parquet')
    LOG.info('forced: done')
    return r


if __name__ == '__main__':
    run()
