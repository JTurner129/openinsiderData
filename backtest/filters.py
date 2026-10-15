"""Phase 7: realistic-trading robustness and filter tests.

Part 1 asks whether the edge survives realistic trading in cheap, illiquid
stocks: per-stock bid-ask spreads (Corwin-Schultz) instead of a flat 0.5%,
buy-and-hold weights instead of daily rebalancing, entry at the next close
instead of the open, a micro-cap factor (IWC) in the regression, and
liquidity floors.

Part 2 tests every candidate filter the same way, choosing on 2006-2015 and
checking on 2016-2026, with Holm correction for the number of tests.
"""
import json
import time

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from .analysis import load_factors
from .common import CACHE, DATA, RESULTS, SCREENS, get_logger, load_config, save_table
from .prices import download_prices, load_price
from .signals import is_top_exec

LOG = get_logger('filters')
TRAIN, TEST = ('2006-01-01', '2015-12-31'), ('2016-01-01', '2100-01-01')
RECENT = ('2020-01-01', '2100-01-01')
HOLDS = {'1m': 21, '3m': 63, '1y': 252}
BASE_SCREENS = ['cluster3', 'exec200k']
FIXED_COST = 0.002          # commission + market impact on top of the spread, round trip
SQRT2 = np.sqrt(2.0)


# --- Market data ---------------------------------------------------------------------
def corwin_schultz(high, low, close):
    """Daily Corwin-Schultz (2012) bid-ask spread estimates from high/low prices,
    with their overnight-return adjustment; negative estimates set to 0."""
    h, l, c = high.copy(), low.copy(), close
    h1, l1 = np.roll(h, -1), np.roll(l, -1)
    # overnight adjustment: shift day t+1's range if it gapped past day t's close
    up = np.where(c > h1, c - h1, 0.0)
    down = np.where(c < l1, c - l1, 0.0)
    h1, l1 = h1 + up + down, l1 + up + down
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.log(h / l) ** 2 + np.log(h1 / l1) ** 2
        gamma = np.log(np.maximum(h, h1) / np.minimum(l, l1)) ** 2
        k = 3 - 2 * SQRT2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s[-1] = np.nan
    return np.clip(s, 0, None)


class Market:
    """Per-ticker arrays aligned to the SPY trading calendar."""

    def __init__(self, cal):
        self.cal = cal
        self.cache = {}

    def get(self, ticker):
        if ticker in self.cache:
            return self.cache[ticker]
        px = load_price(ticker) if isinstance(ticker, str) else None
        if px is None or px.empty:
            self.cache[ticker] = None
            return None
        px = px.dropna(subset=['adj_close'])
        cal = self.cal
        close = px['adj_close'].reindex(cal)
        has = close.notna().values
        prev = px['adj_close'].shift(1).reindex(cal)
        ret = (close / prev - 1).fillna(0.0).values
        opn = px['adj_open'].where(px['adj_open'] > 0, px['adj_close'])
        intraday = (px['adj_close'] / opn - 1).reindex(cal).fillna(0.0).values
        spread = np.full(len(cal), np.nan)
        if 'adj_high' in px:
            s = pd.Series(corwin_schultz(px['adj_high'].values, px['adj_low'].values, px['adj_close'].values),
                          index=px.index)
            spread = s.reindex(cal).values
        dv = px['dollar_volume'].reindex(cal).values if 'dollar_volume' in px else np.full(len(cal), np.nan)
        raw = px['raw_close'].reindex(cal).values
        last = int(np.flatnonzero(has)[-1]) if has.any() else -1
        out = dict(ret=ret, intraday=intraday, close=close.values, has=has, spread=spread, dv=dv, raw=raw, last=last)
        self.cache[ticker] = out
        return out


# --- SEC shares outstanding (market cap, 2011+) ----------------------------------------
def fetch_shares_frames(cfg):
    path = CACHE / 'sec_shares_outstanding.parquet'
    if path.exists():
        return pd.read_parquet(path)
    headers = {'User-Agent': cfg['sec']['user_agent'], 'Accept-Encoding': 'gzip, deflate'}
    rows = []
    for year in range(2009, pd.Timestamp.today().year + 1):
        for q in range(1, 5):
            url = f'https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{year}Q{q}I.json'
            for attempt in range(3):
                time.sleep(0.25 * (2 ** attempt))
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code in (200, 404):
                    break
            if r.status_code != 200:
                continue
            for d in r.json().get('data', []):
                rows.append((int(d['cik']), d['end'], float(d['val'])))
    df = pd.DataFrame(rows, columns=['cik', 'end', 'shares'])
    df['end'] = pd.to_datetime(df['end'])
    df = df[df['shares'] > 0].drop_duplicates(['cik', 'end']).sort_values(['cik', 'end'])
    df.to_parquet(path, index=False)
    LOG.info(f"SEC shares outstanding: {len(df):,} company-quarters for {df['cik'].nunique():,} companies")
    return df


def fetch_book_frames(cfg):
    """Book value (total stockholders' equity) per company-quarter from SEC XBRL frames."""
    path = CACHE / 'sec_book_equity.parquet'
    if path.exists():
        return pd.read_parquet(path)
    headers = {'User-Agent': cfg['sec']['user_agent'], 'Accept-Encoding': 'gzip, deflate'}
    rows = []
    tags = ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']
    for pri, tag in enumerate(tags):
        for year in range(2009, pd.Timestamp.today().year + 1):
            for q in range(1, 5):
                url = f'https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/CY{year}Q{q}I.json'
                for attempt in range(3):
                    time.sleep(0.25 * (2 ** attempt))
                    r = requests.get(url, headers=headers, timeout=60)
                    if r.status_code in (200, 404):
                        break
                if r.status_code != 200:
                    continue
                for d in r.json().get('data', []):
                    rows.append((int(d['cik']), d['end'], float(d['val']), pri))
    df = pd.DataFrame(rows, columns=['cik', 'end', 'book_equity', 'priority'])
    df['end'] = pd.to_datetime(df['end'])
    # Prefer equity attributable to the parent; fall back to total including minority interests.
    df = df.sort_values(['cik', 'end', 'priority']).drop_duplicates(['cik', 'end']).drop(columns='priority')
    df.to_parquet(path, index=False)
    LOG.info(f"SEC book equity: {len(df):,} company-quarters for {df['cik'].nunique():,} companies")
    return df


BOOK_LAG_DAYS = 95   # a quarter's balance sheet is assumed public 95 days after quarter-end


# --- Insider-history features ----------------------------------------------------------
def insider_history_flags(purchases):
    """Per filing: routine (same calendar month purchase in each of the prior
    3 years, after Cohen, Malloy & Pomorski 2012) and first purchase by that
    insider in 2+ years."""
    p = purchases[['accession_number', 'insider_id', 'trade_date', 'filing_date']].copy()
    p['td'] = p['trade_date'].fillna(p['filing_date'])
    p['ym'] = list(zip(p['insider_id'], p['td'].dt.year, p['td'].dt.month))
    seen = set(p['ym'])
    p['routine'] = [all((i, y - k, m) in seen for k in (1, 2, 3)) for i, y, m in p['ym']]
    p = p.sort_values(['insider_id', 'td'])
    gap = p.groupby('insider_id')['td'].diff().dt.days
    p['first_in_2y'] = (gap.isna() | (gap > 730)).astype(float)
    # The data starts in 2006, so "first in 2 years" is unknowable before 2008.
    p.loc[p['td'] < '2008-01-01', 'first_in_2y'] = np.nan
    return p.set_index('accession_number')[['routine', 'first_in_2y']]


SECTORS = [('biotech/pharma', [(2833, 2836), (8731, 8731)]), ('banks', [(6000, 6199)]),
           ('tech', [(3570, 3579), (3660, 3679), (7370, 7379)]), ('energy/mining', [(1000, 1499), (2900, 2999)])]


def sector_of(sic):
    if sic is None or pd.isna(sic):
        return 'unknown'
    for name, ranges in SECTORS:
        if any(a <= sic <= b for a, b in ranges):
            return name
    return 'other'


def build_features(trades, cfg, market):
    """One row per signal with everything the filters need."""
    sig = pd.read_parquet(DATA / 'signals_all_variants.parquet')
    key = ['screen', 'dedupe_days', 'signal_date', 'issuer_cik']
    trades = trades.merge(sig[key + ['accessions']], on=key, how='left')
    purchases = pd.read_parquet(DATA / 'insider_trades.parquet')
    purchases = purchases[purchases['transaction_type'] == 'P'].copy()
    purchases['insider_id'] = purchases['owner_cik'].astype('string').fillna('name:' + purchases['owner_name'].astype(str))
    hist = insider_history_flags(purchases)
    pur = purchases.set_index('accession_number').join(hist)
    pur['top_exec'] = pur['Title'].map(is_top_exec)
    info = json.loads((CACHE / 'sec_company_info.json').read_text())
    use_cs = cfg['backtest'].get('price_source', 'yahoo') == 'compustat'
    if use_cs:
        # Compustat: daily shares outstanding (all years) and quarterly book equity by report date
        from .compustat import fundamentals
        cs_book = {g: x for g, x in fundamentals(cfg).groupby('gvkey')}
        shares_by_cik, book_by_cik = {}, {}
    else:
        shares = fetch_shares_frames(cfg)
        shares_by_cik = {c: g for c, g in shares.groupby('cik')}
        book = fetch_book_frames(cfg)
        book_by_cik = {c: g for c, g in book.groupby('cik')}
    cal = market.cal

    feats = []
    for i, t in trades.iterrows():
        f = {'idx': i}
        acc = [a for a in str(t['accessions'] or '').split(',') if a in pur.index]
        w = pur.loc[acc] if acc else pur.iloc[0:0]
        f['max_own_increase'] = w['Owned'].max() if len(w) else np.nan
        f['median_own_increase'] = w['Owned'].median() if len(w) else np.nan
        f['new_position'] = bool(len(w) and w['Owned'].isna().any())
        f['has_top_exec'] = bool(w['top_exec'].any()) if len(w) else False
        f['has_10pct_owner'] = bool(w['is_ten_pct_owner'].any()) if len(w) else False
        f['directors_only'] = bool(len(w) and w['is_director'].all() and not w['is_officer'].any())
        f['opportunistic'] = bool(len(w) and not w['routine'].any())
        fy = w['first_in_2y'].dropna() if len(w) else pd.Series(dtype=float)
        f['all_first_in_2y'] = bool(fy.astype(bool).all()) if len(fy) == len(w) and len(w) else np.nan
        rec = info.get(str(int(t['issuer_cik'])), {})
        f['sector'] = sector_of(rec.get('sic'))
        s = market.get(t['price_ticker']) if t['price_found'] else None
        if s is not None and pd.notna(t['entry_date']):
            m = int(np.searchsorted(cal, np.datetime64(t['entry_date']), 'left'))
            lo = max(0, m - 20)
            f['adv20'] = np.nanmean(s['dv'][lo:m]) if m > lo else np.nan
            f['spread'] = np.nanmedian(s['spread'][lo:m]) if m > lo else np.nan
            c = s['close']
            prev = c[m - 1] if m >= 1 else np.nan
            for name, n in (('ret_prior_1m', 21), ('ret_prior_3m', 63), ('ret_prior_6m', 126)):
                f[name] = prev / c[m - 1 - n] - 1 if m - 1 - n >= 0 and c[m - 1 - n] > 0 else np.nan
            yr = c[max(0, m - 252):m]
            yr = yr[np.isfinite(yr)]
            f['pct_above_52w_low'] = prev / yr.min() - 1 if len(yr) > 100 else np.nan
            f['pct_below_52w_high'] = 1 - prev / yr.max() if len(yr) > 100 else np.nan
            if use_cs and str(t['price_ticker']).startswith('CS_'):
                gvkey = str(t['price_ticker']).split('_')[1]
                px = load_price(t['price_ticker'])
                so = px['shares_out'].loc[:t['entry_date']].dropna() if px is not None and 'shares_out' in px else None
                if so is not None and len(so):
                    f['market_cap'] = so.iloc[-1] * t['entry_price_raw']
                b = cs_book.get(gvkey)
                if b is not None:
                    b = b[b['public'] <= t['signal_date']]
                    if len(b) and (t['signal_date'] - b['public'].iloc[-1]).days <= 500:
                        f['book_equity'] = b['book_equity'].iloc[-1]
            g = shares_by_cik.get(int(t['issuer_cik']))
            if g is not None:
                g = g[g['end'] <= t['signal_date']]
                if len(g) and (t['signal_date'] - g['end'].iloc[-1]).days <= 400:
                    f['market_cap'] = g['shares'].iloc[-1] * t['entry_price_raw']
            b = book_by_cik.get(int(t['issuer_cik']))
            if b is not None:
                b = b[b['end'] + pd.Timedelta(days=BOOK_LAG_DAYS) <= t['signal_date']]
                if len(b) and (t['signal_date'] - b['end'].iloc[-1]).days <= 500:
                    f['book_equity'] = b['book_equity'].iloc[-1]
        feats.append(f)
    F = pd.DataFrame(feats).set_index('idx')
    out = trades.join(F)
    out['value_to_mcap'] = out['total_value'] / out['market_cap']
    # Price-to-book; undefined (NaN) when book value is zero or negative, which is flagged separately.
    out['price_to_book'] = (out['market_cap'] / out['book_equity']).where(out['book_equity'] > 0)
    out['negative_book'] = out['book_equity'] <= 0
    # Round-trip trading cost: estimated full spread (pay half each way) plus commission/impact.
    # Floor the spread at one tick (1 cent, or 0.01% for sub-dollar stocks); fall back to the
    # screen's median spread when no estimate is available.
    tick = np.where(out['entry_price_raw'] >= 1, 0.01 / out['entry_price_raw'], 0.0001)
    sp = out['spread'].fillna(out.groupby('screen')['spread'].transform('median'))
    out['spread_est'] = np.fmax(sp, tick)
    out['cost_rt'] = out['spread_est'] + FIXED_COST
    return out


# --- Portfolio engine -----------------------------------------------------------------------
def portfolio(trades, market, hold, entry='open', weights='rebalanced', cost=None, rf=None):
    """Daily calendar-time portfolio returns.

    entry: 'open' = buy at the open of the entry day; 'close' = buy at that day's close.
    weights: 'rebalanced' = equal weight every day; 'buyhold' = equal dollars at entry, then drift.
    cost: per-trade round-trip cost (array aligned with trades) or a scalar; half is charged on
    the entry day and half on the exit day.
    """
    cal = market.cal
    n = len(cal)
    num, den, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    costs = np.broadcast_to(np.asarray(0.0 if cost is None else cost, dtype=float), (len(trades),))
    for (_, t), c in zip(trades.iterrows(), costs):
        s = market.get(t['price_ticker'])
        if s is None or pd.isna(t['entry_date']):
            continue
        m = int(np.searchsorted(cal, np.datetime64(t['entry_date']), 'left'))
        start = m if entry == 'open' else m + 1
        end = min(m + hold, s['last'], n - 1)
        if end < start:
            continue
        r = s['ret'][start:end + 1].copy()
        if entry == 'open':
            r[0] = s['intraday'][m]
        r[0] = (1 + r[0]) * (1 - c / 2) - 1
        r[-1] = (1 + r[-1]) * (1 - c / 2) - 1
        if weights == 'buyhold':
            v = np.concatenate([[1.0], np.cumprod(1 + r)[:-1]])   # value at start of each day
        else:
            v = np.ones(len(r))
        num[start:end + 1] += v * r
        den[start:end + 1] += v
        cnt[start:end + 1] += 1
    idx = pd.DatetimeIndex(cal)
    port = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=idx)
    first = idx[cnt > 0].min() if (cnt > 0).any() else None
    if first is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    port = port.loc[first:]
    if rf is not None:
        port = port.fillna(rf.reindex(port.index).ffill().fillna(0.0))
    return port.fillna(0.0), pd.Series(cnt, index=idx).loc[first:]


def regress(port, factors, model, start=None, end=None):
    cols = {'CAPM': ['Mkt-RF'], 'FF5+UMD': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'],
            'FF5+UMD+IWC': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'IWC-RF']}[model]
    d = pd.concat([port.rename('r'), factors], axis=1, join='inner').dropna(subset=['r'] + cols + ['RF'])
    if start:
        d = d.loc[start:end]
    if len(d) < 120:
        return {'alpha': np.nan, 't': np.nan, 'p': np.nan, 'n_days': len(d)}
    res = sm.OLS(d['r'] - d['RF'], sm.add_constant(d[cols])).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return {'alpha': res.params['const'] * 252, 't': res.tvalues['const'], 'p': res.pvalues['const'],
            'se': res.bse['const'] * 252, 'n_days': len(d)}


def trade_returns(trades, market, hold, entry, cost):
    """Per-trade buy-and-hold return over `hold` trading days, net of cost (elapsed horizons only)."""
    out = np.full(len(trades), np.nan)
    costs = np.broadcast_to(np.asarray(cost, dtype=float), (len(trades),))
    for k, ((_, t), c) in enumerate(zip(trades.iterrows(), costs)):
        s = market.get(t['price_ticker'])
        if s is None or pd.isna(t['entry_date']):
            continue
        m = int(np.searchsorted(market.cal, np.datetime64(t['entry_date']), 'left'))
        if m + hold >= len(market.cal):
            continue
        start = m if entry == 'open' else m + 1
        end = min(m + hold, s['last'])
        r = s['ret'][start:end + 1].copy()
        if entry == 'open' and len(r):
            r[0] = s['intraday'][m]
        out[k] = np.prod(1 + r) * (1 - c) - 1
    return out


def bench_returns(trades, market, hold, entry, ticker):
    b = market.get(ticker)
    out = np.full(len(trades), np.nan)
    for k, (_, t) in enumerate(trades.iterrows()):
        if pd.isna(t['entry_date']):
            continue
        m = int(np.searchsorted(market.cal, np.datetime64(t['entry_date']), 'left'))
        if m + hold >= len(market.cal):
            continue
        start = m if entry == 'open' else m + 1
        r = b['ret'][start:m + hold + 1].copy()
        if entry == 'open':
            r[0] = b['intraday'][m]
        out[k] = np.prod(1 + r) - 1
    return out


# --- Part 1: realistic trading -----------------------------------------------------------
REALISM_STEPS = [
    # label, entry, weights, cost column/scale, extra filter
    ('1. As reported (open, daily rebalance, 0.5% cost)', 'open', 'rebalanced', 'flat', None),
    ('2. + estimated spread costs', 'open', 'rebalanced', 'spread', None),
    ('3. + buy-and-hold weights', 'open', 'buyhold', 'spread', None),
    ('4. + enter at next close', 'close', 'buyhold', 'spread', None),
    ('5. Stress: 2× spread costs', 'close', 'buyhold', 'spread2', None),
    ('6. Realistic + price ≥ $2', 'close', 'buyhold', 'spread', ('entry_price_raw', 2)),
    ('7. Realistic + $100K/day volume', 'close', 'buyhold', 'spread', ('adv20', 1e5)),
    ('8. Realistic + $1M/day volume', 'close', 'buyhold', 'spread', ('adv20', 1e6)),
]


def cost_vector(t, kind, flat):
    return {'flat': np.full(len(t), flat), 'spread': t['cost_rt'].values,
            'spread2': (2 * t['spread_est'] + FIXED_COST).values}[kind]


def realism_table(F, market, factors, cfg):
    rows, curves = [], {}
    for screen in SCREENS:
        base = F[(F['screen'] == screen) & F['price_found']]
        for label, entry, weights, ck, filt in REALISM_STEPS:
            t = base if filt is None else base[base[filt[0]] >= filt[1]]
            for hname, hd in HOLDS.items():
                port, n = portfolio(t, market, hd, entry, weights, cost_vector(t, ck, cfg['backtest']['round_trip_cost']),
                                    factors['RF'])
                r = {'screen': screen, 'step': label, 'hold': hname, 'signals': len(t),
                     'avg_positions': n[n > 0].mean() if len(n) else np.nan,
                     'median_spread': t['spread_est'].median(), 'median_adv': t['adv20'].median()}
                for model in ('FF5+UMD', 'FF5+UMD+IWC'):
                    a = regress(port, factors, model)
                    r[f'{model} alpha'], r[f'{model} t'] = a['alpha'], a['t']
                for lab, (a0, a1) in (('test', TEST), ('2020+', RECENT)):
                    a = regress(port, factors, 'FF5+UMD+IWC', a0, a1)
                    r[f'{lab} alpha'], r[f'{lab} t'] = a['alpha'], a['t']
                rows.append(r)
                if hname == '3m':
                    curves[(screen, label)] = port
        LOG.info(f"realism: {screen} done")
    return pd.DataFrame(rows), curves


# --- Part 2: filters ----------------------------------------------------------------------------
def filter_definitions(F):
    q = lambda c, op, v: (lambda d: op(d[c], v))
    ge, lt, le = (lambda a, b: a >= b), (lambda a, b: a < b), (lambda a, b: a <= b)
    defs = [
        ('Liquidity', 'Price ≥ $1', q('entry_price_raw', ge, 1)),
        ('Liquidity', 'Price ≥ $2', q('entry_price_raw', ge, 2)),
        ('Liquidity', 'Price ≥ $5', q('entry_price_raw', ge, 5)),
        ('Liquidity', 'Price ≥ $10', q('entry_price_raw', ge, 10)),
        ('Liquidity', 'Volume ≥ $100K/day', q('adv20', ge, 1e5)),
        ('Liquidity', 'Volume ≥ $1M/day', q('adv20', ge, 1e6)),
        ('Liquidity', 'Volume ≥ $10M/day', q('adv20', ge, 1e7)),
        ('Liquidity', 'Spread ≤ 1%', q('spread_est', le, 0.01)),
        ('Liquidity', 'Spread ≤ 2%', q('spread_est', le, 0.02)),
        ('Size (2011+)', 'Market cap < $50M', q('market_cap', lt, 5e7)),
        ('Size (2011+)', 'Market cap $50M–$300M', lambda d: d['market_cap'].between(5e7, 3e8, inclusive='left')),
        ('Size (2011+)', 'Market cap < $300M', q('market_cap', lt, 3e8)),
        ('Size (2011+)', 'Market cap ≥ $50M', q('market_cap', ge, 5e7)),
        ('Size (2011+)', 'Market cap $300M–$2B', lambda d: d['market_cap'].between(3e8, 2e9)),
        ('Size (2011+)', 'Market cap ≥ $2B', q('market_cap', ge, 2e9)),
        ('Size (2011+)', 'Bought ≥ 0.1% of market cap', q('value_to_mcap', ge, 0.001)),
        ('Size (2011+)', 'Bought ≥ 1% of market cap', q('value_to_mcap', ge, 0.01)),
        ('Value (2011+)', 'Price-to-book ≤ 1', q('price_to_book', le, 1.0)),
        ('Value (2011+)', 'Price-to-book ≤ 1.5', q('price_to_book', le, 1.5)),
        ('Value (2011+)', 'Price-to-book > 3', lambda d: d['price_to_book'] > 3),
        ('Value (2011+)', 'Negative book value', lambda d: d['negative_book'] == True),  # noqa: E712
        ('Conviction', 'Total bought ≥ $500K', q('total_value', ge, 5e5)),
        ('Conviction', 'Total bought ≥ $1M', q('total_value', ge, 1e6)),
        ('Conviction', 'Ownership up ≥ 10%', q('max_own_increase', ge, 0.10)),
        ('Conviction', 'Ownership up ≥ 25%', q('max_own_increase', ge, 0.25)),
        ('Conviction', 'Ownership up ≥ 50%', q('max_own_increase', ge, 0.50)),
        ('Conviction', 'Ownership up ≥ 100% (doubled)', q('max_own_increase', ge, 1.00)),
        ('Conviction', 'Ownership up < 10% (small top-ups)', q('max_own_increase', lt, 0.10)),
        ('Conviction', 'Median insider ownership up ≥ 10%', q('median_own_increase', ge, 0.10)),
        ('Conviction', 'Median insider ownership up ≥ 25%', q('median_own_increase', ge, 0.25)),
        ('Conviction', 'Includes a new position', lambda d: d['new_position']),
        ('Who', 'Cluster includes CEO/CFO/COB', lambda d: d['has_top_exec']),
        ('Who', 'No 10% owners', lambda d: ~d['has_10pct_owner'].astype(bool)),
        ('Who', 'Directors only', lambda d: d['directors_only']),
        ('Who', 'Opportunistic (no routine buyer)', lambda d: d['opportunistic']),
        ('Who', 'All first buy in 2+ years (2008+)', lambda d: d['all_first_in_2y'] == True),  # noqa: E712
        ('Timing', 'Stock down ≥ 10% in prior month', q('ret_prior_1m', le, -0.10)),
        ('Timing', 'Stock down ≥ 20% in prior 3 months', q('ret_prior_3m', le, -0.20)),
        ('Timing', 'Stock down ≥ 30% in prior 3 months', q('ret_prior_3m', le, -0.30)),
        ('Timing', 'Stock down ≥ 50% in prior 6 months', q('ret_prior_6m', le, -0.50)),
        ('Timing', 'Stock up in prior 6 months', q('ret_prior_6m', ge, 0)),
        ('Timing', 'Within 10% of 52-week low', q('pct_above_52w_low', le, 0.10)),
        ('Timing', '≥ 30% below 52-week high', q('pct_below_52w_high', ge, 0.30)),
        ('Timing', '≥ 50% below 52-week high', q('pct_below_52w_high', ge, 0.50)),
        ('Timing', 'Filed within 2 days of trade', q('max_filing_lag_days', le, 2)),
        ('Sector', 'Exclude biotech/pharma', lambda d: d['sector'] != 'biotech/pharma'),
        ('Sector', 'Biotech/pharma only', lambda d: d['sector'] == 'biotech/pharma'),
        ('Sector', 'Exclude banks', lambda d: d['sector'] != 'banks'),
        ('Sector', 'Banks only', lambda d: d['sector'] == 'banks'),
        ('Sector', 'Tech only', lambda d: d['sector'] == 'tech'),
        ('Sector', 'Energy/mining only', lambda d: d['sector'] == 'energy/mining'),
    ]
    return defs


def evaluate(t, market, factors, hold, entry='close', weights='buyhold'):
    port, n = portfolio(t, market, hold, entry, weights, t['cost_rt'].values, factors['RF'])
    out = {'signals': len(t), 'train_signals': int((t['signal_date'] <= TRAIN[1]).sum()),
           'test_signals': int((t['signal_date'] >= TEST[0]).sum()),
           'avg_positions': n[n > 0].mean() if len(n) else np.nan}
    for lab, (a0, a1) in (('full', (None, None)), ('train', TRAIN), ('test', TEST), ('2020+', RECENT)):
        a = regress(port, factors, 'FF5+UMD+IWC', a0, a1)
        out[f'{lab} alpha'], out[f'{lab} t'], out[f'{lab} p'] = a['alpha'], a['t'], a['p']
        if lab == 'full':
            out['full se'] = a.get('se')
    return out


def filter_tests(F, market, factors):
    rows = []
    for screen in BASE_SCREENS:
        base = F[(F['screen'] == screen) & F['price_found']]
        for hname, hd in HOLDS.items():
            b = evaluate(base, market, factors, hd)
            rows.append({'screen': screen, 'hold': hname, 'group': 'Baseline', 'filter': 'No filter (realistic)', **b})
            for group, name, fn in filter_definitions(F):
                mask = fn(base).fillna(False).astype(bool)
                t = base[mask]
                if len(t) < 30:
                    continue
                rows.append({'screen': screen, 'hold': hname, 'group': group, 'filter': name, **evaluate(t, market, factors, hd)})
            LOG.info(f"filters: {screen} {hname} done")
    d = pd.DataFrame(rows)
    # Improvement over the unfiltered baseline, in each period
    base = d[d['filter'] == 'No filter (realistic)'][['screen', 'hold'] + [f'{l} alpha' for l in ('full', 'train', 'test', '2020+')]]
    d = d.merge(base, on=['screen', 'hold'], suffixes=('', ' baseline'))
    for lab in ('full', 'train', 'test', '2020+'):
        d[f'{lab} vs baseline'] = d[f'{lab} alpha'] - d[f'{lab} alpha baseline']
    d = d.drop(columns=[f'{l} alpha baseline' for l in ('full', 'train', 'test', '2020+')])
    # Holm correction across every filter test (excluding the baselines)
    tests = d['filter'] != 'No filter (realistic)'
    p = d.loc[tests, 'full p'].fillna(1.0)
    order = np.argsort(p.values)
    m = len(p)
    adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p.values[order]))
    holm = np.empty(m)
    holm[order] = adj
    d.loc[tests, 'holm p'] = holm
    d['significant after Holm'] = d['holm p'] < 0.05
    d['holds out of sample'] = (d['train vs baseline'] > 0) & (d['test vs baseline'] > 0) & (d['test t'] >= 1.96)
    return d


def combined_filter(F, market, factors, tests):
    """Pick the best filters on the TRAIN period only, combine them, and report the TEST period."""
    rows = []
    defs = {name: fn for _, name, fn in filter_definitions(F)}
    for screen in BASE_SCREENS:
        base = F[(F['screen'] == screen) & F['price_found']]
        for hname, hd in HOLDS.items():
            cand = tests[(tests['screen'] == screen) & (tests['hold'] == hname) & (tests['filter'] != 'No filter (realistic)')
                         & (tests['train vs baseline'] > 0) & (tests['train t'] >= 2) & (tests['train_signals'] >= 100)]
            cand = cand[~cand['group'].isin(['Size (2011+)', 'Value (2011+)'])]   # no SEC fundamentals before 2011
            chosen, t = [], base
            for _, c in cand.sort_values('train alpha', ascending=False).iterrows():
                if len(chosen) == 3:
                    break
                trial = t[defs[c['filter']](t).fillna(False).astype(bool)]
                if (trial['signal_date'] <= TRAIN[1]).sum() < 100:
                    continue
                before = evaluate(t, market, factors, hd)['train alpha']
                after = evaluate(trial, market, factors, hd)['train alpha']
                if after > before:
                    chosen.append(c['filter'])
                    t = trial
            if chosen:
                rows.append({'screen': screen, 'hold': hname, 'filters (chosen on 2006–2015)': ' + '.join(chosen),
                             **evaluate(t, market, factors, hd)})
    return pd.DataFrame(rows)


def run(cfg=None):
    cfg = cfg or load_config()
    bt = cfg['backtest']
    extra = bt.get('extra_benchmarks', ['IWC'])
    download_prices(extra, upgrade=True)
    factors = load_factors(cfg).copy()
    iwc = load_price(extra[0])['adj_close'].pct_change()
    factors['IWC-RF'] = iwc.reindex(factors.index) - factors['RF']
    cal = load_price('SPY').index.values
    market = Market(cal)
    trades = pd.read_parquet(DATA / 'trade_returns_all_variants.parquet')
    trades = trades[(trades['dedupe_days'] == bt['dedupe_days']) & (trades['entry_delay'] == bt['entry_delay_days'])].copy()
    for c in ('signal_date', 'entry_date'):
        trades[c] = pd.to_datetime(trades[c])
    trades = trades.reset_index(drop=True)
    F = build_features(trades, cfg, market)
    save_table(F.drop(columns=['accessions'], errors='ignore'), DATA / 'signal_features', gzip_csv=True)
    LOG.info(f"features built for {len(F):,} signals; median est. spread {F['spread_est'].median():.2%}")

    realism, curves = realism_table(F, market, factors, cfg)
    tests = filter_tests(F, market, factors)
    combo = combined_filter(F, market, factors, tests)

    # Per-trade view of the realistic setup at each hold (net of estimated costs, vs SPY and IWC)
    trade_rows = []
    for screen in SCREENS:
        t = F[(F['screen'] == screen) & F['price_found']]
        for hname, hd in HOLDS.items():
            r = trade_returns(t, market, hd, 'close', t['cost_rt'].values)
            for b in ('SPY', 'IWC'):
                br = bench_returns(t, market, hd, 'close', b)
                x = r - br
                ok = np.isfinite(x)
                trade_rows.append({'screen': screen, 'hold': hname, 'benchmark': b, 'trades': int(ok.sum()),
                                   'mean net return': np.nanmean(r[ok]), 'mean excess': np.nanmean(x[ok]),
                                   'median excess': np.nanmedian(x[ok]), 'hit rate vs benchmark': (x[ok] > 0).mean()})
    trade_tbl = pd.DataFrame(trade_rows)

    capacity = F[F['price_found']].groupby('screen').agg(
        median_price=('entry_price_raw', 'median'), median_adv=('adv20', 'median'),
        median_spread=('spread_est', 'median'), median_cost_rt=('cost_rt', 'median'),
        median_market_cap=('market_cap', 'median'),
        share_price_under_2=('entry_price_raw', lambda x: (x < 2).mean()),
        share_adv_under_100k=('adv20', lambda x: (x < 1e5).mean())).reset_index()
    # Dollars per trade if limited to 10% of one day's average volume
    capacity['median $ per trade at 10% of volume'] = capacity['median_adv'] * 0.10

    RESULTS.mkdir(parents=True, exist_ok=True)
    realism.to_csv(RESULTS / 'realistic_trading.csv', index=False)
    tests.to_csv(RESULTS / 'filter_tests.csv', index=False)
    combo.to_csv(RESULTS / 'combined_filters.csv', index=False)
    trade_tbl.to_csv(RESULTS / 'realistic_per_trade.csv', index=False)
    capacity.to_csv(RESULTS / 'capacity.csv', index=False)
    pd.DataFrame({f'{s}|{l}': c for (s, l), c in curves.items()}).to_parquet(DATA / 'realism_daily_returns.parquet')
    LOG.info('filters: done')
    return realism, tests, combo


if __name__ == '__main__':
    run()
