"""Phase 31: spin-offs OUTSIDE the US (and pre-2005 US) from LSEG Deals + Datastream — an independent sample.

Rules were written down and committed before any of the Datastream prices were examined. They copy the US
rules (Phases 21-24) as closely as weekly data allows; nothing is tuned on this sample.

Events: LSEG Deals (SDC) "Spinoffs", status Completed/Unconditional, target SEDOL converted to ISIN in
Workspace (file eikon/sdc_spinoffs_2026-10-19.xlsx, eikon/convert_sedol_to_isin_DONE.xlsx). Weekly Datastream
series per ISIN (RI local, RI in USD, MV local, MV in USD, P, VO) from 60 days before the effective date.
A series counts as the NEW spin-off share only if its first valid price is within -30 / +120 days of the SDC
effective date (otherwise the code points at the parent, an old listing, or a bond: dropped).
Entry: the first weekly close at least 4 weeks after the first valid price (about 21 trading days).
Hold: 52 weeks. Returns from total return index in USD (X(RI)~U$; local RI if USD missing, then flagged).
Tradeable at entry: market value >= $50M (USD) and price present; average weekly dollar volume (P x VO x MV-USD/MV
local) >= $500K (about $100K a day). Costs: 1.0% round trip (wider spreads abroad) + 0.15% FX each way.
Benchmark: iShares MSCI ACWI (USD) for all; also the iShares MSCI country ETF in USD where it exists
(same 52 weeks). Before ACWI exists (2008) MSCI World ETF URTH is not available either, so pre-2008 trades
use SPY as the global proxy (stated in results).
Reported: per-trade 52-week excess over the benchmark (mean, median, share > 0, month-clustered t), by region
(US, Canada, UK, Europe ex-UK, Korea, Japan/Asia ex-Korea, Australia/NZ, India, other) and by period, plus an
equal-weight calendar-time portfolio (each trade held 52 weeks, idle weeks in the benchmark): CAGR, Sharpe,
max drawdown vs the benchmark. Also: the "cheap" hint (book-to-market) cannot be tested here (no book values);
the "different industry from parent" hint is NOT testable with this export (the parent's industry is only in the
SDC major-industry scheme, the target's only in the macro/mid scheme; they are not comparable). Needs a re-export
with "Target SDC Industry" (same scheme as the parent's) before it can be tested.
Pre-written bar for "the spin-off effect exists outside the US": non-US per-trade mean excess > 0 with
clustered t >= 2, AND the non-US portfolio beats its benchmark on CAGR.
"""
import numpy as np
import pandas as pd

from .common import RESULTS, get_logger

LOG = get_logger('intl_spinoffs')
REGION = {'United States': 'US', 'Canada': 'Canada', 'United Kingdom': 'UK', 'South Korea': 'Korea',
          'Australia': 'Australia/NZ', 'New Zealand': 'Australia/NZ', 'India': 'India', 'Japan': 'Japan/Asia',
          'Hong Kong': 'Japan/Asia', 'China (Mainland)': 'Japan/Asia', 'Singapore': 'Japan/Asia', 'Taiwan': 'Japan/Asia',
          'Malaysia': 'Japan/Asia', 'Thailand': 'Japan/Asia', 'Indonesia': 'Japan/Asia', 'Philippines': 'Japan/Asia'}
EUROPE = {'Sweden', 'Norway', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Switzerland', 'Denmark',
          'Finland', 'Belgium', 'Austria', 'Ireland-Rep', 'Ireland', 'Portugal', 'Greece', 'Poland', 'Luxembourg'}
ETF = {'Canada': 'EWC', 'UK': 'EWU', 'Korea': 'EWY', 'Australia/NZ': 'EWA', 'India': 'INDA', 'US': 'SPY'}
RT_COST = 0.010 + 2 * 0.0015


def region(n):
    if n in REGION:
        return REGION[n]
    if n in EUROPE:
        return 'Europe ex-UK'
    return 'Other'


def parse_blocks(paths):
    """Each DSGRID block is 8 columns: Date, RI, RI USD, MV, MV USD, P, VO, blank. Row 0 holds the ISIN."""
    out = {}
    for p in paths:
        raw = pd.read_excel(p, header=None)
        for b in range((raw.shape[1] + 7) // 8):
            blk = raw.iloc[:, b * 8:b * 8 + 7]
            if blk.shape[1] < 7:
                continue
            isin = str(blk.iloc[0, 0])
            df = blk.iloc[2:].copy()
            df.columns = ['date', 'ri', 'ri_usd', 'mv', 'mv_usd', 'p', 'vo']
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['date']).set_index('date')
            if df['ri'].notna().sum() + df['ri_usd'].notna().sum() > 0:
                out[isin] = df
    return out


def bench_weekly():
    import yfinance as yf
    tick = ['ACWI', 'SPY', 'EWC', 'EWU', 'EWY', 'EWA', 'INDA']
    px = yf.download(tick, start='1993-01-01', auto_adjust=True, progress=False)['Close']
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def run(paths, events_csv='eikon/isin_list.csv', sdc_xlsx='eikon/sdc_spinoffs_2026-10-19.xlsx'):
    S = parse_blocks(paths)
    ev = pd.read_csv(events_csv)
    sdc = pd.read_excel(sdc_xlsx, header=None)
    sdc_ind = sdc[[13, 24, 28]].copy()
    sdc_ind.columns = ['SEDOL', 'target_macro', 'other_macro']
    ev = ev.merge(sdc_ind.drop_duplicates('SEDOL'), on='SEDOL', how='left')
    # parent industry: column 21 is the immediate parent's SDC major industry; use target vs ultimate-parent macro
    sdc_par = sdc[[13, 21, 25]].copy()
    sdc_par.columns = ['SEDOL', 'parent_major', 'target_mid']
    ev = ev.merge(sdc_par.drop_duplicates('SEDOL'), on='SEDOL', how='left')
    ev['Effective'] = pd.to_datetime(ev['Effective'])
    B = bench_weekly()
    rows = []
    for _, e in ev.iterrows():
        s = S.get(e['ISIN'])
        if s is None:
            continue
        ri = s['ri_usd'].where(s['ri_usd'].notna(), np.nan)
        usd = True
        if ri.notna().sum() < 10:
            ri, usd = s['ri'], False
        ri = ri.dropna()
        if len(ri) < 30:
            continue
        first = ri.index[0]
        if not (e['Effective'] - pd.Timedelta(days=30) <= first <= e['Effective'] + pd.Timedelta(days=120)):
            continue
        k = ri.index.searchsorted(first + pd.Timedelta(days=28))
        if k + 52 >= len(ri) + 1 or k >= len(ri):
            continue
        entry, exit_ = ri.index[k], ri.index[min(k + 52, len(ri) - 1)]
        if (exit_ - entry).days < 300:
            continue
        mvu = s['mv_usd'].reindex([entry]).iloc[0] if 'mv_usd' in s else np.nan
        mvl = s['mv'].reindex([entry]).iloc[0]
        w = s.loc[:entry].tail(4)
        fx = (w['mv_usd'] / w['mv']).replace([np.inf, -np.inf], np.nan)
        dv = (w['p'] * w['vo'] * 1000 * fx).mean()
        if not (pd.notna(mvu) and mvu >= 50 and pd.notna(dv) and dv >= 500_000):
            continue
        ret = ri.loc[exit_] / ri.loc[entry] - 1 - RT_COST
        reg = region(e['Nation'])
        bt = 'ACWI' if entry >= pd.Timestamp('2008-04-01') else 'SPY'
        b = B[bt].dropna()
        bret = b.asof(exit_) / b.asof(entry) - 1
        lt = ETF.get(reg)
        lret = np.nan
        if lt and lt in B and B[lt].first_valid_index() is not None and B[lt].first_valid_index() <= entry:
            l = B[lt].dropna()
            lret = l.asof(exit_) / l.asof(entry) - 1
        rows.append({'ISIN': e['ISIN'], 'name': e['Name'], 'nation': e['Nation'], 'region': reg, 'entry': entry,
                     'exit': exit_, 'ret': ret, 'bench': bt, 'excess_global': ret - bret, 'excess_local': ret - lret,
                     'usd_returns': usd, 'mv_usd_m': mvu})
    T = pd.DataFrame(rows)
    T['month'] = T['entry'].dt.to_period('M').astype(str)

    def cl(x, m):
        x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
        m = pd.Series(np.asarray(m)).reset_index(drop=True)
        ok = x.notna()
        x, m = x[ok], m[ok]
        if len(x) < 5:
            return np.nan
        g = (x - x.mean()).groupby(m).sum()
        return x.mean() / (np.sqrt((g ** 2).sum()) / len(x))

    out = []
    groups = [('ALL non-US', T['region'] != 'US'), ('US (SDC list)', T['region'] == 'US')]
    groups += [(r, T['region'] == r) for r in sorted(T['region'].unique()) if r != 'US']
    groups += [('non-US, entry before 2008', (T['region'] != 'US') & (T['entry'] < '2008-01-01')),
               ('non-US, 2008-2015', (T['region'] != 'US') & (T['entry'] >= '2008-01-01') & (T['entry'] < '2016-01-01')),
               ('non-US, 2016+', (T['region'] != 'US') & (T['entry'] >= '2016-01-01'))]
    for name, m in groups:
        x = T[m]
        if len(x) < 5:
            continue
        out.append({'group': name, 'n': len(x), 'mean excess vs global': x['excess_global'].mean(),
                    'median': x['excess_global'].median(), 'share > 0': (x['excess_global'] > 0).mean(),
                    'clustered t': cl(x['excess_global'], x['month']),
                    'mean excess vs local ETF': x['excess_local'].mean(), 'n with local ETF': x['excess_local'].notna().sum()})
    R = pd.DataFrame(out)
    nu = T[T['region'] != 'US']
    # calendar-time portfolio, weekly, non-US
    weeks = pd.date_range(nu['entry'].min(), nu['exit'].max(), freq='W-FRI')
    port = []
    Bw = B.resample('W-FRI').last()
    for wk in weeks[1:]:
        live = nu[(nu['entry'] < wk) & (nu['exit'] >= wk)]
        rs = []
        for _, t in live.iterrows():
            s = S[t['ISIN']]
            ri = s['ri_usd'] if t['usd_returns'] else s['ri']
            ri = ri.dropna()
            a, b_ = ri.asof(wk - pd.Timedelta(days=7)), ri.asof(wk)
            if pd.notna(a) and pd.notna(b_) and a > 0:
                rs.append(b_ / a - 1)
        bt = 'ACWI' if wk >= pd.Timestamp('2008-04-04') else 'SPY'
        br = Bw[bt].pct_change().get(wk, np.nan)
        port.append((wk, np.mean(rs) if rs else br, br))
    P = pd.DataFrame(port, columns=['week', 'spinoffs', 'benchmark']).set_index('week').dropna()
    P['spinoffs'] -= 0  # costs are in the per-trade numbers; portfolio is gross (reported as such)
    def st(r):
        w = (1 + r).cumprod()
        return w.iloc[-1] ** (52 / len(r)) - 1, r.mean() / r.std() * np.sqrt(52), (w / w.cummax() - 1).min()
    sp, bm = st(P['spinoffs']), st(P['benchmark'])
    R['non-US portfolio CAGR (gross)'] = sp[0]
    R['benchmark CAGR'] = bm[0]
    R['portfolio Sharpe (no rf)'] = sp[1]
    R['benchmark Sharpe (no rf)'] = bm[1]
    R['portfolio max DD'] = sp[2]
    R['benchmark max DD'] = bm[2]
    allnu = R.loc[R['group'] == 'ALL non-US'].iloc[0]
    R['effect exists outside US (pre-written bar)'] = bool(allnu['mean excess vs global'] > 0 and allnu['clustered t'] >= 2 and sp[0] > bm[0])
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'intl_spinoffs.csv', index=False)
    T.to_csv(RESULTS / 'intl_spinoff_trades.csv', index=False)
    P.to_csv(RESULTS / 'intl_spinoff_portfolio_weekly.csv')
    LOG.info(f'intl: {len(T)} trades ({len(nu)} non-US); done')
    return R, T, P
