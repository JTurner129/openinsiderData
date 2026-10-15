"""Phase 5: event study, calendar-time portfolios, factor regressions, robustness."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from .common import CACHE, DATA, RESULTS, SCREENS, get_logger, load_config, save_table
from .prices import load_price

LOG = get_logger('analysis')
TRADING_DAYS = 252
ERAS = {'pre-2010': (None, 2009), '2010–2019': (2010, 2019), '2020+': (2020, None)}
FF_URLS = {
    'ff5': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'mom': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip',
}


# --- Fama-French factors ------------------------------------------------------
def _read_french_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        text = zf.read(zf.namelist()[0]).decode('latin-1')
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip()[:8].isdigit()) - 1
    body = []
    for l in lines[start:]:
        if body and not l.strip():
            break
        body.append(l)
    df = pd.read_csv(io.StringIO('\n'.join(body)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
    df.columns = [c.strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors='coerce') / 100.0


def load_factors(cfg):
    path = CACHE / 'ff_factors_daily.parquet'
    if path.exists():
        return pd.read_parquet(path)
    headers = {'User-Agent': cfg['sec']['user_agent']}
    ff5 = _read_french_zip(requests.get(FF_URLS['ff5'], headers=headers, timeout=60).content)
    mom = _read_french_zip(requests.get(FF_URLS['mom'], headers=headers, timeout=60).content)
    mom.columns = ['UMD']
    f = ff5.join(mom, how='inner')
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(path)
    return f


# --- Statistics helpers ---------------------------------------------------------
def clustered_t(values, clusters):
    """t-stat of the mean with standard errors clustered by `clusters`."""
    v = pd.Series(values).astype(float)
    ok = v.notna().values
    v, c = v[ok], pd.Series(clusters)[ok]
    if len(v) < 3 or c.nunique() < 2:
        return np.nan
    res = sm.OLS(v.values, np.ones(len(v))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(c)[0]})
    return float(res.tvalues[0])


def summarize_events(df, horizon):
    r, xs, xi = df[f'ret_{horizon}'], df[f'xs_spy_{horizon}'], df[f'xs_iwm_{horizon}']
    ok = r.notna()
    r, xs, xi, d = r[ok], xs[ok], xi[ok], df.loc[ok]
    n = len(r)
    if n == 0:
        return {'count': 0}
    return {
        'count': n,
        'mean': r.mean(), 'median': r.median(), 'hit_rate': (r > 0).mean(),
        'hit_rate_vs_spy': (xs > 0).mean(),
        'mean_xs_spy': xs.mean(), 'median_xs_spy': xs.median(), 'mean_xs_iwm': xi.mean(),
        't_xs_spy': clustered_t(xs.values, d['signal_date'].dt.to_period('M').astype(str).values),
        't_xs_iwm': clustered_t(xi.values, d['signal_date'].dt.to_period('M').astype(str).values),
        't_raw': clustered_t(r.values, d['signal_date'].dt.to_period('M').astype(str).values),
        'mean_net': d[f'net_{horizon}'].mean(), 'mean_net_xs_spy': d[f'net_xs_spy_{horizon}'].mean(),
        'mean_spy': d[f'spy_{horizon}'].mean(), 'mean_iwm': d[f'iwm_{horizon}'].mean(),
        'truncated_share': d[f'truncated_{horizon}'].mean() if f'truncated_{horizon}' in d else np.nan,
    }


def event_study(trades, horizons):
    rows = []
    for screen in SCREENS:
        s = trades[(trades['screen'] == screen) & trades['price_found']]
        groups = [('All', s)] + [(str(y), g) for y, g in s.groupby('signal_year')]
        for label, g in groups:
            for h in horizons:
                rows.append({'screen': screen, 'signal_year': label, 'horizon': h, **summarize_events(g, h)})
    return pd.DataFrame(rows)


def missing_prices(trades):
    rows = []
    for screen in SCREENS:
        s = trades[trades['screen'] == screen]
        for label, g in [('All', s)] + [(str(y), x) for y, x in s.groupby('signal_year')]:
            entered = g[g['price_status'] != 'not_yet_entered']
            st = entered['price_status'].value_counts()
            n = len(entered)
            rows.append({'screen': screen, 'signal_year': label, 'signals': n,
                         'priced': int(st.get('ok', 0)),
                         'pct_missing': 1 - st.get('ok', 0) / n if n else np.nan,
                         'no_ticker': int(st.get('no_ticker', 0)),
                         'ticker_not_found': int(st.get('ticker_not_found', 0)),
                         'no_price_at_entry': int(st.get('no_price_at_entry', 0)),
                         'price_mismatch': int(st.get('price_mismatch', 0)),
                         'not_yet_entered': int((g['price_status'] == 'not_yet_entered').sum())})
    return pd.DataFrame(rows)


# --- Calendar-time portfolios -------------------------------------------------
class SeriesCache:
    """Per-ticker adjusted return arrays aligned to the master calendar."""

    def __init__(self, cal):
        self.cal = cal
        self.cache = {}

    def get(self, ticker):
        if ticker not in self.cache:
            px = load_price(ticker)
            if px is None or px.empty:
                self.cache[ticker] = None
            else:
                px = px.dropna(subset=['adj_close'])
                close = px['adj_close'].reindex(self.cal)
                has = close.notna().values
                prev = px['adj_close'].shift(1).reindex(self.cal)
                ret = (close / prev - 1).fillna(0.0).values.astype('float32')
                intraday = (px['adj_close'] / px['adj_open'].where(px['adj_open'] > 0, px['adj_close']) - 1)
                intraday = intraday.reindex(self.cal).fillna(0.0).values.astype('float32')
                last = int(np.flatnonzero(has)[-1]) if has.any() else -1
                self.cache[ticker] = (ret, intraday, last)
        return self.cache[ticker]


def calendar_portfolio(trades, hold_days, series, rf):
    """Daily equal-weight return of all stocks within `hold_days` of entry.

    Days with no open positions earn the risk-free rate (cash).
    """
    cal = series.cal
    total = np.zeros(len(cal))
    count = np.zeros(len(cal))
    for _, t in trades.iterrows():
        s = series.get(t['ticker'])
        if s is None or pd.isna(t['entry_date']):
            continue
        ret, intraday, last = s
        m = int(np.searchsorted(cal, np.datetime64(t['entry_date']), 'left'))
        end = min(m + hold_days, last, len(cal) - 1)
        if end < m:
            continue
        total[m] += intraday[m]
        count[m] += 1
        if end > m:
            total[m + 1:end + 1] += ret[m + 1:end + 1]
            count[m + 1:end + 1] += 1
    port = pd.Series(np.where(count > 0, total / np.maximum(count, 1), np.nan), index=cal)
    n = pd.Series(count, index=cal)
    first = n[n > 0].index.min()
    if pd.isna(first):
        return pd.Series(dtype=float), n
    port = port.loc[first:]
    port = port.fillna(rf.reindex(port.index).ffill().fillna(0.0))
    return port, n.loc[first:]


def perf_stats(r, rf, spy):
    r = r.dropna()
    if len(r) < 20:
        return {}
    rf_ = rf.reindex(r.index).ffill().fillna(0.0)
    ex = r - rf_
    years = len(r) / TRADING_DAYS
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    sp = spy.reindex(r.index)
    beta = np.cov(r[sp.notna()], sp.dropna())[0, 1] / sp.var() if sp.notna().sum() > 20 else np.nan
    return {'start': r.index.min().date().isoformat(), 'end': r.index.max().date().isoformat(),
            'cagr': wealth.iloc[-1] ** (1 / years) - 1, 'volatility': r.std() * np.sqrt(TRADING_DAYS),
            'sharpe': ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan,
            'sortino': ex.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan,
            'max_drawdown': dd.min(), 'beta_spy': beta, 'total_return': wealth.iloc[-1] - 1}


def factor_alpha(r, factors, model):
    cols = ['Mkt-RF'] if model == 'CAPM' else ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']
    d = pd.concat([r.rename('r'), factors], axis=1, join='inner').dropna()
    if len(d) < 60:
        return {'model': model, 'n_days': len(d)}
    y = d['r'] - d['RF']
    res = sm.OLS(y, sm.add_constant(d[cols])).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    a, se = res.params['const'], res.bse['const']
    out = {'model': model, 'n_days': len(d), 'alpha_ann': a * TRADING_DAYS, 't_alpha': res.tvalues['const'],
           'p_alpha': res.pvalues['const'], 'ci95_low': (a - 1.96 * se) * TRADING_DAYS,
           'ci95_high': (a + 1.96 * se) * TRADING_DAYS, 'r2': res.rsquared,
           'factor_end': d.index.max().date().isoformat()}
    for c in cols:
        out[f'beta_{c}'] = res.params[c]
    return out


def portfolio_analysis(trades, cfg, factors, series, spy_ret, rf):
    stats, alphas, curves = [], [], {}
    holds = cfg['backtest']['holding_periods']
    for screen in SCREENS:
        t = trades[(trades['screen'] == screen) & trades['price_found']]
        for hname, hd in holds.items():
            port, n = calendar_portfolio(t, hd, series, rf)
            if port.empty:
                continue
            curves[(screen, hname)] = port
            st = perf_stats(port, rf, spy_ret)
            stats.append({'screen': screen, 'holding': hname, 'avg_positions': n[n > 0].mean(),
                          'days_in_cash': int((n == 0).sum()), **st})
            for model in ('CAPM', 'FF5+UMD'):
                alphas.append({'screen': screen, 'holding': hname, **factor_alpha(port, factors, model)})
    return pd.DataFrame(stats), pd.DataFrame(alphas), curves


def benchmark_stats(cfg, rf, start):
    rows, curves = [], {}
    spy_ret = None
    for b in cfg['backtest']['benchmarks']:
        r = load_price(b)['adj_close'].pct_change().loc[start:].dropna()
        if b == 'SPY':
            spy_ret = r
        curves[b] = r
    for b, r in curves.items():
        rows.append({'screen': b, 'holding': '-', **perf_stats(r, rf, spy_ret)})
    return pd.DataFrame(rows), curves


# --- Robustness -----------------------------------------------------------------
def _variant_summary(t, series, rf, factors, label, key, hold=252):
    t = t[t['price_found']]
    xs3 = t['xs_spy_3m'].dropna()
    xs1 = t['xs_spy_1y'].dropna()
    port, _ = calendar_portfolio(t, hold, series, rf)
    alpha = factor_alpha(port, factors, 'FF5+UMD') if not port.empty else {}
    return {key: label, 'signals_priced': len(t), 'mean_xs_spy_3m': xs3.mean(), 'mean_xs_spy_1y': xs1.mean(),
            'hit_rate_vs_spy_1y': (xs1 > 0).mean() if len(xs1) else np.nan,
            'ff5_alpha_ann': alpha.get('alpha_ann'), 't_ff5_alpha': alpha.get('t_alpha')}


def robustness(all_trades, cfg, factors, series, rf, curves):
    bt = cfg['backtest']
    base = all_trades[(all_trades['dedupe_days'] == bt['dedupe_days']) & (all_trades['entry_delay'] == bt['entry_delay_days'])]
    era_rows, dd_rows, delay_rows, liq_rows = [], [], [], []
    for screen in SCREENS:
        b = base[(base['screen'] == screen) & base['price_found']]
        port = curves.get((screen, '1y'))
        for era, (lo, hi) in ERAS.items():
            sel = b[((lo is None) | (b['signal_year'] >= (lo or 0))) & ((hi is None) | (b['signal_year'] <= (hi or 9999)))]
            row = {'screen': screen, 'era': era, 'signals_priced': len(sel)}
            for h in ('3m', '1y'):
                x = sel[f'xs_spy_{h}'].dropna()
                row[f'mean_xs_spy_{h}'] = x.mean()
                row[f'hit_rate_vs_spy_{h}'] = (x > 0).mean() if len(x) else np.nan
            if port is not None:
                p = port.loc[f'{lo or 1900}-01-01':f'{hi or 2100}-12-31']
                a = factor_alpha(p, factors, 'FF5+UMD')
                row['ff5_alpha_ann'] = a.get('alpha_ann')
                row['t_ff5_alpha'] = a.get('t_alpha')
            era_rows.append(row)
        for dd in sorted(all_trades['dedupe_days'].unique()):
            t = all_trades[(all_trades['screen'] == screen) & (all_trades['dedupe_days'] == dd) &
                           (all_trades['entry_delay'] == bt['entry_delay_days'])]
            dd_rows.append({'screen': screen, **_variant_summary(t, series, rf, factors, int(dd), 'dedupe_days')})
        for delay in sorted(all_trades['entry_delay'].unique()):
            t = all_trades[(all_trades['screen'] == screen) & (all_trades['dedupe_days'] == bt['dedupe_days']) &
                           (all_trades['entry_delay'] == delay)]
            lab = 'next open' if delay == 1 else f'open +{delay} trading days'
            delay_rows.append({'screen': screen, **_variant_summary(t, series, rf, factors, lab, 'entry')})
        t = base[base['screen'] == screen]
        liq_rows.append({'screen': screen, **_variant_summary(t, series, rf, factors, 'no filter', 'liquidity')})
        p2 = bt['min_entry_price_variant']
        liq_rows.append({'screen': screen, **_variant_summary(t[t['entry_price_raw'] >= p2], series, rf, factors,
                                                              f'price ≥ ${p2}', 'liquidity')})
    return pd.DataFrame(era_rows), pd.DataFrame(dd_rows), pd.DataFrame(delay_rows), pd.DataFrame(liq_rows)


def headline(es, stats, alphas, trades, missing):
    rows = []
    for screen, (label, _, _) in SCREENS.items():
        e = es[(es['screen'] == screen) & (es['signal_year'] == 'All') & (es['horizon'] == '1y')]
        e = e.iloc[0] if len(e) else {}
        st = stats[(stats['screen'] == screen) & (stats['holding'] == '1y')]
        st = st.iloc[0] if len(st) else {}
        a = alphas[(alphas['screen'] == screen) & (alphas['holding'] == '1y') & (alphas['model'] == 'FF5+UMD')]
        a = a.iloc[0] if len(a) else {}
        mp = missing[(missing['screen'] == screen) & (missing['signal_year'] == 'All')].iloc[0]
        rows.append({'screen': screen, 'Screen': label,
                     'Signals': int((trades['screen'] == screen).sum()),
                     'Priced %': 1 - mp['pct_missing'],
                     'Avg 1y return': e.get('mean'), 'Avg 1y excess vs SPY': e.get('mean_xs_spy'),
                     '1y hit rate': e.get('hit_rate'), 'Sharpe': st.get('sharpe'),
                     'Max drawdown': st.get('max_drawdown'), 'FF5 alpha (ann.)': a.get('alpha_ann'),
                     'FF5 alpha t': a.get('t_alpha'), 'FF5 alpha p': a.get('p_alpha'),
                     'Significant': '✓' if a.get('p_alpha', 1) < 0.05 else '✗'})
    return pd.DataFrame(rows)


def run(cfg=None):
    cfg = cfg or load_config()
    bt = cfg['backtest']
    all_trades = pd.read_parquet(DATA / 'trade_returns_all_variants.parquet')
    for c in ['signal_date', 'entry_date']:
        all_trades[c] = pd.to_datetime(all_trades[c])
    trades = all_trades[(all_trades['dedupe_days'] == bt['dedupe_days']) &
                        (all_trades['entry_delay'] == bt['entry_delay_days'])].copy()
    factors = load_factors(cfg)
    spy = load_price('SPY')
    cal = spy.index.values
    rf = factors['RF']
    spy_ret = spy['adj_close'].pct_change()
    series = SeriesCache(cal)

    es = event_study(trades, list(bt['horizons']))
    miss = missing_prices(trades)
    stats, alphas, curves = portfolio_analysis(trades, cfg, factors, series, spy_ret, rf)
    start = min(c.index.min() for c in curves.values())
    bstats, bcurves = benchmark_stats(cfg, rf, start)
    for b, r in bcurves.items():
        for model in ('CAPM', 'FF5+UMD'):
            alphas = pd.concat([alphas, pd.DataFrame([{'screen': b, 'holding': '-', **factor_alpha(r, factors, model)}])],
                               ignore_index=True)
    stats = pd.concat([stats, bstats], ignore_index=True)
    era, dd, delay, liq = robustness(all_trades, cfg, factors, series, rf, curves)
    head = headline(es, stats, alphas, trades, miss)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = {'headline': head, 'event_study': es, 'missing_prices': miss, 'portfolio_stats': stats,
              'factor_alphas': alphas, 'robust_era': era, 'robust_dedupe': dd, 'robust_entry_delay': delay,
              'robust_liquidity': liq}
    for name, df in tables.items():
        df.to_csv(RESULTS / f'{name}.csv', index=False)
    daily = pd.DataFrame({f'{s}|{h}': c for (s, h), c in curves.items()})
    for b, r in bcurves.items():
        daily[b] = r
    daily.index.name = 'date'
    save_table(daily, DATA / 'portfolio_daily_returns', index=True)
    LOG.info(f"Headline:\n{head.to_string()}")
    return tables, daily


if __name__ == '__main__':
    run()
