"""Phase 29: a practical long-only "Lazy Prices" portfolio for a Trading 212 account (fits a 50-stock pie).

Rules were written down and committed before this was run, after the Phase 9 quintile test (which showed the
signal is strong but mostly on the short side, with high turnover).

Universe: S&P 500 members (Compustat index constituents, 000003) at each quarter-end (Mar, Jun, Sep, Dec),
price >= $1. Score: cosine similarity of the company's most recent 10-K/10-Q filed in the prior 3 months to
its comparable filing a year earlier (Phase 9 scores); higher = less changed.
Portfolios (equal weight at each quarter-end, held 3 months buy-and-hold, then rebalanced):
  T1 (MAIN) the 50 S&P 500 stocks with the HIGHEST full-document similarity
  T2 every scored S&P 500 stock EXCEPT the bottom fifth (the most changed)
  T3 the 50 with the highest Risk Factors (Item 1A) similarity (only stocks with a Risk Factors score)
Costs: 0.25% per side on the share of the portfolio that changes (0.10% spread for these large stocks +
0.15% FX), charged at each rebalance.
Monthly returns from the Compustat monthly panel (next-month returns, delisting-adjusted as in Phase 17).
Benchmarks: SPY and RSP (equal-weight S&P 500 ETF), same months.
Reported for 2005-2015, 2016+ and all: CAGR, Sharpe, max drawdown, average turnover per rebalance; FF5+UMD
alpha with t.
Pre-written bar: T1 beats BOTH SPY and RSP on CAGR in BOTH periods after costs. T2 and T3 are reported but
cannot rescue T1.
"""
import numpy as np
import pandas as pd

from .common import CACHE, RESULTS, get_logger, load_config

LOG = get_logger('text_portfolio')
SIDE = 0.0025


def run(cfg=None):
    import statsmodels.api as sm
    from .compustat import wrds_dir
    from .phase_b import members_on
    from .prices import load_price
    from .text_changes import load_scores
    cfg = cfg or load_config()
    S = load_scores()
    S['filing_date'] = pd.to_datetime(S['filing_date'])
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    co['gvkey'] = co['gvkey'].astype(str)
    S = S.merge(co.dropna(), on='cik').sort_values('filing_date')
    panel = pd.read_parquet(CACHE / 'cs_monthly_panel.parquet', columns=['sec', 'month_end', 'price', 'adv20', 'gvkey'])
    rets = pd.read_parquet(CACHE / 'cs_monthly_returns.parquet', columns=['sec', 'month_end', 'ret_next'])
    panel = panel.merge(rets, on=['sec', 'month_end'])
    panel['gvkey'] = panel['gvkey'].astype(str)
    panel['month_end'] = pd.to_datetime(panel['month_end'])
    panel = panel.sort_values('adv20').drop_duplicates(['gvkey', 'month_end'], keep='last')
    R = panel.pivot(index='month_end', columns='sec', values='ret_next').sort_index()
    months = R.index
    qends = [m for m in months if m.month in (3, 6, 9, 12) and m >= pd.Timestamp('2005-03-01')]
    port = {'T1 top 50 least-changed (full filing)': [], 'T2 S&P 500 minus most-changed fifth': [],
            'T3 top 50 least-changed (Risk Factors)': []}
    prev = {k: pd.Series(dtype=float) for k in port}
    turn = {k: [] for k in port}
    for qe in qends:
        g = panel[(panel['month_end'] == qe) & (panel['price'] >= 1)].copy()
        g = g[members_on(list(zip(g['gvkey'], [qe] * len(g))), ('000003',), cfg)]
        if len(g) < 100:
            continue
        w = S[(S['filing_date'] <= qe) & (S['filing_date'] > qe - pd.DateOffset(months=3))].drop_duplicates('gvkey', keep='last')
        x = g.merge(w[['gvkey', 'sim_cosine', 'rf_sim_cosine']], on='gvkey')
        picks = {}
        a = x.dropna(subset=['sim_cosine'])
        picks['T1 top 50 least-changed (full filing)'] = a.nlargest(50, 'sim_cosine')['sec']
        cut = a['sim_cosine'].quantile(0.2)
        picks['T2 S&P 500 minus most-changed fifth'] = a[a['sim_cosine'] > cut]['sec']
        b = x.dropna(subset=['rf_sim_cosine'])
        picks['T3 top 50 least-changed (Risk Factors)'] = b.nlargest(50, 'rf_sim_cosine')['sec'] if len(b) >= 50 else pd.Series(dtype=str)
        nxt = months[months > qe][:3]
        for k, secs in picks.items():
            if len(secs) == 0:
                continue
            wts = pd.Series(1.0 / len(secs), index=secs.values)
            to = wts.subtract(prev[k], fill_value=0).abs().sum() / 2 if len(prev[k]) else 1.0
            turn[k].append(to)
            # buy and hold for 3 months: track each position's value
            val = wts.copy()
            for i, m in enumerate([qe] + list(nxt[:-1])):
                r = R.loc[m].reindex(val.index).fillna(0.0)
                ret = float((val * r).sum() / val.sum())
                if i == 0:
                    ret -= to * 2 * SIDE
                port[k].append((pd.Timestamp(m).to_period('M') + 1, ret))
                val = val * (1 + r)
            prev[k] = val / val.sum()
    P = pd.DataFrame({k: pd.Series(dict(v)) for k, v in port.items()}).sort_index()
    for t in ('SPY', 'RSP'):
        s = load_price(t)['adj_close'].resample('ME').last().pct_change()
        s.index = s.index.to_period('M')
        P[t] = s.reindex(P.index)
    F = pd.read_parquet(CACHE / 'ff5_umd_daily_full.parquet')
    Fm = (1 + F).resample('ME').prod() - 1
    Fm.index = Fm.index.to_period('M')
    rows = []
    for lab, a, b in (('all', None, None), ('2005–2015', None, '2015-12'), ('2016+', '2016-01', None)):
        for k in P.columns:
            x = P[k].loc[a:b].dropna()
            if len(x) < 24:
                continue
            w = (1 + x).cumprod()
            rf = Fm['RF'].reindex(x.index).fillna(0)
            ex = x - rf
            f = Fm.reindex(x.index).dropna()
            res = sm.OLS(ex.reindex(f.index), sm.add_constant(f[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']])).fit(
                cov_type='HAC', cov_kwds={'maxlags': 6})
            rows.append({'portfolio': k, 'period': lab, 'months': len(x), 'CAGR': w.iloc[-1] ** (12 / len(x)) - 1,
                         'Sharpe': ex.mean() / ex.std() * np.sqrt(12), 'max DD': (w / w.cummax() - 1).min(),
                         'alpha/yr': res.params['const'] * 12, 'alpha t': res.tvalues['const'],
                         'avg turnover per rebalance': np.mean(turn[k]) if k in turn else np.nan})
    out = pd.DataFrame(rows)
    g = lambda n, p: out.loc[(out['portfolio'] == n) & (out['period'] == p), 'CAGR'].iloc[0]
    T1 = 'T1 top 50 least-changed (full filing)'
    out['pre-written bar met (T1)'] = all(g(T1, p) > g('SPY', p) and g(T1, p) > g('RSP', p) for p in ('2005–2015', '2016+'))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / 'text_portfolio.csv', index=False)
    P.to_csv(RESULTS / 'text_portfolio_monthly.csv')
    LOG.info('text_portfolio: done')
    return out


if __name__ == '__main__':
    run()
