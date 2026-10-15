"""Phase 15: a standard market-trend filter on every long-only strategy already tested (not just the winners).

Rules were written down and committed before this was run. Exploratory: the idea came from seeing that most
strategies lost heavily in 2007-08, 2015 and 2022.

Trend: SPY's close above its 200-day simple moving average at the previous close = uptrend.
Applied to each strategy's daily return series:
  F1 strategy in uptrends, cash (risk-free rate) in downtrends.
  F2 strategy in uptrends, SPY in downtrends (stays invested, sheds the strategy's extra risk).
  Every switch costs 0.5% of the portfolio (selling and buying the book, spreads and FX).
Baseline: SPY with the same filter (SPY in uptrends, cash in downtrends).
Strategies (all pre-registered earlier, all included): S&P 500 insider + surprise (L2, 6m); Nasdaq 100
insider + surprise (N2, 1y); all cluster buys + surprise (6m); big-drop tradeable (6m); net-nets (max 20);
DFV (max 20); thrift conversions (1y); biotech below cash; cheap banks + insiders.
Reported: CAGR, Sharpe, max drawdown, all years / 2006-15 / 2016+, number of switches.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config

LOG = get_logger('regime')
SWITCH_COST = 0.005

SERIES = {
    'S&P 500 insider + surprise (6m)': ('largecap_daily', 'L2 S&P 500 + positive earnings surprise|6m'),
    'Nasdaq 100 insider + surprise (1y)': ('nasdaq_daily', 'N2 Nasdaq 100 + positive earnings surprise|1y'),
    'All clusters + surprise (6m)': ('long_only_daily', 'Clusters + positive surprise|6m'),
    'Big-drop tradeable (6m)': ('long_only_daily', 'Big-drop tradeable|6m'),
    'Net-nets (max 20)': ('deep_value_daily', 'D1 Net-nets|max 20 positions'),
    'DFV screen (max 20)': ('deep_value_daily', 'D3 DFV: cheap + hated + alive + smart money|max 20 positions'),
    'Thrift conversions (1y)': ('niche_daily', 'T1 Thrift conversions|1y'),
    'Biotech below cash (1y)': ('niche_daily', 'T3b Biotech below cash|1y'),
    'Cheap banks + insiders (1y)': ('niche_daily', 'T2 Cheap banks + insiders buying|1y'),
}


def run(cfg=None):
    cfg = cfg or load_config()
    from .analysis import load_factors
    from .prices import load_price
    from .strategies import perf
    px = load_price('SPY')['adj_close']
    idx = px.index
    spy = px.pct_change().fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    up = (px > px.rolling(200).mean()).shift(1).fillna(True).astype(bool)
    flips = up.ne(up.shift(1)) & (up.index > idx[200])
    cost = flips.astype(float) * SWITCH_COST
    rows, curves = [], {}

    def report(name, variant, p):
        curves[f'{name}|{variant}'] = p
        for lab, sl in (('all', slice(None)), ('2006–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s_ = perf(p.loc[sl], rf)
            b_ = perf(spy.loc[p.index[0]:].loc[sl], rf)
            rows.append({'strategy': name, 'variant': variant, 'period': lab, 'CAGR': s_.get('CAGR'), 'SPY CAGR': b_.get('CAGR'),
                         'Sharpe': s_.get('Sharpe'), 'SPY Sharpe': b_.get('Sharpe'), 'max DD': s_.get('max drawdown'),
                         'SPY max DD': b_.get('max drawdown'), 'switches': int(flips.loc[p.index[0]:].loc[sl].sum())})

    base = spy.loc[idx[201]:]
    report('SPY', 'no filter', base)
    report('SPY', 'F1 cash in downtrends', pd.Series(np.where(up, spy, rf), index=idx).loc[base.index] - cost.loc[base.index])
    for name, (fname, col) in SERIES.items():
        f = DATA / f'{fname}.parquet'
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if col not in d:
            LOG.warning(f'regime: {col} not in {fname}')
            continue
        s = d[col].reindex(idx)
        s = s.loc[s.first_valid_index():].fillna(0.0)
        s = s.loc[max(s.index[0], idx[201]):]
        u, c = up.loc[s.index], cost.loc[s.index]
        report(name, 'no filter', s)
        report(name, 'F1 cash in downtrends', pd.Series(np.where(u, s, rf.loc[s.index]), index=s.index) - c)
        report(name, 'F2 SPY in downtrends', pd.Series(np.where(u, s, spy.loc[s.index]), index=s.index) - c)
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'regime.csv', index=False)
    pd.DataFrame(curves).to_parquet(DATA / 'regime_daily.parquet')
    LOG.info('regime: done')
    return r


if __name__ == '__main__':
    run()
