"""Phase 27: reality check against a real spin-off fund — Invesco S&P Spin-Off ETF (CSD), since late 2006.

Rules were written down and committed before this was run.

CSD is the closest real-world, investable version of the idea (different rules: S&P index of US spin-offs
with market cap >= $1B, held up to about 4 years, cap-weighted with limits, ETF fees included). Its returns
are actual fund returns, net of its fees and trading.
Compared over CSD's whole history and 2007-15 / 2016+: CSD, our spin-off book (Phase 21), our book limited to
spin-offs >= $1B (Phase 21 R6 rule), SPY and QQQ — CAGR, Sharpe, max drawdown, yearly returns; correlation of
monthly returns between CSD and our book.
Pre-written reading: if CSD lagged SPY over its whole life, that is a warning that the effect is weak in
practice for large spin-offs; if our >= $1B book also lagged where CSD lagged, the backtest and the real fund
agree; if our book beat SPY while CSD lagged, the gap must be explained (small spin-offs, 12-month hold, equal
weight) before trusting it.
"""
import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config

LOG = get_logger('spinoff_etf')


def run(cfg=None):
    cfg = cfg or load_config()
    import yfinance as yf
    from .analysis import load_factors
    from .strategies import perf
    px = yf.download(['CSD', 'SPY', 'QQQ'], start='2006-01-01', auto_adjust=True, progress=False)['Close']
    px.index = pd.to_datetime(px.index).tz_localize(None)
    r = px.pct_change()
    L = pd.read_parquet(DATA / 'spinoff_core_daily.parquet')
    L.index = pd.to_datetime(L.index)
    book = L['spin-off book alone (100%)']
    # >= $1B book: R6 curve is a 30% blend with SPY; rebuild the raw book from the events instead
    E = pd.read_parquet(DATA / 'spinoff_events.parquet')
    big_events = E[E['mcap'] >= 1e9]
    from .combos import SubsetEngine
    from .filters import Market
    from .prices import load_price
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    spy_r = spy_px['adj_close'].pct_change().reindex(idx).fillna(0.0)
    be = big_events.reset_index(drop=True)
    be['entry_date'] = pd.to_datetime(be['entry_date'])
    eng = SubsetEngine(be, Market(spy_px.index.values), 252, be['cost'].values)
    mask = np.ones(len(be))
    act = pd.Series(eng.active(mask), index=idx)
    big = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r)
    start = r['CSD'].first_valid_index()
    S = {'CSD (real spin-off ETF)': r['CSD'], 'our spin-off book (all)': book, 'our spin-off book (>= $1B only)': big,
         'SPY': r['SPY'], 'QQQ': r['QQQ']}
    S = {k: v.loc[start:].dropna() for k, v in S.items()}
    common = S['CSD (real spin-off ETF)'].index
    for k in S:
        S[k] = S[k].reindex(common).fillna(0.0)
    rows = []
    for k, v in S.items():
        for lab, sl in (('all', slice(None)), ('2007–15', slice(None, '2015-12-31')), ('2016+', slice('2016-01-01', None))):
            s = perf(v.loc[sl], rf)
            rows.append({'series': k, 'period': lab, 'CAGR': s.get('CAGR'), 'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown')})
    R = pd.DataFrame(rows)
    m = {k: (1 + v).resample('ME').prod() - 1 for k, v in S.items()}
    R['monthly corr CSD vs our book'] = np.corrcoef(m['CSD (real spin-off ETF)'], m['our spin-off book (all)'])[0, 1]
    R['monthly corr CSD vs our >= $1B book'] = np.corrcoef(m['CSD (real spin-off ETF)'], m['our spin-off book (>= $1B only)'])[0, 1]
    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in S.items()})
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'spinoff_etf.csv', index=False)
    yearly.to_csv(RESULTS / 'spinoff_etf_yearly.csv')
    LOG.info('spinoff_etf: done')
    return R, yearly


if __name__ == '__main__':
    run()


def explain(cfg=None):
    """Pre-registered follow-up (committed before running): why does CSD differ from our book?
    (1) Excess return over SPY by year of life after entry (year 1, 2, 3, 4) for all spin-offs and for >= $1B.
    (2) CSD-like replica from our data: >= $1B spin-offs, enter 21 trading days after first trade, hold 4 years
        (1008 trading days), equal weight; compared with CSD over CSD's history (CAGR, Sharpe, monthly corr).
    (3) Same replica with a 12-month hold. If (2) is close to CSD and (3) is not, the gap comes from holding
        old spin-offs (years 2-4) rather than from our data or costs."""
    cfg = cfg or load_config()
    import yfinance as yf
    from .analysis import load_factors
    from .combos import SubsetEngine
    from .filters import Market
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close']
    spy_r = spy.pct_change().reindex(idx).fillna(0.0)
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    E = pd.read_parquet(DATA / 'spinoff_events.parquet')
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    rows = []
    for _, e in E.iterrows():
        a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:]
        s = spy.loc[e['entry_date']:]
        for k in range(4):
            lo, hi = 252 * k, 252 * (k + 1)
            if len(a) > lo + 1 and len(s) > hi:
                aa = a.iloc[lo:min(hi, len(a) - 1) + 1]
                rows.append({'year': k + 1, 'big': e['mcap'] >= 1e9,
                             'excess': aa.iloc[-1] / aa.iloc[0] - (s.iloc[hi] / s.iloc[lo])})
    Y = pd.DataFrame(rows)
    by_year = pd.concat([Y.groupby('year')['excess'].agg(['count', 'mean', 'median']).assign(group='all'),
                         Y[Y['big']].groupby('year')['excess'].agg(['count', 'mean', 'median']).assign(group='>= $1B')])
    px = yf.download('CSD', start='2006-01-01', auto_adjust=True, progress=False)['Close']
    px = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
    px.index = pd.to_datetime(px.index).tz_localize(None)
    csd = px.pct_change().dropna()
    big = E[E['mcap'] >= 1e9].reset_index(drop=True)
    market = Market(spy_px.index.values)
    out = []
    curves = {'CSD': csd}
    for name, hold in (('replica: >= $1B, hold 4 years', 1008), ('replica: >= $1B, hold 1 year', 252)):
        eng = SubsetEngine(big, market, hold, big['cost'].values)
        mask = np.ones(len(big))
        act = pd.Series(eng.active(mask), index=idx)
        curves[name] = eng.returns(mask, rf.values).reindex(idx).where(act, spy_r)
    common = csd.index.intersection(curves['replica: >= $1B, hold 4 years'].dropna().index)
    for k, v in curves.items():
        v = v.reindex(common).fillna(0.0)
        s = perf(v, rf)
        mm = (1 + v).resample('ME').prod() - 1
        mc = (1 + csd.reindex(common).fillna(0)).resample('ME').prod() - 1
        out.append({'series': k, 'CAGR': s.get('CAGR'), 'Sharpe': s.get('Sharpe'), 'max DD': s.get('max drawdown'),
                    'monthly corr with CSD': np.corrcoef(mm, mc)[0, 1]})
    O = pd.DataFrame(out)
    by_year.to_csv(RESULTS / 'spinoff_by_year_of_life.csv')
    O.to_csv(RESULTS / 'spinoff_csd_replica.csv', index=False)
    return by_year, O
