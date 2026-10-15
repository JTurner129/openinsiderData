"""Phase 26: do insider buying, insider selling, short interest or an activist stake improve spin-offs?

Rules were written down and committed before this was run. 2005-2026 only (SEC Form 4 data starts 2006 and
Compustat short interest mid-2006; the 1994-2004 holdout cannot be used here).

Base: the Phase 21 spin-off events (enter 21 trading days after first trade, hold 12 months, same costs).
Signals known before the entry date (using filings dated before entry):
  I1 insider buying: officer/director open-market purchases totalling >= $50K between the first trading day
     and the day before entry, or in the 90 days before entry (whichever window is longer).
  X1 insider selling: open-market sales by insiders totalling >= $100K in the same window.
  S1 high short interest: latest Compustat short interest before entry >= 5% of shares outstanding.
Add-on signals after entry (a SECOND, separate entry on the same spin-off, next open after the filing date,
held 12 months; only within the first 12 months after first trade):
  I2 later insider buying: officer/director purchases totalling >= $100K within a 30-day window.
  D1 activist / large holder: first Schedule 13D filed on the spin-off.
Reported: per-trade 12-month excess over SPY (stock return x (1 - cost) minus SPY): count, mean, median,
share positive, month-clustered t; for I1/X1/S1 also the complement (spin-offs without the signal) and the
difference. Stand-alone books for each group: CAGR, Sharpe, average positions.
Adoption rule (pre-written): a pre-entry filter is adopted only if spin-offs WITH it beat those WITHOUT it by
>= 5 points per trade with clustered t >= 2 on the difference, and the filtered book still holds >= 6 positions
on average. An add-on signal is adopted only if its trades beat SPY with clustered t >= 2 and beat the base
spin-off trades on average. Anything else is reported as "no evidence".
"""
import numpy as np
import pandas as pd

from .common import CACHE, DATA, RESULTS, get_logger, load_config

LOG = get_logger('spinoff_signals')


def clustered(ex, months):
    ex = pd.Series(np.asarray(ex, float))
    m = pd.Series(np.asarray(months))
    ok = ex.notna().values
    ex, m = ex[ok].reset_index(drop=True), m[ok].reset_index(drop=True)
    if len(ex) < 5:
        return ex.mean(), np.nan, len(ex)
    g = (ex - ex.mean()).groupby(m).sum()
    return ex.mean(), ex.mean() / (np.sqrt((g ** 2).sum()) / len(ex)), len(ex)


def run(cfg=None):
    cfg = cfg or load_config()
    from .combos import SubsetEngine
    from .compustat import wrds_dir
    from .filters import Market
    from .forced_flows import next_day
    from .prices import load_price
    from .strategies import perf
    spy_px = load_price('SPY')
    idx = pd.DatetimeIndex(spy_px.index.values)
    spy = spy_px['adj_close']
    spyr = spy.pct_change().reindex(idx).fillna(0.0)
    from .analysis import load_factors
    rf = load_factors(cfg)['RF'].reindex(idx).ffill().fillna(0.0)
    market = Market(idx.values)
    E = pd.read_parquet(DATA / 'spinoff_events.parquet')
    E['gvkey'] = E['gvkey'].astype(str)
    E['entry_date'] = pd.to_datetime(E['entry_date'])
    E['event'] = pd.to_datetime(E['event'])
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'cik'])
    co['gvkey'] = co['gvkey'].astype(str)
    co['cik'] = pd.to_numeric(co['cik'], errors='coerce')
    E = E.merge(co, on='gvkey', how='left')
    # insider purchases (officer/director) and sales
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['accession_number', 'issuer_cik', 'filing_date', 'value_clean'])
    it = pd.read_parquet(DATA / 'insider_trades.parquet', columns=['accession_number', 'is_director', 'is_officer'])
    pc = pc.merge(it, on='accession_number')
    pc = pc[pc['is_director'] | pc['is_officer']]
    pc['filing_date'] = pd.to_datetime(pc['filing_date']).dt.normalize()
    sales = pd.concat([pd.read_parquet(p, columns=['issuer_cik', 'filing_date', 'sale_value'])
                       for p in sorted((CACHE / 'sec_extra').glob('*q?.parquet'))])
    sales['filing_date'] = pd.to_datetime(sales['filing_date']).dt.normalize()
    buys_by = {int(k): g.sort_values('filing_date') for k, g in pc.groupby('issuer_cik')}
    sells_by = {int(k): g for k, g in sales.groupby('issuer_cik')}
    si = pd.read_parquet(wrds_dir(cfg) / 'compustat_shortint.parquet', columns=['gvkey', 'datadate', 'shortint'])
    si['gvkey'] = si['gvkey'].astype(str)
    si['datadate'] = pd.to_datetime(si['datadate'])
    si_by = {k: g.sort_values('datadate') for k, g in si.groupby('gvkey')}
    d13 = pd.read_parquet(CACHE / 'sec_13d.parquet', columns=['cik', 'filing_date'])
    d13['filing_date'] = pd.to_datetime(d13['filing_date'])
    d13_by = {int(k): g['filing_date'].sort_values() for k, g in d13.groupby('cik')}
    I1, X1, S1, addons = [], [], [], []
    for _, e in E.iterrows():
        lo = min(e['event'], e['entry_date'] - pd.Timedelta(days=90))
        ck = int(e['cik']) if pd.notna(e['cik']) else -1
        b = buys_by.get(ck)
        s = sells_by.get(ck)
        I1.append(bool(b is not None and b[(b['filing_date'] >= lo) & (b['filing_date'] < e['entry_date'])]['value_clean'].sum() >= 50_000))
        X1.append(bool(s is not None and s[(s['filing_date'] >= lo) & (s['filing_date'] < e['entry_date'])]['sale_value'].sum() >= 100_000))
        g = si_by.get(e['gvkey'])
        px = load_price(e['price_ticker'])
        sh = px.loc[:e['entry_date']]['shares_out'].dropna()
        val = np.nan
        if g is not None and len(sh):
            w = g[g['datadate'] < e['entry_date']]
            if len(w):
                val = w['shortint'].iloc[-1] / sh.iloc[-1]
        S1.append(val)
        end = e['event'] + pd.Timedelta(days=365)
        if b is not None:
            w = b[(b['filing_date'] > e['event']) & (b['filing_date'] <= end)]
            d, v = w['filing_date'].values, w['value_clean'].values
            for i in range(len(w)):
                j = np.searchsorted(d, d[i] - np.timedelta64(29, 'D'), 'left')
                if v[j:i + 1].sum() >= 100_000:
                    addons.append({'signal': 'I2 later insider buying', 'price_ticker': e['price_ticker'],
                                   'entry_date': next_day(idx, pd.Timestamp(d[i]), 1), 'cost': e['cost']})
                    break
        f = d13_by.get(ck)
        if f is not None:
            f = f[(f > e['event']) & (f <= end)]
            if len(f):
                addons.append({'signal': 'D1 13D filed', 'price_ticker': e['price_ticker'],
                               'entry_date': next_day(idx, f.iloc[0], 1), 'cost': e['cost']})
    E['I1'], E['X1'], E['si'] = I1, X1, S1
    E['S1'] = E['si'] >= 0.05
    A = pd.DataFrame(addons).dropna(subset=['entry_date'])

    def excess(df):
        out = []
        for _, e in df.iterrows():
            a = load_price(e['price_ticker'])['adj_close'].loc[e['entry_date']:].iloc[:253]
            s = spy.loc[e['entry_date']:].iloc[:253]
            out.append((a.iloc[-1] / a.iloc[0] - 1) * (1 - e['cost']) - (s.iloc[-1] / s.iloc[0] - 1) if len(a) > 1 and len(s) > 1 else np.nan)
        return np.array(out)

    E['ex'] = excess(E)
    E['month'] = E['entry_date'].dt.to_period('M').astype(str)
    if len(A):
        A['ex'] = excess(A)
        A['month'] = pd.to_datetime(A['entry_date']).dt.to_period('M').astype(str)

    def book_stats(df):
        df = df.reset_index(drop=True)
        if len(df) < 5:
            return {}
        eng = SubsetEngine(df, market, 252, df['cost'].values)
        mask = np.ones(len(df))
        act = pd.Series(eng.active(mask), index=idx)
        p = eng.returns(mask, rf.values).reindex(idx).where(act, spyr).loc[act[act].index[0]:]
        cnt = pd.Series(np.asarray((eng.den > 0).sum(axis=0)).ravel(), index=idx).loc[p.index]
        s = perf(p, rf)
        return {'book CAGR': s.get('CAGR'), 'book Sharpe': s.get('Sharpe'), 'avg positions': float(cnt[cnt > 0].mean())}

    rows, adopted = [], []
    base_mean, base_t, base_n = clustered(E['ex'], E['month'])
    rows.append({'group': 'all spin-offs (base)', 'n': base_n, 'mean excess': base_mean, 'median': np.nanmedian(E['ex']),
                 'share > 0': np.nanmean(E['ex'] > 0), 't': base_t, **book_stats(E)})
    for sig, label in (('I1', 'I1 insider buying before entry'), ('X1', 'X1 insider selling before entry'),
                       ('S1', 'S1 short interest >= 5%')):
        known = E['si'].notna() if sig == 'S1' else pd.Series(True, index=E.index)
        w, wo = E[known & E[sig]], E[known & ~E[sig]]
        mw, tw, nw = clustered(w['ex'], w['month'])
        mo, to, no = clustered(wo['ex'], wo['month'])
        dm, dt, _ = clustered(w['ex'] - mo, w['month'])
        bs = book_stats(w)
        rows.append({'group': label, 'n': nw, 'mean excess': mw, 'median': np.nanmedian(w['ex']) if nw else np.nan,
                     'share > 0': np.nanmean(w['ex'] > 0) if nw else np.nan, 't': tw,
                     'without signal: n': no, 'without: mean excess': mo, 'difference': mw - mo, 't (difference)': dt, **bs})
        if nw and mw - mo >= 0.05 and dt >= 2 and bs.get('avg positions', 0) >= 6:
            adopted.append(label)
    for sig in (A['signal'].unique() if len(A) else []):
        g = A[A['signal'] == sig]
        m, t, n = clustered(g['ex'], g['month'])
        rows.append({'group': sig + ' (add-on entry)', 'n': n, 'mean excess': m, 'median': np.nanmedian(g['ex']),
                     'share > 0': np.nanmean(g['ex'] > 0), 't': t, **book_stats(g)})
        if t >= 2 and m > base_mean:
            adopted.append(sig)
    R = pd.DataFrame(rows)
    R['adopted (pre-written rule)'] = ', '.join(adopted) or 'none'
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.to_csv(RESULTS / 'spinoff_signals.csv', index=False)
    LOG.info(f'signals: done; adopted: {adopted or "none"}')
    return R


if __name__ == '__main__':
    run()
