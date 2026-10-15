"""Phase 18: can we predict when small/value beats large/growth (and vice versa)? Tested on 60+ years.

Rules were written down and committed before any data was downloaded.

Data (free): Kenneth French "6 Portfolios Formed on Size and Book-to-Market (2 x 3)", monthly value-weighted
returns and annual Sum BE / Sum ME; FRED monthly BAA, AAA, GS10, TB3MS, CPIAUCSL.
Target at each month-end t: return of SMALL HiBM minus BIG LoBM over months t+1..t+12 (small value minus
large growth).
Predictors, each lagged so it was public at t (FRED series use the value from month t-1; CPI from t-2; the
value spread uses the June breakpoint data for portfolios formed in June, available from July):
  term spread GS10 - TB3MS;  credit spread BAA - AAA;  12-month change in GS10;  CPI inflation (12 months);
  value spread = log(BE/ME of SMALL HiBM / BE/ME of BIG LoBM) (French annual sums);
  style momentum = SMALL HiBM minus BIG LoBM return over the prior 12 months.
Model: each January from 1975, fit an ordinary least-squares regression of the 12-month target on the six
predictors (each standardised with that window's mean and sd), using only months whose 12-month target is
fully known by then (expanding window from July 1963). Predict monthly for the next 12 months.
Rule: hold SMALL HiBM when the prediction is > 0, else BIG LoBM; re-decided monthly; 0.5% cost per switch.
Out-of-sample scoring 1990-2026 (also split 1990-2007 / 2008-2026): CAGR, Sharpe, max drawdown against
always-small-value, always-large-growth, 50/50 monthly rebalanced, and the market; hit rate of the sign;
out-of-sample R-squared against the expanding historical mean (Campbell & Thompson 2008).
Diagnostics: the same for each predictor alone.
Only if the six-predictor rule beats 50/50 on Sharpe in BOTH 1990-2007 and 2008-2026 with out-of-sample
R-squared > 0 will it be applied to our own strategies (a follow-up with its own pre-written rules).
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from .common import CACHE, RESULTS, get_logger

LOG = get_logger('style_timing')
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_2x3_CSV.zip'
FF3 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'
SWITCH_COST = 0.005


def _french_sections(url):
    raw = requests.get(url, timeout=120).content
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode('latin-1')
    return text


def _parse_block(text, title_regex, monthly=True):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if re.search(title_regex, l))
    hdr = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith(','))
    cols = [c.strip() for c in lines[hdr].split(',')[1:]]
    rows = []
    for l in lines[hdr + 1:]:
        parts = [p.strip() for p in l.split(',')]
        if not parts[0] or not parts[0].isdigit():
            break
        if monthly and len(parts[0]) != 6:
            break
        rows.append([parts[0]] + [float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=['date'] + cols)
    if monthly:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        df['date'] = df['date'].astype(int)
    return df.set_index('date')


def load_data():
    path = CACHE / 'style_timing_data.parquet'
    if path.exists():
        return pd.read_parquet(path)
    t = _french_sections(FRENCH)
    ret = _parse_block(t, r'Average Value Weighted Returns -- Monthly') / 100
    bm = _parse_block(t, r'Sum of BE / Sum of ME', monthly=False)
    ff = _parse_block(_french_sections(FF3), r'Mkt-RF', monthly=True) / 100 if False else None
    t3 = _french_sections(FF3)
    lines = t3.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith(',Mkt-RF'))
    rows = []
    for l in lines[hdr + 1:]:
        p = [x.strip() for x in l.split(',')]
        if not p[0].isdigit() or len(p[0]) != 6:
            break
        rows.append([p[0]] + [float(x) for x in p[1:5]])
    ff = pd.DataFrame(rows, columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
    ff['date'] = pd.to_datetime(ff['date'], format='%Y%m') + pd.offsets.MonthEnd(0)
    ff = ff.set_index('date') / 100
    fred = {}
    for s in ('BAA', 'AAA', 'GS10', 'TB3MS', 'CPIAUCSL'):
        d = pd.read_csv(io.StringIO(requests.get(FRED.format(s), timeout=120).text))
        d.columns = ['date', s]
        d['date'] = pd.to_datetime(d['date']) + pd.offsets.MonthEnd(0)
        fred[s] = pd.to_numeric(d.set_index('date')[s], errors='coerce')
    F = pd.DataFrame(fred)
    df = pd.DataFrame({'SV': ret['SMALL HiBM'], 'BG': ret['BIG LoBM'], 'SG': ret['SMALL LoBM'], 'BV': ret['BIG HiBM']})
    df = df.join(ff, how='left').join(F, how='left')
    # value spread from June sums, usable from July of that year to June of the next
    vs = np.log(bm['SMALL HiBM'] / bm['BIG LoBM'])
    df['value_spread'] = [vs.get(d.year if d.month >= 7 else d.year - 1, np.nan) for d in df.index]
    df.to_parquet(path)
    return df


def build(df):
    X = pd.DataFrame(index=df.index)
    X['term'] = (df['GS10'] - df['TB3MS']).shift(1)
    X['credit'] = (df['BAA'] - df['AAA']).shift(1)
    X['rate_chg'] = (df['GS10'] - df['GS10'].shift(12)).shift(1)
    X['inflation'] = (df['CPIAUCSL'] / df['CPIAUCSL'].shift(12) - 1).shift(2)
    X['value_spread'] = df['value_spread']
    spread = df['SV'] - df['BG']
    X['style_mom'] = (1 + df['SV']).rolling(12).apply(np.prod, raw=True) - (1 + df['BG']).rolling(12).apply(np.prod, raw=True)
    fwd = (1 + df['SV']).rolling(12).apply(np.prod, raw=True).shift(-12) - \
        (1 + df['BG']).rolling(12).apply(np.prod, raw=True).shift(-12)
    return X, fwd


def walk_forward(X, y, cols, start_year=1975):
    pred = pd.Series(np.nan, index=X.index)
    hist_mean = pd.Series(np.nan, index=X.index)
    for yr in range(start_year, X.index[-1].year + 1):
        t0 = pd.Timestamp(f'{yr}-01-31')
        known = X.index[X.index <= t0 - pd.offsets.MonthEnd(12)]       # targets fully realised by t0
        tr = pd.concat([X.loc[known, cols], y.loc[known]], axis=1).dropna()
        if len(tr) < 60:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, np.nan)
        Z = (tr[cols] - mu) / sd
        A = np.column_stack([np.ones(len(Z)), Z.values])
        beta, *_ = np.linalg.lstsq(A, tr.iloc[:, -1].values, rcond=None)
        idx = X.index[(X.index >= t0) & (X.index < pd.Timestamp(f'{yr + 1}-01-31'))]
        Zt = ((X.loc[idx, cols] - mu) / sd)
        pred.loc[idx] = beta[0] + Zt.values @ beta[1:]
        hist_mean.loc[idx] = tr.iloc[:, -1].mean()
    return pred, hist_mean


def strategy(df, pred):
    hold_sv = (pred > 0).shift(1)                     # decided at t, held during t+1
    r = np.where(hold_sv == True, df['SV'], np.where(hold_sv == False, df['BG'], np.nan))
    r = pd.Series(r, index=df.index)
    switch = hold_sv.ne(hold_sv.shift(1)) & hold_sv.notna() & hold_sv.shift(1).notna()
    return (r - switch * SWITCH_COST).dropna()


def stats(r, rf):
    r = r.dropna()
    if len(r) < 24:
        return {}
    w = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    return {'CAGR': w.iloc[-1] ** (12 / len(r)) - 1, 'Sharpe': ex.mean() / ex.std() * np.sqrt(12),
            'max DD': (w / w.cummax() - 1).min()}


def run():
    df = load_data()
    X, y = build(df)
    cols_all = ['term', 'credit', 'rate_chg', 'inflation', 'value_spread', 'style_mom']
    rows = []
    mkt = df['Mkt-RF'] + df['RF']
    fifty = 0.5 * df['SV'] + 0.5 * df['BG']
    periods = (('1990–2026', '1990-01-01', None), ('1990–2007', '1990-01-01', '2007-12-31'), ('2008–2026', '2008-01-01', None))
    for name, cols in [('ALL SIX', cols_all)] + [(c, [c]) for c in cols_all]:
        pred, hm = walk_forward(X, y, cols)
        strat = strategy(df, pred)
        for lab, a, b in periods:
            sl = slice(a, b)
            yy, pp, hh = y.loc[sl], pred.loc[sl], hm.loc[sl]
            ok = yy.notna() & pp.notna() & hh.notna()
            r2 = 1 - ((yy[ok] - pp[ok]) ** 2).sum() / ((yy[ok] - hh[ok]) ** 2).sum() if ok.sum() > 24 else np.nan
            hit = ((yy[ok] > 0) == (pp[ok] > 0)).mean() if ok.sum() else np.nan
            row = {'predictors': name, 'period': lab, 'OOS R2': r2, 'hit rate': hit,
                   'share in small/value': float((pp[ok] > 0).mean()) if ok.sum() else np.nan}
            for k, s in (('timing', strat), ('always small-value', df['SV']), ('always large-growth', df['BG']),
                         ('50/50', fifty), ('market', mkt)):
                st = stats(s.loc[sl], df['RF'])
                row[f'{k} CAGR'], row[f'{k} Sharpe'] = st.get('CAGR'), st.get('Sharpe')
                if k == 'timing':
                    row['timing max DD'] = st.get('max DD')
            rows.append(row)
        LOG.info(f'style timing: {name} done')
    r = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'style_timing.csv', index=False)
    passed = r[(r['predictors'] == 'ALL SIX')]
    ok = all(passed.loc[passed['period'] == p, 'timing Sharpe'].iloc[0] > passed.loc[passed['period'] == p, '50/50 Sharpe'].iloc[0]
             for p in ('1990–2007', '2008–2026')) and passed.loc[passed['period'] == '1990–2026', 'OOS R2'].iloc[0] > 0
    LOG.info(f'style timing: pre-registered pass condition met = {ok}')
    return r, ok


if __name__ == '__main__':
    run()
