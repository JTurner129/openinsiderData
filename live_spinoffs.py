"""Spin-off scanner: tells you which new US spin-offs to buy, which you hold, and when to sell each one.

Rules (exactly the ones backtested in Phases 19-24; see backtest/spinoff_core.py):
  * A spin-off = a company that filed an original SEC Form 10-12B and started trading within a year after it.
  * BUY at the open 21 trading days after its first trading day (after the initial forced selling),
    only if the share price is >= $1 and it trades >= $100K a day on average over the prior 20 days.
  * SELL at the open 252 trading days (about 12 months) after the buy. No other exits.
  * Buy every qualifying spin-off at the same size. Keep the spin-off slice at about 30% of the account
    (70% in an index fund), topping up or trimming once a month.

Run it once a day or a few times a week:

    python live_spinoffs.py              # look back over Form 10-12B filings from the last 2 years
    python live_spinoffs.py --years 3

Output: a table on screen and live/spinoffs.csv. Status is one of
  waiting      registered, not trading yet
  watch        trading, buy date not reached yet
  BUY          buy date is today or passed within the last 5 trading days (buy at the next open)
  missed       buy date passed more than 5 trading days ago without a buy logged (your call)
  skip         failed the price / volume check on the buy date
  HOLD         bought, sell date not reached
  SELL         sell date is today or passed
Check each new name by hand before buying: a few Form 10 filers are not true spin-offs (for example a
company re-registering after bankruptcy), and some small ones may not be available on Trading 212.

Data: SEC EDGAR (filing index, company tickers) and Yahoo Finance prices. No WRDS needed.
"""
import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / 'live'
OUT = LIVE / 'spinoffs.csv'
DELAY, HOLD, BUY_WINDOW = 21, 252, 5


def sec_session():
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    s = requests.Session()
    s.headers['User-Agent'] = cfg['sec']['user_agent']
    return s


def form10_filings(s, years):
    today = date.today()
    rows = []
    for y in range(today.year - years, today.year + 1):
        for q in range(1, 5):
            if y == today.year and q > (today.month - 1) // 3 + 1:
                continue
            r = s.get(f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx', timeout=120)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if line[:17].strip() == '10-12B':
                    p = line.split()
                    rows.append({'cik': int(p[-3]), 'form10_date': pd.Timestamp(p[-2]),
                                 'company': ' '.join(p[1:-3])})
            time.sleep(0.2)
    f = pd.DataFrame(rows)
    if f.empty:
        return f
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=years)
    return f[f['form10_date'] >= cutoff].sort_values('form10_date').drop_duplicates('cik')


def tickers(s):
    j = s.get('https://www.sec.gov/files/company_tickers_exchange.json', timeout=60).json()
    t = pd.DataFrame(j['data'], columns=j['fields'])
    t = t[t['exchange'].isin(['NYSE', 'Nasdaq', 'NYSE American', 'CBOE'])]
    return t.drop_duplicates('cik').set_index('cik')[['ticker', 'exchange']]


def trading_days(start, n):
    """The n-th NYSE trading day after start (n=0 -> start). Past days come from SPY's history; future days
    are estimated with US federal holidays (off by a day around Good Friday; recheck the week before)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    import yfinance as yf
    days = pd.DatetimeIndex([])
    if start <= pd.Timestamp(date.today()):
        spy = yf.download('SPY', start=start - pd.Timedelta(days=5), progress=False, auto_adjust=True)
        days = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
        days = days[days >= start]
    if len(days) > n:
        return days[n]
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    last = days[-1] if len(days) else start
    return last + (n - len(days) + 1) * bday


def run(years):
    import yfinance as yf
    s = sec_session()
    F = form10_filings(s, years)
    T = tickers(s)
    F = F.join(T, on='cik', how='left')
    today = pd.Timestamp(date.today())
    rows = []
    for _, f in F.iterrows():
        row = {'company': f['company'], 'cik': f['cik'], 'form10_date': f['form10_date'].date(),
               'ticker': f.get('ticker'), 'exchange': f.get('exchange')}
        if not isinstance(f.get('ticker'), str):
            rows.append({**row, 'status': 'waiting', 'note': 'no exchange ticker yet'})
            continue
        h = yf.download(f['ticker'], period='max', progress=False, auto_adjust=False)
        if h is None or h.empty:
            rows.append({**row, 'status': 'waiting', 'note': 'no price history yet'})
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        first = h.index[0]
        if first < f['form10_date'] or first > f['form10_date'] + pd.Timedelta(days=365):
            rows.append({**row, 'first_trade': first.date(), 'status': 'not a new listing',
                         'note': 'traded before the Form 10 or over a year after it'})
            continue
        buy = h.index[DELAY] if len(h) > DELAY else trading_days(first, DELAY)
        sell = trading_days(buy, HOLD)
        row.update({'first_trade': first.date(), 'buy_date': buy.date(), 'sell_date': sell.date(),
                    'last_close': round(float(h['Close'].iloc[-1]), 2)})
        if today < buy:
            row['status'] = 'watch'
        else:
            w = h.loc[:buy].iloc[:-1].tail(20)
            price, adv = float(w['Close'].iloc[-1]), float((w['Close'] * w['Volume']).mean())
            row.update({'price_before_buy': round(price, 2), 'avg_dollar_volume': round(adv)})
            if price < 1 or adv < 100_000:
                row['status'] = 'skip'
            else:
                row['buy_open'] = round(float(h.loc[buy, 'Open']), 2) if buy in h.index else None
                if today >= sell:
                    row['status'] = 'SELL' if today <= sell + pd.Timedelta(days=10) else 'done'
                elif len(h.loc[buy:]) <= BUY_WINDOW:
                    row['status'] = 'BUY'
                else:
                    row['status'] = 'HOLD'
                if row.get('buy_open'):
                    row['return_since_buy'] = f"{float(h['Close'].iloc[-1]) / row['buy_open'] - 1:+.1%}"
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {'BUY': 0, 'SELL': 1, 'HOLD': 2, 'watch': 3, 'waiting': 4, 'skip': 5, 'done': 6, 'not a new listing': 7}
    out = out.sort_values(['status', 'form10_date'], key=lambda c: c.map(order) if c.name == 'status' else c)
    LIVE.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, default=2)
    args = ap.parse_args()
    out = run(args.years)
    show = out[out['status'].isin(['BUY', 'SELL', 'HOLD', 'watch'])]
    cols = [c for c in ('status', 'ticker', 'company', 'first_trade', 'buy_date', 'sell_date', 'buy_open',
                        'last_close', 'return_since_buy') if c in show]
    pd.set_option('display.width', 200)
    print(show[cols].to_string(index=False) if len(show) else 'No spin-offs to act on or watch right now.')
    print(f'\nFull list ({len(out)} Form 10 filers) saved to {OUT}')
