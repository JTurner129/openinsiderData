"""Spin-off scanner: tells you which new US spin-offs to buy, which you hold, and when to sell each one.

Rules (exactly the ones backtested in Phases 19-24; see backtest/spinoff_core.py):
  * A spin-off = a company that filed an original SEC Form 10-12B and started trading within a year after it.
  * BUY at the open 21 trading days after its first trading day (after the initial forced selling),
    only if the share price is >= $1 and it trades >= $100K a day on average over the prior 20 days.
  * SELL at the open 252 trading days (about 12 months) after the buy. No other exits.
  * Sizing: each new spin-off gets 1/8 of the TOTAL account value on its buy day. Money not in spin-offs sits
    in an index fund (VUAG by default). A buy is paid for by selling the index fund; if the index fund holds
    less than that, sell all of it and trim the rest from the spin-offs already held, in proportion to their
    current values. Sale proceeds go back into the index fund. No monthly rebalancing.

Run it once a day or a few times a week:

    python live_spinoffs.py              # look back over Form 10-12B filings from the last 2 years
    python live_spinoffs.py --years 3

Output: a table on screen and live/spinoffs.csv. Status is one of
  waiting      registered, not trading yet
  watch        trading, buy date not reached yet
  BUY          buy window open: the buy day (trading day 21) or one of the 4 trading days after it
  skip         failed the price / volume check on the buy date
  HOLD         buy window over, sell date not reached (if you did not buy it, it is missed: do not buy late)
  SELL         sell date is today or passed within the last 10 days
  done         sold (12-month hold over)
  not_new      traded before the Form 10 or over a year after it
Trading days follow the NYSE calendar (nyse_calendar.py: real days from SPY's history, NYSE holiday rules ahead).
The classification is done by the daily engine behind the dashboard (spinoff_tracker.py, same code), which also
adds your own trades and alerts. This command is the plain table view of the same pipeline.
Check each new name by hand before buying: a few Form 10 filers are not true spin-offs (for example a
company re-registering after bankruptcy), and some small ones may not be available on Trading 212.

Data: SEC EDGAR (filing index, company tickers) and Yahoo Finance prices. No WRDS needed.
"""
import argparse
import time
from datetime import date
from pathlib import Path

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


def _get(s, url, timeout):
    """GET with up to 3 tries on network errors and HTTP errors other than 404."""
    for attempt in range(3):
        try:
            r = s.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == 2:
                raise
        else:
            if r.status_code in (200, 404) or attempt == 2:
                return r
        time.sleep(2 * (attempt + 1))


def form10_filings(s, years, today=None, errors=None):
    """Original Form 10-12B filings from the last `years` years (EDGAR quarterly form.idx files), one row per CIK
    (its first filing in the window). Quarters that cannot be fetched are added to `errors` when given."""
    today = today or date.today()
    rows = []
    for y in range(today.year - years, today.year + 1):
        for q in range(1, 5):
            if y == today.year and q > (today.month - 1) // 3 + 1:
                continue
            url = f'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx'
            r = _get(s, url, 120)
            if r.status_code != 200:
                # a new quarter's index can be missing for its first days (no filings yet)
                new_quarter = y == today.year and q == (today.month - 1) // 3 + 1 and \
                    (today - date(y, 3 * q - 2, 1)).days < 7
                if errors is not None and not new_quarter:
                    errors.append(f'SEC form.idx {y} Q{q}: HTTP {r.status_code}')
                continue
            for line in r.text.splitlines():
                if line[:17].strip() == '10-12B':
                    p = line.split()
                    rows.append({'cik': int(p[-3]), 'form10_date': pd.Timestamp(p[-2]),
                                 'company': ' '.join(p[1:-3]), 'filename': p[-1]})
            time.sleep(0.2)
    f = pd.DataFrame(rows)
    if f.empty:
        return f
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=years)
    f = f[(f['form10_date'] >= cutoff) & (f['form10_date'] <= pd.Timestamp(today))]
    return f.sort_values('form10_date').drop_duplicates('cik')


def tickers(s):
    j = _get(s, 'https://www.sec.gov/files/company_tickers_exchange.json', 60).json()
    t = pd.DataFrame(j['data'], columns=j['fields'])
    t = t[t['exchange'].isin(['NYSE', 'Nasdaq', 'NYSE American', 'CBOE'])]
    return t.drop_duplicates('cik').set_index('cik')[['ticker', 'exchange', 'name']]   # name = current name


def run(years, out=OUT, today=None, now=None):
    """Classify every Form 10-12B filer of the last `years` years with the daily engine's own code
    (spinoff_tracker.build_pipeline / classify: same calendar, same $1 / $100K window, today's unfinished price
    bar left out) and save the table to `out`. today/now: for tests (default: the New York date and clock now)."""
    from datetime import datetime, timedelta, timezone
    import nyse_calendar as nc
    import spinoff_tracker as st
    now = now or datetime.now(timezone.utc)
    today = today or now.astimezone(st.NY).date()
    us_cut = st.final_bar_dates(today, now)[0]
    filings = st.sec_filings(years, today, [])
    fetch = st.Fetcher()
    cal_from = today - timedelta(days=366 * years + 20)
    spy, _ = fetch('SPY', cal_from, us_cut)
    cal = nc.build_calendar(cal_from, today + timedelta(days=800), [d.date() for d in spy.index])[0]
    items, _, failures = st.build_pipeline(filings, today, cal, us_cut, {}, fetch)
    for x in failures:
        print(f'Warning: no prices for {x["ticker"]} today ({x["why"]})')
    rows = []
    for it in items:
        row = {'company': it['company'], 'cik': it['id'], 'form10_date': it['form10_date'], 'ticker': it['ticker'],
               'exchange': it['exchange'], 'first_trade': it['first_trade'], 'buy_date': it['buy_date'],
               'buy_window_end': it['buy_window_end'], 'sell_date': it['model_sell_date'],
               'buy_open': it['buy_open'], 'last_close': it['last_close'], 'price_before_buy': it['price_before_buy'],
               'avg_dollar_volume': it['avg_dollar_volume'], 'status': it['status'], 'note': it['note']}
        if it['buy_open'] and it['last_close'] and it['status'] in ('BUY', 'HOLD', 'SELL', 'done'):
            row['return_since_buy'] = f"{it['last_close'] / it['buy_open'] - 1:+.1%}"
        rows.append(row)
    out_df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out_df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, default=2)
    args = ap.parse_args()
    out = run(args.years)
    show = out[out['status'].isin(['BUY', 'SELL', 'HOLD', 'watch'])] if len(out) else out
    cols = [c for c in ('status', 'ticker', 'company', 'first_trade', 'buy_date', 'sell_date', 'buy_open',
                        'last_close', 'return_since_buy') if c in show]
    pd.set_option('display.width', 200)
    print(show[cols].to_string(index=False) if len(show) else 'No spin-offs to act on or watch right now.')
    print(f'\nFull list ({len(out)} Form 10 filers) saved to {OUT}')
