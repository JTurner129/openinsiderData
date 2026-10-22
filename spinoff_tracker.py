"""Daily engine for the spin-off tracker dashboard (a claude.ai Artifact with a document database).

    python spinoff_tracker.py scan --out DIR [--db EXPORTS] [--years 2] [--today YYYY-MM-DD]
    python spinoff_tracker.py writes --out DIR --versions LISTINGS.txt

scan
    Finds US spin-offs (an original SEC Form 10-12B, then a first trade on NYSE / Nasdaq / NYSE American / CBOE
    within 365 days), applies the backtested rules and writes the database documents the page reads:
      DIR/docs/market/status.json, DIR/docs/market/pipeline.json, DIR/docs/market/calendar.json,
      DIR/docs/prices/<TICKER>.json, DIR/docs/prices/{VUAG,EQQQ,GBPUSD}.json
    plus DIR/alerts.json (today's actions, personalised with the exported trades / deposits / settings / checks)
    and DIR/needs_check.json (spin-offs whose Form 10 nobody has checked yet).
    --db is a folder of exported documents: <collection>/<doc_id>.json for trades, deposits, settings, checks
    and market/pipeline.json (yesterday's pipeline). Export market too: when the SEC scan fails, yesterday's
    list is used for the alerts, and when Yahoo Finance cannot give a spin-off's prices, its dates and status
    are carried forward from yesterday's item instead of being shown as "waiting".
    Trades and deposits dated after today are ignored until their date (and listed in market/status errors).
    --today simulates a run made before the US open on that New York date (prices are cut to the day before).
    Without --today the alerts follow the clock: after the US close they point to the next trading day.

writes
    Turns DIR/docs into batch files DIR/writes_1.json, writes_2.json, ... for the database tool, adding
    if_version for documents that already exist (from the database's listing text in LISTINGS.txt).

The rules (backtested; do not change them here):
  * Day 0 = first trading day. BUY at the open on NYSE trading day 21 if, over the 20 trading days before it,
    the last close is >= $1 and the average daily dollar volume (close x volume) is >= $100,000.
    The buy window is that day and the 4 trading days after it; after that the spin-off is missed.
  * SELL at the open 252 NYSE trading days after the day you actually bought. No other exits.
  * Each new spin-off gets 1/8 (setting "slice") of the total account; it is paid for by selling the index fund
    (VUAG or EQQQ) and, if that is not enough, by trimming the spin-offs held in proportion to their value.
"""
import argparse
import json
import logging
import math
import re
import shutil
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import live_spinoffs as ls
import nyse_calendar as nc

NY, LONDON = ZoneInfo('America/New_York'), ZoneInfo('Europe/London')
DELAY, WINDOW, HOLD = ls.DELAY, ls.BUY_WINDOW, ls.HOLD      # 21, 5, 252 trading days
MIN_PRICE, MIN_ADV = 1.0, 100_000
SELL_GRACE = 10          # calendar days a model sell stays "SELL" before it becomes "done"
DONE_PRICES = 60         # keep publishing a done spin-off's prices this many days after its model sell
CAL_AHEAD = 600          # the calendar document reaches this many calendar days past today
LOOKBACK = 400           # the engine starts at most this many days back (earlier if the user's data starts earlier)
MODEL_START = date(2026, 10, 13)  # the page's model account starts 2026-10-20; keep fund prices from a week before
EPS = 1e-6
DUST = 0.005             # a sell that leaves less than 0.5% of the shares closes the position (typing rounding)
MIN_DATE = date(2000, 1, 1)   # earlier trade / deposit dates are typing errors (e.g. year 26 -> 0026)
MAX_YEARS_BACK = 15      # the engine never starts more than this many years before today
FX_COST = 0.0015         # Trading 212 FX fee on US stocks, each way
ACTIVE = ('watch', 'BUY', 'HOLD', 'SELL')
ORDER = {s: i for i, s in enumerate(['BUY', 'SELL', 'HOLD', 'watch', 'waiting', 'skip', 'done', 'not_new'])}
INDEX = {'VUAG': ('VUAG.L', 'Close', 1.0),          # accumulating, quoted in GBP
         'EQQQ': ('EQQQ.L', 'Adj Close', 0.01)}     # distributing (Adj Close adds dividends), quoted in pence
FX_ID, FX_SYMBOL = 'GBPUSD', 'GBPUSD=X'
RESERVED = set(INDEX) | {FX_ID}
DEFAULTS = {'index_fund': 'VUAG', 'slice': 8, 'monthly_deposit': 1666.66}
NEXT_CHECK = 'Weekdays 06:45 UK time'
US_FINAL, LSE_FINAL = dtime(16, 15), dtime(16, 45)  # local times after which the day's bar counts as final
ID_RE = re.compile(r'^[A-Z0-9.\-]+$')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DOC_LIMIT = 256 * 1024
MAX_ERRORS = 50
MAX_TICKER_FAILS = 2     # more spin-offs than this without prices today -> the pipeline is not written
YAHOO_BACKOFF = (5, 15, 30)   # seconds to wait before each retry of a failed Yahoo download
YAHOO_PAUSE = 0.4        # minimum seconds between two Yahoo downloads
YAHOO_BREAKER = 3        # after this many symbols in a row fail, Yahoo is treated as down for the rest of the run

logging.getLogger('yfinance').setLevel(logging.CRITICAL)


# ----------------------------------------------------------------------------------------------- small helpers
def iso(d):
    return d.isoformat() if d is not None else None


def r4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def fmt_day(d, today=None):
    """'Thu 5 Nov' (with the year when it is not today's year)."""
    s = f'{d:%a} {d.day} {d:%b}'
    return s if today is None or d.year == today.year else f'{s} {d.year}'


def us_open_uk(d):
    """UK clock time of the 09:30 New York open on date d, e.g. '14:30'."""
    return datetime.combine(d, dtime(9, 30), NY).astimezone(LONDON).strftime('%H:%M')


def money(x):
    return f'£{x:,.0f}'


def shares_txt(x):
    return f'{x:.6f}'.rstrip('0').rstrip('.')


SUFFIX = re.compile(r'(?:,?\s+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.l\.c|plc|n\.?v|'
                    r's\.?a|holdings?))+\.?\s*$', re.I)


def short_name(company):
    s = SUFFIX.sub('', (company or '').strip()).strip(' ,')
    return s or (company or '')


def norm_ticker(t):
    """One spelling for a ticker everywhere (trades, dismissed, pipeline, price documents): upper case, and class
    shares with a dash as SEC and Yahoo write them (BRK.B and brk-b -> BRK-B)."""
    return str(t).strip().upper().replace('.', '-')


def yahoo_symbol(ticker):
    return norm_ticker(ticker)


def _date(v, field):
    if isinstance(v, str) and DATE_RE.match(v.strip()):
        try:
            return date.fromisoformat(v.strip())
        except ValueError:
            pass
    raise ValueError(f'bad {field} {v!r}')


def _num(v, field, positive=False):
    if isinstance(v, bool):
        raise ValueError(f'bad {field} {v!r}')
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'bad {field} {v!r}') from None
    if not math.isfinite(x) or (positive and x <= 0):
        raise ValueError(f'bad {field} {v!r}')
    return x


# ----------------------------------------------------------------------------------------------- exported user data
def _docs(db, coll, errors):
    d = db / coll
    if not d.is_dir():
        return
    for f in sorted(d.glob('*.json')):
        try:
            body = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            errors.append(f'{coll}/{f.stem}: unreadable export skipped ({e.__class__.__name__})')
            continue
        if not isinstance(body, dict):
            errors.append(f'{coll}/{f.stem}: export is not a JSON object, skipped')
            continue
        yield f.stem, body


def _user_date(v):
    """A trade / deposit date; years before 2000 are typing errors (an HTML date input turns '26' into 0026)."""
    d = _date(v, 'date')
    if d < MIN_DATE:
        raise ValueError(f'date {d.isoformat()} is before {MIN_DATE.year}, probably a typing error')
    return d


def clean_trade(doc_id, b):
    typ = str(b.get('type', '')).strip().lower()
    if typ not in ('buy', 'sell'):
        raise ValueError(f'bad type {b.get("type")!r}')
    t = b.get('ticker')
    if not isinstance(t, str) or not t.strip():
        raise ValueError(f'bad ticker {t!r}')
    return {'id': doc_id, 'date': _user_date(b.get('date')), 'type': typ, 'ticker': norm_ticker(t),
            'shares': _num(b.get('shares'), 'shares', True), 'amount_gbp': _num(b.get('amount_gbp'), 'amount_gbp', True),
            'note': str(b.get('note') or ''), 'created_at': str(b.get('created_at') or '')}


def clean_deposit(doc_id, b):
    return {'id': doc_id, 'date': _user_date(b.get('date')), 'amount_gbp': _num(b.get('amount_gbp'), 'amount_gbp'),
            'note': str(b.get('note') or ''), 'created_at': str(b.get('created_at') or '')}


def as_of(data, today):
    """(data with only the trades and deposits dated on or before `today`, ['trades/<id> (date)', ...] left out).
    A future-dated record (a standing order logged ahead, a typo, or a later record in a --today run) must not
    count as held or as money in the account yet."""
    later = [f'{k}/{x["id"]} ({x["date"].isoformat()})' for k in ('trades', 'deposits') for x in data[k]
             if x['date'] > today]
    cur = dict(data, trades=[t for t in data['trades'] if t['date'] <= today],
               deposits=[d for d in data['deposits'] if d['date'] <= today])
    return cur, later


def clean_settings(b, errors):
    s = dict(DEFAULTS, dismissed={})
    fund = b.get('index_fund', s['index_fund'])
    if fund in INDEX:
        s['index_fund'] = fund
    else:
        errors.append(f'settings/main: unknown index_fund {fund!r}, using VUAG')
    sl = b.get('slice', s['slice'])
    try:
        ok = not isinstance(sl, bool) and float(sl) == int(float(sl)) and int(float(sl)) >= 1
    except (TypeError, ValueError):
        ok = False
    if ok:
        s['slice'] = int(float(sl))
    else:
        errors.append(f'settings/main: bad slice {sl!r}, using 8')
    try:
        s['monthly_deposit'] = _num(b.get('monthly_deposit', s['monthly_deposit']), 'monthly_deposit')
    except ValueError as e:
        errors.append(f'settings/main: {e}')
    dis = b.get('dismissed') or {}
    if isinstance(dis, dict):
        s['dismissed'] = {norm_ticker(k): str(v) for k, v in dis.items()}
    else:
        errors.append('settings/main: dismissed is not an object, ignored')
    return s


def cik_key(x):
    x = str(x).strip()
    return x.lstrip('0') or '0' if x.isdigit() else x


def load_exports(db, errors):
    """Exported documents -> {'trades', 'deposits', 'settings', 'checks', 'pipeline'}. Bad files are skipped with an
    error message; a missing folder or collection means no data."""
    data = {'trades': [], 'deposits': [], 'settings': dict(DEFAULTS, dismissed={}), 'checks': {}, 'pipeline': None}
    if not db:
        return data
    db = Path(db)
    if not db.is_dir():
        errors.append(f'export folder {db} not found; running without your data')
        return data
    for kind, fn in (('trades', clean_trade), ('deposits', clean_deposit)):
        for did, b in _docs(db, kind, errors):
            try:
                data[kind].append(fn(did, b))
            except ValueError as e:
                errors.append(f'{kind}/{did}: skipped ({e})')
    for did, b in _docs(db, 'settings', errors):
        if did == 'main':
            data['settings'] = clean_settings(b, errors)
    for did, b in _docs(db, 'checks', errors):
        g = b.get('genuine')
        if g not in (True, False, None):
            errors.append(f'checks/{did}: genuine {g!r} is not true/false/null, treated as null')
            g = None
        data['checks'][cik_key(did)] = {'genuine': g, 'parent': str(b.get('parent') or ''),
                                        'business': str(b.get('business') or ''), 'note': str(b.get('note') or '')}
    for did, b in _docs(db, 'market', errors):
        if did == 'pipeline' and isinstance(b.get('items'), list):
            data['pipeline'] = [x for x in b['items'] if isinstance(x, dict)]
    return data


# ----------------------------------------------------------------------------------------------- prices
class YahooError(Exception):
    """A Yahoo Finance download failed (network, rate limit, outage), as opposed to Yahoo having no prices."""


_last_yahoo_call = [0.0]


def yahoo_history(symbol, start=None):
    """Daily Yahoo Finance history (split-adjusted OHLC, 'Adj Close', Volume) indexed by date, oldest first.
    start=None -> the full history. Empty DataFrame when Yahoo has no prices for the symbol (unknown symbol, or
    nothing in the range). Raises YahooError when the download still fails after retries with backoff (rate
    limit, network, Yahoo down), so a failure is never mistaken for "no prices yet". (Tests replace this.)"""
    import yfinance as yf
    from yfinance.exceptions import YFPricesMissingError, YFTzMissingError
    yf.config.debug.hide_exceptions = False      # raise instead of logging and returning an empty frame
    kw = dict(auto_adjust=False, actions=False)
    if start is None:
        kw['period'] = 'max'
    else:
        kw['start'] = str(start)
    last = None
    for wait in (0,) + YAHOO_BACKOFF:
        time.sleep(max(wait, YAHOO_PAUSE - (time.monotonic() - _last_yahoo_call[0]), 0))
        _last_yahoo_call[0] = time.monotonic()
        try:
            h = yf.Ticker(symbol).history(**kw)
        except (YFPricesMissingError, YFTzMissingError):
            return pd.DataFrame()                 # Yahoo answered: no prices for this symbol / range
        except (ValueError, TypeError) as e:      # a bad request, not worth retrying
            raise YahooError(f'{e.__class__.__name__}: {e}'[:200]) from e
        except Exception as e:
            if getattr(getattr(e, 'response', None), 'status_code', None) == 404:
                return pd.DataFrame()             # unknown symbol
            last = e
            continue
        if h is None or h.empty:
            return pd.DataFrame()
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        idx = pd.DatetimeIndex(h.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        h.index = idx.normalize()
        return h[~h.index.duplicated(keep='last')].sort_index()
    raise YahooError(f'{last.__class__.__name__}: {last}'[:200])


class Fetcher:
    """Yahoo downloads for one run: (history cut to [start, until] without rows that have no close, ok).
    ok=False means the download failed (`failed[symbol]` says why); an empty frame with ok=True means Yahoo has
    no prices. A symbol that failed is not tried again in the same run, and after YAHOO_BREAKER symbols in a
    row fail (each after its retries), Yahoo is treated as down and the rest are not tried."""

    def __init__(self):
        self.failed, self.streak = {}, 0

    def __call__(self, symbol, start, until):
        if symbol in self.failed:
            return pd.DataFrame(), False
        if self.streak >= YAHOO_BREAKER:
            self.failed[symbol] = 'not tried: Yahoo Finance looks down'
            return pd.DataFrame(), False
        try:
            h = yahoo_history(symbol, start)
        except Exception as e:
            self.streak += 1
            self.failed[symbol] = (str(e) if isinstance(e, YahooError) else f'{e.__class__.__name__}: {e}')[:200]
            return pd.DataFrame(), False
        self.streak = 0
        if h is None or len(h) == 0 or 'Close' not in h:
            return pd.DataFrame(), True
        h = h.dropna(subset=['Close'])
        h = h[h.index <= pd.Timestamp(until)]
        if start is not None:
            h = h[h.index >= pd.Timestamp(start)]
        return h, True


def yf_intraday(symbol):
    """The last five days of 30-minute bars (used only to fill a late daily close)."""
    import yfinance as yf
    return yf.Ticker(symbol).history(period='5d', interval='30m', auto_adjust=False)


def final_bar_dates(today, now=None):
    """Last date whose daily bar is final, for (US stocks, LSE funds, FX). now=None: a run before the US open on
    `today` (data to the day before)."""
    if now is None:
        d = today - timedelta(days=1)
        return d, d, d
    ny, lon = now.astimezone(NY), now.astimezone(LONDON)
    us = ny.date() if ny.time() >= US_FINAL else ny.date() - timedelta(days=1)
    lse = lon.date() if lon.time() >= LSE_FINAL else lon.date() - timedelta(days=1)
    return us, lse, lon.date()


def fill_recent_closes(daily, intraday, until):
    """Yahoo often publishes an LSE fund's daily close a day late. Add the missing days after the last daily
    close, up to `until`, from the day's last intraday price (Close of the last bar). Both are Series of prices;
    `intraday` is indexed by tz-aware or naive timestamps. Returns the daily series, extended."""
    if intraday is None or len(intraday) == 0:
        return daily
    last = daily.index.max() if len(daily) else None
    idx = intraday.index.tz_convert(LONDON) if getattr(intraday.index, 'tz', None) is not None else intraday.index
    per_day = intraday.dropna().groupby(pd.DatetimeIndex(idx).normalize().tz_localize(None)).last()
    extra = per_day[(per_day.index > (last if last is not None else pd.Timestamp.min))
                    & (per_day.index <= pd.Timestamp(until))]
    return pd.concat([daily, extra]).sort_index() if len(extra) else daily


def fix_pence_glitches(s):
    """Yahoo sometimes mixes pounds and pence for LSE listings (single days 100x off). Scale those days back."""
    s = s.astype(float).copy()
    if len(s) < 3:
        return s
    med = s.rolling(11, center=True, min_periods=3).median()
    ratio = s / med
    s[ratio < 0.05] *= 100
    s[ratio > 20] /= 100
    return s


def price_doc(pid, ticker, currency, kind, s, ndp=4):
    s = s.dropna()
    s = s[s > 0]
    return {'id': pid, 'ticker': ticker, 'currency': currency, 'kind': kind,
            'dates': [d.date().isoformat() for d in s.index], 'close': [round(float(v), ndp) for v in s.values]}


# ----------------------------------------------------------------------------------------------- SEC filings
def filing_url(cik, filename=None):
    m = re.search(r'(\d{10}-\d{2}-\d{6})\.txt$', str(filename or ''))
    if m:
        acc = m.group(1)
        return f'https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace("-", "")}/{acc}-index.htm'
    return f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-12B'


def sec_filings(years, today, errors):
    """Form 10-12B filers from the last `years` years with their exchange ticker (if any). Raises on failure."""
    s = ls.sec_session()
    q_errors = []
    F = ls.form10_filings(s, years, today=today, errors=q_errors)
    errors.extend(q_errors)
    if q_errors:
        raise RuntimeError(f'{len(q_errors)} EDGAR quarterly index file(s) could not be read')
    if F.empty:
        raise RuntimeError('no Form 10-12B filings found')
    F = F.join(ls.tickers(s), on='cik', how='left')
    out = []
    for _, f in F.iterrows():
        cik = str(int(f['cik']))
        t = f.get('ticker')
        ticker = norm_ticker(t) if isinstance(t, str) and t.strip() else None
        # the name SEC uses now (e.g. 'Qnity Electronics, Inc.'), not the Form 10 filer name ('Novus SpinCo 1, Inc.')
        filer, name = str(f['company']), f.get('name')
        current = name.strip() if ticker and isinstance(name, str) and name.strip() else filer
        out.append({'cik': cik, 'company': current, 'form10_date': f['form10_date'].date(),
                    'form10_url': filing_url(cik, f.get('filename')), 'ticker': ticker,
                    'exchange': f.get('exchange') if isinstance(f.get('exchange'), str) else None,
                    'filed_as': filer if short_name(filer).lower() != short_name(current).lower() else None})
    return out


def filings_from_items(items):
    """Rebuild the filing list from yesterday's pipeline (fallback when the SEC scan fails)."""
    out = []
    for it in items or []:
        try:
            out.append({'cik': cik_key(it['id']), 'company': str(it.get('company') or ''),
                        'form10_date': _date(it.get('form10_date'), 'form10_date'),
                        'form10_url': it.get('form10_url') or filing_url(cik_key(it['id'])),
                        'ticker': (norm_ticker(it['ticker']) or None) if it.get('ticker') else None,
                        'exchange': it.get('exchange')})
        except (KeyError, ValueError):
            continue
    return out


# ----------------------------------------------------------------------------------------------- the rules
def item_flags(f, check=None):
    """Text added to every note of a pipeline item."""
    flag = ' Checked: not a genuine spin-off.' if check and check.get('genuine') is False else ''
    return flag + (f' Filed its Form 10 as {f["filed_as"].rstrip(".")}.' if f.get('filed_as') else '')


def classify(f, h, today, cal, check=None):
    """One pipeline item (model status as if following the rules) from a filing and its price history."""
    it = {'id': f['cik'], 'company': f['company'], 'ticker': f['ticker'], 'exchange': f['exchange'],
          'form10_date': iso(f['form10_date']), 'form10_url': f['form10_url'], 'first_trade': None,
          'buy_date': None, 'buy_window_end': None, 'model_sell_date': None, 'status': 'waiting', 'note': '',
          'last_close': None, 'price_before_buy': None, 'avg_dollar_volume': None, 'buy_open': None}
    flag = item_flags(f, check)

    def done(status, note):
        it['status'], it['note'] = status, (note + flag).strip()
        return it

    if not f['ticker']:
        return done('waiting', 'No exchange ticker yet.')
    if h is None or len(h) == 0:
        return done('waiting', 'No price history yet.')
    first = h.index[0].date()
    it['first_trade'], it['last_close'] = first.isoformat(), r4(h['Close'].iloc[-1])
    if first < f['form10_date']:
        return done('not_new', f'First traded {first}, before its Form 10: not a new listing.')
    if first > f['form10_date'] + timedelta(days=365):
        return done('not_new', f'First traded {first}, more than a year after its Form 10.')
    buy = h.index[DELAY].date() if len(h) > DELAY else nc.nth_trading_day(cal, first, DELAY)
    end, sell = nc.nth_trading_day(cal, buy, WINDOW - 1), nc.nth_trading_day(cal, buy, HOLD)
    it.update(buy_date=iso(buy), buy_window_end=iso(end), model_sell_date=iso(sell))
    if today < buy:
        return done('watch', f'Buy day {buy} (trading day {DELAY}) if it passes the $1 / $100K check; '
                             f'window closes {end}.')
    w = h[h.index < pd.Timestamp(buy)].tail(20)
    if len(w) == 0:
        return done('skip', 'No prices before the buy day.')
    price = float(w['Close'].iloc[-1])
    vol = w['Volume'].fillna(0) if 'Volume' in w else 0
    adv = float((w['Close'] * vol).mean())
    it['price_before_buy'], it['avg_dollar_volume'] = r4(price), int(round(adv)) if math.isfinite(adv) else None
    if not (price >= MIN_PRICE and adv >= MIN_ADV):
        why = ([f'last close ${price:,.2f} < $1'] if not price >= MIN_PRICE else []) + \
              ([f'average dollar volume ${adv:,.0f} < $100,000'] if not adv >= MIN_ADV else [])
        return done('skip', 'Failed the buy-day check: ' + ' and '.join(why) + '.')
    if pd.Timestamp(buy) in h.index and 'Open' in h:
        it['buy_open'] = r4(h.loc[pd.Timestamp(buy), 'Open'])
    if today <= end:
        return done('BUY', f'Buy at the US open, {buy} to {end}; sell 252 trading days after your buy.')
    if today < sell:
        return done('HOLD', f'Model bought {buy}, sells {sell}. Not bought by {end} = missed (do not buy late).')
    if today <= sell + timedelta(days=SELL_GRACE):
        return done('SELL', f'Model sell date {sell}: sell at the US open.')
    return done('done', f'Model holding ended {sell}.')


CARRIED = ' Prices could not be read today: dates and status carried forward from yesterday.'
UNCHECKED = (' Buy day reached, but its prices could not be read today, so the $1 / $100K check has not been run:'
             ' check its last close and daily trading yourself before buying.')


def carry_forward(prev, f, today, cal, check=None):
    """Yesterday's pipeline item for a spin-off whose prices could not be read today, re-classified on the
    calendar (its first trade, buy, window and sell dates do not change; a "waiting" item stays waiting).
    Returns (item, checked); checked=False when the buy day has come but the $1 / $100K check could not be run
    yet (the item is then BUY with avg_dollar_volume None). Returns (None, False) when there is no usable
    item from yesterday."""
    if not isinstance(prev, dict):
        return None, False
    it = classify(f, None, today, cal, check)
    if not prev.get('first_trade'):
        if prev.get('status') != 'waiting':
            return None, False
        it['note'] = (it['note'] + CARRIED).strip()     # not trading yesterday: still waiting (first trade unknown)
        return it, True
    for k in ('first_trade', 'buy_date', 'buy_window_end', 'model_sell_date', 'last_close', 'price_before_buy',
              'avg_dollar_volume', 'buy_open'):
        it[k] = prev.get(k)
    old, flags = prev.get('status'), item_flags(f, check)
    if old in ('not_new', 'skip'):
        it['status'], it['note'] = old, (str(prev.get('note') or '').replace(CARRIED.strip(), '') + CARRIED).strip()
        return it, True
    try:
        buy, end, sell = (_date(prev.get(k), k) for k in ('buy_date', 'buy_window_end', 'model_sell_date'))
    except ValueError:
        return None, False
    px, adv = prev.get('price_before_buy'), prev.get('avg_dollar_volume')
    checked = isinstance(px, (int, float)) and isinstance(adv, (int, float))
    if today < buy:
        status, note = 'watch', f'Buy day {buy} (trading day {DELAY}) if it passes the $1 / $100K check.'
    elif not checked:
        if today > end:
            return None, False            # the window is over and nobody knows whether it passed the check
        return dict(it, status='BUY', note=f'Buy window {buy} to {end}.{UNCHECKED}{flags}{CARRIED}'), False
    elif not (px >= MIN_PRICE and adv >= MIN_ADV):
        status, note = 'skip', 'Failed the buy-day check.'
    elif today <= end:
        status, note = 'BUY', f'Buy at the US open, {buy} to {end}; sell 252 trading days after your buy.'
    elif today < sell:
        status, note = 'HOLD', f'Model bought {buy}, sells {sell}. Not bought by {end} = missed (do not buy late).'
    elif today <= sell + timedelta(days=SELL_GRACE):
        status, note = 'SELL', f'Model sell date {sell}: sell at the US open.'
    else:
        status, note = 'done', f'Model holding ended {sell}.'
    it['status'], it['note'] = status, f'{note}{flags}{CARRIED}'
    return it, True


def build_pipeline(filings, today, cal, us_cut, checks, fetch, prev=None, yahoo_suspect=False):
    """Pipeline items for `filings`, classified with each ticker's full Yahoo history cut at `us_cut`.

    A spin-off whose prices cannot be read today (the download failed, its history could not be read, or Yahoo
    returned nothing although yesterday's item had a first trade, or while other downloads that always have
    prices failed: yahoo_suspect) is a failure, never "waiting": its item is carried forward from yesterday's
    pipeline `prev` (matched by CIK) when possible.
    Returns (items, hists {ticker: history}, failures [{'ticker', 'why', 'carried', 'trusted'}]); trusted=False
    means the item could not be carried forward reliably, so it must not be written to the database."""
    prev_by_cik = {cik_key(p.get('id')): p for p in prev or [] if isinstance(p, dict) and p.get('id') is not None}
    hists, items, failures = {}, [], []
    for f in filings:
        t, chk = f['ticker'], checks.get(cik_key(f['cik']))
        p = prev_by_cik.get(cik_key(f['cik']))
        h, ok, why, it = None, True, None, None
        if t:
            h, ok = fetch(yahoo_symbol(t), None, us_cut)
            if not ok:
                why = f'download failed ({fetch.failed.get(yahoo_symbol(t), "?")})'
            elif len(h) == 0 and p and p.get('first_trade'):
                why = 'no prices returned, although it had prices yesterday'
            elif len(h) == 0 and yahoo_suspect:
                why = 'no prices returned while other Yahoo Finance downloads failed'
            else:
                hists[t] = h
        if why is None:
            try:
                it = classify(f, h, today, cal, chk)
            except Exception as e:   # one odd price history must not stop the run
                why = f'price history could not be read ({e.__class__.__name__}: {e})'
                hists.pop(t, None)
        if why is not None:
            it, trusted = carry_forward(p, f, today, cal, chk)
            carried = it is not None
            if not carried:
                it = classify(f, None, today, cal, chk)
                it['note'] = 'Could not read its price history today.'
            failures.append({'ticker': t, 'why': why, 'carried': carried, 'trusted': carried and trusted})
        items.append(it)
    items.sort(key=lambda it: (ORDER.get(it['status'], 99), it['form10_date'], it['company']))
    return items, hists, failures


def positions(trades):
    """{ticker: {shares, opened, cost_gbp}} from the trade log. `opened` = date of the buy that opened the current
    position (None when flat); cost_gbp = GBP paid for the shares still held (pro rata after part sales).
    A sell that leaves less than DUST (0.5%) of the shares closes the position: Trading 212 shows up to about
    7 decimals, so a sell typed with fewer leaves a tiny remainder that is not a real holding."""
    out = {}
    for t in sorted(trades, key=lambda t: (t['date'], t['type'] != 'buy', t['created_at'])):
        p = out.setdefault(t['ticker'], {'shares': 0.0, 'opened': None, 'cost_gbp': 0.0})
        if t['type'] == 'buy':
            if p['shares'] <= EPS:
                p.update(shares=0.0, opened=t['date'], cost_gbp=0.0)
            p['shares'] += t['shares']
            p['cost_gbp'] += t['amount_gbp']
        else:
            before = p['shares']
            if before > EPS:
                p['cost_gbp'] *= max(0.0, before - t['shares']) / before
            p['shares'] -= t['shares']
            if p['shares'] <= max(EPS, DUST * before):
                p.update(shares=0.0, opened=None, cost_gbp=0.0)
    return out


def account_estimate(deposits, trades, pos, fund, fund_px, fx_px, closes, today):
    """Rough account value in GBP: deposits and spin-off sale proceeds buy the index fund at that day's close,
    spin-off buys sell it. None without deposits or index prices."""
    if not deposits or fund_px is None or len(fund_px) == 0:
        return None

    def px(d):
        s = fund_px[fund_px.index <= pd.Timestamp(d)]
        return float(s.iloc[-1]) if len(s) else float(fund_px.iloc[0])

    ev = [(d['date'], 0, d['amount_gbp']) for d in deposits] + \
         [(t['date'], 1 if t['type'] == 'sell' else 2, t['amount_gbp'] if t['type'] == 'sell' else -t['amount_gbp'])
          for t in trades if t['ticker'] not in RESERVED]
    units = sum(a / px(d) for d, _, a in sorted(ev))
    index_value = units * float(fund_px.iloc[-1])
    fx = float(fx_px.iloc[-1]) if fx_px is not None and len(fx_px) else None
    spin = {}
    for t, p in sorted(pos.items()):
        if p['shares'] <= EPS or t in RESERVED:
            continue
        c = closes.get(t)
        spin[t] = p['shares'] * float(c.iloc[-1]) / fx if c is not None and len(c) and fx else p['cost_gbp']
    total = index_value + sum(spin.values())
    if not total > 0:
        return None
    return {'as_of': today.isoformat(), 'index_fund': fund, 'index_value_gbp': round(index_value, 2),
            'spinoffs_gbp': {t: round(v, 2) for t, v in spin.items()}, 'total_gbp': round(total, 2)}


def funding_text(target, idx_val, spin_vals, fund):
    """How to pay for a buy of `target` GBP; returns (text, new index value, new spin-off values)."""
    if idx_val >= target - 0.5:
        return f' Pay for it by selling about {money(target)} of {fund}.', idx_val - target, spin_vals
    have = max(idx_val, 0.0)
    trim = target - have
    tot = sum(v for v in spin_vals.values() if v > 0)
    if tot <= 0:
        return f' Pay for it with your {fund} (about {money(have)}).', 0.0, spin_vals
    cut = {t: trim * max(v, 0) / tot for t, v in spin_vals.items()}
    parts = sorted(((a, t) for t, a in cut.items() if a >= 0.5), reverse=True)
    listed = ', '.join(f'{t} {money(a)}' for a, t in parts[:6]) + (', ...' if len(parts) > 6 else '')
    text = (f' Sell all your {fund} (about {money(have)}) and trim about {money(trim)} from your spin-offs in '
            f'proportion to their value: {listed}.')
    return text, 0.0, {t: v - cut[t] for t, v in spin_vals.items()}


def us_close_uk(d):
    """UK clock time of the 16:00 New York close on date d, e.g. '21:00'."""
    return datetime.combine(d, dtime(16, 0), NY).astimezone(LONDON).strftime('%H:%M')


def action_day(today, cal, now=None):
    """(day, phase): the NYSE day the user can next trade on, and how. phase 'pre' = today, before the US open;
    'open' = today, the US market is open now; 'next' = a later day (weekend, holiday, or after today's close).
    now=None: a run before the US open (the scheduled run, and --today simulations)."""
    nxt = nc.nth_trading_day(cal, today, 0) if cal else None
    if nxt is None:
        return today, 'pre'
    if nxt != today:
        return nxt, 'next'
    if now is not None:
        t = now.astimezone(NY)
        if t.date() == today and t.time() >= dtime(16, 0):
            return nc.nth_trading_day(cal, today, 1) or today, 'next'
        if t.date() == today and t.time() >= dtime(9, 30):
            return today, 'open'
    return today, 'pre'


def join_push(pieces, tail='', limit=190):
    """'A; B; +3 more' in at most `limit` characters; `tail` (a short warning) is always kept at the end."""
    room = limit - (len(tail) + 2 if tail else 0)
    push = ''
    for i, p in enumerate(pieces):
        rest = len(pieces) - i - 1
        cand = f'{push}; {p}' if push else p
        more = f'; +{rest} more' if rest else ''
        if len(cand) + len(more) > room:
            push = f'{push}; +{len(pieces) - i} more' if push else p[:room]
            break
        push = cand
    return f'{push}; {tail}' if push and tail else push or tail


def build_alerts(today, items, data, cal, closes, fund_px, fx_px, now=None, push_tail='', quiet='No trades today.'):
    """alerts.json: the personal actions for the next time the user can trade (buy / last chance / sell) and
    heads-ups. now: the clock time of a real run (None = before the US open); after the US close the actions
    are for the next trading day, so a buy window that ends today is over. push_tail: a short warning added to
    the push when there are actions; quiet: the push when there are none."""
    cfg = data['settings']
    data = as_of(data, today)[0]
    n, fund, dismissed, checks = cfg['slice'], cfg['index_fund'], cfg['dismissed'], data['checks']
    pos = positions(data['trades'])
    held = {t: p for t, p in pos.items() if p['shares'] > EPS and t not in RESERVED}
    acct = account_estimate(data['deposits'], data['trades'], pos, fund, fund_px, fx_px, closes, today)
    idx_val = acct['index_value_gbp'] if acct else 0.0
    spin_vals = dict(acct['spinoffs_gbp']) if acct else {}
    by_ticker = {it['ticker']: it for it in items if it.get('ticker')}
    act, phase = action_day(today, cal, now)
    when = {'pre': f'today after the US open ({us_open_uk(today)} UK)',
            'open': f'now, while the US market is open (until {us_close_uk(today)} UK)',
            'next': f'at the next US open ({fmt_day(act, today)}, {us_open_uk(act)} UK)'}[phase]
    i_today = bisect_left(cal, today)

    def tdays_until(d):
        return bisect_left(cal, d) - i_today

    def label(t):
        it = by_ticker.get(t)
        return (it['company'] if it else t), (f'{t} ({short_name(it["company"])})' if it else t)

    actions, info = [], []
    for t in sorted(held):
        p = held[t]
        sd = nc.nth_trading_day(cal, p['opened'], HOLD)
        company, lab = label(t)
        if sd is None:
            continue
        if act >= sd:
            due = '12 months are up' if act == sd else f'overdue since {fmt_day(sd, today)}'
            ret = ''
            c, fx = closes.get(t), (float(fx_px.iloc[-1]) if fx_px is not None and len(fx_px) else None)
            if c is not None and len(c) and fx and p['cost_gbp'] > 0:
                val = p['shares'] * float(c.iloc[-1]) / fx * (1 - FX_COST)
                ret = f' At the last close that is about {money(val)}, {val / p["cost_gbp"] - 1:+.1%} on what you paid.'
            actions.append({'type': 'SELL', 'ticker': t, 'company': company, 'date': sd.isoformat(), 'overdue': act > sd,
                            'text': f'Sell all {shares_txt(p["shares"])} {lab} {when}: {due} '
                                    f'(bought {fmt_day(p["opened"], today)}).{ret} Put the money into {fund}.'})
            idx_val += spin_vals.pop(t, 0.0) * (1 - FX_COST)
        elif 0 < tdays_until(sd) <= 5:
            info.append({'type': 'UPCOMING_SELL', 'ticker': t, 'company': company, 'date': sd.isoformat(),
                         'text': f'Coming up: sell {lab} on {fmt_day(sd, today)} after the US open '
                                 f'({us_open_uk(sd)} UK), 12 months after you bought it.'})

    for it in items:
        t, status = it.get('ticker'), it.get('status')
        if not t or t in held or t in dismissed or status not in ('BUY', 'watch'):
            continue
        chk = checks.get(cik_key(it['id']))
        if chk and chk['genuine'] is False:
            continue
        mark = ' (not yet checked as a real spin-off)' if chk is None else \
            ' (spin-off check inconclusive)' if chk['genuine'] is None else ''
        lab = f'{t} ({short_name(it["company"])})'
        buy, end = _date(it['buy_date'], 'buy_date'), _date(it['buy_window_end'], 'buy_window_end')
        if status == 'watch':
            k = tdays_until(buy)
            if 0 < k <= 5:
                info.append({'type': 'UPCOMING_BUY', 'ticker': t, 'company': it['company'], 'date': iso(buy),
                             'text': f'Coming up: buy {lab} on {fmt_day(buy, today)} after the US open '
                                     f'({us_open_uk(buy)} UK) if it passes the $1 / $100K check.{mark}'})
            continue
        if act < buy or act > end:      # after the close on the last day the window is over
            continue
        typ = 'BUY_NEW' if act == buy else 'LAST_CHANCE' if act == end else 'BUY_REMINDER'
        # a BUY carried forward from yesterday without today's prices: the $1 / $100K check has not been run
        unchecked = 'avg_dollar_volume' in it and it['avg_dollar_volume'] is None
        if unchecked:
            mark += (' Its prices could not be read today, so the $1 / $100K check has not been run: buy only if '
                     'its last close is $1 or more and it trades about $100,000 a day or more.')
        amount, fund_txt = '', ''
        if acct:
            target = acct['total_gbp'] / n
            amount = f' (about {money(target)})'
            fund_txt, idx_val, spin_vals = funding_text(target, idx_val, spin_vals, fund)
            spin_vals[t] = target
        if typ == 'BUY_NEW':
            text = (f'Buy {lab} {when}: about 1/{n} of your account{amount}. Window closes {fmt_day(end, today)}.'
                    f'{fund_txt}{mark}')
        elif typ == 'LAST_CHANCE':
            text = f'Last day to buy {lab}: {when}, about 1/{n} of your account{amount}, or let it go.{fund_txt}{mark}'
        else:
            text = (f'Still time to buy {lab}: {when}, about 1/{n} of your account{amount}. '
                    f'Window closes {fmt_day(end, today)}.{fund_txt}{mark}')
        actions.append({'type': typ, 'ticker': t, 'company': it['company'],
                        'date': iso(end if typ == 'LAST_CHANCE' else buy), 'text': text, 'unchecked': unchecked})

    recent = cal[max(0, i_today - 3):i_today]
    for it in items:
        if not it.get('first_trade') or it.get('status') != 'watch' or not recent:
            continue
        first = _date(it['first_trade'], 'first_trade')
        chk = checks.get(cik_key(it['id']))
        if recent[0] <= first <= today and not (chk and chk['genuine'] is False):
            buy = _date(it['buy_date'], 'buy_date')
            info.append({'type': 'NEW_LISTING', 'ticker': it['ticker'], 'company': it['company'], 'date': iso(first),
                         'text': f'New spin-off trading: {it["ticker"]} ({short_name(it["company"])}) since '
                                 f'{fmt_day(first, today)}. Buy day {fmt_day(buy, today)} after the US open '
                                 f'({us_open_uk(buy)} UK) if it passes the $1 / $100K check.'})

    prio = {'BUY_NEW': 0, 'LAST_CHANCE': 1, 'SELL': 2, 'BUY_REMINDER': 3}
    actions.sort(key=lambda a: (prio[a['type']], a['ticker']))
    pieces = []
    for a in actions:
        t = a['ticker']
        if a['type'] == 'BUY_NEW':
            pieces.append(f'BUY {t} today')
        elif a['type'] == 'LAST_CHANCE':
            pieces.append(f'LAST DAY to buy {t}')
        elif a['type'] == 'SELL':
            sd = _date(a['date'], 'date')
            pieces.append(f'SELL {t} (overdue since {sd.day} {sd:%b})' if a['overdue'] else f'SELL {t} (12 months up)')
        else:
            pieces.append(f'BUY {t} (window to {fmt_day(_date(by_ticker[t]["buy_window_end"], "d"), today)})')
        if a.get('unchecked'):
            pieces[-1] += ' (check price first)'
    noteworthy = bool(actions)
    push = join_push(pieces, push_tail) if noteworthy else quiet
    lines = [a['text'] for a in actions] + [x['text'] for x in info]
    if acct:
        sp = acct['spinoffs_gbp']
        lines.append(f'Account about {money(acct["total_gbp"])}: {fund} {money(acct["index_value_gbp"])} + '
                     f'{len(sp)} spin-off{"s" if len(sp) != 1 else ""} {money(sum(sp.values()))} '
                     f'(estimate from your logged deposits and trades).')
    for a in actions:
        a.pop('overdue', None)
        if not a.get('unchecked'):     # kept (true) only on a buy whose $1 / $100K check could not be run today
            a.pop('unchecked', None)
    return {'noteworthy': noteworthy, 'push': push, 'lines': lines,
            'items': actions + info, 'today_ny': today.isoformat(), 'account': acct}


# ----------------------------------------------------------------------------------------------- scan
def write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(body, separators=(',', ':'), allow_nan=False, ensure_ascii=False)
    path.write_text(s, encoding='utf-8')
    return len(s.encode('utf-8'))


def write_doc(out, coll, doc_id, body, errors):
    """Write one database document; a prices document over the size limit loses its oldest points."""
    while True:
        n = len(json.dumps(body, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8'))
        if n <= DOC_LIMIT - 1024:
            break
        if coll == 'prices' and len(body['dates']) > 20:
            k = max(1, len(body['dates']) // 10)
            body['dates'], body['close'] = body['dates'][k:], body['close'][k:]
            continue
        errors.append(f'{coll}/{doc_id}: document too large ({n} bytes), not written')
        return None
    return write_json(out / 'docs' / coll / f'{doc_id}.json', body)


def scan(out, db=None, years=2, today=None, trigger_id=None, next_check=NEXT_CHECK, now=None):
    """Run the daily scan; returns a dict summary. `today` (a date) simulates a pre-open run on that day."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / 'docs', ignore_errors=True)
    for f in out.glob('writes_*.json'):
        f.unlink()
    errors = []
    real = today is None
    now = now or datetime.now(timezone.utc)
    today = today or now.astimezone(NY).date()
    us_cut, lse_cut, fx_cut = final_bar_dates(today, now if real else None)
    data, later = as_of(load_exports(db, errors), today)
    if later:
        errors.append(f'Ignored until their date ({len(later)} dated after {today}): ' + ', '.join(later[:5])
                      + (', ...' if len(later) > 5 else ''))
    first_user = min([t['date'] for t in data['trades']] + [d['date'] for d in data['deposits']], default=None)
    start = min(today - timedelta(days=LOOKBACK), MODEL_START)
    if first_user is not None:
        start = min(start, first_user - timedelta(days=7))
    start = max(start, today - timedelta(days=round(365.25 * MAX_YEARS_BACK)))
    fetch = Fetcher()

    # NYSE calendar: SPY's real days, then the rules (holidays and the known one-off closures)
    cal_from = min(start, today - timedelta(days=366 * years + 10)) - timedelta(days=10)
    spy, _ = fetch('SPY', cal_from, us_cut)
    actual = [d.date() for d in spy.index]
    if not actual:
        errors.append('Yahoo Finance: no SPY history; the calendar is projected from NYSE holiday rules'
                      + (f' ({fetch.failed["SPY"]})' if 'SPY' in fetch.failed else ''))
    cal, last_actual = nc.build_calendar(cal_from, today + timedelta(days=CAL_AHEAD + 200), actual)

    # benchmarks (always have prices: if one is missing, Yahoo is misbehaving and empty answers are suspect)
    docs, fund_px = {}, {}
    for fid, (sym, col, scale) in INDEX.items():
        h, _ = fetch(sym, start, lse_cut)
        if len(h) == 0 or col not in h:
            errors.append(f'Yahoo Finance: no prices for {sym}' + (f' ({fetch.failed[sym]})' if sym in fetch.failed
                                                                    else ''))
            continue
        daily = h[col].dropna()
        if len(daily) and daily.index.max() < pd.Timestamp(lse_cut):
            try:   # the daily close is late: take the day's last intraday price instead
                ih = yf_intraday(sym)
                daily = fill_recent_closes(daily, ih['Close'] if len(ih) else None, lse_cut)
            except Exception as e:  # noqa: BLE001 - a fallback; the daily data still stands
                errors.append(f'Yahoo Finance: intraday {sym} for the latest close failed ({e.__class__.__name__})')
        fund_px[fid] = fix_pence_glitches(daily) * scale
        docs[fid] = price_doc(fid, sym, 'GBP', 'index', fund_px[fid])
    h, _ = fetch(FX_SYMBOL, start, fx_cut)
    fx_px = h['Close'] if len(h) else None
    if fx_px is None:
        errors.append(f'Yahoo Finance: no prices for {FX_SYMBOL}' + (f' ({fetch.failed[FX_SYMBOL]})'
                                                                    if FX_SYMBOL in fetch.failed else ''))
    else:
        docs[FX_ID] = price_doc(FX_ID, FX_SYMBOL, 'USD', 'fx', fx_px)   # US dollars per 1 pound
    yahoo_suspect = not actual or len(fund_px) < len(INDEX) or fx_px is None

    # pipeline (yesterday's list stands in when SEC fails; yesterday's items stand in for failed prices)
    sec_ok, filings, fallback = False, [], False
    try:
        filings = sec_filings(years, today, errors)
        sec_ok = True
    except Exception as e:
        errors.append(f'SEC scan failed: {e.__class__.__name__}: {e}'[:300])
        if data['pipeline']:
            filings, fallback = filings_from_items(data['pipeline']), True
    items, hists, failures = build_pipeline(filings, today, cal, us_cut, data['checks'], fetch, data['pipeline'],
                                            yahoo_suspect)
    for x in failures:
        what = ('status carried forward from yesterday' if x['trusted'] else
                'its buy check could not be run' if x['carried'] else 'status unknown today')
        errors.append(f'Yahoo Finance: no prices for {x["ticker"]} today ({x["why"]}); {what}'[:300])
    untrusted = [x['ticker'] for x in failures if not x['trusted']]
    prices_ok = not untrusted and len(failures) <= MAX_TICKER_FAILS
    scan_ok = sec_ok and bool(actual) and prices_ok
    if sec_ok and not scan_ok:
        why = ([] if actual else ['no SPY history (Yahoo Finance looks unavailable)']) + \
              ([f'no prices for {len(failures)} spin-off{"s" if len(failures) != 1 else ""}']
               if failures and not prices_ok else [])
        errors.append('Pipeline not updated: ' + '; '.join(why))

    # prices of the spin-offs watched, held or recently done
    want = {}
    for it in items:
        t, stt = it['ticker'], it['status']
        recent_done = stt == 'done' and it['model_sell_date'] and \
            _date(it['model_sell_date'], 'd') + timedelta(days=DONE_PRICES) >= today
        if t and (stt in ACTIVE or recent_done):
            want[t] = hists.get(t)
    for t in sorted({tr['ticker'] for tr in data['trades']}):
        if t not in RESERVED and t not in want:
            want[t] = hists.get(t)
    failed_tickers = {x['ticker'] for x in failures}
    closes = {}
    for t, h in sorted(want.items()):
        if not ID_RE.match(t):
            errors.append(f'prices: ticker {t!r} cannot be a document id, skipped')
            continue
        if t in failed_tickers:
            continue                 # already reported; the database keeps yesterday's prices document
        if h is None:
            h, _ = fetch(yahoo_symbol(t), start, us_cut)
        h = h[h.index >= pd.Timestamp(start)] if len(h) else h
        c = pd.to_numeric(h['Close'], errors='coerce').dropna() if len(h) else pd.Series(dtype=float)
        if len(c) == 0:
            errors.append(f'Yahoo Finance: no prices for {t}' + (f' ({fetch.failed[yahoo_symbol(t)]})'
                                                                 if yahoo_symbol(t) in fetch.failed else ''))
            continue
        closes[t] = c
        docs[t] = price_doc(t, t, 'USD', 'spinoff', c)

    # alerts: from today's items whenever the filing list is known (fresh, or yesterday's when SEC failed);
    # a run that is not fully trustworthy says so in the push
    have_list = sec_ok or fallback
    notes = []
    if fallback:
        notes.append('Note: the SEC scan failed, so buy alerts use yesterday\'s spin-off list.')
    elif not sec_ok:
        notes.append('Note: the spin-off scan failed today, so new buys were not checked (sell alerts still work).')
    if not actual:
        notes.append('Note: Yahoo Finance had no SPY history, so trading days come from the NYSE holiday rules '
                     '(exact except for a closure announced at short notice).')
    if failures:
        carried = [x['ticker'] for x in failures if x['carried']]
        lost = [x['ticker'] for x in failures if not x['carried']]
        notes.append('Note: prices could not be read today for ' + ', '.join(x['ticker'] for x in failures) + '.'
                     + (f' Status carried forward from yesterday for {", ".join(carried)}.' if carried else '')
                     + (f' Status unknown today for {", ".join(lost)}.' if lost else ''))
    if not have_list:
        tail, quiet = 'buys not checked', 'No sells due. The spin-off scan failed today, so new buys were not checked.'
    elif not scan_ok:
        tail, quiet = 'scan incomplete', 'No trades found, but today\'s scan was incomplete: see the page.'
    else:
        tail, quiet = '', 'No trades today.'
    alerts = build_alerts(today, items if have_list else [], data, cal, closes,
                          fund_px.get(data['settings']['index_fund']), fx_px, now=now if real else None,
                          push_tail=tail, quiet=quiet)
    alerts['scan_ok'] = scan_ok
    alerts['lines'] += notes
    needs = [{'cik': it['id'], 'ticker': it['ticker'], 'company': it['company'], 'form10_url': it['form10_url']}
             for it in items if it['ticker'] and it['status'] in ACTIVE and cik_key(it['id']) not in data['checks']]

    # write
    sizes = {}
    for pid, body in sorted(docs.items()):
        sizes[f'prices/{pid}'] = write_doc(out, 'prices', pid, body, errors)
    cal_end = today + timedelta(days=CAL_AHEAD)
    sizes['market/calendar'] = write_doc(out, 'market', 'calendar', {
        'days': [d.isoformat() for d in cal if start <= d <= cal_end], 'last_actual': iso(last_actual)}, errors)
    if scan_ok:
        sizes['market/pipeline'] = write_doc(out, 'market', 'pipeline', {'items': items}, errors)
    status = {'generated_at': now.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
              'today_ny': today.isoformat(), 'last_market_day': iso(last_actual), 'scan_ok': scan_ok,
              'errors': [e[:300] for e in errors[:MAX_ERRORS]] + ([f'... and {len(errors) - MAX_ERRORS} more']
                                                                   if len(errors) > MAX_ERRORS else []),
              'counts': {'filings': len(filings) if sec_ok else 0,
                         'tickers': sum(1 for f in filings if f['ticker']) if sec_ok else 0},
              'trigger_id': trigger_id, 'next_check': next_check}
    sizes['market/status'] = write_doc(out, 'market', 'status', status, errors)
    write_json(out / 'alerts.json', alerts)
    write_json(out / 'needs_check.json', needs)
    return {'out': str(out), 'today_ny': today.isoformat(), 'last_market_day': iso(last_actual),
            'scan_ok': scan_ok, 'fallback': fallback, 'statuses': dict(Counter(it['status'] for it in items)),
            'sizes': {k: v for k, v in sizes.items() if v is not None}, 'alerts': alerts, 'needs_check': needs,
            'errors': errors, 'failures': failures}


def print_summary(r):
    order = sorted(r['statuses'].items(), key=lambda kv: ORDER.get(kv[0], 99))
    print(f'Spin-off tracker scan for {r["today_ny"]} (New York date); market data to {r["last_market_day"]}')
    print(f'Scan {"OK" if r["scan_ok"] else "INCOMPLETE (pipeline not updated)"}'
          + (' - SEC failed, alerts use yesterday\'s list' if r['fallback'] else ''))
    if order:
        print(f'Pipeline: {sum(r["statuses"].values())} items (' + ', '.join(f'{k} {v}' for k, v in order) + ')')
    print(f'Today: {r["alerts"]["push"]}')
    for line in r['alerts']['lines']:
        print(f'  - {line}')
    if r['needs_check']:
        print(f'Needs a Form 10 check: ' + ', '.join(x['ticker'] for x in r['needs_check']))
    sizes = r['sizes']
    print(f'Wrote {len(sizes)} documents ({sum(sizes.values()):,} bytes, largest {max(sizes.values(), default=0):,}) '
          f'to {r["out"]}/docs')
    print(f'Errors: {len(r["errors"])}' + ''.join(f'\n  - {e}' for e in r['errors'][:20]))


# ----------------------------------------------------------------------------------------------- writes
LIST_HEAD = re.compile(r'from\s+collection\s+"([^"]+)"', re.I)
LIST_DOC = re.compile(r'^\s*-\s*"([^"]+)"\s.*?\bversion\s+"?([^\s",]+)"?', re.I)


def parse_versions(text):
    """{(collection, doc_id): version} from the database tool's listing text(s)."""
    out, coll = {}, None
    for line in (text or '').splitlines():
        m = LIST_HEAD.search(line)
        if m:
            coll = m.group(1)
            continue
        m = LIST_DOC.match(line)
        if m and coll is not None:
            v = m.group(2)
            out[(coll, m.group(1))] = int(v) if v.isdigit() else v
    return out


def plan_writes(out, versions, max_entries=45, max_bytes=900_000):
    """Batch chunks for every document under out/docs (market/status last)."""
    docs = Path(out).resolve() / 'docs'
    files = sorted(docs.rglob('*.json')) if docs.is_dir() else []

    def key(p):
        coll = p.parent.relative_to(docs).as_posix()
        rank = 2 if (coll, p.stem) == ('market', 'status') else 1 if coll == 'market' else 0
        return rank, coll, p.stem

    chunks, cur, size = [], [], 0
    for p in sorted(files, key=key):
        coll, did = p.parent.relative_to(docs).as_posix(), p.stem
        e = {'op': 'set', 'collection': coll, 'doc_id': did, 'file_path': str(p)}
        if (coll, did) in versions:
            e['if_version'] = versions[(coll, did)]
        n = p.stat().st_size
        if cur and (len(cur) >= max_entries or size + n > max_bytes):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def writes(out, versions_path):
    out = Path(out).resolve()
    vp = Path(versions_path) if versions_path else None
    text = vp.read_text(encoding='utf-8') if vp is not None and vp.exists() else ''
    versions = parse_versions(text)
    for f in out.glob('writes_*.json'):
        f.unlink()
    chunks = plan_writes(out, versions)
    for i, c in enumerate(chunks, 1):
        write_json(out / f'writes_{i}.json', c)
    return chunks, versions


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('scan', help='scan SEC + Yahoo and write the database documents and alerts')
    a.add_argument('--out', required=True)
    a.add_argument('--db', help='folder of exported documents (trades/, deposits/, settings/, checks/)')
    a.add_argument('--years', type=int, default=2)
    a.add_argument('--today', help='YYYY-MM-DD, testing only: simulate a pre-open run on that New York date')
    a.add_argument('--trigger-id', default=None, help='id of the scheduled task (shown to the page)')
    a.add_argument('--next-check', default=NEXT_CHECK)
    b = sub.add_parser('writes', help='turn the documents into database batch files')
    b.add_argument('--out', required=True)
    b.add_argument('--versions', help='file with the database listing text(s) for market and prices')
    args = ap.parse_args(argv)
    if args.cmd == 'scan':
        today = date.fromisoformat(args.today) if args.today else None
        r = scan(args.out, args.db, args.years, today, args.trigger_id, args.next_check)
        print_summary(r)
        return 0
    chunks, versions = writes(args.out, args.versions)
    n = sum(len(c) for c in chunks)
    print(f'{len(chunks)} chunk{"s" if len(chunks) != 1 else ""} ({n} documents, {len(versions)} existing versions '
          f'known): ' + ', '.join(f'writes_{i}.json ({len(c)})' for i, c in enumerate(chunks, 1)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
