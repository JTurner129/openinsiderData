"""NYSE trading-day calendar: real past days from SPY's price history, future days from NYSE holiday rules.

Rules (NYSE full-day closures, as published by NYSE since 2022):
  New Year's Day (Jan 1), Martin Luther King Jr. Day (3rd Mon Jan), Washington's Birthday (3rd Mon Feb),
  Good Friday (Easter Sunday - 2 days), Memorial Day (last Mon May), Juneteenth (Jun 19, from 2022),
  Independence Day (Jul 4), Labor Day (1st Mon Sep), Thanksgiving (4th Thu Nov), Christmas (Dec 25).
  A holiday on a Saturday is observed the Friday before, except New Year's Day: NYSE does not close on
  Friday Dec 31 (e.g. no closure on 31 Dec 2021 for 1 Jan 2022). A holiday on a Sunday is observed the Monday after.

Rules cannot predict one-off closures (e.g. 9 Jan 2025, national day of mourning for President Carter).
The known ones since 2000 are listed in ONE_OFF_CLOSURES (checked against SPY's history: with them, the rules
reproduce every NYSE trading day from 1999 to Oct 2026), so a calendar built from the rules alone is exact
for the past. Past days still come from SPY's real trading history when it is available, so a future
unannounced closure corrects itself once SPY's history covers it; only days after the last SPY bar are
projected with the rules.

No network access is needed for the rule part; `spy_trading_days` lazily imports yfinance.
"""
from bisect import bisect_left
from datetime import date, timedelta

import pandas as pd


def _nth_weekday(year, month, weekday, n):
    """n-th (1-based) given weekday (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def easter(year):
    """Easter Sunday, anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d, saturday_to_friday=True):
    if d.weekday() == 5:
        return d - timedelta(days=1) if saturday_to_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year):
    """{date: name} of NYSE full-day closures in `year` from the rules (no one-off closures)."""
    h = {}

    def add(d, name, sat_to_fri=True):
        o = _observed(d, sat_to_fri)
        if o is not None and o.year == year:
            h[o] = name

    add(date(year, 1, 1), "New Year's Day", sat_to_fri=False)
    h[_nth_weekday(year, 1, 0, 3)] = 'Martin Luther King Jr. Day'
    h[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    h[easter(year) - timedelta(days=2)] = 'Good Friday'
    h[_nth_weekday(year, 5, 0, -1)] = 'Memorial Day'
    if year >= 2022:
        add(date(year, 6, 19), 'Juneteenth')
    add(date(year, 7, 4), 'Independence Day')
    h[_nth_weekday(year, 9, 0, 1)] = 'Labor Day'
    h[_nth_weekday(year, 11, 3, 4)] = 'Thanksgiving Day'
    add(date(year, 12, 25), 'Christmas Day')
    return dict(sorted(h.items()))


# Unscheduled full-day closures since 2000 (the rules above cannot know them). Every one of them is missing
# from SPY's daily history, and SPY has no other day the rules call a closure (checked 1999 to Oct 2026).
ONE_OFF_CLOSURES = {
    date(2001, 9, 11): 'September 11 attacks', date(2001, 9, 12): 'September 11 attacks',
    date(2001, 9, 13): 'September 11 attacks', date(2001, 9, 14): 'September 11 attacks',
    date(2004, 6, 11): 'National day of mourning for President Reagan',
    date(2007, 1, 2): 'National day of mourning for President Ford',
    date(2012, 10, 29): 'Hurricane Sandy', date(2012, 10, 30): 'Hurricane Sandy',
    date(2018, 12, 5): 'National day of mourning for President George H. W. Bush',
    date(2025, 1, 9): 'National day of mourning for President Carter',
}


def nyse_closures(year):
    """{date: name} of every known NYSE full-day closure in `year`: the holiday rules plus ONE_OFF_CLOSURES."""
    h = nyse_holidays(year)
    h.update({d: n for d, n in ONE_OFF_CLOSURES.items() if d.year == year})
    return dict(sorted(h.items()))


def is_rule_trading_day(d):
    return d.weekday() < 5 and d not in nyse_closures(d.year)


def rule_trading_days(start, end):
    """NYSE trading days in [start, end] by the holiday rules and the known one-off closures."""
    start, end = _d(start), _d(end)
    hol = {}
    for y in range(start.year, end.year + 1):
        hol.update(nyse_closures(y))
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


def build_calendar(start, end, actual=None):
    """Trading days in [start, end]: days <= last actual day come from `actual` (real history, e.g. SPY's
    index), later days from the rules. Returns (days: list[date], last_actual: date | None)."""
    start, end = _d(start), _d(end)
    act = sorted({_d(x) for x in (actual if actual is not None else [])})
    last_actual = act[-1] if act else None
    if last_actual is None:
        return rule_trading_days(start, end), None
    days = [d for d in act if start <= d <= end]
    if last_actual < end:
        days += rule_trading_days(max(start, last_actual + timedelta(days=1)), end)
    return days, last_actual


def nth_trading_day(days, d, n):
    """The n-th trading day after d in the sorted list `days` (n=0 -> d itself, or the next trading day if d is
    not one). None when the list is too short."""
    i = bisect_left(days, _d(d))
    if i >= len(days) or i + n >= len(days) or i + n < 0:
        return None
    return days[i + n]


def spy_trading_days(start, end=None):
    """Real NYSE trading days from SPY's daily history (Yahoo Finance), as a list of dates. Empty on failure."""
    import yfinance as yf
    kw = dict(start=str(_d(start)), progress=False, auto_adjust=True, threads=False)
    if end is not None:
        kw['end'] = str(_d(end) + timedelta(days=1))
    try:
        h = yf.download('SPY', **kw)
    except Exception:
        return []
    if h is None or h.empty:
        return []
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return sorted({x.date() for x in idx.normalize()})


def _d(x):
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, date):
        return x if type(x) is date else date(x.year, x.month, x.day)
    return date.fromisoformat(str(x)[:10])
