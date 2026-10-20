"""Tests for nyse_calendar.py and spinoff_tracker.py (no network: SEC and Yahoo are replaced by fakes)."""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import live_spinoffs as ls  # noqa: E402
import nyse_calendar as nc  # noqa: E402
import spinoff_tracker as st  # noqa: E402

def D(x):
    return x if isinstance(x, date) else date.fromisoformat(x)


CAL = nc.rule_trading_days(D('2023-01-01'), D('2029-12-31'))


# ------------------------------------------------------------------------------------------------ NYSE calendar
NYSE = {  # published NYSE full-day closures (one-off closures such as 9 Jan 2025 are not rules; see below)
    2025: ['2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04',
           '2025-09-01', '2025-11-27', '2025-12-25'],
    2026: ['2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25', '2026-06-19', '2026-07-03',
           '2026-09-07', '2026-11-26', '2026-12-25'],
    2027: ['2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31', '2027-06-18', '2027-07-05',
           '2027-09-06', '2027-11-25', '2027-12-24'],
}


@pytest.mark.parametrize('year', sorted(NYSE))
def test_nyse_holidays_match_published_lists(year):
    assert sorted(nc.nyse_holidays(year)) == [D(x) for x in NYSE[year]]


def test_easter():
    assert nc.easter(2025) == D('2025-04-20')
    assert nc.easter(2026) == D('2026-04-05')
    assert nc.easter(2027) == D('2027-03-28')
    assert nc.easter(2038) == D('2038-04-25')
    assert nc.easter(2285) == D('2285-03-22')


def test_weekend_observance():
    # New Year's Day on a Saturday is not observed: 31 Dec 2021 was a normal trading day
    assert D('2022-01-01').weekday() == 5
    assert nc.is_rule_trading_day(D('2021-12-31'))
    assert not any(d.month == 1 and d.day <= 3 for d in nc.nyse_holidays(2022))
    assert nc.is_rule_trading_day(D('2027-12-31'))            # 1 Jan 2028 is a Saturday too
    # other Saturday holidays move to Friday, Sunday holidays to Monday
    assert not nc.is_rule_trading_day(D('2021-12-24'))        # Christmas 2021 on Saturday
    assert not nc.is_rule_trading_day(D('2022-12-26'))        # Christmas 2022 on Sunday
    assert not nc.is_rule_trading_day(D('2022-06-20'))        # Juneteenth 2022 on Sunday
    assert not nc.is_rule_trading_day(D('2026-07-03'))        # 4 July 2026 on Saturday
    assert not nc.is_rule_trading_day(D('2027-07-05'))        # 4 July 2027 on Sunday
    assert not nc.is_rule_trading_day(D('2023-01-02'))        # New Year 2023 on Sunday
    assert nc.is_rule_trading_day(D('2021-06-18'))            # Juneteenth only from 2022
    # NYSE trades on Columbus Day and Veterans Day (the US federal calendar does not)
    assert nc.is_rule_trading_day(D('2026-10-12')) and nc.is_rule_trading_day(D('2026-11-11'))


def test_one_off_closures():
    # the known one-off closures are part of the rules, so a calendar without SPY history is exact for the past
    for d in ('2001-09-11', '2001-09-14', '2004-06-11', '2007-01-02', '2012-10-29', '2012-10-30', '2018-12-05',
              '2025-01-09'):
        assert not nc.is_rule_trading_day(D(d)), d
    assert nc.is_rule_trading_day(D('2025-01-08')) and nc.is_rule_trading_day(D('2025-01-10'))
    assert sorted(nc.nyse_closures(2025)) == sorted([D(x) for x in NYSE[2025]] + [D('2025-01-09')])
    days, last = nc.build_calendar(D('2025-01-01'), D('2025-01-31'), [])
    assert last is None and D('2025-01-09') not in days and len(days) == 20
    # a future closure announced at short notice comes from SPY's real history once it covers that day
    surprise = D('2026-10-14')
    actual = [d for d in nc.rule_trading_days(D('2026-10-01'), D('2026-10-30')) if d != surprise]
    days, last = nc.build_calendar(D('2026-10-01'), D('2026-12-31'), actual)
    assert last == D('2026-10-30') and surprise not in days and D('2026-10-13') in days and D('2026-10-15') in days
    assert D('2026-11-26') not in days and D('2026-11-27') in days   # projected part uses the rules


def test_day_21_and_252_counting():
    assert nc.nth_trading_day(CAL, D('2026-10-01'), 21) == D('2026-10-30')
    assert nc.nth_trading_day(CAL, D('2026-10-30'), 4) == D('2026-11-05')
    assert nc.nth_trading_day(CAL, D('2026-10-30'), 252) == D('2027-11-02')
    # Thanksgiving inside the window: 25 Nov 2025 + 4 trading days = 2 Dec
    assert nc.nth_trading_day(CAL, D('2025-11-25'), 4) == D('2025-12-02')
    # a non-trading start counts from the next trading day
    assert nc.nth_trading_day(CAL, D('2026-10-31'), 0) == D('2026-11-02')
    assert nc.nth_trading_day(CAL, D('2026-11-26'), 1) == D('2026-11-30')
    assert nc.nth_trading_day(CAL, D('2026-10-01'), 10 ** 6) is None
    # independent check: 252 weekdays-minus-holidays
    span = pd.bdate_range('2026-10-31', '2027-11-02')
    hol = set(nc.nyse_holidays(2026)) | set(nc.nyse_holidays(2027))
    assert sum(1 for d in span if d.date() not in hol) == 252


def test_live_spinoffs_docstring():
    assert 'about 30%' not in ls.__doc__ and '1/8' in ls.__doc__ and not hasattr(ls, 'trading_days')


# ------------------------------------------------------------------------------------------------ SEC filings
def idx_line(form, company, cik, filed, acc):
    return f'{form:<17}{company:<62}{cik:<12}{filed}  edgar/data/{cik}/{acc}.txt         '


class FakeSEC:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def get(self, url, timeout):
        self.calls.append(url)
        r = type('R', (), {})()
        q = url.split('/full-index/')[1][:9] if '/full-index/' in url else ''
        r.status_code = 500 if q in self.fail else 200
        r.text = '\n'.join(['Form Type   Company Name', '-' * 40,
                            idx_line('10-12B', 'Vylor Inc.', 2128626, '2026-06-29', '0001193125-26-123456'),
                            idx_line('10-12B/A', 'Vylor Inc.', 2128626, '2026-08-01', '0001193125-26-223456'),
                            idx_line('10-12B', 'Future Co', 7, '2026-12-01', '0000000007-26-000001'),
                            idx_line('10-K', 'Other Co', 1, '2026-06-01', '0000000001-26-000001')]) \
            if q.startswith('2026/QTR3') else 'Form Type\n'
        return r


def test_form10_filings_parsing_and_failed_quarters(monkeypatch):
    monkeypatch.setattr(ls.time, 'sleep', lambda s: None)
    errors = []
    f = ls.form10_filings(FakeSEC(), 2, today=D('2026-10-20'), errors=errors)
    assert errors == [] and f['cik'].tolist() == [2128626]          # amendments, later and other forms ignored
    assert f['filename'].iloc[0] == 'edgar/data/2128626/0001193125-26-123456.txt'
    assert st.filing_url(2128626, f['filename'].iloc[0]) == \
        'https://www.sec.gov/Archives/edgar/data/2128626/000119312526123456/0001193125-26-123456-index.htm'
    assert st.filing_url(5) == 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=5&type=10-12B'
    sec = FakeSEC(fail={'2025/QTR2'})
    ls.form10_filings(sec, 2, today=D('2026-10-20'), errors=errors)
    assert errors == ['SEC form.idx 2025 Q2: HTTP 500'] and sum('2025/QTR2' in u for u in sec.calls) == 3
    assert not any('2026/QTR4' not in u and '2027' in u for u in sec.calls)
    # sec_filings turns a failed quarter into a failed scan (so yesterday's pipeline is kept)
    monkeypatch.setattr(ls, 'sec_session', lambda: FakeSEC(fail={'2025/QTR2'}))
    with pytest.raises(RuntimeError):
        st.sec_filings(2, D('2026-10-20'), [])


# ------------------------------------------------------------------------------------------------ fake prices
def frame(days, close=10.0, volume=1_000_000, adj=None):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    n = len(idx)
    c = [close(i) if callable(close) else close for i in range(n)]
    v = [volume(i) if callable(volume) else volume for i in range(n)]
    a = [adj(i) for i in range(n)] if callable(adj) else c
    return pd.DataFrame({'Open': [x * 1.01 for x in c], 'High': c, 'Low': c, 'Close': c, 'Adj Close': a,
                         'Volume': v}, index=idx)


def days_between(a, b):
    return [d for d in CAL if D(a) <= d <= D(b)]


def filing(cik='1', company='Vylor Inc.', f10='2026-06-29', ticker='VYLR'):
    return {'cik': cik, 'company': company, 'form10_date': D(f10), 'form10_url': st.filing_url(cik),
            'ticker': ticker, 'exchange': 'NYSE' if ticker else None}


def at(today, first='2026-10-01', **kw):
    """Classify VYLR on `today` with prices up to the day before (a pre-open run)."""
    h = frame(days_between(first, D(today) - timedelta(days=1)), **kw)
    return st.classify(filing(), h, D(today), CAL)


# ------------------------------------------------------------------------------------------------ statuses
@pytest.mark.parametrize('today,status', [
    ('2026-10-29', 'watch'),     # day before the buy day
    ('2026-10-30', 'BUY'),       # buy day (trading day 21)
    ('2026-11-05', 'BUY'),       # window end (buy day + 4)
    ('2026-11-06', 'HOLD'),      # day after the window: missed if not bought
    ('2027-11-01', 'HOLD'),      # day before the model sell
    ('2027-11-02', 'SELL'),      # model sell day (252 trading days after the buy)
    ('2027-11-12', 'SELL'),      # sell + 10 calendar days
    ('2027-11-13', 'done'),      # sell + 11
])
def test_status_boundaries(today, status):
    it = at(today)
    assert it['status'] == status
    assert (it['first_trade'], it['buy_date'], it['buy_window_end'], it['model_sell_date']) == \
        ('2026-10-01', '2026-10-30', '2026-11-05', '2027-11-02')
    if status != 'watch':
        assert it['price_before_buy'] == 10.0 and it['avg_dollar_volume'] == 10_000_000
    if status in ('HOLD', 'SELL', 'done'):
        assert it['buy_open'] == 10.1


def test_buy_date_uses_own_history_when_long_enough():
    h = frame(days_between('2026-10-01', '2026-11-20'))
    h = h.drop(pd.Timestamp('2026-10-15'))      # a day missing in the stock's own history
    it = st.classify(filing(), h, D('2026-11-23'), CAL)
    assert it['buy_date'] == '2026-11-02'      # 21st row of its own history, as the backtest counted


def test_liquidity_check():
    # exactly $1 and exactly $100,000 a day pass
    assert at('2026-10-30', close=1.0, volume=100_000)['status'] == 'BUY'
    it = at('2026-10-30', close=0.99, volume=10_000_000)
    assert it['status'] == 'skip' and '$1' in it['note']
    it = at('2026-10-30', close=5.0, volume=19_999)
    assert it['status'] == 'skip' and 'dollar volume' in it['note'] and it['avg_dollar_volume'] == 99_995
    # only the last close before the buy day counts for the price; volume is the 20-day average
    assert at('2026-10-30', close=lambda i: 0.5 if i < 20 else 1.2)['status'] == 'BUY'
    # day 0 (outside the 20-day window) does not count towards volume
    assert at('2026-10-30', close=1.0, volume=lambda i: 10 ** 9 if i == 0 else 0)['status'] == 'skip'
    assert at('2027-12-01', close=0.5)['status'] == 'skip'      # stays skip later on


def test_waiting_and_not_new():
    assert st.classify(filing(ticker=None), None, D('2026-10-20'), CAL)['status'] == 'waiting'
    assert st.classify(filing(), pd.DataFrame(), D('2026-10-20'), CAL)['status'] == 'waiting'
    old = frame(days_between('2023-03-01', '2026-10-19'))
    assert st.classify(filing(), old, D('2026-10-20'), CAL)['status'] == 'not_new'
    late = frame(days_between('2027-07-01', '2027-07-10'))
    it = st.classify(filing(f10='2026-06-29'), late, D('2027-07-12'), CAL)
    assert it['status'] == 'not_new' and 'more than a year' in it['note']


def test_genuine_false_is_noted():
    h = frame(days_between('2026-10-01', '2026-10-29'))
    it = st.classify(filing(), h, D('2026-10-30'), CAL, {'genuine': False})
    assert it['status'] == 'BUY' and 'not a genuine spin-off' in it['note']


# ------------------------------------------------------------------------------------------------ alerts
def item(ticker='VYLR', cik='1', company='Vylor Inc.', status='BUY', first='2026-10-01'):
    buy = nc.nth_trading_day(CAL, D(first), 21)
    return {'id': cik, 'company': company, 'ticker': ticker, 'status': status, 'first_trade': first,
            'buy_date': buy.isoformat(), 'buy_window_end': nc.nth_trading_day(CAL, buy, 4).isoformat(),
            'model_sell_date': nc.nth_trading_day(CAL, buy, 252).isoformat(), 'form10_url': 'u'}


def user(trades=(), deposits=(), checks=None, **settings):
    s = dict(st.DEFAULTS, dismissed={})
    s.update(settings)
    tr = [{'id': str(i), 'date': D(d), 'type': ty, 'ticker': t, 'shares': sh, 'amount_gbp': amt, 'note': '',
           'created_at': f'{d}T12:00:{i:02d}Z'} for i, (d, ty, t, sh, amt) in enumerate(trades)]
    dep = [{'id': str(i), 'date': D(d), 'amount_gbp': a, 'note': '', 'created_at': ''} for i, (d, a) in enumerate(deposits)]
    return {'trades': tr, 'deposits': dep, 'settings': s, 'checks': checks or {}, 'pipeline': None}


def alerts(today, items, data, closes=None, fund=None, fx=None):
    return st.build_alerts(D(today), items, data, CAL, closes or {}, fund, fx)


def test_buy_new_text_and_unchecked():
    a = alerts('2026-10-30', [item()], user())
    assert a['noteworthy'] and a['push'] == 'BUY VYLR today'
    (x,) = a['items']
    assert x['type'] == 'BUY_NEW' and x['ticker'] == 'VYLR' and x['date'] == '2026-10-30'
    # 30 Oct 2026: the UK is already on GMT, New York still on EDT -> the open is 13:30 UK
    assert x['text'].startswith('Buy VYLR (Vylor) today after the US open (13:30 UK): about 1/8 of your account.')
    assert 'Window closes Thu 5 Nov.' in x['text'] and '(not yet checked as a real spin-off)' in x['text']
    a = alerts('2026-10-30', [item()], user(checks={'1': {'genuine': True}}, slice=10))
    assert 'not yet checked' not in a['lines'][0] and '1/10 of your account' in a['lines'][0]
    assert 'inconclusive' in alerts('2026-10-30', [item()], user(checks={'1': {'genuine': None}}))['lines'][0]


def test_us_open_in_uk_time():
    assert st.us_open_uk(D('2026-07-01')) == '14:30'
    assert st.us_open_uk(D('2026-03-10')) == '13:30'     # US on DST, UK not yet
    assert st.us_open_uk(D('2026-12-01')) == '14:30'


def test_buy_skipped_when_held_dismissed_or_not_genuine():
    held = user(trades=[('2026-10-30', 'buy', 'VYLR', 3, 200.0)])
    assert alerts('2026-10-30', [item()], held)['push'] == 'No trades today.'
    assert not alerts('2026-10-30', [item()], user(dismissed={'VYLR': 'Not available on Trading 212'}))['noteworthy']
    assert not alerts('2026-10-30', [item()], user(checks={'1': {'genuine': False}}))['noteworthy']
    a = alerts('2026-10-30', [item()], user())   # a sold-out position is not held: still a buy
    assert a['noteworthy']
    sold = user(trades=[('2026-10-30', 'buy', 'VYLR', 3, 200.0), ('2026-10-31', 'sell', 'VYLR', 3, 210.0)])
    assert alerts('2026-11-02', [item()], sold)['items'][0]['type'] == 'BUY_REMINDER'


def test_reminder_and_last_chance():
    a = alerts('2026-11-02', [item()], user())
    assert a['items'][0]['type'] == 'BUY_REMINDER' and a['push'] == 'BUY VYLR (window to Thu 5 Nov)'
    assert a['lines'][0].startswith('Still time to buy VYLR (Vylor)')
    a = alerts('2026-11-05', [item()], user())
    assert a['items'][0]['type'] == 'LAST_CHANCE' and a['push'] == 'LAST DAY to buy VYLR'
    assert not alerts('2026-11-06', [item(status='HOLD')], user())['noteworthy']


def test_sell_today_and_overdue():
    # bought 30 Oct 2026 (trading day) -> sell 2 Nov 2027
    data = user(trades=[('2026-10-30', 'buy', 'VYLR', 2.5, 150.0)])
    assert not alerts('2027-11-01', [], data)['noteworthy']
    a = alerts('2027-11-02', [], data)
    assert a['push'] == 'SELL VYLR (12 months up)'
    assert a['lines'][0].startswith('Sell all 2.5 VYLR today after the US open') and 'bought Fri 30 Oct 2026' in a['lines'][0]
    a = alerts('2027-11-04', [item(status='HOLD')], data)
    assert a['push'] == 'SELL VYLR (overdue since 2 Nov)' and 'overdue since Tue 2 Nov' in a['lines'][0]
    # buy logged on a Saturday counts from the next trading day (Mon 2 Nov 2026 -> 3 Nov 2027)
    sat = user(trades=[('2026-10-31', 'buy', 'VYLR', 1, 70.0)])
    assert not alerts('2027-11-02', [], sat)['noteworthy'] and alerts('2027-11-03', [], sat)['noteworthy']
    # the position's own buy date is the buy that opened it (sold out then rebought)
    re = user(trades=[('2025-06-02', 'buy', 'ABC', 1, 10.0), ('2025-07-01', 'sell', 'ABC', 1, 11.0),
                      ('2026-10-30', 'buy', 'ABC', 1, 10.0), ('2026-11-10', 'buy', 'ABC', 1, 10.0)])
    assert st.positions(re['trades'])['ABC']['opened'] == D('2026-10-30')
    assert alerts('2027-11-02', [], re)['push'] == 'SELL ABC (12 months up)'
    # a part sale keeps the position open; a full sale ends it
    part = user(trades=[('2026-10-30', 'buy', 'ABC', 2, 10.0), ('2027-01-05', 'sell', 'ABC', 1, 6.0)])
    assert alerts('2027-11-02', [], part)['noteworthy']
    full = user(trades=[('2026-10-30', 'buy', 'ABC', 2, 10.0), ('2027-01-05', 'sell', 'ABC', 2 - 1e-9, 12.0)])
    assert not alerts('2027-11-02', [], full)['noteworthy']


def test_info_items_do_not_make_noise():
    new = item(status='watch', first='2026-10-28')
    a = alerts('2026-10-30', [new], user())
    assert not a['noteworthy'] and a['push'] == 'No trades today.'
    assert [x['type'] for x in a['items']] == ['NEW_LISTING'] and 'Buy day' in a['lines'][0]
    assert alerts('2026-11-02', [new], user())['items'][0]['type'] == 'NEW_LISTING'   # 28, 29, 30 Oct
    assert alerts('2026-11-03', [new], user())['items'] == []   # 28 Oct is more than 3 trading days back
    soon = item(status='watch', first='2026-10-01')
    assert [x['type'] for x in alerts('2026-10-26', [soon], user())['items']] == ['UPCOMING_BUY']


def test_push_is_short_and_ordered():
    items = [item(ticker=f'T{i:02d}', cik=str(i), company=f'Company {i} Inc.') for i in range(30)]
    held = user(trades=[('2025-10-01', 'buy', 'OLDX', 1, 10.0)])
    a = alerts('2026-10-30', items, held)
    assert a['push'].startswith('BUY T00 today; BUY T01 today') and len(a['push']) <= 190
    assert a['push'].endswith('more') and '\n' not in a['push']
    a = alerts('2026-10-30', items[:1], held)
    assert a['push'] == 'BUY T00 today; SELL OLDX (overdue since 2 Oct)'


def test_account_estimate_and_trim_text():
    idx = pd.Series(100.0, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days_between('2026-01-02', '2026-10-29')]))
    fx = pd.Series(1.25, index=idx.index)
    closes = {'AAA': pd.Series(50.0, index=idx.index), 'BBB': pd.Series(25.0, index=idx.index)}
    data = user(deposits=[('2026-01-02', 2000.0)],
                trades=[('2026-03-02', 'buy', 'AAA', 30, 1000.0), ('2026-04-01', 'buy', 'BBB', 30, 600.0)])
    a = alerts('2026-10-30', [item()], data, closes, idx, fx)
    acct = a['account']
    assert acct['index_value_gbp'] == 400.0 and acct['spinoffs_gbp'] == {'AAA': 1200.0, 'BBB': 600.0}
    assert acct['total_gbp'] == 2200.0
    text = a['lines'][0]
    assert 'about £275' in text and 'Sell all your VUAG (about £400)' not in text
    assert 'Pay for it by selling about £275 of VUAG' in text
    data['deposits'][0]['amount_gbp'] = 1700.0      # index fund now holds £100
    text = alerts('2026-10-30', [item()], data, closes, idx, fx)['lines'][0]
    assert 'about £238' in text
    assert 'Sell all your VUAG (about £100) and trim about £138 from your spin-offs' in text
    assert 'AAA £92, BBB £46' in text


def test_sell_alert_shows_return():
    idx = pd.DatetimeIndex([pd.Timestamp('2027-11-01')])
    data = user(trades=[('2026-10-30', 'buy', 'VYLR', 10, 500.0)], deposits=[('2026-10-01', 1000.0)])
    a = alerts('2027-11-02', [], data, {'VYLR': pd.Series(100.0, index=idx)}, pd.Series(10.0, index=idx),
               pd.Series(1.25, index=idx))
    assert 'about £799, +59.8% on what you paid' in a['lines'][0]


# ------------------------------------------------------------------------------------------------ exports
def write(p, body):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body if isinstance(body, str) else json.dumps(body))


def test_malformed_exports_are_skipped(tmp_path):
    db = tmp_path / 'db'
    write(db / 'trades' / 'good.json', {'date': '2026-10-30', 'type': 'buy', 'ticker': ' vylr ', 'shares': 2,
                                        'amount_gbp': 150, 'note': '', 'created_at': ''})
    write(db / 'trades' / 'broken.json', '{"date": "2026-')
    write(db / 'trades' / 'list.json', [1, 2])
    write(db / 'trades' / 'neg.json', {'date': '2026-10-30', 'type': 'buy', 'ticker': 'X', 'shares': -1, 'amount_gbp': 1})
    write(db / 'trades' / 'type.json', {'date': '2026-10-30', 'type': 'hold', 'ticker': 'X', 'shares': 1, 'amount_gbp': 1})
    write(db / 'trades' / 'date.json', {'date': '30/10/2026', 'type': 'buy', 'ticker': 'X', 'shares': 1, 'amount_gbp': 1})
    write(db / 'trades' / 'nan.json', '{"date": "2026-10-30", "type": "buy", "ticker": "X", "shares": NaN, "amount_gbp": 1}')
    write(db / 'deposits' / 'a.json', {'date': '2026-01-05', 'amount_gbp': 1666.66})
    write(db / 'deposits' / 'w.json', {'date': '2026-02-05', 'amount_gbp': -100})
    write(db / 'deposits' / 'bad.json', {'date': '2026-02-30', 'amount_gbp': 5})
    write(db / 'settings' / 'main.json', {'index_fund': 'XYZ', 'slice': 0, 'dismissed': {'abc': 'Skipped'}})
    write(db / 'checks' / '0002128626.json', {'genuine': False, 'parent': 'P', 'business': 'B', 'note': 'n'})
    write(db / 'checks' / '5.json', {'genuine': 'yes'})
    errors = []
    data = st.load_exports(db, errors)
    assert [t['ticker'] for t in data['trades']] == ['VYLR']
    assert sorted(d['amount_gbp'] for d in data['deposits']) == [-100, 1666.66]
    assert data['settings']['index_fund'] == 'VUAG' and data['settings']['slice'] == 8
    assert data['settings']['dismissed'] == {'ABC': 'Skipped'}
    assert data['checks']['2128626']['genuine'] is False and data['checks']['5']['genuine'] is None
    joined = '\n'.join(errors)
    for bad in ('trades/broken', 'trades/list', 'trades/neg', 'trades/type', 'trades/date', 'trades/nan',
                'deposits/bad', 'index_fund', 'slice', 'checks/5'):
        assert bad in joined
    assert st.load_exports(None, []) ['trades'] == []
    errs = []
    assert st.load_exports(tmp_path / 'missing', errs)['trades'] == [] and errs


# ------------------------------------------------------------------------------------------------ full scan (fakes)
TODAY = D('2026-10-30')
OLDX_BUY = CAL[CAL.index(TODAY) - 252]          # bought so that today is its sell day


def fake_yahoo(symbol, start=None):
    actual = [d for d in CAL if d <= TODAY]     # Yahoo also has today's bar: the scan must cut it
    if symbol == 'SPY':
        h = frame(actual, close=500.0)
    elif symbol == 'VYLR':
        h = frame([d for d in actual if d >= D('2026-10-01')], close=70.0)
    elif symbol == 'Q':
        h = frame([d for d in actual if d >= D('2025-10-27')], close=lambda i: 70.0 + i * 0.1)
    elif symbol == 'PNY':
        h = frame([d for d in actual if d >= D('2026-05-01')], close=0.4)
    elif symbol == 'OLD':
        h = frame([d for d in actual if d >= D('2023-01-03')], close=20.0)
    elif symbol == 'OLDX':
        h = frame([d for d in actual if d >= D('2024-06-03')], close=12.0)
    elif symbol == 'VUAG.L':
        h = frame(actual, close=100.0)
    elif symbol == 'EQQQ.L':
        h = frame(actual, close=54000.0, adj=lambda i: 50000.0 + i)       # pence
    elif symbol == 'GBPUSD=X':
        h = frame(actual, close=1.333333333)
    else:
        return pd.DataFrame()
    return h if start is None else h[h.index >= pd.Timestamp(start)]


FILINGS = [
    {'cik': '2128626', 'company': 'Vylor Inc.', 'form10_date': D('2026-06-29'), 'ticker': 'VYLR', 'exchange': 'NYSE'},
    {'cik': '2058873', 'company': 'Novus SpinCo 1, Inc.', 'form10_date': D('2025-04-24'), 'ticker': 'Q', 'exchange': 'NYSE'},
    {'cik': '3', 'company': 'Penny Spinco Corp', 'form10_date': D('2026-03-01'), 'ticker': 'PNY', 'exchange': 'Nasdaq'},
    {'cik': '4', 'company': 'Old Co', 'form10_date': D('2026-02-01'), 'ticker': 'OLD', 'exchange': 'NYSE'},
    {'cik': '5', 'company': 'Not Yet Inc.', 'form10_date': D('2026-09-01'), 'ticker': None, 'exchange': None},
    {'cik': '6', 'company': 'No Prices Inc.', 'form10_date': D('2026-09-02'), 'ticker': 'NOPX', 'exchange': 'NYSE'},
]


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(st, 'yahoo_history', fake_yahoo)
    monkeypatch.setattr(st, 'sec_filings',
                        lambda years, today, errors: [dict(f, form10_url=st.filing_url(f['cik'])) for f in FILINGS])


def make_db(path, fund='EQQQ'):
    write(path / 'trades' / 't1.json', {'date': '2025-11-25', 'type': 'buy', 'ticker': 'Q', 'shares': 10,
                                        'amount_gbp': 560, 'note': '', 'created_at': '2025-11-25T15:00:00Z'})
    write(path / 'trades' / 't2.json', {'date': OLDX_BUY.isoformat(), 'type': 'buy', 'ticker': 'OLDX', 'shares': 50,
                                        'amount_gbp': 450, 'note': '', 'created_at': '2025-10-29T15:00:00Z'})
    write(path / 'deposits' / 'd1.json', {'date': '2025-10-01', 'amount_gbp': 5000, 'note': ''})
    write(path / 'settings' / 'main.json', {'index_fund': fund, 'slice': 8, 'monthly_deposit': 1666.66, 'dismissed': {}})
    write(path / 'checks' / '2058873.json', {'genuine': True, 'parent': 'Novus', 'business': 'x', 'note': ''})
    write(path / 'trades' / 'junk.json', 'not json')


def load(p):
    return json.loads(Path(p).read_text())


def test_full_scan(tmp_path, fakes):
    db, out = tmp_path / 'db', tmp_path / 'out'
    make_db(db)
    (out / 'docs' / 'prices').mkdir(parents=True)
    (out / 'docs' / 'prices' / 'STALE.json').write_text('{}')
    r = st.scan(out, db, years=2, today=TODAY, trigger_id='trig_1')
    docs = out / 'docs'
    status = load(docs / 'market' / 'status.json')
    assert status['today_ny'] == '2026-10-30' and status['last_market_day'] == '2026-10-29'
    assert status['scan_ok'] is True and status['counts'] == {'filings': 6, 'tickers': 5}
    assert status['trigger_id'] == 'trig_1' and status['next_check'] == st.NEXT_CHECK
    assert len(status['errors']) == 1 and 'trades/junk' in status['errors'][0]
    assert set(status) == {'generated_at', 'today_ny', 'last_market_day', 'scan_ok', 'errors', 'counts',
                           'trigger_id', 'next_check'}
    items = {it['ticker'] or it['company']: it for it in load(docs / 'market' / 'pipeline.json')['items']}
    assert {k: v['status'] for k, v in items.items()} == {
        'VYLR': 'BUY', 'Q': 'HOLD', 'PNY': 'skip', 'OLD': 'not_new', 'Not Yet Inc.': 'waiting', 'NOPX': 'waiting'}
    v = items['VYLR']
    assert (v['id'], v['buy_date'], v['buy_window_end'], v['model_sell_date']) == \
        ('2128626', '2026-10-30', '2026-11-05', '2027-11-02')
    assert v['last_close'] == 70.0 and v['buy_open'] is None and v['avg_dollar_volume'] == 70_000_000
    assert set(v) == {'id', 'company', 'ticker', 'exchange', 'form10_date', 'form10_url', 'first_trade', 'buy_date',
                      'buy_window_end', 'model_sell_date', 'status', 'note', 'last_close', 'price_before_buy',
                      'avg_dollar_volume', 'buy_open'}
    # prices: watched/held spin-offs + benchmarks, no stale docs, today's unfinished bar cut
    prices = {p.stem: load(p) for p in (docs / 'prices').glob('*.json')}
    assert set(prices) == {'VYLR', 'Q', 'OLDX', 'VUAG', 'EQQQ', 'GBPUSD'}
    start = '2025-09-24'           # first deposit (1 Oct 2025) - 7 days, earlier than today - 400 days
    assert min(D('2025-10-01') - timedelta(days=7), TODAY - timedelta(days=400)).isoformat() == start
    for k, p in prices.items():
        assert p['id'] == k and p['dates'] == sorted(p['dates']) and len(p['dates']) == len(p['close'])
        assert p['dates'][-1] == '2026-10-29' and p['dates'][0] >= start
    assert prices['VYLR']['dates'][0] == '2026-10-01' and prices['VYLR']['currency'] == 'USD'
    assert prices['VYLR']['kind'] == 'spinoff' and prices['Q']['dates'][0] == '2025-10-27'
    assert prices['OLDX']['dates'][0] >= start
    assert prices['EQQQ']['currency'] == 'GBP' and prices['EQQQ']['kind'] == 'index'
    raw = fake_yahoo('EQQQ.L', start)['Adj Close']                 # Adj Close in pence -> pounds
    assert prices['EQQQ']['close'] == [round(x / 100, 4) for x in raw.iloc[:-1]]
    assert prices['EQQQ']['dates'] == [d.date().isoformat() for d in raw.index[:-1]]
    assert all(505 < x < 510 for x in prices['EQQQ']['close'])
    assert prices['VUAG']['close'][-1] == 100.0
    assert prices['GBPUSD']['close'][-1] == 1.3333 and prices['GBPUSD']['kind'] == 'fx'
    cal = load(docs / 'market' / 'calendar.json')
    assert cal['days'][0] >= start and cal['last_actual'] == '2026-10-29'
    assert cal['days'][-1] <= (TODAY + timedelta(days=600)).isoformat() and '2027-11-02' in cal['days']
    assert '2027-03-26' not in cal['days'] and '2026-11-11' in cal['days']
    # alerts: buy VYLR (unchecked), sell OLDX today; Q is checked and held
    al = load(out / 'alerts.json')
    assert al['noteworthy'] and al['push'] == 'BUY VYLR today; SELL OLDX (12 months up)'
    types = {x['ticker']: x['type'] for x in al['items']}
    assert types == {'VYLR': 'BUY_NEW', 'OLDX': 'SELL'}
    assert '(not yet checked as a real spin-off)' in al['items'][0]['text'] and 'EQQQ' in al['items'][0]['text']
    assert load(out / 'needs_check.json') == [{'cik': '2128626', 'ticker': 'VYLR', 'company': 'Vylor Inc.',
                                              'form10_url': st.filing_url('2128626')}]
    assert all(p.stat().st_size < st.DOC_LIMIT for p in docs.rglob('*.json'))
    assert r['statuses']['BUY'] == 1


def test_scan_without_db_and_real_clock(tmp_path, fakes):
    now = datetime(2026, 10, 30, 5, 45, tzinfo=timezone.utc)          # 06:45 UK? (GMT) = 01:45 New York
    r = st.scan(tmp_path, None, now=now)
    assert r['today_ny'] == '2026-10-30' and r['last_market_day'] == '2026-10-29'
    assert load(tmp_path / 'alerts.json')['push'] == 'BUY VYLR today'
    late = datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc)         # 17:00 New York: today's bar is final
    assert st.scan(tmp_path, None, now=late)['last_market_day'] == '2026-10-30'


def test_final_bar_dates():
    d = D('2026-10-20')
    assert st.final_bar_dates(d) == (D('2026-10-19'),) * 3
    us, lse, fx = st.final_bar_dates(d, datetime(2026, 10, 20, 5, 45, tzinfo=timezone.utc))
    assert (us, lse, fx) == (D('2026-10-19'), D('2026-10-19'), D('2026-10-20'))
    us, lse, fx = st.final_bar_dates(d, datetime(2026, 10, 20, 16, 0, tzinfo=timezone.utc))   # 17:00 UK, 12:00 NY
    assert (us, lse) == (D('2026-10-19'), D('2026-10-20'))


def test_scan_when_sec_fails(tmp_path, fakes, monkeypatch):
    db, out = tmp_path / 'db', tmp_path / 'out'
    make_db(db)
    st.scan(tmp_path / 'first', db, today=TODAY)
    write(db / 'market' / 'pipeline.json', load(tmp_path / 'first' / 'docs' / 'market' / 'pipeline.json'))

    def boom(years, today, errors):
        raise ConnectionError('sec.gov unreachable')
    monkeypatch.setattr(st, 'sec_filings', boom)
    r = st.scan(out, db, today=TODAY)
    assert not (out / 'docs' / 'market' / 'pipeline.json').exists()
    status = load(out / 'docs' / 'market' / 'status.json')
    assert status['scan_ok'] is False and any('SEC scan failed' in e for e in status['errors'])
    assert status['counts'] == {'filings': 0, 'tickers': 0}
    assert {p.stem for p in (out / 'docs' / 'prices').glob('*.json')} == {'VYLR', 'Q', 'OLDX', 'VUAG', 'EQQQ', 'GBPUSD'}
    # from yesterday's list; the push says the scan was incomplete
    assert r['alerts']['push'] == 'BUY VYLR today; SELL OLDX (12 months up); scan incomplete'
    # without yesterday's list: sells still work, buys cannot be known
    (db / 'market' / 'pipeline.json').unlink()
    r = st.scan(out, db, today=TODAY)
    assert r['alerts']['push'] == 'SELL OLDX (12 months up); buys not checked'
    assert r['alerts']['scan_ok'] is False and 'new buys were not checked' in r['alerts']['lines'][-1]
    assert {p.stem for p in (out / 'docs' / 'prices').glob('*.json')} == {'Q', 'OLDX', 'VUAG', 'EQQQ', 'GBPUSD'}
    (db / 'trades' / 't2.json').unlink()
    r = st.scan(out, db, today=TODAY)
    assert not r['alerts']['noteworthy'] and r['alerts']['push'].startswith('No sells due. The spin-off scan failed')


def test_scan_when_yahoo_is_down(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(st, 'yahoo_history', lambda symbol, start=None: pd.DataFrame())
    r = st.scan(tmp_path, None, today=TODAY)
    assert r['scan_ok'] is False and not (tmp_path / 'docs' / 'market' / 'pipeline.json').exists()
    assert not (tmp_path / 'docs' / 'prices').exists()
    assert any('SPY' in e for e in r['errors']) and any('VUAG.L' in e for e in r['errors'])
    cal = load(tmp_path / 'docs' / 'market' / 'calendar.json')
    assert cal['last_actual'] is None and '2026-10-29' in cal['days']


def test_odd_price_history_does_not_stop_the_run(tmp_path, fakes, monkeypatch):
    db = tmp_path / 'db'
    st.scan(tmp_path / 'first', None, today=TODAY)
    write(db / 'market' / 'pipeline.json', load(tmp_path / 'first' / 'docs' / 'market' / 'pipeline.json'))

    def odd(symbol, start=None):
        if symbol == 'Q':
            return pd.DataFrame({'Close': ['x', 'y']}, index=pd.DatetimeIndex(['2025-10-27', '2025-10-28']))
        if symbol == 'PNY':
            raise RuntimeError('Yahoo hiccup')
        return fake_yahoo(symbol, start)
    monkeypatch.setattr(st, 'yahoo_history', odd)
    # without yesterday's pipeline the two statuses are unknown: the run goes on, but the pipeline is not written
    r = st.scan(tmp_path / 'a', None, today=TODAY)
    assert not r['scan_ok'] and r['statuses']['BUY'] == 1 and r['alerts']['push'] == 'BUY VYLR today; scan incomplete'
    assert not (tmp_path / 'a' / 'docs' / 'market' / 'pipeline.json').exists()
    assert any(e.startswith('Yahoo Finance: no prices for Q today (price history could not be read')
               for e in r['errors'])
    assert any('PNY today (download failed (RuntimeError: Yahoo hiccup))' in e for e in r['errors'])
    # with yesterday's pipeline both are carried forward and the pipeline is written
    r = st.scan(tmp_path / 'b', db, today=TODAY)
    assert r['scan_ok'] and r['alerts']['push'] == 'BUY VYLR today'
    items = {it['ticker']: it for it in load(tmp_path / 'b' / 'docs' / 'market' / 'pipeline.json')['items']}
    assert items['PNY']['status'] == 'skip' and items['Q']['status'] == 'HOLD'
    assert items['Q']['note'].endswith(st.CARRIED.strip()) and items['Q']['first_trade'] == '2025-10-27'


def test_pence_glitch_fix():
    s = pd.Series([540.0, 541.0, 5.42, 543.0, 544.0, 54500.0, 546.0])
    assert st.fix_pence_glitches(s).round(2).tolist() == [540.0, 541.0, 542.0, 543.0, 544.0, 545.0, 546.0]


# ------------------------------------------------------------------------------------------------ writes
LISTING = '''2 documents from collection "market" saved as JSON files under "/x/market":
- "status"  71 bytes  version 1  "/x/market/status.json"
-   "pipeline"   15972 bytes   version 12   "/x/market/pipeline.json"
1 document from collection "prices" saved as JSON files under "/x/prices":
- "VUAG"  5444 bytes  version 3  "/x/prices/VUAG.json"
'''


def test_parse_versions():
    v = st.parse_versions(LISTING)
    assert v == {('market', 'status'): 1, ('market', 'pipeline'): 12, ('prices', 'VUAG'): 3}
    assert st.parse_versions('') == {}
    assert st.parse_versions('0 documents from collection "prices" saved as JSON files under "/x/prices":') == {}


def make_docs(out, n_prices=100, big=0):
    for i in range(n_prices):
        write(out / 'docs' / 'prices' / f'T{i:03d}.json', {'id': f'T{i:03d}', 'dates': [], 'close': []})
    for i in range(big):
        write(out / 'docs' / 'prices' / f'BIG{i}.json', {'x': 'y' * 250_000})
    for d in ('status', 'pipeline', 'calendar'):
        write(out / 'docs' / 'market' / f'{d}.json', {'d': d})


def test_writes_without_existing_docs(tmp_path):
    make_docs(tmp_path, 100)
    (tmp_path / 'writes_9.json').write_text('[]')
    rc = st.main(['writes', '--out', str(tmp_path), '--versions', str(tmp_path / 'missing.txt')])
    assert rc == 0 and not (tmp_path / 'writes_9.json').exists()
    chunks = [load(tmp_path / f'writes_{i}.json') for i in (1, 2, 3)]
    assert not (tmp_path / 'writes_4.json').exists()
    assert [len(c) for c in chunks] == [45, 45, 13]
    flat = [e for c in chunks for e in c]
    assert all('if_version' not in e for e in flat)
    assert all(e['op'] == 'set' and Path(e['file_path']).is_absolute() and Path(e['file_path']).exists() for e in flat)
    assert (flat[-1]['collection'], flat[-1]['doc_id']) == ('market', 'status')
    assert {(e['collection'], e['doc_id']) for e in flat} >= {('prices', 'T000'), ('market', 'calendar')}


def test_writes_with_existing_docs_and_size_limit(tmp_path):
    make_docs(tmp_path, 3, big=8)
    (tmp_path / 'v.txt').write_text(LISTING + '1 document from collection "prices" saved as JSON files under "/x":\n'
                                    '- "T001"  9 bytes  version 7  "/x/prices/T001.json"\n')
    chunks, _ = st.writes(tmp_path, tmp_path / 'v.txt')
    for c in chunks:
        assert len(c) <= 45 and sum(Path(e['file_path']).stat().st_size for e in c) <= 900_000
    flat = {(e['collection'], e['doc_id']): e.get('if_version') for c in chunks for e in c}
    assert flat[('market', 'status')] == 1 and flat[('market', 'pipeline')] == 12 and flat[('prices', 'T001')] == 7
    assert flat[('market', 'calendar')] is None and flat[('prices', 'T000')] is None
    assert len(chunks) == 3          # 8 docs of ~250 KB -> 3 per chunk
    assert st.writes(tmp_path / 'nothing', '/dev/null')[0] == []


# ------------------------------------------------------------------------------------------------ review fixes
def first_pipeline(tmp_path, db, today):
    """Run a normal scan (all prices available) on `today` and export its pipeline into db/market, as the
    scheduled session does; returns the exported body."""
    yh = st.yahoo_history
    st.yahoo_history = fake_yahoo
    try:
        st.scan(tmp_path / f'prev-{today}', None, today=D(today))
    finally:
        st.yahoo_history = yh
    body = load(tmp_path / f'prev-{today}' / 'docs' / 'market' / 'pipeline.json')
    write(db / 'market' / 'pipeline.json', body)
    return body


def test_yahoo_empty_for_one_spinoff_is_a_failure_not_waiting(tmp_path, fakes, monkeypatch):
    # yesterday (29 Oct) VYLR was watch; today is its buy day and Yahoo gives nothing for it
    db = tmp_path / 'db'
    make_db(db)
    first_pipeline(tmp_path, db, '2026-10-29')
    monkeypatch.setattr(st, 'yahoo_history', lambda s, start=None: pd.DataFrame() if s == 'VYLR' else fake_yahoo(s, start))
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    v = next(it for it in r['alerts']['items'] if it['ticker'] == 'VYLR')
    # still a buy alert, saying the price check could not be run; the run is marked incomplete
    assert r['alerts']['push'] == 'BUY VYLR today (check price first); SELL OLDX (12 months up); scan incomplete'
    assert v['type'] == 'BUY_NEW' and v['unchecked'] is True and '$1 / $100K check has not been run' in v['text']
    assert not r['scan_ok'] and not (tmp_path / 'out' / 'docs' / 'market' / 'pipeline.json').exists()
    assert any(e == 'Yahoo Finance: no prices for VYLR today (no prices returned, although it had prices '
                    'yesterday); its buy check could not be run' for e in r['errors'])
    assert not (tmp_path / 'out' / 'docs' / 'prices' / 'VYLR.json').exists()     # yesterday's doc is kept
    assert 'unchecked' not in next(it for it in r['alerts']['items'] if it['ticker'] == 'OLDX')
    # a day later (check already done yesterday): carried forward exactly, and the pipeline is written
    first_pipeline(tmp_path, db, '2026-10-30')
    r = st.scan(tmp_path / 'out2', db, today=D('2026-11-02'))
    assert r['scan_ok'] and r['alerts']['push'] == 'SELL OLDX (overdue since 30 Oct); BUY VYLR (window to Thu 5 Nov)'
    items = {it['ticker']: it for it in load(tmp_path / 'out2' / 'docs' / 'market' / 'pipeline.json')['items']}
    v = items['VYLR']
    assert (v['status'], v['buy_date'], v['buy_window_end'], v['model_sell_date']) == \
        ('BUY', '2026-10-30', '2026-11-05', '2027-11-02')
    assert v['avg_dollar_volume'] == 70_000_000 and v['note'].endswith(st.CARRIED.strip())
    # on 6 Nov the carried item moves on to HOLD by the calendar
    r = st.scan(tmp_path / 'out3', db, today=D('2026-11-06'))
    assert r['scan_ok'] and r['statuses']['HOLD'] == 2


def test_yahoo_rate_limit_after_spy(tmp_path, fakes, monkeypatch):
    calls = []

    def limited(symbol, start=None):
        calls.append(symbol)
        if symbol == 'SPY':
            return fake_yahoo(symbol, start)
        raise st.YahooError('YFRateLimitError: Too Many Requests')
    db = tmp_path / 'db'
    make_db(db)
    first_pipeline(tmp_path, db, '2026-10-29')
    monkeypatch.setattr(st, 'yahoo_history', limited)
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    assert not r['scan_ok'] and not (tmp_path / 'out' / 'docs' / 'market' / 'pipeline.json').exists()
    assert calls == ['SPY', 'VUAG.L', 'EQQQ.L', 'GBPUSD=X']   # after 3 failures in a row Yahoo counts as down
    assert any('not tried: Yahoo Finance looks down' in e for e in r['errors'])
    assert 'Pipeline not updated: no prices for 5 spin-offs' in r['errors']
    assert 'Yahoo Finance: no prices for VUAG.L (YFRateLimitError: Too Many Requests)' in r['errors']
    assert 'Yahoo Finance: no prices for Q today (download failed (not tried: Yahoo Finance looks down)); status ' \
           'carried forward from yesterday' in r['errors']
    assert any(e.startswith('Yahoo Finance: no prices for NOPX today') and e.endswith('status carried forward from '
                                                                                    'yesterday') for e in r['errors'])
    # statuses are carried from yesterday on the calendar, so the buy and the sell still come through
    assert r['statuses'] == {'BUY': 1, 'HOLD': 1, 'skip': 1, 'not_new': 1, 'waiting': 2}
    assert r['alerts']['push'] == 'BUY VYLR today (check price first); SELL OLDX (12 months up); scan incomplete'
    # without yesterday's pipeline: nothing is shown as a fake "waiting" in the database
    r = st.scan(tmp_path / 'out2', None, today=TODAY)
    assert not r['scan_ok'] and not (tmp_path / 'out2' / 'docs' / 'market' / 'pipeline.json').exists()
    assert r['alerts']['push'] == "No trades found, but today's scan was incomplete: see the page."


def test_yahoo_silently_empty_after_spy(tmp_path, fakes, monkeypatch):
    # Yahoo answers SPY, then returns empty frames without an error (as yf.download does on a rate limit):
    # the benchmarks are missing, so the empty spin-off answers are not taken as "no prices yet"
    monkeypatch.setattr(st, 'yahoo_history', lambda s, start=None: fake_yahoo(s, start) if s == 'SPY' else pd.DataFrame())
    r = st.scan(tmp_path, None, today=TODAY)
    assert not r['scan_ok'] and not (tmp_path / 'docs' / 'market' / 'pipeline.json').exists()
    assert len(r['failures']) == 5 and all(not x['trusted'] for x in r['failures'])
    assert any(e.endswith('(no prices returned while other Yahoo Finance downloads failed); status unknown today')
               for e in r['errors'])
    assert r['alerts']['push'] == "No trades found, but today's scan was incomplete: see the page."


def test_fetcher_and_yahoo_history_retries(monkeypatch):
    # Fetcher: failures are remembered; three in a row stop further downloads; a success resets the count
    seq = []

    def fake(symbol, start=None):
        seq.append(symbol)
        if symbol.startswith('BAD'):
            raise st.YahooError('down')
        return fake_yahoo('SPY', start) if symbol == 'OK' else pd.DataFrame()
    monkeypatch.setattr(st, 'yahoo_history', fake)
    f = st.Fetcher()
    assert f('BAD1', None, TODAY)[1] is False and f('BAD1', None, TODAY)[1] is False and seq == ['BAD1']
    h, ok = f('OK', D('2026-10-01'), D('2026-10-20'))
    assert ok and h.index[0] == pd.Timestamp('2026-10-01') and h.index[-1] == pd.Timestamp('2026-10-20')
    assert f('NEW', None, TODAY)[1] is True                   # no prices is not a failure
    for s in ('BAD2', 'BAD3', 'BAD4', 'OK2'):
        f(s, None, TODAY)
    assert seq[-3:] == ['BAD2', 'BAD3', 'BAD4'] and 'looks down' in f.failed['OK2'] and f.failed['BAD2'] == 'down'


def test_yahoo_history_retries_only_real_failures(monkeypatch):
    # retries with backoff on rate limits / network errors; no retry when Yahoo says it has no prices
    import yfinance as yf
    from yfinance.exceptions import YFPricesMissingError, YFRateLimitError
    sleeps, tries = [], []
    monkeypatch.setattr(st.time, 'sleep', lambda x: sleeps.append(x))

    class Resp:
        status_code = 404

    class NotFound(Exception):
        response = Resp()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kw):
            tries.append(self.symbol)
            if self.symbol == 'LIMIT2' and tries.count('LIMIT2') <= 2:
                raise YFRateLimitError()
            if self.symbol == 'DOWN':
                raise ConnectionError('reset')
            if self.symbol == 'GONE':
                raise YFPricesMissingError('GONE', '')
            if self.symbol == 'NOPE':
                raise NotFound('HTTP Error 404')
            return frame(days_between('2026-10-01', '2026-10-05')).tz_localize('America/New_York')
    monkeypatch.setattr(yf, 'Ticker', FakeTicker)
    h = st.yahoo_history('LIMIT2')
    assert len(h) == 3 and h.index.tz is None and tries.count('LIMIT2') == 3
    assert [x for x in sleeps if x >= 1] == [5, 15]
    sleeps.clear()
    with pytest.raises(st.YahooError):
        st.yahoo_history('DOWN')
    assert tries.count('DOWN') == 4 and [x for x in sleeps if x >= 1] == [5, 15, 30]
    assert st.yahoo_history('GONE').empty and st.yahoo_history('NOPE').empty
    assert tries.count('GONE') == 1 and tries.count('NOPE') == 1


def test_bad_or_future_dates_in_user_records(tmp_path, fakes):
    db = tmp_path / 'db'
    make_db(db)
    write(db / 'deposits' / 'typo.json', {'date': '0026-07-01', 'amount_gbp': 100, 'note': ''})
    write(db / 'trades' / 'old.json', {'date': '1926-10-01', 'type': 'buy', 'ticker': 'ZZZ', 'shares': 1, 'amount_gbp': 1})
    write(db / 'deposits' / 'later.json', {'date': '2027-06-01', 'amount_gbp': 8000, 'note': ''})
    write(db / 'trades' / 'later.json', {'date': '2026-11-02', 'type': 'sell', 'ticker': 'OLDX', 'shares': 50,
                                         'amount_gbp': 600})
    errors = []
    data = st.load_exports(db, errors)
    assert 'typo' not in [d['id'] for d in data['deposits']] and 'old' not in [t['id'] for t in data['trades']]
    assert any(e.startswith('deposits/typo: skipped (date 0026-07-01 is before 2000') for e in errors)
    assert any(e.startswith('trades/old: skipped (date 1926-10-01') for e in errors)
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    # the typo does not drag the calendar back; the future records are ignored (and listed once)
    assert r['scan_ok'] and r['alerts']['push'] == 'BUY VYLR today; SELL OLDX (12 months up)'
    assert load(tmp_path / 'out' / 'docs' / 'market' / 'calendar.json')['days'][0] >= '2025-09-24'
    assert any(e == 'Ignored until their date (2 dated after 2026-10-30): trades/later (2026-11-02), '
                    'deposits/later (2027-06-01)' for e in r['errors'])
    # one old but valid record cannot push the start more than 15 years back
    write(db / 'deposits' / 'old.json', {'date': '2001-03-01', 'amount_gbp': 5, 'note': ''})
    starts = {}

    def spy(symbol, start=None):
        starts[symbol] = start
        return fake_yahoo(symbol, start)
    st.yahoo_history, yh = spy, st.yahoo_history
    try:
        r = st.scan(tmp_path / 'out', db, today=TODAY)
    finally:
        st.yahoo_history = yh
    assert starts['VUAG.L'] == TODAY - timedelta(days=5479) == D('2011-10-30') and r['scan_ok']


def test_future_dated_records_do_not_count_in_alerts():
    idx = pd.Series(100.0, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days_between('2026-01-02', '2026-10-29')]))
    fx = pd.Series(1.25, index=idx.index)
    base = user(deposits=[('2026-01-02', 8000.0)])
    fut = user(deposits=[('2026-01-02', 8000.0), ('2027-06-01', 8000.0)],
               trades=[('2026-11-02', 'buy', 'VYLR', 3, 900.0)])
    a, b = alerts('2026-10-30', [item()], base, {}, idx, fx), alerts('2026-10-30', [item()], fut, {}, idx, fx)
    assert 'about £1,000' in a['lines'][0] and a['lines'][0] == b['lines'][0] and b['account'] == a['account']
    assert b['push'] == 'BUY VYLR today'          # the buy logged for 2 Nov does not count as held yet


def test_spy_only_failure_keeps_buy_alerts(tmp_path, fakes, monkeypatch):
    db = tmp_path / 'db'
    make_db(db)
    monkeypatch.setattr(st, 'yahoo_history', lambda s, start=None: pd.DataFrame() if s == 'SPY' else fake_yahoo(s, start))
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    assert not r['scan_ok'] and r['last_market_day'] is None and r['statuses']['BUY'] == 1
    assert r['alerts']['push'] == 'BUY VYLR today; SELL OLDX (12 months up); scan incomplete'
    assert any('no SPY history, so trading days come from the NYSE holiday rules' in x for x in r['alerts']['lines'])
    assert not (tmp_path / 'out' / 'docs' / 'market' / 'pipeline.json').exists()
    # with SPY missing, an empty answer for NOPX (no prices yesterday either, no pipeline exported) is suspect too
    assert 'Pipeline not updated: no SPY history (Yahoo Finance looks unavailable); no prices for 1 spin-off' \
        in r['errors']
    first_pipeline(tmp_path, db, '2026-10-29')
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    assert 'Pipeline not updated: no SPY history (Yahoo Finance looks unavailable)' in r['errors']
    assert r['alerts']['push'] == 'BUY VYLR today; SELL OLDX (12 months up); scan incomplete'
    # no SPY and no sells: the quiet push still says the scan was incomplete
    (db / 'trades' / 't2.json').unlink()
    monkeypatch.setattr(st, 'sec_filings', lambda years, today, errors: [])
    r = st.scan(tmp_path / 'out', db, today=TODAY)
    assert r['alerts']['push'] == "No trades found, but today's scan was incomplete: see the page."


def test_push_tail_is_kept_when_long():
    pieces = [f'BUY T{i:02d} today' for i in range(30)]
    p = st.join_push(pieces, 'scan incomplete')
    assert len(p) <= 190 and p.endswith('more; scan incomplete') and p.startswith('BUY T00 today; BUY T01 today')
    assert st.join_push([], 'x') == 'x' and st.join_push(['a', 'b']) == 'a; b'


NOW = lambda iso: datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)   # noqa: E731


def test_alerts_after_the_close_point_to_the_next_trading_day():
    # 5 Nov 2026 is the last day of VYLR's window. Pre-open and during the session it is still a buy ...
    a = st.build_alerts(D('2026-11-05'), [item()], user(), CAL, {}, None, None, now=NOW('2026-11-05T11:45'))
    assert a['push'] == 'LAST DAY to buy VYLR' and 'today after the US open (14:30 UK)' in a['lines'][0]
    a = st.build_alerts(D('2026-11-05'), [item()], user(), CAL, {}, None, None, now=NOW('2026-11-05T16:00'))
    assert a['push'] == 'LAST DAY to buy VYLR' and 'now, while the US market is open (until 21:00 UK)' in a['lines'][0]
    # ... after the US close (16:00 New York = 21:00 UK) the window is over: no alert
    for t in ('2026-11-05T21:00', '2026-11-05T22:00', '2026-11-05T23:30', '2026-11-06T04:59'):
        a = st.build_alerts(D('2026-11-05'), [item()], user(), CAL, {}, None, None, now=NOW(t))
        assert not a['noteworthy'] and a['push'] == 'No trades today.', t
    # after the close on the buy day: a reminder for the next open, not "BUY today"
    a = st.build_alerts(D('2026-10-30'), [item()], user(), CAL, {}, None, None, now=NOW('2026-10-30T20:30'))
    assert a['push'] == 'BUY VYLR (window to Thu 5 Nov)'
    assert 'at the next US open (Mon 2 Nov, 14:30 UK)' in a['lines'][0]
    # sells: after the close on the day before the sell day, sell at the next open; after it on the sell day: overdue
    data = user(trades=[('2026-10-30', 'buy', 'VYLR', 2.5, 150.0)])
    a = st.build_alerts(D('2027-11-01'), [], data, CAL, {}, None, None, now=NOW('2027-11-01T21:30'))
    assert a['push'] == 'SELL VYLR (12 months up)' and 'at the next US open (Tue 2 Nov, 13:30 UK)' in a['lines'][0]
    a = st.build_alerts(D('2027-11-02'), [], data, CAL, {}, None, None, now=NOW('2027-11-02T21:30'))
    assert a['push'] == 'SELL VYLR (overdue since 2 Nov)' and 'next US open (Wed 3 Nov' in a['lines'][0]
    # weekend before the last window day: last chance at the next open
    a = st.build_alerts(D('2026-10-31'), [item(first='2026-09-28')], user(), CAL, {}, None, None)
    assert item(first='2026-09-28')['buy_window_end'] == '2026-11-02'
    assert a['push'] == 'LAST DAY to buy VYLR' and 'at the next US open (Mon 2 Nov' in a['lines'][0]


def test_real_clock_scan_after_the_close(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(st, 'sec_filings', lambda years, today, errors: [
        dict(f, form10_url=st.filing_url(f['cik'])) for f in FILINGS if f['ticker'] == 'VYLR'])
    late = datetime(2026, 10, 30, 22, 0, tzinfo=timezone.utc)
    r = st.scan(tmp_path, None, now=late)
    assert r['alerts']['push'] == 'BUY VYLR (window to Thu 5 Nov)'


def test_tiny_remainder_after_a_sell_closes_the_position():
    d = user(trades=[('2026-10-30', 'buy', 'VYLR', 3.4567891, 250.0), ('2027-11-02', 'sell', 'VYLR', 3.4567, 260.0)])
    assert st.positions(d['trades'])['VYLR']['shares'] == 0 and not alerts('2027-11-10', [], d)['noteworthy']
    # a real part sale (a trim) keeps the position
    d = user(trades=[('2026-10-30', 'buy', 'VYLR', 10, 250.0), ('2027-01-05', 'sell', 'VYLR', 9.9, 260.0)])
    p = st.positions(d['trades'])['VYLR']
    assert abs(p['shares'] - 0.1) < 1e-9 and abs(p['cost_gbp'] - 2.5) < 1e-9
    assert alerts('2027-11-02', [], d)['push'] == 'SELL VYLR (12 months up)'


def test_current_company_name_from_sec(monkeypatch):
    monkeypatch.setattr(ls.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ls, 'sec_session', lambda: FakeSEC())
    names = pd.DataFrame({'ticker': ['VYLR'], 'exchange': ['NYSE'], 'name': ['Vylor Technologies Corp']},
                         index=pd.Index([2128626], name='cik'))
    monkeypatch.setattr(ls, 'tickers', lambda s: names)
    (f,) = st.sec_filings(2, D('2026-10-20'), [])
    assert f['company'] == 'Vylor Technologies Corp' and f['filed_as'] == 'Vylor Inc.'
    h = frame(days_between('2026-10-01', '2026-10-29'))
    it = st.classify(f, h, D('2026-10-30'), CAL)
    assert it['company'] == 'Vylor Technologies Corp' and it['note'].endswith('Filed its Form 10 as Vylor Inc.')
    a = alerts('2026-10-30', [it], user())
    assert a['lines'][0].startswith('Buy VYLR (Vylor Technologies) today')
    names['name'] = ['VYLOR INC']                 # same name, other spelling: no "filed as"
    assert st.sec_filings(2, D('2026-10-20'), [])[0]['filed_as'] is None


def test_class_share_tickers_match(tmp_path):
    db = tmp_path / 'db'
    write(db / 'trades' / 'a.json', {'date': '2026-10-30', 'type': 'buy', 'ticker': 'abc.a', 'shares': 3, 'amount_gbp': 200})
    write(db / 'settings' / 'main.json', {'dismissed': {'xyz.b': 'Not available on Trading 212'}})
    data = st.load_exports(db, [])
    assert data['trades'][0]['ticker'] == 'ABC-A' and data['settings']['dismissed'] == {'XYZ-B': 'Not available on Trading 212'}
    assert alerts('2026-11-02', [item(ticker='ABC-A')], data)['push'] == 'No trades today.'
    assert not alerts('2026-11-02', [item(ticker='XYZ-B', cik='2')], data)['noteworthy']
    assert st.filings_from_items([dict(item(ticker='BRK.B'), form10_date='2026-06-29')])[0]['ticker'] == 'BRK-B'
    assert st.norm_ticker(' brk.b ') == 'BRK-B' == st.yahoo_symbol('BRK.B')


def test_live_spinoffs_run_uses_engine_rules(tmp_path, fakes, monkeypatch):
    # day 0 has a huge volume, the 20 days before the buy day trade only $50K a day: the engine says skip
    # (the old command counted day 0 instead of day 20 and said buy)
    def thin(symbol, start=None):
        if symbol == 'VYLR':
            h = frame(days_between('2026-10-01', TODAY), close=5.0, volume=lambda i: 10_000_000 if i == 0 else 10_000)
            return h if start is None else h[h.index >= pd.Timestamp(start)]
        return fake_yahoo(symbol, start)
    monkeypatch.setattr(st, 'yahoo_history', thin)
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)      # 08:00 New York: today's bar is not final
    out = ls.run(2, out=tmp_path / 'spinoffs.csv', now=now)
    rows = {r['ticker'] if isinstance(r['ticker'], str) else r['company']: r for r in out.to_dict('records')}
    assert rows['VYLR']['status'] == 'skip' and rows['VYLR']['avg_dollar_volume'] == 50_000
    assert rows['VYLR']['buy_date'] == '2026-10-30' and rows['Q']['status'] == 'HOLD' and rows['OLD']['status'] == 'not_new'
    assert (tmp_path / 'spinoffs.csv').exists()
    items = {it['ticker']: it for it in st.scan(tmp_path / 'scan', None, today=TODAY)['alerts']['items']}
    assert 'VYLR' not in items
