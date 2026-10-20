// Tests the pure logic block of spinoff_tracker.html (the <script id="spin-logic"> block),
// evaluated in node:vm exactly as written in the page. Run: node test_logic.mjs
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, 'spinoff_tracker.html'), 'utf8');
const m = html.match(/<script id="spin-logic">([\s\S]*?)<\/script>/);
if (!m) throw new Error('logic block <script id="spin-logic"> not found');
const ctx = vm.createContext({});
vm.runInContext(m[1], ctx, { filename: 'spin-logic.js' });
const L = ctx.SpinLogic;
assert.ok(L && typeof L.computeAll === 'function', 'SpinLogic exposed');

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log('  ok   ' + name); }
  catch (e) { fail++; console.log('  FAIL ' + name + '\n       ' + (e && e.message ? e.message.split('\n').join('\n       ') : e)); }
}
const near = (a, b, tol = 1e-6, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} expected ${b}, got ${a}`);
const plain = (x) => JSON.parse(JSON.stringify(x)); // strip vm realm prototypes for deepEqual
function deepFreeze(o) { if (o && typeof o === 'object' && !Object.isFrozen(o)) { Object.freeze(o); Object.values(o).forEach(deepFreeze); } return o; }
// weekday-only calendar (no holidays) for hand counting
function weekdays(start, n) { const out = []; let d = start; while (out.length < n) { if (!L.isWeekend(d)) out.push(d); d = L.addDays(d, 1); } return out; }
const ser = (dates, close, extra = {}) => ({ dates, close, ...extra });

console.log('dates and times');
test('isYMD rejects impossible dates', () => {
  assert.equal(L.isYMD('2026-02-29'), false); assert.equal(L.isYMD('2028-02-29'), true);
  assert.equal(L.isYMD('2026-13-01'), false); assert.equal(L.isYMD('2026-1-01'), false); assert.equal(L.isYMD(20261020), false);
});
test('addDays / daysBetween use calendar days', () => {
  assert.equal(L.addDays('2026-10-31', 1), '2026-11-01');
  assert.equal(L.addDays('2026-03-29', 1), '2026-03-30'); // UK clocks change; still one day
  assert.equal(L.daysBetween('2025-10-17', '2026-10-20'), 368);
  assert.equal(L.daysBetween('2026-10-20', '2026-10-20'), 0);
});
test('todayNY / todayLondon follow the zone, not the machine', () => {
  assert.equal(L.todayNY(Date.parse('2026-10-21T03:30:00Z')), '2026-10-20'); // 23:30 EDT
  assert.equal(L.todayNY(Date.parse('2026-12-01T04:59:00Z')), '2026-11-30'); // 23:59 EST
  assert.equal(L.todayNY(Date.parse('2026-12-01T05:00:00Z')), '2026-12-01');
  assert.equal(L.todayLondon(Date.parse('2026-10-20T23:30:00Z')), '2026-10-21'); // 00:30 BST
  assert.equal(L.todayLondon(Date.parse('2026-12-20T23:30:00Z')), '2026-12-20'); // GMT
});
test('US open in UK time handles the mismatched clock changes', () => {
  assert.equal(L.usOpenUK('2026-10-20'), '14:30');
  assert.equal(L.usOpenUK('2026-10-27'), '13:30'); // UK back to GMT 25 Oct, US on EDT until 1 Nov
  assert.equal(L.usOpenUK('2026-11-03'), '14:30');
  assert.equal(L.usOpenUK('2026-03-10'), '13:30'); // US on EDT from 8 Mar, UK on GMT until 29 Mar
  assert.equal(L.usOpenUK('2026-03-31'), '14:30');
});
test('fmtDate', () => {
  assert.equal(L.fmtDate('2026-10-20'), 'Tue 20 Oct');
  assert.equal(L.fmtDate('2026-10-20', { year: true, weekday: false }), '20 Oct 2026');
  assert.equal(L.weekdayOf('2026-10-24'), 'Saturday');
});

console.log('calendar');
test('252 trading days after a buy (weekday-only calendar, hand count)', () => {
  const days = weekdays('2026-01-05', 300);
  // index 2 = Wed 7 Jan; +252 = index 254 = 50 weeks (Mon 21 Dec) + 4 weekdays = Fri 25 Dec
  assert.equal(L.tradingDayAfter(days, '2026-01-07', 252), '2026-12-25');
  // Saturday buy -> day 0 is Monday 12 Jan (index 5); index 257 = Wed 30 Dec
  assert.equal(L.tradingDayAfter(days, '2026-01-10', 252), '2026-12-30');
  assert.equal(L.tradingDayAfter(days, '2025-12-01', 252), null); // before calendar start
  assert.equal(L.tradingDayAfter(days, '2026-12-01', 252), null); // beyond the end
});
test('real NYSE calendar: Fri 17 Oct 2025 + 252 = Tue 20 Oct 2026 (262 weekdays - 10 holidays)', () => {
  const fx = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
  const days = fx.collections.market.calendar.days;
  for (const hol of ['2025-11-27', '2025-12-25', '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07']) {
    assert.equal(L.isTradingDay(days, hol), false, hol + ' should be a holiday');
  }
  assert.equal(L.tradingDayAfter(days, '2025-10-17', 252), '2026-10-20');
  assert.equal(L.tradingDaysBetween(days, '2025-10-17', '2026-10-20'), 252);
});
test('tradingDaysBetween counts a < d <= b', () => {
  const days = weekdays('2026-01-05', 30);
  assert.equal(L.tradingDaysBetween(days, '2026-01-05', '2026-01-09'), 4);
  assert.equal(L.tradingDaysBetween(days, '2026-01-09', '2026-01-12'), 1); // Fri -> Mon
  assert.equal(L.tradingDaysBetween(days, '2026-01-10', '2026-01-12'), 1); // from a Saturday
  assert.equal(L.tradingDaysBetween(days, '2026-01-12', '2026-01-12'), 0);
});
test('sellDateFor falls back to the pipeline date beyond the calendar', () => {
  const days = weekdays('2026-01-05', 100);
  assert.deepEqual(plain(L.sellDateFor(days, '2026-02-02', '2027-02-01')), { date: '2027-02-01', source: 'pipeline' });
  assert.equal(L.sellDateFor(days, '2026-02-02', null).source, 'approx');
});

console.log('prices and valuation');
test('priceOn forward-fills and back-fills', () => {
  const s = ser(['2026-01-05', '2026-01-06', '2026-01-08'], [10, 11, 12]);
  assert.equal(L.priceOn(s, '2026-01-07'), 11);
  assert.equal(L.priceOn(s, '2026-01-08'), 12);
  assert.equal(L.priceOn(s, '2026-01-01'), 10);
  assert.equal(L.priceOn(s, '2026-02-01'), 12);
  assert.equal(L.priceOn(null, '2026-02-01'), null);
});
test('index pot units: deposit, spin-off buy, spin-off sell', () => {
  const fund = ser(['2026-01-02', '2026-02-02', '2026-03-02'], [100, 125, 150]);
  const deposits = L.cleanDeposits([{ id: 'a', date: '2026-01-02', amount_gbp: 1000 }]);
  const trades = L.cleanTrades([
    { id: 'b', date: '2026-02-02', type: 'buy', ticker: 'xyz', shares: 5, amount_gbp: 250 },
    { id: 'c', date: '2026-03-02', type: 'sell', ticker: 'XYZ', shares: 5, amount_gbp: 300 }
  ]);
  const pot = L.indexPot(L.buildEvents(deposits, trades), fund);
  near(pot.units, 10, 1e-12); // 10 - 250/125 + 300/150
  near(L.stepAt(pot.timeline, '2026-02-15'), 8, 1e-12);
  near(L.stepAt(pot.timeline, '2025-12-31'), 0, 1e-12);
  assert.equal(pot.warnings.length, 0);
});
test('same date order: deposits, then sells, then buys (no false overspend warning)', () => {
  const fund = ser(['2026-01-02', '2026-02-02'], [100, 125]);
  const deposits = L.cleanDeposits([{ id: 'a', date: '2026-01-02', amount_gbp: 1000 }, { id: 'b', date: '2026-02-02', amount_gbp: 500 }]);
  const trades = L.cleanTrades([{ id: 'c', date: '2026-02-02', type: 'buy', ticker: 'XYZ', shares: 1, amount_gbp: 1500 }]);
  const ev = L.buildEvents(deposits, trades);
  assert.deepEqual(plain(ev.map((e) => e.kind)), ['deposit', 'deposit', 'buy']);
  const pot = L.indexPot(ev, fund);
  near(pot.units, 2, 1e-12); // 10 + 4 - 12
  assert.equal(pot.warnings.length, 0);
});
test('overspending records produce a warning, not a crash', () => {
  const fund = ser(['2026-02-02'], [100]);
  const pot = L.indexPot(L.buildEvents([], L.cleanTrades([{ id: 'x', date: '2026-02-02', type: 'buy', ticker: 'AB', shares: 1, amount_gbp: 2000 }])), fund);
  near(pot.units, -20, 1e-12);
  assert.equal(pot.warnings.length, 1);
  assert.match(pot.warnings[0].text, /Your records spend more than the account held on Mon 2 Feb 2026/);
});
test('spin-off value converts USD to GBP', () => {
  near(L.spinValueGBP(10, 25, 1.25), 200, 1e-12);
  assert.equal(L.spinValueGBP(10, 25, null), null);
});

// a small hand-checkable account
function smallInput(over = {}) {
  const days = weekdays('2025-12-29', 400);
  return {
    today: '2026-03-03', todayUK: '2026-03-03', nowMs: Date.parse('2026-03-03T15:00:00Z'),
    market: { calendar: { days, last_actual: '2026-03-02' }, pipeline: { items: [] }, status: null },
    prices: {
      VUAG: ser(['2026-01-02', '2026-01-05', '2026-03-02'], [100, 100, 110]),
      EQQQ: ser(['2026-01-02', '2026-03-02'], [400, 380]),
      GBPUSD: ser(['2026-01-02', '2026-03-02'], [1.25, 1.25]),
      XYZ: ser(['2026-01-05', '2026-03-02'], [20, 30])
    },
    checks: {},
    deposits: [{ id: 'd1', date: '2026-01-02', amount_gbp: 8000, note: '', created_at: '' }],
    trades: [{ id: 't1', date: '2026-01-05', type: 'buy', ticker: 'XYZ', shares: 62.5, amount_gbp: 1000, note: '', created_at: '' }],
    settings: { index_fund: 'VUAG', slice: 8, monthly_deposit: 1666.66, dismissed: {} },
    ...over
  };
}
test('account value with FX: 70 VUAG units x 110 + 62.5 XYZ x $30 / 1.25 = 9,200', () => {
  const D = L.computeAll(deepFreeze(smallInput()));
  near(D.account.pot, 7700, 1e-9);
  near(D.account.value, 9200, 1e-9);
  near(D.head.you.moneyIn, 8000, 1e-9);
  near(D.head.you.gain, 1200, 1e-9);
  near(D.head.you.gainPct, 0.15, 1e-12);
  near(D.head.sp.value, 8800, 1e-9);  // 80 units x 110
  near(D.head.nq.value, 7600, 1e-9);  // 20 units x 380
  near(D.head.vsSP, 400, 1e-9);
  near(D.head.vsNQ, 1600, 1e-9);
  assert.equal(D.head.annualised, false); // first deposit < 365 days old
  assert.equal(D.head.you.xirr, null);
  assert.equal(D.series.dates[0], '2026-01-02');
  assert.equal(D.series.dates.at(-1), '2026-03-02');
  near(D.series.account[0], 8000, 1e-9);
  near(D.series.account[1], 8000, 1e-9); // 70 x 100 + 62.5 x 20 / 1.25
});
test('slice: account / 8 per spin-off in the window', () => {
  const inp = smallInput();
  inp.market = { ...inp.market, pipeline: { items: [
    { id: '1', company: 'Alpha', ticker: 'ALP', status: 'BUY', buy_date: '2026-03-02', buy_window_end: '2026-03-06' },
    { id: '2', company: 'Beta', ticker: 'BET', status: 'BUY', buy_date: '2026-03-03', buy_window_end: '2026-03-09' }
  ] } };
  const D = L.computeAll(deepFreeze(inp));
  near(D.sliceAmount, 1150, 1e-9);
  const buys = D.actions.filter((a) => a.kind === 'buy');
  assert.equal(buys.length, 2);
  buys.forEach((b) => near(b.amount, 1150, 1e-9));
  assert.equal(D.plan.mode, 'fund');
  near(D.plan.fundSell, 2300, 1e-9);
});
test('funding plan: trim others in proportion when the fund falls short', () => {
  const p = L.fundingPlan(1500, [{ ticker: 'A', value: 3000 }, { ticker: 'B', value: 1000 }], 2000, 'VUAG');
  assert.equal(p.mode, 'trim');
  near(p.fundSell, 1500, 1e-12); near(p.trim, 500, 1e-12);
  assert.deepEqual(plain(p.trims.map((t) => [t.ticker, Math.round(t.amount * 100) / 100])), [['A', 375], ['B', 125]]);
  const q = L.fundingPlan(2500, [{ ticker: 'A', value: 3000 }], 2000, 'EQQQ');
  assert.equal(q.mode, 'fund'); near(q.fundSell, 2000, 1e-12); assert.equal(q.fund, 'EQQQ');
  const r = L.fundingPlan(500, [{ ticker: 'A', value: 3000, dueSell: true }, { ticker: 'B', value: 1000 }], 2000, 'VUAG');
  assert.equal(r.mode, 'fund'); near(r.fromSales, 3000, 1e-12); near(r.fundSell, 0, 1e-12); near(r.fundBuy, 1000, 1e-12); // today's sale funds it, rest to the fund
  const s = L.fundingPlan(-50, [], 100, 'VUAG');
  assert.equal(s.mode, 'trim'); near(s.shortfall, 100, 1e-12);
});

console.log('XIRR');
test('one year, +10% over 366 days', () => {
  near(L.xirr([{ date: '2020-01-01', amount: -1000 }, { date: '2021-01-01', amount: 1100 }]), Math.pow(1.1, 365 / 366) - 1, 1e-9);
});
test('Microsoft XIRR reference example = 37.3362535%', () => {
  const r = L.xirr([
    { date: '2008-01-01', amount: -10000 }, { date: '2008-03-01', amount: 2750 }, { date: '2008-10-30', amount: 4250 },
    { date: '2009-02-15', amount: 3250 }, { date: '2009-04-01', amount: 2750 }
  ]);
  near(r, 0.373362535, 1e-6);
});
test('large loss converges (bisection fallback region)', () => {
  near(L.xirr([{ date: '2020-01-01', amount: -1000 }, { date: '2020-12-31', amount: 10 }]), -0.99, 1e-6);
});
test('no sign change -> null', () => {
  assert.equal(L.xirr([{ date: '2020-01-01', amount: -1000 }, { date: '2021-01-01', amount: -5 }]), null);
});

console.log('positions');
test('trim, close, reopen, same-day round trip, orphan sell', () => {
  const trades = L.cleanTrades([
    { id: '1', date: '2026-01-05', type: 'buy', ticker: 'A', shares: 10, amount_gbp: 100 },
    { id: '2', date: '2026-02-02', type: 'sell', ticker: 'A', shares: 4, amount_gbp: 50 },
    { id: '3', date: '2026-03-02', type: 'sell', ticker: 'A', shares: 6, amount_gbp: 90 },
    { id: '4', date: '2026-04-01', type: 'buy', ticker: 'A', shares: 5, amount_gbp: 60 },
    { id: '6', date: '2026-05-01', type: 'sell', ticker: 'B', shares: 3, amount_gbp: 33, created_at: '2026-05-01T10:00:00Z' },
    { id: '5', date: '2026-05-01', type: 'buy', ticker: 'B', shares: 3, amount_gbp: 30, created_at: '2026-05-01T11:00:00Z' },
    { id: '7', date: '2026-05-04', type: 'sell', ticker: 'C', shares: 1, amount_gbp: 5 }
  ]);
  const { list, warnings } = L.positions(trades);
  const A = list.filter((p) => p.ticker === 'A');
  assert.equal(A.length, 2);
  assert.equal(A[0].closed, true); assert.equal(A[0].closeDate, '2026-03-02');
  near(A[0].cost, 100, 1e-12); near(A[0].proceeds, 140, 1e-12);
  assert.equal(A[1].closed, false); near(A[1].shares, 5, 1e-12); assert.equal(A[1].openDate, '2026-04-01');
  const B = list.filter((p) => p.ticker === 'B');
  assert.equal(B.length, 1); assert.equal(B[0].closed, true); near(B[0].proceeds - B[0].cost, 3, 1e-12);
  assert.equal(warnings.length, 1); assert.match(warnings[0].text, /Sale of C/);
});
test('open position: return counts trims; sell date and countdown from the calendar', () => {
  const inp = smallInput({ today: '2026-03-03' });
  const D = L.computeAll(deepFreeze(inp));
  assert.equal(D.open.length, 1);
  const p = D.open[0];
  near(p.value, 1500, 1e-9); near(p.ret, 0.5, 1e-12); near(p.spRet, 0.10, 1e-12);
  // weekday calendar from Mon 29 Dec 2025: buy Mon 5 Jan = index 5; +252 = index 257 = 51 weeks (Mon 21 Dec 2026) + 2 = Wed 23 Dec 2026
  assert.equal(p.sellDate, '2026-12-23');
  assert.equal(p.sellSource, 'calendar');
  // elapsed: 5 Jan -> 3 Mar = 41 weekdays
  assert.equal(p.elapsed, 41);
  assert.equal(p.daysLeft, 252 - 41);
  assert.equal(p.due, false);
});
test('closed position: return vs VUAG over the same dates', () => {
  const inp = smallInput();
  inp.trades = [...inp.trades, { id: 't2', date: '2026-03-02', type: 'sell', ticker: 'XYZ', shares: 62.5, amount_gbp: 1480, note: '', created_at: '' }];
  const D = L.computeAll(deepFreeze(inp));
  assert.equal(D.open.length, 0); assert.equal(D.closed.length, 1);
  const c = D.closed[0];
  near(c.ret, 0.48, 1e-12); near(c.spRet, 0.10, 1e-12); near(c.excess, 0.38, 1e-12);
  assert.equal(c.daysHeld, 56);
  assert.equal(D.stats.n, 1); assert.equal(D.stats.wins, 1); near(D.stats.avgExcess, 0.38, 1e-12);
  near(D.account.pot, (80 - 10 + 1480 / 110) * 110, 1e-9);
});

console.log('actions and filtering');
function actionInput() {
  const inp = smallInput({ today: '2026-12-23', todayUK: '2026-12-23', nowMs: Date.parse('2026-12-23T15:00:00Z') });
  inp.prices = { ...inp.prices, OLD: ser(['2025-12-29', '2026-12-22'], [10, 12]) };
  inp.trades = [...inp.trades, { id: 't0', date: '2025-12-30', type: 'buy', ticker: 'OLD', shares: 100, amount_gbp: 800, note: '', created_at: '' }];
  inp.deposits = [{ id: 'd0', date: '2025-12-29', amount_gbp: 1000, note: '', created_at: '' }, ...inp.deposits];
  inp.market = { ...inp.market, pipeline: { items: [
    { id: '10', company: 'Late Window', ticker: 'LATE', status: 'BUY', buy_date: '2026-12-21', buy_window_end: '2026-12-28' },
    { id: '11', company: 'Early Window', ticker: 'EARL', status: 'BUY', buy_date: '2026-12-17', buy_window_end: '2026-12-23' },
    { id: '12', company: 'Pending Check', ticker: 'PEND', status: 'watch', buy_date: '2026-12-23', buy_window_end: '2026-12-30' },
    { id: '13', company: 'Hidden Co', ticker: 'HIDE', status: 'BUY', buy_date: '2026-12-22', buy_window_end: '2026-12-29' },
    { id: '14', company: 'Tracking Stock', ticker: 'FAKE', status: 'BUY', buy_date: '2026-12-22', buy_window_end: '2026-12-29' },
    { id: '15', company: 'Already Mine', ticker: 'XYZ', status: 'HOLD', first_trade: '2025-12-01', buy_date: '2026-01-05', buy_window_end: '2026-01-09' },
    { id: '16', company: 'Stale Signal', ticker: 'STAL', status: 'BUY', buy_date: '2026-12-10', buy_window_end: '2026-12-16' },
    { id: '17', company: 'Future One', ticker: 'FUT', status: 'watch', buy_date: '2027-01-05', buy_window_end: '2027-01-11' },
    { id: '18', company: 'Not Listed', ticker: null, status: 'waiting', form10_date: '2026-12-01' },
    { id: '19', company: 'Unclear Co', ticker: 'UNC', status: 'watch', buy_date: '2027-01-06', buy_window_end: '2027-01-12' }
  ] } };
  inp.checks = { '14': { genuine: false, note: 'tracking stock' }, '11': { genuine: true, parent: 'Parent Inc.', business: 'Widgets' }, '19': { genuine: null } };
  inp.settings = { ...inp.settings, dismissed: { HIDE: 'Not available on Trading 212' } };
  return inp;
}
test('order: overdue sell, sell due today, then buys by window end', () => {
  const D = L.computeAll(deepFreeze(actionInput()));
  // OLD bought Tue 30 Dec 2025 (index 1) -> +252 = index 253; index 250 = Mon 14 Dec 2026 (50 weeks), so 253 = Thu 17 Dec -> overdue. XYZ -> Wed 23 Dec = today.
  assert.deepEqual(plain(D.actions.map((a) => a.kind + ':' + a.ticker + (a.overdue ? ':overdue' : ''))),
    ['sell:OLD:overdue', 'sell:XYZ', 'buy:EARL', 'buy:LATE', 'buy:PEND']);
  const pend = D.actions.find((a) => a.ticker === 'PEND');
  assert.equal(pend.pending, true);
  assert.equal(D.actions.find((a) => a.ticker === 'EARL').lastDay, true);
  assert.equal(D.actions[0].sellDate, '2026-12-17');
  assert.equal(D.actions[0].openUK, '14:30');
});
test('dismissed, genuine=false, held and stale signals are filtered out of buys', () => {
  const D = L.computeAll(deepFreeze(actionInput()));
  const buyT = D.actions.filter((a) => a.kind === 'buy').map((a) => a.ticker);
  for (const t of ['HIDE', 'FAKE', 'XYZ', 'STAL']) assert.ok(!buyT.includes(t), t + ' must not be a buy');
  const missed = Object.fromEntries(D.groups.missed.map((it) => [it.ticker, it.reason]));
  assert.match(missed.HIDE, /You hid it: Not available on Trading 212/);
  assert.match(missed.FAKE, /Not a genuine spin-off: tracking stock/);
  assert.equal(missed.STAL, 'Buy window passed');
  assert.ok(!('XYZ' in missed), 'held spin-off is not "missed"');
  assert.deepEqual(plain(D.groups.coming.map((it) => it.ticker)), ['FUT', 'UNC']); // genuine null still shown
  assert.deepEqual(plain(D.groups.waiting.map((it) => it.company)), ['Not Listed']);
  assert.deepEqual(plain(D.groups.open.map((it) => it.ticker)), ['EARL', 'LATE', 'PEND']);
});
test('both sells today fund the buys first (plan uses today\'s sale proceeds)', () => {
  const D = L.computeAll(deepFreeze(actionInput()));
  assert.equal(D.plan.mode, 'fund');
  assert.ok(D.plan.fromSales > 0);
  near(D.plan.need, D.sliceAmount * 3, 1e-9);
  near(D.plan.fundSell, Math.max(0, D.plan.need - D.plan.fromSales), 1e-9);
});
test('coming up lists future buys and sells by date with trading-day counts', () => {
  const D = L.computeAll(deepFreeze(actionInput()));
  assert.deepEqual(plain(D.coming.map((c) => c.kind + ':' + c.ticker + ':' + c.date)), ['buy:FUT:2027-01-05', 'buy:UNC:2027-01-06']);
  assert.equal(D.coming[0].tradingDays, 9); // weekday calendar, no holidays: 24 Dec .. 5 Jan
});
test('nothing to do -> empty actions; next item is the first coming entry', () => {
  const inp = smallInput();
  const D = L.computeAll(deepFreeze(inp));
  assert.equal(D.actions.length, 0);
  assert.equal(D.coming[0].kind, 'sell');
});

console.log('health, validation, csv');
test('stale and failed scans are flagged', () => {
  const inp = smallInput();
  inp.market = { ...inp.market, status: { generated_at: '2026-02-26T05:47:00Z', today_ny: '2026-02-26', last_market_day: '2026-02-25', scan_ok: false, errors: ['EDGAR timed out'], trigger_id: 'trig_x', next_check: 'Weekdays 06:45 UK time' } };
  const D = L.computeAll(deepFreeze(inp));
  assert.equal(D.health.stale, true); assert.equal(D.health.scanOk, false);
  assert.deepEqual(plain(D.health.errors), ['EDGAR timed out']); assert.equal(D.health.triggerId, 'trig_x');
  const inp2 = smallInput(); inp2.market = { ...inp2.market, status: { ...inp.market.status, generated_at: '2026-03-03T05:47:00Z', scan_ok: true, errors: [] } };
  assert.equal(L.computeAll(deepFreeze(inp2)).health.stale, false);
});
test('validateTrade', () => {
  const ok = L.validateTrade({ date: '2026-10-20', type: 'buy', ticker: ' brk.b ', shares: '41.2371', amount_gbp: '£1,234.50' }, '2026-10-20');
  assert.equal(ok.ok, true); assert.equal(ok.value.ticker, 'BRK.B'); near(ok.value.amount_gbp, 1234.5, 1e-12); near(ok.value.shares, 41.2371, 1e-12);
  const bad = L.validateTrade({ date: '2026-10-21', type: 'buy', ticker: 'AB$', shares: '0', amount_gbp: '-5' }, '2026-10-20');
  assert.equal(bad.ok, false);
  assert.deepEqual(Object.keys(bad.errors).sort(), ['amount_gbp', 'date', 'shares', 'ticker']);
  assert.match(bad.errors.date, /future/);
});
test('validateDeposit signs withdrawals', () => {
  const w = L.validateDeposit({ date: '2026-10-01', amount: '500', kind: 'out' }, '2026-10-20');
  assert.equal(w.value.amount_gbp, -500);
  assert.equal(L.validateDeposit({ date: '2026-10-01', amount: '0' }, '2026-10-20').ok, false);
});
test('records CSV escapes quotes and commas', () => {
  const csv = L.recordsCSV([{ id: 'd', date: '2026-10-01', amount_gbp: 1666.66, note: 'Monthly, "Oct"', created_at: 'x' }],
    [{ id: 't', date: '2026-10-02', type: 'buy', ticker: 'CLDR', shares: 10, amount_gbp: 100, note: '=SUM(A1)', created_at: 'y' }]);
  const lines = csv.trim().split('\r\n');
  assert.equal(lines[0], 'record,date,type,ticker,shares,amount_gbp,note,created_at');
  assert.equal(lines[1], 'deposit,2026-10-01,deposit,,,1666.66,"Monthly, ""Oct""",x');
  assert.equal(lines[2], "trade,2026-10-02,buy,CLDR,10,100.00,'=SUM(A1),y");
});
test('money and percent formatting', () => {
  assert.equal(L.fmtMoney(1666.66), '£1,666.66');
  assert.equal(L.fmtMoney(-210.4, { dp: 0 }), '−£210');
  assert.equal(L.fmtMoney(512.2, { sign: true, dp: 0 }), '+£512');
  assert.equal(L.fmtMoney(-0.001), '£0.00');
  assert.equal(L.fmtPct(0.1474), '+14.7%');
  assert.equal(L.fmtPct(-0.05), '−5.0%');
  assert.equal(L.fmtCompact(31299), '£31.3k');
  assert.equal(L.fmtShares(41.237100), '41.2371');
  assert.equal(L.parseAmount('1,666.66'), 1666.66);
});

console.log('fixture end to end');
test('fixture: actions, positions and an independent valuation agree', () => {
  const fx = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
  const c = fx.collections;
  const docs = (m) => Object.entries(m).map(([id, v]) => ({ ...v, id }));
  const input = deepFreeze({
    today: fx.today, todayUK: fx.today, nowMs: Date.parse(fx.now_utc),
    market: c.market, prices: c.prices, checks: c.checks, trades: docs(c.trades), deposits: docs(c.deposits), settings: c.settings.main
  });
  const D = L.computeAll(input);
  assert.deepEqual(plain(D.actions.map((a) => a.kind + ':' + a.ticker)), ['sell:NFIN', 'buy:OSTW', 'buy:CLDR', 'buy:PCHS']);
  assert.equal(D.actions[0].overdue, false); assert.equal(D.actions[0].sellDate, fx.today);
  assert.equal(D.actions.find((a) => a.ticker === 'OSTW').pending, true);
  assert.deepEqual(plain(D.open.map((p) => p.ticker)), ['NFIN', 'CRVD', 'ELMF']);
  assert.deepEqual(plain(D.closed.map((p) => p.ticker)), ['HRMT']);
  const hr = Object.values(c.trades).filter((t) => t.ticker === 'HRMT');
  near(D.closed[0].ret, hr[1].amount_gbp / hr[0].amount_gbp - 1, 1e-12);
  assert.ok(D.groups.missed.some((it) => it.ticker === 'KRNX' && /You hid it/.test(it.reason)));
  assert.ok(D.groups.missed.some((it) => it.ticker === 'SMTK' && /Not a genuine/.test(it.reason)));
  assert.ok(D.groups.missed.some((it) => it.ticker === 'BLSW'));
  assert.equal(D.groups.waiting.length, 2);
  assert.deepEqual(plain(D.groups.coming.map((it) => it.ticker)), ['MRDP', 'LKSB']);
  assert.equal(D.head.annualised, true);
  assert.ok(D.head.you.xirr > -1 && D.head.you.xirr < 2);
  // independent valuation with linear scans
  const p = (s, d) => { let v = null; for (let i = 0; i < s.dates.length; i++) { if (s.dates[i] <= d) v = s.close[i]; } return v ?? s.close[0]; };
  const asOf = D.asOf; const fund = c.prices.VUAG;
  let units = 0; const held = {};
  const ev = [...docs(c.deposits).map((d) => ({ d: d.date, o: 0, a: d.amount_gbp })), ...docs(c.trades).map((t) => ({ d: t.date, o: t.type === 'sell' ? 1 : 2, t }))].sort((x, y) => (x.d < y.d ? -1 : x.d > y.d ? 1 : x.o - y.o));
  for (const e of ev) {
    if (!e.t) units += e.a / p(fund, e.d);
    else if (e.t.type === 'buy') { units -= e.t.amount_gbp / p(fund, e.d); held[e.t.ticker] = (held[e.t.ticker] || 0) + e.t.shares; }
    else { units += e.t.amount_gbp / p(fund, e.d); held[e.t.ticker] -= e.t.shares; }
  }
  let total = units * p(fund, asOf);
  for (const [t, sh] of Object.entries(held)) if (sh > 1e-6) total += sh * p(c.prices[t], asOf) / p(c.prices.GBPUSD, asOf);
  near(D.account.value, total, 1e-6, 'account value');
  near(D.sliceAmount, total / 8, 1e-6, 'slice');
  const moneyIn = Object.values(c.deposits).reduce((s, d) => s + d.amount_gbp, 0);
  near(D.head.you.moneyIn, moneyIn, 1e-6);
  console.log(`       fixture: value £${total.toFixed(2)}, money in £${moneyIn.toFixed(2)}, slice £${(total / 8).toFixed(2)}, XIRR ${(D.head.you.xirr * 100).toFixed(2)}%, vs S&P £${D.head.vsSP.toFixed(2)}`);
});
test('EQQQ as the index fund switches the pot, benchmarks unchanged', () => {
  const fx = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
  const c = fx.collections;
  const docs = (m) => Object.entries(m).map(([id, v]) => ({ ...v, id }));
  const base = { today: fx.today, todayUK: fx.today, nowMs: Date.parse(fx.now_utc), market: c.market, prices: c.prices, checks: c.checks, trades: docs(c.trades), deposits: docs(c.deposits) };
  const a = L.computeAll({ ...base, settings: { ...c.settings.main, index_fund: 'VUAG' } });
  const b = L.computeAll({ ...base, settings: { ...c.settings.main, index_fund: 'EQQQ' } });
  assert.equal(b.fundCode, 'EQQQ');
  assert.notEqual(Math.round(a.account.pot), Math.round(b.account.pot));
  near(a.head.sp.value, b.head.sp.value, 1e-9); near(a.head.nq.value, b.head.nq.value, 1e-9);
  assert.equal(b.plan.fund, 'EQQQ');
});
test('empty state: no records, no market data', () => {
  const D = L.computeAll({ today: '2026-10-20', nowMs: Date.now() });
  assert.equal(D.hasRecords, false); assert.equal(D.series, null); assert.equal(D.head, null);
  assert.equal(D.actions.length, 0); assert.equal(D.health.hasData, false);
});

console.log('risk level and exposure');
test('risk level: 8 = Pure, 27 = Lower risk, other 4..40 custom, anything else falls back to 8', () => {
  assert.equal(L.riskLevel(8).name, 'Pure'); assert.equal(L.riskLevel(27).name, 'Lower risk');
  assert.match(L.riskLevel(8).text, /about 22% a year, worst fall about 60%/);
  assert.match(L.riskLevel(27).text, /about 30% in spin-offs and 70% stays in the index fund/);
  assert.equal(L.riskLevel(12).key, 'custom');
  const n = (v) => L.normSettings({ slice: v }).slice;
  assert.equal(n(4), 4); assert.equal(n(40), 40); assert.equal(n(27), 27);
  assert.equal(n(3), 8); assert.equal(n(41), 8); assert.equal(n(8.5), 8); assert.equal(n('x'), 8); assert.equal(n(undefined), 8);
  assert.equal(L.validSlice(4), true); assert.equal(L.validSlice(2), false); assert.equal(L.validSlice(NaN), false);
});
test('exposure: 62.5 XYZ x $30 / 1.25 = 1,500 of 9,200 in spin-offs (16.3%)', () => {
  const D = L.computeAll(deepFreeze(smallInput()));
  near(D.account.spinValue, 1500, 1e-9);
  near(D.account.spinShare, 1500 / 9200, 1e-12);
  assert.equal(D.risk.name, 'Pure');
  const E = L.computeAll(deepFreeze(smallInput({ trades: [] })));
  near(E.account.spinShare, 0, 1e-12);
});
test('slice 27 (lower risk): each buy gets account / 27', () => {
  const inp = smallInput({ settings: { index_fund: 'VUAG', slice: 27, monthly_deposit: 1666.66, dismissed: {} } });
  inp.market = { ...inp.market, pipeline: { items: [{ id: '1', company: 'Alpha', ticker: 'ALP', status: 'BUY', buy_date: '2026-03-02', buy_window_end: '2026-03-06' }] } };
  const D = L.computeAll(deepFreeze(inp));
  near(D.sliceAmount, 9200 / 27, 1e-9);
  assert.equal(D.risk.name, 'Lower risk');
  near(D.plan.fundSell, 9200 / 27, 1e-9);
});

// ---- the REAL engine output (copy of the scan's docs folder), when present ----
const REAL = '/tmp/claude-0/-home-user/cfe4416a-1a27-57e3-996d-df500b1ef602/scratchpad/trk_real_docs';
if (existsSync(REAL)) {
  console.log('real engine output (' + REAL + ')');
  const load = () => { const o = {}; for (const c of readdirSync(REAL)) { o[c] = {}; for (const f of readdirSync(join(REAL, c))) o[c][f.slice(0, -5)] = JSON.parse(readFileSync(join(REAL, c, f), 'utf8')); } return o; };
  const recs = JSON.parse(readFileSync(join(here, 'real_records.json'), 'utf8'));
  const docsOf = (m) => Object.entries(m).map(([id, v]) => ({ ...v, id }));
  const run = (real, withRecords, nowIso = '2026-10-20T17:30:00Z') => L.computeAll(deepFreeze({
    today: L.todayNY(Date.parse(nowIso)), todayUK: L.todayLondon(Date.parse(nowIso)), nowMs: Date.parse(nowIso),
    market: real.market, prices: real.prices, checks: {},
    trades: withRecords ? docsOf(recs.trades) : [], deposits: withRecords ? docsOf(recs.deposits) : [], settings: withRecords ? recs.settings.main : undefined
  }));
  test('real docs, no records: VYLR is the next buy (Fri 30 Oct), nothing to do today', () => {
    const D = run(load(), false);
    assert.equal(D.actions.length, 0);
    assert.equal(D.health.hasData, true); assert.equal(D.health.scanOk, true); assert.equal(D.health.stale, false);
    assert.equal(D.health.triggerId, null);
    assert.deepEqual(plain(D.groups.coming.map((it) => [it.ticker, it.raw.buy_date])), [['VYLR', '2026-10-30']]);
    assert.equal(D.coming[0].ticker, 'VYLR'); assert.equal(D.coming[0].tradingDays, 8);
    assert.equal(D.groups.waiting.length, 11);
    assert.equal(D.groups.missed.length, 13 + 4 + 3); // HOLD the user doesn't hold, done, not_new
    assert.equal(D.head, null); assert.equal(D.account.value, null);
    assert.equal(D.isTradingDay, true); assert.equal(D.openUK, '14:30');
  });
  test('real docs + sample records: positions, sell dates from the real calendar, valuation with FX', () => {
    const real = load();
    const D = run(real, true);
    assert.deepEqual(plain(D.open.map((p) => [p.ticker, p.sellDate])), [['SOLS', '2026-11-19'], ['VSNT', '2027-01-19'], ['HONA', '2027-07-19'], ['ADIG', '2027-08-31']]);
    // the page's own sell date for a rule-following buy equals the engine's model_sell_date
    for (const p of D.open) assert.equal(p.sellDate, real.market.pipeline.items.find((i) => i.ticker === p.ticker).model_sell_date, p.ticker);
    assert.equal(D.open[0].daysLeft, 22);
    assert.deepEqual(plain(D.closed.map((c) => c.ticker)), ['RNA']);
    assert.ok(D.closed[0].ret < 0);
    assert.equal(D.actions.length, 0);
    assert.ok(D.groups.missed.every((it) => !['SOLS', 'VSNT', 'HONA', 'ADIG', 'RNA'].includes(it.ticker)), 'held/bought tickers leave the missed list');
    const sols = D.open[0], s = L.normSeries(real.prices.SOLS), g = L.normSeries(real.prices.GBPUSD);
    near(sols.value, sols.shares * L.priceOn(s, D.asOf) / L.priceOn(g, D.asOf), 1e-9);
    assert.equal(D.asOf, '2026-10-20'); // VUAG has an LSE close for today
    assert.ok(D.account.spinShare > 0.3 && D.account.spinShare < 0.7, 'exposure ' + D.account.spinShare);
    assert.equal(D.warnings.length, 0, JSON.stringify(plain(D.warnings)));
    assert.equal(D.head.annualised, true); assert.ok(D.head.you.xirr > -1 && D.head.you.xirr < 2);
  });
  test('real docs with Yahoo down: null last_market_day / last_actual and extra status fields are handled', () => {
    const real = load();
    Object.assign(real.market.status, { last_market_day: null, scan_ok: false, errors: ['SPY download failed: Yahoo returned 503'], engine: 'spinoff_tracker.py', alerts: { noteworthy: false } });
    real.market.calendar.last_actual = null;
    const D = run(real, true, '2026-10-26T13:00:00Z');
    assert.equal(D.health.lastMarketDay, null); assert.equal(D.health.scanOk, false);
    assert.equal(D.health.errors.length, 1);
    assert.equal(D.today, '2026-10-26'); assert.equal(D.openUK, '13:30'); // UK on GMT, US still on EDT
    assert.equal(D.open.length, 4);
  });
} else console.log('(skipping real engine output tests: ' + REAL + ' not found)');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
