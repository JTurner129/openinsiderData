// Builds fixture.json: a synthetic dataset in the exact db contract shapes used by
// spinoff_tracker.html. Usage: node make_fixture.mjs [--today YYYY-MM-DD] [--out fixture.json]
// Everything is placed relative to "today", so the scenario (one sell due today, two buys
// in their window, one pending price check, ...) holds for any weekday you pick.
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const TODAY = arg('--today', '2026-10-20');
const OUT = arg('--out', join(here, 'fixture.json'));

// ---------- dates ----------
const pad = (n) => String(n).padStart(2, '0');
const P = (s) => new Date(Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10), 12));
const F = (d) => `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
const add = (s, n) => { const d = P(s); d.setUTCDate(d.getUTCDate() + n); return F(d); };
const dow = (s) => P(s).getUTCDay();
const ymd = (y, m, d) => `${y}-${pad(m)}-${pad(d)}`;
function nthWeekday(y, m, wd, n) { // n>=1, or -1 for last
  if (n > 0) { let d = ymd(y, m, 1); while (dow(d) !== wd) d = add(d, 1); return add(d, 7 * (n - 1)); }
  let d = add(ymd(m === 12 ? y + 1 : y, m === 12 ? 1 : m + 1, 1), -1); while (dow(d) !== wd) d = add(d, -1); return d;
}
function easter(y) {
  const a = y % 19, b = Math.floor(y / 100), c = y % 100, d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3), h = (19 * a + b - d - g + 15) % 30, i = Math.floor(c / 4), k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31), day = ((h + l - 7 * m + 114) % 31) + 1;
  return ymd(y, month, day);
}
function observedUS(s) { const w = dow(s); return w === 6 ? add(s, -1) : w === 0 ? add(s, 1) : s; }
function nyseHolidays(y) {
  const H = new Set();
  const ny = ymd(y, 1, 1); if (dow(ny) === 0) H.add(add(ny, 1)); else if (dow(ny) !== 6) H.add(ny);
  H.add(nthWeekday(y, 1, 1, 3)); H.add(nthWeekday(y, 2, 1, 3)); H.add(add(easter(y), -2));
  H.add(nthWeekday(y, 5, 1, -1)); if (y >= 2022) H.add(observedUS(ymd(y, 6, 19))); H.add(observedUS(ymd(y, 7, 4)));
  H.add(nthWeekday(y, 9, 1, 1)); H.add(nthWeekday(y, 11, 4, 4)); H.add(observedUS(ymd(y, 12, 25)));
  return H;
}
function ukHolidays(y) {
  const H = new Set();
  const sub = (s) => { let d = s; while (dow(d) === 0 || dow(d) === 6 || H.has(d)) d = add(d, 1); H.add(d); };
  sub(ymd(y, 1, 1)); H.add(add(easter(y), -2)); H.add(add(easter(y), 1));
  H.add(nthWeekday(y, 5, 1, 1)); H.add(nthWeekday(y, 5, 1, -1)); H.add(nthWeekday(y, 8, 1, -1));
  sub(ymd(y, 12, 25)); sub(ymd(y, 12, 26));
  return H;
}
const weekday = (s) => dow(s) !== 0 && dow(s) !== 6;
function daysBetween(a, b, holi) {
  const out = []; for (let d = a; d <= b; d = add(d, 1)) { if (weekday(d) && !holi(d)) out.push(d); } return out;
}
const nyHol = {}; const isNyHol = (d) => (nyHol[d.slice(0, 4)] ||= nyseHolidays(+d.slice(0, 4))).has(d);
const ukHol = {}; const isUkHol = (d) => (ukHol[d.slice(0, 4)] ||= ukHolidays(+d.slice(0, 4))).has(d);

// ---------- deterministic random ----------
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
// geometric walk over `dates` from p0; optionally pinned so price at pinDate = p0*(1+pinRet) measured from startDate
function walk(dates, p0, vol, drift, seed, pin, dp = 2) {
  const r = rng(seed); let lp = Math.log(p0); const out = [];
  for (let i = 0; i < dates.length; i++) { if (i) lp += drift + vol * gauss(r); out.push(lp); }
  if (pin) {
    const i0 = Math.max(0, dates.findIndex((d) => d >= pin.from)); const i1 = dates.findIndex((d) => d >= pin.to); const j1 = i1 < 0 ? dates.length - 1 : i1;
    const want = Math.log(1 + pin.ret) - (out[j1] - out[i0]);
    for (let i = 0; i < out.length; i++) { const f = i <= i0 ? 0 : i >= j1 ? 1 : (i - i0) / (j1 - i0); out[i] += want * f; }
  }
  const m = 10 ** dp;
  return out.map((x) => Math.round(Math.exp(x) * m) / m);
}

// ---------- calendar ----------
const firstDeposit = (() => { const d = P(TODAY); d.setUTCMonth(d.getUTCMonth() - 13, 1); let s = F(d); while (!weekday(s)) s = add(s, 1); return s; })();
const engineStart = [add(firstDeposit, -7), add(TODAY, -400)].sort()[0];
const allNY = daysBetween(add(engineStart, -150), add(TODAY, 600), isNyHol); // internal: room for first trades before the engine start
const nyDays = allNY.filter((d) => d >= engineStart); // exported calendar
const iT = allNY.indexOf(TODAY);
if (iT < 0) throw new Error(`--today ${TODAY} is not an NYSE trading day; pick a weekday that is not a holiday`);
const lastMarketDay = allNY[iT - 1];
const cal = { days: nyDays, last_actual: lastMarketDay };
const nyAt = (i) => allNY[i];
const histNY = allNY.filter((d) => d <= lastMarketDay);
const lseDays = daysBetween(engineStart, add(TODAY, -1), isUkHol);
const fxDays = daysBetween(engineStart, add(TODAY, -1), () => false);

// ---------- index funds + FX ----------
const series = (id, ticker, currency, kind, dates, close) => ({ id, ticker, currency, kind, dates, close });
const whole = (dates, ret) => ({ from: dates[0], to: dates.at(-1), ret });
const vuag = walk(lseDays, 84.2, 0.0085, 0, 11, whole(lseDays, 0.152));
const eqqq = walk(lseDays, 412.5, 0.0115, 0, 23, whole(lseDays, 0.214));
const gbpusd = walk(fxDays, 1.3412, 0.0042, 0, 37, whole(fxDays, -0.012), 4);
const prices = {
  VUAG: series('VUAG', 'VUAG.L', 'GBP', 'index', lseDays, vuag),
  EQQQ: series('EQQQ', 'EQQQ.L', 'GBP', 'index', lseDays, eqqq),
  GBPUSD: series('GBPUSD', 'GBPUSD=X', 'USD', 'fx', fxDays, gbpusd),
};
const px = (s, d) => { const i = s.dates.findLastIndex((x) => x <= d); return s.close[i >= 0 ? i : 0]; };

// ---------- spin-offs ----------
let seed = 100;
const items = [], checks = {};
function spin({ cik, company, ticker, exchange, firstIdx, status, p0 = 20, vol = 0.026, drift = 0.0004, pin, note = '', form10Lead = 75, url = true, check }) {
  const first = firstIdx == null ? null : nyAt(firstIdx);
  const form10 = first ? add(first, -form10Lead) : add(TODAY, -form10Lead);
  const buy = firstIdx == null ? null : nyAt(firstIdx + 21);
  const wEnd = firstIdx == null ? null : nyAt(firstIdx + 25);
  const sell = firstIdx == null ? null : nyAt(firstIdx + 21 + 252);
  let doc = null;
  if (ticker && first && first <= lastMarketDay) {
    const from = first > engineStart ? first : engineStart;
    const dates = histNY.filter((d) => d >= from);
    const pinAbs = pin ? { from: buy, to: pin.to === 'sell' ? sell : (pin.to || lastMarketDay), ret: pin.ret } : null;
    doc = series(ticker, ticker, 'USD', 'spinoff', dates, walk(dates, p0, vol, drift, seed++, pinAbs));
    prices[ticker] = doc;
  }
  const lastClose = doc ? doc.close.at(-1) : null;
  const before = doc && buy ? px(doc, nyAt(firstIdx + 20)) : null;
  const vol20 = doc ? Math.round(p0 * 210000 * (0.6 + (seed % 7) / 5)) : null;
  items.push({
    id: cik, company, ticker: ticker || null, exchange: exchange || null, form10_date: form10,
    form10_url: url ? `https://www.sec.gov/Archives/edgar/data/${+cik}/000${cik.slice(-6)}2500${pad(seed % 90)}/form10-12b.htm` : null,
    first_trade: first, buy_date: buy, buy_window_end: wEnd, model_sell_date: sell, status, note,
    last_close: lastClose, price_before_buy: buy && buy <= lastMarketDay ? before : null,
    avg_dollar_volume: doc ? vol20 : null, buy_open: buy && buy <= lastMarketDay && doc ? Math.round(px(doc, buy) * 0.995 * 100) / 100 : null,
  });
  if (check) checks[cik] = { genuine: check.genuine, parent: check.parent || '', business: check.business || '', note: check.note || '', checked_at: `${form10}T07:10:00Z` };
  return { first, buy, wEnd, sell, doc };
}
const idxOf = (d) => allNY.findIndex((x) => x >= d);

// The four the user trades
const HRMT = spin({ cik: '0001943318', company: 'Harbor Ridge Materials Inc.', ticker: 'HRMT', exchange: 'NYSE', firstIdx: idxOf(add(firstDeposit, 14)) - 21, status: 'done', p0: 31.4, pin: { ret: 0.27, to: 'sell' }, check: { genuine: true, parent: 'Ridgeline Industries Corp.', business: 'Aggregates and cement for US Gulf Coast construction' } });
const NFIN = spin({ cik: '0001951207', company: 'Northfield Instruments Corp.', ticker: 'NFIN', exchange: 'NASDAQ', firstIdx: iT - 252 - 21, status: 'SELL', p0: 18.75, pin: { ret: 0.34 }, check: { genuine: true, parent: 'Northfield Group Inc.', business: 'Test and measurement instruments for semiconductor fabs' } });
const CRVD = spin({ cik: '0001968844', company: 'Corvid Aerospace Holdings', ticker: 'CRVD', exchange: 'NYSE', firstIdx: idxOf(add(TODAY, -250)) - 21, status: 'HOLD', p0: 42.1, pin: { ret: -0.11 }, check: { genuine: true, parent: 'Atlas Dynamics Corp.', business: 'Aircraft interiors and cabin systems' } });
const ELMF = spin({ cik: '0001972031', company: 'Elmstead Foods Co.', ticker: 'ELMF', exchange: 'NYSE', firstIdx: idxOf(add(TODAY, -125)) - 21, status: 'HOLD', p0: 24.6, pin: { ret: 0.072 }, check: { genuine: true, parent: 'Continental Brands Inc.', business: 'Frozen and chilled prepared meals' } });
// HRMT closes when its 252 days are up
// Buy windows open today
spin({ cik: '0001988120', company: 'Calder Logistics Inc.', ticker: 'CLDR', exchange: 'NYSE', firstIdx: iT - 22, status: 'BUY', p0: 16.2, note: 'Passed: last close and average volume above the minimums', check: { genuine: true, parent: 'Calder Industries Inc.', business: 'Freight brokerage and dedicated truckload' } });
spin({ cik: '0001990457', company: 'Pinecrest Health Services', ticker: 'PCHS', exchange: 'NASDAQ', firstIdx: iT - 21, status: 'BUY', p0: 9.4, note: 'Passed: last close and average volume above the minimums' });
spin({ cik: '0001991302', company: 'Ostara Water Technologies', ticker: 'OSTW', exchange: 'NYSE American', firstIdx: iT - 23, status: 'watch', p0: 7.15, note: 'Buy day reached; price check runs with the next price update', check: { genuine: true, parent: 'Ostara Holdings Inc.', business: 'Municipal water treatment equipment' } });
spin({ cik: '0001987765', company: 'Kearney Rail Systems', ticker: 'KRNX', exchange: 'NYSE', firstIdx: iT - 22, status: 'BUY', p0: 27.8, note: 'Passed: last close and average volume above the minimums', check: { genuine: true, parent: 'Kearney Corp.', business: 'Railcar leasing and maintenance' } });
// Coming up
spin({ cik: '0001993310', company: 'Meridian Payments Corp.', ticker: 'MRDP', exchange: 'NASDAQ', firstIdx: iT + 7 - 21, status: 'watch', p0: 22.3, check: { genuine: true, parent: 'Meridian Financial Group', business: 'Merchant payment processing' } });
spin({ cik: '0001994982', company: 'Lakeshore Brands Inc.', ticker: 'LKSB', exchange: 'NYSE', firstIdx: iT + 16 - 21, status: 'watch', p0: 13.9, check: { genuine: null, note: 'Form 10 describes a split-off exchange offer; distribution terms not final' } });
spin({ cik: '0001995013', company: 'Summit Tracking Holdings', ticker: 'SMTK', exchange: 'NASDAQ', firstIdx: iT + 4 - 21, status: 'watch', p0: 11.0, check: { genuine: false, parent: 'Summit Capital LP', note: 'IPO of a minority stake, not distributed to shareholders' } });
// Registered, not trading
spin({ cik: '0001996541', company: 'Vantor Energy Holdings', ticker: null, firstIdx: null, status: 'waiting', form10Lead: 34, note: 'Form 10 filed; no exchange listing yet' });
spin({ cik: '0001997120', company: 'Glenmoor Insurance Group', ticker: null, firstIdx: null, status: 'waiting', form10Lead: 12, note: 'Form 10 filed; no exchange listing yet', check: { genuine: true, parent: 'Glenmoor Holdings plc', business: 'Specialty commercial insurance' } });
// Missed / skipped
spin({ cik: '0001979904', company: 'Brightline Software Inc.', ticker: 'BLSW', exchange: 'NASDAQ', firstIdx: idxOf(add(TODAY, -190)) - 21, status: 'HOLD', p0: 35.0, check: { genuine: true, parent: 'Bright Horizons Tech Corp.', business: 'Field-service scheduling software' } });
spin({ cik: '0001983377', company: 'Tidewater Microsystems', ticker: 'TDWM', exchange: 'NASDAQ', firstIdx: idxOf(add(TODAY, -70)) - 21, status: 'skip', p0: 1.3, vol: 0.04, drift: -0.004, note: 'Last close $0.84 on the buy day, below the $1 minimum' });
spin({ cik: '0001410022', company: 'Atlas Legacy Corp.', ticker: 'ALGC', exchange: 'NYSE', firstIdx: null, status: 'not_new', note: 'First traded 2019-05-02, years before this Form 10' });
items.find((x) => x.ticker === 'ALGC').first_trade = '2019-05-02';

// ---------- the user's records ----------
const deposits = {}, trades = {};
let n = 0;
const created = (d) => `${d}T19:${pad(10 + (n % 40))}:00.000Z`;
const dep = (date, amt, note) => { deposits[`d${String(++n).padStart(3, '0')}`] = { date, amount_gbp: amt, note, created_at: created(date) }; };
dep(firstDeposit, 12000, 'Money already in the ISA');
{ let d = firstDeposit; for (;;) { const p = P(d); p.setUTCMonth(p.getUTCMonth() + 1, 1); d = F(p); while (!weekday(d)) d = add(d, 1); if (d > TODAY) break; dep(d, 1666.66, 'Monthly deposit'); } }
const trade = (date, type, ticker, doc, gbp, note = '') => {
  const fx = px(prices.GBPUSD, date), usd = px(doc, date);
  const shares = type === 'buy' ? Math.round(gbp / 1.0015 * fx / usd * 1e4) / 1e4 : gbp;
  const amount = type === 'buy' ? gbp : Math.round(shares * usd / fx * 0.9985 * 100) / 100;
  trades[`t${String(++n).padStart(3, '0')}`] = { date, type, ticker, shares, amount_gbp: amount, note, created_at: created(date) };
  return { shares, amount };
};
const hrmtBuy = trade(HRMT.buy, 'buy', 'HRMT', HRMT.doc, 1515.0);
trade(NFIN.buy, 'buy', 'NFIN', NFIN.doc, 1742.35);
trade(HRMT.sell, 'sell', 'HRMT', HRMT.doc, hrmtBuy.shares, '12 months up');
trade(CRVD.buy, 'buy', 'CRVD', CRVD.doc, 2786.4);
trade(ELMF.buy, 'buy', 'ELMF', ELMF.doc, 3519.9);

const status = {
  generated_at: `${TODAY}T05:47:12Z`, today_ny: TODAY, last_market_day: lastMarketDay, scan_ok: true, errors: [],
  counts: { filings: 47, tickers: Object.keys(prices).length - 3 }, trigger_id: 'trig_01FixtureDailyCheck', next_check: 'Weekdays 06:45 UK time',
};
const settings = { index_fund: 'VUAG', slice: 8, monthly_deposit: 1666.66, dismissed: { KRNX: 'Not available on Trading 212' } };

const fixture = {
  today: TODAY,
  now_utc: `${TODAY}T13:05:00Z`,
  collections: {
    market: { status, pipeline: { items }, calendar: cal },
    prices, checks, trades, deposits, settings: { main: settings },
  },
};
writeFileSync(OUT, JSON.stringify(fixture));
console.log(`wrote ${OUT}: today ${TODAY}, ${nyDays.length} NYSE days, ${Object.keys(prices).length} price docs, ${items.length} pipeline items, ${Object.keys(trades).length} trades, ${Object.keys(deposits).length} deposits`);
console.log(`HRMT buy ${HRMT.buy} sell ${HRMT.sell}; NFIN buy ${NFIN.buy} sell ${NFIN.sell}; CRVD buy ${CRVD.buy} sell ${CRVD.sell}; ELMF buy ${ELMF.buy} sell ${ELMF.sell}`);
