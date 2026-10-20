// Screenshot + smoke harness for spinoff_tracker.html.
// Wraps the page in the platform's document skeleton, injects a mock window.claude
// (db backed by fixture.json, user, downloads, mcp), freezes the clock at the fixture's
// "now", then screenshots every tab at 390px and 1100px in light and dark, checks for
// horizontal overflow and console errors, and drives a few interactions.
//
// Real-data mode: the same checks against the REAL engine output (a copy of the scan's docs folder,
// --real-docs DIR), once with no user records, once with the sample records in real_records.json
// (make_real_records.mjs), and once with Yahoo down (null last_market_day / last_actual, scan_ok false).
//
// Run: PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers NODE_PATH=$(npm root -g) node shot.mjs [--out DIR] [--only today,records]
//      [--mode all|fixture|real] [--real-docs DIR]
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const OUT = arg('--out', '/tmp/claude-0/-home-user/cfe4416a-1a27-57e3-996d-df500b1ef602/scratchpad/dash_shots');
const MODE = arg('--mode', 'all');
const REAL_DOCS = arg('--real-docs', '/tmp/claude-0/-home-user/cfe4416a-1a27-57e3-996d-df500b1ef602/scratchpad/trk_real_docs');
mkdirSync(OUT, { recursive: true });

const page = readFileSync(join(here, 'spinoff_tracker.html'), 'utf8');
const fixture = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
// The platform skeleton (charset, viewport-fit=cover, small reset, safe-area padding on :root).
const skeleton = `<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
${page}
</body></html>`;
const pagePath = join(OUT, 'wrapped.html');
writeFileSync(pagePath, skeleton);
const url = pathToFileURL(pagePath).href;

// ---- mock window.claude (runs in the page before any page script) ----
function installMock({ fx, noDb, noMcp }) {
  const clone = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
  const store = clone(fx.collections);
  const listeners = {};
  window.__writes = []; window.__mcpCalls = []; window.__downloads = [];
  let n = 0;
  const meta = { fromCache: false, hasPendingWrites: false };
  const snapDoc = (id, body) => Object.freeze({ id, exists: body !== undefined, data: () => (body === undefined ? undefined : Object.freeze(clone(body))), metadata: meta });
  const deliver = (col, fn) => setTimeout(() => {
    const m = store[col] || {};
    const docs = Object.keys(m).sort().map((id) => snapDoc(id, m[id]));
    fn(Object.freeze({ docs, size: docs.length, empty: !docs.length, docChanges: () => [], metadata: meta }));
  }, 5);
  const emit = (col) => (listeners[col] || []).forEach((fn) => deliver(col, fn));
  const docRef = (col, id) => Object.freeze({
    id, path: col + '/' + id,
    get: async () => snapDoc(id, (store[col] || {})[id]),
    set: async (data) => {
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw { code: 'invalid_argument', message: 'body must be an object' };
      (store[col] ||= {})[id] = clone(data); window.__writes.push(['set', col, id, clone(data)]); emit(col);
    },
    update: async (data) => { if (!(store[col] || {})[id]) throw { code: 'invalid_argument', message: 'missing' }; Object.assign(store[col][id], clone(data)); window.__writes.push(['update', col, id]); emit(col); },
    delete: async () => { if (store[col]) delete store[col][id]; window.__writes.push(['delete', col, id]); emit(col); },
    onSnapshot: (next) => { setTimeout(() => next(snapDoc(id, (store[col] || {})[id])), 5); return () => {}; },
    acquire: async () => ({ acquired: true }),
    collection: () => { throw new TypeError('nested collections are not mocked'); },
  });
  const colRef = (col) => {
    const q = {
      path: col,
      doc: (id) => docRef(col, id || 'm' + (++n) + Math.random().toString(36).slice(2, 10)),
      add: async (data) => { const r = docRef(col, 'a' + (++n)); await r.set(data); return r; },
      onSnapshot: (next) => { (listeners[col] ||= []).push(next); deliver(col, next); return () => {}; },
      get: async () => { const m = store[col] || {}; const docs = Object.keys(m).sort().map((id) => snapDoc(id, m[id])); return { docs, size: docs.length, empty: !docs.length, docChanges: () => [], metadata: meta }; },
      where: () => q, orderBy: () => q, limit: () => q,
    };
    return Object.freeze(q);
  };
  const db = Object.freeze({ doc: (p) => { const s = p.split('/'); return docRef(s[0], s[1]); }, collection: (p) => colRef(p) });
  const user = Object.freeze({ isOwner: async () => true, canEdit: async () => true, can: async () => true, id: async () => 'u_test', me: async () => ({ id: 'u_test', name: '', avatarUrl: '', color: '#888', email: null, isOwner: true, canEdit: true }) });
  const downloads = Object.freeze({ save: async (r) => { window.__downloads.push({ filename: r.filename, size: String(r.data).length, head: String(r.data).slice(0, 200) }); return { status: 'saved' }; } });
  const mcp = Object.freeze({
    callTool: async (server, tool, input) => { window.__mcpCalls.push({ server, tool, input }); return { content: [{ type: 'text', text: '{"ok":true}' }], payload: { ok: true } }; },
    listTools: async () => ({ servers: [{ server: 'Claude Code Remote', authStatus: 'connected', tools: [{ name: 'fire_trigger', description: '' }] }] }),
    watchTool: () => () => {}, invalidate: async () => {}, server: async () => ({}), describeTool: async () => { throw { code: 'bad_request' }; },
  });
  const caps = { db: noDb ? null : db, user, downloads, mcp: noMcp ? null : mcp };
  window.claude = Object.freeze({ use: (name) => new Promise((res) => setTimeout(() => res(name in caps ? caps[name] : null), 20)) });
}

function variant(name) {
  const fx = JSON.parse(JSON.stringify(fixture));
  if (name === 'stale') {
    const s = fx.collections.market.status;
    s.generated_at = '2026-10-14T05:47:12Z'; s.today_ny = '2026-10-14'; s.scan_ok = false;
    s.errors = ['SEC EDGAR full-text search timed out after 3 retries', 'No Yahoo prices for PCHS'];
  }
  if (name === 'firstrun') {
    fx.collections = { market: {}, prices: {}, checks: {}, trades: {}, deposits: {}, settings: {} };
  }
  return fx;
}

// Real engine output as a fixture: {today, now_utc, collections}
function realFixture({ records = false, yahooDown = false } = {}) {
  const collections = { market: {}, prices: {}, checks: {}, trades: {}, deposits: {}, settings: {} };
  for (const col of readdirSync(REAL_DOCS)) {
    collections[col] ||= {};
    for (const f of readdirSync(join(REAL_DOCS, col))) if (f.endsWith('.json')) collections[col][f.slice(0, -5)] = JSON.parse(readFileSync(join(REAL_DOCS, col, f), 'utf8'));
  }
  let now = '2026-10-20T17:30:00Z';
  if (records) {
    const r = JSON.parse(readFileSync(join(here, 'real_records.json'), 'utf8'));
    collections.trades = r.trades; collections.deposits = r.deposits; collections.settings = r.settings;
  }
  if (yahooDown) {
    Object.assign(collections.market.status, { generated_at: '2026-10-26T05:51:02Z', today_ny: '2026-10-26', last_market_day: null, scan_ok: false,
      errors: ['SPY download failed (Yahoo returned 503); kept yesterday\'s spin-off list'], engine_version: 'test', alerts: { noteworthy: false } });
    collections.market.calendar.last_actual = null;
    now = '2026-10-26T13:00:00Z';
  }
  return { today: now.slice(0, 10), now_utc: now, collections };
}

const TABS = ['today', 'performance', 'spinoffs', 'records'];
const problems = [];
const shots = [];

async function open(browser, { width, scheme, fx, noDb = false, noMcp = false, label }) {
  const phone = width < 600;
  const ctx = await browser.newContext({ viewport: { width, height: phone ? 844 : 900 }, deviceScaleFactor: phone ? 2 : 1, colorScheme: scheme, ignoreHTTPSErrors: true, locale: 'en-GB', timezoneId: 'Europe/London' });
  const pg = await ctx.newPage();
  const errs = [];
  pg.on('console', (m) => { if (m.type() === 'error' && !/fonts\.(googleapis|gstatic)/.test(m.text())) errs.push(m.text()); });
  pg.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  await pg.clock.setFixedTime(new Date(fx.now_utc || '2026-10-20T13:05:00Z'));
  await pg.addInitScript(installMock, { fx, noDb, noMcp });
  await pg.goto(url, { waitUntil: 'load' });
  await pg.waitForTimeout(noDb ? 400 : 350);
  await pg.evaluate(() => document.fonts && document.fonts.ready);
  return { ctx, pg, errs, label };
}
async function overflow(pg) {
  return pg.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
}
async function snapTabs(o, name, tabs = TABS) {
  for (const t of tabs) {
    await o.pg.click('#tab-' + t);
    await o.pg.waitForTimeout(t === 'performance' ? 250 : 120);
    const { sw, iw } = await overflow(o.pg);
    if (sw > iw) problems.push(`${name}/${t}: horizontal overflow scrollWidth ${sw} > innerWidth ${iw}`);
    const file = join(OUT, `${name}_${t}.png`);
    await o.pg.screenshot({ path: file, fullPage: true });
    shots.push(file);
  }
}
function report(o) { o.errs.forEach((e) => problems.push(`${o.label}: console error: ${e}`)); }

const browser = await chromium.launch({ proxy: process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: 'localhost,127.0.0.1' } : undefined, args: ['--ignore-certificate-errors'] });
try {
  const only = arg('--only', null);
  if (MODE !== 'real') {
  for (const scheme of ['light', 'dark']) {
    for (const width of [390, 1100]) {
      const name = `${width}_${scheme}`;
      const o = await open(browser, { width, scheme, fx: variant('normal'), label: name });
      await snapTabs(o, name, only ? only.split(',') : TABS);
      report(o); await o.ctx.close();
    }
  }

  // Interactions on the phone layout
  {
    const o = await open(browser, { width: 390, scheme: 'light', fx: variant('normal'), label: 'interact' });
    const pg = o.pg;
    // expand the first buy card's steps, screenshot the Today tab
    await pg.click('#tab-today');
    const buyCard = pg.locator('article[aria-label="Buy OSTW"]');
    await buyCard.locator('summary').click();
    await buyCard.screenshot({ path: join(OUT, 'interact_buycard_steps.png') }); shots.push(join(OUT, 'interact_buycard_steps.png'));
    // Check now -> mcp fire_trigger with the status trigger id
    await pg.click('#checkNowBtn');
    await pg.waitForTimeout(100);
    const calls = await pg.evaluate(() => window.__mcpCalls);
    if (!(calls.length === 1 && calls[0].tool === 'fire_trigger' && calls[0].input.trigger_id === 'trig_01FixtureDailyCheck' && calls[0].server === 'Claude Code Remote')) problems.push('Check now did not call fire_trigger correctly: ' + JSON.stringify(calls));
    const msg = await pg.textContent('.checknow__msg');
    if (!/Started a check; results in a few minutes/.test(msg || '')) problems.push('Check now message missing: ' + msg);
    // I bought it -> sheet prefilled; submit with shares
    await pg.locator('article[aria-label="Buy CLDR"] button', { hasText: 'I bought it' }).click();
    await pg.waitForTimeout(80);
    const pre = await pg.evaluate(() => ({ t: document.getElementById('tradeTicker').value, a: document.getElementById('tradeAmount').value, buy: document.getElementById('tradeTypeBuy').checked, d: document.getElementById('tradeDate').value, focus: document.activeElement.id }));
    if (pre.t !== 'CLDR' || !pre.buy || !(+pre.a > 0) || pre.d !== '2026-10-20' || pre.focus !== 'tradeShares') problems.push('Buy sheet prefill wrong: ' + JSON.stringify(pre));
    await pg.screenshot({ path: join(OUT, 'interact_sheet.png') }); shots.push(join(OUT, 'interact_sheet.png'));
    await pg.click('#tradeSubmit'); // shares empty -> validation error
    const err = await pg.textContent('#tradeSharesErr');
    if (!/number of shares/.test(err || '')) problems.push('Missing shares validation not shown: ' + err);
    await pg.fill('#tradeShares', '296.1');
    await pg.click('#tradeSubmit');
    await pg.waitForTimeout(150);
    let w = await pg.evaluate(() => window.__writes);
    const tw = w.find((x) => x[1] === 'trades');
    if (!tw || tw[3].ticker !== 'CLDR' || tw[3].type !== 'buy' || tw[3].shares !== 296.1 || !(tw[3].amount_gbp > 0) || !tw[3].created_at) problems.push('Trade write wrong: ' + JSON.stringify(tw));
    const stillBuy = await pg.locator('article[aria-label="Buy CLDR"]').count();
    if (stillBuy) problems.push('CLDR buy card still shown after recording the buy');
    // I sold it -> sell sheet, amount empty and focused
    await pg.locator('article[aria-label="Sell NFIN"] button', { hasText: 'I sold it' }).click();
    await pg.waitForTimeout(80);
    const spre = await pg.evaluate(() => ({ sell: document.getElementById('tradeTypeSell').checked, sh: document.getElementById('tradeShares').value, focus: document.activeElement.id, lbl: document.getElementById('tradeAmountLbl').textContent }));
    if (!spre.sell || !(+spre.sh > 0) || spre.focus !== 'tradeAmount' || !/received/.test(spre.lbl)) problems.push('Sell sheet prefill wrong: ' + JSON.stringify(spre));
    await pg.keyboard.press('Escape');
    if (!(await pg.locator('#sheet').isHidden())) problems.push('Escape did not close the sheet');
    // Not on Trading 212 -> settings.dismissed
    await pg.locator('article[aria-label="Buy PCHS"] button', { hasText: 'Not on Trading 212' }).click();
    await pg.waitForTimeout(120);
    w = await pg.evaluate(() => window.__writes);
    const sw = w.filter((x) => x[1] === 'settings').pop();
    if (!sw || sw[3].dismissed.PCHS !== 'Not available on Trading 212' || sw[3].dismissed.KRNX !== 'Not available on Trading 212' || sw[3].slice !== 8) problems.push('Dismiss write wrong: ' + JSON.stringify(sw));
    await pg.screenshot({ path: join(OUT, 'interact_after_actions.png'), fullPage: true }); shots.push(join(OUT, 'interact_after_actions.png'));
    // Records: quick deposit, delete with two-step confirm, CSV
    await pg.click('#tab-records');
    await pg.click('#depQuick');
    await pg.waitForTimeout(120);
    w = await pg.evaluate(() => window.__writes);
    const dw = w.filter((x) => x[1] === 'deposits').pop();
    if (!dw || dw[3].amount_gbp !== 1666.66 || dw[3].date !== '2026-10-20') problems.push('Quick deposit write wrong: ' + JSON.stringify(dw));
    const firstDel = pg.locator('#recTrades .rec button', { hasText: 'Delete' }).first();
    await firstDel.click();
    await pg.waitForTimeout(60);
    const confirmShown = await pg.locator('#recTrades .rec__confirm').count();
    if (!confirmShown) problems.push('Delete confirm not shown');
    await pg.locator('#recTrades').screenshot({ path: join(OUT, 'interact_delete_confirm.png') }); shots.push(join(OUT, 'interact_delete_confirm.png'));
    await pg.locator('#recTrades .rec__confirm button', { hasText: 'Keep' }).click();
    await pg.click('#recDownload');
    await pg.waitForTimeout(60);
    const dl = await pg.evaluate(() => window.__downloads);
    if (!(dl.length === 1 && /\.csv$/.test(dl[0].filename) && /^record,date,type/.test(dl[0].head))) problems.push('CSV download wrong: ' + JSON.stringify(dl));
    // Settings: switch to EQQQ and save
    await pg.check('#setFundEQQQ');
    await pg.check('#setRiskCustom');
    await pg.fill('#setSlice', '10');
    await pg.click('#setSave');
    await pg.waitForTimeout(120);
    w = await pg.evaluate(() => window.__writes);
    const st = w.filter((x) => x[1] === 'settings').pop();
    if (!st || st[3].index_fund !== 'EQQQ' || st[3].slice !== 10) problems.push('Settings write wrong: ' + JSON.stringify(st));
    // tab via hash
    await pg.evaluate(() => { location.hash = '#performance'; });
    await pg.waitForTimeout(150);
    if ((await pg.getAttribute('#tab-performance', 'aria-selected')) !== 'true') problems.push('hash #performance did not switch tab');
    report(o); await o.ctx.close();
  }

  // Chart hover on desktop
  {
    const o = await open(browser, { width: 1100, scheme: 'light', fx: variant('normal'), label: 'hover' });
    await o.pg.click('#tab-performance');
    await o.pg.waitForTimeout(250);
    const box = await o.pg.locator('#chart svg').boundingBox();
    await o.pg.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);
    await o.pg.waitForTimeout(80);
    await o.pg.locator('#chart').screenshot({ path: join(OUT, 'hover_chart_1100.png') }); shots.push(join(OUT, 'hover_chart_1100.png'));
    report(o); await o.ctx.close();
  }
  {
    const o = await open(browser, { width: 390, scheme: 'dark', fx: variant('normal'), label: 'hover-phone' });
    await o.pg.click('#tab-performance');
    await o.pg.waitForTimeout(250);
    const box = await o.pg.locator('#chart svg').boundingBox();
    await o.pg.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.5);
    await o.pg.waitForTimeout(80);
    await o.pg.locator('#chart').screenshot({ path: join(OUT, 'hover_chart_390_dark.png') }); shots.push(join(OUT, 'hover_chart_390_dark.png'));
    report(o); await o.ctx.close();
  }

  // Absence and degraded states
  {
    const o = await open(browser, { width: 390, scheme: 'light', fx: variant('normal'), noDb: true, label: 'nodb' });
    await snapTabs(o, 'nodb_390_light', ['today', 'records']);
    report(o); await o.ctx.close();
  }
  {
    const o = await open(browser, { width: 390, scheme: 'light', fx: variant('stale'), noMcp: true, label: 'stale' });
    await snapTabs(o, 'stale_390_light', ['today']);
    if (await o.pg.locator('#checkNowBtn').count()) problems.push('Check now shown without mcp');
    report(o); await o.ctx.close();
  }
  {
    const o = await open(browser, { width: 390, scheme: 'light', fx: variant('firstrun'), label: 'firstrun' });
    await snapTabs(o, 'firstrun_390_light', ['today', 'performance', 'spinoffs']);
    report(o); await o.ctx.close();
  }
  } // fixture mode

  // ---- real engine output ----
  if (MODE !== 'fixture') {
    if (!existsSync(REAL_DOCS)) problems.push('real docs folder not found: ' + REAL_DOCS);
    else {
      const expectText = async (o, sel, re, what) => { const t = (await o.pg.textContent(sel)) || ''; if (!re.test(t)) problems.push(`${o.label}: ${what}: ${JSON.stringify(t.slice(0, 200))}`); };
      for (const [width, scheme] of [[390, 'light'], [1100, 'dark']]) {
        const o = await open(browser, { width, scheme, fx: realFixture(), label: `real_norec_${width}_${scheme}` });
        await snapTabs(o, `real_norec_${width}_${scheme}`, only ? only.split(',') : TABS);
        await o.pg.click('#tab-today');
        await expectText(o, '#todayActions', /Start by logging your money/, 'onboarding card missing');
        await expectText(o, '#todayComing', /VYLR/, 'VYLR not in coming up');
        if (await o.pg.locator('#checkNowBtn').count()) problems.push(o.label + ': Check now shown although trigger_id is null');
        report(o); await o.ctx.close();
      }
      for (const [width, scheme] of [[390, 'light'], [390, 'dark'], [1100, 'light']]) {
        const o = await open(browser, { width, scheme, fx: realFixture({ records: true }), label: `real_rec_${width}_${scheme}` });
        await snapTabs(o, `real_rec_${width}_${scheme}`, only ? only.split(',') : TABS);
        await o.pg.click('#tab-today');
        await expectText(o, '#todayActions', /Nothing to do today[\s\S]*VYLR[\s\S]*Friday 30 Oct/, 'nothing-to-do card should name the VYLR buy');
        await expectText(o, '#todaySummary', /Spin-offs: \d+% of your account/, 'exposure line missing');
        await o.pg.click('#tab-spinoffs');
        await expectText(o, '#spinPositions', /SOLS[\s\S]*Thursday 19 Nov 2026/, 'SOLS sell date');
        report(o); await o.ctx.close();
      }
      {
        const o = await open(browser, { width: 390, scheme: 'light', fx: realFixture({ records: true, yahooDown: true }), label: 'real_yahoodown' });
        await snapTabs(o, 'real_yahoodown_390_light', ['today']);
        await expectText(o, '#todayStatus', /didn’t finish[\s\S]*Yahoo returned 503/, 'scan_ok false banner');
        report(o); await o.ctx.close();
      }
      // risk level control on real data
      {
        const o = await open(browser, { width: 390, scheme: 'light', fx: realFixture({ records: true }), label: 'real_risk' });
        const pg = o.pg;
        await pg.click('#tab-records');
        if (!(await pg.isChecked('#setRisk8'))) problems.push('risk: Pure not selected for slice 8');
        if (!(await pg.locator('#setSliceField').isHidden())) problems.push('risk: custom field visible for slice 8');
        await pg.check('#setRiskCustom');
        if (await pg.locator('#setSliceField').isHidden()) problems.push('risk: custom field hidden after choosing Custom');
        await pg.fill('#setSlice', '41');
        await pg.click('#setSave');
        await expectText(o, '#setSliceErr', /4 to 40/, 'custom slice 41 should be refused');
        await pg.fill('#setSlice', '12');
        await pg.click('#setSave');
        await pg.waitForTimeout(120);
        let w = (await pg.evaluate(() => window.__writes)).filter((x) => x[1] === 'settings').pop();
        if (!w || w[3].slice !== 12) problems.push('risk: custom 12 not saved: ' + JSON.stringify(w));
        await pg.check('#setRisk27');
        await pg.click('#setSave');
        await pg.waitForTimeout(120);
        w = (await pg.evaluate(() => window.__writes)).filter((x) => x[1] === 'settings').pop();
        if (!w || w[3].slice !== 27) problems.push('risk: lower risk 27 not saved: ' + JSON.stringify(w));
        await pg.locator('#settingsPanel').screenshot({ path: join(OUT, 'real_settings_risk.png') }); shots.push(join(OUT, 'real_settings_risk.png'));
        await pg.click('#tab-today');
        await expectText(o, '#todaySummary', /Risk level: Lower risk \(1\/27 each\)/, 'summary should show the new risk level');
        report(o); await o.ctx.close();
      }
    }
  }
} finally {
  await browser.close();
}

console.log('screenshots:');
shots.forEach((s) => console.log('  ' + s));
if (problems.length) { console.log('\nPROBLEMS:'); problems.forEach((p) => console.log('  - ' + p)); process.exit(1); }
console.log('\nno overflow, no console errors, interactions OK');
