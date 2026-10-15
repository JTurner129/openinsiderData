// Opens report.html in headless Chromium and checks that every chart rendered
// and no table is empty. Usage: node tests/check_report.mjs [path/to/report.html]
import { createRequire } from 'module';
import path from 'path';

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }

const file = path.resolve(process.argv[2] || 'report.html');
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + file, { waitUntil: 'load', timeout: 120000 });
await page.waitForTimeout(3000);

const result = await page.evaluate(() => {
  const plots = [...document.querySelectorAll('.js-plotly-plot')];
  const charts = plots.map(p => {
    const title = p.querySelector('.gtitle')?.textContent || '(untitled)';
    const traces = (p.data || []).length;
    const marks = p.querySelectorAll('.trace, .hm, .heatmaplayer image, .point, path.js-line, .bars path').length;
    const vals = a => Array.isArray(a) ? a.flat() : (ArrayBuffer.isView(a) ? Array.from(a) : []);
    const finite = (p.data || []).some(t => ['y', 'z', 'x'].some(k =>
      vals(t[k]).some(v => v !== null && v !== undefined && !(typeof v === 'number' && isNaN(v)))));
    // Histograms keep data binary-encoded; count drawn, non-degenerate bars instead.
    const drawnBars = [...p.querySelectorAll('.bars path')].filter(el => {
      const d = el.getAttribute('d') || ''; return d && d !== 'M0,0Z' && !/^M[\d.]+,([\d.]+)V\1Z$/.test(d);
    }).length;
    // Scatter data is binary-encoded too; count drawn markers.
    const drawnPoints = p.querySelectorAll('.scatterlayer .point').length;
    return { title, traces, marks, finite: finite || drawnBars > 0 || drawnPoints > 0 };
  });
  const tables = [...document.querySelectorAll('table')].map(t => ({
    head: [...t.querySelectorAll('th')].slice(0, 3).map(th => th.textContent).join(' | '),
    rows: t.querySelectorAll('tbody tr').length,
  }));
  return { charts, tables };
});
await page.screenshot({ path: path.join(path.dirname(file), 'results', 'report_screenshot.png'), fullPage: false });
await browser.close();

let bad = 0;
for (const c of result.charts) {
  const ok = c.traces > 0 && c.marks > 0 && c.finite;
  if (!ok) { bad++; console.log('EMPTY CHART:', c.title, JSON.stringify(c)); }
}
for (const t of result.tables) {
  if (t.rows === 0) { bad++; console.log('EMPTY TABLE:', t.head); }
}
console.log(`charts: ${result.charts.length}, tables: ${result.tables.length}, problems: ${bad}, page errors: ${errors.length}`);
errors.slice(0, 5).forEach(e => console.log('  error:', e));
process.exit(bad || errors.length ? 1 : 0);
