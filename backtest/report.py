"""Build report.html (interactive Plotly), PNG charts and results.xlsx."""
import html
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .common import CHARTS, DATA, RESULTS, ROOT, SCREENS, get_logger, load_config

LOG = get_logger('report')
for _n in ('kaleido', 'choreographer'):
    logging.getLogger(_n).setLevel(logging.WARNING)

# Reference categorical palette (fixed order) and chart chrome.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SCREEN_COLOR = {k: SERIES[i] for i, k in enumerate(SCREENS)}
BENCH_STYLE = {'SPY': dict(color='#52514e', dash='dash'), 'IWM': dict(color='#898781', dash='dot')}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
POS, NEG, MID = '#008300', '#d03b3b', '#f0efec'
HORIZONS = ['1w', '1m', '3m', '6m', '1y']
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def pct(x, d=1):
    return '–' if x is None or pd.isna(x) else f'{x * 100:.{d}f}%'


def money(x):
    return '–' if x is None or pd.isna(x) else f'${x:,.0f}'


def num(x, d=2):
    return '–' if x is None or pd.isna(x) else f'{x:.{d}f}'


def base_layout(fig, title, xtitle=None, ytitle=None, height=420):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor='left', y=0.985, yref='container', yanchor='top',
                   font=dict(size=16, color=INK)),
        font=dict(family=FONT, size=12, color=INK2), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height + 40, margin=dict(l=60, r=20, t=110, b=50), hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor='white', font=dict(family=FONT, color=INK)))
    fig.update_xaxes(title=xtitle, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


class Charts:
    def __init__(self):
        self.figs = []   # (name, fig)
        CHARTS.mkdir(parents=True, exist_ok=True)

    def add(self, name, fig, caption):
        self.figs.append((name, fig))
        div = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displaylogo': False, 'responsive': True})
        return f'<figure class="chart">{div}<figcaption>{html.escape(caption)}</figcaption></figure>'

    def write_pngs(self):
        for name, fig in self.figs:
            fig.write_image(CHARTS / f'{name}.png', width=1100, height=fig.layout.height or 420, scale=2)


# --- Chart builders -----------------------------------------------------------
def heatmap(es, screen, min_n):
    d = es[(es['screen'] == screen) & (es['signal_year'] != 'All')]
    years = sorted(d['signal_year'].unique())
    z, grey, custom, notes = [], [], [], []
    for y in years:
        row, grow, crow = [], [], []
        for h in HORIZONS:
            c = d[(d['signal_year'] == y) & (d['horizon'] == h)]
            n = int(c['count'].iloc[0]) if len(c) and pd.notna(c['count'].iloc[0]) else 0
            v = c['mean_xs_spy'].iloc[0] if n else np.nan
            ok = n >= min_n and pd.notna(v)
            row.append(v * 100 if ok else np.nan)
            grow.append(0 if (n and not ok) else np.nan)
            crow.append([n, '' if ok else f' (fewer than {min_n} signals: greyed out)', v * 100 if n else np.nan])
            if n:
                notes.append((HORIZONS.index(h), years.index(y), v * 100, n, ok))
        z.append(row)
        grey.append(grow)
        custom.append(crow)
    arr = np.array(z, dtype=float)
    zmax = max(np.nanpercentile(np.abs(arr), 95), 1) if np.isfinite(arr).any() else 10
    fig = go.Figure()
    xi, yi = list(range(len(HORIZONS))), list(range(len(years)))
    yl = [[y] * len(HORIZONS) for y in years]
    xl = [HORIZONS] * len(years)
    custom = [[c + [yy, xx] for c, yy, xx in zip(cr, yr, xr)] for cr, yr, xr in zip(custom, yl, xl)]
    fig.add_trace(go.Heatmap(z=grey, x=xi, y=yi, colorscale=[[0, '#d6d5cf'], [1, '#d6d5cf']], showscale=False,
                             customdata=custom, xgap=2, ygap=2,
                             hovertemplate='Signal year %{customdata[3]}, horizon %{customdata[4]}<br>Mean excess vs SPY: %{customdata[2]:.1f}%'
                                           '<br>Signals: %{customdata[0]}%{customdata[1]}<extra></extra>'))
    fig.add_trace(go.Heatmap(
        z=z, x=xi, y=yi, colorscale=[[0, NEG], [0.5, MID], [1, POS]], zmin=-zmax, zmax=zmax, zmid=0,
        xgap=2, ygap=2, customdata=custom, colorbar=dict(title='Excess vs SPY', ticksuffix='%'),
        hovertemplate='Signal year %{customdata[3]}, horizon %{customdata[4]}<br>Mean excess vs SPY: %{z:.1f}%<br>Signals: %{customdata[0]}<extra></extra>'))
    for h, y, v, n, ok in notes:
        strong = ok and abs(v) / zmax > 0.55
        fig.add_annotation(x=h, y=y, text=f'{v:.1f}%  <span style="font-size:10px">n={n}</span>', showarrow=False,
                           font=dict(size=11, color='white' if strong else (INK if ok else MUTED)))
    base_layout(fig, f'{SCREENS[screen][0]}: mean excess return vs SPY by signal year', 'Horizon after entry',
                'Signal year', height=max(380, 24 * len(years) + 140))
    fig.update_layout(hovermode='closest')
    fig.update_yaxes(tickvals=yi, ticktext=years, autorange='reversed', gridcolor='rgba(0,0,0,0)', zeroline=False)
    fig.update_xaxes(tickvals=xi, ticktext=HORIZONS, gridcolor='rgba(0,0,0,0)', side='top', zeroline=False,
                     title='Horizon after entry')
    return fig


def log_ticks(fig, values, **kw):
    """Readable dollar ticks on a log axis (1-2-5 sequence)."""
    lo, hi = max(min(values), 1), max(values)
    ticks = [m * 10 ** e for e in range(2, 9) for m in (1, 2, 5) if lo / 1.5 <= m * 10 ** e <= hi * 1.5]
    fig.update_yaxes(type='log', tickvals=ticks, ticktext=[f'${t / 1000:,.0f}K' if t < 1e6 else f'${t / 1e6:,.1f}M'
                                                          for t in ticks], **kw)


def horizon_bars(es, screen):
    d = es[(es['screen'] == screen) & (es['signal_year'] == 'All')].set_index('horizon').reindex(HORIZONS)
    fig = go.Figure()
    for name, col, color, pattern in [(SCREENS[screen][0], 'mean', SCREEN_COLOR[screen], ''),
                                      ('SPY (same dates)', 'mean_spy', BENCH_STYLE['SPY']['color'], '/'),
                                      ('IWM (same dates)', 'mean_iwm', BENCH_STYLE['IWM']['color'], '.')]:
        fig.add_trace(go.Bar(x=HORIZONS, y=d[col] * 100, name=name, marker=dict(color=color, pattern_shape=pattern,
                             line=dict(color=SURFACE, width=2)),
                             text=[pct(v) for v in d[col]], textposition='outside', textfont=dict(color=INK2, size=10),
                             hovertemplate='%{x}: %{y:.1f}%<extra>' + name + '</extra>'))
    base_layout(fig, f'{SCREENS[screen][0]}: average return by horizon vs benchmarks', 'Horizon after entry',
                'Average return (%)')
    fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(ticksuffix='%')
    return fig


def equity_drawdown(daily, screen, hold='1y'):
    col = f'{screen}|{hold}'
    r = daily[col].dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    vals = []
    for name, series, style in [(SCREENS[screen][0], r, dict(color=SCREEN_COLOR[screen])),
                                ('SPY', daily['SPY'].reindex(r.index).fillna(0), BENCH_STYLE['SPY']),
                                ('IWM', daily['IWM'].reindex(r.index).fillna(0), BENCH_STYLE['IWM'])]:
        w = 10_000 * (1 + series).cumprod()
        dd = w / w.cummax() - 1
        vals += [w.min(), w.max()]
        fig.add_trace(go.Scatter(x=w.index, y=w, name=name, line=dict(width=2, **style), legendgroup=name,
                                 hovertemplate=name + ': $%{y:,.0f}<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name=name, line=dict(width=1.5, **style), legendgroup=name,
                                 showlegend=False, hovertemplate=name + ' drawdown: %{y:.1f}%<extra></extra>'),
                      row=2, col=1)
    base_layout(fig, f'{SCREENS[screen][0]}: growth of $10,000 (calendar-time portfolio, 1-year hold) and drawdown',
                height=560)
    log_ticks(fig, vals, title='Portfolio value (log scale)', row=1, col=1)
    fig.update_yaxes(title='Drawdown (%)', ticksuffix='%', row=2, col=1)
    fig.update_xaxes(title='Date', row=2, col=1)
    return fig


def histogram(trades, screen):
    r = trades[(trades['screen'] == screen)]['ret_1y'].dropna() * 100
    fig = go.Figure()
    if len(r):
        lo, hi = np.percentile(r, [0.5, 99.5])
        clipped = r.clip(lo, hi)
        fig.add_trace(go.Histogram(x=clipped, nbinsx=60, marker=dict(color=SCREEN_COLOR[screen], line=dict(color=SURFACE, width=1)),
                                   name='Trades', hovertemplate='1-year return %{x}<br>Trades: %{y}<extra></extra>'))
        for v, label, dash in [(r.mean(), f'Mean {r.mean():.1f}%', 'solid'), (r.median(), f'Median {r.median():.1f}%', 'dash')]:
            fig.add_vline(x=v, line=dict(color=INK, width=1.5, dash=dash))
            fig.add_annotation(x=v, y=1, yref='paper', text=label, showarrow=False, yanchor='bottom',
                               xanchor='left' if 'Mean' in label else 'right', font=dict(color=INK, size=11))
    base_layout(fig, f'{SCREENS[screen][0]}: distribution of 1-year returns per trade', '1-year return (%)',
                'Number of trades')
    fig.update_xaxes(ticksuffix='%')
    fig.update_layout(hovermode='closest', showlegend=False)
    return fig


def all_equity(daily, hold='1y'):
    fig = go.Figure()
    vals = []
    start = min(daily[f'{s}|{hold}'].first_valid_index() for s in SCREENS if f'{s}|{hold}' in daily)
    for s in SCREENS:
        r = daily[f'{s}|{hold}'].dropna()
        w = 10_000 * (1 + r).cumprod()
        vals += [w.min(), w.max()]
        fig.add_trace(go.Scatter(x=w.index, y=w, name=SCREENS[s][0], line=dict(width=2, color=SCREEN_COLOR[s]),
                                 hovertemplate=SCREENS[s][0] + ': $%{y:,.0f}<extra></extra>'))
    for b in ['SPY', 'IWM']:
        r = daily[b].loc[start:].fillna(0)
        w = 10_000 * (1 + r).cumprod()
        vals += [w.min(), w.max()]
        fig.add_trace(go.Scatter(x=w.index, y=w, name=b, line=dict(width=2, **BENCH_STYLE[b]),
                                 hovertemplate=b + ': $%{y:,.0f}<extra></extra>'))
    base_layout(fig, 'All screens: growth of $10,000 (calendar-time portfolios, 1-year hold)', 'Date',
                'Portfolio value (log scale)', height=520)
    log_ticks(fig, vals)
    return fig


def alpha_bars(alphas, hold='1y'):
    d = alphas[(alphas['holding'] == hold) & (alphas['model'] == 'FF5+UMD') & alphas['screen'].isin(SCREENS)]
    d = d.set_index('screen').reindex(list(SCREENS))
    fig = go.Figure()
    for model, opacity, pattern in [('CAPM', 0.45, '/'), ('FF5+UMD', 1.0, '')]:
        dm = alphas[(alphas['holding'] == hold) & (alphas['model'] == model)].set_index('screen').reindex(list(SCREENS))
        fig.add_trace(go.Bar(
            x=[SCREENS[s][0] for s in SCREENS], y=dm['alpha_ann'] * 100, name=f'{model} alpha',
            marker=dict(color=[SCREEN_COLOR[s] for s in SCREENS], opacity=opacity, pattern_shape=pattern,
                        line=dict(color=SURFACE, width=2)),
            error_y=dict(type='data', symmetric=False, array=(dm['ci95_high'] - dm['alpha_ann']) * 100,
                         arrayminus=(dm['alpha_ann'] - dm['ci95_low']) * 100, color=INK2, thickness=1.2, width=5),
            customdata=np.c_[dm['t_alpha'], dm['ci95_low'] * 100, dm['ci95_high'] * 100],
            hovertemplate='%{x}<br>' + model + ' alpha: %{y:.1f}% a year<br>95% CI: %{customdata[1]:.1f}% to %{customdata[2]:.1f}%'
                          '<br>t-stat: %{customdata[0]:.2f}<extra></extra>'))
    base_layout(fig, 'Annualized alpha per screen (1-year hold) with 95% confidence intervals', None,
                'Alpha (% per year)', height=460)
    fig.update_layout(barmode='group', hovermode='closest')
    fig.update_yaxes(ticksuffix='%')
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig


def excess_lines(es):
    fig = go.Figure()
    for s in SCREENS:
        d = es[(es['screen'] == s) & (es['signal_year'] == 'All')].set_index('horizon').reindex(HORIZONS)
        fig.add_trace(go.Scatter(x=HORIZONS, y=d['mean_xs_spy'] * 100, name=SCREENS[s][0], mode='lines+markers',
                                 line=dict(width=2, color=SCREEN_COLOR[s]), marker=dict(size=8),
                                 hovertemplate=SCREENS[s][0] + ': %{y:.1f}%<extra></extra>'))
    base_layout(fig, 'Average excess return vs SPY across horizons', 'Horizon after entry', 'Mean excess vs SPY (%)')
    fig.update_yaxes(ticksuffix='%')
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig


REALISM_SHORT = {
    '1. As reported (open, daily rebalance, 0.5% cost)': 'As reported',
    '2. + estimated spread costs': '+ spread costs',
    '3. + buy-and-hold weights': '+ buy-and-hold',
    '4. + enter at next close': '+ next close',
    '5. Stress: 2× spread costs': '2× costs',
    '6. Realistic + price ≥ $2': '+ price ≥ $2',
    '7. Realistic + $100K/day volume': '+ $100K/day',
    '8. Realistic + $1M/day volume': '+ $1M/day',
}


def realism_chart(realism, hold='3m'):
    d = realism[realism['hold'] == hold]
    fig = go.Figure()
    for s in SCREENS:
        x = d[d['screen'] == s]
        fig.add_trace(go.Bar(x=[REALISM_SHORT.get(v, v) for v in x['step']], y=x['FF5+UMD+IWC alpha'] * 100,
                             name=SCREENS[s][0], marker=dict(color=SCREEN_COLOR[s], line=dict(color=SURFACE, width=2)),
                             customdata=np.c_[x['FF5+UMD+IWC t'], x['signals']],
                             hovertemplate=SCREENS[s][0] + '<br>%{x}: %{y:.1f}% a year<br>t = %{customdata[0]:.2f}, '
                                                           'signals %{customdata[1]:,}<extra></extra>'))
    base_layout(fig, f'Alpha as trading assumptions get more realistic ({hold} hold, FF5 + momentum + micro-cap factor)',
                'Assumptions (each step adds to the previous; steps 6–8 add a filter to step 4)', 'Alpha (% per year)', height=480)
    fig.update_layout(barmode='group', hovermode='closest')
    fig.update_yaxes(ticksuffix='%')
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig


def filter_scatter(tests, hold='3m'):
    d = tests[(tests['hold'] == hold) & (tests['filter'] != 'No filter (realistic)')]
    fig = go.Figure()
    for s, symbol in zip(['cluster3', 'exec200k'], ['circle', 'diamond']):
        x = d[d['screen'] == s]
        fig.add_trace(go.Scatter(
            x=x['train vs baseline'] * 100, y=x['test vs baseline'] * 100, mode='markers', name=SCREENS[s][0],
            marker=dict(size=10, symbol=symbol, color=SCREEN_COLOR[s], line=dict(color=SURFACE, width=2)),
            text=x['filter'], customdata=np.c_[x['test t'], x['test_signals']],
            hovertemplate='%{text}<br>2006–15 change: %{x:+.1f}%/yr<br>2016–26 change: %{y:+.1f}%/yr'
                          '<br>2016–26 t = %{customdata[0]:.2f}, signals %{customdata[1]:,}<extra>' + SCREENS[s][0] + '</extra>'))
    base_layout(fig, f'Does a filter that helped in 2006–2015 still help in 2016–2026? ({hold} hold)',
                'Change in alpha vs no filter, 2006–2015 (%/yr)', 'Change in alpha vs no filter, 2016–2026 (%/yr)', height=520)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(ticksuffix='%')
    fig.update_yaxes(ticksuffix='%')
    fig.update_layout(hovermode='closest')
    return fig


def survivors_chart(tests):
    d = tests[tests['holds out of sample']].copy()
    if d.empty:
        d = tests[tests['filter'] != 'No filter (realistic)'].nlargest(10, 'test alpha')
    d = d.sort_values('test alpha').tail(15)
    labels = [f"{SCREENS[s][0].split(' (')[0]} · {f} · {h}" for s, f, h in zip(d['screen'], d['filter'], d['hold'])]
    se = (d['test alpha'] / d['test t']).abs()
    fig = go.Figure(go.Bar(
        y=labels, x=d['test alpha'] * 100, orientation='h',
        marker=dict(color=[SCREEN_COLOR[s] for s in d['screen']], line=dict(color=SURFACE, width=2)),
        error_x=dict(type='data', array=1.96 * se * 100, color=INK2, thickness=1.2, width=4),
        customdata=np.c_[d['test t'], d['test_signals'], d['train alpha'] * 100],
        hovertemplate='%{y}<br>2016–26 alpha: %{x:.1f}%/yr (t = %{customdata[0]:.2f}, %{customdata[1]:,} signals)'
                      '<br>2006–15 alpha: %{customdata[2]:.1f}%/yr<extra></extra>'))
    base_layout(fig, 'Filters that helped in both periods: 2016–2026 alpha after estimated costs, with 95% intervals',
                'Alpha 2016–2026 (% per year)', None, height=max(380, 34 * len(d) + 160))
    fig.update_xaxes(ticksuffix='%')
    fig.update_layout(hovermode='closest', showlegend=False, margin=dict(l=380))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    return fig


def decay_bars(alphas):
    holds = ['1m', '3m', '6m', '1y']
    fig = go.Figure()
    for s in SCREENS:
        d = alphas[(alphas['screen'] == s) & (alphas['model'] == 'FF5+UMD')].set_index('holding').reindex(holds)
        fig.add_trace(go.Bar(x=holds, y=d['alpha_ann'] * 100, name=SCREENS[s][0],
                             marker=dict(color=SCREEN_COLOR[s], line=dict(color=SURFACE, width=2)),
                             customdata=d['t_alpha'],
                             hovertemplate=SCREENS[s][0] + ' (%{x} hold): %{y:.1f}% a year, t=%{customdata:.2f}<extra></extra>'))
    base_layout(fig, 'FF5+momentum alpha by holding period (where the edge decays)', 'Holding period',
                'Alpha (% per year)')
    fig.update_layout(barmode='group', hovermode='closest')
    fig.update_yaxes(ticksuffix='%')
    return fig


# --- HTML tables ---------------------------------------------------------------
def cell_color(v, scale=0.3):
    if v is None or pd.isna(v):
        return ''
    a = min(abs(v) / scale, 1) * 0.55 + 0.08
    rgb = '0,131,0' if v > 0 else '208,59,59'
    return f'background:rgba({rgb},{a:.2f})'


def html_table(df, formats, color_cols=(), color_scale=0.3, cls='tbl'):
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            f = formats.get(c)
            txt = f(v) if f else ('–' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            style = cell_color(v, color_scale) if c in color_cols else ''
            tds.append(f'<td style="{style}">{html.escape(str(txt))}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def signal_table_json(trades, screen):
    t = trades[trades['screen'] == screen].sort_values('signal_date', ascending=False)
    rows = []
    for _, r in t.iterrows():
        tk = r['ticker'] or ''
        if r.get('ticker_source') == 'sec_current':
            tk = f"{tk or '?'} → {r['price_ticker']}"
        rows.append([r['signal_date'].strftime('%Y-%m-%d'), tk, r['company_name'] or '',
                     int(r['n_insiders']), round(float(r['total_value'] or 0)),
                     *[None if pd.isna(r[f'ret_{h}']) else round(float(r[f'ret_{h}']) * 100, 1) for h in HORIZONS],
                     *[None if pd.isna(r[f'xs_spy_{h}']) else round(float(r[f'xs_spy_{h}']) * 100, 1) for h in ['1y']],
                     r['price_status']])
    return rows


# --- Narrative -------------------------------------------------------------------
def summary_bullets(head, es, alphas, miss, T):
    """Plain-English findings, computed from the result tables."""
    b = []
    sig = head[head['Significant'] == '✓']
    lo, hi = head['FF5 alpha (ann.)'].min(), head['FF5 alpha (ann.)'].max()
    capm = alphas[(alphas['holding'] == '1y') & (alphas['model'] == 'CAPM') & alphas['screen'].isin(SCREENS)]
    al = head.set_index('screen')['FF5 alpha (ann.)']
    lead = ('On paper, all of this works.' if len(sig) == len(head) else
            'On paper, some of this works.' if len(sig) else 'On paper, none of this clearly works.')
    tighter = ('The simple ≥3-insider cluster screen does at least as well as the stricter ones; requiring 5 or 10 insiders '
               'does not add return.' if al['cluster3'] >= max(al['cluster5'], al['cluster10']) else
               'Requiring more insiders in a cluster raises alpha.')
    b.append(f'<strong>{lead}</strong> {len(sig)} of {len(head)} screens earn statistically significant '
             f'risk-adjusted returns. Fama-French 5-factor + momentum alpha in the calendar-time portfolio ranges from '
             f'{pct(lo)} to {pct(hi)} a year (CAPM alpha {pct(capm["alpha_ann"].min())} to {pct(capm["alpha_ann"].max())}). '
             f'{tighter}')
    y1 = es[(es['signal_year'] == 'All') & (es['horizon'] == '1y') & es['screen'].isin(SCREENS)]
    b.append(f'<strong>The typical trade loses to the index; a few big winners carry the average.</strong> The average 1-year '
             f'excess return vs SPY is {pct(y1["mean_xs_spy"].min())} to {pct(y1["mean_xs_spy"].max())}, but the median is '
             f'{pct(y1["median_xs_spy"].min())} to {pct(y1["median_xs_spy"].max())} and only {pct(y1["hit_rate_vs_spy"].min(), 0)}–'
             f'{pct(y1["hit_rate_vs_spy"].max(), 0)} of signals beat SPY over a year. You need to hold many names to catch the outliers.')
    era = T['robust_era']
    e = era[era['era'].isin(['pre-2010', '2020+'])].pivot(index='screen', columns='era', values='ff5_alpha_ann').reindex(list(SCREENS))
    t20 = era[era['era'] == '2020+'].set_index('screen')['t_ff5_alpha'].reindex(list(SCREENS))
    weak = [SCREENS[k][0] for k, t in t20.items() if not abs(t) >= 1.96]
    b.append(f'<strong>The edge has shrunk.</strong> FF5 alpha fell from {pct(e["pre-2010"].min())}–{pct(e["pre-2010"].max())} '
             f'a year before 2010 to {pct(e["2020+"].min())}–{pct(e["2020+"].max())} since 2020'
             + (f', where it is no longer statistically significant for {html.escape(", ".join(weak))}.' if weak else '.'))
    liq = T['robust_liquidity']
    lp = liq.pivot(index='screen', columns='liquidity', values='ff5_alpha_ann')
    filt = [c for c in lp.columns if c != 'no filter'][0]
    cut = 1 - (lp[filt] / lp['no filter'])
    dl = T['robust_entry_delay']
    d3 = dl.pivot(index='screen', columns='entry', values='mean_xs_spy_3m')
    late = [c for c in d3.columns if c != 'next open'][0]
    b.append(f'<strong>Much of it comes from penny stocks and speed.</strong> Requiring a share price of at least $2 at entry '
             f'cuts alpha by {pct(cut.min(), 0)}–{pct(cut.max(), 0)}, and these are small, thinly traded stocks where daily '
             f'rebalancing and the 0.5% cost assumption flatter real-world results. Entering two days later instead of at the '
             f'next open costs about {pct((d3["next open"] - d3[late]).mean())} of 3-month excess return.')
    rp = RESULTS / 'realistic_trading.csv'
    if rp.exists():
        r = pd.read_csv(rp)
        r = r[(r['screen'] == 'cluster3') & (r['hold'] == '1y')].set_index('step')
        first, real = r.index[0], [i for i in r.index if i.startswith('4.')][0]
        bh = [i for i in r.index if i.startswith('3.')][0]
        sp = [i for i in r.index if i.startswith('2.')][0]
        b.append(f'<strong>Under realistic trading, most of the edge disappears.</strong> For clusters ≥3 held a year, alpha '
                 f'(after a micro-cap factor) falls from {pct(r.loc[first, "FF5+UMD+IWC alpha"])} as reported to '
                 f'{pct(r.loc[real, "FF5+UMD+IWC alpha"])} (t = {num(r.loc[real, "FF5+UMD+IWC t"])}) with estimated spreads, '
                 f'buy-and-hold weights and next-close entry, and to {pct(r.loc[real, "test alpha"])} in 2016–2026 '
                 f'(t = {num(r.loc[real, "test t"])}). The biggest single drop is from daily rebalancing to buy-and-hold '
                 f'weights (−{pct(r.loc[sp, "FF5+UMD+IWC alpha"] - r.loc[bh, "FF5+UMD+IWC alpha"])} a year), not from costs '
                 f'(−{pct(r.loc[first, "FF5+UMD+IWC alpha"] - r.loc[sp, "FF5+UMD+IWC alpha"])}). See <a href="#realistic">Realistic trading</a> and '
                 f'<a href="#filters">Filter tests</a>.')
    mx = miss[miss['signal_year'] == 'All']
    early = miss[miss['signal_year'].isin([str(y) for y in range(2006, 2012)])]
    early_cov = 1 - (early['signals'] * early['pct_missing']).sum() / early['signals'].sum()
    b.append(f'<strong>How confident to be: moderately, and expect less in practice.</strong> Only '
             f'{pct(1 - mx["pct_missing"].max(), 0)}–{pct(1 - mx["pct_missing"].min(), 0)} of signals could be priced (Yahoo lacks '
             f'most delisted companies), falling to {pct(early_cov, 0)} for 2006–2011, the years with the strongest results. '
             f'Delisted stocks are disproportionately failures, so these numbers almost certainly <strong>overstate</strong> '
             f'returns. Re-running Phase 3 with a source that includes delisted stocks is the single most important next step.')
    return b


HOW_TO_READ = """
<ul>
<li><strong>Return</strong>: the stock's change from the entry price (next trading day's open after the filing) to the close a set number of trading days later, including dividends.</li>
<li><strong>Excess return</strong>: the stock's return minus what SPY (or IWM) returned over exactly the same days. Positive means the signal beat simply buying the index.</li>
<li><strong>Hit rate</strong>: the share of signals with a positive return (or, "vs SPY", that beat SPY). Above 50% means more winners than losers.</li>
<li><strong>Sharpe ratio</strong>: return per unit of volatility, above the risk-free rate, per year. Around 0.5 is typical of the stock market; above 1 is strong.</li>
<li><strong>Max drawdown</strong>: the worst peak-to-trough fall of the portfolio. It shows how painful the strategy was to hold.</li>
<li><strong>Alpha</strong>: return left over after accounting for exposure to the market (CAPM) or to the market plus size, value, profitability, investment and momentum factors (FF5+UMD). It is the part not explained by known risk premia. ✓ means it is statistically significant (p &lt; 0.05); ✗ means it could plausibly be luck.</li>
<li><strong>t-stat</strong>: how many standard errors an estimate is from zero. Roughly, |t| above 2 is significant. Event-study t-stats cluster standard errors by signal month, because signals in the same month move together.</li>
</ul>
"""

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7;--border:rgba(11,11,11,0.10)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:28px;margin:8px 0 4px}h2{font-size:22px;margin:48px 0 8px;padding-top:12px;border-top:1px solid var(--grid)}h3{font-size:17px;margin:28px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 20px;margin:16px 0}
nav a{margin-right:14px;color:#256abf;text-decoration:none;font-size:14px}nav{margin:8px 0 16px;line-height:2}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;background:var(--surface)}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
td:first-child,th:first-child{text-align:left}.headline td:nth-child(1){font-weight:600}
figure.chart{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px;margin:16px 0}
figcaption{color:var(--ink2);font-size:13px;padding:4px 8px 6px}
.warn{border-left:4px solid #ec835a;background:#fff7f2}.note{color:var(--ink2);font-size:13px}
.sigtools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}.sigtools input{padding:6px 10px;border:1px solid var(--grid);border-radius:6px;min-width:240px}
.sigtable th{cursor:pointer;user-select:none}.sigtable th:hover{color:var(--ink)}.pager button{padding:4px 10px;border:1px solid var(--grid);background:var(--surface);border-radius:6px;cursor:pointer}
.sig{font-weight:700}.yes{color:#006300}.no{color:#d03b3b}
"""

TABLE_JS = r"""
const SIGCOLS=['Signal date','Ticker','Company','Insiders','Total $ bought','1w','1m','3m','6m','1y','1y vs SPY','Price status'];
function fmtCell(i,v){if(v===null||v===undefined||v==='')return i>=5&&i<=10?'–':'';if(i===4)return '$'+Math.round(v).toLocaleString();if(i>=5&&i<=10)return v.toFixed(1)+'%';return String(v);}
function bg(i,v){if(i<5||i>10||v===null)return '';const a=Math.min(Math.abs(v)/40,1)*0.55+0.08;return v>0?`background:rgba(0,131,0,${a.toFixed(2)})`:`background:rgba(208,59,59,${a.toFixed(2)})`;}
function makeTable(id,rows){const el=document.getElementById(id);let q='',sortI=0,asc=false,page=0;const per=50;
 el.innerHTML=`<div class="sigtools"><input type="search" placeholder="Search ticker, company or date…"><span class="count note"></span><span class="pager"><button data-d="-1">‹ Prev</button> <span class="pg note"></span> <button data-d="1">Next ›</button></span></div><div class="tablewrap"><table class="sigtable"><thead><tr>${SIGCOLS.map((c,i)=>`<th data-i="${i}">${c}</th>`).join('')}</tr></thead><tbody></tbody></table></div>`;
 const tb=el.querySelector('tbody');
 function render(){let r=rows.filter(x=>!q||(x[0]+' '+x[1]+' '+x[2]).toLowerCase().includes(q));
  r.sort((a,b)=>{const x=a[sortI],y=b[sortI];if(x===y)return 0;if(x===null)return 1;if(y===null)return -1;return (x<y?-1:1)*(asc?1:-1);});
  const pages=Math.max(1,Math.ceil(r.length/per));page=Math.min(page,pages-1);
  tb.innerHTML=r.slice(page*per,page*per+per).map(x=>'<tr>'+x.map((v,i)=>`<td style="${bg(i,v)}">${fmtCell(i,v)}</td>`).join('')+'</tr>').join('');
  el.querySelector('.count').textContent=r.length.toLocaleString()+' signals';el.querySelector('.pg').textContent=`page ${page+1} of ${pages}`;
  el.querySelectorAll('th').forEach(th=>{const i=+th.dataset.i;th.textContent=SIGCOLS[i]+(i===sortI?(asc?' ▲':' ▼'):'');});}
 el.querySelector('input').addEventListener('input',e=>{q=e.target.value.toLowerCase();page=0;render();});
 el.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const i=+th.dataset.i;asc=(i===sortI)?!asc:false;sortI=i;render();}));
 el.querySelectorAll('.pager button').forEach(b=>b.addEventListener('click',()=>{page=Math.max(0,page+ +b.dataset.d);render();}));
 render();}
"""


def write_excel(tables):
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
    path = RESULTS / 'results.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = 'B2'
            for i, c in enumerate(df.columns, 1):
                letter = get_column_letter(i)
                ws.column_dimensions[letter].width = min(max(len(str(c)) + 2, 10), 40)
                lc = str(c).lower()
                is_pct = any(k in lc for k in ['mean', 'median', 'hit', 'return', 'cagr', 'vol', 'drawdown', 'alpha_ann',
                                                'ci95', 'pct', 'xs_', 'avg', 'priced %', 'excess', 'alpha (ann.)'])
                if is_pct and pd.api.types.is_numeric_dtype(df[c]) and len(df):
                    for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        for cell in row:
                            cell.number_format = '0.0%'
                    if any(k in lc for k in ['xs', 'excess', 'alpha', 'mean', 'return', 'cagr', 'drawdown']):
                        ws.conditional_formatting.add(f'{letter}2:{letter}{len(df) + 1}', ColorScaleRule(
                            start_type='num', start_value=-0.3, start_color='F8B4B4', mid_type='num', mid_value=0,
                            mid_color='F7F7F5', end_type='num', end_value=0.3, end_color='9FD89F'))
                elif str(c).startswith('t_') or lc.endswith(' t'):
                    for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        for cell in row:
                            cell.number_format = '0.00'
                    ws.conditional_formatting.add(f'{letter}2:{letter}{len(df) + 1}', ColorScaleRule(
                        start_type='num', start_value=-3, start_color='F8B4B4', mid_type='num', mid_value=0,
                        mid_color='F7F7F5', end_type='num', end_value=3, end_color='9FD89F'))
    return path


def build(cfg=None):
    cfg = cfg or load_config()
    bt = cfg['backtest']
    os.environ.setdefault('BROWSER_PATH', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    T = {n: pd.read_csv(RESULTS / f'{n}.csv') for n in
         ['headline', 'event_study', 'missing_prices', 'portfolio_stats', 'factor_alphas', 'robust_era',
          'robust_dedupe', 'robust_entry_delay', 'robust_liquidity']}
    T['event_study']['signal_year'] = T['event_study']['signal_year'].astype(str)
    T['missing_prices']['signal_year'] = T['missing_prices']['signal_year'].astype(str)
    trades = pd.read_parquet(DATA / 'trade_returns.parquet')
    trades['signal_date'] = pd.to_datetime(trades['signal_date'])
    daily = pd.read_parquet(DATA / 'portfolio_daily_returns.parquet').set_index('date')
    daily.index = pd.to_datetime(daily.index)
    scrape = json.loads((DATA / 'scrape_summary.json').read_text())
    meta = json.loads((DATA / 'signals_meta.json').read_text())
    counts = pd.read_csv(DATA / 'signal_counts_by_year.csv')
    pclean = json.loads((DATA / 'price_cleaning.json').read_text())
    es, alphas, miss, head = T['event_study'], T['factor_alphas'], T['missing_prices'], T['headline']

    ch = Charts()
    parts = []
    nav = ''.join(f'<a href="#{k}">{html.escape(v[0])}</a>' for k, v in SCREENS.items())
    parts.append(f'<h1>Insider-buying signals: backtest report</h1>'
                 f'<p class="sub">SEC Form 4 open-market purchases, filings {scrape["earliest_filing_date"]} to '
                 f'{scrape["latest_filing_date"]} ({scrape["rows"]:,} purchase filings). Signal date = filing date. '
                 f'Entry = next trading day\'s open. Prices: Yahoo Finance, adjusted for splits and dividends.</p>'
                 f'<nav><a href="#summary">Summary</a>{nav}<a href="#compare">Comparison</a><a href="#robust">Robustness</a>'
                 f'<a href="#realistic">Realistic trading</a><a href="#filters">Filter tests</a>'
                 f'<a href="#data">Data &amp; coverage</a><a href="#method">Method</a></nav>')

    # Summary
    h = head.copy()
    h['FF5 alpha (ann.)'] = [f'{pct(a)} {s}' for a, s in zip(h['FF5 alpha (ann.)'], h['Significant'])]
    show = h[['Screen', 'Signals', 'Priced %', 'Avg 1y return', 'Avg 1y excess vs SPY', '1y hit rate', 'Sharpe',
              'Max drawdown', 'FF5 alpha (ann.)', 'FF5 alpha t']]
    bullets = summary_bullets(head, es, alphas, miss, T)
    parts.append('<h2 id="summary">Summary</h2>')
    parts.append(html_table(show, {'Priced %': pct, 'Avg 1y return': pct, 'Avg 1y excess vs SPY': pct, '1y hit rate': pct,
                                   'Sharpe': num, 'Max drawdown': pct, 'FF5 alpha t': num, 'Signals': lambda v: f'{v:,}'},
                            color_cols=['Avg 1y excess vs SPY'], cls='headline'))
    parts.append('<p class="note">Signals counted after the 90-day de-duplication. Returns, hit rate and excess use priced signals '
                 'whose 1-year horizon has fully elapsed. Sharpe, drawdown and alpha come from the calendar-time portfolio '
                 '(equal weight in every stock within 1 year of its entry). Alpha is annualized; ✓ = significant at 5% '
                 '(Newey-West t-stat).</p>')
    parts.append('<div class="card"><ul>' + ''.join(f'<li>{x}</li>' for x in bullets) + '</ul></div>')
    parts.append('<details class="card"><summary><strong>How to read this</strong></summary>' + HOW_TO_READ + '</details>')
    high_missing = miss[(miss['signal_year'] == 'All') & (miss['pct_missing'] > 0.05)]
    if len(high_missing):
        parts.append('<div class="card warn"><strong>Survivorship warning.</strong> More than 5% of signals have no usable '
                     'price for ' + ', '.join(html.escape(SCREENS[s][0]) for s in high_missing['screen']) +
                     '. Yahoo Finance drops most delisted companies, so these are missing, not dropped, and the remaining sample '
                     'is tilted toward survivors. For a cleaner test, re-run Phase 3 with a source that includes delisted stocks '
                     '(Norgate Data, Tiingo, EODHD or CRSP). See <a href="#data">Data &amp; coverage</a>.</div>')

    # Per screen
    for s, (label, kind, param) in SCREENS.items():
        rule = (f'At least {param} distinct insiders and at least ${bt["cluster_min_value"]:,} bought within a rolling '
                f'{bt["cluster_window_days"]}-day filing window; funds and trusts excluded.') if kind == 'cluster' else \
               (f'A single purchase filing of at least ${param:,} by a CEO, CFO or Chairman of the Board.')
        hs = head[head['screen'] == s].iloc[0]
        parts.append(f'<h2 id="{s}">{html.escape(label)}</h2><p class="sub">{rule} {hs["Signals"]:,} signals '
                     f'({pct(hs["Priced %"])} with prices).</p>')
        parts.append(ch.add(f'{s}_heatmap', heatmap(es, s, bt['min_signals_heatmap']),
                            'Green = signals that year beat SPY on average; red = they lagged. Grey cells have fewer than '
                            f'{bt["min_signals_heatmap"]} signals, too few to trust. Look for consistent colour down a column.'))
        parts.append(ch.add(f'{s}_horizon_bars', horizon_bars(es, s),
                            'Each group compares the average signal return with SPY and IWM over the same dates. '
                            'The gap between the coloured bar and the grey bars is the average edge.'))
        parts.append(ch.add(f'{s}_equity', equity_drawdown(daily, s),
                            'Top: what $10,000 would have grown to holding every signal for a year, vs the indexes. '
                            'Bottom: how far each fell from its previous peak.'))
        parts.append(ch.add(f'{s}_histogram', histogram(trades, s),
                            'When the mean sits well right of the median, a few large winners drive the average. '
                            'Extreme 0.5% tails are clipped into the edge bins for readability.'))
        parts.append(f'<h3>Every signal</h3><p class="note">Click a column header to sort; type to search. Returns are from '
                     f'entry; blank horizons have not elapsed yet. Rows without prices are listed with their status.</p>'
                     f'<div id="tbl_{s}"></div>')

    # Comparison
    parts.append('<h2 id="compare">Comparison across screens</h2>')
    parts.append(ch.add('compare_equity', all_equity(daily),
                        'Lines above SPY and IWM beat the index over the full period. Click legend entries to hide lines.'))
    parts.append(ch.add('compare_alpha', alpha_bars(alphas),
                        'Bars whose error bars cross zero are not statistically distinguishable from no edge.'))
    parts.append(ch.add('compare_excess_by_horizon', excess_lines(es),
                        'Shows when the edge appears and whether it keeps growing or fades as the holding horizon lengthens.'))
    parts.append(ch.add('compare_alpha_by_holding', decay_bars(alphas),
                        'Shorter holding periods concentrate on the days right after the filing; compare bar heights to see where the edge sits.'))
    ps = T['portfolio_stats']
    ps_show = ps.copy()
    ps_show['Screen'] = ps_show['screen'].map(lambda k: SCREENS[k][0] if k in SCREENS else k)
    ps_show = ps_show[['Screen', 'holding', 'start', 'cagr', 'volatility', 'sharpe', 'sortino', 'max_drawdown', 'beta_spy',
                       'avg_positions']]
    ps_show.columns = ['Screen', 'Hold', 'Start', 'CAGR', 'Volatility', 'Sharpe', 'Sortino', 'Max drawdown', 'Beta to SPY',
                       'Avg positions']
    parts.append('<h3>Calendar-time portfolio statistics</h3>')
    parts.append(html_table(ps_show, {'CAGR': pct, 'Volatility': pct, 'Sharpe': num, 'Sortino': num, 'Max drawdown': pct,
                                      'Beta to SPY': num, 'Avg positions': lambda v: num(v, 0)}, color_cols=['CAGR']))
    al = alphas.copy()
    al['Screen'] = al['screen'].map(lambda k: SCREENS[k][0] if k in SCREENS else k)
    al = al[['Screen', 'holding', 'model', 'alpha_ann', 't_alpha', 'ci95_low', 'ci95_high', 'beta_Mkt-RF', 'r2', 'n_days']]
    al.columns = ['Screen', 'Hold', 'Model', 'Alpha (ann.)', 't-stat', '95% CI low', '95% CI high', 'Market beta', 'R²', 'Days']
    parts.append('<h3>Factor regressions</h3>')
    parts.append(html_table(al, {'Alpha (ann.)': pct, 't-stat': num, '95% CI low': pct, '95% CI high': pct,
                                 'Market beta': num, 'R²': num, 'Days': lambda v: f'{int(v):,}' if pd.notna(v) else '–'},
                            color_cols=['Alpha (ann.)']))

    # Robustness
    parts.append('<h2 id="robust">Robustness</h2><p class="sub">Excess returns are vs SPY. Alpha is FF5+momentum from the '
                 '1-year-hold calendar-time portfolio restricted to each subset.</p>')
    rfmt = {'mean_xs_spy_3m': pct, 'mean_xs_spy_1y': pct, 'hit_rate_vs_spy_3m': pct, 'hit_rate_vs_spy_1y': pct,
            'ff5_alpha_ann': pct, 't_ff5_alpha': num, 'signals_priced': lambda v: f'{int(v):,}'}
    names = {'mean_xs_spy_3m': '3m excess', 'mean_xs_spy_1y': '1y excess', 'hit_rate_vs_spy_3m': '3m hit vs SPY',
             'hit_rate_vs_spy_1y': '1y hit vs SPY', 'ff5_alpha_ann': 'FF5 alpha', 't_ff5_alpha': 'alpha t',
             'signals_priced': 'Priced signals', 'dedupe_days': 'Dedupe window (days)', 'entry': 'Entry',
             'era': 'Era', 'liquidity': 'Liquidity filter'}
    for key, title in [('robust_era', 'By era'), ('robust_dedupe', 'De-duplication window'),
                       ('robust_entry_delay', 'Entry timing'), ('robust_liquidity', 'Liquidity filter (price at entry)')]:
        d = T[key].copy()
        d.insert(0, 'Screen', d.pop('screen').map(lambda k: SCREENS[k][0]))
        f = {names.get(c, c): fn for c, fn in rfmt.items()}
        d = d.rename(columns=names)
        parts.append(f'<h3>{title}</h3>')
        parts.append(html_table(d, f, color_cols=['3m excess', '1y excess', 'FF5 alpha']))
    cost = es[(es['signal_year'] == 'All')].copy()
    cost['Screen'] = cost['screen'].map(lambda k: SCREENS[k][0])
    cost = cost[['Screen', 'horizon', 'count', 'mean', 'mean_net', 'mean_xs_spy', 'mean_net_xs_spy']]
    cost.columns = ['Screen', 'Horizon', 'Signals', 'Mean return', f'After {bt["round_trip_cost"]:.1%} costs',
                    'Excess vs SPY', 'Excess after costs']
    parts.append('<h3>Effect of round-trip trading costs</h3>')
    parts.append(html_table(cost, {'Mean return': pct, f'After {bt["round_trip_cost"]:.1%} costs': pct,
                                   'Excess vs SPY': pct, 'Excess after costs': pct, 'Signals': lambda v: f'{int(v):,}'},
                            color_cols=['Excess after costs']))
    esh = es.copy()
    esh['Screen'] = esh['screen'].map(lambda k: SCREENS[k][0])
    esh = esh[esh['signal_year'] == 'All'][['Screen', 'horizon', 'count', 'mean', 'median', 'hit_rate', 'hit_rate_vs_spy',
                                            'mean_xs_spy', 'median_xs_spy', 'mean_xs_iwm', 't_xs_spy', 't_xs_iwm']]
    esh.columns = ['Screen', 'Horizon', 'Signals', 'Mean', 'Median', 'Hit rate', 'Hit vs SPY', 'Excess vs SPY',
                   'Median excess', 'Excess vs IWM', 't (vs SPY)', 't (vs IWM)']
    parts.append('<h3>Event study, all years</h3><p class="note">t-stats use standard errors clustered by signal month. '
                 'Per-year rows are in results.xlsx (sheet event_study).</p>')
    parts.append(html_table(esh, {c: pct for c in ['Mean', 'Median', 'Hit rate', 'Hit vs SPY', 'Excess vs SPY',
                                                   'Median excess', 'Excess vs IWM']} |
                            {'t (vs SPY)': num, 't (vs IWM)': num, 'Signals': lambda v: f'{int(v):,}'},
                            color_cols=['Excess vs SPY', 'Excess vs IWM']))

    # Realistic trading and filter tests (Phase 7), if they have been run
    if (RESULTS / 'realistic_trading.csv').exists():
        realism = pd.read_csv(RESULTS / 'realistic_trading.csv')
        tests = pd.read_csv(RESULTS / 'filter_tests.csv')
        combo = pd.read_csv(RESULTS / 'combined_filters.csv')
        cap = pd.read_csv(RESULTS / 'capacity.csv')
        per = pd.read_csv(RESULTS / 'realistic_per_trade.csv')
        T.update({'realistic_trading': realism, 'filter_tests': tests, 'combined_filters': combo, 'capacity': cap,
                  'realistic_per_trade': per})
        parts.append('<h2 id="realistic">Does it survive realistic trading?</h2><p class="sub">The headline numbers assume '
                     'a flat 0.5% round-trip cost, daily rebalancing and entry at the open. Here each assumption is replaced '
                     'step by step. Alpha is from a regression on the Fama-French 5 factors, momentum <em>and</em> the iShares '
                     'Micro-Cap ETF (IWC), so the plain micro-cap premium does not count as skill.</p>')
        parts.append(ch.add('realism_steps', realism_chart(realism),
                            'If the edge were an artefact of cheap-stock trading, the bars would collapse toward zero as '
                            'costs, buy-and-hold weights and next-close entry are added.'))
        rr = realism[realism['hold'] == '3m'].copy()
        rr['Screen'] = rr['screen'].map(lambda k: SCREENS[k][0])
        rr = rr[['Screen', 'step', 'signals', 'median_spread', 'FF5+UMD alpha', 'FF5+UMD+IWC alpha', 'FF5+UMD+IWC t',
                 'test alpha', 'test t', '2020+ alpha', '2020+ t']]
        rr.columns = ['Screen', 'Assumptions', 'Signals', 'Median est. spread', 'FF5+UMD alpha', '+ micro-cap factor',
                      't', '2016–26 alpha', '2016–26 t', '2020+ alpha', '2020+ t']
        parts.append('<h3>Step by step (3-month hold)</h3><p class="note">Spread = Corwin–Schultz estimate from daily '
                     'highs and lows over the 20 days before entry, plus 0.2% for commission and impact, charged half on '
                     'entry and half on exit. This estimator overstates spreads for liquid stocks (it gives Apple about '
                     '0.4%), so costs here are conservative. Other holding periods are in results.xlsx.</p>')
        parts.append(html_table(rr, {'Median est. spread': pct, 'FF5+UMD alpha': pct, '+ micro-cap factor': pct, 't': num,
                                     '2016–26 alpha': pct, '2016–26 t': num, '2020+ alpha': pct, '2020+ t': num,
                                     'Signals': lambda v: f'{int(v):,}'},
                                color_cols=['+ micro-cap factor', '2016–26 alpha', '2020+ alpha']))
        c2 = cap.copy()
        c2['Screen'] = c2['screen'].map(lambda k: SCREENS[k][0])
        c2 = c2[['Screen', 'median_price', 'median_adv', 'median_spread', 'median_cost_rt', 'median_market_cap',
                 'share_price_under_2', 'share_adv_under_100k', 'median $ per trade at 10% of volume']]
        c2.columns = ['Screen', 'Median price', 'Median $ volume/day', 'Median est. spread', 'Median round-trip cost',
                      'Median market cap (2011+)', 'Under $2', 'Under $100K/day', '$ per trade at 10% of a day']
        parts.append('<h3>How tradeable are these stocks?</h3>')
        parts.append(html_table(c2, {'Median price': lambda v: f'${v:,.2f}', 'Median $ volume/day': money,
                                     'Median est. spread': pct, 'Median round-trip cost': pct,
                                     'Median market cap (2011+)': money, 'Under $2': pct, 'Under $100K/day': pct,
                                     '$ per trade at 10% of a day': money}))
        pt = per.copy()
        pt['Screen'] = pt['screen'].map(lambda k: SCREENS[k][0])
        pt = pt[['Screen', 'hold', 'benchmark', 'trades', 'mean net return', 'mean excess', 'median excess', 'hit rate vs benchmark']]
        pt.columns = ['Screen', 'Hold', 'vs', 'Trades', 'Mean net return', 'Mean excess', 'Median excess', 'Hit rate']
        parts.append('<h3>Per trade, realistic (next-close entry, estimated costs)</h3>')
        parts.append(html_table(pt, {'Mean net return': pct, 'Mean excess': pct, 'Median excess': pct, 'Hit rate': pct,
                                     'Trades': lambda v: f'{int(v):,}'}, color_cols=['Mean excess', 'Median excess']))

        n_tests = int((tests['filter'] != 'No filter (realistic)').sum())
        parts.append(f'<h2 id="filters">Filter tests</h2><p class="sub">{n_tests} filter tests on the two base screens '
                     f'(clusters ≥3 and CEO/CFO/COB ≥$200K) at 1-month, 3-month and 1-year holds, all under the realistic '
                     f'setup above. A filter "holds" if it improved alpha in 2006–2015 <em>and</em> in 2016–2026, and the '
                     f'2016–2026 alpha is significant. With this many tests some will look good by chance, so the Holm '
                     f'column corrects the p-value for the number of tests.</p>')
        parts.append(ch.add('filter_scatter', filter_scatter(tests),
                            'Points in the top-right helped in both periods. Points scattered evenly around the centre would '
                            'mean filter results are mostly noise.'))
        parts.append(ch.add('filter_survivors', survivors_chart(tests),
                            'Only filters that held up out of sample are shown; bars whose interval crosses zero are not reliable.'))
        if len(combo):
            cb = combo.copy()
            cb['Screen'] = cb['screen'].map(lambda k: SCREENS[k][0])
            cb = cb[['Screen', 'hold', 'filters (chosen on 2006–2015)', 'train_signals', 'test_signals', 'train alpha',
                     'test alpha', 'test t', '2020+ alpha', '2020+ t']]
            cb.columns = ['Screen', 'Hold', 'Filters picked using 2006–2015 only', 'Signals 2006–15', 'Signals 2016–26',
                          'Alpha 2006–15', 'Alpha 2016–26', 't 2016–26', 'Alpha 2020+', 't 2020+']
            parts.append('<h3>Best combination, chosen without looking at 2016–2026</h3><p class="note">Up to three filters were '
                         'added one at a time, each only if it raised 2006–2015 alpha. The 2016–2026 columns are the honest '
                         'out-of-sample test.</p>')
            parts.append(html_table(cb, {'Alpha 2006–15': pct, 'Alpha 2016–26': pct, 't 2016–26': num, 'Alpha 2020+': pct,
                                         't 2020+': num}, color_cols=['Alpha 2016–26', 'Alpha 2020+']))
        ft = tests.copy()
        ft['Screen'] = ft['screen'].map(lambda k: SCREENS[k][0])
        ft = ft.sort_values(['screen', 'hold', 'test vs baseline'], ascending=[True, True, False])
        ft = ft[['Screen', 'hold', 'group', 'filter', 'signals', 'train alpha', 'test alpha', 'test t', 'train vs baseline',
                 'test vs baseline', '2020+ alpha', 'holm p', 'holds out of sample']]
        ft['holds out of sample'] = ft['holds out of sample'].map({True: '✓', False: ''})
        ft.columns = ['Screen', 'Hold', 'Group', 'Filter', 'Signals', 'Alpha 2006–15', 'Alpha 2016–26', 't 2016–26',
                      'Δ vs none 2006–15', 'Δ vs none 2016–26', 'Alpha 2020+', 'Holm p (full period)', 'Holds']
        parts.append('<h3>Every filter</h3>')
        parts.append(html_table(ft, {'Alpha 2006–15': pct, 'Alpha 2016–26': pct, 't 2016–26': num, 'Δ vs none 2006–15': pct,
                                     'Δ vs none 2016–26': pct, 'Alpha 2020+': pct,
                                     'Holm p (full period)': lambda v: '–' if pd.isna(v) else f'{v:.3f}',
                                     'Signals': lambda v: f'{int(v):,}'},
                                color_cols=['Δ vs none 2006–15', 'Δ vs none 2016–26']))

    # Data & coverage
    parts.append('<h2 id="data">Data &amp; coverage</h2>')
    rp = pd.DataFrame(list(scrape['rows_per_year'].items()), columns=['Year', 'Purchase filings'])
    parts.append(f'<div class="card"><p><strong>Source:</strong> {html.escape(scrape["source"])}, quarters '
                 f'{scrape["quarters_available"][0]}–{scrape["quarters_available"][1]}. Failed quarters: '
                 f'{len(scrape["failed_quarters"]) or "none"}. Empty months: {len(scrape["empty_months"]) or "none"}. '
                 f'Filings without a ticker: {scrape["rows_without_ticker"]:,}.</p>'
                 f'<p><strong>Cleaning:</strong> {meta["prices_corrected"]:,} purchase prices more than 10× away from the same '
                 f'company\'s other purchases that month were repaired; {meta["values_excluded"]:,} filings worth over $1B '
                 f'(almost all typos) were excluded from dollar totals.</p>'
                 f'<p><strong>Fund/trust filter:</strong> {meta["fund_companies_dropped"]:,} companies dropped '
                 f'({meta["fund_dropped_by_sic"]:,} by SIC {bt["fund_sic_range"][0]}–{bt["fund_sic_range"][1]}, '
                 f'{meta["fund_dropped_by_name"]:,} by name), removing {meta["fund_raw_signals_dropped"]:,} raw cluster signals. '
                 f'Note SIC 6798 (REITs) falls inside this range, so REITs are excluded too.</p>'
                 f'<p><strong>Prices:</strong> Yahoo had data for {pclean["tickers_found"]:,} of {pclean["tickers_requested"]:,} '
                 f'tickers tried (filed tickers plus the companies\' current SEC tickers). Bad prints removed: '
                 f'{pclean["spikes_fixed"]:,} one-day V-shaped spikes (≥5× and back) in {pclean["tickers_with_spikes"]:,} tickers, '
                 f'{pclean["zero_days_dropped"]:,} days with a zero price, {pclean["opens_fixed"]:,} implausible opens.</p></div>')
    parts.append('<h3>Purchase filings per year</h3>' + html_table(rp, {'Purchase filings': lambda v: f'{int(v):,}'}))
    c2 = counts.rename(columns={k: v[0] for k, v in SCREENS.items()}).rename(columns={'signal_year': 'Signal year'})
    parts.append('<h3>Signals per screen per year</h3>' + html_table(c2, {}))
    mp = miss.copy()
    mp['Screen'] = mp['screen'].map(lambda k: SCREENS[k][0])
    mp = mp[['Screen', 'signal_year', 'signals', 'priced', 'priced_via_new_ticker', 'pct_missing', 'no_ticker',
             'ticker_not_found', 'no_price_at_entry', 'price_mismatch', 'not_yet_entered']]
    mp.columns = ['Screen', 'Year', 'Signals', 'Priced', 'Priced via new ticker', '% missing', 'No ticker', 'Not on Yahoo',
                  'No price at entry', 'Price mismatch', 'Not yet entered']
    parts.append('<h3>Missing prices per screen per year</h3><p class="note">"Not on Yahoo" is mostly delisted companies. '
                 '"Price mismatch" means Yahoo\'s price at entry was more than 2× away from the insiders\' purchase price, '
                 'which usually means the ticker now belongs to a different company; those are treated as missing rather '
                 'than trusted.</p>')
    parts.append(html_table(mp, {'% missing': pct}))

    parts.append('<h2 id="method">Method and assumptions</h2><div class="card">' + METHOD.format(**bt) + '</div>')

    from plotly.offline import get_plotlyjs
    plotly_js = f'<script type="text/javascript">{get_plotlyjs()}</script>'
    tables_js = ''.join(f'makeTable("tbl_{s}",{json.dumps(signal_table_json(trades, s))});' for s in SCREENS)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Insider Buying Backtest</title><style>{CSS}</style>{plotly_js}</head><body><main>{"".join(parts)}</main>'
            f'<script>{TABLE_JS}{tables_js}</script></body></html>')
    out = ROOT / 'report.html'
    out.write_text(page, encoding='utf-8')
    LOG.info(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB); writing {len(ch.figs)} PNGs")
    ch.write_pngs()
    xl = write_excel({**T, 'signal_counts_by_year': counts})
    LOG.info(f"Wrote {xl}")
    return out


METHOD = """
<ul>
<li><strong>Data.</strong> SEC EDGAR Insider Transactions Data Sets (Form 4, originals only; amendments excluded to avoid double counting). Only transaction code P (open-market or private purchase) with acquired flag A. One row per filing, like OpenInsider: quantities and values summed across the filing's lines, value-weighted price, earliest trade date. The first listed reporting owner is the insider; joint filers (a person and their LLC) count once.</li>
<li><strong>Signal date</strong> is the SEC filing date, never the trade date. Entry is the open of the trading day after the filing date (filings after 5:30pm ET are already dated the next business day by the SEC). Where Yahoo has no open, the close is used and flagged.</li>
<li><strong>Clusters</strong>: rolling {cluster_window_days}-day window of filing dates per company (by SEC CIK, so ticker changes do not split a cluster). The signal fires on the first day the window holds the required distinct insiders and ≥ ${cluster_min_value:,} total; a new cluster can fire only after the condition lapses.</li>
<li><strong>Top executives</strong>: titles matched with a tested regular expression for CEO / chief executive, CFO / chief financial / principal financial officer, and Chairman / COB (not vice or deputy chairs, not "former"). Single filings only.</li>
<li><strong>De-duplication</strong>: after a company signals on a screen, further signals from it on that screen are ignored for {dedupe_days} days.</li>
<li><strong>Returns</strong>: adjusted close at entry + N trading days ({horizons}) divided by adjusted open at entry. Benchmarks over identical dates. A horizon counts only if it has fully elapsed. If a stock stops trading before the horizon, its last close is used and the trade is flagged as truncated. Costs: {round_trip_cost:.1%} round trip.</li>
<li><strong>Calendar-time portfolios</strong>: each day, equal weight in every stock still inside its holding period; days with no positions earn the risk-free rate. Alpha from OLS of daily excess returns on Mkt-RF (CAPM) and on Mkt-RF, SMB, HML, RMW, CMA and UMD (Kenneth French data library), Newey-West standard errors with 5 lags, annualized ×252.</li>
<li><strong>Ticker check</strong>: Yahoo's split-unadjusted price at entry must be within {price_mismatch_tolerance}× of the insiders' purchase price, otherwise the signal is treated as missing (likely a reused ticker).</li>
<li><strong>Known limitations</strong>: survivorship bias from Yahoo's missing delisted stocks; SEC bulk data begins in 2006 (OpenInsider has some 2003–2005 filings); a few legitimate purchases over $1B are excluded from dollar totals along with the typos.</li>
</ul>
"""


if __name__ == '__main__':
    build()
