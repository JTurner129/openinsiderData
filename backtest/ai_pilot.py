"""Phase 13: can an AI's judgment pick the winners among screened candidates? A hindsight-free pilot.

Rules were written down and committed before the model was downloaded or run.

Model: Meta Llama 3.1 8B Instruct (4-bit GGUF), whose training data ends in December 2023 (Meta model card),
run locally with llama.cpp, temperature 0. It cannot know what happened to any stock after that.

Candidates: every entry dated 2024-02-01 or later from
  (a) tradeable big-drop insider clusters (>= 3 insiders, >= $200K, stock down >= 50% in 6 months,
      >= $100K a day traded), and
  (b) S&P 500 officer/director purchases >= $100K (test L1), and
  (c) the DFV screen without the smart-money condition (test D4).
Input: a fact sheet built only from data public at the entry date (company name, sector code, market cap,
6- and 12-month price change, valuation, cash, debt, trailing earnings, revenue trend, short interest,
insider purchases with titles and amounts, recent 8-K item codes). Same prompt for every candidate.
Output: JSON with a 1-10 score, buy / pass, and three short reasons.

Measures (pre-registered): Spearman correlation between score and the 6-month return minus SPY (entries
with a complete 6 months); average 6-month return minus SPY for score >= 7 against score <= 4; and
"buy" against "pass". Small sample: treated as a pilot, not proof.

Amendment (committed before any model run, because the 8B model was far too slow on this machine's CPU):
use Llama 3.2 3B Instruct (4-bit GGUF; same December 2023 training cutoff per Meta's model card); keep every
candidate from (a); from (b) and (c) take a fixed random sample of 150 each (seed 20261015).
"""
import json
import re

import numpy as np
import pandas as pd

from .common import DATA, RESULTS, get_logger, load_config, save_table
from .compustat import wrds_dir

LOG = get_logger('ai_pilot')
MODEL = '/home/user/models/llama32-3b-q4.gguf'
START = pd.Timestamp('2024-02-01')
SEED = 20261015
OUT = DATA / 'ai_pilot_scores.parquet'

SYSTEM = ('You are a careful fundamental stock analyst. You judge whether a stock is likely to beat the '
          'S&P 500 over the next 6 months, using only the facts given. Reply with JSON only.')
ASK = ('Facts as of {date}:\n{facts}\n\nReturn JSON exactly like: {{"score": <1-10, 10 = very likely to beat the '
       'S&P 500 over 6 months>, "decision": "buy" or "pass", "reasons": ["...", "...", "..."]}}')


def candidates(cfg):
    rng = np.random.default_rng(SEED)
    F = pd.read_parquet(DATA / 'signal_features.parquet')
    F['signal_date'] = pd.to_datetime(F['signal_date'])
    a = F[(F['screen'] == 'cluster3') & F['price_found'] & (F['ret_prior_6m'] <= -0.5) & (F['adv20'] >= 1e5) &
          (F['signal_date'] >= START)]
    a = pd.DataFrame({'set': 'a big-drop cluster', 'price_ticker': a['price_ticker'], 'issuer_cik': a['issuer_cik'],
                      'date': a['signal_date']})
    L = pd.read_parquet(DATA / 'largecap_signals_any.parquet')
    L['signal_date'] = pd.to_datetime(L['signal_date'])
    L = L[L['sp500'] & (L['signal_date'] >= START)]
    L = L.iloc[rng.permutation(len(L))[:150]]
    b = pd.DataFrame({'set': 'b S&P 500 insider buy', 'price_ticker': L['price_ticker'], 'issuer_cik': L['issuer_cik'],
                      'date': L['signal_date']})
    D = pd.read_parquet(DATA / 'deep_value_entries.parquet')
    D['month_end'] = pd.to_datetime(D['month_end'])
    D = D[D['strategy'].str.startswith('D4') & (D['month_end'] >= START)]
    D = D.iloc[rng.permutation(len(D))[:150]]
    c = pd.DataFrame({'set': 'c DFV screen', 'price_ticker': D['sec'], 'issuer_cik': np.nan, 'date': D['month_end']})
    return pd.concat([a, b, c], ignore_index=True)


def fact_sheet(row, co, q, pc, si):
    from .prices import load_price
    sec = row['price_ticker']
    gv = sec.split('_')[1]
    d = row['date']
    px = load_price(sec)
    if px is None:
        return None
    h = px.loc[:d]
    if len(h) < 130:
        return None
    price = h['raw_close'].iloc[-1]
    shares = h['shares_out'].dropna().iloc[-1] if 'shares_out' in h and h['shares_out'].notna().any() else np.nan
    mcap = price * shares / 1e6
    r6 = h['adj_close'].iloc[-1] / h['adj_close'].iloc[-127] - 1
    r12 = h['adj_close'].iloc[-1] / h['adj_close'].iloc[-253] - 1 if len(h) > 253 else np.nan
    c = co.get(gv, {})
    lines = [f"Company: {c.get('conm', 'unknown')} (SIC industry code {c.get('sic', 'n/a')})",
             f"Share price ${price:,.2f}; market cap ${mcap:,.0f}M; price change: 6 months {r6:+.0%}, 12 months {r12:+.0%}"]
    g = q.get(gv)
    if g is not None:
        g = g[g['public'] <= d].tail(8)
        if len(g):
            last = g.iloc[-1]
            ttm = g.tail(4)
            rev, ni = ttm['saleq'].sum(), ttm['niq'].sum()
            prev = g.iloc[-8:-4]['saleq'].sum() if len(g) >= 8 else np.nan
            # pd.NA cannot be tested for truth; treat it like None (same result as before for every other value)
            nz = lambda x: 0 if x is None or x is pd.NA else (x or 0)
            debt = nz(last['dlttq']) + nz(last['dlcq'])
            lines.append(f"Trailing 12-month revenue ${rev:,.0f}M ({(rev / prev - 1):+.0%} vs prior year); net income ${ni:,.0f}M"
                         if prev and np.isfinite(prev) and prev > 0 else f"Trailing 12-month revenue ${rev:,.0f}M; net income ${ni:,.0f}M")
            lines.append(f"Cash ${last['cheq']:,.0f}M; debt ${debt:,.0f}M; book equity ${last['ceqq']:,.0f}M; "
                         f"current assets ${last['actq']:,.0f}M; total liabilities ${last['ltq']:,.0f}M")
            if np.isfinite(last.get('epspxq', np.nan)):
                lines.append(f"Latest quarterly EPS ${last['epspxq']:.2f} (reported {last['public'].date()})")
    s = si.get(sec)
    if s is not None:
        s = s[(s['datadate'] <= d) & (s['datadate'] > d - pd.Timedelta(days=45))]
        if len(s) and shares:
            lines.append(f"Short interest {s['shortint'].iloc[-1] / shares:.1%} of shares")
    p = pc.get(gv)
    if p is not None:
        p = p[(p['filing_date'] <= d) & (p['filing_date'] > d - pd.Timedelta(days=90))]
        if len(p):
            tops = p.groupby(['owner_name', 'Title'])['value_clean'].sum().sort_values(ascending=False).head(5)
            lines.append('Insider open-market purchases in the last 90 days: ' +
                         '; '.join(f"{n.title()} ({t}) ${v:,.0f}" for (n, t), v in tops.items()))
    return '\n'.join(lines)


def run(cfg=None):
    cfg = cfg or load_config()
    from llama_cpp import Llama
    from .compustat import security_map
    C = candidates(cfg)
    co = pd.read_parquet(wrds_dir(cfg) / 'compustat_company.parquet', columns=['gvkey', 'conm', 'sic'])
    co = {str(r.gvkey): {'conm': r.conm, 'sic': r.sic} for r in co.itertuples()}
    q = pd.read_parquet(wrds_dir(cfg) / 'compustat_fundq_full.parquet')
    for col in ('saleq', 'niq', 'cheq', 'dlttq', 'dlcq', 'ceqq', 'actq', 'ltq', 'epspxq'):
        q[col] = pd.to_numeric(q[col], errors='coerce')
    q['datadate'] = pd.to_datetime(q['datadate'])
    q['public'] = pd.to_datetime(q['rdq'], errors='coerce').fillna(q['datadate'] + pd.Timedelta(days=45))
    q = {str(k): g.sort_values('public') for k, g in q.groupby('gvkey')}
    pc = pd.read_parquet(DATA / 'purchases_clean.parquet', columns=['issuer_cik', 'filing_date', 'owner_name', 'Title', 'value_clean'])
    pc['filing_date'] = pd.to_datetime(pc['filing_date'])
    sm = security_map(cfg)[['cik_i', 'gvkey']].dropna().drop_duplicates()
    pc = pc.merge(sm, left_on='issuer_cik', right_on='cik_i')
    pc = {str(k): g for k, g in pc.groupby('gvkey')}
    si = pd.read_parquet(wrds_dir(cfg) / 'compustat_shortint.parquet')
    si['datadate'] = pd.to_datetime(si['datadate'])
    si['sec'] = 'CS_' + si['gvkey'].astype(str) + '_' + si['iid'].astype(str)
    si = {k: g.sort_values('datadate') for k, g in si.groupby('sec')}
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=['set', 'price_ticker', 'date'])
    key = set(zip(done['set'], done['price_ticker'], pd.to_datetime(done['date'])))
    llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=4, verbose=False, seed=SEED)
    rows = done.to_dict('records')
    LOG.info(f'ai_pilot: {len(C)} candidates, {len(key)} already scored')
    for i, r in C.iterrows():
        if (r['set'], r['price_ticker'], r['date']) in key:
            continue
        facts = fact_sheet(r, co, q, pc, si)
        if facts is None:
            continue
        out = llm.create_chat_completion(messages=[{'role': 'system', 'content': SYSTEM},
                                                   {'role': 'user', 'content': ASK.format(date=r['date'].date(), facts=facts)}],
                                         max_tokens=160, temperature=0)
        text = out['choices'][0]['message']['content']
        m = re.search(r'\{.*\}', text, re.S)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
        rows.append({'set': r['set'], 'price_ticker': r['price_ticker'], 'date': r['date'], 'facts': facts,
                     'score': pd.to_numeric(j.get('score'), errors='coerce'), 'decision': str(j.get('decision', '')).lower(),
                     'reasons': json.dumps(j.get('reasons', [])), 'raw': text})
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        LOG.info(f"ai_pilot: {len(rows)}/{len(C)} scored ({r['set']}, {r['price_ticker']}, score {rows[-1]['score']})")
    LOG.info('ai_pilot: scoring done')
    return evaluate(cfg)


def evaluate(cfg=None):
    from .prices import load_price
    from scipy.stats import spearmanr
    S = pd.read_parquet(OUT)
    spy = load_price('SPY')['adj_close']
    res = []
    for _, r in S.iterrows():
        px = load_price(r['price_ticker'])
        a = px['adj_close'].loc[pd.Timestamp(r['date']) + pd.Timedelta(days=1):]
        b = spy.loc[pd.Timestamp(r['date']) + pd.Timedelta(days=1):]
        if len(b) < 127:
            res.append(np.nan)
            continue
        end = b.index[126]
        a = a.loc[:end]
        res.append((a.iloc[-1] / a.iloc[0] - 1) - (b.loc[end] / b.iloc[0] - 1) if len(a) > 1 else np.nan)
    S['excess_6m'] = res
    out = []
    for name, g in [('all', S)] + list(S.groupby('set')):
        g = g.dropna(subset=['score', 'excess_6m'])
        if len(g) < 10:
            continue
        rho, p = spearmanr(g['score'], g['excess_6m'])
        out.append({'set': name, 'n': len(g), 'spearman': rho, 'p': p,
                    'excess score>=7': g.loc[g['score'] >= 7, 'excess_6m'].mean(), 'n>=7': int((g['score'] >= 7).sum()),
                    'excess score<=4': g.loc[g['score'] <= 4, 'excess_6m'].mean(), 'n<=4': int((g['score'] <= 4).sum()),
                    'excess buy': g.loc[g['decision'] == 'buy', 'excess_6m'].mean(),
                    'excess pass': g.loc[g['decision'] == 'pass', 'excess_6m'].mean(),
                    'excess all': g['excess_6m'].mean()})
    r = pd.DataFrame(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_csv(RESULTS / 'ai_pilot.csv', index=False)
    save_table(S.drop(columns=['raw']), DATA / 'ai_pilot_scored')
    LOG.info('ai_pilot: done')
    return r
