"""Extract the data the insider-buying backtest needs from WRDS.

Run this ON WRDS (JupyterHub at https://wrds-jupyter.wharton.upenn.edu, or the
WRDS Cloud), where `import wrds` connects without a password:

    python extract_wrds.py                 # everything except the optional insider table
    python extract_wrds.py --with-insiders # also Thomson/LSEG insider filings (large)
    python extract_wrds.py --start 2005-01-01

It writes Parquet files to ~/insider_wrds/ and prints a summary. Each table is
checked against your subscription first: anything missing is skipped and
reported, not fatal.

WRDS data is licensed: keep these files private (a private repository or your
own machine), never in a public repository.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

OUT = Path.home() / 'insider_wrds'

# Column wishes per table; whichever exist in your WRDS version are pulled.
CIZ_DAILY = ['permno', 'dlycaldt', 'dlyprc', 'dlyopen', 'dlyhigh', 'dlylow', 'dlybid', 'dlyask', 'dlyclose',
             'dlyvol', 'dlyret', 'dlyretx', 'dlyretmissflg', 'dlydelflg', 'shrout', 'dlycumfacpr', 'dlycumfacshr',
             'dlyprcflg']
LEGACY_DAILY = ['permno', 'date', 'prc', 'openprc', 'askhi', 'bidlo', 'bid', 'ask', 'vol', 'ret', 'retx', 'shrout',
                'cfacpr', 'cfacshr']
CIZ_SECINFO = ['permno', 'secinfostartdt', 'secinfoenddt', 'ticker', 'issuernm', 'cusip', 'primaryexch', 'sharetype',
               'securitytype', 'securitysubtype', 'usincflg', 'issuertype', 'tradingstatusflg', 'siccd']
LEGACY_NAMES = ['permno', 'namedt', 'nameenddt', 'ticker', 'comnam', 'ncusip', 'shrcd', 'exchcd', 'siccd']
CIZ_DELIST = ['permno', 'delistingdt', 'delret', 'delretmisstype', 'delactiontype', 'delstatustype', 'delreasontype',
              'delpaymenttype']
LEGACY_DELIST = ['permno', 'dlstdt', 'dlret', 'dlretx', 'dlstcd']
CCM = ['gvkey', 'lpermno', 'lpermco', 'linktype', 'linkprim', 'linkdt', 'linkenddt']
COMPANY = ['gvkey', 'cik', 'conm', 'sic', 'naics', 'loc', 'ipodate', 'dldte', 'dlrsn']
FUNDQ = ['gvkey', 'datadate', 'rdq', 'fyearq', 'fqtr', 'ceqq', 'seqq', 'cshoq', 'atq', 'ltq', 'prccq', 'niq', 'saleq']
INSIDERS = ['personid', 'owner', 'rolecode1', 'rolecode2', 'rolecode3', 'rolecode4', 'cusip6', 'cusip2', 'ticker',
            'cname', 'trandate', 'fdate', 'acqdisp', 'trancode', 'shares', 'tprice', 'ownership', 'sharesheld',
            'cleanse', 'seqnum', 'sectitle', 'formtype', 'secid', 'dcn', 'amend']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def columns(db, schema, table):
    q = ("select column_name from information_schema.columns "
         f"where table_schema = '{schema}' and table_name = '{table}'")
    try:
        return set(db.raw_sql(q)['column_name'].str.lower())
    except Exception:
        return set()


def first_table(db, candidates):
    """First (schema, table) that exists, with its columns."""
    for schema, table in candidates:
        cols = columns(db, schema, table)
        if cols:
            return schema, table, cols
    return None, None, set()


def pick(wanted, available):
    have = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    return have, missing


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.parquet'
    try:
        df.to_parquet(path, index=False)
    except Exception:          # pyarrow missing: fall back to gzipped CSV
        path = OUT / f'{name}.csv.gz'
        df.to_csv(path, index=False)
    log(f'  wrote {path.name}: {len(df):,} rows, {path.stat().st_size / 1e6:.1f} MB')
    return path


def extract_table(db, summary, name, candidates, wanted, where='', chunk_by=None, start=None, end=None):
    schema, table, cols = first_table(db, candidates)
    if not schema:
        log(f'{name}: not available in your subscription (tried {", ".join(s + "." + t for s, t in candidates)}); skipped')
        summary[name] = 'not available'
        return None
    have, missing = pick(wanted, cols)
    log(f'{name}: {schema}.{table}, {len(have)} columns' + (f' (not present: {", ".join(missing)})' if missing else ''))
    sel = ', '.join(have)
    if chunk_by:
        total = 0
        for y in range(int(start[:4]), int(end[:4]) + 1):
            q = (f"select {sel} from {schema}.{table} where {chunk_by} between '{max(start, f'{y}-01-01')}' "
                 f"and '{min(end, f'{y}-12-31')}'" + (f' and {where}' if where else ''))
            df = db.raw_sql(q)
            total += len(df)
            if len(df):
                save(df, f'{name}_{y}')
        summary[name] = f'{schema}.{table}: {total:,} rows'
    else:
        q = f'select {sel} from {schema}.{table}' + (f' where {where}' if where else '')
        df = db.raw_sql(q)
        save(df, name)
        summary[name] = f'{schema}.{table}: {len(df):,} rows'
    return schema, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default=pd.Timestamp.today().strftime('%Y-%m-%d'))
    ap.add_argument('--with-insiders', action='store_true', help='also pull Thomson/LSEG insider filings (large)')
    ap.add_argument('--insiders-start', default='1986-01-01')
    args = ap.parse_args()

    import wrds
    db = wrds.Connection()
    summary = {}
    log(f'Writing to {OUT}')

    # 1. CRSP security info (which permnos are US common stocks, ticker history)
    sec = extract_table(db, summary, 'crsp_security_info',
                        [('crsp', 'stksecurityinfohist'), ('crsp_a_stock', 'stksecurityinfohist')], CIZ_SECINFO)
    if not sec:
        extract_table(db, summary, 'crsp_security_info',
                      [('crsp', 'stocknames'), ('crsp_a_stock', 'stocknames')], LEGACY_NAMES)

    # 2. CRSP daily stock file, one file per year, US common stocks only
    schema, table, cols = first_table(db, [('crsp', 'dsf_v2'), ('crsp_a_stock', 'dsf_v2')])
    if schema:
        common = ("permno in (select distinct permno from {s}.stksecurityinfohist where sharetype = 'NS' "
                  "and securitytype = 'EQTY' and securitysubtype = 'COM' and usincflg = 'Y')").format(s=schema)
        extract_table(db, summary, 'crsp_daily', [(schema, table)], CIZ_DAILY, where=common,
                      chunk_by='dlycaldt', start=args.start, end=args.end)
    else:
        extract_table(db, summary, 'crsp_daily', [('crsp', 'dsf'), ('crsp_a_stock', 'dsf')], LEGACY_DAILY,
                      where="permno in (select distinct permno from crsp.stocknames where shrcd in (10, 11))",
                      chunk_by='date', start=args.start, end=args.end)
        log('NOTE: legacy CRSP daily file used; it may stop before the latest year.')

    # 3. Delistings (date, return, reason)
    if not extract_table(db, summary, 'crsp_delistings', [('crsp', 'stkdelists'), ('crsp_a_stock', 'stkdelists')], CIZ_DELIST):
        extract_table(db, summary, 'crsp_delistings', [('crsp', 'dsedelist'), ('crsp_a_stock', 'dsedelist')], LEGACY_DELIST)

    # 4. CRSP/Compustat link and Compustat company file (CIK)
    extract_table(db, summary, 'ccm_link', [('crsp', 'ccmxpf_lnkhist'), ('crsp_a_ccm', 'ccmxpf_lnkhist')], CCM,
                  where="linktype in ('LU', 'LC') and linkprim in ('P', 'C')")
    extract_table(db, summary, 'compustat_company', [('comp', 'company'), ('comp_na_daily_all', 'company')], COMPANY)

    # 5. Compustat quarterly fundamentals (book value, shares) from 2003
    extract_table(db, summary, 'compustat_fundq', [('comp', 'fundq'), ('comp_na_daily_all', 'fundq')], FUNDQ,
                  where="indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' "
                        "and datadate >= '2003-01-01'")

    # 6. Optional: Thomson/LSEG insider filings, Form 4 table, from 1986
    if args.with_insiders:
        extract_table(db, summary, 'tfn_insiders', [('tfn', 'table1'), ('tr_insiders', 'table1')], INSIDERS,
                      chunk_by='fdate', start=args.insiders_start, end=args.end)

    db.close()
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9
    print('\n=== Summary ===')
    for k, v in summary.items():
        print(f'{k:22s} {v}')
    print(f'\n{len(list(OUT.iterdir()))} files, {total:.2f} GB in {OUT}')
    print('Keep these files private (WRDS licence). Next: push them to a PRIVATE GitHub repository.')


if __name__ == '__main__':
    sys.exit(main())
