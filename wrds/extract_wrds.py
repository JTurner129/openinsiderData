"""Extract the data the insider-buying backtest needs from WRDS.

Works with CRSP if you have it, or with Compustat alone: Compustat Security
Daily (comp.secd) keeps delisted companies, and comp.company carries the SEC
CIK, so prices can be matched to insider filings without CRSP.

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
SECD = ['gvkey', 'iid', 'datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'cshoc', 'ajexdi', 'trfd',
        'curcdd', 'exchg', 'tpci', 'prcstd']
SECURITY = ['gvkey', 'iid', 'tic', 'cusip', 'exchg', 'excntry', 'tpci', 'dldtei', 'dlrsni', 'secstat', 'ibtic']
LISTED_EXCHANGES = (11, 12, 14)      # NYSE, NYSE American (AMEX), Nasdaq
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


def readable(db, schema, table):
    """True if this account may actually read the table (WRDS shows tables you cannot read)."""
    try:
        db.raw_sql(f'select * from {schema}.{table} limit 1')
        return True
    except Exception as e:
        log(f'  {schema}.{table}: no access ({str(e).splitlines()[0][:80]})')
        return False


def first_table(db, candidates):
    """First (schema, table) that exists AND this account can read, with its columns."""
    for schema, table in candidates:
        cols = columns(db, schema, table)
        if cols and readable(db, schema, table):
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
        log(f'{name}: not available to your account (tried {", ".join(s + "." + t for s, t in candidates)}); skipped')
        summary[name] = 'not available'
        return None
    try:
        return _extract(db, summary, name, schema, table, cols, wanted, where, chunk_by, start, end)
    except Exception as e:
        log(f'{name}: failed ({str(e).splitlines()[0][:120]}); skipped')
        summary[name] = 'failed: ' + str(e).splitlines()[0][:80]
        return None


def _extract(db, summary, name, schema, table, cols, wanted, where, chunk_by, start, end):
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
    ap.add_argument('--ciks', default=str(Path(__file__).with_name('insider_ciks.csv')),
                    help='CSV of insider-purchase company CIKs (from the repo); their OTC prices are included')
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

    # 6. Compustat Security Daily: prices for listed US common stocks, plus any
    #    exchange (including OTC) for companies with insider purchases.
    #    Quarterly files keep each under GitHub's 100 MB limit.
    extract_table(db, summary, 'compustat_security', [('comp', 'security'), ('comp_na_daily_all', 'security')], SECURITY)
    schema, table, cols = first_table(db, [('comp', 'secd'), ('comp_na_daily_all', 'secd')])
    if not schema:
        log('compustat_daily: Security Daily (comp.secd) is not in your subscription; prices cannot come from WRDS')
        summary['compustat_daily'] = 'not available'
    else:
        ciks_path = Path(args.ciks)
        insider_gvkeys = []
        if ciks_path.exists():
            ciks = pd.read_csv(ciks_path)['cik'].astype(int).astype(str)
            cik_list = ','.join(f"'{c}'" for c in ciks)
            # CIK may be stored with or without leading zeros
            try:
                g = db.raw_sql(f"select gvkey from comp.company where ltrim(cik, '0') in ({cik_list})")
                insider_gvkeys = g['gvkey'].astype(str).tolist()
            except Exception as e:
                log(f'compustat_daily: CIK lookup failed ({str(e).splitlines()[0][:80]}); listed stocks only')
            log(f'compustat_daily: {len(insider_gvkeys):,} companies with insider purchases matched by CIK')
        else:
            log(f'compustat_daily: {ciks_path} not found; pulling exchange-listed stocks only')
        have, missing = pick(SECD, cols)
        log(f'compustat_daily: {schema}.{table}, {len(have)} columns' + (f' (not present: {", ".join(missing)})' if missing else ''))
        gv = ','.join(f"'{x}'" for x in insider_gvkeys) or "''"
        cond = (f"tpci = '0' and curcdd = 'USD' and (exchg in {LISTED_EXCHANGES} or gvkey in ({gv}))")
        total, failed = 0, []
        for q_start in pd.date_range(args.start, args.end, freq='QS'):
            label = f'compustat_daily_{q_start.year}q{q_start.quarter}'
            if any((OUT / f'{label}.{ext}').exists() for ext in ('parquet', 'csv.gz')):
                log(f'  {label}: already saved, skipping')
                continue
            q_end = min(q_start + pd.offsets.QuarterEnd(0), pd.Timestamp(args.end))
            try:
                df = db.raw_sql(f"select {', '.join(have)} from {schema}.{table} where datadate between "
                                f"'{q_start.date()}' and '{q_end.date()}' and {cond}")
            except Exception as e:
                log(f'  {label}: failed ({str(e).splitlines()[0][:100]}); re-run to retry')
                failed.append(label)
                continue
            total += len(df)
            if len(df):
                save(df, label)
        if failed:
            log(f'compustat_daily: {len(failed)} quarter(s) failed; run the script again to retry them')
        summary['compustat_daily'] = f'{schema}.{table}: {total:,} new rows' + (f', {len(failed)} quarters failed' if failed else '')

    # 7. Optional: Thomson/LSEG insider filings, Form 4 table, from 1986
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
