"""Second WRDS pull: fuller Compustat fundamentals for every US company, plus index membership and ratings.

Run on the WRDS JupyterHub terminal, in the same folder as before:

    python3 extract_wrds_v2.py

It writes Parquet files to the current folder and skips any it already wrote (safe to re-run).
Tables your subscription can't read are skipped with a message. Expect roughly 0.5-1 GB.
"""
import os
import sys

import wrds

OUT = os.getcwd()
USA = "(select gvkey from comp.company where fic = 'USA')"
STD = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"

FUNDQ = ['gvkey', 'datadate', 'rdq', 'fyearq', 'fqtr', 'fyr', 'atq', 'ltq', 'ceqq', 'seqq', 'cshoq', 'prccq',
         'niq', 'ibq', 'saleq', 'revtq', 'cogsq', 'xsgaq', 'xintq', 'txtq', 'dpq', 'oiadpq', 'actq', 'lctq',
         'cheq', 'rectq', 'invtq', 'ppentq', 'dlttq', 'dlcq', 'epspxq', 'epsfxq', 'ajexq', 'oancfy', 'capxy',
         'prstkcy', 'sstky', 'dvy', 'xrdq']
FUNDA = ['gvkey', 'datadate', 'fyear', 'fyr', 'at', 'lt', 'ceq', 'seq', 'csho', 'prcc_f', 'ni', 'ib', 'sale',
         'revt', 'cogs', 'xsga', 'xint', 'txt', 'dp', 'oiadp', 'ebit', 'ebitda', 'act', 'lct', 'che', 'rect',
         'invt', 'ppent', 'dltt', 'dlc', 're', 'wcap', 'oancf', 'capx', 'prstkc', 'sstk', 'dvc', 'dvt', 'xrd',
         'emp', 'pi', 'epspx', 'ajex', 'txdb', 'itcb', 'pstk', 'pstkl', 'pstkrv']


def readable(db, schema, table):
    try:
        db.raw_sql(f'select * from {schema}.{table} limit 1')
        return True
    except Exception as e:
        print(f'  skip {schema}.{table}: {str(e).splitlines()[0][:120]}')
        return False


def save(db, name, sql, date_cols=()):
    path = os.path.join(OUT, f'{name}.parquet')
    if os.path.exists(path):
        print(f'{name}: already saved, skipping')
        return
    print(f'{name}: querying...', flush=True)
    df = db.raw_sql(sql, date_cols=list(date_cols))
    df.to_parquet(path, index=False)
    print(f'{name}: {len(df):,} rows -> {path}', flush=True)


def present(db, schema, table, cols):
    """Only ask for columns the table actually has (older subscriptions lack a few)."""
    have = set(db.raw_sql(f"select column_name from information_schema.columns "
                          f"where table_schema = '{schema}' and table_name = '{table}'")['column_name'])
    missing = [c for c in cols if c not in have]
    if missing:
        print(f'  {schema}.{table}: not available, skipped: {", ".join(missing)}')
    return [c for c in cols if c in have]


def main():
    db = wrds.Connection()
    if readable(db, 'comp', 'fundq'):
        cols = present(db, 'comp', 'fundq', FUNDQ)
        save(db, 'compustat_fundq_full',
             f"select {', '.join(cols)} from comp.fundq where {STD} and datadate >= '2003-01-01' and gvkey in {USA}",
             ['datadate', 'rdq'])
    if readable(db, 'comp', 'funda'):
        cols = present(db, 'comp', 'funda', FUNDA)
        save(db, 'compustat_funda',
             f"select {', '.join(cols)} from comp.funda where {STD} and datadate >= '2002-01-01' and gvkey in {USA}",
             ['datadate'])
    if readable(db, 'comp', 'idxcst_his'):
        save(db, 'compustat_index_members', 'select gvkey, iid, gvkeyx, "from", thru from comp.idxcst_his',
             ['from', 'thru'])
    if readable(db, 'comp', 'idx_index'):
        save(db, 'compustat_index_names', 'select gvkeyx, conm, indexcat, idxcstflg, tic from comp.idx_index')
    if readable(db, 'comp', 'adsprate'):
        save(db, 'compustat_ratings',
             f"select gvkey, datadate, splticrm, spsdrm from comp.adsprate where gvkey in {USA}",
             ['datadate'])
    print('\nDone. Files in', OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.parquet') and ('full' in f or 'funda' in f or 'index' in f or 'rating' in f):
            print(f'  {f}  {os.path.getsize(os.path.join(OUT, f)) / 1e6:,.1f} MB')


if __name__ == '__main__':
    sys.exit(main())
