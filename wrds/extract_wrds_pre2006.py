"""Third WRDS pull: Compustat history before 2005, for out-of-sample tests on 1990-2004.

The strategies were all designed on 2006-2026, so 1990-2004 is a clean holdout: the rules are already fixed.
Pulls daily prices for every US-dollar common stock of a US company (including companies that later died),
1990-2004, plus quarterly and annual fundamentals going back far enough to compute trailing ratios.

Run on the WRDS JupyterHub terminal, in any folder:

    python3 extract_wrds_pre2006.py
    python3 extract_wrds_pre2006.py --start 1983-01-01 --end 1989-12-31   # earlier years (prices only)

Writes Parquet files to the current folder and skips any already written (safe to re-run if it stops).
Expect roughly 0.5-0.8 GB and 20-60 minutes.
"""
import argparse
import os
import sys

import pandas as pd
import wrds

OUT = os.getcwd()
START, END = '1990-01-01', '2004-12-31'
USA = "(select gvkey from comp.company where fic = 'USA')"
STD = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"
SECD = ['gvkey', 'iid', 'datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'cshoc', 'ajexdi', 'trfd',
        'curcdd', 'exchg', 'tpci', 'prcstd']
FUNDQ = ['gvkey', 'datadate', 'rdq', 'fyearq', 'fqtr', 'fyr', 'atq', 'ltq', 'ceqq', 'seqq', 'cshoq', 'prccq',
         'niq', 'ibq', 'saleq', 'revtq', 'cogsq', 'xsgaq', 'xintq', 'txtq', 'dpq', 'oiadpq', 'actq', 'lctq',
         'cheq', 'rectq', 'invtq', 'ppentq', 'dlttq', 'dlcq', 'epspxq', 'epsfxq', 'ajexq', 'oancfy', 'capxy',
         'prstkcy', 'sstky', 'dvy', 'xrdq']
FUNDA = ['gvkey', 'datadate', 'fyear', 'fyr', 'at', 'lt', 'ceq', 'seq', 'csho', 'prcc_f', 'ni', 'ib', 'sale',
         'revt', 'cogs', 'xsga', 'xint', 'txt', 'dp', 'oiadp', 'ebit', 'ebitda', 'act', 'lct', 'che', 'rect',
         'invt', 'ppent', 'dltt', 'dlc', 're', 'wcap', 'oancf', 'capx', 'prstkc', 'sstk', 'dvc', 'dvt', 'xrd',
         'emp', 'pi', 'epspx', 'ajex', 'txdb', 'itcb', 'pstk', 'pstkl', 'pstkrv']


def present(db, schema, table, cols):
    have = set(db.raw_sql(f"select column_name from information_schema.columns "
                          f"where table_schema = '{schema}' and table_name = '{table}'")['column_name'])
    return [c for c in cols if c in have]


def save(db, name, sql, date_cols=()):
    path = os.path.join(OUT, f'{name}.parquet')
    if os.path.exists(path):
        print(f'{name}: already saved, skipping', flush=True)
        return
    df = db.raw_sql(sql, date_cols=list(date_cols))
    df.to_parquet(path, index=False)
    print(f'{name}: {len(df):,} rows', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=END)
    args = ap.parse_args()
    db = wrds.Connection()
    cols = present(db, 'comp', 'fundq', FUNDQ)
    save(db, 'compustat_fundq_full_pre2003',
         f"select {', '.join(cols)} from comp.fundq where {STD} and datadate >= '1987-01-01' "
         f"and datadate < '2003-01-01' and gvkey in {USA}", ['datadate', 'rdq'])
    cols = present(db, 'comp', 'funda', FUNDA)
    save(db, 'compustat_funda_pre2002',
         f"select {', '.join(cols)} from comp.funda where {STD} and datadate >= '1986-01-01' "
         f"and datadate < '2002-01-01' and gvkey in {USA}", ['datadate'])
    cols = present(db, 'comp', 'secd', SECD)
    cond = "tpci = '0' and curcdd = 'USD' and gvkey in " + USA
    for q in pd.date_range(args.start, args.end, freq='QS'):
        name = f'compustat_daily_all_{q.year}q{q.quarter}'
        q_end = min(q + pd.offsets.QuarterEnd(0), pd.Timestamp(args.end))
        try:
            save(db, name, f"select {', '.join(cols)} from comp.secd where datadate between "
                           f"'{q.date()}' and '{q_end.date()}' and {cond}", ['datadate'])
        except Exception as e:
            print(f'{name}: failed ({str(e).splitlines()[0][:100]}); run the script again to retry', flush=True)
    db.close()
    files = [f for f in os.listdir(OUT) if f.endswith('.parquet') and ('pre20' in f or f[-14:-10] < '2005')]
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in files) / 1e9
    print(f'\nDone: {len(files)} files, {total:.2f} GB in {OUT}. Push them to the private repo as before.')


if __name__ == '__main__':
    sys.exit(main())
