"""Pull the Markit Securities Finance 2011 US equity sample (real borrow fees) to calibrate the fee estimates.

Run on the WRDS terminal:  python3 extract_markit_sample.py
"""
import os

import wrds

db = wrds.Connection()
for table in ('amereqty2011', 'amereqtyi2011'):
    path = f'markit_{table}.parquet'
    if os.path.exists(path):
        print(f'{table}: already saved')
        continue
    try:
        cols = db.raw_sql(f"select column_name from information_schema.columns "
                          f"where table_schema = 'mrktsamp_msf' and table_name = '{table}'")['column_name'].tolist()
        print(f'{table}: {len(cols)} columns: {", ".join(cols)}')
        df = db.raw_sql(f'select * from mrktsamp_msf.{table}')
        df.to_parquet(path, index=False)
        print(f'{table}: {len(df):,} rows -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
    except Exception as e:
        print(f'{table}: failed: {str(e).splitlines()[0][:150]}')
