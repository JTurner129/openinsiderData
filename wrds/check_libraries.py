"""List the WRDS libraries this login can see, highlighting securities-lending (borrow fee) data.

Run on the WRDS terminal:  python3 check_libraries.py
"""
import wrds

db = wrds.Connection()
libs = sorted(db.list_libraries())
keys = ('markit', 'msf', 'secfin', 'lending', 'borrow', 'short')
hits = [l for l in libs if any(k in l.lower() for k in keys)]
print(f'{len(libs)} libraries visible')
print('Possible borrow-fee / securities-lending libraries:', hits or 'none')
for lib in hits:
    try:
        print(f'  {lib}: {db.list_tables(lib)[:15]}')
    except Exception as e:
        print(f'  {lib}: not readable ({str(e).splitlines()[0][:80]})')
print('All libraries:', ', '.join(libs))
