# %%
import os
import wfdb
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mitdb')

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220',
       '223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232',
       '233','234']


def count_symbols(records, target_symbols=('L', 'R')):
    counts = {}
    for rec in records:
        path = os.path.join(DATA_DIR, rec)
        ann = wfdb.rdann(path, 'atr')
        c = Counter(ann.symbol)
        relevant = {s: c[s] for s in target_symbols if c[s] > 0}
        if relevant:
            counts[rec] = relevant
    return counts


print("--- DS1 (train) ---")
ds1_counts = count_symbols(DS1)
for rec, c in ds1_counts.items():
    print(f"  {rec}: {c}")
if not ds1_counts:
    print("  (no record with L or R)")

print("\n--- DS2 (test) ---")
ds2_counts = count_symbols(DS2)
for rec, c in ds2_counts.items():
    print(f"  {rec}: {c}")
if not ds2_counts:
    print("  (no record with L or R)")
