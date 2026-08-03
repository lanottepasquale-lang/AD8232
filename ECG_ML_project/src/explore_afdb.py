# %%
import os
import wfdb
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'afdb')
os.makedirs(DATA_DIR, exist_ok=True)

# Piccolo sottoinsieme per iniziare (6 record su 23 disponibili in totale)
SUBSET = ['04015', '04043', '04048', '05091', '07162', '08434']

# %%
for rec in SUBSET:
    wfdb.dl_database('afdb', dl_dir=DATA_DIR, records=[rec])

# %%
for rec in SUBSET:
    path = os.path.join(DATA_DIR, rec)
    record = wfdb.rdrecord(path)
    rhythm_ann = wfdb.rdann(path, 'atr')  # annotazioni di ritmo (aux_note)

    durata_min = len(record.p_signal) / record.fs / 60
    rhythms = Counter(rhythm_ann.aux_note)

    print(f"\nRecord {rec} - durata {durata_min:.1f} min, fs={record.fs}Hz")
    print("Ritmi presenti (conteggio cambi di stato, non durata):")
    for rhythm, count in rhythms.most_common():
        print(f"  {rhythm!r}: {count}")