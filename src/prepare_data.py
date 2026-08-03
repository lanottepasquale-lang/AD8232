# %%
import os
import numpy as np
import wfdb

from utils import DS1, DS2, bandpass_filter, extract_beats

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mitdb')

# %%
# Scarica tutti i record del MIT-BIH Arrhythmia Database (una sola volta)
os.makedirs(DATA_DIR, exist_ok=True)
wfdb.dl_database('mitdb', dl_dir=DATA_DIR)

# %%
def load_records(record_list):
    X_all, y_all = [], []
    for rec_name in record_list:
        path = os.path.join(DATA_DIR, rec_name)
        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')

        signal = record.p_signal[:, 0]          # prima derivazione (MLII)
        signal_clean = bandpass_filter(signal)

        X, y = extract_beats(signal_clean, annotation.sample, annotation.symbol)
        X_all.append(X)
        y_all.append(y)
        print(f"Record {rec_name}: {len(y)} battiti estratti")

    return np.concatenate(X_all), np.concatenate(y_all)

# %%
print("Preparo il training set (DS1)...")
X_train, y_train = load_records(DS1)

print("Preparo il test set (DS2)...")
X_test, y_test = load_records(DS2)

# %%
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

print("Fatto.")
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Distribuzione classi (train):", {c: (y_train == c).sum() for c in np.unique(y_train)})