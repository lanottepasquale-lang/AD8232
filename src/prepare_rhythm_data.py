# %%
import os
import random
import numpy as np
import wfdb

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'afdb')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RHYTHM_MAP = {'(N': 'N', '(AFIB': 'AFIB'}
FS = 250
WINDOW_BEATS = 40
STEP_BEATS = 15

# %%
# Elenco ufficiale dei record disponibili, preso direttamente da PhysioNet
# (evita di scrivere a mano nomi di file che potrebbero contenere errori)
all_records = wfdb.get_record_list('afdb')
print(f"Record disponibili in afdb: {len(all_records)}")
print(all_records)

# %%
# Scarica tutto (salta automaticamente i file gia' presenti da prima)
wfdb.dl_database('afdb', dl_dir=DATA_DIR, records=all_records)

# %%
# Split per paziente 70/30, con seed fisso per riproducibilita'
random.seed(42)
shuffled = all_records.copy()
random.shuffle(shuffled)
split_point = int(len(shuffled) * 0.7)
TRAIN_RECORDS = shuffled[:split_point]
TEST_RECORDS = shuffled[split_point:]

print(f"\nTrain ({len(TRAIN_RECORDS)} pazienti): {TRAIN_RECORDS}")
print(f"Test  ({len(TEST_RECORDS)} pazienti): {TEST_RECORDS}")


def label_for_peak(peak_sample, ann_samples, ann_labels):
    idx = np.searchsorted(ann_samples, peak_sample, side='right') - 1
    idx = max(0, min(idx, len(ann_labels) - 1))
    return ann_labels[idx]


def rr_features(rr_window):
    mean_rr = np.mean(rr_window)
    sdnn = np.std(rr_window)
    diffs = np.diff(rr_window)
    rmssd = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0
    pnn50 = np.mean(np.abs(diffs) > 0.05) if len(diffs) > 0 else 0.0
    cv = sdnn / (mean_rr + 1e-8)
    iqr = np.percentile(rr_window, 75) - np.percentile(rr_window, 25)
    hist, _ = np.histogram(rr_window, bins=8, density=False)
    probs = hist / (hist.sum() + 1e-8)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return [mean_rr, sdnn, rmssd, pnn50, cv, iqr, entropy]


def extract_windows(record_name):
    path = os.path.join(DATA_DIR, record_name)
    try:
        ann = wfdb.rdann(path, 'atr')
        qrs = wfdb.rdann(path, 'qrs')
    except Exception as e:
        print(f"  Saltato {record_name}: impossibile leggere le annotazioni ({e})")
        return np.empty((0, 7)), np.array([])

    ann_labels = [RHYTHM_MAP.get(note) for note in ann.aux_note]
    peaks = qrs.sample
    if len(peaks) < WINDOW_BEATS + 1:
        return np.empty((0, 7)), np.array([])

    peak_labels = [label_for_peak(p, ann.sample, ann_labels) for p in peaks]
    rr_all = np.diff(peaks) / FS

    X, y = [], []
    for i in range(0, len(rr_all) - WINDOW_BEATS, STEP_BEATS):
        window_labels = peak_labels[i: i + WINDOW_BEATS + 1]
        if None in window_labels or len(set(window_labels)) != 1:
            continue
        rr_window = rr_all[i: i + WINDOW_BEATS]
        X.append(rr_features(rr_window))
        y.append(window_labels[0])

    return np.array(X), np.array(y)


def build_set(records):
    X_all, y_all, rec_all = [], [], []
    for rec in records:
        X, y = extract_windows(rec)
        if len(y) > 0:
            print(f"Record {rec}: {len(y)} finestre - {dict((l, int((y == l).sum())) for l in set(y))}")
        X_all.append(X)
        y_all.append(y)
        rec_all.extend([rec] * len(y))
    return np.concatenate(X_all), np.concatenate(y_all), np.array(rec_all)


# %%
print("\nTraining set:")
X_train, y_train, rec_train = build_set(TRAIN_RECORDS)
print("\nTest set:")
X_test, y_test, rec_test = build_set(TEST_RECORDS)

print("\nTotali - train:", X_train.shape, "test:", X_test.shape)

# %%
np.save(os.path.join(OUT_DIR, 'X_rhythm_train.npy'), X_train)
np.save(os.path.join(OUT_DIR, 'y_rhythm_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'X_rhythm_test.npy'), X_test)
np.save(os.path.join(OUT_DIR, 'y_rhythm_test.npy'), y_test)
np.save(os.path.join(OUT_DIR, 'rec_test.npy'), rec_test)

# Salviamo anche l'elenco dei record di test, serve a train_rhythm.py
with open(os.path.join(OUT_DIR, 'test_records.txt'), 'w') as f:
    f.write('\n'.join(TEST_RECORDS))

print("Salvato in data/")