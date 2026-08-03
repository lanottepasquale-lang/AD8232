# %%
import os
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'test_records.txt')) as f:
    TEST_RECORDS = [line.strip() for line in f if line.strip()]

# %%
X_train = np.load(os.path.join(DATA_DIR, 'X_rhythm_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_rhythm_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(DATA_DIR, 'X_rhythm_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_rhythm_test.npy'), allow_pickle=True)
rec_test = np.load(os.path.join(DATA_DIR, 'rec_test.npy'), allow_pickle=True)

print("Train:", X_train.shape, "Test:", X_test.shape)

# %%
# Diagnostica: le due classi si separano nelle feature grezze?
feat_names = ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'cv', 'iqr', 'entropy']
for i, name in enumerate(feat_names):
    afib_vals = X_train[y_train == 'AFIB', i]
    n_vals = X_train[y_train == 'N', i]
    print(f"{name:10s}  AFIB: media={afib_vals.mean():.3f} std={afib_vals.std():.3f}   "
          f"N: media={n_vals.mean():.3f} std={n_vals.std():.3f}")

# %%
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
classes = list(le.classes_)
print("\nClassi:", classes)

# %%
afib_idx = classes.index('AFIB')
n_idx = classes.index('N')
afib_count = int((y_train_enc == afib_idx).sum())
target_n = min(afib_count * 2, int((y_train_enc == n_idx).sum()))

under = RandomUnderSampler(sampling_strategy={n_idx: target_n}, random_state=42)

pipeline = ImbPipeline([
    ('under', under),
    ('clf', RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'
    ))
])

print(f"Sotto-campiono N a {target_n} esempi (AFIB resta a {afib_count})")
pipeline.fit(X_train, y_train_enc)

# %%
y_pred = pipeline.predict(X_test)

print("\n--- Classification report ---")
print(classification_report(y_test_enc, y_pred, target_names=classes, zero_division=0))

print("\n--- Confusion matrix ---")
print("Classi (ordine righe/colonne):", classes)
print(confusion_matrix(y_test_enc, y_pred))

# %%
# Diagnostica: il fallimento su AFIB e' concentrato su un singolo paziente?
print("\n--- Recall AFIB per singolo record di test ---")
afib_code = le.transform(['AFIB'])[0]
for rec in TEST_RECORDS:
    mask = (rec_test == rec) & (y_test_enc == afib_code)
    n_examples = int(mask.sum())
    if n_examples > 0:
        rec_recall = (y_pred[mask] == afib_code).mean()
        print(f"Record {rec}: recall AFIB = {rec_recall:.2f} (su {n_examples} esempi)")
    else:
        print(f"Record {rec}: nessun esempio AFIB in test")

# %%
joblib.dump(pipeline, os.path.join(MODELS_DIR, 'rhythm_model_rf.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'rhythm_label_encoder.pkl'))
print("\nModello salvato in models/rhythm_model_rf.pkl")