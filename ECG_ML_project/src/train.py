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

# %%
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'), allow_pickle=True)

print("Train:", X_train.shape, "Test:", X_test.shape)

# %%
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
classes = list(le.classes_)
print("Classi:", classes)

# %%
# Solo sotto-campionamento di N (nessun SMOTE): e' la versione che ha dato
# i risultati piu' solidi su N e V finora.
n_idx = classes.index('N')

under = RandomUnderSampler(
    sampling_strategy={n_idx: 4000},
    random_state=42
)

pipeline = ImbPipeline([
    ('under', under),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

print("Addestramento in corso...")
pipeline.fit(X_train, y_train_enc)

# %%
y_pred = pipeline.predict(X_test)

print("\n--- Classification report ---")
print(classification_report(y_test_enc, y_pred, target_names=classes, zero_division=0))

print("\n--- Confusion matrix ---")
print("Classi (ordine righe/colonne):", classes)
print(confusion_matrix(y_test_enc, y_pred))

# %%
joblib.dump(pipeline, os.path.join(MODELS_DIR, 'ecg_model_rf.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
print("\nModello salvato in models/ecg_model_rf.pkl")