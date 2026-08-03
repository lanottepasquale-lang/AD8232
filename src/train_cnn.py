# %%
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SIGNAL_LEN = 240  # pre=130 + post=110, definiti in utils.extract_beats

# %%
X_train_full = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train_full = np.load(os.path.join(DATA_DIR, 'y_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'), allow_pickle=True)

print("Train:", X_train_full.shape, "Test:", X_test.shape)

# %%
# Escludiamo Q: con soli 8 esempi in tutto il dataset, il suo class_weight
# calcolato automaticamente diventa enorme e destabilizza l'addestramento
# della rete, penalizzando anche le altre classi minoritarie (L, R).
mask_train = y_train_full != 'Q'
mask_test = y_test != 'Q'
X_train_full = X_train_full[mask_train]
y_train_full = y_train_full[mask_train]
X_test = X_test[mask_test]
y_test = y_test[mask_test]
print("Dopo aver rimosso Q - Train:", X_train_full.shape, "Test:", X_test.shape)

# %%
le = LabelEncoder()
y_train_full_enc = le.fit_transform(y_train_full)
y_test_enc = le.transform(y_test)
classes = list(le.classes_)
print("Classi:", classes)

# %%
# Separiamo segnale grezzo (per la CNN) e feature RR (branch separato)
sig_train_full = X_train_full[:, :SIGNAL_LEN]
feat_train_full = X_train_full[:, SIGNAL_LEN:]
sig_test = X_test[:, :SIGNAL_LEN]
feat_test = X_test[:, SIGNAL_LEN:]

# Le feature RR hanno scale diverse dal segnale gia' normalizzato, le
# standardizziamo separatamente
feat_scaler = StandardScaler()
feat_train_full = feat_scaler.fit_transform(feat_train_full)
feat_test = feat_scaler.transform(feat_test)

# %%
# Split di validazione, solo per monitorare il training (il test DS2 resta
# intatto per la valutazione finale, come nel Random Forest)
sig_train, sig_val, feat_train, feat_val, y_train, y_val = train_test_split(
    sig_train_full, feat_train_full, y_train_full_enc,
    test_size=0.1, stratify=y_train_full_enc, random_state=42
)

# %%
def build_model(signal_len, n_features, n_classes):
    signal_input = keras.Input(shape=(signal_len, 1), name='signal')
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(signal_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    feat_input = keras.Input(shape=(n_features,), name='rr_features')
    f = layers.Dense(16, activation='relu')(feat_input)

    combined = layers.Concatenate()([x, f])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    output = layers.Dense(n_classes, activation='softmax')(combined)

    model = keras.Model(inputs=[signal_input, feat_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(SIGNAL_LEN, feat_train.shape[1], len(classes))
model.summary()

# %%
class_weights_arr = compute_class_weight(
    'balanced', classes=np.arange(len(classes)), y=y_train
)
class_weight = dict(enumerate(class_weights_arr))
print("Class weight:", {classes[k]: round(v, 2) for k, v in class_weight.items()})

# %%
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

history = model.fit(
    {'signal': sig_train[..., None], 'rr_features': feat_train},
    y_train,
    validation_data=(
        {'signal': sig_val[..., None], 'rr_features': feat_val},
        y_val
    ),
    epochs=60,
    batch_size=128,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

# %%
y_pred_proba = model.predict({'signal': sig_test[..., None], 'rr_features': feat_test})
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n--- Classification report ---")
print(classification_report(y_test_enc, y_pred, target_names=classes, zero_division=0))

print("\n--- Confusion matrix ---")
print("Classi (ordine righe/colonne):", classes)
print(confusion_matrix(y_test_enc, y_pred))

# %%
model.save(os.path.join(MODELS_DIR, 'ecg_model_cnn.keras'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
joblib.dump(feat_scaler, os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
print("\nModello salvato in models/ecg_model_cnn.keras")