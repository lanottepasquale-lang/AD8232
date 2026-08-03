# %%
import sys
import os
import pandas as pd

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')


def latest_log():
    files = sorted(f for f in os.listdir(LOGS_DIR) if f.endswith('.csv'))
    if not files:
        raise FileNotFoundError("Nessun log trovato in logs/")
    return os.path.join(LOGS_DIR, files[-1])


# %%
path = sys.argv[1] if len(sys.argv) > 1 else latest_log()
print(f"Log analizzato: {path}\n")

df = pd.read_csv(path)
durata = df['t_sec'].max()
print(f"Durata sessione: {durata:.1f}s ({durata/60:.1f} min)\n")

# %%
print("--- Battiti per tipo ---")
beats = df[df['event_type'] == 'beat']
print(beats['label'].value_counts().to_string())

# %%
print("\n--- Eventi (frequenza, pattern, qualita' segnale, dispositivo) ---")
events = df[df['event_type'] == 'event']
if len(events) > 0:
    print(events[['t_sec', 'label', 'confidence', 'detail']].to_string(index=False))
else:
    print("Nessuno")

# %%
print("\n--- Episodi di ritmo (cambi AFib/normale/incerto) ---")
rhythm = df[df['event_type'] == 'rhythm']
if len(rhythm) > 0:
    print(rhythm[['t_sec', 'label', 'confidence']].to_string(index=False))
else:
    print("Nessuno")

# %%
segments = df[df['event_type'] == 'segment']
if len(segments) > 0:
    print("\n--- Segmenti del tour (se presenti) ---")
    print(segments[['t_sec', 'label', 'detail']].to_string(index=False))