"""Configurazione centralizzata: soglie, finestre e percorsi usati da
inference.py. Tenerli qui separati rende piu' semplice ritarare il sistema
(specialmente dopo il collegamento al segnale reale da ESP32/AD8232) senza
dover toccare la logica applicativa."""

import os

# --- Percorsi ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'mitdb')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# --- Segnale ---
FS = 360
PRE = 130
POST = 110

# --- Classificazione battito ---
BEAT_CONF_THRESHOLD = 0.5

# --- Classificazione ritmo ---
RHYTHM_CONF_THRESHOLD = 0.75
RHYTHM_WINDOW_BEATS = 40

# --- Regole su frequenza ---
TACHY_BPM = 100
BRADY_BPM = 60
PAUSE_SEC = 2.0
MIN_CONSECUTIVE_RATE = 4

# --- Qualita' del segnale ---
# Calibrate su MIT-BIH (segnale gia' pulito): con l'AD8232 reale, osserva il
# rumore di base a riposo e ricalibra questi valori di conseguenza.
RAW_FLAT_STD_THRESHOLD = 0.01
BEAT_SATURATION_ZSCORE = 6.0

# --- Pattern di battito ---
PATTERN_HISTORY_LEN = 6
PVC_BURDEN_WINDOW_SEC = 60
PVC_BURDEN_MIN_BEATS = 10
PVC_BURDEN_ALERT_FRACTION = 0.10

# --- Rilevatore QRS: search-back ---
QRS_SEARCHBACK_MULTIPLIER = 1.5
QRS_SEARCHBACK_THRESHOLD_FACTOR = 0.5

MIN_PHYSIOLOGICAL_RR_SEC = 0.28   # sotto questo, quasi certo doppio conteggio dello stesso QRS
T_WAVE_WINDOW_SEC = 0.36          # finestra in cui un candidato debole e' probabile onda T
T_WAVE_ENERGY_RATIO = 0.5         # sotto questa frazione dell'energia del QRS precedente, e' onda T

# --- Onde P e T (approssimazione intervalli PR e RT) ---
# NOTA IMPORTANTE: sono misure picco-picco, non onset-offset come nello
# standard clinico rigoroso. Servono come indicatori di tendenza/soglia,
# non come misura diagnostica precisa. Calibrate su MIT-BIH: da
# ricalibrare quasi certamente sul segnale reale da AD8232.
P_SEARCH_START_MS = 280    # limite piu' lontano dal picco R (ms prima)
P_SEARCH_END_MS = 60       # limite piu' vicino al picco R (ms prima)
P_MIN_AMPLITUDE = 0.05     # soglia minima per considerare valido un candidato P
P_SEARCH_MIN_RR_SEC = 0.5  # sotto questo RR (>120 bpm) la ricerca P e' saltata

T_SEARCH_START_MS = 100    # limite piu' vicino al picco R (ms dopo)
T_SEARCH_END_MS = 500      # limite piu' lontano (ms dopo), adattato su next_rr
T_MIN_AMPLITUDE = 0.05

PR_LONG_MS = 200   # blocco AV di primo grado
PR_SHORT_MS = 120  # pre-eccitazione o ritmo giunzionale/atriale basso

RTC_LONG_MS = 460  # QTc lungo (formula di Fridericia), soglia indicativa
RTC_SHORT_MS = 350 # QTc corto, soglia indicativa

MIN_CONSECUTIVE_PQT = 3   # battiti consecutivi richiesti prima di dichiarare un'anomalia sostenuta

# --- Debug/visualizzazione onde P e T ---
SHOW_PQT_MARKERS = True     # disegna un marcatore su ogni P/T trovato
PRINT_PQT_PER_BEAT = True   # stampa PR/RT per ogni battito, non solo le anomalie sostenute

PQT_APPLICABLE_LABELS = {'N', 'S'}  # PR/RT calcolati solo su battiti a QRS stretto e non ectopico

SERIAL_PORT = "/dev/cu.usbserial-0001"   # la tua porta reale (verifica con: ls /dev/cu.*)
BAUD_RATE = 115200