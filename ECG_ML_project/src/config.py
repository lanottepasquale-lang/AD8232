"""Centralized configuration: thresholds, windows, and paths used by
inference.py. Keeping them separated here makes it easier to recalibrate the system
(especially after connecting to the real signal from the ESP32/AD8232) without
having to modify the application logic."""

import os

# --- path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'mitdb')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# --- signal ---
FS = 360
PRE = 130
POST = 110

# --- beat classification ---
BEAT_CONF_THRESHOLD = 0.5

# --- rithm classification ---
RHYTHM_CONF_THRESHOLD = 0.75
RHYTHM_WINDOW_BEATS = 40

# --- frequence rules ---
TACHY_BPM = 100
BRADY_BPM = 60
PAUSE_SEC = 2.0
MIN_CONSECUTIVE_RATE = 4

# --- Signal Quality ---
# Calibrated on MIT-BIH (already clean signal): with the real AD8232, observe the
# baseline noise at rest and recalibrate these values accordingly.
RAW_FLAT_STD_THRESHOLD = 0.01
BEAT_SATURATION_ZSCORE = 6.0

# --- Beat Pattern ---
PATTERN_HISTORY_LEN = 6
PVC_BURDEN_WINDOW_SEC = 60
PVC_BURDEN_MIN_BEATS = 10
PVC_BURDEN_ALERT_FRACTION = 0.10

# --- QRS Detector: search-back ---
QRS_SEARCHBACK_MULTIPLIER = 1.5
QRS_SEARCHBACK_THRESHOLD_FACTOR = 0.5

MIN_PHYSIOLOGICAL_RR_SEC = 0.28   # below this, almost certainly double counting of the same QRS
T_WAVE_WINDOW_SEC = 0.36          # window where a weak candidate is likely a T wave
T_WAVE_ENERGY_RATIO = 0.5         # below this fraction of the previous QRS energy, it's a T wave

# --- P and T waves (approximation of PR and RT intervals) ---
# IMPORTANT NOTE: these are peak-to-peak measurements, not onset-offset as in the
# rigorous clinical standard. They serve as trend/threshold indicators,
# not as precise diagnostic measurements. Calibrated on MIT-BIH: must
# almost certainly be recalibrated on the real AD8232 signal.
P_SEARCH_START_MS = 280   # furthest limit from the R peak (ms before)
P_SEARCH_END_MS = 60      # closest limit to the R peak (ms before)
P_MIN_AMPLITUDE = 0.05    # minimum threshold to consider a P candidate valid
P_SEARCH_MIN_RR_SEC = 0.5 # below this RR (>120 bpm) the P search is skipped

T_SEARCH_START_MS = 100   # closest limit to the R peak (ms after)
T_SEARCH_END_MS = 500     # furthest limit (ms after), adapted to next_rr
T_MIN_AMPLITUDE = 0.05

PR_LONG_MS = 200  # first-degree AV block
PR_SHORT_MS = 120 # pre-excitation or low junctional/atrial rhythm

RTC_LONG_MS = 460  # Long QTc (Fridericia's formula), indicative threshold
RTC_SHORT_MS = 350 # Short QTc, indicative threshold

MIN_CONSECUTIVE_PQT = 3   # consecutive beats required before declaring a sustained anomaly

# --- Debug/visualization of P and T waves ---
SHOW_PQT_MARKERS = True   # draws a marker on each found P/T
PRINT_PQT_PER_BEAT = True # prints PR/RT for each beat, not just sustained anomalies

PQT_APPLICABLE_LABELS = {'N', 'S'} # PR/RT calculated only on narrow and non-ectopic QRS beats

SERIAL_PORT = "/dev/cu.usbserial-0001"  # your actual port (verify with: ls /dev/cu.*)
BAUD_RATE = 115200

# --- TCP channel with ecg_realtime.py (connect) ---
# The script opens a CLIENT socket to these HOST/OUT_PORTs (see
# AD8232/ecg_realtime.py) and sends us JSON batches of ALREADY filtered samples
# (0.5-40Hz Butterworth filter applied on that end). They must match exactly.
ANALYZER_HOST = '127.0.0.1'
ANALYZER_PORT = 65432
