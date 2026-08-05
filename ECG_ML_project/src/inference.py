# %%
import os
import time
import csv
from datetime import datetime
from collections import deque
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
import joblib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from config import *

print(">>> DETECTOR VERSION: with T-wave discrimination and closely spaced peak fusion <<<")

os.makedirs(LOGS_DIR, exist_ok=True)

beat_model = keras.models.load_model(os.path.join(MODELS_DIR, 'ecg_model_cnn.keras'))
beat_le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
beat_feat_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))

rhythm_model = joblib.load(os.path.join(MODELS_DIR, 'rhythm_model_rf.pkl'))
rhythm_le = joblib.load(os.path.join(MODELS_DIR, 'rhythm_label_encoder.pkl'))

BEAT_COLORS = {'N': 'gray', 'S': 'orange', 'V': 'red', 'F': 'purple',
               'L': 'brown', 'R': 'brown', 'incerto': 'gold', 'artefatto': 'black'}
BEAT_INFO = {
    'N': ('normal', False),
    'S': ('supraventricular (atrial extrasystole)', True),
    'V': ('ventricular (ventricular extrasystole)', True),
    'F': ('fusion', True),
    'L': ("left bundle branch block (low reliability)", True),
    'R': ("right bundle branch block (low reliability)", True),
    'incerto': ('beat not classified with certainty', True),
    'artefatto': ('probable motion/noise artifact', True),
}
RHYTHM_INFO = {
    'N': 'normal',
    'AFIB': 'ATRIAL FIBRILLATION',
    'incerto': "rhythm not classified with certainty",
}
EVENT_DESCRIPTIONS = {
    'pausa': 'rhythm pause',
    'segnale_scadente': "low quality signal (check electrodes)",
    'elettrodo_scollegato': 'disconnected electrode (lead-off)',
}


class SessionLogger:
    """Writes each event to CSV in real-time, ensuring data remains
    accessible even after the program is closed."""

    def __init__(self, logs_dir=LOGS_DIR):
        filename = f"ecg_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = os.path.join(logs_dir, filename)
        self.file = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t_sec', 'event_type', 'label', 'confidence', 'detail'])

    def log(self, t, event_type, label, confidence='', detail=''):
        self.writer.writerow([f"{t:.3f}", event_type, label, confidence, detail])
        self.file.flush()

    def close(self):
        self.file.close()


def causal_bandpass(sig, fs=360, low=0.5, high=40):
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return lfilter(b, a, sig)


def check_lead_off_esp32():
    """Integration point for the physical reading of the AD8232 LO+/LO- pins
    (digital read on ESP32) when transitioning from simulation to the live
    signal. In simulation from MIT-BIH files, a lead-off signal does not exist,
    thus it always returns False: it never triggers here, but the structure
    is already prepared for hardware connection."""
    return False


def raw_block_is_flat(raw_block):
    """Computationally inexpensive and informative heuristic check: an excessively low
    standard deviation in a raw block almost always indicates a disconnected
    electrode or a stalled ADC. Threshold calibrated on MIT-BIH: recalibrate
    on actual hardware."""
    return np.std(raw_block) < RAW_FLAT_STD_THRESHOLD


class RealTimeQRSDetector:
    """Simplified Pan-Tompkins for block streaming, featuring:
    - slowly decaying adaptive envelope
    - proper block boundary management
    - search-back protocol if excessive time elapses since the last peak
    - fusion of excessively close peaks (same wide QRS double-counted)
    - T-wave discrimination: a weak candidate within a proximate window
      from the preceding QRS is discarded as a T-wave, not counted
      as a beat (original Pan-Tompkins step, omitted in previous
      versions of this detector)."""

    def __init__(self, fs=360):
        self.fs = fs
        self.b, self.a = butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        self.win_size = int(0.15 * fs)
        self.refractory = int(0.2 * fs)
        self.tail_squared = np.zeros(self.win_size - 1)
        self.last_filtered_sample = 0.0
        self.running_max = None
        self.decay = 0.9995
        self.threshold_fraction = 0.35
        self.pending_val = None
        self.pending_global_index = None
        self.global_index = 0
        self.last_peak_global = -10 ** 9
        self.rr_estimate_samples = fs * 0.8
        self.min_physiological_rr = int(MIN_PHYSIOLOGICAL_RR_SEC * fs)
        self.t_wave_window = int(T_WAVE_WINDOW_SEC * fs)
        self.last_confirmed_val = None

    def process_block(self, raw_block):
        filtered, self.zi = lfilter(self.b, self.a, raw_block, zi=self.zi)
        extended = np.concatenate(([self.last_filtered_sample], filtered))
        diff = np.diff(extended)
        self.last_filtered_sample = filtered[-1]
        squared = diff ** 2

        ext_squared = np.concatenate([self.tail_squared, squared])
        integrated = np.convolve(ext_squared, np.ones(self.win_size) / self.win_size, mode='valid')
        self.tail_squared = ext_squared[-(self.win_size - 1):]

        if self.pending_val is not None:
            combined = np.concatenate(([self.pending_val], integrated))
            start_global = self.pending_global_index
        else:
            combined = integrated
            start_global = self.global_index

        peaks_found = []
        for local_i in range(1, len(combined) - 1):
            global_i = start_global + local_i
            val = combined[local_i]
            if self.running_max is None:
                self.running_max = val
            else:
                self.running_max = max(val, self.running_max * self.decay)
            base_threshold = self.threshold_fraction * self.running_max

            time_since_last = global_i - self.last_peak_global
            if time_since_last > QRS_SEARCHBACK_MULTIPLIER * self.rr_estimate_samples:
                threshold = base_threshold * QRS_SEARCHBACK_THRESHOLD_FACTOR
            else:
                threshold = base_threshold

            is_local_max = val > combined[local_i - 1] and val >= combined[local_i + 1]
            if not is_local_max or time_since_last < self.refractory:
                continue
            if val <= threshold:
                continue

            # Too close to the preceding peak to constitute a distinct beat:
            # probable double counting of the same wide QRS. Retains the
            # peak exhibiting higher energy, discards the other.
            if time_since_last < self.min_physiological_rr:
                if peaks_found and val > combined[peaks_found[-1] - start_global]:
                    peaks_found[-1] = global_i
                    self.last_peak_global = global_i
                    self.last_confirmed_val = val
                continue

            # Candidate within the standard T-wave window, exhibiting markedly
            # lower energy than the preceding QRS: probable T-wave, not
            # a novel beat. Discarded, not merged.
            if (time_since_last < self.t_wave_window
                    and self.last_confirmed_val is not None
                    and val < T_WAVE_ENERGY_RATIO * self.last_confirmed_val):
                continue

            interval = time_since_last
            self.rr_estimate_samples = 0.875 * self.rr_estimate_samples + 0.125 * interval
            self.last_peak_global = global_i
            self.last_confirmed_val = val
            peaks_found.append(global_i)

        self.pending_val = combined[-1]
        self.pending_global_index = start_global + len(combined) - 1
        self.global_index += len(integrated)
        return peaks_found


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


class AnomalyAnalyzer:
    """Integrates multi-level analysis on each newly detected peak: RR rules
    (tachy/brady/pause), beat classifier, rhythm classifier,
    bigeminy/trigeminy patterns, PVC burden over the last minute,
    and PR/RT(QT) interval approximation via P wave and T wave search
    surrounding the R peak. Manages a one-beat delay required for next_rr,
    identical to utils.extract_beats utilized during training."""

    def __init__(self):
        self.peaks = []
        self.rr_history = deque(maxlen=RHYTHM_WINDOW_BEATS)
        self.last_rhythm_label = 'N'
        self.last_rhythm_conf = None
        self.tachy_streak = 0
        self.brady_streak = 0
        self.rate_state = 'normale'
        self.recent_beat_labels = deque(maxlen=MIN_CONSECUTIVE_RATE)
        self.pattern_history = deque(maxlen=PATTERN_HISTORY_LEN)
        self.last_pattern = None
        self.pvc_window = deque()
        self.pvc_alert_active = False
        self.pr_long_streak = 0
        self.pr_short_streak = 0
        self.rt_long_streak = 0
        self.rt_short_streak = 0
        self.pr_state = 'normale'
        self.rt_state = 'normale'

    def refine_peak(self, raw_idx, wideband_signal, back_ms=140, fwd_ms=40):
        back = int(back_ms / 1000 * FS)
        fwd = int(fwd_ms / 1000 * FS)
        lo, hi = max(0, raw_idx - back), min(len(wideband_signal), raw_idx + fwd)
        if hi <= lo:
            return raw_idx
        return lo + int(np.argmax(wideband_signal[lo:hi]))

    def new_peak(self, raw_idx, wideband_signal):
        peak = self.refine_peak(raw_idx, wideband_signal)
        self.peaks.append(peak)

        results = {'peak_idx': None, 'beat': None, 'events': [], 'rhythm': None, 'pqt': None}
        if len(self.peaks) < 2:
            return None

        rr = (self.peaks[-1] - self.peaks[-2]) / FS
        self.rr_history.append(rr)
        bpm = 60 / rr if rr > 0 else 0

        if bpm > TACHY_BPM:
            self.tachy_streak += 1
            self.brady_streak = 0
        elif bpm < BRADY_BPM:
            self.brady_streak += 1
            self.tachy_streak = 0
        else:
            self.tachy_streak = 0
            self.brady_streak = 0

        new_state = self.rate_state
        if self.tachy_streak >= MIN_CONSECUTIVE_RATE:
            new_state = 'tachicardia'
        elif self.brady_streak >= MIN_CONSECUTIVE_RATE:
            new_state = 'bradicardia'
        elif self.tachy_streak == 0 and self.brady_streak == 0:
            new_state = 'normale'

        if new_state != self.rate_state:
            if new_state == 'tachicardia':
                v_count = sum(1 for l in self.recent_beat_labels if l == 'V')
                extra = 'possible ventricular' if v_count >= 2 else 'possible supraventricular (e.g., SVTA)'
                results['events'].append(('inizio_tachicardia', bpm, extra))
            elif new_state != 'normale':
                results['events'].append(('inizio_' + new_state, bpm, ''))
            elif self.rate_state != 'normale':
                results['events'].append(('fine_' + self.rate_state, bpm, ''))
            self.rate_state = new_state

        if rr > PAUSE_SEC:
            results['events'].append(('pausa', rr, ''))

        if len(self.rr_history) == RHYTHM_WINDOW_BEATS:
            feats = np.array([rr_features(list(self.rr_history))])
            proba = rhythm_model.predict_proba(feats)[0]
            conf = float(np.max(proba))
            if conf >= RHYTHM_CONF_THRESHOLD:
                self.last_rhythm_label = rhythm_le.inverse_transform([np.argmax(proba)])[0]
            else:
                self.last_rhythm_label = 'incerto'
            self.last_rhythm_conf = conf
        results['rhythm'] = (self.last_rhythm_label, self.last_rhythm_conf)

        beat_result = self._classify_beat(len(self.peaks) - 2, wideband_signal)
