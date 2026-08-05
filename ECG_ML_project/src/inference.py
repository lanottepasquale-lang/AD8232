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

print(">>> VERSIONE DETECTOR: con discriminazione onda T e fusione picchi ravvicinati <<<")

os.makedirs(LOGS_DIR, exist_ok=True)

beat_model = keras.models.load_model(os.path.join(MODELS_DIR, 'ecg_model_cnn.keras'))
beat_le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
beat_feat_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))

rhythm_model = joblib.load(os.path.join(MODELS_DIR, 'rhythm_model_rf.pkl'))
rhythm_le = joblib.load(os.path.join(MODELS_DIR, 'rhythm_label_encoder.pkl'))

BEAT_COLORS = {'N': 'gray', 'S': 'orange', 'V': 'red', 'F': 'purple',
               'L': 'brown', 'R': 'brown', 'incerto': 'gold', 'artefatto': 'black'}
BEAT_INFO = {
    'N': ('normale', False),
    'S': ('sopraventricolare (estrasistole atriale)', True),
    'V': ('ventricolare (extrasistole ventricolare)', True),
    'F': ('fusione', True),
    'L': ("blocco di branca sinistra (bassa affidabilita')", True),
    'R': ("blocco di branca destra (bassa affidabilita')", True),
    'incerto': ('battito non classificato con sicurezza', True),
    'artefatto': ('probabile artefatto da movimento/rumore', True),
}
RHYTHM_INFO = {
    'N': 'normale',
    'AFIB': 'FIBRILLAZIONE ATRIALE',
    'incerto': "ritmo non classificato con sicurezza",
}
EVENT_DESCRIPTIONS = {
    'pausa': 'pausa nel ritmo',
    'segnale_scadente': "segnale di bassa qualita' (controllare elettrodi)",
    'elettrodo_scollegato': 'elettrodo scollegato (lead-off)',
}


class SessionLogger:
    """Scrive ogni evento su CSV in tempo reale, cosi' i dati restano
    consultabili anche dopo la chiusura del programma."""

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
    """Punto di aggancio per la lettura reale dei pin LO+/LO- dell'AD8232
    (lettura digitale su ESP32) quando si passera' dalla simulazione al
    segnale live. In simulazione da file MIT-BIH non esiste un segnale di
    lead-off, quindi restituisce sempre False: qui non scatta mai, ma la
    struttura e' gia' pronta per il collegamento hardware."""
    return False


def raw_block_is_flat(raw_block):
    """Controllo economico e informativo: una deviazione standard troppo
    bassa in un blocco grezzo indica quasi sempre un elettrodo scollegato o
    un ADC bloccato. Soglia calibrata su MIT-BIH: ricalibrare su hardware reale."""
    return np.std(raw_block) < RAW_FLAT_STD_THRESHOLD


class RealTimeQRSDetector:
    """Pan-Tompkins semplificato per streaming a blocchi, con:
    - inviluppo adattivo a decadimento lento
    - gestione corretta dei confini tra blocchi
    - search-back se passa troppo tempo dall'ultimo picco
    - fusione di picchi troppo ravvicinati (stesso QRS largo contato doppio)
    - discriminazione dell'onda T: un candidato debole entro una finestra
      ravvicinata dal QRS precedente viene scartato come onda T, non contato
      come battito (passaggio del Pan-Tompkins originale, omesso nelle
      versioni precedenti di questo detector)."""

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

            # Troppo vicino al picco precedente per essere un battito distinto:
            # probabile doppio conteggio dello stesso QRS largo. Tiene il
            # picco con energia maggiore, scarta l'altro.
            if time_since_last < self.min_physiological_rr:
                if peaks_found and val > combined[peaks_found[-1] - start_global]:
                    peaks_found[-1] = global_i
                    self.last_peak_global = global_i
                    self.last_confirmed_val = val
                continue

            # Candidato entro la finestra tipica di un'onda T, con energia
            # nettamente inferiore al QRS precedente: probabile onda T, non
            # un nuovo battito. Scartato, non unito.
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
    """Combina piu' livelli di analisi su ogni nuovo picco: regole su RR
    (tachi/bradi/pausa), classificatore di battito, classificatore di
    ritmo, pattern di bigeminismo/trigeminismo, carico di PVC nell'ultimo
    minuto, e approssimazione degli intervalli PR/RT(QT) tramite ricerca
    di onda P e onda T attorno al picco R. Gestisce il ritardo di un
    battito necessario per next_rr, esattamente come in utils.extract_beats
    usato nel training."""

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
                extra = 'possibile ventricolare' if v_count >= 2 else 'possibile sopraventricolare (es. SVTA)'
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
        if beat_result:
            peak_idx, (label, conf) = beat_result
            results['peak_idx'], results['beat'] = beat_result
            t_beat = peak_idx / FS

            self.recent_beat_labels.append(label)
            self.pattern_history.append(label)

            current_pattern = self._detect_pvc_pattern()
            if current_pattern != self.last_pattern:
                if current_pattern:
                    results['events'].append(('inizio_' + current_pattern, None, ''))
                elif self.last_pattern:
                    results['events'].append(('fine_' + self.last_pattern, None, ''))
                self.last_pattern = current_pattern

            self.pvc_window.append((t_beat, label == 'V'))
            while self.pvc_window and t_beat - self.pvc_window[0][0] > PVC_BURDEN_WINDOW_SEC:
                self.pvc_window.popleft()
            if len(self.pvc_window) >= PVC_BURDEN_MIN_BEATS:
                burden = sum(1 for _, is_v in self.pvc_window if is_v) / len(self.pvc_window)
                if burden >= PVC_BURDEN_ALERT_FRACTION and not self.pvc_alert_active:
                    self.pvc_alert_active = True
                    results['events'].append(('inizio_pvc_elevato', burden * 100, ''))
                elif burden < PVC_BURDEN_ALERT_FRACTION and self.pvc_alert_active:
                    self.pvc_alert_active = False
                    results['events'].append(('fine_pvc_elevato', burden * 100, ''))

            if label in PQT_APPLICABLE_LABELS:
                pqt = self._detect_pr_qt(len(self.peaks) - 2, wideband_signal)
            else:
                pqt = None
            results['pqt'] = pqt

            pr = pqt['pr_ms'] if pqt else None
            if pr is not None:
                if pr > PR_LONG_MS:
                    self.pr_long_streak += 1
                    self.pr_short_streak = 0
                elif pr < PR_SHORT_MS:
                    self.pr_short_streak += 1
                    self.pr_long_streak = 0
                else:
                    self.pr_long_streak = 0
                    self.pr_short_streak = 0

                new_pr_state = self.pr_state
                if self.pr_long_streak >= MIN_CONSECUTIVE_PQT:
                    new_pr_state = 'prolungato'
                elif self.pr_short_streak >= MIN_CONSECUTIVE_PQT:
                    new_pr_state = 'corto'
                elif self.pr_long_streak == 0 and self.pr_short_streak == 0:
                    new_pr_state = 'normale'

                if new_pr_state != self.pr_state:
                    if new_pr_state != 'normale':
                        results['events'].append(('inizio_pr_' + new_pr_state, pr, ''))
                    elif self.pr_state != 'normale':
                        results['events'].append(('fine_pr_' + self.pr_state, pr, ''))
                    self.pr_state = new_pr_state

            rtc = pqt['rtc_ms'] if pqt else None
            if rtc is not None:
                if rtc > RTC_LONG_MS:
                    self.rt_long_streak += 1
                    self.rt_short_streak = 0
                elif rtc < RTC_SHORT_MS:
                    self.rt_short_streak += 1
                    self.rt_long_streak = 0
                else:
                    self.rt_long_streak = 0
                    self.rt_short_streak = 0

                new_rt_state = self.rt_state
                if self.rt_long_streak >= MIN_CONSECUTIVE_PQT:
                    new_rt_state = 'prolungato'
                elif self.rt_short_streak >= MIN_CONSECUTIVE_PQT:
                    new_rt_state = 'corto'
                elif self.rt_long_streak == 0 and self.rt_short_streak == 0:
                    new_rt_state = 'normale'

                if new_rt_state != self.rt_state:
                    if new_rt_state != 'normale':
                        results['events'].append(('inizio_rt_' + new_rt_state, rtc, ''))
                    elif self.rt_state != 'normale':
                        results['events'].append(('fine_rt_' + self.rt_state, rtc, ''))
                    self.rt_state = new_rt_state

        return results

    def _detect_pvc_pattern(self):
        if len(self.pattern_history) < PATTERN_HISTORY_LEN:
            return None
        labels = list(self.pattern_history)

        for phase in (0, 1):
            v_pos = {phase, phase + 2, phase + 4}
            if all(labels[i] == 'V' for i in v_pos) and all(labels[i] != 'V' for i in set(range(6)) - v_pos):
                return 'bigeminismo'

        for phase in (0, 1, 2):
            v_pos = {phase, phase + 3}
            if all(labels[i] == 'V' for i in v_pos) and all(labels[i] != 'V' for i in set(range(6)) - v_pos):
                return 'trigeminismo'

        return None

    def _classify_beat(self, i, wideband_signal):
        peak = self.peaks[i]
        if peak - PRE < 0 or peak + POST >= len(wideband_signal):
            return None

        rr = np.diff(self.peaks) / FS
        prev_rr = rr[i - 1] if i > 0 else np.nan
        next_rr = rr[i] if i < len(rr) else np.nan
        local_window = rr[max(0, i - 5):i]
        local_avg_rr = np.mean(local_window) if len(local_window) > 0 else prev_rr

        if np.isnan(prev_rr):
            prev_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(next_rr):
            next_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(local_avg_rr):
            local_avg_rr = 0.8
        ratio = prev_rr / (local_avg_rr + 1e-8)

        beat = wideband_signal[peak - PRE: peak + POST]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        features = beat_feat_scaler.transform([[prev_rr, next_rr, local_avg_rr, ratio]])

        proba = beat_model({
            'signal': tf.convert_to_tensor(beat[None, :, None], dtype=tf.float32),
            'rr_features': tf.convert_to_tensor(features, dtype=tf.float32)
        }, training=False).numpy()[0]

        conf = float(np.max(proba))
        label = beat_le.inverse_transform([np.argmax(proba)])[0] if conf >= BEAT_CONF_THRESHOLD else 'incerto'
        return peak, (label, conf)

    def _detect_pr_qt(self, i, wideband_signal):
        """Approssima gli intervalli PR e RT (proxy del QT) cercando i
        picchi di onda P e onda T attorno al picco R gia' confermato. Sono
        misure picco-picco, non onset-offset come nello standard clinico
        rigoroso: da leggere come indicatori di tendenza, non come misura
        diagnostica precisa. Chiamato solo per battiti N/S (QRS stretto):
        su battiti ectopici a QRS largo (V/R/L) la morfologia distorta
        rende una ricerca 'massimo in finestra' inaffidabile, come in
        clinica reale, dove il QT si misura sui battiti sinusali, non
        sui PVC."""
        peak = self.peaks[i]
        rr = np.diff(self.peaks) / FS
        prev_rr = rr[i - 1] if i > 0 else None
        next_rr = rr[i] if i < len(rr) else None

        result = {'p_peak': None, 'pr_ms': None, 't_peak': None, 'rt_ms': None, 'rtc_ms': None}

        # --- Onda T ---
        t_start = peak + int(T_SEARCH_START_MS / 1000 * FS)
        max_end_ms = T_SEARCH_END_MS
        if next_rr is not None:
            max_end_ms = min(max_end_ms, next_rr * 1000 * 0.6)
        t_end = peak + int(max_end_ms / 1000 * FS)
        t_end = min(t_end, len(wideband_signal) - 1)

        if t_end > t_start:
            window = wideband_signal[t_start:t_end]
            if len(window) > 0 and np.max(window) > T_MIN_AMPLITUDE:
                t_offset = int(np.argmax(window))
                t_peak = t_start + t_offset
                result['t_peak'] = t_peak
                result['rt_ms'] = (t_peak - peak) / FS * 1000
                rr_for_correction = prev_rr if prev_rr else next_rr
                if rr_for_correction and rr_for_correction > 0:
                    rt_sec = result['rt_ms'] / 1000
                    result['rtc_ms'] = rt_sec / (rr_for_correction ** (1 / 3)) * 1000  # Fridericia

        # --- Onda P ---
        hr_ok = prev_rr is not None and prev_rr > P_SEARCH_MIN_RR_SEC
        rhythm_ok = self.last_rhythm_label != 'AFIB'
        if hr_ok and rhythm_ok:
            p_start = peak - int(P_SEARCH_START_MS / 1000 * FS)
            p_end = peak - int(P_SEARCH_END_MS / 1000 * FS)
            p_start = max(0, p_start)
            if p_end > p_start and p_end > 0:
                window = wideband_signal[p_start:p_end]
                if len(window) > 0 and np.max(window) > P_MIN_AMPLITUDE:
                    p_offset = int(np.argmax(window))
                    p_peak = p_start + p_offset
                    result['p_peak'] = p_peak
                    result['pr_ms'] = (peak - p_peak) / FS * 1000

        return result
    
    def _detect_pvc_pattern(self):
        """Bigeminismo: un V ogni due battiti. Trigeminismo: un V ogni tre.
        Controlla tutte le fasi possibili su una finestra di 6 etichette."""
        if len(self.pattern_history) < PATTERN_HISTORY_LEN:
            return None
        labels = list(self.pattern_history)

        for phase in (0, 1):
            v_pos = {phase, phase + 2, phase + 4}
            if all(labels[i] == 'V' for i in v_pos) and all(labels[i] != 'V' for i in set(range(6)) - v_pos):
                return 'bigeminismo'

        for phase in (0, 1, 2):
            v_pos = {phase, phase + 3}
            if all(labels[i] == 'V' for i in v_pos) and all(labels[i] != 'V' for i in set(range(6)) - v_pos):
                return 'trigeminismo'

        return None

    def _classify_beat(self, i, wideband_signal):
        peak = self.peaks[i]
        if peak - PRE < 0 or peak + POST >= len(wideband_signal):
            return None

        rr = np.diff(self.peaks) / FS
        prev_rr = rr[i - 1] if i > 0 else np.nan
        next_rr = rr[i] if i < len(rr) else np.nan
        local_window = rr[max(0, i - 5):i]
        local_avg_rr = np.mean(local_window) if len(local_window) > 0 else prev_rr

        if np.isnan(prev_rr):
            prev_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(next_rr):
            next_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(local_avg_rr):
            local_avg_rr = 0.8
        ratio = prev_rr / (local_avg_rr + 1e-8)

        beat = wideband_signal[peak - PRE: peak + POST]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        features = beat_feat_scaler.transform([[prev_rr, next_rr, local_avg_rr, ratio]])

        proba = beat_model({
            'signal': tf.convert_to_tensor(beat[None, :, None], dtype=tf.float32),
            'rr_features': tf.convert_to_tensor(features, dtype=tf.float32)
        }, training=False).numpy()[0]

        conf = float(np.max(proba))
        label = beat_le.inverse_transform([np.argmax(proba)])[0] if conf >= BEAT_CONF_THRESHOLD else 'incerto'
        return peak, (label, conf)


def check_signal_quality(raw_block, t_end, logger, state):
    """Monitor informativo (non blocca la classificazione): segnala quando
    un blocco grezzo ha varianza sospetta, indicativa di un possibile
    elettrodo scollegato o ADC bloccato."""
    is_flat = raw_block_is_flat(raw_block)
    if is_flat and not state['quality_alert']:
        state['quality_alert'] = True
        print(f"t={t_end:6.2f}s  {EVENT_DESCRIPTIONS['segnale_scadente']}  [ANOMALIA]")
        logger.log(t_end, 'event', 'segnale_scadente', '', EVENT_DESCRIPTIONS['segnale_scadente'])
    elif not is_flat and state['quality_alert']:
        state['quality_alert'] = False
        print(f"t={t_end:6.2f}s  segnale tornato regolare")
        logger.log(t_end, 'event', 'segnale_regolare', '', '')


def check_lead_off(t_end, logger, state):
    """Gate reale (non solo informativo): se attivo, salta la rilevazione
    picchi per questo blocco. In simulazione non scatta mai."""
    off = check_lead_off_esp32()
    if off and not state['lead_off_alert']:
        state['lead_off_alert'] = True
        print(f"t={t_end:6.2f}s  {EVENT_DESCRIPTIONS['elettrodo_scollegato']}  [ANOMALIA]")
        logger.log(t_end, 'event', 'elettrodo_scollegato', '', EVENT_DESCRIPTIONS['elettrodo_scollegato'])
    elif not off and state['lead_off_alert']:
        state['lead_off_alert'] = False
        print(f"t={t_end:6.2f}s  elettrodo ricollegato")
        logger.log(t_end, 'event', 'elettrodo_ricollegato', '', '')
    return off


def handle_result(result, end_sample_time, logger, ax, state, prefix=''):
    if result is None:
        return

    t = result['peak_idx'] / FS if result['peak_idx'] is not None else end_sample_time

    if result['beat']:
        peak_idx, (label, conf) = result['peak_idx'], result['beat']
        desc, is_anomaly = BEAT_INFO.get(label, (label, True))
        state['beat_total_count'] += 1
        if is_anomaly:
            state['beat_anomaly_count'] += 1
        tag = "[ANOMALIA]" if is_anomaly else ""
        print(f"{prefix}t={t:6.2f}s  battito: {label} - {desc}  {tag}  confidenza {conf:.2f}")
        logger.log(t, 'beat', label, f"{conf:.2f}", f"{prefix}{desc}")

        state['scatter_x'].append(t)
        state['scatter_y'].append(state['signal'][peak_idx])
        state['scatter_c'].append(BEAT_COLORS.get(label, 'blue'))
        if is_anomaly:
            ax.annotate(label, (t, state['signal'][peak_idx]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=9, color='red', fontweight='bold', ha='center')

        # --- Visualizzazione diretta di P e T, indipendente dagli allarmi ---
        pqt = result.get('pqt')
        if pqt:
            p_str = f"P a {pqt['pr_ms']:.0f}ms prima" if pqt['p_peak'] is not None else "P non rilevata"
            t_str = f"T a {pqt['rt_ms']:.0f}ms dopo (RTc {pqt['rtc_ms']:.0f}ms)" if pqt['t_peak'] is not None else "T non rilevata"
            if PRINT_PQT_PER_BEAT:
                print(f"{prefix}         {p_str}  |  {t_str}")
            logger.log(t, 'pqt',
                       f"PR={pqt['pr_ms']:.0f}" if pqt['pr_ms'] is not None else "PR=nd",
                       f"RTc={pqt['rtc_ms']:.0f}" if pqt['rtc_ms'] is not None else "RTc=nd",
                       f"{prefix}beat_t={t:.2f}")

            if SHOW_PQT_MARKERS:
                if pqt['p_peak'] is not None:
                    tp = pqt['p_peak'] / FS
                    ax.scatter([tp], [state['signal'][pqt['p_peak']]],
                               s=18, color='deepskyblue', zorder=2, marker='v')
                if pqt['t_peak'] is not None:
                    tt = pqt['t_peak'] / FS
                    ax.scatter([tt], [state['signal'][pqt['t_peak']]],
                               s=18, color='seagreen', zorder=2, marker='^')

    for event_name, value, extra in result['events']:
        if event_name.startswith('inizio_') or event_name.startswith('fine_'):
            stato, azione = event_name.split('_', 1)
            etichetta = f"[{stato.upper()} {azione.upper().replace('_', ' ')}]"
            if azione in ('tachicardia', 'bradicardia'):
                state['current_rate_state'] = azione if stato == 'inizio' else 'normale'
                valore_str = f"{value:.0f} bpm"
            elif azione == 'pvc_elevato':
                valore_str = f"{value:.0f}% battiti V (ultimo minuto)"
            elif azione.startswith('pr_') or azione.startswith('rt_'):
                valore_str = f"{value:.0f} ms"
            else:
                valore_str = ''
            suffisso = f"  ({extra})" if extra else ""
            print(f"{prefix}t={t:6.2f}s  {etichetta}  {valore_str}{suffisso}")
            logger.log(t, 'event', event_name, valore_str, f"{prefix}{extra}".strip())
            if stato == 'inizio':
                ax.axvline(t, color='darkred', linestyle='--', linewidth=1, alpha=0.6)
        else:
            desc = EVENT_DESCRIPTIONS.get(event_name, event_name)
            detail = f" di {value:.2f}s" if event_name == 'pausa' else ''
            print(f"{prefix}t={t:6.2f}s  {desc}{detail}  [ANOMALIA]")
            logger.log(t, 'event', event_name,
                       f"{value:.2f}s" if event_name == 'pausa' else '', f"{prefix}{desc}".strip())
            ax.axvline(t, color='darkred', linestyle='--', linewidth=1, alpha=0.6)

    if result['rhythm']:
        rhythm_label, rhythm_conf = result['rhythm']
        if rhythm_label != state['current_rhythm']:
            state['current_rhythm'] = rhythm_label
            nome = RHYTHM_INFO.get(rhythm_label, rhythm_label)
            conf_str = f"{rhythm_conf:.2f}" if rhythm_conf is not None else "n/d"
            print(f"{prefix}t={t:6.2f}s  ritmo: {nome}  [CAMBIO RITMO]  confidenza {conf_str}")
            logger.log(t, 'rhythm', rhythm_label, conf_str, f"{prefix}{nome}")


def run_simulation(record_name='100', speed=4.0, seconds_to_show=6, start_sec=0):
    record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    raw_signal = record.p_signal[:, 0]
    wideband_signal = causal_bandpass(raw_signal, fs=FS)

    if start_sec > 0:
        start_sample = int(start_sec * FS)
        raw_signal = raw_signal[start_sample:]
        wideband_signal = wideband_signal[start_sample:]

    detector = RealTimeQRSDetector(fs=FS)
    analyzer = AnomalyAnalyzer()
    logger = SessionLogger()
    print(f"Log della sessione: {logger.path}")

    state = {
        'signal': wideband_signal,
        'scatter_x': [], 'scatter_y': [], 'scatter_c': [],
        'beat_anomaly_count': 0, 'beat_total_count': 0,
        'current_rhythm': 'N', 'current_rate_state': 'normale',
        'quality_alert': False, 'lead_off_alert': False,
    }

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], color='steelblue', linewidth=0.8)
    scat = ax.scatter([], [], s=40, zorder=3)
    ax.set_xlim(0, seconds_to_show)
    ax.set_ylim(np.min(wideband_signal) * 1.2, np.max(wideband_signal) * 1.2)
    ax.set_xlabel('secondi')
    title_base = f'ECG live (simulato) - record {record_name}'

    try:
        block_size = FS
        n_blocks = len(raw_signal) // block_size
        for b in range(n_blocks):
            start, end = b * block_size, (b + 1) * block_size
            raw_block = raw_signal[start:end]
            t_block_end = end / FS

            check_signal_quality(raw_block, t_block_end, logger, state)
            if check_lead_off(t_block_end, logger, state):
                continue

            for p in detector.process_block(raw_block):
                result = analyzer.new_peak(p, wideband_signal)
                handle_result(result, t_block_end, logger, ax, state, prefix=f'[{record_name}] ')

            t_now = end / FS
            t_min = max(0, t_now - seconds_to_show)
            idx_min = int(t_min * FS)
            line.set_data(np.arange(idx_min, end) / FS, wideband_signal[idx_min:end])
            ax.set_xlim(t_min, max(t_now, seconds_to_show))

            visible = [(x, y, c) for x, y, c in
                       zip(state['scatter_x'], state['scatter_y'], state['scatter_c']) if x >= t_min]
            if visible:
                xs, ys, cs = zip(*visible)
                scat.set_offsets(np.c_[xs, ys])
                scat.set_color(cs)

            ax.set_title(f"{title_base}  |  ritmo: {state['current_rhythm']}  |  "
                         f"frequenza: {state['current_rate_state']}  |  "
                         f"anomalie battito: {state['beat_anomaly_count']}/{state['beat_total_count']}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(block_size / FS / speed)
    finally:
        logger.close()
        print(f"\nLog salvato in: {logger.path}")

    plt.ioff()
    plt.show()


def run_demo_tour(segments=None, speed=6.0, seconds_to_show=6, segment_duration=20):
    """Scorre automaticamente una serie di segmenti presi da record diversi,
    ciascuno per un tempo limitato, per mostrare piu' tipi di anomalia in
    un'unica esecuzione con un unico log finale. Il grafico viene pulito
    (ax.cla()) all'inizio di ogni segmento: senza questo, marcatori P/T,
    linee verticali ed etichette del segmento precedente restano visibili
    e si sovrappongono a quelli del segmento nuovo, perche' ogni segmento
    riparte il proprio asse temporale da zero e le porzioni di asse x si
    sovrappongono da un segmento all'altro."""
    if segments is None:
        segments = [
            ('100', 0, 'Tracciato normale (baseline)'),
            ('106', 161, 'Tachicardia ventricolare'),
            ('234', 831, 'Tachicardia sopraventricolare (SVTA)'),
            ('200', 93, 'Tachicardia ventricolare (episodio 2)'),
        ]

    logger = SessionLogger()
    print(f"Log del tour: {logger.path}")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))

    try:
        for record_name, start_sec, description in segments:
            print(f"\n=== Segmento: record {record_name} - {description} (da t={start_sec}s) ===")
            logger.log(0, 'segment', record_name, '', description)

            record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
            raw_signal = record.p_signal[:, 0]
            wideband_signal = causal_bandpass(raw_signal, fs=FS)

            start_sample = int(start_sec * FS)
            raw_signal = raw_signal[start_sample:]
            wideband_signal = wideband_signal[start_sample:]

            max_samples = int(segment_duration * FS)
            raw_signal = raw_signal[:max_samples]
            wideband_signal = wideband_signal[:max_samples]

            # Pulizia completa del grafico: rimuove marcatori P/T, linee
            # verticali ed etichette lasciati dal segmento precedente
            ax.cla()
            line, = ax.plot([], [], color='steelblue', linewidth=0.8)
            scat = ax.scatter([], [], s=40, zorder=3)

            detector = RealTimeQRSDetector(fs=FS)
            analyzer = AnomalyAnalyzer()
            state = {
                'signal': wideband_signal,
                'scatter_x': [], 'scatter_y': [], 'scatter_c': [],
                'beat_anomaly_count': 0, 'beat_total_count': 0,
                'current_rhythm': 'N', 'current_rate_state': 'normale',
                'quality_alert': False, 'lead_off_alert': False,
            }

            ax.set_xlim(0, seconds_to_show)
            ax.set_ylim(np.min(wideband_signal) * 1.2, np.max(wideband_signal) * 1.2)
            ax.set_xlabel("secondi (dall'inizio del segmento)")
            title_base = f'{record_name} - {description}'

            block_size = FS
            n_blocks = len(raw_signal) // block_size
            for b in range(n_blocks):
                start, end = b * block_size, (b + 1) * block_size
                raw_block = raw_signal[start:end]
                t_block_end = end / FS

                check_signal_quality(raw_block, t_block_end, logger, state)
                if check_lead_off(t_block_end, logger, state):
                    continue

                for p in detector.process_block(raw_block):
                    result = analyzer.new_peak(p, wideband_signal)
                    handle_result(result, t_block_end, logger, ax, state, prefix=f'[{record_name}] ')

                t_now = end / FS
                t_min = max(0, t_now - seconds_to_show)
                idx_min = int(t_min * FS)
                line.set_data(np.arange(idx_min, end) / FS, wideband_signal[idx_min:end])
                ax.set_xlim(t_min, max(t_now, seconds_to_show))

                visible = [(x, y, c) for x, y, c in
                           zip(state['scatter_x'], state['scatter_y'], state['scatter_c']) if x >= t_min]
                if visible:
                    xs, ys, cs = zip(*visible)
                    scat.set_offsets(np.c_[xs, ys])
                    scat.set_color(cs)

                ax.set_title(f"{title_base}  |  ritmo: {state['current_rhythm']}  |  "
                             f"frequenza: {state['current_rate_state']}  |  "
                             f"anomalie: {state['beat_anomaly_count']}/{state['beat_total_count']}")

                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(block_size / FS / speed)
    finally:
        logger.close()
        print(f"\nLog completo del tour: {logger.path}")

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_demo_tour()
