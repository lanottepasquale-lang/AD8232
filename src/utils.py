import numpy as np
from scipy.signal import butter, filtfilt

AAMI_MAP = {
    'N': 'N', 'e': 'N', 'j': 'N',        # normale (L/R rimossi da qui)
    'L': 'L',                             # blocco di branca sinistra
    'R': 'R',                             # blocco di branca destra
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}
# Record ufficiali dello split DS1 (train) / DS2 (test) secondo de Chazal et al.
DS1 = ['101','106','108','109','111','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220',
       '223','230']
DS2 = ['100','103','105','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232',
       '233','234']

def bandpass_filter(sig, fs=360, low=0.5, high=40):
    """Filtro passa-banda per pulire il segnale ECG."""
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)


def extract_beats(signal, r_peaks, symbols, pre=130, post=110, fs=360):
    """Ritaglia la finestra attorno a ogni picco R e aggiunge feature basate
    sugli intervalli RR. Finestra pre allargata a 130 campioni (361 ms) per
    includere per intero l'onda P, che precede il QRS di circa 150-250 ms
    ed e' l'indicatore clinico principale dei battiti S."""
    X, y = [], []
    n = len(r_peaks)
    rr = np.diff(r_peaks) / fs

    for i in range(n):
        peak = r_peaks[i]
        sym = symbols[i]
        if sym not in AAMI_MAP:
            continue
        if peak - pre < 0 or peak + post >= len(signal):
            continue

        beat = signal[peak - pre: peak + post]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        prev_rr = rr[i - 1] if i > 0 else np.nan
        next_rr = rr[i] if i < n - 1 else np.nan
        local_window = rr[max(0, i - 5):i]
        local_avg_rr = np.mean(local_window) if len(local_window) > 0 else prev_rr

        if np.isnan(prev_rr):
            prev_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(next_rr):
            next_rr = local_avg_rr if not np.isnan(local_avg_rr) else 0.8
        if np.isnan(local_avg_rr):
            local_avg_rr = 0.8

        ratio = prev_rr / (local_avg_rr + 1e-8)

        features = np.array([prev_rr, next_rr, local_avg_rr, ratio])
        X.append(np.concatenate([beat, features]))
        y.append(AAMI_MAP[sym])

    return np.array(X), np.array(y)