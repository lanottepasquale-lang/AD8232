import sys
import time
import serial
import numpy as np
from scipy.signal import butter, lfilter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

sys.path.insert(0, 'src')
from config import FS, SERIAL_PORT, BAUD_RATE
from inference import (RealTimeQRSDetector, AnomalyAnalyzer, SessionLogger,
                        BEAT_INFO, RHYTHM_INFO, EVENT_DESCRIPTIONS,
                        check_signal_quality, check_lead_off)

FINESTRA_SECONDI = 5
FINESTRA = FINESTRA_SECONDI * FS


class StreamingBandpass:
    """Filtro causale a stato persistente (lfilter con zi), da usare al
    posto di filtfilt in un ciclo live: filtfilt richiederebbe conoscere
    campioni futuri, impossibile per definizione con dati che arrivano in
    questo istante, e reintrodurrebbe instabilita' proprio sugli ultimi
    campioni se riapplicato ripetutamente su un buffer scorrevole.
    Non esiste in inference.py: la' causal_bandpass e' pensata per essere
    chiamata una sola volta su un array gia' completo (la simulazione da
    file MIT-BIH), non blocco per blocco in streaming."""

    def __init__(self, fs, low=0.5, high=40):
        self.b, self.a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def process(self, raw_block):
        filtered, self.zi = lfilter(self.b, self.a, raw_block, zi=self.zi)
        return filtered


def gestisci_risultato(result, t_now, logger, curva_anomalie_x, curva_anomalie_y,
                        curva_anomalie_colori, state):
    """Equivalente di handle_result in inference.py, ma per PyQtGraph
    invece di matplotlib: stampa/logga allo stesso modo, ma non disegna
    su un Axes (handle_result non e' riusabile qui per quel motivo)."""
    if result is None:
        return

    if result['beat']:
        peak_idx, (label, conf) = result['beat']
        desc, is_anomaly = BEAT_INFO.get(label, (label, True))
        state['beat_total_count'] += 1
        if is_anomaly:
            state['beat_anomaly_count'] += 1
        tag = "[ANOMALIA]" if is_anomaly else ""
        print(f"t={t_now:6.1f}s  battito: {label} - {desc}  {tag}  confidenza {conf:.2f}")
        logger.log(t_now, 'beat', label, f"{conf:.2f}", desc)
        if is_anomaly:
            curva_anomalie_x.append(t_now)
            curva_anomalie_y.append(state['ultimo_valore_filtrato'])
            curva_anomalie_colori.append('r')

    for event_name, value, extra in result['events']:
        if event_name.startswith('inizio_') or event_name.startswith('fine_'):
            stato, azione = event_name.split('_', 1)
            if azione in ('tachicardia', 'bradicardia'):
                state['current_rate_state'] = azione if stato == 'inizio' else 'normale'
                valore_str = f"{value:.0f} bpm"
            else:
                valore_str = f"{value:.0f}" if value is not None else ''
            print(f"t={t_now:6.1f}s  [{stato.upper()} {azione.upper()}]  {valore_str}  ({extra})")
            logger.log(t_now, 'event', event_name, valore_str, extra)
        else:
            desc = EVENT_DESCRIPTIONS.get(event_name, event_name)
            print(f"t={t_now:6.1f}s  {desc}  [ANOMALIA]")
            logger.log(t_now, 'event', event_name, '', desc)

    if result['rhythm']:
        rhythm_label, rhythm_conf = result['rhythm']
        if rhythm_label != state['current_rhythm']:
            state['current_rhythm'] = rhythm_label
            nome = RHYTHM_INFO.get(rhythm_label, rhythm_label)
            print(f"t={t_now:6.1f}s  ritmo: {nome}  [CAMBIO RITMO]")
            logger.log(t_now, 'rhythm', rhythm_label,
                       f"{rhythm_conf:.2f}" if rhythm_conf else '', nome)


# --- Connessione seriale: UNICA, aperta una sola volta ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)
ser.reset_input_buffer()

# --- Filtro, buffer, detector, analyzer, log ---
bandpass = StreamingBandpass(fs=FS)
buffer_dati = []
detector = RealTimeQRSDetector(fs=FS)
analyzer = AnomalyAnalyzer()
logger = SessionLogger()
print(f"Log della sessione: {logger.path}")

state = {
    'beat_anomaly_count': 0, 'beat_total_count': 0,
    'current_rhythm': 'N', 'current_rate_state': 'normale',
    'quality_alert': False, 'lead_off_alert': False,
    'ultimo_valore_filtrato': 0,
}
anomalie_x, anomalie_y, anomalie_colori = [], [], []

# --- UI PyQtGraph ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="ECG Monitor - live")
win.resize(1000, 600)
win.setBackground('k')

plot = win.addPlot(title="ECG live")
plot.setLabel('left', 'Ampiezza filtrata')
plot.setLabel('bottom', 'Tempo', units='s')
plot.showGrid(x=True, y=True, alpha=0.4)
curva = plot.plot(pen=pg.mkPen('g', width=1.5))
marker_anomalie = pg.ScatterPlotItem(size=10, brush='r')
plot.addItem(marker_anomalie)


def aggiorna():
    nuovi_valori = []
    try:
        while ser.in_waiting > 0:
            riga = ser.readline().decode('utf-8', errors='ignore').strip()
            if riga:
                nuovi_valori.append(float(riga.split(',')[0]))
    except ValueError:
        pass

    if not nuovi_valori:
        return

    raw_block = np.array(nuovi_valori)
    filtered_block = bandpass.process(raw_block)
    buffer_dati.extend(filtered_block.tolist())
    state['ultimo_valore_filtrato'] = filtered_block[-1]
    t_now = len(buffer_dati) / FS

    check_signal_quality(raw_block, t_now, logger, state)
    if not check_lead_off(t_now, logger, state):
        for p_offset in detector.process_block(raw_block):
            p_absolute = len(buffer_dati) - len(filtered_block) + p_offset
            result = analyzer.new_peak(p_absolute, buffer_dati)
            gestisci_risultato(result, t_now, logger, anomalie_x, anomalie_y, anomalie_colori, state)

    visibile = buffer_dati[-FINESTRA:]
    asse_tempo = (np.arange(len(visibile)) + max(0, len(buffer_dati) - FINESTRA)) / FS
    curva.setData(x=asse_tempo, y=visibile)

    t_min = max(0, t_now - FINESTRA_SECONDI)
    marker_x = [x for x in anomalie_x if x >= t_min]
    marker_y = [y for x, y in zip(anomalie_x, anomalie_y) if x >= t_min]
    marker_anomalie.setData(x=marker_x, y=marker_y)

    plot.setTitle(f"ritmo: {state['current_rhythm']}  |  frequenza: {state['current_rate_state']}  |  "
                  f"anomalie: {state['beat_anomaly_count']}/{state['beat_total_count']}")


timer = QtCore.QTimer()
timer.timeout.connect(aggiorna)
timer.start(20)

try:
    sys.exit(app.exec_())
finally:
    timer.stop()
    ser.close()
    logger.close()
    print(f"Log salvato in: {logger.path}")