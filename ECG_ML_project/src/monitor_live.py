import sys
import socket
import json
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

sys.path.insert(0, 'src')
from config import FS, ANALYZER_HOST, ANALYZER_PORT
from inference import (RealTimeQRSDetector, AnomalyAnalyzer, SessionLogger,
                        BEAT_INFO, RHYTHM_INFO, EVENT_DESCRIPTIONS,
                        check_signal_quality, check_lead_off)

FINESTRA_SECONDI = 5
FINESTRA = FINESTRA_SECONDI * FS

# NOTA ARCHITETTURA (cambiata rispetto alla versione precedente):
# non apriamo piu' noi la porta seriale. E' ecg_realtime.py (del collega)
# che possiede la seriale, applica il suo filtro Butterworth 0.5-40Hz e ci
# manda i campioni GIA' filtrati via TCP, in batch JSON da 180 campioni
# (vedi AD8232/ecg_realtime.py: buffer_batch, N_BATCH=180). Noi diventiamo
# un server TCP che sta in ascolto e riceve. Niente da filtrare qui: farlo
# di nuovo vorrebbe dire applicare il passa-banda due volte, distorcendo
# il segnale (specialmente il transiente del filtro IIR, gia' gestito una
# volta sola da lui con z_state).


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


# --- Server TCP: in ascolto sulla stessa porta a cui si connette
# ecg_realtime.py come client. Non bloccante: accept()/recv() non devono
# mai fermare il loop Qt, esattamente come prima non dovevano bloccarlo
# le letture seriali (stesso schema di polling, cambia solo la fonte). ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ANALYZER_HOST, ANALYZER_PORT))
server_socket.listen(1)
server_socket.setblocking(False)
print(f"In ascolto su {ANALYZER_HOST}:{ANALYZER_PORT} — avvia ora ecg_realtime.py sull'altro lato.")

client_conn = None   # diventa la connessione con ecg_realtime.py una volta accettata
recv_buffer = b""    # accumula byte grezzi tra una recv() e la successiva

# --- Buffer, detector, analyzer, log (invariati) ---
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
    global client_conn, recv_buffer

    # 1. Accetta la connessione da ecg_realtime.py quando arriva (una sola
    # volta: dopo la prima accept() client_conn resta valido finche' lui
    # non chiude la connessione).
    if client_conn is None:
        try:
            client_conn, addr = server_socket.accept()
            client_conn.setblocking(False)
            print(f"Connesso a ecg_realtime.py: {addr}")
        except BlockingIOError:
            return  # nessun client ancora in connessione, normale

    # 2. Legge tutti i byte disponibili in questo istante, senza bloccare.
    try:
        while True:
            chunk = client_conn.recv(65536)
            if not chunk:
                print("ecg_realtime.py ha chiuso la connessione.")
                client_conn = None
                return
            recv_buffer += chunk
    except BlockingIOError:
        pass  # nessun altro dato pendente in questo istante, e' normale

    # 3. Ogni riga e' un array JSON di campioni GIA' filtrati, terminato da
    # '\n' (vedi: pacchetto = json.dumps(batch_da_inviare) + '\n' in
    # ecg_realtime.py). Un recv() puo' consegnare piu' righe insieme, o
    # una riga a meta': per questo si accumula in recv_buffer e si
    # processano solo le righe complete.
    nuovi_valori = []
    while b'\n' in recv_buffer:
        linea, recv_buffer = recv_buffer.split(b'\n', 1)
        if not linea:
            continue
        try:
            nuovi_valori.extend(json.loads(linea.decode('utf-8')))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # pacchetto corrotto/troncato, scartato

    if not nuovi_valori:
        return

    # Dati gia' filtrati da ecg_realtime.py: nessun filtro da applicare qui.
    filtered_block = np.array(nuovi_valori, dtype=float)
    buffer_dati.extend(filtered_block.tolist())
    state['ultimo_valore_filtrato'] = filtered_block[-1]
    t_now = len(buffer_dati) / FS

    # NOTA: qui riceviamo solo il segnale gia' filtrato, non il grezzo.
    # check_signal_quality() valuta quindi la varianza del filtrato, non
    # del grezzo come nella versione precedente (serviva il vero raw_block
    # per replicare esattamente il controllo originale calibrato su
    # MIT-BIH) - da ricalibrare RAW_FLAT_STD_THRESHOLD di conseguenza.
    check_signal_quality(filtered_block, t_now, logger, state)
    if not check_lead_off(t_now, logger, state):
        for p_offset in detector.process_block(filtered_block):
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
    if client_conn is not None:
        client_conn.close()
    server_socket.close()
    logger.close()
    print(f"Log salvato in: {logger.path}")