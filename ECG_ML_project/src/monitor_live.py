import sys
import socket
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtWidgets

sys.path.insert(0, 'src')
from config import FS, ANALYZER_HOST, ANALYZER_PORT
from inference import (RealTimeQRSDetector, AnomalyAnalyzer, SessionLogger,
                       BEAT_INFO, BEAT_COLORS, RHYTHM_INFO, EVENT_DESCRIPTIONS,
                       check_signal_quality, check_lead_off)

FINESTRA_SECONDI = 5
FINESTRA = FINESTRA_SECONDI * FS

# ARCHITECTURE NOTE (changed from the previous version):
# we no longer open the serial port ourselves. It is ecg_realtime.py (the colleague's script)
# that owns the serial port, applies its 0.5-40Hz Butterworth filter and
# sends us the ALREADY filtered samples via TCP, in JSON batches of 180 samples
# (see AD8232/ecg_realtime.py: buffer_batch, N_BATCH=180). We act as
# a listening TCP server. No filtering is required here: doing it
# again would mean applying the band-pass filter twice, thereby distorting
# the signal (especially the IIR filter transient, already handled
# once on that end with z_state).


def gestisci_risultato(result, t_now, logger, state):
    """Equivalent to handle_result in inference.py, but adapted for PyQtGraph
    rather than matplotlib. In addition to printing/logging, it now populates the same
    visual information as the original: a colored marker for the beat type
    (BEAT_COLORS) positioned on the true peak amplitude, triangles for P/T waves,
    dashed vertical lines for events - all accumulated in state['...'] and rendered in aggiorna()."""
    if result is None:
        return

    t = result['peak_idx'] / FS if result['peak_idx'] is not None else t_now

    if result['beat']:
        peak_idx, (label, conf) = result['peak_idx'], result['beat']
        desc, is_anomaly = BEAT_INFO.get(label, (label, True))
        state['beat_total_count'] += 1
        if is_anomaly:
            state['beat_anomaly_count'] += 1
        tag = "[ANOMALY]" if is_anomaly else ""
        print(f"t={t:6.1f}s  beat: {label} - {desc}  {tag}  confidence {conf:.2f}")
        logger.log(t, 'beat', label, f"{conf:.2f}", desc)

        # Marker on the R peak, color-coded by beat type - same
        # logic and palette as inference.py (BEAT_COLORS), positioned
        # on the actual signal amplitude at the peak point (not an approximated
        # value as in the previous version of this file).
        state['beat_x'].append(t)
        state['beat_y'].append(state['buffer_dati'][peak_idx])
        state['beat_color'].append(BEAT_COLORS.get(label, 'blue'))

        # --- P and T waves, independent of alarms (same logic as
        # inference.py: P = downward blue triangle, T = upward green triangle) ---
        pqt = result.get('pqt')
        if pqt:
            p_str = f"P at {pqt['pr_ms']:.0f}ms prior" if pqt['p_peak'] is not None else "P undetected"
            t_str = f"T at {pqt['rt_ms']:.0f}ms post (RTc {pqt['rtc_ms']:.0f}ms)" if pqt['t_peak'] is not None else "T undetected"
            print(f"         {p_str}  |  {t_str}")
            logger.log(t, 'pqt',
                       f"PR={pqt['pr_ms']:.0f}" if pqt['pr_ms'] is not None else "PR=nd",
                       f"RTc={pqt['rtc_ms']:.0f}" if pqt['rtc_ms'] is not None else "RTc=nd",
                       f"beat_t={t:.2f}")
            if pqt['p_peak'] is not None:
                state['p_x'].append(pqt['p_peak'] / FS)
                state['p_y'].append(state['buffer_dati'][pqt['p_peak']])
            if pqt['t_peak'] is not None:
                state['t_x'].append(pqt['t_peak'] / FS)
                state['t_y'].append(state['buffer_dati'][pqt['t_peak']])

    for event_name, value, extra in result['events']:
        if event_name.startswith('inizio_') or event_name.startswith('fine_'):
            stato, azione = event_name.split('_', 1)
            if azione in ('tachicardia', 'bradicardia'):
                state['current_rate_state'] = azione if stato == 'inizio' else 'normale'
                valore_str = f"{value:.0f} bpm"
            else:
                valore_str = f"{value:.0f}" if value is not None else ''
            print(f"t={t:6.1f}s  [{stato.upper()} {azione.upper()}]  {valore_str}  ({extra})")
            logger.log(t, 'event', event_name, valore_str, extra)
            if stato == 'inizio':
                state['event_lines'].append(t)
        else:
            desc = EVENT_DESCRIPTIONS.get(event_name, event_name)
            print(f"t={t:6.1f}s  {desc}  [ANOMALY]")
            logger.log(t, 'event', event_name, '', desc)
            state['event_lines'].append(t)

    if result['rhythm']:
        rhythm_label, rhythm_conf = result['rhythm']
        if rhythm_label != state['current_rhythm']:
            state['current_rhythm'] = rhythm_label
            nome = RHYTHM_INFO.get(rhythm_label, rhythm_label)
            print(f"t={t:6.1f}s  rhythm: {nome}  [RHYTHM CHANGE]")
            logger.log(t, 'rhythm', rhythm_label,
                       f"{rhythm_conf:.2f}" if rhythm_conf else '', nome)


# --- TCP Server: listening on the same port to which
# ecg_realtime.py connects as a client. Non-blocking: accept()/recv() must
# never halt the Qt loop, exactly as serial reads shouldn't have blocked it
# previously (same polling paradigm, only the data source changes). ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ANALYZER_HOST, ANALYZER_PORT))
server_socket.listen(1)
server_socket.setblocking(False)
print(f"Listening on {ANALYZER_HOST}:{ANALYZER_PORT} — now launch ecg_realtime.py on the remote end.")

client_conn = None   # becomes the connection with ecg_realtime.py once accepted
recv_buffer = b""    # accumulates raw bytes between consecutive recv() calls

# --- Buffer, detector, analyzer, log (unchanged) ---
buffer_dati = []
detector = RealTimeQRSDetector(fs=FS)
analyzer = AnomalyAnalyzer()
logger = SessionLogger()
print(f"Session log: {logger.path}")

state = {
    'beat_anomaly_count': 0, 'beat_total_count': 0,
    'current_rhythm': 'N', 'current_rate_state': 'normale',
    'quality_alert': False, 'lead_off_alert': False,
    'ultimo_valore_filtrato': 0,
    'buffer_dati': buffer_dati,  # reference to the same buffer, convenient for reading the true amplitude in markers
    # Plot markers (equivalent to scatter_x/y/c in inference.py,
    # but utilizing separate lists for beats/P/T instead of a single scatter plot,
    # because PyQtGraph manages distinct symbols with distinct ScatterPlotItems)
    'beat_x': [], 'beat_y': [], 'beat_color': [],
    'p_x': [], 'p_y': [],
    't_x': [], 't_y': [],
    'event_lines': [],  # timestamps (seconds) where a dashed vertical line should be drawn
}

# --- PyQtGraph UI ---
# --- PyQtGraph UI, same visual palette as inference.py (light background,
# steelblue trace, colored markers by beat type + P/T triangles) ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="ECG Monitor - live")
win.resize(1000, 600)
win.setBackground('w')

plot = win.addPlot(title="Live ECG")
plot.setLabel('left', 'Filtered Amplitude')
plot.setLabel('bottom', 'Time', units='s')
plot.showGrid(x=False, y=False)
curva = plot.plot(pen=pg.mkPen('steelblue', width=1))

# Markers on R peaks, colored by beat type (BEAT_COLORS): same
# function as the unified scatter plot in inference.py, but here the color is passed
# per-point to each setData() instead of a single fixed property.
marker_battiti = pg.ScatterPlotItem(size=10, symbol='o', pen=pg.mkPen('k', width=0.5))
plot.addItem(marker_battiti)

# Downward blue triangle = P wave (visually verified with a test
# render: the 't' symbol renders pointing downwards in PyQtGraph, not upwards
# as the path coordinates read in isolation would suggest - Qt
# inverts the Y axis between symbol space and screen space)
marker_p = pg.ScatterPlotItem(size=9, symbol='t', brush=pg.mkBrush('deepskyblue'), pen=pg.mkPen(None))
plot.addItem(marker_p)

# Upward green triangle = T wave ('t1' symbol, visually verified)
marker_t = pg.ScatterPlotItem(size=9, symbol='t1', brush=pg.mkBrush('seagreen'), pen=pg.mkPen(None))
plot.addItem(marker_t)

# Dashed vertical lines for events (tachy/brady/pause/pattern/pvc):
# managed as InfiniteLine objects dynamically added/removed because,
# unlike scatter plots, a single "setData" for multiple lines does not exist.
event_line_items = {}  # map of t (seconds) -> InfiniteLine object already on the plot


def aggiorna():
    global client_conn, recv_buffer

    # 1. Accepts the connection from ecg_realtime.py when it arrives (only
    # once: after the first accept() client_conn remains valid until it
    # closes the connection).
    if client_conn is None:
        try:
            client_conn, addr = server_socket.accept()
            client_conn.setblocking(False)
            print(f"Connected to ecg_realtime.py: {addr}")
        except BlockingIOError:
            return  # no client currently connecting, normal state

    # 2. Reads all currently available bytes, non-blocking.
    connessione_chiusa = False
    try:
        while True:
            chunk = client_conn.recv(65536)
            if not chunk:
                connessione_chiusa = True
                break
            recv_buffer += chunk
    except BlockingIOError:
        pass  # no other pending data at this moment, normal state

    # 3. Each line is a JSON array of ALREADY filtered samples, terminated by
    # '\n' (see: pacchetto = json.dumps(batch_da_inviare) + '\n' in
    # ecg_realtime.py). A recv() may yield multiple lines together, or
    # half a line: for this reason it accumulates in recv_buffer and
    # processes only complete lines.
    nuovi_valori = []
    while b'\n' in recv_buffer:
        linea, recv_buffer = recv_buffer.split(b'\n', 1)
        if not linea:
            continue
        try:
            nuovi_valori.extend(json.loads(linea.decode('utf-8')))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # corrupted/truncated packet, discarded

    if connessione_chiusa:
        print("ecg_realtime.py has closed the connection.")
        client_conn = None
        # NB: not an early return - nuovi_valori could already
        # contain the final batch, arrived completely during the
        # last recv() prior to EOF: it must be analyzed regardless.

    if not nuovi_valori:
        return

    # Data already filtered by ecg_realtime.py: no filter to apply here.
    filtered_block = np.array(nuovi_valori, dtype=float)
    buffer_dati.extend(filtered_block.tolist())
    state['ultimo_valore_filtrato'] = filtered_block[-1]
    t_now = len(buffer_dati) / FS

    # NOTE: here we only receive the already filtered signal, not the raw one.
    # check_signal_quality() therefore evaluates the variance of the filtered signal, not
    # the raw one as in the previous version (the true raw_block was required
    # to perfectly replicate the original check calibrated on
    # MIT-BIH) - RAW_FLAT_STD_THRESHOLD must be recalibrated accordingly.
    check_signal_quality(filtered_block, t_now, logger, state)
    if not check_lead_off(t_now, logger, state):
        for p_absolute in detector.process_block(filtered_block):
            # p_absolute is ALREADY an absolute index in the detector's internal
            # reference frame (global_index), which increments exactly by
            # len(block) on each call, in perfect lockstep with
            # buffer_dati - NO additional offset must be added.
            # Verified by cross-referencing with run_simulation() in inference.py
            # (the already validated offline pathway), which uses "p" directly:
            #     for p in detector.process_block(raw_block):
            #         result = analyzer.new_peak(p, wideband_signal)
            # The previous version of this file erroneously added
            # len(buffer_dati) - len(filtered_block) + p_offset, a double
            # count that caused the index to diverge unboundedly over
            # time (verified: up to 6696 sample error over 20s of
            # synthetic signal) - pre-existing bug, not introduced by the
            # TCP channel transition.
            result = analyzer.new_peak(p_absolute, buffer_dati)
            gestisci_risultato(result, t_now, logger, state)

    visibile = buffer_dati[-FINESTRA:]
    asse_tempo = (np.arange(len(visibile)) + max(0, len(buffer_dati) - FINESTRA)) / FS
    curva.setData(x=asse_tempo, y=visibile)

    t_min = max(0, t_now - FINESTRA_SECONDI)

    # Filtered view restricted to the visible window, WITHOUT modifying the lists
    # in state: those remain complete for the entire session, so at program end
    # the plot of the full session can be exported, not just
    # the last window displayed on screen.
    def _finestra(xs, *altre):
        i = 0
        while i < len(xs) and xs[i] < t_min:
            i += 1
        if i == 0:
            return (xs,) + altre
        return (xs[i:],) + tuple(l[i:] for l in altre)

    bx, by, bc = _finestra(state['beat_x'], state['beat_y'], state['beat_color'])
    marker_battiti.setData(x=bx, y=by, brush=[pg.mkBrush(c) for c in bc])

    px, py = _finestra(state['p_x'], state['p_y'])
    marker_p.setData(x=px, y=py)

    tx, ty = _finestra(state['t_x'], state['t_y'])
    marker_t.setData(x=tx, y=ty)

    # Event lines: remove those exiting the window, append the new ones
    # among those currently visible (state['event_lines'] remains comprehensive anyway)
    eventi_visibili = [t for t in state['event_lines'] if t >= t_min]
    for t_evento in list(event_line_items.keys()):
        if t_evento < t_min:
            plot.removeItem(event_line_items.pop(t_evento))
    for t_evento in eventi_visibili:
        if t_evento not in event_line_items:
            linea = pg.InfiniteLine(
                pos=t_evento, angle=90,
                pen=pg.mkPen('darkred', width=1, style=QtCore.Qt.DashLine))
            plot.addItem(linea)
            event_line_items[t_evento] = linea

    plot.setTitle(f"Live ECG Monitoring  |  rhythm: {state['current_rhythm']}  |  "
                  f"rate: {state['current_rate_state']}  |  "
                  f"anomalies: {state['beat_anomaly_count']}/{state['beat_total_count']}")


timer = QtCore.QTimer()
timer.timeout.connect(aggiorna)
timer.start(20)

try:
    sys.exit(app.exec())
finally:
    timer.stop()
    if client_conn is not None:
        client_conn.close()
    server_socket.close()

    # Saves an image of the ENTIRE session (full signal + all
    # accumulated markers, not just the final WINDOW_SECONDS window
    # displayed on screen), matching the CSV log timestamp so the two
    # files remain paired and easily retrievable together.
    try:
        if buffer_dati:
            curva.setData(x=np.arange(len(buffer_dati)) / FS, y=buffer_dati)
            marker_battiti.setData(
                x=state['beat_x'], y=state['beat_y'],
                brush=[pg.mkBrush(c) for c in state['beat_color']])
            marker_p.setData(x=state['p_x'], y=state['p_y'])
            marker_t.setData(x=state['t_x'], y=state['t_y'])

            for t_evento in list(event_line_items.keys()):
                plot.removeItem(event_line_items.pop(t_evento))
            for t_evento in state['event_lines']:
                linea = pg.InfiniteLine(
                    pos=t_evento, angle=90,
                    pen=pg.mkPen('darkred', width=1, style=QtCore.Qt.DashLine))
                plot.addItem(linea)

            plot.setTitle(f"Live ECG Monitoring (complete session)  |  "
                          f"rhythm: {state['current_rhythm']}  |  "
                          f"rate: {state['current_rate_state']}  |  "
                          f"anomalies: {state['beat_anomaly_count']}/{state['beat_total_count']}")
            plot.enableAutoRange()
            app.processEvents()  # forces redraw before capturing the image

            # Same name/timestamp as the log, only prefix and extension differ
            plot_path = logger.path.replace('ecg_log_', 'ecg_plot_').replace('.csv', '.png')
            esportatore = pg.exporters.ImageExporter(plot)
            # NOTE: for very extended sessions (hours), the entire signal is
            # nevertheless compressed into a single fixed-width image -
            # it technically remains complete (all beats/markers are present),
            # but visually illegible to the naked eye without zooming heavily
            # on the saved file. For sessions lasting a few minutes, this is adequate.
            esportatore.parameters()['width'] = max(2000, len(buffer_dati) // FS * 40)
            esportatore.export(plot_path)
            print(f"Complete session plot saved to: {plot_path}")
    except Exception as e:
        print(f"Unable to save complete session plot: {e}")

    logger.close()
    print(f"Log saved to: {logger.path}")
