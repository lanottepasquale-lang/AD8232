import sys
import socket
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtWidgets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, 'src')
from config import FS, ANALYZER_HOST, ANALYZER_PORT
from inference import (RealTimeQRSDetector, AnomalyAnalyzer, SessionLogger,
                        BEAT_INFO, BEAT_COLORS, RHYTHM_INFO, EVENT_DESCRIPTIONS,
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


def gestisci_risultato(result, t_now, logger, state):
    """Equivalente di handle_result in inference.py, ma per PyQtGraph
    invece di matplotlib. Oltre a stampare/loggare, ora popola le stesse
    informazioni visive dell'originale: pallino colorato per tipo di
    triangoli per le onde P/T, linee verticali tratteggiate per gli
    eventi - tutto accumulato in state['...'] e disegnato in aggiorna()."""
    if result is None:
        return

    t = result['peak_idx'] / FS if result['peak_idx'] is not None else t_now

    if result['beat']:
        peak_idx, (label, conf) = result['peak_idx'], result['beat']
        desc, is_anomaly = BEAT_INFO.get(label, (label, True))
        state['beat_total_count'] += 1
        if is_anomaly:
            state['beat_anomaly_count'] += 1
        tag = "[ANOMALIA]" if is_anomaly else ""
        print(f"t={t:6.1f}s  battito: {label} - {desc}  {tag}  confidenza {conf:.2f}")
        logger.log(t, 'beat', label, f"{conf:.2f}", desc)

        # Pallino sul picco R, colorato per tipo di battito - stessa
        # logica e stessa palette di inference.py (BEAT_COLORS), posizionato
        # sulla vera ampiezza del segnale nel punto del picco (non un valore
        # approssimato come nella versione precedente di questo file).
        state['beat_x'].append(t)
        state['beat_y'].append(state['buffer_dati'][peak_idx])
        state['beat_color'].append(BEAT_COLORS.get(label, 'blue'))

        # --- Onde P e T, indipendenti dagli allarmi (stessa logica di
        # inference.py: P = triangolo giu' azzurro, T = triangolo su verde) ---
        pqt = result.get('pqt')
        if pqt:
            p_str = f"P a {pqt['pr_ms']:.0f}ms prima" if pqt['p_peak'] is not None else "P non rilevata"
            t_str = f"T a {pqt['rt_ms']:.0f}ms dopo (RTc {pqt['rtc_ms']:.0f}ms)" if pqt['t_peak'] is not None else "T non rilevata"
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
            print(f"t={t:6.1f}s  {desc}  [ANOMALIA]")
            logger.log(t, 'event', event_name, '', desc)
            state['event_lines'].append(t)

    if result['rhythm']:
        rhythm_label, rhythm_conf = result['rhythm']
        if rhythm_label != state['current_rhythm']:
            state['current_rhythm'] = rhythm_label
            nome = RHYTHM_INFO.get(rhythm_label, rhythm_label)
            print(f"t={t:6.1f}s  ritmo: {nome}  [CAMBIO RITMO]")
            logger.log(t, 'rhythm', rhythm_label,
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
    'buffer_dati': buffer_dati,  # riferimento allo stesso buffer, comodo per leggere l'ampiezza vera nei marker
    # Marker per il grafico (equivalenti a scatter_x/y/c di inference.py,
    # ma con liste separate per battiti/P/T invece di un unico scatter,
    # perche' PyQtGraph gestisce simboli diversi con ScatterPlotItem diversi)
    'beat_x': [], 'beat_y': [], 'beat_color': [],
    'p_x': [], 'p_y': [],
    't_x': [], 't_y': [],
    'event_lines': [],  # tempi (secondi) in cui disegnare una linea verticale tratteggiata
}

# --- UI PyQtGraph ---
# --- UI PyQtGraph, stessa palette visiva di inference.py (sfondo chiaro,
# traccia steelblue, marker colorati per tipo di battito + triangoli P/T) ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="ECG Monitor - live")
win.resize(1000, 600)
win.setBackground('w')

plot = win.addPlot(title="ECG live")
plot.setLabel('left', 'Ampiezza filtrata')
plot.setLabel('bottom', 'Tempo', units='s')
plot.showGrid(x=False, y=False)
curva = plot.plot(pen=pg.mkPen('steelblue', width=1))

# Pallini sui picchi R, colore per tipo di battito (BEAT_COLORS): stesso
# ruolo dello scatter unico in inference.py, ma qui il colore viene passato
# per-punto ad ogni setData() invece di un'unica proprieta' fissa.
marker_battiti = pg.ScatterPlotItem(size=10, symbol='o', pen=pg.mkPen('k', width=0.5))
plot.addItem(marker_battiti)

# Triangolo giu' azzurro = onda P (verificato visivamente con un render di
# prova: il simbolo 't' rende verso il basso in PyQtGraph, non verso l'alto
# come suggerirebbero le coordinate del path lette in isolamento - Qt
# inverte l'asse Y tra spazio del simbolo e spazio schermo)
marker_p = pg.ScatterPlotItem(size=9, symbol='t', brush=pg.mkBrush('deepskyblue'), pen=pg.mkPen(None))
plot.addItem(marker_p)

# Triangolo su verde = onda T (simbolo 't1', verificato visivamente)
marker_t = pg.ScatterPlotItem(size=9, symbol='t1', brush=pg.mkBrush('seagreen'), pen=pg.mkPen(None))
plot.addItem(marker_t)

# Linee verticali tratteggiate per eventi (tachi/bradi/pausa/pattern/pvc):
# gestite come oggetti InfiniteLine aggiunti/rimossi dinamicamente perche',
# a differenza degli scatter, non esiste un unico "setData" per piu' linee.
event_line_items = {}  # mappa t (secondi) -> oggetto InfiniteLine gia' sul plot


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
    connessione_chiusa = False
    try:
        while True:
            chunk = client_conn.recv(65536)
            if not chunk:
                connessione_chiusa = True
                break
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

    if connessione_chiusa:
        print("ecg_realtime.py ha chiuso la connessione.")
        client_conn = None
        # NB: non un return anticipato - nuovi_valori potrebbe gia'
        # contenere l'ultimo batch, arrivato per intero proprio
        # nell'ultima recv() prima dell'EOF: va comunque analizzato.

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
        for p_absolute in detector.process_block(filtered_block):
            # p_absolute e' GIA' un indice assoluto nel riferimento interno
            # del detector (global_index), che cresce esattamente di
            # len(block) ad ogni chiamata, in perfetto lockstep con
            # buffer_dati - NON va sommato nessun offset aggiuntivo.
            # Confermato confrontando con run_simulation() in inference.py
            # (il percorso offline gia' validato), che usa "p" cosi' com'e':
            #     for p in detector.process_block(raw_block):
            #         result = analyzer.new_peak(p, wideband_signal)
            # La versione precedente di questo file sommava per errore
            # len(buffer_dati) - len(filtered_block) + p_offset, un doppio
            # conteggio che faceva divergere l'indice senza limite nel
            # tempo (verificato: fino a 6696 campioni di errore in 20s di
            # segnale sintetico) - bug preesistente, non introdotto dal
            # passaggio al canale TCP.
            result = analyzer.new_peak(p_absolute, buffer_dati)
            gestisci_risultato(result, t_now, logger, state)

    visibile = buffer_dati[-FINESTRA:]
    asse_tempo = (np.arange(len(visibile)) + max(0, len(buffer_dati) - FINESTRA)) / FS
    curva.setData(x=asse_tempo, y=visibile)

    t_min = max(0, t_now - FINESTRA_SECONDI)

    # Vista filtrata alla sola finestra visibile, SENZA modificare le liste
    # in state: quelle restano complete per tutta la sessione, cosi' a fine
    # programma si puo' esportare il grafico dell'intera sessione, non solo
    # dell'ultima finestra mostrata a schermo.
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

    # Linee evento: rimuovi quelle uscite dalla finestra, aggiungi le nuove
    # tra quelle visibili ora (state['event_lines'] resta comunque completo)
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

    plot.setTitle(f"Monitoraggio ECG live  |  ritmo: {state['current_rhythm']}  |  "
                  f"frequenza: {state['current_rate_state']}  |  "
                  f"anomalie: {state['beat_anomaly_count']}/{state['beat_total_count']}")


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

    # Salva l'intera sessione segmentandola in tracciati impaginati su PDF.
    # Questo approccio garantisce il mantenimento della scala temporale e
    # l'integrità vettoriale del segnale biopotenziale.
    try:
        if buffer_dati:
            # 1. Definizione dei parametri metrici e temporali
            SECONDI_PER_RIGA = 10  # standard per foglio orizzontale a 25 mm/s
            RIGHE_PER_PAGINA = 4   # 40 secondi totali per pagina (A4 landscape)
            
            n_campioni = len(buffer_dati)
            t_totale = n_campioni / FS
            t_array = np.arange(n_campioni) / FS
            
            # Generazione della path per il referto PDF
            pdf_path = logger.path.replace('ecg_log_', 'ecg_report_').replace('.csv', '.pdf')
            print(f"Generazione del referto vettoriale PDF in corso: {pdf_path}")
            
            with PdfPages(pdf_path) as pdf:
                n_segmenti = int(np.ceil(t_totale / SECONDI_PER_RIGA))
                
                for riga_idx in range(n_segmenti):
                    # Inizializzazione di una nuova pagina (figura)
                    if riga_idx % RIGHE_PER_PAGINA == 0:
                        fig, axes = plt.subplots(RIGHE_PER_PAGINA, 1, figsize=(11.69, 8.27))
                        titolo = (f"Monitoraggio ECG | Ritmo: {state['current_rhythm']} | "
                                  f"Frequenza: {state['current_rate_state']} | "
                                  f"Anomalie: {state['beat_anomaly_count']}/{state['beat_total_count']}")
                        fig.suptitle(titolo, fontsize=12, fontweight='bold')
                        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.98, top=0.90, bottom=0.05)
                        
                        # Normalizzazione degli assi qualora si scegliesse 1 riga per pagina
                        if RIGHE_PER_PAGINA == 1:
                            axes = [axes]
                            
                    ax = axes[riga_idx % RIGHE_PER_PAGINA]
                    
                    # Calcolo dei limiti temporali e degli indici per lo slicing dell'epoca corrente
                    t_start = riga_idx * SECONDI_PER_RIGA
                    t_end = t_start + SECONDI_PER_RIGA
                    
                    idx_start = int(t_start * FS)
                    idx_end = min(int(t_end * FS), n_campioni)
                    
                    # Rendering del tracciato
                    ax.plot(t_array[idx_start:idx_end], buffer_dati[idx_start:idx_end], 
                            color='black', linewidth=1)
                    
                    # Generazione del reticolo millimetrato (simulazione carta ECG)
                    ax.set_xlim(t_start, t_end)
                    # Griglia maggiore (0.2s = 5mm) e minore (0.04s = 1mm)
                    ax.set_xticks(np.arange(t_start, t_end + 0.2, 0.2))
                    ax.set_xticks(np.arange(t_start, t_end + 0.04, 0.04), minor=True)
                    ax.grid(which='major', color='#ff9999', linestyle='-', linewidth=0.8)
                    ax.grid(which='minor', color='#ffcccc', linestyle='-', linewidth=0.3)
                    
                    # Vettorizzazione dei marker spaziali nell'intervallo temporale corrente
                    # 1. Battiti R
                    beat_mask = (np.array(state['beat_x']) >= t_start) & (np.array(state['beat_x']) < t_end)
                    if np.any(beat_mask):
                        bx = np.array(state['beat_x'])[beat_mask]
                        by = np.array(state['beat_y'])[beat_mask]
                        bcolors = np.array(state['beat_color'])[beat_mask]
                        ax.scatter(bx, by, c=bcolors, marker='o', zorder=3)

                    # 2. Onde P e T
                    p_mask = (np.array(state['p_x']) >= t_start) & (np.array(state['p_x']) < t_end)
                    if np.any(p_mask):
                        ax.scatter(np.array(state['p_x'])[p_mask], np.array(state['p_y'])[p_mask], 
                                   color='blue', marker='v', zorder=3, label='P')
                        
                    t_mask = (np.array(state['t_x']) >= t_start) & (np.array(state['t_x']) < t_end)
                    if np.any(t_mask):
                        ax.scatter(np.array(state['t_x'])[t_mask], np.array(state['t_y'])[t_mask], 
                                   color='green', marker='^', zorder=3, label='T')

                    # 3. Linee d'evento (Markers verticali)
                    for t_evento in state['event_lines']:
                        if t_start <= t_evento < t_end:
                            ax.axvline(x=t_evento, color='darkred', linestyle='--', linewidth=1)

                    ax.set_ylabel("Ampiezza", fontsize=8)
                    
                    # Manteniamo le label dell'asse temporale solo sull'ultimo asse della pagina
                    if riga_idx % RIGHE_PER_PAGINA == RIGHE_PER_PAGINA - 1 or riga_idx == n_segmenti - 1:
                        ax.set_xlabel("Tempo (s)", fontsize=9)
                    else:
                        ax.set_xticklabels([])

                    # Chiusura e scrittura della pagina in memoria se satura o se i dati sono terminati
                    if riga_idx % RIGHE_PER_PAGINA == RIGHE_PER_PAGINA - 1 or riga_idx == n_segmenti - 1:
                        # Rimozione termica degli assi vuoti nell'ultima pagina (incompleta)
                        if riga_idx == n_segmenti - 1 and (riga_idx % RIGHE_PER_PAGINA) < (RIGHE_PER_PAGINA - 1):
                            for empty_idx in range((riga_idx % RIGHE_PER_PAGINA) + 1, RIGHE_PER_PAGINA):
                                fig.delaxes(axes[empty_idx])
                        
                        pdf.savefig(fig)
                        plt.close(fig)
                        
            print(f"Plot multipagina della sessione completato: {pdf_path}")
            
            # Aggiorna anche l'interfaccia PyQt per l'ultimo frame
            app.processEvents()
            
    except Exception as e:
        print(f"Errore critico durante l'esportazione del referto PDF: {e}")

    logger.close()
    print(f"Log salvato in: {logger.path}")
