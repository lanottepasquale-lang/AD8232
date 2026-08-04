import sys
import serial
import socket
import time
import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter, lfilter_zi
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# --- PARAMETRI STRUMENTALI ---
PORTA_IN = "COM3"       # INSERISCI LA TUA PORTA (es. COM3 su Windows o /dev/ttyUSB0 su Mac/Linux)
BAUD_RATE = 115200   
FS = 360  # Frequenza di campionamento in Hz
DT=1/FS          
N_BATCH=180
FINESTRA = 1800    #5 secondi di campioni (1800 campioni a 360 Hz)
HOST = '127.0.0.1'
PORTA_OUT = 65432

def trova_battiti(ecg_filtrato, fs):
    # 1. Definiamo un'altezza minima per considerare un picco valido.
    altezza_minima = 0.6 * np.max(ecg_filtrato)
    
    # 2. Definiamo la distanza temporale minima (periodo refrattario).
    # Fisiologicamente, la frequenza cardiaca umana supera raramente i 200 bpm.
    # 200 bpm = 3.33 battiti al secondo = intervallo minimo di ~0.3 secondi.
    distanza_minima = int(0.3 * fs) # Convertito in numero di campioni
    
    # 3. Il motore analitico di Scipy
    # Restituisce gli indici esatti in cui la pendenza si inverte da positiva a negativa
    picchi, _ = find_peaks(ecg_filtrato, height=altezza_minima, distance=distanza_minima)
    
    return picchi


# --- CONFIGURAZIONE DEL FILTRO ---
def progetta_filtro(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = progetta_filtro(0.5, 40.0, FS)

# --- INIZIALIZZAZIONE DELLO STATO DEL FILTRO ---
z_state = lfilter_zi(b, a)
primo_dato_ricevuto = False # Flag per la calibrazione iniziale

# --- INIZIALIZZAZIONE HARDWARE E MEMORIA ---
ser = serial.Serial(PORTA_IN, BAUD_RATE)
buffer_dati = np.zeros(FINESTRA)
asse_x_tempo = np.arange(FINESTRA) / FS  # Array statico per l'asse dei tempi (secondi)

storia_grezza = [] 
buffer_filtrato_plot = np.zeros(FINESTRA)

# --- SETUP RETE TCP (Client) ---
buffer_batch = [] # Serbatoio per i 180 campioni
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    print(f"[RETE] Connessione al motore neurale su {HOST}:{PORTA_OUT}...")
    client_socket.connect((HOST, PORTA_OUT))
    print("[RETE] Connesso con successo.")
except ConnectionRefusedError:
    print("[ERRORE CRITICO] Il server analizza.py non è in ascolto!")
    sys.exit(1) # Meglio fermare tutto se la rete neurale non è pronta

# --- SETUP INTERFACCIA GRAFICA (PyQtGraph) ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="Monitor ECG Real-Time")
win.resize(1000, 600)
win.setBackground('k') # Sfondo nero per imitare la carta medica

plot = win.addPlot(title="Segnale ECG (Filtro Passa-Banda 0.5 - 40 Hz)")
plot.setLabel('left', 'Ampiezza Filtrata', units='ADC Units')
plot.setLabel('bottom', 'Tempo', units='s')
plot.setYRange(-1500, 1500)
plot.setXRange(0, FINESTRA / FS)
plot.getAxis('bottom').setPen('w')
plot.getAxis('left').setPen('w')

# COSTRUZIONE DELLA GRIGLIA ECG STANDARD (Ticks statici)
asse_x = plot.getAxis('bottom')
asse_y = plot.getAxis('left')

durata_totale = FINESTRA / FS
# Ticks temporali: Quadrati grandi 0.2s, piccoli 0.04s
ticks_x_major = [(x, str(round(x, 1))) for x in np.arange(0, durata_totale + 0.1, 0.2)]
ticks_x_minor = [(x, '') for x in np.arange(0, durata_totale + 0.1, 0.04)]
asse_x.setTicks([ticks_x_major, ticks_x_minor])

# Ticks di ampiezza: arbitrari per un segnale ADC grezzo
ticks_y_major = [(y, str(y)) for y in np.arange(-1500, 1501, 500)]
ticks_y_minor = [(y, '') for y in np.arange(-1500, 1501, 100)]
asse_y.setTicks([ticks_y_major, ticks_y_minor])

plot.showGrid(x=True, y=True, alpha=0.4)
curva = plot.plot(pen=pg.mkPen('g', width=1.5)) # Tracciato verde

# Creiamo il testo iniziale in verde (formato RGB) e ancoriamo l'angolo in alto a sinistra
testo_bpm = pg.TextItem(text="BPM: --", color=(0, 255, 0), anchor=(0, 0))

# Impostiamo il font per renderlo più leggibile (opzionale ma consigliato)
font = testo_bpm.textItem.font()
font.setPointSize(14)
font.setBold(True)
testo_bpm.setFont(font)

# Posizionamento: coordinate basate sui tuoi assi (Tempo, Ampiezza)
# Avendo i ticks Y fino a 1500, lo posizioniamo in alto.
# Regola l'ascissa (es. 0.1) in base a dove vuoi che appaia sull'asse dei tempi.
testo_bpm.setPos(0.1, 1400) 

# Aggiungiamo l'oggetto vettoriale al grafico
plot.addItem(testo_bpm)

# --- MOTORE DI AGGIORNAMENTO E CONTROLLO ---
def aggiorna_grafico():
    global buffer_filtrato_plot, storia_grezza, z_state, primo_dato_ricevuto, testo_bpm
    
    try:
        nuovi_dati = []
        # 1. Svuotamento rapido del buffer seriale
        while ser.in_waiting > 0:
            dato_grezzo = ser.readline().decode('utf-8').strip()
            if dato_grezzo:
                valore = int(dato_grezzo)
                nuovi_dati.append(valore)
                storia_grezza.append(valore)
        
        # 2. Elaborazione matematica del chunk (se ci sono nuovi dati)
        if nuovi_dati:
            chunk = np.array(nuovi_dati)
            
            # Calibrazione dello stato iniziale al primissimo campione 
            # per azzerare il transiente di accensione del filtro IIR.
            if not primo_dato_ricevuto:
                z_state = z_state * chunk[0]
                primo_dato_ricevuto = True
            
            # Filtro causale (lfilter) in streaming. 
            # Prende in ingresso il chunk e lo stato precedente, restituisce chunk filtrato e nuovo stato.
            chunk_filtrato, z_state = lfilter(b, a, chunk, zi=z_state)

# --- NUOVO: LOGICA DI BATCHING E INVIO SOCKET ---
            # Aggiungiamo i nuovi dati filtrati al serbatoio (li convertiamo in lista per praticità)
            buffer_batch.extend(chunk_filtrato.tolist())
            
            # Ciclo while: matematicamente robusto nel caso in cui un blocco ritardi 
            # e arrivino più di 360 campioni tutti insieme.
            while len(buffer_batch) >= N_BATCH:
                # Estraiamo esattamente la finestra temporale richiesta (i primi 180)
                batch_da_inviare = buffer_batch[:N_BATCH]
                
                # Rimuoviamo i dati estratti dal serbatoio (FIFO)
                del buffer_batch[:N_BATCH] 
                
                # Serializziamo il tensore e inviamolo alla rete neurale
                try:
                    pacchetto = (json.dumps(batch_da_inviare) + '\n').encode('utf-8')
                    client_socket.sendall(pacchetto)
                except Exception as e:
                    print(f"[RETE] Errore di invio socket: {e}")
            # ------------------------------------------------

            # 3. Aggiornamento del buffer circolare per il plotting
            n_nuovi = len(chunk_filtrato)
            if n_nuovi >= FINESTRA:
                # Caso limite: sono arrivati più dati della dimensione della finestra
                buffer_filtrato_plot = chunk_filtrato[-FINESTRA:]
            else:
                # Shift a sinistra e inserimento in coda (FIFO)
                buffer_filtrato_plot[:-n_nuovi] = buffer_filtrato_plot[n_nuovi:]
                buffer_filtrato_plot[-n_nuovi:] = chunk_filtrato
            
            # --- ANALISI E TELEMETRIA SUL SEGNALE STABILE ---
            picchi = trova_battiti(buffer_filtrato_plot, FS)
            
            if len(picchi) >= 2:
                distanza_intervalli = np.diff(picchi) / FS  
                ultimo_intervallo = distanza_intervalli[-1]
                bpm_attuale = int(60 / ultimo_intervallo)
                
                anomalia = False
                if ultimo_intervallo >= 1.2 or ultimo_intervallo < 0.4 or bpm_attuale < 40 or bpm_attuale > 100:
                    anomalia = True
                    
                comando = f"{bpm_attuale},{int(anomalia)}\n"
                ser.write(comando.encode('utf-8'))

                testo_bpm.setText(f"BPM: {bpm_attuale}")

                if anomalia:
                    testo_bpm.setColor((255, 0, 0))  # Rosso in caso di anomalia (tachicardia/bradicardia/artefatti)
                else:
                    testo_bpm.setColor((0, 255, 0))  # Verde se il battito è fisiologico

            else:
                comando = "0,0\n"
                ser.write(comando.encode('utf-8'))

                testo_bpm.setText("BPM: --")
                testo_bpm.setColor((0, 255, 0))

            # 4. Rendering grafico dell'array filtrato aggiornato
            curva.setData(x=asse_x_tempo, y=buffer_filtrato_plot)
            
    except ValueError:
        pass

# --- SCHEDULAZIONE DEL TEMPO REAL-TIME ---
timer = QtCore.QTimer()
timer.timeout.connect(aggiorna_grafico)
timer.start(20) # Polling ogni 20 ms

# --- ESECUZIONE E SALVATAGGIO ---
try:
    sys.exit(app.exec_())
finally:
    timer.stop()
    ser.close()
    client_socket.close() # <-- CHIUSURA DEL SOCKET
    print("Acquisizione terminata. Porta seriale e socket rilasciati.")
    
    # Salvataggio dati post-processing
    array_grezzo = np.array(storia_grezza)
    if len(array_grezzo) > 33:
        array_filtrato = filtfilt(b, a, array_grezzo)
        dati_da_salvare = np.column_stack((array_grezzo, array_filtrato))
        np.savetxt("dati_ecg_definitivi.txt", dati_da_salvare, fmt="%d\t%.3f", header="Grezzo\tFiltrato", comments="")
        print(f"Dataset consolidato: scritti {len(array_grezzo)} campioni.")