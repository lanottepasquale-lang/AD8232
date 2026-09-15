import sys
import serial
import socket
import time
import json
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# --- PARAMETRI STRUMENTALI ---
PORTA_IN = "/dev/cu.usbserial-0001"       # Verifica la porta
BAUD_RATE = 115200   
FS = 360  # Frequenza di campionamento in Hz
DT = 1/FS          
N_BATCH = 180
FINESTRA = 1800    # 5 secondi di campioni
HOST = '127.0.0.1'
PORTA_OUT = 65432

# MODIFICA: fattore di correzione per il mismatch di scala ADC.
# Il firmware ESP32 (main.py) usa sensore_ecg.read_u16(), che riscala
# il valore nativo a 12 bit (0-4095) su un range a 16 bit (0-65535),
# moltiplicandolo per 16. Questo gonfia l'ampiezza del segnale filtrato
# fino a decine di migliaia di unita', molto oltre il range fisiologico
# atteso, causando il clipping visivo osservato nel plot (setYRange
# troppo stretto rispetto al segnale reale).
# Se in futuro il firmware viene aggiornato per usare read() (12 bit
# nativi) invece di read_u16(), impostare questo fattore a 1.0.
FATTORE_CORREZIONE_SCALA = 16.0

def trova_battiti(ecg_filtrato, fs):
    altezza_minima = 0.6 * np.max(ecg_filtrato)
    distanza_minima = int(0.3 * fs) 
    picchi, _ = find_peaks(ecg_filtrato, height=altezza_minima, distance=distanza_minima)
    return picchi

# --- CONFIGURAZIONE DEL FILTRO (Topologia SOS) ---
def progetta_filtro_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # La topologia SOS garantisce la stabilità numerica frazionando il polinomio
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

sos = progetta_filtro_sos(0.5, 40.0, FS)

# --- INIZIALIZZAZIONE DELLO STATO DEL FILTRO ---
z_state = sosfilt_zi(sos)
primo_dato_ricevuto = False 

# --- INIZIALIZZAZIONE HARDWARE E MEMORIA ---
ser = serial.Serial(PORTA_IN, BAUD_RATE)
buffer_dati = np.zeros(FINESTRA)
asse_x_tempo = np.arange(FINESTRA) / FS  

storia_grezza = [] 
buffer_filtrato_plot = np.zeros(FINESTRA)

# --- SETUP RETE TCP (Client) ---
buffer_batch = [] 
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    print(f"[RETE] Connessione al motore neurale su {HOST}:{PORTA_OUT}...")
    client_socket.connect((HOST, PORTA_OUT))
    print("[RETE] Connesso con successo.")
except ConnectionRefusedError:
    print("[ERRORE CRITICO] Il server non è in ascolto. Arresto del sistema.")
    sys.exit(1)

# --- SETUP INTERFACCIA GRAFICA ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="Monitor ECG Real-Time")
win.resize(1000, 600)
win.setBackground('k') 

plot = win.addPlot(title="Segnale ECG (Filtro Passa-Banda 0.5 - 40 Hz)")
plot.setLabel('left', 'Ampiezza Filtrata', units='ADC Units')
plot.setLabel('bottom', 'Tempo', units='s')

# MODIFICA: range Y ridotto da ±1500 a ±500, coerente con il segnale
# dopo la correzione di scala (FATTORE_CORREZIONE_SCALA). Valore di
# partenza plausibile: verificare con un print di min/max sui primi
# secondi di acquisizione reale e regolare se necessario.
plot.setYRange(-500, 500)
plot.setXRange(0, FINESTRA / FS)
plot.getAxis('bottom').setPen('w')
plot.getAxis('left').setPen('w')

asse_x = plot.getAxis('bottom')
asse_y = plot.getAxis('left')

durata_totale = FINESTRA / FS
ticks_x_major = [(x, str(round(x, 1))) for x in np.arange(0, durata_totale + 0.1, 0.2)]
ticks_x_minor = [(x, '') for x in np.arange(0, durata_totale + 0.1, 0.04)]
asse_x.setTicks([ticks_x_major, ticks_x_minor])

# MODIFICA: tick coerenti con il nuovo range ±500
ticks_y_major = [(y, str(y)) for y in np.arange(-150, 150, 25)]
ticks_y_minor = [(y, '') for y in np.arange(-150, 150, 5)]
asse_y.setTicks([ticks_y_major, ticks_y_minor])

plot.showGrid(x=True, y=True, alpha=0.4)
curva = plot.plot(pen=pg.mkPen('g', width=1.5)) 

testo_bpm = pg.TextItem(text="BPM: --", color=(0, 255, 0), anchor=(0, 0))
font = testo_bpm.textItem.font()
font.setPointSize(14)
font.setBold(True)
testo_bpm.setFont(font)
# MODIFICA: posizione verticale del testo BPM riportata dentro al
# nuovo range visibile (prima era 1400, fuori scala col nuovo ±500)
testo_bpm.setPos(0.1, 450)
plot.addItem(testo_bpm)

# --- MOTORE DI AGGIORNAMENTO ---
def aggiorna_grafico():
    global buffer_filtrato_plot, storia_grezza, z_state, primo_dato_ricevuto, testo_bpm
    
    try:
        nuovi_dati = []
        while ser.in_waiting > 0:
            dato_grezzo = ser.readline().decode('utf-8').strip()
            if dato_grezzo:
                # MODIFICA: riscalatura del valore grezzo per compensare
                # il mismatch introdotto da read_u16() sul firmware ESP32.
                valore = int(dato_grezzo) / FATTORE_CORREZIONE_SCALA
                nuovi_dati.append(valore)
                storia_grezza.append(valore)
        
        if nuovi_dati:
            chunk = np.array(nuovi_dati)
            
            # Calibrazione stato iniziale (condizione al contorno)
            if not primo_dato_ricevuto:
                z_state = z_state * chunk[0]
                primo_dato_ricevuto = True
            
            # Esecuzione filtro ricorsivo con topologia SOS
            chunk_filtrato, z_state = sosfilt(sos, chunk, zi=z_state)

            buffer_batch.extend(chunk_filtrato.tolist())
            
            while len(buffer_batch) >= N_BATCH:
                batch_da_inviare = buffer_batch[:N_BATCH]
                del buffer_batch[:N_BATCH] 
                
                try:
                    pacchetto = (json.dumps(batch_da_inviare) + '\n').encode('utf-8')
                    client_socket.sendall(pacchetto)
                except Exception as e:
                    print(f"[RETE] Errore di invio socket: {e}")

            n_nuovi = len(chunk_filtrato)
            if n_nuovi >= FINESTRA:
                buffer_filtrato_plot = chunk_filtrato[-FINESTRA:]
            else:
                buffer_filtrato_plot[:-n_nuovi] = buffer_filtrato_plot[n_nuovi:]
                buffer_filtrato_plot[-n_nuovi:] = chunk_filtrato
            
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
                    testo_bpm.setColor((255, 0, 0))  
                else:
                    testo_bpm.setColor((0, 255, 0))  

            else:
                comando = "0,0\n"
                ser.write(comando.encode('utf-8'))
                testo_bpm.setText("BPM: --")
                testo_bpm.setColor((0, 255, 0))

            curva.setData(x=asse_x_tempo, y=buffer_filtrato_plot)
            
    except ValueError:
        pass

timer = QtCore.QTimer()
timer.timeout.connect(aggiorna_grafico)
timer.start(20) 

try:
    sys.exit(app.exec_())
finally:
    timer.stop()
    ser.close()
    client_socket.close() 
    print("Acquisizione terminata. Porta seriale e socket rilasciati.")
    
    # --- POST-PROCESSING ---
    array_grezzo = np.array(storia_grezza)
    if len(array_grezzo) > 33:
        # Sostituito filtfilt con sosfiltfilt per coerenza matematica
        array_filtrato = sosfiltfilt(sos, array_grezzo)
        dati_da_salvare = np.column_stack((array_grezzo, array_filtrato))
        np.savetxt("dati_ecg_definitivi.txt", dati_da_salvare, fmt="%d\t%.3f", header="Grezzo\tFiltrato", comments="")
        print(f"Dataset consolidato: scritti {len(array_grezzo)} campioni.")
