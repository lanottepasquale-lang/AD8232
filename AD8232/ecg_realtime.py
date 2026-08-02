import sys
import serial
import numpy as np
from scipy.signal import butter, filtfilt,find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# --- PARAMETRI STRUMENTALI ---
PORTA = "COM3"       # INSERISCI LA TUA PORTA (es. COM3 su Windows o /dev/ttyUSB0 su Mac/Linux)
BAUD_RATE = 115200   
FS = 200             # Frequenza di campionamento in Hz
FINESTRA = 1000      #5 secondi di campioni (1000 campioni a 200 Hz)


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

# --- INIZIALIZZAZIONE HARDWARE E MEMORIA ---
ser = serial.Serial(PORTA, BAUD_RATE)
buffer_dati = np.zeros(FINESTRA)
asse_x_tempo = np.arange(FINESTRA) / FS  # Array statico per l'asse dei tempi (secondi)
storia_grezza = [] 

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

# --- MOTORE DI AGGIORNAMENTO E CONTROLLO ---
def aggiorna_grafico():
    global buffer_dati, storia_grezza
    
    try:
        # Svuota il buffer fisico e aggiorna la memoria RAM
        while ser.in_waiting > 0:
            dato_grezzo = ser.readline().decode('utf-8').strip()
            
            if dato_grezzo:
                valore = int(dato_grezzo)
                storia_grezza.append(valore)
                
                buffer_dati[:-1] = buffer_dati[1:]
                buffer_dati[-1] = valore
                
# Calcolo matematico del filtro
        buffer_filtrato = filtfilt(b, a, buffer_dati)

        # Analisi dei picchi (la topologia vettoriale)
        picchi = trova_battiti(buffer_filtrato, FS)
        
        # Validazione statistica: calcoliamo i BPM solo se abbiamo almeno 2 picchi
        if len(picchi) >= 2:
            distanza_intervalli = np.diff(picchi) / FS  
            
            # Per la telemetria real-time, isoliamo l'intervallo più recente 
            # (l'ultimo elemento dell'array) per avere una latenza minima
            ultimo_intervallo = distanza_intervalli[-1]
            bpm_attuale = int(60 / ultimo_intervallo) # Cast forzato a intero!
            
            # Valutazione clinica dell'anomalia sul battito corrente
            anomalia = False
            if ultimo_intervallo >= 1.2 or ultimo_intervallo < 0.4 or bpm_attuale < 40 or bpm_attuale > 100:
                anomalia = True
                
            # Compilazione e invio di un singolo pacchetto CSV pulito
            comando = f"{bpm_attuale},{int(anomalia)}\n"
            ser.write(comando.encode('utf-8'))
            
        else:
            # Condizione di sicurezza: segnale piatto, artefatti o elettrodi staccati.
            # Invio del segnale di silenziamento all'ESP32.
            comando = "0,0\n"
            ser.write(comando.encode('utf-8'))

        # Rendering grafico accelerato
        curva.setData(x=asse_x_tempo, y=buffer_filtrato)
        
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
    print("Acquisizione terminata. Porta seriale rilasciata.")
    
    # Salvataggio dati post-processing
    array_grezzo = np.array(storia_grezza)
    if len(array_grezzo) > 33:
        array_filtrato = filtfilt(b, a, array_grezzo)
        dati_da_salvare = np.column_stack((array_grezzo, array_filtrato))
        np.savetxt("dati_ecg_definitivi.txt", dati_da_salvare, fmt="%d\t%.3f", header="Grezzo\tFiltrato", comments="")
        print(f"Dataset consolidato: scritti {len(array_grezzo)} campioni.")