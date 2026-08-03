import sys
import uselect
import time
from machine import Pin,ADC, Timer, PWM

led_rosso=Pin(14, Pin.OUT)
led_giallo=Pin(27, Pin.OUT)
led_verde=Pin(26, Pin.OUT)
led_rosso.value(0) #spento all' inizio
led_giallo.value(0) #spento all' inizio
led_verde.value(0) #spento all' inizio

buzzer=PWM(25)
buzzer.freq(1000)
buzzer.duty(0)

sensore_ecg = ADC(Pin(34))
sensore_ecg.init(atten=ADC.ATTN_11DB)

lo_p = Pin(32, Pin.IN)
lo_m = Pin(33, Pin.IN)

# 3. La funzione di interrupt che scatta a ogni "tic" del timer
def leggi_e_invia(timer):
    if lo_p.value() == 1 or lo_m.value() == 1:      
        # Se il segnale è fuori range, invia un valore speciale (0)
        led_giallo.value(1) # Accende il LED giallo per indicare che gli elettrodi sono staccati
        print(0)
    else:
        # Legge il valore (0-4095) e lo stampa sulla porta seriale (USB)
        valore = sensore_ecg.read_u16()
        print(valore)

# 4. Inizializzazione del Timer Hardware (Timer 0)
timer_campionamento = Timer(0)

# 5. Avvia il timer: freq=360 significa 360 Hz (360 campioni al secondo)
timer_campionamento.init(freq=360, mode=Timer.PERIODIC, callback=leggi_e_invia)


# --- SETUP RICEZIONE SERIALE ---
poller = uselect.poll()
poller.register(sys.stdin, uselect.POLLIN)

# --- VARIABILI DI STATO (Macchina a Stati Finiti) ---
buffer_seriale = ""      # Variabile stringa per accumulare i caratteri in arrivo
bpm_target = 0           # Valore dei BPM attualmente impostati
ultimo_toggle_ms = time.ticks_ms() #cronometro per il lampeggio dei led e buzzer
stato_led_rosso = 0   
stato_led_verde = 0  
anomalia=0       

# --- CICLO MAIN ---
while True:
    
    # 1. LETTURA SERIALE (Non bloccante)
    if poller.poll(0):
        char = sys.stdin.read(1) # Legge 1 byte
        
        if char == '\n':
            # Il PC ha terminato l'invio del pacchetto CSV
            try:
                # 1. Separazione spaziale dei dati
                dati_ricevuti = buffer_seriale.split(',')
                
                # 2. Controllo dimensionale (Integrità del pacchetto)
                if len(dati_ricevuti) == 2:
                    bpm_target = int(dati_ricevuti[0])
                    anomalia = int(dati_ricevuti[1])
                    
            except ValueError:
                pass # Ignora pacchetti corrotti da disturbi sul cavo USB

            # 3. Svuotamento dell'accumulatore (Fondamentale!)
            buffer_seriale = ""
        else:       
            buffer_seriale += char # Accumula i caratteri in arrivo

    # 2. LOGICA DI LAMPEGGIO ASINCRONA (Blink without Delay)
    if bpm_target > 0:
        semi_periodo_ms = int(30000 / bpm_target) # Calcolo del semi-periodo di oscillazione in millisecondi.
        
        if time.ticks_diff(time.ticks_ms(), ultimo_toggle_ms) >= semi_periodo_ms:
            if anomalia==1:
                # Inverte lo stato del LED
                stato_led_rosso = not stato_led_rosso
                led_rosso.value(stato_led_rosso)
                stato_led_verde = 0 
                led_verde.value(stato_led_verde) # Spegne il LED verde quando il rosso è acceso
                buzzer.duty(512*stato_led_rosso) # Accende il buzzer solo quando il LED è acceso

            else:
                stato_led_verde = not stato_led_verde
                led_verde.value(stato_led_verde)
                stato_led_rosso = 0
                led_rosso.value(stato_led_rosso) # Spegne il LED rosso quando il verde è acceso
                buzzer.duty(512*stato_led_verde) # Accende il buzzer solo quando il LED è acceso


            #Aggiorna il cronometro
            ultimo_toggle_ms = time.ticks_ms()