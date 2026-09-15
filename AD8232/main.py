import sys
import uselect
import time
import network
import bluetooth
from machine import Pin, ADC, Timer, PWM

# --- 1. SPEGNIMENTO DEL WI-FI E BLUETOOTH ---
wlan_sta = network.WLAN(network.STA_IF)
wlan_sta.active(False)
wlan_ap = network.WLAN(network.AP_IF)
wlan_ap.active(False)

try:
    ble = bluetooth.BLE()
    ble.active(False)
except Exception as e:
    pass

print("Moduli RF disattivati. Ambiente pulito per l'ECG.")

# --- 2. SETUP HARDWARE ---
led_rosso = Pin(14, Pin.OUT)
led_giallo = Pin(27, Pin.OUT)
led_verde = Pin(26, Pin.OUT)
led_rosso.value(0)
led_giallo.value(0)
led_verde.value(0)

buzzer = PWM(Pin(25))
buzzer.freq(5000)
buzzer.duty(0)

sensore_ecg = ADC(Pin(34))
sensore_ecg.init(atten=ADC.ATTN_11DB)

lo_p = Pin(32, Pin.IN)
lo_m = Pin(33, Pin.IN)

# --- 3. VARIABILI DI STATO (Inizializzate PRIMA del Timer) ---
allarme_elettrodi = False 
buffer_seriale = ""      
bpm_target = 0           
ultimo_toggle_ms = time.ticks_ms() 
stato_led_rosso = 0   
stato_led_verde = 0  
anomalia = 0       

# --- 4. GESTIONE DELL'INTERRUPT ---
def leggi_e_invia(timer):
    global allarme_elettrodi, stato_led_verde, stato_led_rosso 
    
    # Valutazione booleana dello stato dei terminali
    anomalia_contatto = (lo_p.value() == 1) or (lo_m.value() == 1)
    
    if anomalia_contatto:      
        led_giallo.value(1)
        if not allarme_elettrodi:
            buzzer.duty(512) 
            allarme_elettrodi = True
    else:
        led_giallo.value(0) 
        
        if allarme_elettrodi:
            allarme_elettrodi = False
            # Verifica conservativa dello stato richiesto dal Main Loop
            if stato_led_verde == 1 or stato_led_rosso == 1:
                buzzer.duty(512)
            else:
                buzzer.duty(0)
            
        valore = sensore_ecg.read()

# --- 5. INIZIALIZZAZIONE TIMER HW ---
timer_campionamento = Timer(0)
timer_campionamento.init(freq=360, mode=Timer.PERIODIC, callback=leggi_e_invia)

# --- SETUP RICEZIONE SERIALE ---
poller = uselect.poll()
poller.register(sys.stdin, uselect.POLLIN)

# --- 6. CICLO MAIN ---
while True:
    
    # 1. LETTURA SERIALE (Non bloccante)
    if poller.poll(0):
        char = sys.stdin.read(1)
        
        if char == '\n':
            try:
                dati_ricevuti = buffer_seriale.split(',')
                if len(dati_ricevuti) == 2:
                    bpm_target = int(dati_ricevuti[0])
                    anomalia = int(dati_ricevuti[1])
            except ValueError:
                pass 
            buffer_seriale = ""
        else:       
            buffer_seriale += char 

    # 2. LOGICA DI LAMPEGGIO ASINCRONA
    if bpm_target > 0:
        semi_periodo_ms = int(30000 / bpm_target) 
        
        if time.ticks_diff(time.ticks_ms(), ultimo_toggle_ms) >= semi_periodo_ms:
            if anomalia == 1:
                # Cast esplicito a int per garantire l'integrità dei dati nella ISR
                stato_led_rosso = int(not stato_led_rosso)
                led_rosso.value(stato_led_rosso)
                stato_led_verde = 0 
                led_verde.value(stato_led_verde) 
                
                # Attiva il buzzer solo se non c'è già un allarme elettrodi in corso
                if not allarme_elettrodi:
                    buzzer.duty(512 * stato_led_rosso) 
            else:
                stato_led_verde = int(not stato_led_verde)
                led_verde.value(stato_led_verde)
                stato_led_rosso = 0
                led_rosso.value(stato_led_rosso) 
                
                if not allarme_elettrodi:
                    buzzer.duty(512 * stato_led_verde) 

            ultimo_toggle_ms = time.ticks_ms()
