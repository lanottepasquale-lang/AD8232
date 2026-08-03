import numpy as np
import matplotlib.pyplot as plt



def dft(x):
    X=np.array([])
    for i in range(N):
        supp=0
        for j in range(N):
            supp += x[j]*np.exp(-2j*np.pi*i*j/N)
        X=np.append(X,supp)
    return X

def anti_dft(X):
    x=np.array([])
    for i in range(N):
        supp=0
        for j in range(N):
            supp += X[j]*np.exp(2j*np.pi*i*j/N)
        x=np.append(x,supp/N)
    return x


matrice = np.loadtxt("dati_ecg_definitivi.txt", delimiter="\t", skiprows=1)

datas=matrice[:, 0]
filtered = matrice[:, 1]

num_samples = len(datas)
freq= 200 #sono stati campionati 200 dati al secondo 
tau= 1/freq #periodo di campionamento
f_nyquist = freq / 2  #frequenza di Nyquist, massima frequenza che può essere rappresentata senza aliasing 

frequency_grid=np.arange(0, f_nyquist+1, 1) #inizio compreso, fine escluso, quindi arrivo perfetto a f_nyquist, non lo supero
frequency_grid=np.append(frequency_grid, np.arange(-f_nyquist+1, 0, 1)) 
temporal_grid=np.arange(0, num_samples * tau, tau)  #asse temporale

indici_ordinati = np.argsort(frequency_grid) 

datas_fourier = dft(datas)
spectrum = np.sqrt(datas_fourier * np.conj(datas_fourier))  # spettro di ampiezza

#plot spettro dati non filtrato

plt.plot(temportal_grid, datas, label="Dati non filtrati", color="#4c3bdd")
plt.grid(linewidth=0.5, alpha=0.5)
plt.title("Segnale non filtrato")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")
plt.show()

plt.plot(frequency_grid[indici_ordinati], spectrum[indici_ordinati], color="#4c3bdd")
plt.grid(linewidth=0.5, alpha=0.5)
plt.title("Spettro di ampiezza non filtrato")
plt.xlabel("Frequenza")
plt.ylabel("Ampiezza")
plt.show()


#implementazione filtro passa banda per dati ECG, elimino le frequenze al di fuori dell'intervallo 0.5-40 Hz, uso due approcci, uso esp decrescente e filtro butterworth
# esponential decay filter

datas_filtered_exp = np.array([])

for  freq in frequency_grid:
    if 0.5 <= abs(freq) <= 40:
        datas_filtered_exp = np.append(datas_filtered_exp, datas_fourier[np.where(frequency_grid == freq)])
    elif abs(freq) < 0.5:
        datas_filtered_exp = np.append(datas_filtered_exp, datas_fourier[np.where(frequency_grid == freq)] * np.exp(- 5* (0.5 - abs(freq))))

    else:
        datas_filtered_exp = np.append(datas_filtered_exp, datas_fourier[np.where(frequency_grid == freq)] * np.exp(- 5* (abs(freq) - 40)))

spectrum_filtered_exp = np.sqrt(datas_filtered_exp * np.conj(datas_filtered_exp))  # spettro di ampiezza filtrato

plt.plot(frequency_grid[indici_ordinati], spectrum_filtered_exp[indici_ordinati], color="#4c3bdd")
plt.grid(linewidth=0.5, alpha=0.5)
plt.title("Spettro di ampiezza filtrato")
plt.xlabel("Frequenza")
plt.ylabel("Ampiezza")
plt.show()

datas_filtered_exp_time = anti_dft(datas_filtered_exp)
plt.plot(temporal_grid, datas_filtered_exp_time, label="Dati filtrati", color="#4c3bdd")
plt.grid(linewidth=0.5, alpha=0.5)
plt.title("Segnale filtrato")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")
plt.show()

errori=np.abs(datas_filtered_exp_time - filtered)

plt.plot(temporal_grid, errori, label="Errori", color="#4c3bdd")
plt.grid(linewidth=0.5, alpha=0.5)
plt.title("Errori del filtro, exp vs butterworth")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")
plt.show()