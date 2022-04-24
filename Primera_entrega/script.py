import numpy as np
import soundfile as sf 
import pandas as pd
import matplotlib.pyplot as plt

#Parte 1: Función de sintetización de ruido rosa

#Primera consigna:

def ruidoRosa_voss(t, fs=44100, ncols=16):

    nrows = int(t*fs)

    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # el numero total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)

    ## Centrado de el array en 0
    total = total - total.mean()

    ## Normalizado
    valor_max = max(abs(max(total)), abs(min(total)))
    total = total / valor_max

    # Agregar generación de archivo de audio .wav
    sf.write('ruido_rosa.wav', total, fs)

    return total

ruido_rosa = ruidoRosa_voss(10)

#Segunda consigna:

def dominio_temporal(t,fs):

    #Eje x: tiempo
    eje_x = np.linspace(0,t+1,t*fs)
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = ruido_rosa
    plt.ylabel("Amplitud Normalizada")
    
    plt.title("Gráfico: dominio temporal de la señal")
    plt.plot(eje_x, eje_y)
    return plt.show()

dominio_temporal(10,44100)

#Tercera consigna:

#Parte 2: Funcion de generación de sine sweep logaritmico + filtro inverso

#Parte 3: Función adquisicion y reproducción