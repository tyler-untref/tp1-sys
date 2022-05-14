#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 23:47:01 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 
import sounddevice as sd
import pandas as pd
import matplotlib.pyplot as plt

# Primera consigna: Función de sintetización de ruido rosa

# Primer punto: definición de la función

def ruidoRosa_voss(t,fs,ncols):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    
    .. Nota:: si 'ruidoRosa.wav' existe, este será sobreescrito
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del ruido generado.
    ncols: int
        Determina el número de fuentes aleatorias a agregar.
    fs: int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
    
    returns: NumPy array
        Datos de la señal generada.
    
    Ejemplo
    -------
    Generar un `.wav` desde un numpy array de 10 segundos con ruido rosa a una 
    frecuencia de muestreo de 44100 Hz.
    
        import numpy as np
        import soundfile as sf
        from scipy import signal
        
        ruidoRosa_voss(10)
    """

    nrows = int(float(t)*fs)

    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # El número total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)

    # Centrado del array en 0
    total = total - total.mean()

    # Normalizado
    valor_max = max(abs(max(total)), abs(min(total)))
    total = total / valor_max

    # Agregar generación de archivo de audio .wav
    sf.write('ruido_rosa.wav', total, fs)

    return total

ruido_rosa = ruidoRosa_voss(10,44100,16)

# Segundo punto: gráfico del dominio temporal

def dominio_temporal(t,fs):
    """
    Genera un array de t segundos con n = t*fs muestras y plotea 
    la amplitud de la función ruido rosa dependiente del tiempo 
    
    """
    #Eje x: tiempo
    eje_x = np.linspace(0,t,t*fs)
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = ruido_rosa
    plt.ylabel("Amplitud Normalizada")
    
    plt.title("Gráfico: dominio temporal de la señal")
    plt.plot(eje_x, eje_y)
    return plt.show()

dominio_temporal(10,44100)

# Tercer punto: gráfico del domino espectral

#Lee el archivo .txt
df = pd.read_csv(r'C:\Users\Tyler\Documents\UNTREF\Señales_y_sistemas\Práctica\TP\tp1-sys\Primera_entrega\espectro-ruido_rosa.txt', delimiter="\t")
#Lo convierte a un array
array = df.to_numpy()

#Eje x: frecuencia
eje_x = array[:,0]
plt.xlabel("Frecuencia (Hz)")

#Eje y: amplitud
eje_y = array[0,:]
plt.ylabel("Amplitud (dB)")

plt.title("Gráfico: Espectro de la señal")
plt.plot(eje_x, eje_y)
grafico = plt.show() 

#El problema es que las dimensiones tienen que ser iguales

# Cuarto punto:

sd.play(ruido_rosa,44100)
sd.wait()

#-----------------------------------------------------------------------------

# Segunda consigna: Funcion de generación de sine sweep logaritmico + filtro inverso

# Primer punto: definición de la función

def sine_sweep(t, f1, f2, fs):
    
    """
    Genera un Sine Sweep utilizando las ecuaciones de Meng, Q.
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del sine sweep.
    f1 : int
         Frecuencia inferior en Hz.
    f2 : int
        Frecuencia superior en Hz.
    fs : int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.    

    Returns: NumPy array
        Datos de la señal generada.
    -------
    None.
    """
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2
    tiempo = np.linspace(0,t+1,t*fs)
    R = np.log(w2/w1)
    L = t/R
    K = (t*w1)/R

    resultado = np.sin(K*(np.exp(tiempo/L)-1))

    # Generación del archivo .wav
    sf.write('sine_sweep.wav', resultado, 44100)
    
    return resultado

sine_sweep = sine_sweep(10, 20, 20000, 44100)

#Reproducción de los resultados 

sd.play(sine_sweep)
sd.wait()

# Filtro inverso ?

# Segundo punto: Gráfico del espectro

#Lee el archivo .txt
df = pd.read_csv(r'C:\Users\Tyler\Documents\UNTREF\Señales_y_sistemas\Práctica\TP\tp1-sys\Primera_entrega\espectro_sine_sweep.txt', delimiter="\t")
#Lo convierte a un array
array = df.to_numpy()

#Eje x: frecuencia
eje_x = array[:,0]
plt.xlabel("Frecuencia (Hz)")

#Eje y: amplitud
eje_y = array[0,:]
plt.ylabel("Amplitud (dB)")

plt.title("Gráfico: Espectro de la señal")
plt.plot(eje_x, eje_y)
grafico = plt.show() 

#El problema es que las dimensiones tienen que ser iguales


#-----------------------------------------------------------------------------

#No supimos ni cómo encarar la tercera consigna (funcion de adquisición y repr)







