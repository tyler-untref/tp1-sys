import numpy as np
import soundfile as sf 
import sounddevice as sd
import pandas as pd
import matplotlib.pyplot as plt

# Segunda consigna: Funcion de generación de sine sweep logaritmico + filtro inverso

# Primer punto: definición de la función

def sine_sweep(t, w1, w2, fs):
    
    """
    Genera un Sine Sweep utilizando las ecuaciones de Meng, Q.
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del sine sweep.
    w1 : int
         Frecuencia angular inferior en rad/s
    w2 : int
        Frecuencia angular superior en rad/s.
    fs : int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.    

    Returns: NumPy array
        Datos de la señal generada.
    -------
    None.
    """
    
    tiempo = np.linspace(0,t+1,t*fs)
    R = np.log(w2/w1)
    L = t/R
    K = (t*w1)/R

    resultado = np.sin(K*(np.exp(tiempo/L)-1))

    # Generación del archivo .wav
    sf.write('sine_sweep.wav', resultado, 44100)
    
    return resultado

sine_sweep = sine_sweep(10, 20, 20000, 44100)

# Filtro inverso ?

#Reproducción de los resultados 

sd.play(sine_sweep)
sd.wait()


# Segundo punto: Gráfico del espectro




