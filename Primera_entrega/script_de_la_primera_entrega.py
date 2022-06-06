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
from scipy import signal
import scipy.io.wavfile as sio
# Primera consigna: Función de sintetización de ruido rosa

# Primer punto: definición de la función

def ruidoRosa_voss(t,fs,ncols):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    
    .Nota: si 'ruidoRosa.wav' existe, este será sobreescrito.
    
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


# Segundo punto: funcion gráfico del dominio temporal
def dominio_temporal(data):
    """
    Grafica una señal de entrada en función del tiempo. 
    
    Parametro
    ----------
    data: tupla. 
          El primer valor es un numpy array de 1D y el segundo es su fs.
             
    """
    archivo = data[0]
    fs = data[1]
    
    #Eje x: tiempo
    eje_x = np.linspace(0,(len(archivo)/fs),len(archivo))
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = archivo
    plt.ylabel("Amplitud normalizada")
    # plt.yscale('log')
    
    plt.plot(eje_x, eje_y)
    plt.show()      


#-----------------------------------------------------------------------------

#Segunda consigna: Funcion de generación de sine sweep logaritmico + filtro inverso

def gen_sine_sweep(t, f1, f2, fs):
    """
    Genera un Sine Sweep utilizando las ecuaciones de Meng, Q. y su filtro
    inverso. 
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del sine sweep.
    f1 : int
          Frecuencia inferior en Hz.
    f2 : int
        Frecuencia superior en Hz.
    fs : int
        Frecuencia de muestreo en Hz de la señal.      

    Returns: Tupla de dos elementos, con 1 Numpy array en cada elemento.
        El primer elemento corresponde al array del sine sweep generado y el 
        segundo elemento corresponde al filtro inverso del sine sweep generado.
    -------
    
    """
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2
    tiempo = np.linspace(0,t,t*fs)
    R = np.log(w2/w1)
    L = t/R
    K = (t*w1)/R

    frec_inst = (K/L)*np.exp(tiempo/L)
    mod_m = w1/(2*np.pi*frec_inst)
    
    sine_sweep_t = np.sin(K*(np.exp(tiempo/L) - 1))
    
    filtro_inverso = mod_m * sine_sweep_t[::-1]
    filtro_inverso = filtro_inverso/max(filtro_inverso)
    
    # Generación del archivo .wav
    sf.write('sine_sweep.wav', sine_sweep_t, fs)
    sf.write('filtro_inverso.wav', filtro_inverso, fs)
    
    return sine_sweep_t, filtro_inverso 

t =10
sine_sweep = gen_sine_sweep(t, 20, 20000, 48000)


## test con la convolucion:    
convolucion = signal.convolve(sine_sweep[0], sine_sweep[1])
sf.write('convolucion.wav', convolucion, 48000)



# #Generacion de filtro inverso del sine_sweep_grabado aula de informatica

# grabado, fs = sf.read('ss-grabado.wav')
# grabado = grabado[(fs):(31*fs)]
# w1 = 2*np.pi*88
# w2 = 2*np.pi*11314
# tiempo = np.linspace(0,len(grabado)/fs,len(grabado))
# R = np.log(w2/w1)
# L = t/R
# K = (t*w1)/R

# frec_inst = (K/L)*np.exp(tiempo/L)
# mod_m = w1/(2*np.pi*frec_inst)
# filtro_inverso = mod_m*((-1)*grabado)
    
# # Generación del archivo .wav
# sf.write('filtro_inverso.wav', filtro_inverso, 44100)


#-----------------------------------------------------------------------------


#Funcion de reproducción y adquisición
def reproduccion_adquisicion(archivo):
    '''
    Reproduce la señal de audio utilizando la salida física de audio 
    (seleccionada por default) y graba la reproducción de la misma, utilizando 
    la entrada física (seleccionada por default) de audio de la computadora.
    
    .Nota: La función escribe un archivo.wav llamado 'sine_sweep_generado_y_
    reproducido.wav'
    
    Parámetro
    ----------
    archivo : string
        nombre del archvio .wav a ser reproducido.

    Returns
    -------
    grabacion : Numpy Array
        devuelve un vector con la señal grabada.

    '''
    #, entrada='digital input', salida='digital output', lat_entrada='high', lat_salida='high')
                              
                             
    vector, fs = sf.read(archivo)
    # sd.default.device = (entrada, salida)
    # sd.default.latency = (lat_entrada, lat_salida)
    grabacion = sd.playrec(vector, fs, channels=2)
    sio.write('sine_sweep_generado_y_reproducido.wav', fs, grabacion)
    return grabacion


audio = reproduccion_adquisicion('sine_sweep.wav')


