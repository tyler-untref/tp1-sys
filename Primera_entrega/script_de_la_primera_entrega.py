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

def dominio_temporal(archivo):
    """
    Grafica una señal de entrada en función del tiempo. 
    
    Parametro
    ----------
    archivo: Numpy array de 1D
             
    """
    
    archivo, fs = archivo
    #Eje x: tiempo
    eje_x = np.linspace(0,(len(archivo)/fs),len(archivo))
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = archivo
    plt.ylabel("Amplitud Normalizada")
    
    plt.title("Gráfico: Dominio temporal de la señal")
    plt.plot(eje_x, eje_y)
    return plt.show()  


# sd.play(ruido_rosa,44100)
# sd.wait()


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
        El primer elemento corresponde al array del sine sweep y el segundo
        elemento corresponde al filtro inverso.
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
    
    sine_sweep_t = np.sin(K*(np.exp(tiempo/L)-1))
    
    filtro_inverso = mod_m*sine_sweep_t[::-1]
    
    # Generación del archivo .wav
    sf.write('sine_sweep.wav', sine_sweep_t, fs)
    sf.write('filtro_inverso.wav', filtro_inverso, fs)
    return sine_sweep_t, filtro_inverso 

t =10
sine_sweep = gen_sine_sweep(t, 20, 20000, 48000)

#Reproducción de los resultados 

# sd.play(sine_sweep[0])
# sd.wait()

# #generacion de filtro inverso del sine_sweep_grabado aula de informatica
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



##Grafico del sine sweep generado

#Eje x: frecuencia
eje_x_ss = np.linspace(0,t,t*48000)

#Eje y: amplitud
eje_y_ss = sine_sweep[0]

plt.figure(1)
plt.plot(eje_x_ss, eje_y_ss,'r')
plt.xlabel("Tiempo (t)")
plt.ylabel("Amplitud (dB)")
plt.title("esto es del sine sweep")


##Grafico del filtro inverso

#Eje x: frecuencia
eje_x_fi = np.linspace(0,t,t*48000)

#Eje y: amplitud
eje_y_fi = sine_sweep[1]

plt.figure(2)
plt.plot(eje_x_fi, eje_y_fi)
plt.xlabel("Tiempo (t)")
plt.ylabel("Amplitud (dB)")
plt.title("esto es del filtro inverso")


## test con la convolucion:
    
convolucion = signal.convolve(sine_sweep[0], sine_sweep[1])
sf.write('convolucion.wav', convolucion, 48000)
#sd.play(convolucion)

#Grafico de la convolucion
#Eje x: frecuencia
eje_x_c = np.linspace(0,(len(convolucion)/48000),len(convolucion))

#Eje y: amplitud
eje_y_c = convolucion

plt.figure(3)
plt.plot(eje_x_c, eje_y_c,'g')
plt.xlabel("Tiempo (t)")
plt.ylabel("Amplitud (dB)")
plt.title("esto es de la convolucion")


    
    
#-----------------------------------------------------------------------------

# Tercera Consigna: Funcion de adquisicion y reproduccion

# def adquisicion_reproduccion(archivo,t):
    
#     """
#     Recibe como argumento un string con el nombre de archivo de audio a ser 
#     reproducido y un int el cual refiere al tiempo de duracion de la 
#         reproducción del archivo. 
    
#     Devuelve una reproducción de audio utiizando sd.read
    
#     """
    
#     (archivo, fs) = sf.read(archivo)
#     archivo = archivo[0:((t*fs)+1)]
#     return sd.play(archivo)

# prueba = adquisicion_reproduccion('miles.wav', 5)





