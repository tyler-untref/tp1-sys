#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:25:30 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 
import sounddevice as sd
import pandas as pd
import scipy.io.wavfile as sio


#Función sintetización de ruido rosa
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


#Función de sintetización de respuesta al impulso
def sintetizacion_RI(diccionario, fs):
    '''
    Genera una respuesta al impulso sintetizada a partir de valores previamente
    definidos de T_60
    
    .Nota: La función escribe un archivo.wav llamado ´sintetizacion RI.wav´.

    Parametros
    ----------
    diccionario : diccionario
         un diccionario cuyas claves son las frecuencias centrales 
         y sus valores son los T60 correspondientes a cada frecuencia
         central de banda de octava como establece la norma IEC61260 
         (IEC61260,1995).
    fs : int
         frecuencia de muestreo deseada de la señal resultante.  

    Returns
    -------
    suma : Numpy Array
       con la respuesta al impulso sintetizada de valores de T60 definidos 
       para cada frecuencia central y con duracion de t segundos.
    
    '''
    t = 3
    suma = np.zeros(t*fs)
    lista_f = list(diccionario)
    # print(lista_f)
    lista_t60 = list(diccionario.values())
    # print(lista_t60)
    tiempo = np.linspace(0, t, t*fs)
    for i in range(len(diccionario)):
        f_i = lista_f[i]
        t60i = lista_t60[i]
        tau_i = ((-1)*np.log(10**(-3)))/t60i
        tau_i = tau_i
        y_i = np.exp(tau_i*tiempo)*np.cos(2*np.pi*f_i*tiempo)
        # print(suma)
        suma = suma + y_i 
    suma = suma[::-1]
    suma = (suma/max(abs(suma)))
    
    sf.write('sintetizacion_RI.wav', suma, fs)
    return suma, fs


#Función generación de sine sweep logaritmico + filtro inverso
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
    sf.write('sine_sweep_generado.wav', sine_sweep_t, fs)
    sf.write('filtro_inverso_generado.wav', filtro_inverso, fs)
    
    return sine_sweep_t, filtro_inverso 



#Funcion de reproducción y adquisición
def reproduccion_adquisicion(archivo):
    '''
    Reproduce la señal de audio utilizando la salida física de audio 
    (seleccionada por default) y la graba utilizando la entrada física 
    (seleccionada por default) de audio de la computadora.
    
    .Nota: La función escribe un archivo.wav llamado 'sine_sweep_generado_y_
    reproducido.wav'
    
    Parámetro
    ----------
    archivo : string
        nombre del archivo .wav a ser reproducido.

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
    grabacion = grabacion[:,0] #lo convierto a mono
    sio.write('sine_sweep_generado_y_grabado.wav', fs, grabacion)
    return grabacion


if __name__ == '__main__':
    ruido_rosa = ruidoRosa_voss(5, 48000, 16)
    
    dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
           2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}
    resp_imp_sint = sintetizacion_RI(dic_t60, 48000)
    resp_imp_sint = resp_imp_sint[0]
    from funciones_grafico_de_dominio_temporal import dominio_temporal
    
    dominio_temporal((resp_imp_sint, 48000))
    tupla_sweep_fi = gen_sine_sweep(10, 20, 20000, 48000)
    ss_generado = tupla_sweep_fi[0]
    fi_generado = tupla_sweep_fi[1]
    
    ss_repr_grabado = reproduccion_adquisicion('sine_sweep_generado.wav')
    

