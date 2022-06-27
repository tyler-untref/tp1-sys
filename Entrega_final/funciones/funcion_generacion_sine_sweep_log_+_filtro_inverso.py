#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:49:33 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 

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
    sf.write('sine_sweep.wav', sine_sweep_t, fs)
    sf.write('filtro_inverso.wav', filtro_inverso, fs)
    
    return sine_sweep_t, filtro_inverso 
