#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:13:06 2022

@author: pedro.f.ridano
"""
import numpy as np 

#Funcion D50
def d_50(vector, fs):
    '''
    Calcula el parámetro acústico D50 a partir de una respuesta al impulso.

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal.

    Returns
    -------
    resultado: float
        valor del parámetro D50.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_50 = int((50/1000) * fs)
    resultado = 100 * integral[muestra_50] / (integral[-1])
    
    return resultado