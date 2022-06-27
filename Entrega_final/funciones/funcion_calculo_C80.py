#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:13:07 2022

@author: pedro.f.ridano
"""
import numpy as np 

#Funcion C80
def c_80(vector, fs):
    '''
    Calcula el parámetro acústico C80 a partir de una respuesta al impulso.

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal.

    Returns
    -------
    resultado: float
        valor del parámetro acústico C80.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_80 = int((80/1000) * fs)
    resultado = 10 * np.log10(integral[muestra_80] / (integral[-1] - integral[muestra_80]))
    
    return resultado
    
