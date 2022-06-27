#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:06 2022

@author: pedro.f.ridano
"""

import numpy as np


#Función cálculo del EDT
def edt(vector, fs):
    '''
    Calcula el parámetro acústico EDT a partir de un vector (respuesta al impulso) 

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal de entrada.

    Returns
    -------
    valor_edt = int
        valor del parámetro EDT.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    inicial = 0
    final = hallar_caida(vector, 10)
    recta = regresion_entre(tiempo, vector, inicial, final)
    valor_edt = len(recta) / fs 
    return valor_edt

