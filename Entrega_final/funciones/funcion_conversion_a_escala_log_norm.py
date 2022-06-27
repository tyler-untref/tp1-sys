#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:03 2022

@author: pedro.f.ridano
"""

import numpy as np


#Función conversión a escala logarítmica normalizada
def conversion_log_norm(vector):
    '''
    Convierte una señal de entrada lineal en una señal con escala logaritmica
    normalizada.
    
    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso con escala lineal.

    Returns
    -------
    vector_log : Numpy Array
        respuesta al impulso con escala logaritmica normalizada.

    '''
    vector_max = max(np.abs(vector))
    vector_log = 10*np.log10(vector/(vector_max))
    return vector_log