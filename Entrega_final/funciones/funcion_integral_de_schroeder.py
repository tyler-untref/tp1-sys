#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:05 2022

@author: pedro.f.ridano
"""

import numpy as np

#Función integral de Schroeder
def integral_de_schroeder(resp_imp, fs):
    """
    Calcula la integral de Schroeder de la señal (respuesta al impulso) para así 
    trabajar con una señal más adecuada al momento de calcular los parámetros 
    acústicos.

    Parametros
    ----------
    resp_al_imp: Numpy Array
        respuesta al impulso ya suavizada.     
   
    Returns 
    -------
    resultado: Numpy Array
       integral de schroeder de la señal.

    """
    t = len(resp_imp)/fs
    n = len(resp_imp)
    extr_inf = 0
    extr_sup = t 
    delta_int = ((extr_sup - extr_inf)*fs)/n
    
    resp_imp = resp_imp ** 2
    resultado = delta_int*np.cumsum(resp_imp)
     
    return resultado