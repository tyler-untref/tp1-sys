#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:06 2022

@author: pedro.f.ridano
"""

import numpy as np


#Funcion T60 a partir del T10, T20, T30
def t_60(vector, fs, metodo):
    '''
    Calcula el T60 a partir del T10, T20 o T30.

    Parametros
    ----------
    archivo : Numpy Array
        recta obtenida a partir de la regresión lineal por cuadrados minimos. 
    fs : int
        fs del archivo original.
    metodo : string
        tipo de tiempo de reverberacion a partir del cual la función calculará 
        el T_60. 
        Por ejemplo, si metodo = 't_10' la función calcula el T60 a partir del T10, 
                     si 't_20' la función calcula el T60 a partir del T20,
                     si 't_30' la función calcula el T60 a partir del T30.

    Returns
    -------
    resultado : float
        valor del parámetro T60.
    

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    multiplicador = {'t_10': 6, 't_20': 3, 't_30': 2}
    if metodo == 't_10':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 15)
        recta = regresion_entre(tiempo, vector, inicial, final)
        
    if metodo == 't_20':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 25)
        recta = regresion_entre(tiempo, vector, inicial, final)
        
  
    if metodo == 't_30':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 35)
        recta = regresion_entre(tiempo, vector, inicial, final)
    
    resultado = multiplicador[metodo] * (len(recta)) / fs
    
    return resultado
