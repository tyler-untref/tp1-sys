#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:04 2022

@author: pedro.f.ridano
"""

import numpy as np

#Función suavizado de la señal (filtro de media móvil)    
def filtro_media_movil(archivo):
    '''
    Aplica el filtro de media móvil a una señal de entrada, utilizando una
    ventana fija de 501 muestras. 

    Parametros
    ----------
    archivo : numpy array
        señal a ser filtrada.

    Returns
    -------
    suavizada : numpy array
        señal filtrada.

    '''
    ventana = 501
    suavizada = np.zeros(len(archivo)-ventana)
    for i in range(0, len(archivo)-ventana):
        suavizada[i] = np.mean(archivo[i:i+ventana])
    suavizada = suavizada/max(suavizada)    
    return suavizada