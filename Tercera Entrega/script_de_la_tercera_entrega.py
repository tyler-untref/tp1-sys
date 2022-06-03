#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:32:49 2022

@author: pedro.f.ridano
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal
import scipy.integrate as integrate

# Primera consigna: Realizar una función que aplique la transformada de Hilbert 
# a una señal de entrada. 

def signo(w):
    """
    Calcula la función signo a partir de un valor de frecuencia

    Parameters
    ----------
    w : float
        frecuencia en radianes

    Returns el valor del signo del número ingresado en la entrada
    -------
    None.

    """
    if w == 0:
        signo = 0
    elif w > 0:
        signo = 1
    else: signo = -1    
      
    return signo       
           
    

def transformada_de_hilbert(entrada):
    """
    Aplica la transformada de Hilbert a una señal de entrada.
    
    qué hacemos si algun componente de t = 0 ?
    
    Parameters
    
    ----------
    entrada : numpy array
        

    Returns numpy array 
    -------
    None.

    """
    print(len(entrada))
    t = np.arange(1, len(entrada)+1)
    print(len(t))
    
    resp_al_imp = 1/(np.pi*t)
    return resp_al_imp 
    
    # scipy.signal.convolve(in1, in2, mode='full', method='auto')
    
# Segunda consigna: 

# Función integral de Schroeder

def integral_de_schroeder(impulso, resp_al_imp):
    """
    Permite calcular una curva de decaimiento más suavizada de la
    respuesta al impulso para así trabajar con una señal más adecuada 
    al momento de calcular los parámetros acústicos.

    Parameters
    ----------
   
    impulso: 
    resp_al_imp:     
   
    Returns numpy array 
    -------
    None.


    """
    
    resultado = integrate.quad((resp_al_imp)**2, impulso[-1])
    
    





    