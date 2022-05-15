#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:13:55 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 
import sounddevice as sd
import pandas as pd
import scipy 
# import matplotlib.pyplot as plt

# Primera consigna; Función de carga de archivos de audio

def carga(lista):
    '''
    Recibe como argumento una lista formada por strings que refieren a los 
    archivos de audio a ser cargados

    Devuelve un diccionario cuyas claves son los nombres
    de los archivos y cuyos elementos son tuplas. En la primera posicion
    de cada tupla se encuentra el array de Numpy con el vector de audio y en la
    segunda posicion la frecuencia de muestreo del vector
    -------
    '''
    diccionario = {} 
    for i in lista:
        array = sf.read(i)
        diccionario[i]=array
    return diccionario
    
dic = carga(['usina_main_s1_p5.wav', 'minster1_000_ortf_48k.wav'])
            

# Segunda Consigna; Función de sintetización de respuesta al impulso

dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
           2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}

def sintetizacion_R_I(vector):
    """
    Recibe como argumento un diccionario cuyas claves son las frecuencias 
    centrales y sus elementos son los T60 correspondientes a cada frecuencia 
    central de banda de octava y de tercio de octava como establece 
    la norma IEC61260 (IEC61260,1995).

    Devuelve una respuesta al impulso sintetizada en formato archivo 
    de audio.wav de valores de T60 definidos para cada
    frecuencia central y de duracion 2 segundos.
    -------
    """
    suma = np.zeros(0)
    lista_f = list((vector))
    lista_t60 = list(dic_t60.values())
    cant_elementos = np.arange(0, len(vector))
    for i in cant_elementos:
        t = 2
        fi = lista_f[i]
        t60i = lista_t60[i]
        pi = np.log(10**-3)/t60i
        yi = np.exp(pi*t)*np.cos(2*np.pi*fi*t)
        suma = suma + yi
    sint = sf.write('sintetizacion IR.wav', suma, 41000)
    return sint
    
