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
import matplotlib.pyplot as plt

# # Primera consigna; Función de carga de archivos de audio

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
            

#Segunda Consigna; Función de sintetización de respuesta al impulso


def sintetizacion_R_I(vector):
    """
    Recibe como argumento un diccionario cuyas claves son las frecuencias 
    centrales y sus elementos son los T60 correspondientes a cada frecuencia 
    central de banda de octava y de tercio de octava como establece 
    la norma IEC61260 (IEC61260,1995).

    La funcion escribe un archivo.wav llamado ´sintetizacion RI.wav´
    y devuelve un array con la respuesta al impulso sintetizada de valores de 
    T60 definidos para cada frecuencia central y de duracion t segundos. 
    -------
    """
    t = 3
    fs = 44100
    suma = np.zeros(t*fs)
    lista_f = list(vector)
    # print(lista_f)
    lista_t60 = list(vector.values())
    # print(lista_t60)
    tiempo = np.linspace(0, t, t*fs)
    for i in range(len(vector)):
        f_i = lista_f[i]
        t60i = lista_t60[i]
        pi_i = np.log(10**(-3))/t60i
        y_i = np.exp(pi_i*tiempo)*np.cos(2*np.pi*f_i*tiempo)
        # print(suma)
        suma = suma + y_i 
    suma = suma/max(abs(suma))
    sf.write('sintetizacion RI.wav', suma, fs)
    return suma

dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
            2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}

sint = sintetizacion_R_I(dic_t60)

sd.play(sint)
sd.wait()

    
# Tercera Consigna: Funcion de obtencion de la respuesta al impulso

def obtencion_RI(string1, string2):
    """
    Recibe como argumento dos strings, siendo el primero el nombre del 
    archivo .wav con el sine sweep grabado en el recinto y el segundo el
    nombre del archivo .wav del filtro inverso generado para ese sine sweep.

    Genera un archivo de audio.wav con la respuesta al impulso calculada 
    a partir de la convolucion entre los dos archivos de audio 
    y devuelve un numpy array con dicho vector de audio 
    -------

    """
    sine_sweep, fs = sf.read(string1)
    fourier_ss = np.fft.fft(sine_sweep) 
    
    filtro_inverso, fs = sf.read(string2)    
    fourier_fi = np.fft.fft(filtro_inverso) 

    producto = fourier_ss * fourier_fi
    resp_imp = abs(np.fft.ifft(producto))
    
    sf.write('respuesta al impulso.wav', resp_imp, fs)
    
    return resp_imp

#Nota: No pudimos hacer funcionar este código pero nos pareció importante adjuntarlo igual dado que sentimos que 
#no estamos tan lejos de lograrlo.

R_I_obtenido = obtencion_RI('sine_sweep.wav', 'filtro_inverso.wav')

sd.play(R_I_obtenido)
sd.wait()

# Cuarta Consigna: Funcion filtros Norma IEC61260

def filtrado(archivo, rango, orden):
    """
    Recibe como argumento tres parámetros: un string1 con el nombre del archivo
    a filtrar, un string2 pudiendo ser 'o' para un filtrado por bandas de octava
    o 't' para un filtrado por tercios de octava, y por último un int que indica
    el orden del filtro.

    Devuelve tres diccionarios, los tres tienen la misma información en las 
    claves: las frecuencias centrales de las bandas que fueron filtradas. 
    El primero tiene como valores, los arrays de la señal filtrada. 
    El segundo tiene como valores la respuesta en frecuencia del filtro 
    para cada frecuencia central correspondiente. 
    El último tiene como valores los arrays con las frecuencias angulares
    
    """
# no se nos ocurrió cómo interpretar esta consigna 
    

# Quinta Consigna: Funcion conversion a escala logaritmica normalizada

def conversion_log_norm(RI):
    """
    Recibe como argumento un array (la respuesta al impulso con escala lineal)
    
    Devuelve el mismo array pero convertido a escala logaritmica
    """
    RI_max = max(np.abs(RI))
    RI_log = 20*np.log10(RI/(RI_max))
    return RI_log

sint_log_norm = conversion_log_norm(sint)


