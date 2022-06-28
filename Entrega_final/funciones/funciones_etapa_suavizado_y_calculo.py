#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 10:33:41 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
from numpy.linalg import inv
from funciones_intermedias import hallar_caida
    

#Función suavizado de la señal (filtro de media móvil)    
def filtro_media_movil(vector):
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
    suavizada = np.zeros(len(vector) - ventana)
    for i in range(0, len(vector)-ventana):
        suavizada[i] = np.mean(vector[i:i+ventana])
    print(len(suavizada))
    print(np.shape(suavizada))
    print(suavizada)    
    suavizada = suavizada/max(suavizada)    
    return suavizada[::-1]


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
    resultado  = resultado[-1] - resultado #integral hasta el final - integral en cada valor 
    resultado = resultado[:-1] #elimino el ultimo valor (porque es =0)
    #integral_db = conversion_log_norm(schroeder_resp_imp_sint)
 
    return resultado


#Funcion regresión lineal por cuadrados minimos
def regr_cuad_min(vector, fs, grado=1):
    '''
    Calcula la aproximación por regresión utilizando el método de cuadrados 
    mínimos dada una señal de entrada, en función del tiempo que dura la señal. 

    Parametros
    ----------
    vector : Numpy Array.
            señal de entrada.
    fs : int
        frecuencia de sampleo de la señal de entrada.        
    grado : int, opcional.
           si es =1 realiza la regresión lineal, si es =2 realiza la regresión
           cuadrática, etc.
           el valor por default es 1.

    Returns
    -------
    interpolacion : Numpy Array.
        señal aproximada a partir de la interpolación.

    '''
    vector_x = np.linspace(0, len(vector)/fs, len(vector))
    vector_y = vector
    columnas = grado + 1
    filas = len(vector_x)
    M = np.zeros([filas, columnas])
    
    for i in range(0, filas):
        for j in range(0, columnas):
            M[i, j] = (vector_x[i])**j
    #print(M) 
    T = M.transpose()
    M_coef = inv((T@M))@(T@vector_y)    
    M_coef = M_coef[::-1]
    #R = np.corrcoef(M_coef)
    interpolacion = np.polyval(M_coef, vector_x)

    return interpolacion

def regresion_entre(vector_x, vector_y, inicial, final):
    '''
    Realiza la regresión lineal entre los valores de la imagen de una señal 
    y los valores de su pre-imagen dentro de un intervalo dado [final-inicial]

    Parametros
    ----------
    vector_x : Numpy Array
        vector correspondiente a los valores de la pre-imagen. Por ej: eje temporal
    vector_y : Numpy Array
        vector correspondiente a los valores de la imagen.
    inicial : int
        primer valor del intervalo.
    final : int
        último valor del intervalo.

    Returns
    -------
    salida: Numpy Array
        señal que representa la recta de regresión entre el intervalo dado.

    '''
    coefs = regr_cuad_min(vector_x, vector_y)
    salida = np.polyval(coefs, vector_x)
    salida = salida[inicial:final]
    return salida

# ----------------------------------------------------------------------------

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

    
def regresion_entre_np(vector_x, vector_y, inicial, final):
    '''
    Realiza la regresión lineal entre los valores de la imagen de una señal 
    y los valores de su pre-imagen dentro de un intervalo dado [final-inicial]

    Parametros
    ----------
    vector_x : Numpy Array
        vector correspondiente a los valores de la pre-imagen. Por ej: eje temporal
    vector_y : Numpy Array
        vector correspondiente a los valores de la imagen.
    inicial : int
        primer valor del intervalo.
    final : int
        último valor del intervalo.

    Returns
    -------
    salida: Numpy Array
        señal que representa la recta de regresión entre el intervalo dado.

    '''
    A = np.vstack([vector_x, np.ones(len(vector_x))]).T
    m, c = np.linalg.lstsq(A, vector_y)
    R = np.corrcoef(vector_x,vector_y)
    return m, c , R



if __name__ == '__main__':
    elveden, fs = sf.read('elveden_hall_suffolk_england.wav')
    sd.play(elveden)
    elveden = elveden[:, 0]
    
    #suavizo la señal
    suavizada_elveden = filtro_media_movil(elveden)
    # from funciones_grafico_de_dominio_temporal import dominio_temporal
    # dominio_temporal((elveden, fs))
    integral_elveden = integral_de_schroeder(suavizada_elveden, fs)
    
    vector_tiempo = np.linspace(0, len(elveden)/fs, len(elveden))
    
    regresion_entre_np(vector_tiempo, elveden, 0, len(elveden))
    




