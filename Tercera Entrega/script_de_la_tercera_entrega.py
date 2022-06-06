#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:32:49 2022

@author: pedro.f.ridano
"""

import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import sounddevice as sd
import pandas as pd
from scipy import signal
import scipy.integrate as integrate
from numpy.linalg import inv

def valor_min(vector):
    '''
    Halla el valor mínimo de una señal   

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    minimo : float
        valor minimo de la señal

    '''
    vector = abs(vector)
    minimo = 1
    for i in vector:
        if i < minimo and i != 0:
            minimo = i
    
    return minimo      

def reemplaza_ceros(vector):
    '''
    Reemplaza los ceros de una señal por el valor mínimo de la misma.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    vector : Numpy Array
        señal con los ceros anteriores reemplazados por el valor mínimo de la misma.

    '''
    minimo = valor_min(vector)
    for i in range(len(vector)):
        if vector[i] == 0 :
            vector[i] = minimo    
    return vector


def elimina_inicio(vector):
    '''
    Elimina el ruido y silencios al comienzo de una señal exponencial decreciente.

    Parametros
    ----------
    vector : Numpy Array
        señal a ser eliminado su ruido inicial.

    Returns
    -------
    vector: Numpy Array
        señal sin el ruido o silencio inicial.

    '''
    indice_max = np.argmax(vector)
    vector = vector[indice_max:]
   
    return vector 


def corta_inf(vector):
    '''
    Elimina valores de tipo inf. de la señal de entrada.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    salida : Numpy Array
        señal sin los valores de tipo inf.

    '''
    output = []
    for val in vector:
         if np.isfinite(val):
              output.append(val)
    salida = np.array(output)
    return salida


def elimina_valores(vector, valor):
    '''
    Elimina un valor específico de la señal de entrada.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.
    valor : int
        valor entero que se desea eliminar de la señal.

    Returns
    -------
    salida : Numpy Array
        señal sin el valor seleccionado.

    '''
    output = []
    for val in vector:
         if val != valor:
              output.append(val)
    salida = np.array(output)
    return salida


#Traigo la funcion de dominio temporal de la primer entrega
def dominio_temporal(data):
    """
    Grafica una señal de entrada en función del tiempo. 
    
    Parametro
    ----------
    data: tupla. 
          El primer valor es un numpy array de 1D y el segundo es su fs.
             
    """
    archivo = data[0]
    fs = data[1]
    
    #Eje x: tiempo
    eje_x = np.linspace(0,(len(archivo)/fs),len(archivo))
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = archivo
    plt.ylabel("Amplitud [dB]")
    # plt.yscale('log')
    
    plt.plot(eje_x, eje_y, 'g')
    plt.show()      


#Variacion de la funcion de dominio temporal de la primera entrega
def dominio_temporal_2(data_1, data_2, fs):
    """
    Grafica dos señales de entrada en función del tiempo de forma simultánea. 
    
    .Nota: Ambas señales deberán tener la misma frecuencia de sampleo.
    
    Parámetro
    ----------
    data: tupla
        -El primer valor es un numpy array de 1D 
        -El segundo valor es otro numpy array de 1D
        -El tercer valor es la frecuencia de sampleo
             
    """
   
    #Eje x_1: tiempo
    eje_x_1 = np.linspace(0,(len(data_1)/fs),len(data_1))
    plt.xlabel("Tiempo (s)")
    
    #Eje x_2: tiempo
    eje_x_2 = np.linspace(0,(len(data_2)/fs),len(data_2))
    plt.xlabel("Tiempo (s)")
    
    #Eje y_1: amplitud normalizada
    eje_y_1 = data_1
    plt.ylabel("Amplitud [dB]")
    
    #Eje y_2: amplitud normalizada
    eje_y_2 = data_2
    plt.ylabel("Amplitud [dB]")
    
    plt.plot(eje_x_1, eje_y_1, 'orange', label='respuesta al impulso')
    plt.legend()
    plt.plot(eje_x_2, eje_y_2, label='respuesta al impulso suavizada')
    plt.legend()
    plt.show()     



#Primera consigna: Realizar una función que aplique un suavizado a la señal
#filtro de media movil    
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


#prueba
resp_imp, fs = sf.read('usina_main_s1_p5.wav')
resp_imp = resp_imp[:,0]
resp_imp = elimina_inicio(resp_imp)
resp_imp = abs(resp_imp)

resp_imp_media_movil = filtro_media_movil(resp_imp)
# resp_imp = reemplaza_ceros(resp_imp) 
# resp_imp_db = 20.0 * np.log10(abs(resp_imp))

#Traigo la funcion de la segunda entrega
def conversion_log_norm(RI):
    '''
    Convierte una señal de entrada lineal en una señal con escala logaritmica
    normalizada.
    
    Parametros
    ----------
    RI : Numpy Array
        respuesta al impulso con escala lineal.

    Returns
    -------
    RI_log : Numpy Array
        respuesta al impulso con escala logaritmica normalizada.

    '''
    RI_max = max(np.abs(RI))
    RI_log = 20*np.log10(RI/(RI_max))
    return RI_log

resp_imp_media_movil_db = conversion_log_norm(resp_imp_media_movil)


##Comentar sobre el resultado obtenido. ¿Qué está visualizando? 
##comparar con la señal original en el mismo gráfico.
##Se observan una gran diferencia. ¿Qué método de filtrado es más efectivo?


#Segunda consigna: 

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
    

#prueba
sch = integral_de_schroeder(resp_imp_media_movil, fs)
schroeder = sch[-1] - sch #integral hasta el final - integral en cada valor 
schroeder = schroeder[:-1] #elimino el ultimo valor (porque es =0)
schroeder_db = 10*np.log10(schroeder/max(schroeder))



#Funcion regresión lineal por cuadrados minimos
def regr_cuad_min(vector_x, vector_y, grado=1):
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
    return M_coef

abcisa = np.linspace(0, len(schroeder_db)/fs, len(schroeder_db))
coefs = regr_cuad_min(abcisa, schroeder_db) 
interpolacion = np.polyval(coefs, abcisa)


#Funciones intermedias para calcular los parámetros acústicos según Norma ISO 3382

#Función para calcular inicial, final
def hallar_caida(vector, valor_caida):
    '''
    Calcula la pre-imagen de la señal al decaer un valor determinado en amplitud.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.
    valor_caida : int
        valor de caída de interés.

    Returns
    -------
    abcisa_caida : int
        muestra de la señal en la que la misma decae al valor de interés.

    '''
    abcisa_caida = 0
    for i in range(0, len(vector)):
        if vector[i] <= -valor_caida:
            abcisa_caida = i
            break
    return abcisa_caida 


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


t60 = t_60(schroeder_db, 48000, 't_20')


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
    












