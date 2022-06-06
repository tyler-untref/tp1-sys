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

# EPS = np.finfo(float).eps
def valor_min(vector):
    vector = abs(vector)
    minimo = 1
    for i in vector:
        if i < minimo and i != 0:
            minimo = i
    return minimo      

def elimina_inicio(vector):
    indice_max = np.argmax(vector)
    return vector[indice_max:]

def reemplaza_ceros(vector):
    minimo = valor_min(vector)
    for i in range(len(vector)):
        if vector[i] == 0 :
            vector[i] = minimo    
    return vector

def corta_inf(vector):
    output = []
    for val in vector:
         if np.isfinite(val):
              output.append(val)
    return output

def elimina_valores(vector, valor):
    output = []
    for val in vector:
         if val != valor:
              output.append(val)
    return output


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
    plt.ylabel("Amplitud Normalizada")
    # plt.yscale('log')
    
    plt.title("Gráfico: Dominio temporal de la señal")
    plt.plot(eje_x, eje_y)
    plt.show()      


#Primera consigna: Realizar una función que aplique un suavizado a la señal


##filtro de media movil    
def filtro_media_movil(archivo):
    '''
    Aplica el filtro de media móvil a una señal de entrada, utilizando una
    ventana fija de 10 muestras. 

    Parametros
    ----------
    archivo : numpy array
        array de la señal a ser filtrada.

    Returns
    -------
    suavizada : numpy array
        array con la señal filtrada.

    '''
    ventana = 501
    suavizada = np.zeros(len(archivo)-ventana)
    for i in range(0, len(archivo)-ventana):
        suavizada[i] = np.mean(archivo[i:i+ventana])
    #compenzamos el delay concatenando ceros al array
    #suavizada = np.hstack([np.zeros(ventana//2), suavizada])
    return suavizada


#prueba
resp_imp, fs = sf.read('usina_main_s1_p5.wav')
resp_imp = resp_imp[:,0]
resp_imp = elimina_inicio(resp_imp)
resp_imp = abs(resp_imp)

# #quito el ruido de fondo inicial ?
# resp_imp = resp_imp[int(0.09)*fs:len(resp_imp)]

resp_imp_media_movil = filtro_media_movil(resp_imp)
# resp_imp = reemplaza_ceros(resp_imp) 
# resp_imp_db = 20.0 * np.log10(abs(resp_imp))


##Comentar sobre el resultado obtenido. ¿Qué está visualizando? 
##comparar con la señal original en el mismo gráfico.
##Se observan una gran diferencia. ¿Qué método de filtrado es más efectivo?



#Segunda consigna: 

##Función integral de Schroeder

def integral_de_schroeder(resp_imp, fs):
    """
    Permite calcular una curva de decaimiento más suavizada de la
    respuesta al impulso para así trabajar con una señal más adecuada 
    al momento de calcular los parámetros acústicos.

    Parameters
    ----------
    resp_al_imp:     
   
    Returns numpy array 
    -------
    None.


    """
    t = len(resp_imp)/fs
    n = len(resp_imp)
    extr_inf = 0
    extr_sup = t 
    delta_int = ((extr_sup - extr_inf)*fs)/n
    
    resp_imp = resp_imp ** 2
    resultado = delta_int*np.cumsum(resp_imp)
     

    return resultado
    
sch = integral_de_schroeder(resp_imp_media_movil, fs)
# sch = sch[-1]-sch
schroeder = sch[-1] - sch #integral hasta el final - integral en cada valor 
schroeder = schroeder[:-1] #elimino el ultimo valor (=0)
schroeder_db = 10*np.log10(schroeder/max(schroeder))


#Comentar lo que se observa en la visualización.
resp_imp_media_movil_db = 20*np.log10(resp_imp_media_movil/max(resp_imp_media_movil))


##Tercera consigna


def elimina_inf(vector):
    for i in range(0, len(vector)):
        if np.isfinite(vector[i]) == True:
            np.hstack((vector, vector[i]))    
    return vector
    


# sch_db = corta_inf(sch_db)

#Funcion regresion lineal por cuadrados minimos
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


#Funciones para calcular los parámetros acústicos según Norma ISO 3382

#para calcular inicial, final
def hallar_caida(vector, valor_caida):
    abcisa_caida = 0
    for i in range(0, len(vector)):
        if vector[i] <= -valor_caida:
            abcisa_caida = i
            break
    return abcisa_caida 

#regresion entre
def regresion_entre(vector_x, vector_y, inicial, final):
    coefs = regr_cuad_min(vector_x, vector_y)
    salida = np.polyval(coefs, vector_x)
    return salida[inicial:final]
       
def edt(vector, fs):
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    inicial = 0
    final = hallar_caida(vector, 10)
    recta = regresion_entre(tiempo, vector, inicial, final)
    return 6 * len(recta) / fs

#Funcion T60 a partir del T10, T20, T30
def t_60(vector, fs, metodo):
    '''
    Calcula el T60 a partir del T_10, T_20 o T_30.

    ---> Para calcular el T60 la funcion busca el valor de -5 dBFS en el array 
          y realiza una resta entre la pre imagen de ese punto con el punto de 
          caida -15,-25 o -35 dBFS, dependiendo de la opcion que se haya elegido, 
          luego multiplica ese valor por 6, 3 o 2 y se obtiene el T60.

    Parameters
    ----------
    archivo : numpy array
        señal recta interpolada por cuadrados minimos. 
    fs : int
        fs del archivo original.
    string : string
        tipo de tiempo de reverberacion a partir de los cuales la función
        calculará el T_60. 
        Por ejemplo, si 'T_10' la función calcula a partir del T_10, 
                      si 'T_20' la función calcula a partir del T_20,
                      si 'T_30' la función calcula a partir del T_30.

    Returns
    -------
    resultado : float
    

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

    return multiplicador[metodo] * (len(recta)) / fs


t60 = t_60(schroeder_db, 48000, 't_30')


#Funcion C80
def c_80(vector, fs):
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_80 = int((80/1000) * fs)
    return 10 * np.log10(integral[muestra_80] / (integral[-1] - integral[muestra_80]))
    
#Funcion D50
def d_50(vector, fs):
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_50 = int((50/1000) * fs)
    return 100 * integral[muestra_50] / (integral[-1])











