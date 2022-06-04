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
    plt.yscale('log')
    
    plt.title("Gráfico: Dominio temporal de la señal")
    plt.plot(eje_x, eje_y)
    return plt.show()      


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
    ventana = 10
    suavizada = np.zeros(len(archivo)-ventana)
    for i in range(0, len(archivo)-ventana):
        suavizada[i] = np.mean(archivo[i:i+ventana])
    #compenzamos el delay concatenando ceros al array
    suavizada = np.hstack([np.zeros(ventana//2), suavizada])
    return suavizada


#prueba
resp_imp, fs = sf.read('usina_main_s1_p5.wav')

# #quito el ruido de fondo inicial ?
# resp_imp = resp_imp[int(0.09)*fs:len(resp_imp)]

resp_imp_media_movil = filtro_media_movil(resp_imp)

##grafico
plt.figure(1)
grafico_1 = dominio_temporal((resp_imp, fs))

plt.figure(2)
grafico_2 = dominio_temporal((resp_imp_media_movil, fs))

##suena
sd.play(resp_imp_media_movil)

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
    extr_sup = t*1000 #lo hago tender a un numero muy grande
    delta_int = (extr_sup - extr_inf)/n
    
    resp_imp = (resp_imp[::-1])**2
    resultado = delta_int*np.cumsum(resp_imp)
    resultado = resultado[::-1]
    
    #grafica resultados
    # length = resultado.shape[0]/fs
    # time = np.linspace(0, length, resultado.shape[0])
    # plt.rcParams['figure.figsize'] = (10,5) # set plot size
    # plt.scatter(time,resultado)
    # plt.xlabel("Tiempo [s]")
    # plt.ylabel("Amplitud [dB]")

    return resultado
    
sch = integral_de_schroeder(resp_imp_media_movil, fs)    
#sch_db = 10.0 * np.log10(sch / np.max(sch))


# length = sch.shape[0]/fs
# time = np.linspace(0, length, sch.shape[0])
# plt.figure(1)
# plt.plot(time, resp_imp_media_movil, 'b--')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('comparacion')
# plt.plot(time, sch, 'r')
# plt.grid()

dominio_temporal((sch, fs))

# sd.play(sch, fs)

#Comentar lo que se observa en la visualización.



#Tercera consigna

#Funcion regresion lineal por cuadrados minimos

#el output es en dBFS .?


#Funciones para calcular los parámetros acústicos según Norma ISO 3382

#Funcion EDT

#Funcion T60 a partir del T10, T20, T30
def T_60(archivo, fs, metodo):
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
                     SI 'T_30' la función calcula a partir del T_30.

    Returns
    -------
    resultado : float
    

    '''
    if metodo == 'T_10':
        for i in archivo:
            if archivo[i] == -5:
                pre_im = i
            if archivo[i] == -15:
                pre_im_caida = i
        muestra_t60 = pre_im_caida-pre_im    
        t_60 = muestra_t60/fs
    
    if metodo == 'T_20':
        for i in archivo:
            if archivo[i] == -5:
                pre_im = i
            if archivo[i] == -25:
                pre_im_caida = i
        muestra_t60 = pre_im_caida-pre_im    
        t_60 = muestra_t60/fs
    return t_60
  
    if metodo == 'T_30':
        for i in archivo:
            if archivo[i] == -5:
                pre_im = i
            if archivo[i] == -35:
                pre_im_caida = i
        muestra_t60 = pre_im_caida-pre_im    
        t_60 = muestra_t60/fs

    return t_60



#prueba 
#voy a armar un array con valores y testearla


#Funcion D50

#Funcion C80











