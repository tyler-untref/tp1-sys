#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:47:20 2022

@author: pedro.f.ridano
"""
import numpy as np
import matplotlib.pyplot as plt

#Funciones gráfico de dominio temporal
def dominio_temporal(data):
    """
    Grafica una señal de entrada en función del tiempo. 
    
    Parametro
    ----------
    data: tupla. 
          El primer valor es un numpy array de 1D y el segundo es su fs.
             
    """
    vector = data[0]
    fs = data[1]
    
    #Eje x: tiempo
    eje_x = np.linspace(0,(len(vector)/fs),len(vector))
    plt.xlabel("Tiempo (s)")
    
    #Eje y: amplitud normalizada
    eje_y = vector
    eje_y = vector
    plt.ylabel("Amplitud normalizada")
    # plt.yscale('log')
    
    plt.plot(eje_x, eje_y)
    plt.show()      
    
def dominio_temporal_2(data_1, data_2, fs):
    """
    Grafica dos señales de entrada en función del tiempo de forma simultánea. 
    
    .Nota: Ambas señales deberán tener la misma frecuencia de sampleo.
    
    Parámetro
    ----------
    data_1: Numpy Array
        señal a ser graficada.
    data_2: Numpy Array
        señal a ser graficada.
    fs: int
      frecuencia de sampleo de las señales.      
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
    

    