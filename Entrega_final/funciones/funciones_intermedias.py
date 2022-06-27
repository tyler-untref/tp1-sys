#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:02 2022

@author: pedro.f.ridano
"""

import numpy as np


#Funciones intermedias
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


       

