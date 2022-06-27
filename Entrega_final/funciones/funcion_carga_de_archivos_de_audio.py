#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:03 2022

@author: pedro.f.ridano
"""

import soundfile as sf 

#Función de carga de archivos de audio (dataset)
def carga(lista):
    '''
    Recibe archivos de audio y los guarda en un diccionario.
    
    Parametros
    ----------
    lista: lista 
       formada por strings que refieren a los archivos de audio .wav
       a ser cargados.

    Returns
    -------
    diccionario : diccionario 
         un diccionario cuyas claves son los nombres de los archivos .wav 
         y cuyos valores son tuplas. 
         En la primera posicion de cada tupla se encuentra el array de Numpy 
         con el vector de audio y en la segunda posición la frecuencia de 
         muestreo de la señal.
    '''
    diccionario = {} 
    for i in lista:
        array, fs = sf.read(i)
        diccionario[i]=(array, fs)
    return diccionario
  
if __name__ == '__main__':
    dic = carga(['usina_main_s1_p5.wav', 'councilchamber_s1_r4_ir_1_96000.wav'])