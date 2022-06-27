#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 10:25:20 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 
import sounddevice as sd
from scipy import signal


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
  
    
#Funcion de obtención de la respuesta al impulso
def obtencion_RI(ss_grabado, wav_f_i):
    '''
    Calcula la respuesta al impulso a partir de la convolución entre el sine 
    sweep grabado en el recinto y el filtro inverso ya generado para ese sine
    sweep.
    
    .Nota: La función genera un archivo de audio .wav con la respuesta al 
    impulso llamado 'respuesta al impulso.wav' 

    Parametros
    ----------
    ss_grabado : string
        nombre del archivo .wav con el sine sweep grabado en el recinto.
    wav_f_i : string
        nombre del archivo .wav del filtro inverso generado para ese sine sweep.

    Returns
    -------
    resp_imp : Numpy Array
        con la respuesta al impulso calculada.
    Además, graba un archivo .wav con la respuesta al impulso generada.
    '''
    sine_sweep, fs = sf.read(ss_grabado)
    
    filtro_inverso, fs = sf.read(wav_f_i)    
    
    resp_imp = signal.fftconvolve(sine_sweep, filtro_inverso) 
    
    sf.write('respuesta_al_impulso.wav', resp_imp, fs)
    
    return resp_imp
   

#Función conversión a escala logarítmica normalizada
def conversion_log_norm(vector):
    '''
    Convierte una señal de entrada lineal en una señal con escala logaritmica
    normalizada.
    
    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso con escala lineal.
    Returns
    -------
    vector_log : Numpy Array
        respuesta al impulso con escala logaritmica normalizada.
    '''
    vector_max = max(np.abs(vector))
    vector_log = 10*np.log10(np.abs(vector)/(vector_max))
    return vector_log   
  

    
def filtrado(vector, fs, rango, fc, orden=4):
    """
    Aplica un filtro pasabanda a la señal de entrada, por octavas o por tercios 
    de octava según la norma IEC61260.
    
    Parametros
    ----------
    vector: Numpy Array
       array de la señal filtrar.
    fs : int
        frecuencia de muestreo de la señal de entrada
    rango: string
       pudiendo ser 'o' para un filtrado por bandas de octava
       o 't' para un filtrado por tercios de octava.
    fc: int
     frecuencia central en Hz.
    orden: int
       orden del filtro.
        
    Returns:
    -------
    lista: lista
      -el primer elemento es la frecuencia central fc de la banda que 
       fue filtrada 
      -el segundo elemento es un Numpy Array de la señal filtrada para esa fc.
      -el tercer elemento es la respuesta en frecuencia del filtro 
       para esa fc.
      -el cuarto elemento es un Numpy Array con las frecuencias angulares.
      
    """
    
    if np.shape(vector) == (len(vector), 2):
        vector = vector[:, 1]
        
    G = 1
    #Octava: G = 1/2, 
    #1/3 de Octava: G = 1/6
    
    if rango == 'o':
        G = 1/2
        if rango == 't':
            G = 1/6
        
    
    factor = np.power(2, G)
    fc_Hz = fc
    print('fc_hz es: ', fc_Hz)
    
    #Calculo los extremos de la banda a partir de la frecuencia central
    f1_Hz = fc_Hz / factor
    f2_Hz = fc_Hz * factor
    
    print('Frecuencia de corte inferior: ', round(f1_Hz), 'Hz')
    print('Frecuencia de corte superior: ', round(f2_Hz), 'Hz')
    
    #Extraemos los coeficientes del filtro 
    b,a = signal.iirfilter(orden, [2*np.pi*f1_Hz,2*np.pi*f2_Hz], btype='band', 
                           analog=True, ftype='butter') 
    
    #Defino sos para aplicar el filtro 
    sos = signal.iirfilter(orden, [f1_Hz,f2_Hz], btype='band', analog=False, 
                           ftype='butter', fs=fs, output='sos')
    
        
    #Dados los coeficientes del filtro, calculo la resp en frecuencia
    #w: array con las frecuencias angulares con las que h fue calculado.
    #h: array con la respuesta en frecuencia.
    w, h = signal.freqs(b,a)
    
    #Aplico el filtro al audio
    filt = signal.sosfilt(sos, vector)
    
    #defino las salidas
    lista = [fc_Hz, 9*filt, h, w]
    return lista    
  
    
  
if __name__ == '__main__':
    dic = carga(['usina_main_s1_p5.wav', 'elveden_hall_suffolk_england.wav'])
    array_usina = dic['usina_main_s1_p5.wav']
    array_usina = array_usina[0]
    array_elveden_hall = dic['elveden_hall_suffolk_england.wav']
    array_elveden_hall = array_elveden_hall[0]
  
    resp_imp_ss_grabado = obtencion_RI('sine_sweep_generado_y_grabado.wav', 'filtro_inverso_generado.wav')
    
    resp_imp_ss_grabado_log = conversion_log_norm(resp_imp_ss_grabado)
    
    filtrada = filtrado(resp_imp_ss_grabado_log, 48000, 'o' , 1000)
    sd.play(filtrada[1])
    
    

    
    
    
    
    
    
    