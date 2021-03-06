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
from scipy import signal
import matplotlib.pyplot as plt

#traigo la funcion dominio_temporal de la primer entrega:    
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
    
    plt.plot(eje_x, eje_y)
           

#Primera consigna: Función de carga de archivos de audio
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
         un diccionario cuyas claves son los nombres de los archivos 
         .wav y cuyos valores son tuplas. 
         En la primera posicion de cada tupla se encuentra el array 
         de Numpy con el vector de audio y en la segunda posición la
         frecuencia de muestreo de array.
    '''
    diccionario = {} 
    for i in lista:
        array, fs = sf.read(i)
        diccionario[i]=(array, fs)
    return diccionario
    
dic = carga(['usina_main_s1_p5.wav', 'minster1_000_ortf_48k.wav'])
            

#Segunda Consigna: Función de sintetización de respuesta al impulso
def sintetizacion_R_I(diccionario):
    '''
    Genera una respuesta al impulso sintetizada a partir de valores previamente
    definidos de T_60.
    
    .Nota: La función escribe un archivo.wav llamado ´sintetizacion RI.wav´.

    Parametros
    ----------
    diccionario : diccionario
         un diccionario cuyas claves son las frecuencias centrales 
         y sus valores son los T60 correspondientes a cada frecuencia
         central de banda de octava como establece la norma IEC61260 
         (IEC61260,1995).

    Returns
    -------
    suma : Numpy Array
       con la respuesta al impulso sintetizada de valores de T60 definidos 
       para cada frecuencia central y con duracion de t segundos.
    
    '''
    t = 3
    fs = 44100
    suma = np.zeros(t*fs)
    lista_f = list(diccionario)
    # print(lista_f)
    lista_t60 = list(diccionario.values())
    # print(lista_t60)
    tiempo = np.linspace(0, t, t*fs)
    for i in range(len(diccionario)):
        f_i = lista_f[i]
        t60i = lista_t60[i]
        tau_i = ((-1)*np.log(10**(-3)))/t60i
        tau_i = tau_i
        y_i = np.exp(tau_i*tiempo)*np.cos(2*np.pi*f_i*tiempo)
        # print(suma)
        suma = suma + y_i 
    suma = suma[::-1]
    suma = suma/max(abs(suma))
    
    sf.write('sintetizacion RI.wav', suma, fs)
    return suma


dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
            2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}

sint = sintetizacion_R_I(dic_t60)

    
# Tercera Consigna: Funcion de obtencion de la respuesta al impulso
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

    '''
    sine_sweep, fs = sf.read(ss_grabado)
    
    filtro_inverso, fs = sf.read(wav_f_i)    
    
    resp_imp = signal.fftconvolve(sine_sweep, filtro_inverso) 
    
    sf.write('respuesta al impulso.wav', resp_imp, fs)
    
    return resp_imp


grabado, fs = sf.read('ss-grabado.wav')
grabado_cortado = grabado[(fs):(31*fs)]

sf.write('grabado_cortado.wav', grabado_cortado, 44100)

#generacion de filtro inverso del sine_sweep_grabado aula de informatica
grabado, fs = sf.read('ss-grabado.wav')
grabado = grabado[(fs):(31*fs)]
t =10
w1 = 2*np.pi*88
w2 = 2*np.pi*11314
tiempo = np.linspace(0,len(grabado)/fs,len(grabado))
R = np.log(w2/w1)
L = t/R
K = (t*w1)/R

frec_inst = (K/L)*np.exp(tiempo/L)
mod_m = w1/(2*np.pi*frec_inst)
filtro_inverso = mod_m*((-1)*grabado)
    
# Generación del archivo .wav
sf.write('filtro_inverso.wav', filtro_inverso, fs)

R_I_obtenido = obtencion_RI('grabado_cortado.wav', 'filtro_inverso.wav')


# Cuarta Consigna: Funcion filtros Norma IEC61260

def filtrado(archivo, rango, fc, orden):
    """
    Aplica un filtro pasabanda a la señal de entrada, por octavas o por tercios 
    de octava según la norma IEC61260.
    
    Parametros
    ----------
    archivo: string
       nombre del archivo .wav a filtrar.
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
      -el segundo elemento es array de la señal filtrada para esa fc.
      -el tercer elemento es la respuesta en frecuencia del filtro 
       para esa fc.
      -el cuarto elemento es un array con las frecuencias angulares.
      
    """
    archivo, fs = sf.read(archivo) 
    
    #Octava: G = 1/2, 
    #1/3 de Octava: G = 1/6
    if rango == 'o':
        G = 1/2
        if rango == 't':
            G = 1/6
        else:
            print('decida si quiere un filtrado por octava o por tercios de octava')
    
    factor = np.power(2, G)
    fc_Hz = fc
    
    #Calculo los extremos de la banda a partir de la frecuencia central
    f1_Hz=fc_Hz/factor
    f2_Hz=fc_Hz*factor
    print('Frecuencia de corte inferior: ', round(f1_Hz), 'Hz')
    print('Frecuencia de corte superior: ', round(f2_Hz), 'Hz')
    
    #Extraemos los coeficientes del filtro 
    b,a = signal.iirfilter(orden, [2*np.pi*f1_Hz,2*np.pi*f2_Hz], rs=60, btype='band', 
                           analog=True, ftype='butter') 
    
    #Defino sos para aplicar el filtro 
    sos = signal.iirfilter(orden, [f1_Hz,f2_Hz], rs=60, btype='band', analog=False, 
                           ftype='butter', fs=fs, output='sos')
    
    
    #Dados los coeficientes del filtro, calculo la resp en frecuencia
    #w: array con las frecuencias angulares con las que h fue calculado.
    #h: array con la respuesta en frecuencia.
    w, h = signal.freqs(b,a)
    
    #Aplico el filtro al audio
    filt = signal.sosfilt(sos, archivo)
    
    #defino las salidas
    lista = [fc_Hz, filt, h, w]
    return lista


# #prueba
# filtrado_dic = filtrado('miles_mono.wav', 'o', 3)
# array_filtrado_dic_filt = filtrado_dic[1]
# array_filtrado_dic_h = filtrado_dic[2]
# array_filtrado_w = filtrado_dic[3]

# print(array_filtrado_dic_filt)
# print('pausa')
# print(array_filtrado_dic_h)
# print('pausa')
# print(array_filtrado_w)


# #grafico temporal de la señal filtrada
# dominio_temporal((array_filtrado_dic_filt, 44100))

# #espectro de la respuesta del filtro (no termino de ver si está bien o no)
# plt.semilogx(array_filtrado_w, 20*np.log10(abs(array_filtrado_dic_h)))
# plt.xlabel('Frecuencia')
# plt.ylabel('Amplitud [dB]')
# plt.grid()
# plt.show() 

#me guardo la señal filtrada
#sf.write('miles_filtrado.wav', array_filtrado_dic_filt, 48000)



#Quinta Consigna: Funcion conversion a escala logaritmica normalizada
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

sint_log_norm = conversion_log_norm(sint)




