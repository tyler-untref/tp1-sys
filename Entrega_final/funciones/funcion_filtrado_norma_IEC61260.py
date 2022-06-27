#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:04 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 
from scipy import signal
    
def filtrado(archivo, rango, fc, orden=4):
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
    G = 1
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
    print('fc_hz es: ', fc_Hz)
    
    #Calculo los extremos de la banda a partir de la frecuencia central
    f1_Hz = fc_Hz / factor
    f2_Hz = fc_Hz * factor
    
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

if __name__ == '__main__':
    from funciones_grafico_de_dominio_temporal import dominio_temporal
    
    filtrado_yeah = filtrado('usina_main_s1_p5.wav', 'o' , 1000)
    filtrado_yeah_mono = (250000*filtrado_yeah[1])[:, 0]
    dominio_temporal((filtrado_yeah_mono, 48000))

