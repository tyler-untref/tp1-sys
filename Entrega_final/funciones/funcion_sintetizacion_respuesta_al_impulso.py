#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:01 2022

@author: pedro.f.ridano
"""

import numpy as np
import soundfile as sf 


#Función de sintetización de respuesta al impulso
def sintetizacion_RI(diccionario, fs):
    '''
    Genera una respuesta al impulso sintetizada a partir de valores previamente
    definidos de T_60
    
    .Nota: La función escribe un archivo.wav llamado ´sintetizacion RI.wav´.

    Parametros
    ----------
    diccionario : diccionario
         un diccionario cuyas claves son las frecuencias centrales 
         y sus valores son los T60 correspondientes a cada frecuencia
         central de banda de octava como establece la norma IEC61260 
         (IEC61260,1995).
    fs : int
         frecuencia de muestreo deseada de la señal resultante.  

    Returns
    -------
    suma : Numpy Array
       con la respuesta al impulso sintetizada de valores de T60 definidos 
       para cada frecuencia central y con duracion de t segundos.
    
    '''
    t = 3
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
    
    sf.write('sintetizacion_RI.wav', suma, fs)
    return suma


if __name__ == '__main__':
    #cargo los valores del T60 por banda de octava, calculados a partir del T30, 
    #y sacados de la tabla de OpenAir
    dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
               2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}
    
    #llamo a la funcion
    resp_imp_usina_sint = sintetizacion_RI(dic_t60, 48000)

    









