#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:01 2022

@author: pedro.f.ridano
"""

import soundfile as sf 
import sounddevice as sd
import scipy.io.wavfile as sio

#Funcion de reproducción y adquisición
def reproduccion_adquisicion(archivo):
    '''
    Reproduce la señal de audio utilizando la salida física de audio 
    (seleccionada por default) y graba la reproducción de la misma, utilizando 
    la entrada física (seleccionada por default) de audio de la computadora.
    
    .Nota: La función escribe un archivo.wav llamado 'sine_sweep_generado_y_
    reproducido.wav'
    
    Parámetro
    ----------
    archivo : string
        nombre del archvio .wav a ser reproducido.

    Returns
    -------
    grabacion : Numpy Array
        devuelve un vector con la señal grabada.

    '''
    #, entrada='digital input', salida='digital output', lat_entrada='high', lat_salida='high')
                              
                             
    vector, fs = sf.read(archivo)
    # sd.default.device = (entrada, salida)
    # sd.default.latency = (lat_entrada, lat_salida)
    grabacion = sd.playrec(vector, fs, channels=2)
    sio.write('sine_sweep_generado_y_reproducido.wav', fs, grabacion)
    return grabacion





