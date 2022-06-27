#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:33:23 2022

@author: pedro.f.ridano
"""
#librerias
import sounddevice as sd

#funciones
from funciones_etapa_generacion_y_adquisicion import gen_sine_sweep
from funciones_etapa_procesam_y_filtrado import carga
from funciones_etapa_procesam_y_filtrado import obtencion_RI
from funciones_etapa_procesam_y_filtrado import conversion_log_norm
from funciones_etapa_procesam_y_filtrado import filtrado
from funciones_grafico_de_dominio_temporal import dominio_temporal
from funciones_grafico_de_dominio_temporal import dominio_temporal

#genero el ss y el fi y los asigno a variables 
gen_sine_sweep(10, 20, 20000, 48000)

#levanto los .wav generados, y los asigno a variables
dic = carga(['sine_sweep_generado.wav', 'filtro_inverso_generado.wav'])

tupla_ss = dic['sine_sweep_generado.wav']
ss = tupla_ss[0]
fs = tupla_ss[1]

tupla_fi = dic['filtro_inverso_generado.wav']
fi = tupla_fi[0]

#obtengo la respuesta al impulso a partir de los .wav generados
resp_imp_generada = obtencion_RI('sine_sweep_generado.wav', 'filtro_inverso_generado.wav')
dominio_temporal((resp_imp_generada, fs))

#convierto a escala log norm
resp_imp_generada_log = conversion_log_norm(resp_imp_generada)



#filtro segun norma IEC61620 
filtrada_generada = filtrado(resp_imp_generada_log, 48000, 'o', 1000)

dominio_temporal((filtrada_generada[1], fs))

# sonido = sd.play(filtrada_generada[1])










