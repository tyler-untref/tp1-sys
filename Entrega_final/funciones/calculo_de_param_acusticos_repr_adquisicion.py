#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 07:50:57 2022

@author: pedro.f.ridano
"""

import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

#funciones
from funciones_etapa_generacion_y_adquisicion import reproduccion_adquisicion
from funciones_etapa_procesam_y_filtrado import carga
from funciones_etapa_procesam_y_filtrado import obtencion_RI
from funciones_etapa_procesam_y_filtrado import conversion_log_norm
from funciones_etapa_procesam_y_filtrado import filtrado
from funciones_etapa_suavizado_y_calculo import filtro_media_movil
from funciones_etapa_suavizado_y_calculo import integral_de_schroeder
from funciones_etapa_suavizado_y_calculo import regr_cuad_min
from funciones_etapa_suavizado_y_calculo import edt
from funciones_etapa_suavizado_y_calculo import t_60
from funciones_etapa_suavizado_y_calculo import d_50
from funciones_etapa_suavizado_y_calculo import c_80
from funciones_grafico_de_dominio_temporal import dominio_temporal

#reproduzco el sine_sweep_generado
ss_repr_grabado = reproduccion_adquisicion('sine_sweep_generado.wav')

#obtengo la RI con el f_inv
resp_imp_ss_grabado = obtencion_RI('sine_sweep_generado_y_grabado.wav', 'filtro_inverso_generado.wav')

#convierto a escala log norm
resp_imp_ss_grabado_log = conversion_log_norm(resp_imp_ss_grabado)

dominio_temporal((resp_imp_ss_grabado, 48000))

#filtro segun norma IEC61620 
filtrada_ss_grabado = filtrado(resp_imp_ss_grabado_log, 48000, 'o', 1000)

# suavizo la señal
suavizada_ss_grabado = filtro_media_movil(filtrada_ss_grabado[1])

#integral de schroeder (para qué?)
integral_ss_grabado = integral_de_schroeder(suavizada_ss_grabado, 48000)

# #calculo la recta de regresion 
regr_lineal_ss_grabado = regr_cuad_min(integral_ss_grabado, 48000)

# #calculo de parámetros acústicos
# valor_edt = edt(regr_lineal, fs)

# valor_t60 = t_60(regr_lineal, fs, 't_20')

# valor_d50 = d_50(regr_lineal, fs)

# valor_c80 = c_80(regr_lineal, fs)