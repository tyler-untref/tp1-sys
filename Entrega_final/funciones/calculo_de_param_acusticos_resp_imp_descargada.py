#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:35:39 2022

@author: pedro.f.ridano
"""

import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

#funciones
from funciones_etapa_procesam_y_filtrado import carga
from funciones_etapa_procesam_y_filtrado import conversion_log_norm
from funciones_etapa_procesam_y_filtrado import filtrado
from funciones_etapa_suavizado_y_calculo import filtro_media_movil
from funciones_etapa_suavizado_y_calculo import integral_de_schroeder
from funciones_etapa_suavizado_y_calculo import regr_cuad_min
from funciones_etapa_suavizado_y_calculo import edt
from funciones_etapa_suavizado_y_calculo import t_60
from funciones_etapa_suavizado_y_calculo import d_50
from funciones_etapa_suavizado_y_calculo import c_80

#cargo la funcion
resp_imp_usina_d = carga(['usina_main_s1_p5.wav'])
resp_imp_usina_d = resp_imp_usina_d[0]

#la convierto a escala log norm
resp_imp_usina_d_log = conversion_log_norm(resp_imp_usina_d)

#filtro segun norma
filtrada_usina_d = filtrado(resp_imp_usina_d_log, 48000, 'o', 1000)

#filtro media movil
suavizada_usina_d = filtro_media_movil(filtrada_usina_d)

#integral de schroeder
integral_usina_d = integral_de_schroeder(suavizada_usina_d, 48000)

#regresion lineal
#calculo la recta de regresion 
regr_lineal_ss_grabado = regr_cuad_min(integral_usina_d, 48000)

#calculo de parámetros acústicos
valor_edt_usina_d = edt(integral_usina_d, 48000)

valor_t60_usina_d = t_60(integral_usina_d, 48000, 't_20')

valor_d50_usina_d = d_50(integral_usina_d, 48000)

valor_c80_usina_d = c_80(integral_usina_d, 48000)




