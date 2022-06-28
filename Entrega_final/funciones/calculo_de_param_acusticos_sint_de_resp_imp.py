#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 21:29:18 2022

@author: pedro.f.ridano
"""
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

#funciones
from funciones_etapa_generacion_y_adquisicion import sintetizacion_RI 
from funciones_etapa_procesam_y_filtrado import conversion_log_norm
from funciones_etapa_procesam_y_filtrado import filtrado
from funciones_etapa_suavizado_y_calculo import filtro_media_movil
from funciones_etapa_suavizado_y_calculo import integral_de_schroeder
from funciones_etapa_suavizado_y_calculo import regr_cuad_min
from funciones_etapa_suavizado_y_calculo import edt
from funciones_etapa_suavizado_y_calculo import t_60
from funciones_etapa_suavizado_y_calculo import d_50
from funciones_etapa_suavizado_y_calculo import c_80
from funciones_grafico_de_dominio_temporal import dominio_temporal_2

#cargo los valores del T60 por banda de octava, calculados a partir del T30, 
#y sacados de la tabla de OpenAir
dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
           2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}
    
#llamo a la funcion
resp_imp_sint_usina = sintetizacion_RI(dic_t60, 48000)

#guardo los valores de salida de la funcion sintetizacion
resp_imp_sint_usina = resp_imp_sint_usina[0]

resp_imp_sint_usina = resp_imp_sint_usina

# dominio_temporal((resp_imp_sint_usina, 48000))

#convierto a escala log norm
resp_imp_sint_usina_log = conversion_log_norm(resp_imp_sint_usina)

# dominio_temporal((resp_imp_sint_usina_log, 48000))

#filtro segun norma IEC61620 
filtrada_sint = filtrado(resp_imp_sint_usina_log, 48000, 'o' , 1000)

# suavizo la señal
suavizada_sint = filtro_media_movil(filtrada_sint[1])
# dominio_temporal((suavizada_sint, 48000))

#integral de schroeder (para qué?)
integral_sint = integral_de_schroeder(suavizada_sint, 48000)
integral_sint_log = conversion_log_norm(integral_sint)
# dominio_temporal((integral_sint_log, 48000))

#calculo la recta de regresion 
# regr_lineal = regr_cuad_min(integral_sint_log, 48000)

#usando la funcion de numpy de least squares


# dominio_temporal_2(integral_sint_log, regr_lineal, 48000)

#calculo de parámetros acústicos
# valor_edt = edt(integral_sint, 48000)

# valor_t60 = t_60(integral_sint, 48000, 't_20')

# valor_d50 = d_50(integral_sint, 48000)

# valor_c80 = c_80(integral_sint, 48000)







