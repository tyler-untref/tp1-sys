#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:13:07 2022

@author: pedro.f.ridano
"""

#Función cálculo de parámetros acústicos
def parametros_acusticos(vector, fs, parametro, metodo=None):
    '''
    Calcula los parámetros acústicos de una señal procesada.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.
    fs : int
        frecuencia de muestreo de la señal.
    parametro : string
        .parametro acustico a ser calculado:
            puede ser 'edt', 't_60', 'd_50', 'c_80'.
    metodo : string, optional
        metodo a partir del cual se calcula el T60
        puede ser: 't_10', 't_20', 't30'. The default is None.

    Returns
    -------
    salida : float
        valor del parametro acustico calculado.

    '''
    if parametro == 'edt':
        salida = edt(vector, fs)
    if parametro == 't_60':
        salida = t_60(vector, fs, metodo)
    if parametro == 'd_50':
        salida = d_50(vector, fs)
    if parametro == 'c_80':
        salida = c_80(vector, fs)
    return salida    
