#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:52:05 2022

@author: pedro.f.ridano
"""

import numpy as np
from numpy.linalg import inv


#Funcion regresión lineal por cuadrados minimos
def regr_cuad_min(vector_x, vector_y, grado=1):
    columnas = grado + 1
    filas = len(vector_x)
    M = np.zeros([filas, columnas])
    for i in range(0, filas):
        for j in range(0, columnas):
            M[i, j] = (vector_x[i])**j
    #print(M) 
    T = M.transpose()
    M_coef = inv((T@M))@(T@vector_y)    
    M_coef = M_coef[::-1]
    return M_coef

