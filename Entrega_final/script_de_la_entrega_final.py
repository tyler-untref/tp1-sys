#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 19:41:12 2022

@author: pedro.f.ridano
"""

import numpy as np
from numpy.linalg import inv
import soundfile as sf 
import sounddevice as sd
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as sio
import scipy.integrate as integrate


#Funciones gráfico de dominio temporal
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
    plt.ylabel("Amplitud normalizada")
    # plt.yscale('log')
    
    plt.plot(eje_x, eje_y)
    plt.show()      
    
def dominio_temporal_2(data_1, data_2, fs):
    """
    Grafica dos señales de entrada en función del tiempo de forma simultánea. 
    
    .Nota: Ambas señales deberán tener la misma frecuencia de sampleo.
    
    Parámetro
    ----------
    data_1: Numpy Array
        señal a ser graficada.
    data_2: Numpy Array
        señal a ser graficada.
    fs: int
      frecuencia de sampleo de las señales.      
    """
   
    #Eje x_1: tiempo
    eje_x_1 = np.linspace(0,(len(data_1)/fs),len(data_1))
    plt.xlabel("Tiempo (s)")
    
    #Eje x_2: tiempo
    eje_x_2 = np.linspace(0,(len(data_2)/fs),len(data_2))
    plt.xlabel("Tiempo (s)")
    
    #Eje y_1: amplitud normalizada
    eje_y_1 = data_1
    plt.ylabel("Amplitud [dB]")
    
    #Eje y_2: amplitud normalizada
    eje_y_2 = data_2
    plt.ylabel("Amplitud [dB]")
    
    plt.plot(eje_x_1, eje_y_1, 'orange', label='respuesta al impulso')
    plt.legend()
    plt.plot(eje_x_2, eje_y_2, label='respuesta al impulso suavizada')
    plt.legend()
    plt.show()     
    

#Función sintetización de ruido rosa
def ruidoRosa_voss(t,fs,ncols):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    
    .Nota: si 'ruidoRosa.wav' existe, este será sobreescrito.
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del ruido generado.
    ncols: int
        Determina el número de fuentes aleatorias a agregar.
    fs: int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
    
    returns: NumPy array
        Datos de la señal generada.
    
    """

    nrows = int(float(t)*fs)

    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # El número total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)

    # Centrado del array en 0
    total = total - total.mean()

    # Normalizado
    valor_max = max(abs(max(total)), abs(min(total)))
    total = total / valor_max

    # Agregar generación de archivo de audio .wav
    sf.write('ruido_rosa.wav', total, fs)

    return total

ruido_rosa = ruidoRosa_voss(10,44100,16)


#Función generación de sine sweep logaritmico + filtro inverso
def gen_sine_sweep(t, f1, f2, fs):
    """
    Genera un Sine Sweep utilizando las ecuaciones de Meng, Q. y su filtro
    inverso. 
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del sine sweep.
    f1 : int
          Frecuencia inferior en Hz.
    f2 : int
        Frecuencia superior en Hz.
    fs : int
        Frecuencia de muestreo en Hz de la señal.      

    Returns: Tupla de dos elementos, con 1 Numpy array en cada elemento.
        El primer elemento corresponde al array del sine sweep generado y el 
        segundo elemento corresponde al filtro inverso del sine sweep generado.
    -------
    
    """
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2
    tiempo = np.linspace(0,t,t*fs)
    R = np.log(w2/w1)
    L = t/R
    K = (t*w1)/R

    frec_inst = (K/L)*np.exp(tiempo/L)
    mod_m = w1/(2*np.pi*frec_inst)
    
    sine_sweep_t = np.sin(K*(np.exp(tiempo/L) - 1))
    
    filtro_inverso = mod_m * sine_sweep_t[::-1]
    filtro_inverso = filtro_inverso/max(filtro_inverso)
    
    # Generación del archivo .wav
    sf.write('sine_sweep.wav', sine_sweep_t, fs)
    sf.write('filtro_inverso.wav', filtro_inverso, fs)
    
    return sine_sweep_t, filtro_inverso 

t =10
sine_sweep = gen_sine_sweep(t, 20, 20000, 48000)


#test con la convolución:    
convolucion = signal.convolve(sine_sweep[0], sine_sweep[1])
sf.write('convolucion.wav', convolucion, 48000)
dominio_temporal((convolucion, 48000))


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


audio = reproduccion_adquisicion('sine_sweep.wav')



#Funcion de obtención de la respuesta al impulso
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


#Generacion de filtro inverso del sine_sweep_grabado aula de informatica

grabado, fs = sf.read('ss-grabado.wav')
grabado = grabado[(fs):(31*fs)]
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
sf.write('filtro_inverso.wav', filtro_inverso, 44100)


respuesta_al_imp_obtenida = obtencion_RI('ss-grabado.wav', filtro_inverso)



#Función de sintetización de respuesta al impulso
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
    fs = 48000
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

#cargo los valores del T60 por banda de octava, calculados a partir del T30, 
#y sacados de la tabla de OpenAir
dic_t60 = {31.25: 2.15, 62.5: 1.48, 125: 1.63, 250: 1.91, 500: 2.08, 1000: 2.09, 
            2000: 1.82, 4000: 1.6, 8000: 1.18, 16000: 1.11}


#-----------------------------------------------------------------------------
#Funciones intermedias
def valor_min(vector):
    '''
    Halla el valor mínimo de una señal   

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    minimo : float
        valor minimo de la señal

    '''
    vector = abs(vector)
    minimo = 1
    for i in vector:
        if i < minimo and i != 0:
            minimo = i
    
    return minimo      

def reemplaza_ceros(vector):
    '''
    Reemplaza los ceros de una señal por el valor mínimo de la misma.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    vector : Numpy Array
        señal con los ceros anteriores reemplazados por el valor mínimo de la misma.

    '''
    minimo = valor_min(vector)
    for i in range(len(vector)):
        if vector[i] == 0 :
            vector[i] = minimo    
    return vector

def elimina_inicio(vector):
    '''
    Elimina el ruido y silencios al comienzo de una señal exponencial decreciente.

    Parametros
    ----------
    vector : Numpy Array
        señal a ser eliminado su ruido inicial.

    Returns
    -------
    vector: Numpy Array
        señal sin el ruido o silencio inicial.

    '''
    indice_max = np.argmax(vector)
    vector = vector[indice_max:]
   
    return vector 

def elimina_valores(vector, valor):
    '''
    Elimina un valor específico de la señal de entrada.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.
    valor : int
        valor entero que se desea eliminar de la señal.

    Returns
    -------
    salida : Numpy Array
        señal sin el valor seleccionado.

    '''
    output = []
    for val in vector:
         if val != valor:
              output.append(val)
    salida = np.array(output)
    return salida

def corta_inf(vector):
    '''
    Elimina valores de tipo inf. de la señal de entrada.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.

    Returns
    -------
    salida : Numpy Array
        señal sin los valores de tipo inf.

    '''
    output = []
    for val in vector:
         if np.isfinite(val):
              output.append(val)
    salida = np.array(output)
    return salida

#Funciones intermedias para calcular los parámetros acústicos según Norma ISO 3382

#Función para calcular inicial, final
def hallar_caida(vector, valor_caida):
    '''
    Calcula la pre-imagen de la señal al decaer un valor determinado en amplitud.

    Parametros
    ----------
    vector : Numpy Array
        señal de entrada.
    valor_caida : int
        valor de caída de interés.

    Returns
    -------
    abcisa_caida : int
        muestra de la señal en la que la misma decae al valor de interés.

    '''
    abcisa_caida = 0
    for i in range(0, len(vector)):
        if vector[i] <= -valor_caida:
            abcisa_caida = i
            break
    return abcisa_caida 

def regresion_entre(vector_x, vector_y, inicial, final):
    '''
    Realiza la regresión lineal entre los valores de la imagen de una señal 
    y los valores de su pre-imagen dentro de un intervalo dado [final-inicial]

    Parametros
    ----------
    vector_x : Numpy Array
        vector correspondiente a los valores de la pre-imagen. Por ej: eje temporal
    vector_y : Numpy Array
        vector correspondiente a los valores de la imagen.
    inicial : int
        primer valor del intervalo.
    final : int
        último valor del intervalo.

    Returns
    -------
    salida: Numpy Array
        señal que representa la recta de regresión entre el intervalo dado.

    '''
    coefs = regr_cuad_min(vector_x, vector_y)
    salida = np.polyval(coefs, vector_x)
    salida = salida[inicial:final]
    return salida
       
#-----------------------------------------------------------------------------

#Función conversión a escala logarítmica normalizada
def conversion_log_norm(vector):
    '''
    Convierte una señal de entrada lineal en una señal con escala logaritmica
    normalizada.
    
    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso con escala lineal.

    Returns
    -------
    vector_log : Numpy Array
        respuesta al impulso con escala logaritmica normalizada.

    '''
    vector_max = max(np.abs(vector))
    vector_log = 10*np.log10(vector/(vector_max))
    return vector_log


#Función de carga de archivos de audio (dataset)
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
         un diccionario cuyas claves son los nombres de los archivos .wav 
         y cuyos valores son tuplas. 
         En la primera posicion de cada tupla se encuentra el array de Numpy 
         con el vector de audio y en la segunda posición la frecuencia de 
         muestreo de la señal.
    '''
    diccionario = {} 
    for i in lista:
        array, fs = sf.read(i)
        diccionario[i]=(array, fs)
    return diccionario
    
dic = carga(['usina_main_s1_p5.wav', 'councilchamber_s1_r4_ir_1_96000.wav'])



#Función filtrado Norma IEC61260

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


#Función suavizado de la señal (filtro de media móvil)    
def filtro_media_movil(archivo):
    '''
    Aplica el filtro de media móvil a una señal de entrada, utilizando una
    ventana fija de 501 muestras. 

    Parametros
    ----------
    archivo : numpy array
        señal a ser filtrada.

    Returns
    -------
    suavizada : numpy array
        señal filtrada.

    '''
    ventana = 501
    suavizada = np.zeros(len(archivo)-ventana)
    for i in range(0, len(archivo)-ventana):
        suavizada[i] = np.mean(archivo[i:i+ventana])
    suavizada = suavizada/max(suavizada)    
    return suavizada


#Función integral de Schroeder
def integral_de_schroeder(resp_imp, fs):
    """
    Calcula la integral de Schroeder de la señal (respuesta al impulso) para así 
    trabajar con una señal más adecuada al momento de calcular los parámetros 
    acústicos.

    Parametros
    ----------
    resp_al_imp: Numpy Array
        respuesta al impulso ya suavizada.     
   
    Returns 
    -------
    resultado: Numpy Array
       integral de schroeder de la señal.

    """
    t = len(resp_imp)/fs
    n = len(resp_imp)
    extr_inf = 0
    extr_sup = t 
    delta_int = ((extr_sup - extr_inf)*fs)/n
    
    resp_imp = resp_imp ** 2
    resultado = delta_int*np.cumsum(resp_imp)
     
    return resultado
    
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


#Procesamiento de la señal resp_imp_usina
primera_tupla = list(dic.values())[0] #busco la resp al imp usina, en el diccionario
resp_imp_usina = primera_tupla[0] #array de la resp al imp de la usina
resp_imp_usina_fs = primera_tupla[1] #busco la fs del array
resp_imp_usina = resp_imp_usina[:,0] #trabajo con la señal mono
resp_imp_usina = elimina_inicio(resp_imp_usina) #elimino el ruido inicial
resp_imp_usina = abs(resp_imp_usina) #trabajo con el modulo
resp_imp_usina_media_movil = filtro_media_movil(resp_imp_usina) #suavizo la señal
 

#Cálculo de la integral de schroeder
sch_resp_imp_usina = integral_de_schroeder(resp_imp_usina_media_movil, resp_imp_usina_fs)
schroeder_resp_imp_usina = sch_resp_imp_usina[-1] - sch_resp_imp_usina #integral hasta el final - integral en cada valor 
schroeder_resp_imp_usina = schroeder_resp_imp_usina[:-1] #elimino el ultimo valor (porque es =0)
schroeder_resp_imp_usina_db = conversion_log_norm(schroeder_resp_imp_usina)

#Cálculo de la recta de regresión lineal
abcisa_usina = np.linspace(0, len(schroeder_resp_imp_usina_db)/resp_imp_usina_fs, len(schroeder_resp_imp_usina_db))
coefs_usina = regr_cuad_min(abcisa_usina, schroeder_resp_imp_usina_db) 
interpolacion = np.polyval(coefs_usina, abcisa_usina)



##Procesamiento de la señal resp_imp_usina_sint
resp_imp_usina_sint = resp_imp_usina_sint
resp_imp_usina_sint_fs = 48000 #armo esta variable para trabajar mas cómodo
resp_imp_usina_sint_media_movil = filtro_media_movil(resp_imp_usina_sint) #suavizo la señal

#Cálculo de la integral de schroeder
sch_resp_imp_sint = integral_de_schroeder(resp_imp_usina_sint_media_movil, resp_imp_usina_sint_fs)
schroeder_resp_imp_sint = sch_resp_imp_sint[-1] - sch_resp_imp_sint #integral hasta el final - integral en cada valor 
schroeder_resp_imp_sint = schroeder_resp_imp_sint[:-1] #elimino el ultimo valor (porque es =0)
schroeder_resp_imp_sint_db = conversion_log_norm(schroeder_resp_imp_sint)

#Cálculo de la recta de regresión lineal
abcisa_usina_sint = np.linspace(0, len(schroeder_resp_imp_sint_db)/resp_imp_usina_sint_fs, len(schroeder_resp_imp_sint_db))
coefs_usina_sint = regr_cuad_min(abcisa_usina_sint, schroeder_resp_imp_sint_db) 
interpolacion_usina_sint = np.polyval(coefs_usina_sint, abcisa_usina_sint)





#Funciones para cálculo de parámetros acústicos

#Función cálculo del EDT
def edt(vector, fs):
    '''
    Calcula el parámetro acústico EDT a partir de un vector (respuesta al impulso) 

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal de entrada.

    Returns
    -------
    valor_edt = int
        valor del parámetro EDT.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    inicial = 0
    final = hallar_caida(vector, 10)
    recta = regresion_entre(tiempo, vector, inicial, final)
    valor_edt = len(recta) / fs 
    return valor_edt


#Funcion T60 a partir del T10, T20, T30
def t_60(vector, fs, metodo):
    '''
    Calcula el T60 a partir del T10, T20 o T30.

    Parametros
    ----------
    archivo : Numpy Array
        recta obtenida a partir de la regresión lineal por cuadrados minimos. 
    fs : int
        fs del archivo original.
    metodo : string
        tipo de tiempo de reverberacion a partir del cual la función calculará 
        el T_60. 
        Por ejemplo, si metodo = 't_10' la función calcula el T60 a partir del T10, 
                     si 't_20' la función calcula el T60 a partir del T20,
                     si 't_30' la función calcula el T60 a partir del T30.

    Returns
    -------
    resultado : float
        valor del parámetro T60.
    

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    multiplicador = {'t_10': 6, 't_20': 3, 't_30': 2}
    if metodo == 't_10':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 15)
        recta = regresion_entre(tiempo, vector, inicial, final)
        
    if metodo == 't_20':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 25)
        recta = regresion_entre(tiempo, vector, inicial, final)
        
  
    if metodo == 't_30':
        inicial = hallar_caida(vector, 5)
        final = hallar_caida(vector, 35)
        recta = regresion_entre(tiempo, vector, inicial, final)
    
    resultado = multiplicador[metodo] * (len(recta)) / fs
    
    return resultado

#prueba
#t60 = t_60(schroeder_db, 48000, 't_20')


#Funcion D50
def d_50(vector, fs):
    '''
    Calcula el parámetro acústico D50 a partir de una respuesta al impulso.

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal.

    Returns
    -------
    resultado: float
        valor del parámetro D50.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_50 = int((50/1000) * fs)
    resultado = 100 * integral[muestra_50] / (integral[-1])
    
    return resultado

#Funcion C80
def c_80(vector, fs):
    '''
    Calcula el parámetro acústico C80 a partir de una respuesta al impulso.

    Parametros
    ----------
    vector : Numpy Array
        señal del tipo respuesta al impulso.
    fs : int
        frecuencia de muestreo de la señal.

    Returns
    -------
    resultado: float
        valor del parámetro acústico C80.

    '''
    tiempo = np.linspace(0, len(vector)/fs, len(vector))
    
    delta_t = tiempo[1] - tiempo[0]
    integral = delta_t * np.cumsum(vector ** 2) #integral hasta
    muestra_80 = int((80/1000) * fs)
    resultado = 10 * np.log10(integral[muestra_80] / (integral[-1] - integral[muestra_80]))
    
    return resultado
    

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







