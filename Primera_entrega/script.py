import numpy as np
import soundfile as sf 
import sounddevice as sd
import pandas as pd
# import matplotlib.pyplot as plt

#Parte 1: Función de sintetización de ruido rosa

#Primera consigna:

def ruidoRosa_voss(t, fs=int(44100), ncols=int(16)):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    
    .. Nota:: si 'ruidoRosa.wav' existe, este será sobreescrito
    
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
    
    Ejemplo
    -------
    Generar un `.wav` desde un numpy array de 10 segundos con ruido rosa a una 
    frecuencia de muestreo de 44100 Hz.
    
        import numpy as np
        import soundfile as sf
        from scipy import signal
        
        ruidoRosa_voss(10)
    """
    
    nrows = int(float(t)*fs)

    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # el numero total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)

    ## Centrado de el array en 0
    total = total - total.mean()

    ## Normalizado
    valor_max = max(abs(max(total)), abs(min(total)))
    total = total / valor_max

    # Agregar generación de archivo de audio .wav
    sf.write('ruido_rosa.wav', total, fs)

    return total

ruido_rosa = ruidoRosa_voss(10)

#Segunda consigna:

# def dominio_temporal(t,fs):

#     #Eje x: tiempo
#     eje_x = np.linspace(0,t+1,t*fs)
#     plt.xlabel("Tiempo (s)")
    
#     #Eje y: amplitud normalizada
#     eje_y = ruido_rosa
#     plt.ylabel("Amplitud Normalizada")
    
#     plt.title("Gráfico: dominio temporal de la señal")
#     plt.plot(eje_x, eje_y)
#     return plt.show()

# dominio_temporal(10,44100)

#Tercera consigna:

    
#Cuarta consigna:

sd.play(ruido_rosa, 44100)    


#Parte 2: Funcion de generación de sine sweep logaritmico + filtro inverso

def sine_sweep(t, w1, w2, fs=44100):
    """
    Genera un Sine Sweep utilizando las ecuaciones de Meng, Q.
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del sine sweep.
    w1 : int
         Frecuencia angular inferior en rad/s
    w2 : int
        Frecuencia angular superior en rad/s.
    fs : int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.    

    Returns: NumPy array
        Datos de la señal generada.
    -------
    None.

    """
        
    R = np.log(w2/w1)
    L = t/R
    K = (t*w1)/R
    
    return np.sin(K*(np.exp(t/L)-1))

sine_sweep = sine_sweep(10, 20, 20000)


#Parte 3: Función adquisicion y reproducción









