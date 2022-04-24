import numpy as np
import scipy
import soundfile as sf
from scipy import signal
from scipy.io.wavfile import write
import sounddevice as sd
import soundfile as sf 
import pandas as pd
# import matplotlib.pyplot as plt

fs=44100

def ruidoRosa_voss(t, fs=44100, ncols=16):

    nrows = int(t*fs)

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
    #wav = write('ruido rosa', int(fs/10), total.astype(np.int16))

    return total

ruido_rosa = ruidoRosa_voss(10)

# amplitude = np.iinfo(np.int16).max
# data = amplitude*ruido_rosa 
# wav = write('ruido rosa', int(fs/10), data.astype(np.int16))

# sf.write('ruido rosa', ruido_rosa, fs,)

#eje_x = np.array([len(10*fs)])


sd.play(ruido_rosa, 44100)
sd.wait()