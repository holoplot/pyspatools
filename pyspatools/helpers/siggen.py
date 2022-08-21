import numpy as np
from .helpers import db2amp, lin_map
from .const import *

__all__ = ['cosine', 'zero_padding']


def _get_pcm_range(dtype):
    if dtype.upper() == 'PCM24':
        ymin = PCM24_SIGNED_MIN
        ymax = PCM24_SIGNED_MAX
    elif dtype.upper() == 'PCM16':
        ymin = PCM16_SIGNED_MIN
        ymax = PCM16_SIGNED_MAX
    elif dtype.upper() == 'PCM32':
        ymin = PCM32_SIGNED_MIN
        ymax = PCM32_SIGNED_MAX
    elif dtype.upper() == 'PCM8':
        ymin = PCM8_SIGNED_MIN
        ymax = PCM8_SIGNED_MAX
    else:
        raise AttributeError('Unsupported dtype')
    return ymin, ymax


def cosine(freq=440, amp_db=0.0, dur=1.0, sr=48000, channels=1, dtype='float64'):
    """
    Cosine signal generator

    Parameters
    ----------
    freq : int, float
        Signal will have only 1 consistent frequency
    amp_db : int, float
        Amplitude in decibel
    dur : float
        Duration in seconds
    sr : int
        Sampling rate
    channels : int
        Channel count
    dtype : str, optional
        float32, float64, PCM16, PCM24, PCM32

    Returns
    -------
    numpy.ndarray
        The signal array
    """
    sig = db2amp(amp_db) * np.cos(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)))
    if dtype.lower() == 'float32':
        sig = sig.astype(np.float32)
    elif 'pcm' in dtype.lower():
        out_min, out_max = _get_pcm_range(dtype)
        sig = lin_map(x=sig, in_min=-1., in_max=1., out_min=out_min, out_max=out_max)
        sig = sig.astype(np.int32)

    l = sig.shape[0]
    sig = np.repeat(sig, channels)
    return sig.reshape((channels, l))


def zero_padding(x, front=0, back=0):
    """
    Pad signal per channel base on the amount of samples

    Parameters
    ----------
    x : numpy.ndarray
        Signal array
    front : int, optional
        The amount of 0s to be added to the front
    back : int, optional
        The amount of 0s to be added to the back

    Returns
    -------
    result : numpy.ndarray
        The zero padded signal
    """
    result = np.ndarray((x.shape[0], x.shape[1] + front + back), dtype=x.dtype)
    for i in range(x.shape[0]):
        result[i] = np.pad(x[i], (front, back))
    return result
