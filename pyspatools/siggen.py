import numpy as np
from scipy import signal

from .helpers import lin_map
from .helpers.const import *
from .signal import AudioSignal

__all__ = ['cos', 'sin', 'sawtooth', 'pink']


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


def _normalize(x):
    """Return the normalized input array"""
    # d is a (n x dimension) np array
    x -= np.min(x, axis=0)
    x /= np.ptp(x, axis=0)
    return x


def _convert_dtype(sig, dtype):
    if dtype.lower() == 'float32':
        sig = sig.astype(np.float32)
    elif 'pcm' in dtype.lower():
        out_min, out_max = _get_pcm_range(dtype)
        sig = lin_map(x=sig, in_min=-1., in_max=1., out_min=out_min, out_max=out_max)
        sig = sig.astype(np.int32)

    return sig.reshape((1, sig.shape[0]))


def cos(freq=440, amp=1.0, dur=1.0, sr=48000, dtype='float32'):
    """
    Cosine signal generator

    Parameters
    ----------
    freq : int, float
        Signal will have only 1 consistent frequency
    amp_db : int, float
        Amplitude
    dur : float
        Duration in seconds
    sr : int
        Sampling rate
    dtype : str, optional
        float32, float64, PCM16, PCM24, PCM32

    Returns
    -------
    numpy.ndarray
        The signal array
    """
    sig = amp * np.cos(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)))
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)


def sin(freq=440, amp=1.0, dur=1.0, sr=48000, dtype='float32'):
    """
    Sine signal generator

    Parameters
    ----------
    freq : int, float
        Signal will have only 1 consistent frequency
    amp : int, float
        Amplitude
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
    sig = amp * np.sin(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)))
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)


def sawtooth(freq=440, amp=1.0, dur=1.0, sr=44800, dtype='float32'):
    """
    Generate sawtooth wave signal.

    Parameters
    ----------
    freq : int, float
        signal frequency (Default value = 440)
    amp : int, float
        signal amplitude
    dur : int, float
        duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
    sr : int
        sampling rate
    channels : int
        number of channels (Default value = 1)
    Returns
    -------
    numpy.ndarray
    """
    sig = amp * signal.sawtooth(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)))
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)


def pink(amp=1.0, dur=1.0, sr=48000, dtype='float64'):
    """
    Generate pink noise

    Parameters
    ----------
    type : string
        type of noise, currently available: 'white' and 'pink' (Default value = 'white')
    amp : int, float
        signal amplitude (Default value = 1.0)
    dur : int, float
        duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
    num_rows : int
        number of rows (samples). dur and num_rows only use one of the two(Default value = None)
    sr : int
        sampling rate (Default value = 44100)
    channels : int
        number of channels (Default value = 1)
    cn : list of string
        channel names as a list. The size needs to match the number of channels (Default value = None)
    label : string
        identifier of the object (Default value = "square")
    Returns
    -------
    Asig
    """
    # Based on Paul Kellet's method
    b0, b1, b2, b3, b4, b5, b6 = 0, 0, 0, 0, 0, 0, 0
    sig = []
    length = int(dur * sr)
    for i in range(length):
        white = np.random.random() * 1.98 - 0.99
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        b3 = 0.86650 * b3 + white * 0.3104856
        b4 = 0.55000 * b4 + white * 0.5329522
        b5 = -0.7616 * b5 - white * 0.0168980
        sig.append(b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362)
        b6 = white * 0.115926
    sig = _normalize(sig) * amp
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)
