import numpy as np
from scipy import signal

from .helpers.const import *
from .signal import AudioSignal

__all__ = ["cos", "sin", "sawtooth", "pink"]


def _normalize(x):
    """Return the normalized input array"""
    # d is a (n x dimension) np array
    x -= np.min(x, axis=0)
    x /= np.ptp(x, axis=0)
    return x


def _convert_dtype(sig, dtype):
    """
    Convert the data type of the signal as per dtype argument.

    Parameters
    ----------
    sig : numpy.ndarray
        The signal array
    dtype : str
        Data type to which the signal is to be converted

    Returns
    -------
    numpy.ndarray
        The signal array with the specified data type
    """
    if dtype == "float32":
        return sig.astype(np.float32)
    elif dtype == "float64":
        return sig.astype(np.float64)
    elif dtype == "PCM16":
        return (sig * 32767).astype(np.int16)
    elif dtype == "PCM24":
        return (sig * 8388607).astype(np.int32)  # Using int32 to represent 24-bit as Python doesn't have a native 24-bit int
    elif dtype == "PCM32":
        return (sig * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def _duplicate_channels(sig, channels):
    """
    Duplicate the signal across the channels
    """
    return np.tile(sig[:, np.newaxis], (1, channels))


def cos(freq=440, amp=1.0, dur=1.0, sr=48000, phase=0, channels=1, dtype="float32"):
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
    phase : int, float
        Phase in degress

    Returns
    -------
    numpy.ndarray
        The signal array
    """
    phase_rad = np.deg2rad(phase)
    sig = amp * np.cos(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)) + phase_rad)
    if channels > 1:
        sig = _duplicate_channels(sig, channels)

    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)


def sawtooth(freq=440, amp=1.0, dur=1.0, sr=44800, phase=0, channels=1, dtype="float32"):
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
    dtype : str
        data type (Default value = "float32")
    phase : int, float
        phase in degress
    Returns
    -------
    numpy.ndarray
    """
    phase_rad = np.deg2rad(phase)
    phase_fraction = phase_rad / (2 * np.pi)  # Convert phase from radians to fraction of cycle
    sig = amp * signal.sawtooth(2 * np.pi * freq * np.linspace(0, dur, int(dur * sr)) + phase_fraction)
    if channels > 1:
        sig = _duplicate_channels(sig, channels)
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)


def pink(amp=1.0, dur=1.0, sr=48000, channels=1, dtype="float32"):
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
    for _ in range(length):
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
    if channels > 1:
        sig = _duplicate_channels(sig, channels)
    sig = _convert_dtype(sig, dtype)
    return AudioSignal(sig=sig, sr=sr)
