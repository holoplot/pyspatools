from ..helpers.const import *
import numpy as np
import scipy.signal
import pyloudnorm

__all__ = ['correlation', 'latency', 'lkfs', 'pcm_to_float']


def pcm_to_float(x, bitdepth):
    if bitdepth.upper() == 'PCM24':
        ymax = PCM24_SIGNED_MAX
    elif bitdepth.upper() == 'PCM16':
        ymax = PCM16_SIGNED_MAX
    elif bitdepth.upper() == 'PCM32':
        ymax = PCM32_SIGNED_MAX
    elif bitdepth.upper() == 'PCM8':
        ymax = PCM8_SIGNED_MAX
    else:
        return x
    result = np.ndarray(shape=x.shape, dtype=np.float)
    for i in range(len(x)):
        result[i, :] = x[i, :] / ymax
    return result


def latency(x, threshold=0, offset=0) -> list:
    """
    Iterate through each channel and returns the sample

    Parameters
    ----------
    x : numpy.ndarray
        Audio signals
    threshold : int or float
        The signal threshold value that consider a valid signal
    offset : int
        A sample offset as the before this offset maybe cause by other factor (if known) than the latency

    Returns
    -------
    results : list
        A list of latency in sample per channel

    """
    if not isinstance(offset, int) or offset < 0:
        raise AttributeError("offset must be positive int")
    results = []
    for ch in x:
        where = np.where(np.abs(ch[offset:]) > threshold)
        try:
            results.append(where[0][0])
        except IndexError:
            raise ValueError(f"Couldn't find starting signal with given threshold {threshold} ")
    return results


def lkfs(x, sr, bitdepth):
    """
    Loudness, K-weighted, relative to full scale implementation based on ITU-R BS.1770 standard.
    Credit: https://github.com/csteinmetz1/pyloudnorm

    Parameters
    ----------
    x : numpy.ndarray
        Input multichannel audio signals.
    sr : int
        Sampling rate

    Returns
    -------
    list
        A list of scala for a single loudness value in dB per channel

    """
    x_float = pcm_to_float(x, bitdepth)
    meter = pyloudnorm.Meter(sr)
    return [meter.integrated_loudness(ch) for ch in x_float]


def correlation(a, b, mode='full', method='auto'):

    """
    Compute the correlation and its corresponding lags
    Parameters
    ----------
    a : numpy.ndarray
    b : numpy.ndarray
    mode : str
    method : str

    Returns
    -------
    corr
    lags

    """

    corrs = []
    lags = []
    for ch_a, ch_b in zip(a, b):
        corrs.append(scipy.signal.correlate(ch_a, ch_b, mode=mode, method=method))
        lags.append(scipy.signal.correlation_lags(ch_a.size, ch_b.size, mode=mode))
    return np.array(corrs, dtype=a.dtype), np.array(lags, dtype=np.int32)

