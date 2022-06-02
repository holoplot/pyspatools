import numpy as np

__all__ = ['lin_map', 'left_trim', 'db2amp', 'amp2db']


def lin_map(x, in_min, in_max, out_min, out_max):
    """
    Linear mapping a scala
    Parameters
    ----------
    x : float
        input value
    in_min : float
        input's minimum value
    in_max : float
        input's maximum value
    out_min : float
        output's minimum value
    out_max : float
        output's maximum value

    Returns
    -------
    float
        mapped output
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def left_trim(x, start=None):
    """
    Iterate through each channel, find the first non-zero sample. Pick the minimum value and
    left trim all channels to that. This can removed any zero padding.

    Parameters
    ----------
    x : numpy.ndarray
        Multichannel array
    start : int
        If not None, instead of finding the first non zero sample, all channels trim to pos


    Returns
    -------
    numpy.ndarray
        The left trimmed array
    """

    trimmed_waves = []
    first_nonzero_sample = []
    if not start:
        for channel, i in zip(x, range(len(x))):
            try:
                first_nonzero_sample.append(np.where(channel != 0)[0][0])
            except IndexError:
                first_nonzero_sample.append(0)
        # Use the lowest delay as reference.
        start = min(first_nonzero_sample)
    for channel in x:
        trimmed_waves.append(channel[start:])
    return np.ndarray(trimmed_waves)


def db2amp(db):
    """
    Convert db to amplitude

    Parameters
    ----------
    db : numbers.Number
        decibel value

    Returns
    -------
    numbers.Number
        amplitube value
    """
    return 10 ** (db / 20.0)


def amp2db(amp):
    """
    Convert amplitude to db

    Parameters
    ----------
    amp : number.Number
        amplitube value

    Returns
    -------
    numbers.Number
        decibel value
    """
    return 20 * np.log10(amp)
