import numpy as np

__all__ = ["lin_map", "db2amp", "amp2db"]


def lin_map(x, in_min, in_max, out_min, out_max):
    """
    Linear mapping a scala
    Parameters
    ----------
    x : numpy.ndarray
        input signal array
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
    numpy.ndarray
        mapped output
    """
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def db2amp(db: float) -> float:
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


def amp2db(amp: float) -> float:
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
