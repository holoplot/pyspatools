import numpy as np

__all__ = ['get_latency']


def get_latency(x, threshold=0, offset=0) -> list:
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
