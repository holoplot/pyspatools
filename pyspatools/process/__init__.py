import numpy as np

__all__ = ['left_trim']


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


