from typing import Optional
import numpy as np
from scipy.signal import stft as scistft


__all__ = ['stft', 'spectrum']


def stft(x, sr, window='hann', nperseg=256, noverlap=None,
         nfft=None, detrend=False, return_onesided=True,
         boundary='zeros', padded=True):
    """
    Compute Short Time Fourier Transform using scipy.signal.stft for all channels

    Parameters
    ----------
    x : numpy ndarray
        Input multichannel audio signals.
    sr : int
        Sampling rate
    window : str, optional
        Type of window function, default is 'hann', other options can be found at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    nperseg : int, optional
        Number of samples per stft segement (Default value = 256)
    noverlap : int, optional
        Number of samples to overlap between segments
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If detrend is a string,
        it is passed as the type argument to the detrend function.
        If it is a function, it takes a segment and returns a detrended segment.
        If detrend is False, no detrending is done. Defaults to False.
    return_onesided: bool, optional
        If True, return a one-sided spectrum for real data.
        If False return a two-sided spectrum. Defaults to True, but for complex data,
        a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends,
        and how to generate the new values, in order to center the first windowed segment on the first input point.
        This has the benefit of enabling reconstruction of the first input point
        when the employed window function starts at zero.
        Valid options are ['even', 'odd', 'constant', 'zeros', None].
        Defaults to ‘zeros’, for zero padding extension.
        I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] for nperseg=3.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to make the signal fit exactly
        into an integer number of window segments, so that all of the signal is included in the output.
        Defaults to True. Padding occurs after boundary extension, if boundary is not None,
        and padded is True, as is the default.

    Returns
    -------
    freqs : list
        A list of ndarray for each channel, each channel contains the array of sample frequencies
    times : list
        A list of ndarray for each channel, each contains the segment time for that channel.
    Zxxs : list
        A list of STFT of x for each channel

    """
    # TODO maybe more clean if results are np.ndarray instead of list
    freqs = []
    times = []
    Zxxs = []

    for each_ch in x:
        f, t, Zxx = scistft(each_ch, fs=sr, window=window, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, detrend=detrend,
                            return_onesided=return_onesided, boundary=boundary,
                            padded=padded, axis=0)
        freqs.append(f)
        times.append(t)
        Zxxs.append((Zxx))

    return freqs, times, Zxxs


def spectrum(x: np.ndarray, n: Optional[int] = None):
    """
    Calculate the absolute spectrum of each channel

    Parameters
    ----------
    x : numpy ndarray
        Input multichannel audio signals.
    n : int or None
        If None (default), the length of fft is half of the signal, otherwise the transformation results in
        half of n points.

    Returns
    -------
    np.ndarray
        An array of absolute fft spectrum for each channel

    """
    return np.array([np.abs(np.fft.rfft(ch, n=n)) for ch in x])

