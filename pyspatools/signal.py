from typing import Optional
from typing import Union

import numpy as np
import pyloudnorm
import scipy
import soundfile
from scipy import signal

from .helpers.const import PCM16_SIGNED_MAX
from .helpers.const import PCM24_SIGNED_MAX
from .helpers.const import PCM32_SIGNED_MAX
from .helpers.const import PCM8_SIGNED_MAX


def pcm_to_float(x: np.ndarray, bitrate: int) -> np.ndarray:
    if bitrate == 24:
        ymax = PCM24_SIGNED_MAX
    elif bitrate == 16:
        ymax = PCM16_SIGNED_MAX
    elif bitrate == 32:
        ymax = PCM32_SIGNED_MAX
    elif bitrate == 8:
        ymax = PCM8_SIGNED_MAX
    else:
        return x
    result = np.ndarray(shape=x.shape, dtype=np.float32)
    for i in range(len(x)):
        result[i, :] = x[i, :] / ymax
    return result


class AudioSignal():
    def __init__(self, sig: Union[np.ndarray, str], sr: int = 48000):
        """
        Base class for that holds the audio array and processing methods

        :param sig: If numpy.ndarray, this will be the signal array. If str, this is a filepath that reads a 24bit PCM wav file
        :param sr: Sampling rate. If sig is str, sr will be overwritten

        """
        self.sr = sr
        if isinstance(sig, str):
            # Currently only support PCM24
            arry, sr = soundfile.read(file=sig, dtype='int32', always_2d=True)
            arry = np.transpose(np.right_shift(arry, 8))
            self.sig = pcm_to_float(arry, bitrate=24)
            self.sr = sr
        else:
            self.sig = sig

    @property
    def shape(self) -> tuple:
        return self.sig.shape

    @property
    def dtype(self) -> np.dtype:
        return self.sig.dtype

    @property
    def channels(self) -> int:
        return self.sig.shape[0]

    @property
    def normalized_max(self):
        max = []
        for each_channel in self.sig:
            max.append(np.max(np.abs(each_channel)))
        return max

    @property
    def length(self):
        return self.sig.shape[1]

    def left_trim(self):
        """
        Left trim signal per channel to the first nonzero sample index. Use the minimal index across all channels.
        """
        first_nonzero_sample = []
        for channel in self.sig:
            try:
                first_nonzero_sample.append(np.where(channel != 0)[0][0])
            except IndexError:
                first_nonzero_sample.append(0)
        start_idx = min(first_nonzero_sample)
        self.sig = self.sig[:, start_idx:]
        return self

    def stft(self, window='hann', nperseg=256, noverlap=None,
             nfft=None, detrend=False, return_onesided=True,
             boundary='zeros', padded=True) -> tuple:
        """
        Compute Short Time Fourier Transform using scipy.signal.stft for all channels

        Parameters
        ----------
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

        for each_ch in self.sig:
            f, t, Zxx = scipy.stft(each_ch, fs=self.sr, window=window, nperseg=nperseg,
                                noverlap=noverlap, nfft=nfft, detrend=detrend,
                                return_onesided=return_onesided, boundary=boundary,
                                padded=padded, axis=0)
            freqs.append(f)
            times.append(t)
            Zxxs.append(Zxx)

        return freqs, times, Zxxs

    def spectrum(self, n: Optional[int] = None) -> np.ndarray:
        """
        Calculate the absolute spectrum of each channel

        Parameters
        ----------
        n : int or None
            If None (default), the length of fft is half of the signal, otherwise the transformation results in
            half of n points.

        Returns
        -------
        np.ndarray
            An array of absolute fft spectrum for each channel

        """
        return np.array([np.abs(np.fft.rfft(ch, n=n)) for ch in self.sig])


    def latency(self, threshold: float = 1.0, offset=0) -> list:
        """
        Iterate through each channel and returns first index that is over threshold value.

        Parameters
        ----------
        threshold : float
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
        for ch in self.sig:
            where = np.where(np.abs(ch[offset:]) > threshold)
            try:
                results.append(where[0][0])
            except IndexError:
                raise ValueError(f"Couldn't find starting signal with given threshold {threshold} ")
        return results


    def lkfs(self, bitrate: Optional[int] = None):
        """
        Loudness, K-weighted, relative to full scale implementation based on ITU-R BS.1770 standard.
        Credit: https://github.com/csteinmetz1/pyloudnorm.

        Parameters
        ----------
        bitrate : int, optional
            The loudness calculation only works with float signal to 1.0. So for PCM signal
            the bitrate is required.
        Returns
        -------
        list
            A list of scala for a single loudness value in dB per channel

        """
        if bitrate:
            data = pcm_to_float(self.sig, bitrate)
        else:
            data = self.sig

        meter = pyloudnorm.Meter(self.sr)

        return [meter.integrated_loudness(ch) for ch in data]

    def zero_padding(self, front=0, back=0):
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

        """
        result = np.ndarray((self.sig.shape[0], self.sig.shape[1] + front + back), dtype=self.sig.dtype)
        for i in range(self.sig.shape[0]):
            result[i] = np.pad(self.sig[i], (front, back))
        self.sig = result

    def save(self, path: str):
        """
        Save signal to PCM24 format
        """
        if not path.endswith('.wav'):
            raise AttributeError("Only accept .wav format in path")
        soundfile.write(path, self.sig, self.sr, 'PCM_24')


    def iirfilter(
        self,
        cutoff_freqs,
        btype="highpass",
        ftype="butter",
        order=4,
        filter="lfilter",
        rp=None,
        rs=None,
    ):
        """iirfilter based on scipy.signal.iirfilter

        Parameters
        ----------
        cutoff_freqs : float or [float, float]
            Cutoff frequency or frequencies.
        btype : str
            Filter type (Default value = 'bandpass')
        ftype : str
            Tthe type of IIR filter. e.g. 'butter', 'cheby1', 'cheby2', 'elip', 'bessel' (Default value = 'butter')
        order : int
            Filter order (Default value = 4)
        filter : str
            The scipy.signal method to call when applying the filter coeffs to the signal.
            by default it is set to scipy.signal.lfilter (one-dimensional).
        rp : float
            For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB) (Default value = None)
        rs : float
            For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB) (Default value = None)

        Returns
        -------
        _ : Asig
            new Asig with the filter applied. also you can access b, a coefficients by doing self._['b']
            and self._['a']

        """
        # TODO scipy.signal.__getattribute__ error
        Wn = np.array(cutoff_freqs) * 2 / self.sr
        b, a = signal.iirfilter(
            order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        return signal.__getattribute__(filter)(b, a, self.sig, axis=1)


    def find_peaks(self, height=None, threshold=None, distance=None,
                   prominence=None, width=None, wlen=None,
                   rel_height=0.5, plateau_size=None):
        results = []
        for i in range(self.channels):
            results.append(signal.find_peaks(self.sig[i, :], height=None,
                                             threshold=None, distance=None,
                                             prominence=None, width=None,
                                             wlen=None, rel_height=0.5,
                                             plateau_size=None))
        return results
