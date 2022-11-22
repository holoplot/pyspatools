from typing import Optional
from typing import Union

import numpy as np
import pyloudnorm
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
    for i in range(x.shape[1]):
        result[:, i] = x[:, i] / ymax
    return result


class AudioSignal:
    def __init__(self, sig: Union[np.ndarray, str], sr: int = 48000):
        """
        Base class for that holds the audio array and processing methods

        :param sig: If numpy.ndarray, this will be the signal array. If str,
            this is a filepath that reads a 24bit PCM wav file
        :param sr: Sampling rate. If sig is str, sr will be overwritten

        """
        self.sr = sr
        if isinstance(sig, str):
            # Currently only support PCM24
            arry, sr = soundfile.read(file=sig, dtype="int32", always_2d=True)
            arry = np.right_shift(arry, 8)
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
        return self.sig.shape[1]

    @property
    def normalized_max(self) -> list:
        max = []
        for i in range(self.channels):
            max.append(np.max(np.abs(self.sig[:, i])))
        return max

    @property
    def length(self) -> int:
        return self.sig.shape[0]

    def left_trim(self):
        """
        Left trim signal per channel to the first nonzero sample index.
            Use the minimal index across all channels.
        """
        first_nonzero_sample = []
        for i in range(self.channels):
            try:
                first_nonzero_sample.append(np.where(self.sig[:, i] != 0)[0][0])
            except IndexError:
                first_nonzero_sample.append(0)
        start_idx = min(first_nonzero_sample)
        self.sig = self.sig[:, start_idx:]
        return self

    def stft(
        self,
        window="hann",
        nperseg=256,
        noverlap=None,
        nfft=None,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
    ) -> tuple:
        """
        Compute Short Time Fourier Transform using scipy.signal.stft for all channels

        :param window:
            Type of window function, default is 'hann', other options can be found at:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
        :param nperseg:
            Number of samples per stft segement (Default value = 256)
        :param noverlap:
            Number of samples to overlap between segments
        :param nfft:
            Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
        :param detrend:
            Specifies how to detrend each segment. If detrend is a string,
            it is passed as the type argument to the detrend function.
            If detrend is False, no detrending is done. Defaults to False.
        :param return_onesided:
            If True, return a one-sided spectrum for real data.
            If False return a two-sided spectrum. Defaults to True, but for complex data,
            a two-sided spectrum is always returned.
        :param boundary:
            Specifies whether the input signal is extended at both ends,
            and how to generate the new values, in order to center the first windowed segment on the first input point.
            This has the benefit of enabling reconstruction of the first input point
            when the employed window function starts at zero.
            Valid options are ['even', 'odd', 'constant', 'zeros', None].
            Defaults to ‘zeros’, for zero padding extension.
            I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] for nperseg=3.
        :param padded:
            Specifies whether the input signal is zero-padded at the end to make the signal fit exactly
            into an integer number of window segments, so that all of the signal is included in the output.
            Defaults to True. Padding occurs after boundary extension, if boundary is not None,
            and padded is True, as is the default.

        :returns freqs: A list of ndarray for each channel, each channel contains the array of sample frequencies
        :returns times: A list of ndarray for each channel, each contains the segment time for that channel.
        :returns Zxxs: A list of STFT of x for each channel
        """
        # TODO maybe more clean if results are np.ndarray instead of list
        freqs = []
        times = []
        Zxxs = []

        for i in range(self.channels):
            f, t, Zxx = signal.stft(
                self.sig[:, i],
                fs=self.sr,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                boundary=boundary,
                padded=padded,
                axis=0,
            )
            freqs.append(f)
            times.append(t)
            Zxxs.append(Zxx)

        return freqs, times, Zxxs

    def spectrum(self, n: Optional[int] = None) -> np.ndarray:
        """
        Calculate the absolute spectrum of each channel

        :param n: If None (default), the length of fft is half of the signal, otherwise the transformation results in
            half of n points.
        :returns: An array of absolute fft spectrum for each channel

        """
        return np.array(
            [np.abs(np.fft.rfft(self.sig[:, i], n=n)) for i in range(self.sig.shape[1])]
        )

    def latency(self, threshold: float = 1.0, offset=0) -> list:
        """
        Iterate through each channel and returns first index that is over threshold value.

        :param threshold: The signal threshold value that consider a valid signal
        :param offset: A sample offset as the before this offset maybe cause by other factor (if known) than the latency
        :returns results: A list of latency in sample per channel
        """
        if not isinstance(offset, int) or offset < 0:
            raise AttributeError("offset must be positive int")
        results = []
        for i in range(self.channels):
            where = np.where(np.abs(self.sig[offset:, i]) > threshold)
            try:
                results.append(where[0][0])
            except IndexError:
                raise ValueError(
                    f"Couldn't find starting signal with given threshold {threshold} "
                )
        return results

    def lkfs(self, bitrate: Optional[int] = None) -> list:
        """
        Loudness, K-weighted, relative to full scale implementation based on ITU-R BS.1770 standard.
        Credit: https://github.com/csteinmetz1/pyloudnorm.

        :param bitrate: The loudness calculation only works with float signal to 1.0. So for PCM signal
            the bitrate is required.
        :returns: A list of scala for a single loudness value in dB per channel
        """
        if bitrate:
            data = pcm_to_float(self.sig, bitrate)
        else:
            data = self.sig

        meter = pyloudnorm.Meter(self.sr)

        return [meter.integrated_loudness(data[:, i]) for i in range(data.shape[1])]

    def zero_padding(self, front=0, back=0) -> None:
        """
        Pad signal per channel base on the amount of samples

        :param front: The amount of 0s to be added to the front
        :param back: The amount of 0s to be added to the back
        """
        result = np.ndarray(
            (self.sig.shape[0] + front + back, self.sig.shape[1]), dtype=self.sig.dtype
        )
        for i in range(self.channels):
            result[:, i] = np.pad(self.sig[:, i], (front, back))
        self.sig = result

    def save(self, path: str) -> None:
        """
        Save signal to PCM24 format

        :param path: File path of the output file.
        """
        if not path.endswith(".wav"):
            raise AttributeError("Only accept .wav format in path")
        soundfile.write(path, self.sig, self.sr, "PCM_24")

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
        """
        iirfilter based on scipy.signal.iirfilter

        :param cutoff_freqs: Cutoff frequency or frequencies.
        :param btype: Filter type (Default value = 'highpass')
        :param ftype: Tthe type of IIR filter. e.g. 'butter', 'cheby1', 'cheby2', 'elip', 'bessel' (Default value = 'butter')
        :param order: Filter order (Default value = 4)
        :param filter: The scipy.signal method to call when applying the filter coeffs to the signal.
                    By default it is set to scipy.signal.lfilter (one-dimensional).
        :param rp: For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB) (Default value = None)
        :param rs: For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB) (Default value = None)

        :returns: A numpy array of the filter signaled.

        """
        # TODO scipy.signal.__getattribute__ error
        Wn = np.array(cutoff_freqs) * 2 / self.sr
        b, a = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        return getattr(signal, filter)(b, a, self.sig, axis=0)

    def find_peaks(
        self,
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    ) -> list:
        """
        Take the sig array and return a list of peaks for each channel. Please refer to
        scipy.signal.find_peaks: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        """
        results = []
        for i in range(self.channels):
            results.append(
                signal.find_peaks(
                    self.sig[:, i],
                    height=height,
                    threshold=threshold,
                    distance=distance,
                    prominence=prominence,
                    width=width,
                    wlen=wlen,
                    rel_height=rel_height,
                    plateau_size=plateau_size,
                )
            )
        return results
