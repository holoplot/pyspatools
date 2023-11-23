from typing import Optional
from pathlib import Path

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


class AudioBase:
    def __init__(self, input: np.ndarray | str | Path, sr: int = 48000):
        """
        Base class for that holds the audio array and processing methods

        :param input: Input file path or numpy.ndarray
        :param sr: Sample rate
        """
        if isinstance(input, str) or isinstance(input, Path):
            info = soundfile.info(input)
            self.array, self.sr = soundfile.read(input, always_2d=True)
            # TODO check if array is int to begin with
            match info.subtype:
                case "PCM_16":
                    bitrate = 16
                case "PCM_24":
                    bitrate = 24
                case "PCM_32":
                    bitrate = 32
                case "PCM_8":
                    bitrate = 8
                case _:
                    raise AttributeError(f"Only support PCM_16, PCM_24, PCM_32, PCM_8")
            self.array = pcm_to_float(self.array, bitrate)
            
        elif isinstance(input, np.ndarray):
            self.array = input
            self.sr = sr
        else:
            raise AttributeError("Only accept str/path load file or numpy.ndarray as direct input")

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

    @property
    def duration(self) -> float:
        return self.length / self.sr

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
        self.sig = self.sig[start_idx:, :]
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

    def to_mono(self):
        """
        Mix channels to mono signal.
        """
        blend = np.ones(self.channels) / self.channels
        if len(blend) != self.channels:
            raise AttributeError("len(blend) != self.channels")
        else:
            self.sig = np.sum(self.sig * blend, axis=1)

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

    def lkfs_per_channel(self) -> list:
        """
        k-weighted loudness per channel
        """
        return [lkfs(self.array[:, i]) for i in range(self.array.shape[1])]

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
        return AudioSignal(getattr(signal, filter)(b, a, self.sig, axis=0),
                           sr=self.sr)

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

