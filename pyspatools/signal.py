from typing import Optional

import numpy as np
import pyloudnorm
from scipy.signal import stft as scistft
import soundfile

from .helpers.const import PCM8_SIGNED_MAX, PCM16_SIGNED_MAX, PCM24_SIGNED_MAX, PCM32_SIGNED_MAX


class AudioSignal():
    def __init__(self, data: np.ndarray, sr: int):
        """
        Base class for that holds the audio array and processing methods
        Parameters
        ----------
        data : numpy.ndarray
            Multi-channel array in the form of (channels, signals)
        sr : int
            Sampling rate
        """
        self.data = data
        self.sr = sr

    @property
    def channels(self):
        return self.data.shape[0]

    @property
    def length(self):
        return self.data.shape[1]

    def left_trim(self):
        """
        Left trim signal per channel to the first nonzero sample index. Use the minimal index across all channels. 
        """
        first_nonzero_sample = []
        for channel in self.data:
            try:
                first_nonzero_sample.append(np.where(channel != 0)[0][0])
            except IndexError:
                first_nonzero_sample.append(0)
        start_idx = min(first_nonzero_sample)
        self.data = self.data[:, start_idx:]
        return self

    def pcm_to_float(self, bitrate: int):
        if bitrate == 24:
            ymax = PCM24_SIGNED_MAX
        elif bitrate == 16:
            ymax = PCM16_SIGNED_MAX
        elif bitrate == 32:
            ymax = PCM32_SIGNED_MAX
        elif bitrate == 8:
            ymax = PCM8_SIGNED_MAX
        else:
            return self.data
        result = np.ndarray(shape=self.data.shape, dtype=np.float32)
        for i in range(len(self.data)):
            result[i, :] = self.data[i, :] / ymax
        return result

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

        for each_ch in self.data:
            f, t, Zxx = scistft(each_ch, fs=self.sr, window=window, nperseg=nperseg,
                                noverlap=noverlap, nfft=nfft, detrend=detrend,
                                return_onesided=return_onesided, boundary=boundary,
                                padded=padded, axis=0)
            freqs.append(f)
            times.append(t)
            Zxxs.append((Zxx))

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
        return np.array([np.abs(np.fft.rfft(ch, n=n)) for ch in self.data])


    def latency(self, threshold=0, offset=0) -> list:
        """
        Iterate through each channel and returns first index that is over threshold value.

        Parameters
        ----------
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
        for ch in self.data:
            where = np.where(np.abs(ch[offset:]) > threshold)
            try:
                results.append(where[0][0])
            except IndexError:
                raise ValueError(f"Couldn't find starting signal with given threshold {threshold} ")
        return results


    def lkfs(self, bitrate=None):
        """
        Loudness, K-weighted, relative to full scale implementation based on ITU-R BS.1770 standard.
        Credit: https://github.com/csteinmetz1/pyloudnorm

        Returns
        -------
        list
            A list of scala for a single loudness value in dB per channel

        """
        if bitrate:
            data = self.pcm_to_float(bitrate)
        else:
            data = self.data

        meter = pyloudnorm.Meter(self.sr)
        
        return [meter.integrated_loudness(ch) for ch in data]


class AudioFile(AudioSignal):
    def __init__(self, path: str, bitrate: int = 24, transpose: bool = True):
        """
        This is the main object for pyspatool that load 

        Parameters
        ----------
        path : str
            Path to audio source
        bitrate : int
            Bit rate, currently only support 24bit
        transpose : bool
            Whether to apply transpose to the read file signal.
        """
        self.bitrate = bitrate
        if self.bitrate==24:
            _dtype = 'int32'
        else:
            raise AttributeError('Unsupported bitrate, currently: 24')
        data, sr = soundfile.read(file=path, dtype=_dtype, always_2d=True)
        if transpose:
            data = np.transpose(np.right_shift(data, 8))
        super().__init__(data, sr)

    def save(self, path: str):
        soundfile.write(path, self.data, self.sr)


