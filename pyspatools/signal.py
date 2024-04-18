from copy import copy
from typing import Union, Optional
from os import PathLike

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
    def __init__(self, sig: Union[np.ndarray, PathLike], sr: int = 48000):
        """
        Base class for that holds the audio array and processing methods

        :param sig: If numpy.ndarray, this will be the signal array. If str,
            this is a filepath that reads a 24bit PCM wav file
        :param sr: Sampling rate. If sig is str, sr will be overwritten

        """
        self.sr = sr
        if isinstance(sig, PathLike):
            # Currently only support PCM24
            self.sig, self.sr = soundfile.read(sig, always_2d=True)
        else:
            self.sig = sig

        # Turn mono signal from shape (n, ) to (n, 1)
        try:
            _ = self.sig.shape[1]
        except IndexError:
            self.sig = np.expand_dims(self.sig, axis=1)

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
            when the employed window function starts at zero.sigal
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

    def spectrum(self, n: int | None = None) -> np.ndarray:
        """
        Calculate the absolute spectrum of each channel

        :param n: If None (default), the length of fft is half of the signal, otherwise the transformation results in
            half of n points.
        :returns: An array of absolute fft spectrum for each channel

        """
        return np.array(
            [np.abs(np.fft.rfft(self.sig[:, i], n=n)) for i in range(self.sig.shape[1])]
        )

    @staticmethod
    def to_mono(sig: np.ndarray) -> np.ndarray:
        """
        Mix channels to mono signal.
        """
        channels = sig.shape[1]
        blend = np.ones(channels) / channels
        if len(blend) != channels:
            raise AttributeError("len(blend) != self.channels")
        else:
            return np.sum(sig * blend, axis=1)

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
    
    def right_trim(self):
        """
        Right trim signal per channel to the last nonzero sample index.
            Use the maximal index across all channels.
        """
        last_nonzero_sample = []
        for i in range(self.channels):
            try:
                last_nonzero_sample.append(np.where(self.sig[:, i] != 0)[0][-1])
            except IndexError:
                last_nonzero_sample.append(self.length)
        end_idx = max(last_nonzero_sample)
        self.sig = self.sig[:end_idx, :]
        return self

    def pitch_detection(self) -> float:
        """
        Find its fundamental frequency based on peak value in spectrum
        """
        sig = copy(self.sig)
        if self.channels > 1:
            sig = self.to_mono(sig)

        rfftspec = np.fft.rfft(sig, axis=0)
        freqs = np.linspace(0, self.sr / 2, self.length // 2 + 1)

        return freqs[np.argmax(rfftspec)]


    def pitch_detection_per_channel(self) -> list:
        """
        Take an AudioData and find its fundamental frequency based on peak value in spectrum for each channel separately.
        """
        freqs = np.linspace(0, self.sr / 2, self.length // 2 + 1)
        fundamental_freqs = []

        for i in range(self.channels):
            channel = self.sig[:, i]

            rfftspec = np.fft.rfft(channel, axis=0)

            peak_index = np.argmax(np.abs(rfftspec))

            fundamental_freqs.append(freqs[peak_index])

        return fundamental_freqs

    def detect_dropouts(self, threshold=0.01, min_duration=0.01, ignore_after=10) -> list:
        """
        Detect dropouts in an audio signal, processing each channel individually.

        Parameters:
        - threshold: The amplitude threshold below which a signal is considered a dropout. Default is 0.01.
        - min_duration: The minimum duration (in seconds) for a segment to be considered a dropout. Default is 0.01 seconds.
        - ignore_after: Ignore dropouts after this time in seconds. Default is 10 seconds.

        Returns:
        A list of lists, where each sublist contains tuples. Each tuple contains the start and end times (in seconds)
        of detected dropouts for a channel.
        """
        signal = self.sig
        sr = self.sr

        # Initialize the list to hold dropout information for each channel
        dropouts_per_channel = []

        for i in range(self.channels):
            channel = signal[:, i]

            normalized_signal = np.abs(channel / np.max(np.abs(channel)))

            below_threshold = normalized_signal < threshold

            # Convert sample index to time
            time_index = np.arange(len(channel)) / sr

            # Identify contiguous regions below threshold
            dropouts = []
            dropout_start = None
            for i in range(len(below_threshold)):
                if below_threshold[i]:
                    if dropout_start is None:
                        dropout_start = i
                else:
                    if dropout_start is not None:
                        if time_index[i] - time_index[dropout_start] >= min_duration:
                            dropouts.append((time_index[dropout_start], time_index[i]))
                        dropout_start = None

            # Check if the last segment is a dropout
            if dropout_start is not None and time_index[-1] - time_index[dropout_start] >= min_duration:
                dropouts.append((time_index[dropout_start], time_index[-1]))

            # Ignore dropouts that occur after the specified time
            dropouts = [dropout for dropout in dropouts if dropout[0] <= ignore_after]

            dropouts_per_channel.append(dropouts)

        return dropouts_per_channel


def combine_signals(signal1: AudioSignal, signal2: AudioSignal) -> AudioSignal:
    """
    Combine two AudioSignal objects into one with multiple channels, assuming both signals
    have the same sampling rate and number of samples.

    Parameters:
    - signal1: The first AudioSignal object.
    - signal2: The second AudioSignal object.

    Returns:
    An AudioSignal object with the combined channels of both input signals.
    """
    # Extract the raw signal data and ensure preconditions are met
    if signal1.sr != signal2.sr:
        raise ValueError("Sampling rates do not match.")
    if signal1.length != signal2.length:
        raise ValueError("Signal lengths do not match.")

    # Determine the new number of channels
    total_channels = signal1.channels + signal2.channels

    # Create a new array for the combined signal
    combined_signal = np.zeros((signal1.length, total_channels))

    # Assign the data from the original signals to the combined signal
    combined_signal[:, :signal1.channels] = signal1.sig
    combined_signal[:, signal1.channels:] = signal2.sig

    new_signal = AudioSignal(sig=combined_signal, sr=signal1.sr)
    return new_signal

