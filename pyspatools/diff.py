from typing import Union

import numpy as np

from .signal import AudioSignal, AudioFile


__all__ = ['Diff', 'right_trim']


def right_trim(a: Union[AudioSignal, AudioFile], b: Union[AudioSignal, AudioFile]) -> tuple:
    """
    Compare two signals and right trim to match the length to the smaller one
    """
    if a.length > b.length:
        a.data = a.data[:, :b.length]
    else:
        b.data = b.data[:, :a.length]
    return a, b


class Diff():
    def __init__(self, a: Union[AudioSignal, AudioFile], b: Union[AudioSignal, AudioFile], tolerance: int = 0):
        """A Diff provide comparison between the AudioSignal of two inputs."""
        self.a = a
        self.b = b
        if self.a.sr != self.b.sr:
            raise AttributeError("a and b have different sampling rate.")
        self.sr = self.a.sr
        if self.a.channels != self.b.channels:
            raise AttributeError("a and b have different channels")
        if self.a.length != self.b.length:
            self.a, self.b = right_trim(self.a, self.b)
        self.channels = self.a.channels
        self.tolerance = tolerance


    @property
    def delta(self) -> np.ndarray:
        """Element differences between the data of a and b"""
        return self.a.data - self.b.data

    @property
    def where(self) -> list:
        """A list of indices where the absolute delta value is greater than tolerance per channel."""
        if self.delta:
            return [np.where(np.abs(ch_delta) > self.tolerance)[0] for ch_delta in self.delta]
        else:
            return []

    @property
    def channel_max(self) -> np.ndarray:
        """An array of max value per channel"""
        return np.amax(self.delta, axis=1)

    @property
    def channel_min(self) -> np.ndarray:
        """An array of min value per channel"""
        return np.amin(self.delta, axis=1)
