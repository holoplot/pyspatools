import numpy as np

from .signal import AudioSignal


__all__ = ["Diff", "match_size"]


def match_size(a: AudioSignal, b: AudioSignal) -> tuple:
    """
    Compare two signals and right trim to match the length to the smaller one
    """
    if a.length > b.length:
        a.sig = a.sig[:, : b.length]
    else:
        b.sig = b.sig[:, : a.length]
    return a, b


class Diff:
    def __init__(self, a: AudioSignal, b: AudioSignal, tolerance: int = 0):
        """A Diff provide comparison between the AudioSignal of two inputs."""
        self.a = a
        self.b = b
        if self.a.sr != self.b.sr:
            raise AttributeError("a and b have different sampling rate.")
        self.sr = self.a.sr
        if self.a.channels != self.b.channels:
            raise AttributeError("a and b have different channels")
        if self.a.length != self.b.length:
            self.a, self.b = match_size(self.a, self.b)
        self.channels = self.a.channels
        self.tolerance = tolerance

    @property
    def delta(self) -> np.ndarray:
        """Element differences between the sig of a and b"""
        return self.a.sig - self.b.sig

    @property
    def where(self) -> list:
        """A list of indices where the absolute delta value is greater than tolerance per channel."""
        return [
            list(np.where(np.abs(ch_delta) > self.tolerance)[0])
            for ch_delta in self.delta
        ]

    @property
    def within_tolerance(self):
        return False if np.any(self.where) else True

    @property
    def channel_max(self) -> np.ndarray:
        """An array of max value per channel"""
        return np.amax(self.delta, axis=1)

    @property
    def channel_min(self) -> np.ndarray:
        """An array of min value per channel"""
        return np.amin(self.delta, axis=1)
