from dataclasses import dataclass
import numpy as np


__all__ = ['Diff']


@dataclass
class Diff:
    a: np.ndarray
    b: np.ndarray
    sr: int
    tolerance: int = 0

    @property
    def delta(self) -> np.ndarray:
        """Element differences between a and b"""
        return self.a - self.b

    @property
    def where(self) -> list:
        """A list of indices where the absolute delta value is greater than tolerance per channel."""
        return [np.where(np.abs(ch_delta) > self.tolerance)[0] for ch_delta in self.delta]

    @property
    def channel_max(self) -> np.ndarray:
        """An array of max value per channel"""
        return [np.where(np.abs(item) > self.tolerance)[0] for item in np.amax(self.delta, axis=1)]

    @property
    def channel_min(self) -> np.ndarray:
        """An array of min value per channel"""
        # return np.amin(self.delta, axis=1)
        return [np.where(np.abs(item) > self.tolerance)[0] for item in np.amax(self.delta, axis=1)]

    @property
    def channel_max_indices(self) -> list:
        """A list with an array of each channel where the indices of channel_max value occurred"""
        return [np.where(self.delta[i, :] == self.channel_max[i])[0] for i in range(len(self.channel_max))]

    @property
    def channel_min_indices(self) -> list:
        """A list with an array of each channel where the indices of channel_min value occurred"""
        return [np.where(self.delta[i, :] == self.channel_min[i])[0] for i in range(len(self.channel_min))]
