from dataclasses import dataclass
import numpy as np


def _array_module(array):
    """Get numpy or cupy depending on the array type."""
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


@dataclass
class Spectrogram:
    """
    Auditory spectrogram representation.

    Attributes:
        data: Spectrogram array [n_freq × n_time] (numpy or cupy)
        times: Time axis in seconds [n_time]
        freqs: Center frequencies in Hz [n_freq]
        sr: Sample rate of original audio
    """

    data: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    sr: int

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def n_freqs(self) -> int:
        return self.data.shape[0]

    @property
    def n_times(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def to_numpy(self) -> np.ndarray:
        from .backend import to_numpy

        return to_numpy(self.data)


@dataclass
class RSF:
    """
    Rate-Scale-Frequency representation.

    Attributes:
        data: RSF array [n_frames × n_rates × n_scales × n_freq] (numpy or cupy)
        times: Frame times in seconds [n_frames]
        rates: Temporal modulation rates in Hz [n_rates]
        scales: Spectral modulation scales in cycles/octave [n_scales]
        freqs: Center frequencies in Hz [n_freq]
    """

    data: np.ndarray
    times: np.ndarray
    rates: np.ndarray
    scales: np.ndarray
    freqs: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def n_rates(self) -> int:
        return self.data.shape[1]

    @property
    def n_scales(self) -> int:
        return self.data.shape[2]

    @property
    def n_freqs(self) -> int:
        return self.data.shape[3]

    def to_numpy(self) -> np.ndarray:
        from .backend import to_numpy

        return to_numpy(self.data)

    def mean_over_time(self):
        return self.data.mean(axis=0)

    def mean_over_freq(self):
        return self.data.mean(axis=3)

    def _split_by_direction(self):
        mid = self.n_rates // 2
        return self.data[:, :mid, :, :], self.data[:, mid:, :, :]

    def upward_rates(self) -> np.ndarray:
        return self.rates[: self.n_rates // 2]

    def downward_rates(self) -> np.ndarray:
        return self.rates[self.n_rates // 2 :]

    def rate_scale_matrix(self, fold: bool = False):
        rs = self.data.mean(axis=(0, 3)).T

        if not fold:
            return rs

        return self._fold_rates_scales()

    def rate_scale_matrix_split(self):
        up_data, down_data = self._split_by_direction()
        return up_data.mean(axis=(0, 3)).T, down_data.mean(axis=(0, 3)).T

    def _fold_rates_scales(self):
        xp = _array_module(self.data)
        upward_rs, downward_rs = self.rate_scale_matrix_split()
        rs_folded = (xp.flip(upward_rs, axis=1) + downward_rs) / 2
        return xp.concatenate([xp.flip(rs_folded, axis=1), rs_folded], axis=1)
