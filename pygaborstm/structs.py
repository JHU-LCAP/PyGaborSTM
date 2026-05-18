"""Data classes for spectrogram and RSF representations."""

from dataclasses import dataclass
import numpy as np


def _array_module(array):
    """Return numpy or cupy depending on the array's backend."""
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


@dataclass
class Spectrogram:
    """Auditory spectrogram with frequency and time axes.

    Parameters
    ----------
    data : np.ndarray
        Spectrogram array of shape ``(n_freq, n_time)``. May be a numpy
        or cupy array depending on the backend that produced it.
    times : np.ndarray
        Time axis in seconds, length ``n_time``.
    freqs : np.ndarray
        Center frequencies in Hz, length ``n_freq``.
    sr : int
        Sample rate (Hz) of the original audio.
    """

    data: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    sr: int

    @property
    def shape(self) -> tuple:
        """Shape of ``data`` as ``(n_freq, n_time)``."""
        return self.data.shape

    @property
    def n_freqs(self) -> int:
        """Number of frequency channels."""
        return self.data.shape[0]

    @property
    def n_times(self) -> int:
        """Number of time frames."""
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """Total duration in seconds (last time minus first; ``0.0`` if a single frame)."""
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def to_numpy(self) -> np.ndarray:
        """Return ``data`` as a numpy array, copying from the GPU if needed.

        Returns
        -------
        np.ndarray
            Host-side copy of the spectrogram data.
        """
        from .backend import to_numpy

        return to_numpy(self.data)


@dataclass
class RSF:
    """Rate-Scale-Frequency representation produced by the Gabor stage.

    Parameters
    ----------
    data : np.ndarray
        RSF array of shape ``(n_frames, n_rates, n_scales, n_freq)``.
        May be numpy or cupy.
    times : np.ndarray
        Frame center times in seconds, length ``n_frames``.
    rates : np.ndarray
        Temporal modulation rates in Hz, length ``n_rates``. The first
        half is negative (upward sweeps), the second half is positive
        (downward sweeps).
    scales : np.ndarray
        Spectral modulation scales in cycles/octave, length ``n_scales``.
    freqs : np.ndarray
        Center frequencies in Hz, length ``n_freq``.
    """

    data: np.ndarray
    times: np.ndarray
    rates: np.ndarray
    scales: np.ndarray
    freqs: np.ndarray

    @property
    def shape(self) -> tuple:
        """Shape of ``data`` as ``(n_frames, n_rates, n_scales, n_freq)``."""
        return self.data.shape

    @property
    def n_frames(self) -> int:
        """Number of RSF frames along the time axis."""
        return self.data.shape[0]

    @property
    def n_rates(self) -> int:
        """Number of temporal modulation rates."""
        return self.data.shape[1]

    @property
    def n_scales(self) -> int:
        """Number of spectral modulation scales."""
        return self.data.shape[2]

    @property
    def n_freqs(self) -> int:
        """Number of frequency channels."""
        return self.data.shape[3]

    def to_numpy(self) -> np.ndarray:
        """Return ``data`` as a numpy array, copying from the GPU if needed.

        Returns
        -------
        np.ndarray
            Host-side copy of the RSF data.
        """
        from .backend import to_numpy

        return to_numpy(self.data)

    def mean_over_time(self):
        """Average across the time/frame axis.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_rates, n_scales, n_freq)``.
        """
        return self.data.mean(axis=0)

    def mean_over_freq(self):
        """Average across the frequency axis.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_frames, n_rates, n_scales)``.
        """
        return self.data.mean(axis=3)

    def _split_by_direction(self):
        mid = self.n_rates // 2
        return self.data[:, :mid, :, :], self.data[:, mid:, :, :]

    def upward_rates(self) -> np.ndarray:
        """Negative-rate half of the rates axis (upward-sweeping ripples).

        Returns
        -------
        np.ndarray
            First half of ``rates``, length ``n_rates // 2``.
        """
        return self.rates[: self.n_rates // 2]

    def downward_rates(self) -> np.ndarray:
        """Positive-rate half of the rates axis (downward-sweeping ripples).

        Returns
        -------
        np.ndarray
            Second half of ``rates``, length ``n_rates // 2``.
        """
        return self.rates[self.n_rates // 2 :]

    def rate_scale_matrix(self, fold: bool = False):
        """Reduce the RSF to a 2D scale-by-rate matrix.

        Parameters
        ----------
        fold : bool, default False
            If True, average the upward and downward halves into a
            single symmetric matrix, then mirror it back to full width
            so the output shape is preserved.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_scales, n_rates)``.
        """
        rs = self.data.mean(axis=(0, 3)).T

        if not fold:
            return rs

        return self._fold_rates_scales()

    def rate_scale_matrix_split(self):
        """Reduce the RSF to two scale-by-rate matrices, by sweep direction.

        Returns
        -------
        tuple of np.ndarray
            ``(upward, downward)``, each of shape
            ``(n_scales, n_rates // 2)``.
        """
        up_data, down_data = self._split_by_direction()
        return up_data.mean(axis=(0, 3)).T, down_data.mean(axis=(0, 3)).T

    def _fold_rates_scales(self):
        xp = _array_module(self.data)
        upward_rs, downward_rs = self.rate_scale_matrix_split()
        rs_folded = (xp.flip(upward_rs, axis=1) + downward_rs) / 2
        return xp.concatenate([xp.flip(rs_folded, axis=1), rs_folded], axis=1)
