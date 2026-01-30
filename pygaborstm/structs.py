from dataclasses import dataclass
import numpy as np


@dataclass
class Spectrogram:
    """
    Auditory spectrogram representation.

    Attributes:
        data: Spectrogram array [n_freq × n_time]
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
        """Duration in seconds."""
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def to_numpy(self) -> np.ndarray:
        """Return raw data array."""
        return self.data


@dataclass
class RSF:
    """
    Rate-Scale-Frequency representation.

    Attributes:
        data: RSF array [n_frames × n_rates × n_scales × n_freq]
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
        """Return raw data array."""
        return self.data

    def mean_over_time(self) -> np.ndarray:
        """
        Collapse time dimension.

        Returns:
            Array [n_rates × n_scales × n_freq]
        """
        return self.data.mean(axis=0)

    def mean_over_freq(self) -> np.ndarray:
        """
        Collapse frequency dimension.

        Returns:
            Array [n_frames × n_rates × n_scales]
        """
        return self.data.mean(axis=3)

    def rate_scale_matrix(self, fold: bool = False) -> np.ndarray:
        """
        Get 2D rate-scale representation (averaged over time and frequency).

        Args:
            fold: If True, fold positive/negative rates for symmetric visualization

        Returns:
            Array [n_scales × n_rates]
        """
        rs = self.data.mean(axis=(0, 3)).T  # [n_scales × n_rates]

        if not fold:
            return rs

        return self._fold_rates_scales(rs)

    def _fold_rates_scales(self, rs: np.ndarray) -> np.ndarray:
        """
        Fold RSF by averaging positive and negative rates.

        Creates symmetric visualization by averaging responses at
        matching positive/negative rates.

        Args:
            rs: Rate-scale matrix [n_scales × n_rates]

        Returns:
            Folded matrix [n_scales × n_rates] (symmetric)
        """
        n_rates_half = rs.shape[1] // 2

        rs_left = rs[:, :n_rates_half]       # Negative rates
        rs_right = rs[:, n_rates_half:]      # Positive rates

        # Flip left so magnitudes align
        rs_left_flipped = np.flip(rs_left, axis=1)

        # Average positive and negative
        rs_folded = (rs_left_flipped + rs_right) / 2

        # Mirror back for symmetric visualization
        rs_folded_mirrored = np.concatenate([
            np.flip(rs_folded, axis=1),
            rs_folded
        ], axis=1)

        return rs_folded_mirrored