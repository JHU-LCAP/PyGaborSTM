"""
Gabor Filterbank and RSF Extraction

Implements 2D Gabor filters for spectro-temporal modulation analysis,
following Bellur & Elhilali (2017).

Pipeline:
    1. Create 2D Gabor filters tuned to (rate, scale) pairs
    2. Convolve with auditory spectrogram
    3. Integrate over time windows to produce RSF representation
"""

import numpy as np
from typing import Tuple

from .config import Config
from .structs import Spectrogram, RSF
from .backend import get_array_module, get_signal_module, to_numpy


# Default Gabor parameter options (for adaptive tuning)
PARAM_OPTIONS = {
    "sigma_t": np.array(
        [1 / 1.4, 1 / 1.6, 1 / 1.8, 1 / 2.0, 1 / 2.2, 1 / 2.4, 1 / 2.6]
    ),
    "sigma_f": np.array(
        [1 / 1.4, 1 / 1.6, 1 / 1.8, 1 / 2.0, 1 / 2.2, 1 / 2.4, 1 / 2.6]
    ),
    "theta": np.radians(np.array([-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5])),
    "alpha": np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
}
DEFAULT_PARAM_IDX = 3


class GaborFilterbank:
    """
    2D Gabor filterbank for spectro-temporal modulation analysis.

    Filters are tuned to different rates (temporal modulation, Hz) and
    scales (spectral modulation, cycles/octave).
    """

    # Resolution presets: (n_rates_positive, n_scales)
    RESOLUTION_PRESETS = {
        "low": (5, 6),  # 60 filters (paper default)
        "medium": (10, 12),  # 240 filters
        "high": (20, 20),  # 800 filters
        "ultra": (32, 32),  # 2048 filters
    }

    def __init__(self, config: Config | None = None):
        cfg = config or Config()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.rsf_frame_size_ms = cfg.rsf_frame_size_ms
        self.rsf_frame_shift_ms = cfg.rsf_frame_shift_ms
        self.use_gpu = cfg.use_gpu

        # Get array and signal modules
        self.xp = get_array_module(self.use_gpu)
        self.signal = get_signal_module(self.use_gpu)

        self.frmlen_ms = cfg.frmlen_ms
        self.bandwidth_oct = cfg.octaves
        self.time_per_frame = self.frmlen_ms / 1000.0

        self.rates, self.scales = self._get_rates_scales(cfg.resolution)

    def _get_rates_scales(self, resolution: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rate and scale arrays based on resolution preset."""
        if resolution not in self.RESOLUTION_PRESETS:
            raise ValueError(
                f"Invalid resolution '{resolution}'. "
                f"Choose from {list(self.RESOLUTION_PRESETS.keys())}"
            )

        n_rates_pos, n_scales = self.RESOLUTION_PRESETS[resolution]

        # Positive rates: 2-32 Hz (log spaced)
        rates_pos = np.logspace(np.log2(2), np.log2(32), n_rates_pos, base=2)
        # Full rates: negative (upward) + positive (downward)
        rates = np.concatenate([-rates_pos[::-1], rates_pos])

        # Scales: 0.25-8 cycles/octave (log spaced)
        scales = np.logspace(np.log2(0.25), np.log2(8), n_scales, base=2)

        return rates, scales

    def _create_gabor_filter(
        self,
        omega: float,
        Omega: float,
        T,
        F,
        sigma_t_mult: float = 0.5,
        sigma_f_mult: float = 0.5,
        theta: float = 0.0,
        alpha: float = 1.0,
    ):
        """
        Create a 2D Gabor filter.


        F(ω, Ω, t, f) = α/(2πσ_t σ_f) * exp(-0.5(t₁²/σ_t² + f₁²/σ_f²)) * exp(2πj(ωt + Ωf))

        Args:
            omega: Temporal modulation rate (Hz)
            Omega: Spectral modulation scale (cycles/octave)
            T: Time meshgrid
            F: Frequency meshgrid
            sigma_t_mult: Time bandwidth multiplier
            sigma_f_mult: Frequency bandwidth multiplier
            theta: Filter orientation (radians)
            alpha: Gain factor

        Returns:
            Complex 2D Gabor filter
        """
        xp = self.xp
        
        omega_abs = max(abs(omega), 0.5)

        sigma_t = sigma_t_mult / omega_abs
        sigma_f = sigma_f_mult / Omega

        # Rotated coordinates
        t1 = T * xp.cos(theta) + F * xp.sin(theta)
        f1 = -T * xp.sin(theta) + F * xp.cos(theta)

        # Gaussian envelope
        gaussian = (alpha / (2 * xp.pi * sigma_t * sigma_f)) * xp.exp(
            -0.5 * (t1**2 / sigma_t**2 + f1**2 / sigma_f**2)
        )

        # Complex sinusoidal carrier
        carrier = xp.exp(2j * xp.pi * (omega * T + Omega * F))

        return gaussian * carrier

    def _apply_gabor_filter(
        self,
        spec,
        omega: float,
        Omega: float,
        filter_params: Tuple[float, float, float, float] = (0.5, 0.5, 0.0, 1.0),
    ):
        """
        Apply a single Gabor filter to the spectrogram.

        Args:
            spec: Spectrogram [time × freq]
            omega: Temporal modulation rate (Hz)
            Omega: Spectral modulation scale (cycles/octave)
            filter_params: (sigma_t_mult, sigma_f_mult, theta, alpha)

        Returns:
            Magnitude of filtered response [time × freq]
        """
        xp = self.xp
        
        n_time, n_freq = spec.shape
        octaves_per_bin = self.bandwidth_oct / n_freq

        # Create coordinate grids
        t_grid = (xp.arange(n_time) - n_time / 2) * self.time_per_frame
        f_grid = (xp.arange(n_freq) - n_freq / 2) * octaves_per_bin
        T, F = xp.meshgrid(t_grid, f_grid, indexing="ij")

        sigma_t_mult, sigma_f_mult, theta, alpha = filter_params
        gabor_kernel = self._create_gabor_filter(
            omega, Omega, T, F, sigma_t_mult, sigma_f_mult, theta, alpha
        )

        filtered = self.signal.fftconvolve(spec, gabor_kernel, mode="same")
        return xp.abs(filtered)

    def compute(
        self,
        spectrogram: Spectrogram,
        params: np.ndarray | None = None,
    ) -> RSF:
        """
        Compute RSF representation from auditory spectrogram.

        Integrates Gabor filter responses over 500ms windows with 10ms shift.

        Args:
            spectrogram: Auditory spectrogram (Spectrogram object or [freq × time] array)
            params: Optional filter parameter indices [n_filters × 4]

        Returns:
            RSF object with shape [n_frames × n_rates × n_scales × n_freq]
        """
        xp = self.xp
        
        if isinstance(spectrogram, Spectrogram):
            spec = xp.asarray(spectrogram.data.T)  # [time × freq]
            freqs = spectrogram.freqs
        else:
            spec = xp.asarray(spectrogram.T)
            freqs = np.arange(spec.shape[1])

        n_time, n_freq = spec.shape

        # Decode filter parameters
        if params is None:
            params = self._get_default_params()
        decoded_params = self._decode_params(params)

        # Frame parameters
        window_size = int(self.rsf_frame_size_ms / 1000.0 / self.time_per_frame)
        frame_shift = max(
            1, int(self.rsf_frame_shift_ms / 1000.0 / self.time_per_frame)
        )
        n_frames = max(1, (n_time - window_size) // frame_shift + 1)

        if n_frames == 1:
            window_size = n_time

        # Output array
        rsf_data = xp.zeros((n_frames, len(self.rates), len(self.scales), n_freq))

        # Apply all filters
        n_scales = len(self.scales)
        for i, omega in enumerate(self.rates):
            for j, Omega in enumerate(self.scales):
                filter_idx = i * n_scales + j
                filter_params = tuple(decoded_params[filter_idx])

                # Apply Gabor filter: R(ω, Ω, t, f|Λ)
                filtered = self._apply_gabor_filter(spec, omega, Omega, filter_params)

                # Integrate over time windows: T(ω, Ω, f|Λ) = ∫R dt
                for k in range(n_frames):
                    start = k * frame_shift
                    end = min(start + window_size, n_time)
                    rsf_data[k, i, j, :] = xp.mean(filtered[start:end, :], axis=0)

        # Transfer back to CPU
        rsf_data = to_numpy(rsf_data)

        # Build time axis for frames
        frame_period = self.rsf_frame_shift_ms / 1000.0
        times = np.arange(n_frames) * frame_period

        return RSF(
            data=rsf_data,
            times=times,
            rates=self.rates,
            scales=self.scales,
            freqs=freqs,
        )

    def _get_default_params(self) -> np.ndarray:
        """Get default parameter indices for all filters."""
        n_filters = len(self.rates) * len(self.scales)
        return np.full((n_filters, 4), DEFAULT_PARAM_IDX, dtype=np.int32)

    def _decode_params(self, indices: np.ndarray) -> np.ndarray:
        """Convert parameter indices to actual values."""
        return np.column_stack(
            [
                PARAM_OPTIONS["sigma_t"][indices[:, 0]],
                PARAM_OPTIONS["sigma_f"][indices[:, 1]],
                PARAM_OPTIONS["theta"][indices[:, 2]],
                PARAM_OPTIONS["alpha"][indices[:, 3]],
            ]
        )