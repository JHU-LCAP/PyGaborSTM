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
from .backend import get_array_module, get_signal_module, to_numpy, next_fast_len, get_dtypes, get_available_memory


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

    # Resolution presets (multipliers relative to "low")
    RESOLUTION_MULTIPLIERS = {
        "low": 1,
        "medium": 2,
        "high": 4,
        "ultra": 6,
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
        self.float_dtype, self.complex_dtype = get_dtypes()

        self.frmlen_ms = cfg.frmlen_ms
        self.bandwidth_oct = cfg.octaves
        self.time_per_frame = self.frmlen_ms / 1000.0

        # Get rates/scales based on config and resolution
        self.rates, self.scales = self._get_rates_scales(cfg)

    def _get_rates_scales(self, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rate and scale arrays based on config and resolution."""
        cfg_rates = np.asarray(cfg.rates, dtype=np.float64)
        cfg_scales = np.asarray(cfg.scales, dtype=np.float64)
        
        if cfg.resolution not in self.RESOLUTION_MULTIPLIERS:
            raise ValueError(
                f"Invalid resolution '{cfg.resolution}'. "
                f"Choose from {list(self.RESOLUTION_MULTIPLIERS.keys())}"
            )
        
        multiplier = self.RESOLUTION_MULTIPLIERS[cfg.resolution]
        
        if multiplier == 1:
            return cfg_rates, cfg_scales
        
        pos_rates = cfg_rates[cfg_rates > 0]
        rate_min, rate_max = pos_rates.min(), pos_rates.max()
        scale_min, scale_max = cfg_scales.min(), cfg_scales.max()
        
        n_rates_pos = len(pos_rates) * multiplier
        n_scales = len(cfg_scales) * multiplier
        
        rates_pos = np.logspace(np.log2(rate_min), np.log2(rate_max), n_rates_pos, base=2)
        rates = np.concatenate([-rates_pos[::-1], rates_pos])
        scales = np.logspace(np.log2(scale_min), np.log2(scale_max), n_scales, base=2)
        
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

        return (gaussian * carrier).astype(self.complex_dtype)

    def _apply_gabor_filter(
        self,
        spec_fft,
        pad_shape,
        crop_t: int,
        crop_f: int,
        n_time: int,
        n_freq: int,
        omega: float,
        Omega: float,
        T,
        F,
        filter_params: Tuple[float, float, float, float] = (0.5, 0.5, 0.0, 1.0),
    ):
        """
        Apply a single Gabor filter using precomputed spectrogram FFT.

        Args:
            spec_fft: Precomputed FFT of spectrogram (computed once, reused)
            pad_shape: Padded FFT dimensions
            crop_t: Time crop offset for "same" mode
            crop_f: Frequency crop offset for "same" mode
            n_time: Original time dimension
            n_freq: Original frequency dimension
            omega: Temporal modulation rate (Hz)
            Omega: Spectral modulation scale (cycles/octave)
            T: Precomputed time meshgrid
            F: Precomputed frequency meshgrid
            filter_params: (sigma_t_mult, sigma_f_mult, theta, alpha)

        Returns:
            Magnitude of filtered response [time × freq]
        """
        xp = self.xp

        sigma_t_mult, sigma_f_mult, theta, alpha = filter_params
        gabor_kernel = self._create_gabor_filter(
            omega, Omega, T, F, sigma_t_mult, sigma_f_mult, theta, alpha
        )

        kernel_fft = xp.fft.fft2(gabor_kernel, s=pad_shape)
        filtered_full = xp.fft.ifft2(spec_fft * kernel_fft, s=pad_shape)

        # crop to "same" size and take magnitude
        filtered = xp.abs(
            filtered_full[crop_t : crop_t + n_time, crop_f : crop_f + n_freq]
        )
        return filtered

    def _prepare_spectrogram(self, spectrogram: Spectrogram):
        """Parse spectrogram input into array and frequency axis."""
        xp = self.xp

        if isinstance(spectrogram, Spectrogram):
            spec = xp.asarray(spectrogram.data.T, dtype=self.float_dtype)
            freqs = spectrogram.freqs
        else:
            spec = xp.asarray(spectrogram.T, dtype=self.float_dtype)
            freqs = np.arange(spec.shape[1])

        return spec, freqs

    def _compute_frame_params(self, n_time: int) -> Tuple[int, int, int]:
        """Compute RSF frame windowing parameters."""
        window_size = int(self.rsf_frame_size_ms / 1000.0 / self.time_per_frame)
        frame_shift = max(
            1, int(self.rsf_frame_shift_ms / 1000.0 / self.time_per_frame)
        )
        n_frames = max(1, (n_time - window_size) // frame_shift + 1)

        if n_frames == 1:
            window_size = n_time

        return window_size, frame_shift, n_frames

    def _build_meshgrid(self, n_time: int, n_freq: int):
        """Build time-frequency coordinate grids for Gabor filter construction."""
        xp = self.xp
        octaves_per_bin = self.bandwidth_oct / n_freq
        t_grid = (xp.arange(n_time, dtype=self.float_dtype) - n_time / 2) * self.time_per_frame
        f_grid = (xp.arange(n_freq, dtype=self.float_dtype) - n_freq / 2) * octaves_per_bin
        return xp.meshgrid(t_grid, f_grid, indexing="ij")

    def _precompute_spec_fft(self, spec, n_time: int, n_freq: int):
        """Compute padded FFT of spectrogram for reuse across all filters."""
        xp = self.xp
        pad_shape = (
            next_fast_len(n_time + n_time - 1, self.use_gpu),
            next_fast_len(n_freq + n_freq - 1, self.use_gpu),
        )
        spec_fft = xp.fft.fft2(spec, s=pad_shape)
        crop_t = (n_time - 1) // 2
        crop_f = (n_freq - 1) // 2
        return spec_fft, pad_shape, crop_t, crop_f

    def _build_frame_indices(self, n_frames: int, frame_shift: int,
                             window_size: int, n_time: int):
        """Precompute frame integration indices for vectorized windowing."""
        xp = self.xp
        starts = xp.arange(n_frames) * frame_shift
        offsets = xp.arange(window_size)
        frame_indices = starts[:, None] + offsets[None, :]
        return xp.clip(frame_indices, 0, n_time - 1)

    def _build_all_kernels(self, T, F, decoded_params):
        """Build all Gabor kernels and stack into a 3D tensor."""
        xp = self.xp
        n_scales = len(self.scales)
        kernels = []

        for i, omega in enumerate(self.rates):
            for j, Omega in enumerate(self.scales):
                filter_idx = i * n_scales + j
                sigma_t_mult, sigma_f_mult, theta, alpha = decoded_params[filter_idx]
                kernel = self._create_gabor_filter(
                    omega, Omega, T, F, sigma_t_mult, sigma_f_mult, theta, alpha
                )
                kernels.append(kernel)

        return xp.stack(kernels)  # [n_filters x n_time x n_freq]

    def _auto_batch_size(self, n_filters: int, pad_shape: Tuple[int, int]) -> int:
        """
        Calculate batch size based on available memory.

        At peak, each filter in a batch needs ~3 padded FFT arrays
        (batch_fft, filtered_fft, filtered_full) simultaneously alive.
        Uses 50% of available memory to leave headroom.
        """
        bytes_per_complex = np.dtype(self.complex_dtype).itemsize
        mem_per_filter = 3 * pad_shape[0] * pad_shape[1] * bytes_per_complex
        available = get_available_memory(self.use_gpu)
        batch_size = max(1, int(available * 0.5 / mem_per_filter))
        return min(batch_size, n_filters)

    def _apply_filters_batched(self, spec_fft, pad_shape, crop_t, crop_f,
                                n_time, n_freq, kernels, frame_indices,
                                batch_size: int | None = None):
        """
        Apply all Gabor filters using batched tensor operations.

        Processes filters in chunks of batch_size to control memory usage.
        At low resolution (60 filters), runs in one batch. At higher
        resolutions, loops over batches while keeping tensor ops inside.

        Args:
            spec_fft: Precomputed spectrogram FFT [pad_time x pad_freq]
            pad_shape: Padded FFT dimensions
            crop_t, crop_f: Crop offsets for "same" mode
            n_time, n_freq: Original spectrogram dimensions
            kernels: Stacked Gabor kernels [n_filters x n_time x n_freq]
            frame_indices: Precomputed frame indices [n_frames x window_size]
            batch_size: Filters per batch. None = auto (fits to available memory).

        Returns:
            RSF data [n_frames x n_rates x n_scales x n_freq]
        """
        xp = self.xp
        n_filters = kernels.shape[0]
        n_frames = frame_indices.shape[0]

        if batch_size is None:
            batch_size = self._auto_batch_size(n_filters, pad_shape)

        # Accumulate results across batches
        rsf_flat = xp.zeros((n_filters, n_frames, n_freq))

        for start in range(0, n_filters, batch_size):
            end = min(start + batch_size, n_filters)
            batch = kernels[start:end]

            # Batch FFT this chunk of kernels
            batch_fft = xp.fft.fft2(batch, s=pad_shape, axes=(-2, -1))

            # Broadcast multiply with cached spec_fft
            filtered_fft = batch_fft * spec_fft[None, :, :]

            # Batch IFFT
            filtered_full = xp.fft.ifft2(filtered_fft, s=pad_shape, axes=(-2, -1))

            # Crop to "same" size and take magnitude
            filtered = xp.abs(
                filtered_full[:, crop_t : crop_t + n_time, crop_f : crop_f + n_freq]
            )

            # Batch frame integration
            rsf_flat[start:end] = filtered[:, frame_indices, :].mean(axis=2)

        # Reshape to [n_frames x n_rates x n_scales x n_freq]
        n_rates = len(self.rates)
        n_scales = len(self.scales)
        rsf_data = rsf_flat.reshape(n_rates, n_scales, n_frames, n_freq)
        rsf_data = rsf_data.transpose(2, 0, 1, 3)

        return rsf_data

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
        spec, freqs = self._prepare_spectrogram(spectrogram)
        n_time, n_freq = spec.shape

        if params is None:
            params = self._get_default_params()
        decoded_params = self._decode_params(params)

        window_size, frame_shift, n_frames = self._compute_frame_params(n_time)

        # Precompute shared data
        T, F = self._build_meshgrid(n_time, n_freq)
        spec_fft, pad_shape, crop_t, crop_f = self._precompute_spec_fft(spec, n_time, n_freq)
        frame_indices = self._build_frame_indices(n_frames, frame_shift, window_size, n_time)

        # Build all kernels, apply as one batched operation
        kernels = self._build_all_kernels(T, F, decoded_params)
        rsf_data = self._apply_filters_batched(
            spec_fft, pad_shape, crop_t, crop_f,
            n_time, n_freq, kernels, frame_indices
        )

        rsf_data = to_numpy(rsf_data)
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