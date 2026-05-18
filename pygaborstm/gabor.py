"""
Gabor Filterbank and RSF Extraction

Implements 2D Gabor filters for spectro-temporal modulation analysis,
following Bellur & Elhilali (2017).

Pipeline:
    1. Cached: 2D Gabor kernels tuned to (rate, scale) pairs, plus their FFTs
    2. Hot path: FFT input spectrogram → broadcast multiply → batch IFFT
       → magnitude → frame integration

Memory-adaptive: at high resolutions the kernel FFT cache can exceed GPU
memory. When it would, falls back to streaming mode (kernels are rebuilt
and FFT'd per chunk inside the compute loop). Slower per call but works on
any GPU, including lower-memory devices like the Jetson Orin Nano.
"""

import warnings
import numpy as np
from typing import Tuple

from .config import Config
from .structs import Spectrogram, RSF
from .backend import (
    get_array_module,
    to_numpy,
    next_fast_len,
    get_dtypes,
    get_available_memory,
)


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

    RESOLUTION_MULTIPLIERS = {
        "low": 1,
        "medium": 2,
        "high": 4,
        "ultra": 8,
        "max": 16,
        "overkill": 32,
    }

    # Above this kernel count, warn at construction time: compute work scales
    # linearly with n_kernels regardless of GPU memory or streaming mode.
    _KERNEL_COUNT_WARN_THRESHOLD = 5000

    # Fraction of available GPU memory we're willing to dedicate to the
    # kernel FFT cache. Below this, build the cache; above it, stream.
    _CACHE_MEMORY_BUDGET = 0.7

    def __init__(self, config: Config | None = None):
        cfg = config or Config()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.rsf_frame_size_ms = cfg.rsf_frame_size_ms
        self.rsf_frame_shift_ms = cfg.rsf_frame_shift_ms
        self.use_gpu = cfg.use_gpu

        self.xp = get_array_module(self.use_gpu)
        self.float_dtype, self.complex_dtype = get_dtypes()

        self.frmlen_ms = cfg.frmlen_ms
        self.bandwidth_oct = cfg.octaves
        self.time_per_frame = self.frmlen_ms / 1000.0

        self.rates, self.scales = self._get_rates_scales(cfg)
        self._n_rates = len(self.rates)
        self._n_scales = len(self.scales)
        self._n_kernels = self._n_rates * self._n_scales

        if self._n_kernels > self._KERNEL_COUNT_WARN_THRESHOLD:
            warnings.warn(
                f"Resolution '{cfg.resolution}' produces {self._n_kernels} "
                f"kernels ({self._n_rates} rates × {self._n_scales} scales). "
                f"Compute scales linearly with this regardless of available "
                f"memory — expect significantly longer per-file processing. "
                f"Lower resolution if interactive performance matters.",
                ResourceWarning,
                stacklevel=2,
            )

        # Lazy cache state (rebuilt on input shape change)
        self._cached_shape = None
        self._T = None
        self._F = None
        self._pad_shape = None
        self._crop_t = None
        self._crop_f = None
        self._frame_indices = None
        self._n_frames = None
        self._batch_size = None
        self._kernel_ffts = None  # cached when it fits; None → streaming path

    # ----- rates/scales (config-dependent, computed at init) ------------------

    def _get_rates_scales(self, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
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

        rates_pos = np.logspace(
            np.log2(rate_min), np.log2(rate_max), n_rates_pos, base=2
        )
        rates = np.concatenate([-rates_pos[::-1], rates_pos])
        scales = np.logspace(np.log2(scale_min), np.log2(scale_max), n_scales, base=2)
        return rates, scales

    # ----- lazy cache (rebuilt on input shape change) -------------------------

    def _ensure_shape_cache(self, n_time: int, n_freq: int) -> None:
        """Rebuild shape-dependent caches if input dimensions changed."""
        if (n_time, n_freq) == self._cached_shape:
            return

        xp = self.xp

        # Meshgrid
        octaves_per_bin = self.bandwidth_oct / n_freq
        t_grid = (
            xp.arange(n_time, dtype=self.float_dtype) - n_time / 2
        ) * self.time_per_frame
        f_grid = (
            xp.arange(n_freq, dtype=self.float_dtype) - n_freq / 2
        ) * octaves_per_bin
        self._T, self._F = xp.meshgrid(t_grid, f_grid, indexing="ij")

        # FFT padding
        self._pad_shape = (
            next_fast_len(2 * n_time - 1, self.use_gpu),
            next_fast_len(2 * n_freq - 1, self.use_gpu),
        )
        self._crop_t = (n_time - 1) // 2
        self._crop_f = (n_freq - 1) // 2

        # Frame integration
        window_size = int(self.rsf_frame_size_ms / 1000.0 / self.time_per_frame)
        frame_shift = max(
            1, int(self.rsf_frame_shift_ms / 1000.0 / self.time_per_frame)
        )
        n_frames = max(1, (n_time - window_size) // frame_shift + 1)
        if n_frames == 1:
            window_size = n_time

        self._n_frames = n_frames
        starts = xp.arange(n_frames) * frame_shift
        offsets = xp.arange(window_size)
        self._frame_indices = xp.clip(starts[:, None] + offsets[None, :], 0, n_time - 1)

        # Batch size
        bytes_per_complex = np.dtype(self.complex_dtype).itemsize
        mem_per_filter = 3 * self._pad_shape[0] * self._pad_shape[1] * bytes_per_complex
        available = get_available_memory(self.use_gpu)
        self._batch_size = min(
            self._n_kernels, max(1, int(available * 0.5 / mem_per_filter))
        )

        # Invalidate kernel cache (shape changed)
        self._kernel_ffts = None
        self._cached_shape = (n_time, n_freq)

    def _ensure_kernel_cache(self, decoded_params: np.ndarray) -> None:
        """Build and cache kernel FFTs if they fit in memory.

        If the full cache would exceed the memory budget, emits a
        ResourceWarning and leaves self._kernel_ffts as None — the signal
        to compute_device that the streaming path should be used instead.
        """
        if self._kernel_ffts is not None:
            return

        xp = self.xp
        K = self._n_kernels

        # Pre-allocation check: would the cache fit?
        bytes_per_fft = (
            self._pad_shape[0]
            * self._pad_shape[1]
            * np.dtype(self.complex_dtype).itemsize
        )
        cache_bytes = K * bytes_per_fft
        available = get_available_memory(self.use_gpu)

        if cache_bytes > available * self._CACHE_MEMORY_BUDGET:
            warnings.warn(
                f"Kernel FFT cache would need {cache_bytes / 1e9:.2f} GB; "
                f"only {available / 1e9:.2f} GB available. Falling back to "
                f"streaming mode — kernels rebuilt each compute() call. "
                f"This works on any GPU but is slower than the cached path.",
                ResourceWarning,
                stacklevel=2,
            )
            return  # _kernel_ffts stays None → streaming dispatch

        # Cache fits. Build it in chunks so peak construction memory stays
        # bounded (the old all-at-once fft2 would itself OOM at high N
        # even when the final cache had room).
        self._kernel_ffts = xp.empty(
            (K, *self._pad_shape),
            dtype=self.complex_dtype,
        )
        for start in range(0, K, self._batch_size):
            end = min(start + self._batch_size, K)
            kernels_chunk = self._build_kernels_range(decoded_params, start, end)
            self._kernel_ffts[start:end] = xp.fft.fft2(
                kernels_chunk,
                s=self._pad_shape,
                axes=(-2, -1),
            ).astype(self.complex_dtype)
            del kernels_chunk
            if self.use_gpu:
                xp.cuda.Stream.null.synchronize()

    # ----- kernel construction -----------------------------------------------

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
        """
        xp = self.xp

        omega_abs = max(abs(omega), 0.5)
        sigma_t = sigma_t_mult / omega_abs
        sigma_f = sigma_f_mult / Omega

        t1 = T * xp.cos(theta) + F * xp.sin(theta)
        f1 = -T * xp.sin(theta) + F * xp.cos(theta)

        gaussian = (alpha / (2 * xp.pi * sigma_t * sigma_f)) * xp.exp(
            -0.5 * (t1**2 / sigma_t**2 + f1**2 / sigma_f**2)
        )
        carrier = xp.exp(2j * xp.pi * (omega * T + Omega * F))

        return (gaussian * carrier).astype(self.complex_dtype)

    def _build_kernels_range(self, decoded_params, start: int, end: int):
        """Build kernels [start:end) in flat (rate, scale) ordering.

        Used by both the cached build (loops over the full range in chunks)
        and the streaming compute (builds one chunk per outer iteration).
        """
        xp = self.xp
        kernels = []
        for k_idx in range(start, end):
            i = k_idx // self._n_scales
            j = k_idx % self._n_scales
            omega = self.rates[i]
            Omega = self.scales[j]
            sigma_t_mult, sigma_f_mult, theta, alpha = decoded_params[k_idx]
            kernel = self._create_gabor_filter(
                omega,
                Omega,
                self._T,
                self._F,
                sigma_t_mult,
                sigma_f_mult,
                theta,
                alpha,
            )
            kernels.append(kernel)
        return xp.stack(kernels)

    # ----- hot path ----------------------------------------------------------

    def _apply_filters_batched(self, spec_fft, kernel_ffts):
        """
        Apply all filters using batched tensor operations with a pre-built
        kernel FFT cache.

        Args:
            spec_fft: FFT of input spectrogram [pad_time x pad_freq]
            kernel_ffts: Pre-FFT'd Gabor kernels [n_kernels x pad_time x pad_freq]

        Returns:
            RSF data [n_frames x n_rates x n_scales x n_freq]
        """
        xp = self.xp
        n_time = self._cached_shape[0]
        n_freq = self._cached_shape[1]
        ct, cf = self._crop_t, self._crop_f
        K = self._n_kernels

        rsf_flat = xp.zeros((K, self._n_frames, n_freq), dtype=self.float_dtype)

        for start in range(0, K, self._batch_size):
            end = min(start + self._batch_size, K)

            filtered_fft = kernel_ffts[start:end] * spec_fft[None, :, :]
            filtered_full = xp.fft.ifft2(filtered_fft, s=self._pad_shape, axes=(-2, -1))
            filtered = xp.abs(filtered_full[:, ct : ct + n_time, cf : cf + n_freq])
            rsf_flat[start:end] = filtered[:, self._frame_indices, :].mean(axis=2)

            if self.use_gpu:
                xp.cuda.Stream.null.synchronize()

        rsf_data = rsf_flat.reshape(
            self._n_rates, self._n_scales, self._n_frames, n_freq
        )
        return rsf_data.transpose(2, 0, 1, 3)

    def _apply_filters_streaming(self, spec_fft, decoded_params):
        """
        Apply filters by building kernels inline per-chunk (no full cache).

        Same structure as _apply_filters_batched, but each iteration
        constructs and FFTs its own slice of the kernel bank instead of
        indexing a prebuilt cache. Peak memory is one chunk's worth of
        kernels regardless of n_kernels, so this path works at any
        resolution given enough patience.

        Args:
            spec_fft: FFT of input spectrogram [pad_time x pad_freq]
            decoded_params: Per-kernel parameter array [n_kernels x 4]

        Returns:
            RSF data [n_frames x n_rates x n_scales x n_freq]
        """
        xp = self.xp
        n_time = self._cached_shape[0]
        n_freq = self._cached_shape[1]
        ct, cf = self._crop_t, self._crop_f
        K = self._n_kernels

        rsf_flat = xp.zeros((K, self._n_frames, n_freq), dtype=self.float_dtype)

        for start in range(0, K, self._batch_size):
            end = min(start + self._batch_size, K)

            # Build + FFT this chunk's kernels (the difference vs cached path)
            kernels_chunk = self._build_kernels_range(decoded_params, start, end)
            kernel_ffts_chunk = xp.fft.fft2(
                kernels_chunk,
                s=self._pad_shape,
                axes=(-2, -1),
            ).astype(self.complex_dtype)
            del kernels_chunk

            # Same convolution + integration as the cached path
            filtered_fft = kernel_ffts_chunk * spec_fft[None, :, :]
            del kernel_ffts_chunk
            filtered_full = xp.fft.ifft2(filtered_fft, s=self._pad_shape, axes=(-2, -1))
            del filtered_fft
            filtered = xp.abs(filtered_full[:, ct : ct + n_time, cf : cf + n_freq])
            del filtered_full
            rsf_flat[start:end] = filtered[:, self._frame_indices, :].mean(axis=2)
            del filtered

            if self.use_gpu:
                xp.cuda.Stream.null.synchronize()

        rsf_data = rsf_flat.reshape(
            self._n_rates, self._n_scales, self._n_frames, n_freq
        )
        return rsf_data.transpose(2, 0, 1, 3)

    # ----- param helpers -----------------------------------------------------

    def _get_default_params(self) -> np.ndarray:
        return np.full((self._n_kernels, 4), DEFAULT_PARAM_IDX, dtype=np.int32)

    def _decode_params(self, indices: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                PARAM_OPTIONS["sigma_t"][indices[:, 0]],
                PARAM_OPTIONS["sigma_f"][indices[:, 1]],
                PARAM_OPTIONS["theta"][indices[:, 2]],
                PARAM_OPTIONS["alpha"][indices[:, 3]],
            ]
        )

    # ----- public API --------------------------------------------------------

    def compute_device(self, spec_device, params: np.ndarray | None = None):
        """
        Hot path. Process spectrogram on device, return RSF on device.

        Memory-adaptive dispatch:
          - Default mode (params=None): try to cache kernel FFTs. If they
            fit, use the batched cached path. If not, fall back to streaming.
          - GA mode (params given): always stream. Kernels change per call,
            so caching has no benefit, and this avoids GA's own OOM risk at
            high resolution.

        Args:
            spec_device: Device array (numpy or cupy) of shape (n_time, n_freq)
            params: Optional filter parameter indices [n_filters x 4].

        Returns:
            Device array of shape (n_frames, n_rates, n_scales, n_freq)
        """
        xp = self.xp
        n_time, n_freq = spec_device.shape

        self._ensure_shape_cache(n_time, n_freq)
        spec_fft = xp.fft.fft2(spec_device, s=self._pad_shape)

        if params is None:
            decoded = self._decode_params(self._get_default_params())
            self._ensure_kernel_cache(decoded)
            if self._kernel_ffts is not None:
                return self._apply_filters_batched(spec_fft, self._kernel_ffts)
            return self._apply_filters_streaming(spec_fft, decoded)

        # GA mode: always stream
        decoded = self._decode_params(params)
        return self._apply_filters_streaming(spec_fft, decoded)

    def compute(
        self,
        spectrogram: Spectrogram,
        params: np.ndarray | None = None,
    ) -> RSF:
        """
        Compute RSF representation from spectrogram. Returns RSF dataclass on host.

        Args:
            spectrogram: Auditory spectrogram (Spectrogram object or array)
            params: Optional filter parameter indices [n_filters x 4]

        Returns:
            RSF object with shape [n_frames x n_rates x n_scales x n_freq]
        """
        xp = self.xp

        if isinstance(spectrogram, Spectrogram):
            spec_device = xp.asarray(spectrogram.data.T, dtype=self.float_dtype)
            freqs = spectrogram.freqs
        else:
            spec_device = xp.asarray(spectrogram.T, dtype=self.float_dtype)
            freqs = np.arange(spec_device.shape[1])

        device_out = self.compute_device(spec_device, params)
        rsf_data = to_numpy(device_out)

        frame_period = self.rsf_frame_shift_ms / 1000.0
        times = np.arange(rsf_data.shape[0]) * frame_period

        return RSF(
            data=rsf_data,
            times=times,
            rates=self.rates,
            scales=self.scales,
            freqs=freqs,
        )
