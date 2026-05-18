"""
Auditory Spectrogram (Cochlear Model)

Implements the auditory spectrogram computation from the NSL Toolbox,
following Chi, Ru & Shamma (2005) and used in Bellur & Elhilali (2017).

Pipeline:
    y1: Cochlear filtering (gammatone filterbank)
    y2: Transduction (hair cell response)
    y3: Lateral inhibition
    y4: Half-wave rectification
    y5: Leaky integration + compression
"""

import numpy as np

from .config import Config
from .structs import Spectrogram
from .backend import (
    get_array_module,
    get_signal_module,
    to_numpy,
    next_fast_len,
    get_dtypes,
)

# Optional GPU fast path for the y1 stage. A single CUDA kernel launch
# runs all SOS cascades in parallel, replacing the per-channel sosfilt
# loop. Falls back to the loop when unavailable (CPU mode, no nvrtc,
# float64 pipeline, etc.).
try:
    from .gammatone_kernel import (
        batched_sosfilt as _batched_sosfilt_impl,
        is_available as _kernel_is_available,
    )
except ImportError:
    _batched_sosfilt_impl = None

    def _kernel_is_available() -> bool:
        return False


class AuditorySpectrogram:
    """
    Compute auditory spectrogram mimicking peripheral auditory processing.

    y(t,f) = (max(δf(a(t) * hc(t,f)), 0) * w(t,τ))^(1/3)
    """

    def __init__(self, config: Config | None = None):
        cfg = config or Config()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.f_min = cfg.f_min
        self.octaves = cfg.octaves
        self.tau_ms = cfg.tau_ms
        self.frmlen_ms = cfg.frmlen_ms
        self.use_gpu = cfg.use_gpu

        self.xp = get_array_module(self.use_gpu)
        self.signal = get_signal_module(self.use_gpu)
        self.float_dtype, self.complex_dtype = get_dtypes()

        self.filter_order = 4
        frame_adjustment = 2 ** (self.filter_order - 1)
        self.alph = np.exp(-1 / (self.tau_ms * frame_adjustment))

        self.f_max = self.f_min * (2**self.octaves)
        self.center_freqs = self._create_frequency_scale()

        # Precompute at init (config-dependent, never changes)
        self._L_frm = int((self.frmlen_ms / 1000.0) * self.sample_rate)
        self._init_gammatone_filters()
        self._init_y5_kernel()
        self._init_y1_fast_path()

        # Lazy cache (input-length dependent, rebuilt on shape change)
        self._cached_n_samples = None
        self._y5_kernel_fft = None
        self._y5_n_fft = None
        self._y5_pad = None

    # ----- init helpers (run once) -------------------------------------------

    def _create_frequency_scale(self) -> np.ndarray:
        return np.logspace(
            np.log2(self.f_min), np.log2(self.f_max), self.n_filters, base=2.0
        )

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(self.float_dtype)
        audio = audio - np.mean(audio)
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        return audio

    def _init_gammatone_filters(self) -> None:
        """Build SOS coefficients and stack on device (no per-call transfer)."""
        xp = self.xp
        erb_scale = 0.6
        T = 1.0 / self.sample_rate

        ERB = 24.7 * (4.37 * self.center_freqs / 1000.0 + 1.0) * erb_scale
        B = 1.019 * 2 * np.pi * ERB

        sos_list = []
        for fc, bw in zip(self.center_freqs, B):
            omega = 2 * np.pi * fc
            r = np.exp(-bw * T)
            theta = omega * T

            a0, a1, a2 = 1.0, -2.0 * r * np.cos(theta), r * r
            b0, b1, b2 = 1.0, 0.0, 0.0
            sos = np.array([[b0, b1, b2, a0, a1, a2]] * self.filter_order)

            w = 2 * np.pi * fc / self.sample_rate
            z = np.exp(1j * w)
            H_section = (b0 + b1 * z**-1 + b2 * z**-2) / (a0 + a1 * z**-1 + a2 * z**-2)
            gain = np.abs(H_section**self.filter_order)
            if gain > 0:
                sos[0, 0] = b0 / gain

            sos_list.append(sos)

        # Stack to (n_filters, filter_order, 6) and transfer once.
        # SOS stays float64 for sosfilt numerical stability.
        sos_stack = np.stack(sos_list, axis=0)
        self._sos_device = xp.asarray(sos_stack)

    def _init_y5_kernel(self) -> None:
        """Precompute leaky integration kernel (config-dependent only).
        The FFT is lazily computed in _ensure_cache since it depends on input length."""
        tau_sec = self.tau_ms / 1000.0
        self._tau_samples = int(tau_sec * self.sample_rate)
        t = np.arange(self._tau_samples, dtype=np.float64) / self.sample_rate
        kernel = np.exp(-t / tau_sec)
        kernel = kernel / kernel.sum()
        self._y5_kernel_host = kernel.astype(self.float_dtype)

    def _init_y1_fast_path(self) -> None:
        """Resolve the GPU batched-SOS kernel for the y1 stage, if available.

        Pre-converts the SOS bank to float32 once. The kernel runs the
        biquad recurrence in float32 (~8x faster than float64 on consumer
        Ampere, where FP64 throughput is throttled to 1/64 of FP32). The
        kernel currently only handles float32 audio, so we gate on that.
        """
        self._batched_sosfilt = None
        self._sos_device_f32 = None
        if (
            self.use_gpu
            and _batched_sosfilt_impl is not None
            and _kernel_is_available()
            and self.float_dtype == np.float32
        ):
            self._batched_sosfilt = _batched_sosfilt_impl
            self._sos_device_f32 = self._sos_device.astype(np.float32)

    # ----- lazy cache (rebuilt on input length change) -----------------------

    def _ensure_cache(self, n_samples: int) -> None:
        """Rebuild cached FFT data if input length changed."""
        if n_samples == self._cached_n_samples:
            return

        xp = self.xp
        n_conv = n_samples + self._tau_samples - 1
        n_fft = next_fast_len(n_conv, self.use_gpu)

        kernel_device = xp.asarray(self._y5_kernel_host)
        self._y5_n_fft = n_fft
        self._y5_pad = (self._tau_samples - 1) // 2
        # [1, n_fft] for broadcasting across all channels
        self._y5_kernel_fft = xp.fft.rfft(kernel_device, n=n_fft)[None, :]
        self._cached_n_samples = n_samples

    # ----- pipeline stages ---------------------------------------------------

    def _y1_cochlear_filter(self, audio):
        """Stage 1: Gammatone filterbank using cached SOS coefficients.

        GPU fast path: single batched-SOS kernel launch, float32 internal.
        Fallback: per-channel scipy/cupyx sosfilt loop at self.float_dtype.
        """
        xp = self.xp
        audio_device = xp.asarray(audio, dtype=self.float_dtype)

        if self._batched_sosfilt is not None:
            return self._batched_sosfilt(
                self._sos_device_f32,
                audio_device,
                gain=2.0,
                precision="float32",
            )

        n_samples = audio_device.shape[0]
        output = xp.empty((self.n_filters, n_samples), dtype=self.float_dtype)
        for i in range(self.n_filters):
            output[i, :] = 2.0 * self.signal.sosfilt(self._sos_device[i], audio_device)
        return output

    def _y2_transduction(self, y1):
        """Stage 2: Hair cell transduction (derivative + compression)."""
        xp = self.xp
        y2 = xp.diff(y1, axis=1, prepend=y1[:, 0:1])
        return xp.tanh(y2 * 0.5)

    def _y3_lateral_inhibition(self, y2):
        """Stage 3: Lateral inhibitory network."""
        xp = self.xp
        y3 = xp.zeros_like(y2)
        y3[:-1, :] = y2[:-1, :] - y2[1:, :]
        y3[-1, :] = y2[-1, :]
        return y3

    def _y4_rectification(self, y3):
        """Stage 4: Half-wave rectification."""
        return self.xp.maximum(y3, 0)

    def _y5_integration(self, y4):
        """Stage 5: Leaky integration using cached kernel FFT (batch across all channels)."""
        xp = self.xp
        n_fft = self._y5_n_fft

        Y4_fft = xp.fft.rfft(y4, n=n_fft, axis=1)
        y5_full = xp.fft.irfft(Y4_fft * self._y5_kernel_fft, n=n_fft, axis=1)

        pad = self._y5_pad
        n = y4.shape[1]
        return y5_full[:, pad : pad + n]

    def _downsample(self, spectrogram):
        """Downsample using device-native resample_poly (no CPU round-trip on GPU)."""
        return self.signal.resample_poly(spectrogram, up=1, down=self._L_frm, axis=1)

    # ----- public API --------------------------------------------------------

    def compute_device(self, audio: np.ndarray):
        """
        Hot path. Returns spectrogram on device in (n_time, n_freq) orientation.

        This is the orientation GaborFilterbank.compute_device() expects.
        No host transfer — stays on GPU if use_gpu=True.

        Args:
            audio: 1D numpy array

        Returns:
            Device array (numpy or cupy) of shape (n_time, n_freq)
        """
        xp = self.xp

        if audio.ndim > 1:
            audio = audio.flatten()

        audio = self._preprocess_audio(audio)
        self._ensure_cache(len(audio))

        y1 = self._y1_cochlear_filter(audio)
        y2 = self._y2_transduction(y1)
        y3 = self._y3_lateral_inhibition(y2)
        y4 = self._y4_rectification(y3)
        y5 = self._y5_integration(y4)
        y5 = xp.cbrt(y5)
        y5 = self._downsample(y5)

        return y5.T  # (n_freq, n_time) → (n_time, n_freq)

    def compute(self, audio: np.ndarray) -> Spectrogram:
        """
        Compute auditory spectrogram. Returns Spectrogram dataclass on host.

        Spectrogram.data uses the legacy (n_freq, n_time) orientation.

        Args:
            audio: Input audio signal (1D array)

        Returns:
            Spectrogram object with data and metadata
        """
        device_out = self.compute_device(audio)  # (n_time, n_freq)
        host = to_numpy(device_out).T  # (n_freq, n_time)

        frame_period = self.frmlen_ms / 1000.0
        times = np.arange(host.shape[1]) * frame_period

        return Spectrogram(
            data=host,
            times=times,
            freqs=self.center_freqs,
            sr=self.sample_rate,
        )
