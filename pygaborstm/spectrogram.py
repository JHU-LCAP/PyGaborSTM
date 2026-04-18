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

from scipy.signal import resample_poly

from .config import Config
from .structs import Spectrogram
from .backend import get_array_module, get_signal_module, to_numpy


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

        # Get array and signal modules (numpy/cupy)
        self.xp = get_array_module(self.use_gpu)
        self.signal = get_signal_module(self.use_gpu)

        self.filter_order = 4
        frame_adjustment = 2 ** (self.filter_order - 1)
        self.alph = np.exp(-1 / (self.tau_ms * frame_adjustment))

        self.f_max = self.f_min * (2**self.octaves)
        self.center_freqs = self._create_frequency_scale()
        self._init_gammatone_filters()

    def _create_frequency_scale(self) -> np.ndarray:
        """Create logarithmically spaced center frequencies."""
        return np.logspace(
            np.log2(self.f_min), np.log2(self.f_max), self.n_filters, base=2.0
        )

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to zero mean and unit max amplitude."""
        audio = audio - np.mean(audio)
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        return audio

    def _init_gammatone_filters(self):
        """Pre-compute gammatone filter coefficients (SOS format)."""
        filter_order = 4
        erb_scale = 0.6
        T = 1.0 / self.sample_rate

        ERB = 24.7 * (4.37 * self.center_freqs / 1000.0 + 1.0) * erb_scale
        B = 1.019 * 2 * np.pi * ERB

        self._gammatone_sos = []

        for fc, bw in zip(self.center_freqs, B):
            omega = 2 * np.pi * fc
            r = np.exp(-bw * T)
            theta = omega * T

            a0, a1, a2 = 1.0, -2.0 * r * np.cos(theta), r * r
            b0, b1, b2 = 1.0, 0.0, 0.0

            sos = np.array([[b0, b1, b2, a0, a1, a2]] * filter_order)

            # Normalize gain at center frequency
            w = 2 * np.pi * fc / self.sample_rate
            z = np.exp(1j * w)
            H_section = (b0 + b1 * z**-1 + b2 * z**-2) / (a0 + a1 * z**-1 + a2 * z**-2)
            gain = np.abs(H_section**filter_order)

            if gain > 0:
                sos[0, 0] = b0 / gain

            self._gammatone_sos.append(sos)

    def _y1_cochlear_filter(self, audio: np.ndarray):
        """Stage 1: Apply gammatone filterbank."""
        xp = self.xp

        audio_device = xp.asarray(audio)
        n_samples = len(audio)
        output = xp.zeros((self.n_filters, n_samples))

        for i, sos in enumerate(self._gammatone_sos):
            sos_device = xp.asarray(sos)
            output[i, :] = 2.0 * self.signal.sosfilt(sos_device, audio_device)

        return output

    def _y2_transduction(self, y1):
        """Stage 2: Hair cell transduction (derivative + compression)."""
        xp = self.xp
        y2 = xp.diff(y1, axis=1, prepend=y1[:, 0:1])
        scale = 0.5
        return xp.tanh(y2 * scale)

    def _y3_lateral_inhibition(self, y2):
        """Stage 3: Lateral inhibitory network."""
        xp = self.xp
        y3 = xp.zeros_like(y2)
        y3[:-1, :] = y2[:-1, :] - y2[1:, :]
        y3[-1, :] = y2[-1, :]
        return y3

    def _y4_rectification(self, y3):
        """Stage 4: Half-wave rectification."""
        xp = self.xp
        return xp.maximum(y3, 0)

    def _y5_integration(self, y4):
        """Stage 5: Leaky temporal integration."""
        xp = self.xp

        tau_sec = self.tau_ms / 1000.0
        tau_samples = int(tau_sec * self.sample_rate)
        t = xp.arange(tau_samples) / self.sample_rate

        kernel = xp.exp(-t / tau_sec)
        kernel = kernel / kernel.sum()

        y5 = xp.zeros_like(y4)
        for i in range(y4.shape[0]):
            y5[i, :] = xp.convolve(y4[i, :], kernel, mode="same")

        return y5

    def _downsample(self, spectrogram):
        """Downsample to frame rate using polyphase filtering."""
        xp = self.xp
        
        # Calculate downsampling factor
        L_frm = int((self.frmlen_ms / 1000.0) * self.sample_rate)
        
        # resample_poly(x, up, down) - we want to downsample by L_frm
        # up=1, down=L_frm gives us 1/L_frm of the original samples
        spectrogram_np = to_numpy(spectrogram)
        
        downsampled = resample_poly(spectrogram_np, up=1, down=L_frm, axis=1)
        
        return xp.asarray(downsampled)

    def compute(self, audio: np.ndarray) -> Spectrogram:
        """
        Compute auditory spectrogram.

        Args:
            audio: Input audio signal (1D array)

        Returns:
            Spectrogram object with data and metadata
        """
        if audio.ndim > 1:
            audio = audio.flatten()

        audio = self._preprocess_audio(audio)

        y1 = self._y1_cochlear_filter(audio)
        y2 = self._y2_transduction(y1)
        y3 = self._y3_lateral_inhibition(y2)
        y4 = self._y4_rectification(y3)
        y5 = self._y5_integration(y4)
        y5 = self.xp.cbrt(y5) # causes artifacts in MRF for 32Hz
        # y5 = self.xp.log1p(self.xp.abs(y5))
        y5 = self._downsample(y5)

        # Transfer back to CPU for output
        y5 = to_numpy(y5)

        # Build time axis
        frame_period = self.frmlen_ms / 1000.0
        times = np.arange(y5.shape[1]) * frame_period

        return Spectrogram(
            data=y5,
            times=times,
            freqs=self.center_freqs,
            sr=self.sample_rate,
        )
