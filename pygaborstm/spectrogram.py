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

from pathlib import Path
import numpy as np
from scipy import signal

from .config import SpectrogramConfig
from .structs import Spectrogram


class AuditorySpectrogram:
    """
    Compute auditory spectrogram mimicking peripheral auditory processing.

    y(t,f) = (max(δf(a(t) * hc(t,f)), 0) * w(t,τ))^(1/3)
    """

    def __init__(self, config: SpectrogramConfig | None = None):
        cfg = config or SpectrogramConfig()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.f_min = cfg.f_min
        self.octaves = cfg.octaves
        self.tau_ms = cfg.tau_ms
        self.frmlen_ms = cfg.frmlen_ms
        self.constant_Q = cfg.constant_Q

        self.filter_order = 4
        frame_adjustment = 2 ** (self.filter_order - 1)
        self.alph = np.exp(-1 / (self.tau_ms * frame_adjustment))

        self.f_max = self.f_min * (2 ** self.octaves)
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
            H_section = (b0 + b1 * z**-1 + b2 * z**-2) / \
                (a0 + a1 * z**-1 + a2 * z**-2)
            gain = np.abs(H_section ** filter_order)

            if gain > 0:
                sos[0, 0] = b0 / gain

            self._gammatone_sos.append(sos)

    def _y1_cochlear_filter(self, audio: np.ndarray) -> np.ndarray:
        """Stage 1: Apply gammatone filterbank."""
        n_samples = len(audio)
        output = np.zeros((self.n_filters, n_samples))

        for i, sos in enumerate(self._gammatone_sos):
            output[i, :] = 2.0 * signal.sosfilt(sos, audio)

        return output

    def _y2_transduction(self, y1: np.ndarray) -> np.ndarray:
        """Stage 2: Hair cell transduction (derivative + compression)."""
        y2 = np.diff(y1, axis=1, prepend=y1[:, 0:1])
        scale = 0.5
        return np.tanh(y2 * scale)

    def _y3_lateral_inhibition(self, y2: np.ndarray) -> np.ndarray:
        """Stage 3: Lateral inhibitory network."""
        y3 = np.zeros_like(y2)
        y3[:-1, :] = y2[:-1, :] - y2[1:, :]
        y3[-1, :] = y2[-1, :]
        return y3

    def _y4_rectification(self, y3: np.ndarray) -> np.ndarray:
        """Stage 4: Half-wave rectification."""
        return np.maximum(y3, 0)

    def _y5_integration(self, y4: np.ndarray) -> np.ndarray:
        """Stage 5: Leaky temporal integration."""
        tau_sec = self.tau_ms / 1000.0
        tau_samples = int(tau_sec * self.sample_rate)
        t = np.arange(tau_samples) / self.sample_rate

        kernel = np.exp(-t / tau_sec)
        kernel = kernel / kernel.sum()

        y5 = np.zeros_like(y4)
        for i in range(y4.shape[0]):
            y5[i, :] = np.convolve(y4[i, :], kernel, mode="same")

        return y5

    def _downsample(self, spectrogram: np.ndarray) -> np.ndarray:
        """Downsample to frame rate."""
        shft = 0
        L_frm = int((self.frmlen_ms / 1000.0) * self.sample_rate * (2 ** shft))

        n_samples = spectrogram.shape[1]
        n_frames = int(np.ceil(n_samples / L_frm))

        if n_samples < n_frames * L_frm:
            pad_width = ((0, 0), (0, n_frames * L_frm - n_samples))
            spectrogram = np.pad(spectrogram, pad_width, mode="constant")

        return spectrogram[:, L_frm - 1:: L_frm]

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
        y5 = np.cbrt(y5)
        y5 = self._downsample(y5)

        # Build time axis
        frame_period = self.frmlen_ms / 1000.0
        times = np.arange(y5.shape[1]) * frame_period

        return Spectrogram(
            data=y5,
            times=times,
            freqs=self.center_freqs,
            sr=self.sample_rate,
        )


def auditory_spectrogram(audio: np.ndarray, config: SpectrogramConfig | None = None) -> Spectrogram:
    """Functional interface for computing auditory spectrogram."""
    return AuditorySpectrogram(config).compute(audio)
