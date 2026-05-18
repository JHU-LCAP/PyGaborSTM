"""
Core class for PyGaborSTM.

Usage:
    import pygaborstm as stm

    model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))

    # Individual stages (returns dataclasses on host)
    spec = model.spectrogram(audio)
    rsf = model.rsf(spec)

    # Full pipeline on device (no intermediate host transfers)
    rsf = model.compute(audio)

    # Device-only hot path (stays on GPU, no host transfer)
    rsf_device = model.compute_device(audio)
"""

import numpy as np

from .config import Config
from .spectrogram import AuditorySpectrogram
from .gabor import GaborFilterbank
from .structs import Spectrogram, RSF
from .backend import to_numpy


class PyGaborSTM:
    """
    Main interface for spectro-temporal modulation analysis.

    Args:
        config: Configuration object (optional, uses defaults if None)

    Example:
        >>> model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
        >>> spec = model.spectrogram(audio)
        >>> rsf = model.rsf(spec)
        >>> # or, full pipeline with no intermediate host transfers:
        >>> rsf = model.compute(audio)
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._spec_model = AuditorySpectrogram(self.config)
        self._gabor_model = GaborFilterbank(self.config)

    def spectrogram(self, audio: np.ndarray) -> Spectrogram:
        """
        Compute auditory spectrogram from audio.

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            Spectrogram object (data on host)
        """
        return self._spec_model.compute(audio)

    def rsf(self, spec: Spectrogram) -> RSF:
        """
        Compute RSF representation from spectrogram.

        Args:
            spec: Auditory spectrogram

        Returns:
            RSF object (data on host)
        """
        return self._gabor_model.compute(spec)

    # NOTE: chained compute_device runs ~15ms slower than calling
    # spectrogram()/rsf() separately on single files. Cause appears to be
    # device-side allocation pattern interacting with cuFFT — gabor stage runs
    # ~17ms when spec arrives via host-DMA upload but ~41ms when spec arrives
    # from a prior device kernel. Batch throughput is unaffected. Investigate
    # with nsys if it ever matters.
    def compute_device(self, audio: np.ndarray):
        """
        Full pipeline on device. No intermediate host transfers.

        Spectrogram output stays on device and feeds directly into the
        Gabor stage. Only use this when you don't need the intermediate
        Spectrogram dataclass.

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            Device array (numpy or cupy) of shape (n_frames, n_rates, n_scales, n_freq)
        """
        spec_device = self._spec_model.compute_device(audio)
        return self._gabor_model.compute_device(spec_device)

    def compute(self, audio: np.ndarray) -> RSF:
        """
        Full pipeline: audio → spectrogram → RSF.

        Chains both stages on device with no intermediate host transfer,
        then copies the final result to host and wraps in an RSF dataclass.

        Args:
            audio: Audio signal (1D numpy array)

        Returns:
            RSF object (data on host)
        """
        rsf_device = self.compute_device(audio)
        rsf_data = to_numpy(rsf_device)

        frame_period = self.config.rsf_frame_shift_ms / 1000.0
        times = np.arange(rsf_data.shape[0]) * frame_period

        return RSF(
            data=rsf_data,
            times=times,
            rates=self._gabor_model.rates,
            scales=self._gabor_model.scales,
            freqs=self._spec_model.center_freqs,
        )