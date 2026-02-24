"""
Core class for PyGaborSTM.

Usage:
    import pygaborstm as stm
    
    model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
    spec = model.spectrogram(audio)
    rsf = model.rsf(spec)
"""

import numpy as np

from .config import Config
from .spectrogram import AuditorySpectrogram
from .gabor import GaborFilterbank
from .structs import Spectrogram, RSF


class PyGaborSTM:
    """
    Main interface for spectro-temporal modulation analysis.
    
    Args:
        config: Configuration object (optional, uses defaults if None)
    
    Example:
        >>> model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
        >>> spec = model.spectrogram(audio)
        >>> rsf = model.rsf(spec)
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
            Spectrogram object
        """
        return self._spec_model.compute(audio)

    def rsf(self, spec: Spectrogram) -> RSF:
        """
        Compute RSF representation from spectrogram.

        Args:
            spec: Auditory spectrogram

        Returns:
            RSF object
        """
        return self._gabor_model.compute(spec)