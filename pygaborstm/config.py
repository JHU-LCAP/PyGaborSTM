"""Configuration dataclass for PyGaborSTM.

Default values match Bellur & Elhilali (2017).
"""

from dataclasses import dataclass, field
import numpy as np

from .constants import STANDARD_RATES, STANDARD_SCALES


@dataclass
class Config:
    """Configuration for the full PyGaborSTM pipeline.

    Parameters
    ----------
    use_gpu : bool, default False
        If True, use CuPy for GPU acceleration. Silently falls back to
        NumPy with a warning if CuPy is unavailable.
    sample_rate : int, default 16000
        Audio sample rate in Hz. Must match the input signal.
    n_filters : int, default 128
        Number of cochlear (gammatone) channels in the spectrogram stage.
    f_min : float, default 180.0
        Lowest filter center frequency in Hz.
    octaves : float, default 5.3
        Total frequency range in octaves. ``f_max = f_min * 2**octaves``.
    tau_ms : float, default 8.0
        Time constant (ms) for the leaky-integration stage of the
        spectrogram.
    frmlen_ms : float, default 16.0
        Spectrogram frame length in ms (downsampling factor after y5).
    rates : np.ndarray, default ``STANDARD_RATES``
        Base set of temporal modulation rates (Hz). Used as anchor points;
        ``resolution`` may interpolate additional rates between them.
    scales : np.ndarray, default ``STANDARD_SCALES``
        Base set of spectral modulation scales (cycles/octave). Anchor
        points for ``resolution`` interpolation.
    resolution : {"low", "medium", "high", "ultra", "max", "overkill"}, default "low"
        Density multiplier for the rate and scale grids. Each level
        doubles the number of kernels along both axes.
    rsf_frame_size_ms : int, default 500
        RSF integration window length in ms.
    rsf_frame_shift_ms : int, default 10
        RSF hop size in ms.
    """

    # General
    use_gpu: bool = False
    sample_rate: int = 16000

    # Spectrogram
    n_filters: int = 128
    f_min: float = 180.0
    octaves: float = 5.3
    tau_ms: float = 8.0
    frmlen_ms: float = 16.0

    # RSF / Gabor
    rates: np.ndarray = field(default_factory=lambda: STANDARD_RATES.copy())
    scales: np.ndarray = field(default_factory=lambda: STANDARD_SCALES.copy())
    resolution: str = "low"
    rsf_frame_size_ms: int = 500
    rsf_frame_shift_ms: int = 10
