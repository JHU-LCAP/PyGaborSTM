"""
Configuration for PyGaborSTM.

Default values match Bellur & Elhilali (2017).
"""

from dataclasses import dataclass, field


@dataclass
class SpectrogramConfig:
    """Configuration for auditory spectrogram."""

    sample_rate: int = 16000
    n_filters: int = 128
    f_min: float = 180.0
    octaves: float = 5.3
    tau_ms: float = 8.0
    frmlen_ms: float = 16.0
    constant_Q: float = 8.0


@dataclass
class GaborConfig:
    """Configuration for Gabor filterbank."""

    sample_rate: int = 16000
    n_freq_bins: int = 128
    resolution: str = "low"
    rsf_frame_size_ms: int = 500
    rsf_frame_shift_ms: int = 10


@dataclass
class Config:
    """Top-level configuration."""

    use_gpu: bool = False
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    gabor: GaborConfig = field(default_factory=GaborConfig)
