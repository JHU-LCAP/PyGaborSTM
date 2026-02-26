"""
Configuration for PyGaborSTM.

Default values match Bellur & Elhilali (2017).
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for PyGaborSTM."""

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
    resolution: str = "low"
    rsf_frame_size_ms: int = 500
    rsf_frame_shift_ms: int = 10
