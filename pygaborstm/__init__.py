from .config import Config, SpectrogramConfig, GaborConfig
from .structs import Spectrogram, RSF
from .spectrogram import AuditorySpectrogram, auditory_spectrogram
from .gabor import GaborFilterbank, rsf
from .core import load, compute_rsf
from .plotting import (
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_rsf,
    plot_rsf_grid,
    plot_filterbank,
)
from .utils import (
    generate_tone,
    generate_three_tones,
    generate_broadband_noise,
    generate_harmonic_complex,
    generate_moving_ripple,
    generate_ripple_set,
    save_three_tones,
    save_noise,
    save_harmonic_complexes,
)

__version__ = "0.1.0"

__all__ = [
    # Core functions
    "load",
    "compute_rsf",
    # Functional API
    "auditory_spectrogram",
    "rsf",
    # Classes
    "AuditorySpectrogram",
    "GaborFilterbank",
    # Config
    "Config",
    "SpectrogramConfig",
    "GaborConfig",
    # Data structures
    "Spectrogram",
    "RSF",
    # Plotting
    "plot_spectrogram",
    "plot_spectrogram_grid",
    "plot_rsf",
    "plot_rsf_grid",
    "plot_filterbank",
    # Stimulus generation
    "generate_tone",
    "generate_three_tones",
    "generate_broadband_noise",
    "generate_harmonic_complex",
    "generate_moving_ripple",
    "generate_ripple_set",
    "save_three_tones",
    "save_noise",
    "save_harmonic_complexes",
]
