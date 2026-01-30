from .config import Config, SpectrogramConfig, GaborConfig
from .structs import Spectrogram, RSF
from .spectrogram import AuditorySpectrogram, auditory_spectrogram
from .gabor import GaborFilterbank, rsf
from .core import load, compute_rsf
from .plotting import plot_spectrogram, plot_rsf, plot_filterbank

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
    "plot_rsf",
    "plot_filterbank",
]