"""
PyGaborSTM: Spectro-temporal modulation analysis library.

Usage:
    import pygaborstm as stm

    # Create model
    model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))

    # Compute
    spec = model.spectrogram(audio)
    rsf = model.rsf(spec)

    # Plot
    stm.plot.spectrogram(spec)
    stm.plot.rsf(rsf)
"""

from .config import Config
from .structs import Spectrogram, RSF
from .core import PyGaborSTM
from . import analysis
from . import plot
from . import structs

__version__ = "0.1.0"

__all__ = [
    # Main class
    "PyGaborSTM",
    # Config
    "Config",
    # Data structures
    "Spectrogram",
    "RSF",
    # Namespaced modules
    "plot",
    "structs",
]
