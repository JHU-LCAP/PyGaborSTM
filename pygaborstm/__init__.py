"""PyGaborSTM: spectro-temporal modulation analysis.

Extracts Rate-Scale-Frequency (RSF) representations from audio via a
bio-inspired auditory spectrogram followed by a 2D Gabor filterbank,
following Chi, Ru & Shamma (2005) and Bellur & Elhilali (2017).

Public API
----------
PyGaborSTM
    Main user-facing class. Use :meth:`PyGaborSTM.compute` for the full
    pipeline, or :meth:`PyGaborSTM.spectrogram` and
    :meth:`PyGaborSTM.rsf` for the individual stages.
Config
    Configuration dataclass for all pipeline parameters.
Spectrogram, RSF
    Output dataclasses returned by the corresponding stages.
plot, analysis, structs
    Namespaced submodules.

Examples
--------
>>> import pygaborstm as stm
>>> model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
>>> spec = model.spectrogram(audio)
>>> rsf = model.rsf(spec)
>>> stm.plot.spectrogram(spec)
>>> stm.plot.rsf(rsf)
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
    "analysis",
    "plot",
    "structs",
]
