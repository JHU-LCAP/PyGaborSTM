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
structs
    Namespaced submodule of the output dataclasses.
plot, analysis
    Namespaced submodules, imported on first attribute access. They need
    the plotting extra::

        pip install 'pygaborstm[viz]'

    Importing :mod:`pygaborstm` itself never imports matplotlib.

Examples
--------
>>> import pygaborstm as stm
>>> model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
>>> spec = model.spectrogram(audio)
>>> rsf = model.rsf(spec)
>>> stm.plot.plt_spectrogram(spec)   # needs pygaborstm[viz]
>>> stm.plot.plt_rsf(rsf)
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from types import ModuleType
from typing import TYPE_CHECKING

from . import structs
from ._optional import VIZ_MODULES, missing_extra_message
from .config import Config
from .core import PyGaborSTM
from .structs import RSF, Spectrogram

# Resolved by type checkers and IDEs without importing matplotlib at runtime.
if TYPE_CHECKING:
    # Redundant aliases mark these as re-exports; they are public but kept out
    # of __all__ so star-import does not require the viz extra.
    from . import analysis as analysis
    from . import plot as plot

try:
    __version__ = _dist_version("pygaborstm")
except PackageNotFoundError:  # source tree that was never installed
    __version__ = "0.0.0.dev0"

#: Submodules imported on first attribute access (PEP 562), so that
#: `import pygaborstm` costs numpy and scipy and nothing else.
_LAZY_SUBMODULES = frozenset({"plot", "analysis"})

__all__ = [  # noqa: RUF022 - grouped by meaning, not alphabetised
    # Main class
    "PyGaborSTM",
    # Config
    "Config",
    # Data structures
    "Spectrogram",
    "RSF",
    # Namespaced modules. plot and analysis are deliberately absent: they
    # are reachable as attributes, but listing them here would make
    # `from pygaborstm import *` require the viz extra.
    "structs",
    # Metadata
    "__version__",
]


def __getattr__(name: str) -> ModuleType:
    if name not in _LAZY_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f".{name}", __name__)
    except ImportError as exc:
        # Only translate a missing optional dependency. A genuine ImportError
        # from inside plot.py or analysis.py must surface unchanged.
        missing = (exc.name or "").split(".")[0]
        if missing not in VIZ_MODULES:
            raise
        raise ImportError(
            missing_extra_message(missing, extra="viz", feature=f"pygaborstm.{name}")
        ) from exc
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    # _LAZY_SUBMODULES explicitly: they are public and should be discoverable
    # before they have been touched, even though they are not in __all__.
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)
