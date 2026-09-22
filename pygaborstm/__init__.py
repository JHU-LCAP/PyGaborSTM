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


def build_info() -> dict[str, str | None]:
    """Where this installation came from.

    Between releases the version alone does not identify a build, since every
    commit reports the same ``X.Y.Z.dev0``. This reads the record pip writes at
    install time (PEP 610 ``direct_url.json``) to recover the exact commit.

    Returns
    -------
    dict
        ``version`` always. ``source`` is one of ``"git"``, ``"vcs"``,
        ``"archive"``, ``"local"``, ``"editable"``, ``"index"`` or
        ``"unknown"``. ``commit`` is the resolved sha for a git install and
        None otherwise; ``url`` is the origin when one was recorded.

    Examples
    --------
    >>> import pygaborstm as stm
    >>> stm.build_info()["commit"]
    'a1b2c3d4e5f6...'

    Notes
    -----
    A git install records the commit pip actually resolved, so pinning a branch
    still yields a concrete sha. Released installs from an index carry no such
    record: there the version is the identifier.

    This never raises. Anything it cannot establish is reported as
    ``"unknown"`` rather than guessed, because a confident wrong answer about
    provenance is worse than admitting ignorance.
    """
    info: dict[str, str | None] = {
        "version": __version__,
        "source": "unknown",
        "commit": None,
        "url": None,
    }
    try:
        return _provenance(info)
    except Exception:  # pragma: no cover - diagnostics must never break import
        return info


def _record_describes_this_import(dist) -> bool:
    """True if ``dist``'s files are the ones that actually got imported."""
    from pathlib import Path

    try:
        located = dist.locate_file("pygaborstm/__init__.py")
        if located is None:
            return False
        return Path(located).resolve() == Path(__file__).resolve()
    except Exception:
        return False


def _provenance(info: dict[str, str | None]) -> dict[str, str | None]:
    """Fill in ``info`` from the installed distribution's PEP 610 record."""
    import json
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("pygaborstm")
    except (PackageNotFoundError, ValueError):
        # A source tree that was never installed. Not an index install.
        return info

    raw = dist.read_text("direct_url.json")
    if raw is None:
        # Installed, but with no direct URL recorded: came from an index.
        info["source"] = "index"
        return info

    try:
        record = json.loads(raw)
    except ValueError:
        return info
    if not isinstance(record, dict):
        return info

    url = record.get("url")
    info["url"] = url if isinstance(url, str) else None

    vcs_info = record.get("vcs_info")
    dir_info = record.get("dir_info")
    if isinstance(vcs_info, dict):
        # PEP 610 allows hg, bzr and svn too; only git carries a sha we can
        # hand back as a commit.
        if vcs_info.get("vcs") == "git":
            commit = vcs_info.get("commit_id")
            info["source"] = "git"
            info["commit"] = commit if isinstance(commit, str) else None
            if not _record_describes_this_import(dist):
                # Metadata lookup returns the first match on sys.path, which
                # need not be the copy that got imported. A commit from some
                # other installation is worse than no commit at all.
                info["source"] = "unknown"
                info["commit"] = None
        else:
            info["source"] = "vcs"
    elif isinstance(dir_info, dict):
        info["source"] = "editable" if dir_info.get("editable") else "local"
    elif isinstance(record.get("archive_info"), dict):
        info["source"] = "archive"
    return info


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
    "build_info",
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
