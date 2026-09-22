"""Helpers for optional, extras-gated dependencies.

Imports nothing beyond the standard library: this module is loaded by
``pygaborstm/__init__.py`` and must stay cheap.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

#: Top-level modules provided by the ``viz`` extra.
VIZ_MODULES = frozenset({"matplotlib", "seaborn"})


def missing_extra_message(module: str, *, extra: str, feature: str) -> str:
    return (
        f"{feature} requires the optional dependency '{module}', which is not "
        f"installed.\n\n"
        f"Install it with:\n"
        f"    pip install 'pygaborstm[{extra}]'\n\n"
        f"The core pipeline (Config, PyGaborSTM, Spectrogram, RSF) works "
        f"without it."
    )


def require(module: str, *, extra: str, feature: str) -> ModuleType:
    """Import ``module``, or raise an ImportError saying how to get it."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            missing_extra_message(module, extra=extra, feature=feature)
        ) from exc
