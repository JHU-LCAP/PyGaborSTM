"""Post-install smoke test for a published pygaborstm distribution.

Run against an *installed* distribution, from a directory that is not the
repository root::

    cd "$(mktemp -d)"
    python /path/to/PyGaborSTM/scripts/smoke_install.py

Checks, in order:

1. pygaborstm was imported from an installed location, not a checkout. This is the
   check the script exists for: run from the repository root, ``import
   pygaborstm`` resolves to ``./pygaborstm/`` and everything below passes even
   against an empty wheel.
2. The installed version agrees with ``__version__`` and, when set, with
   ``EXPECT_VERSION``.
3. Importing the package does not import matplotlib or seaborn.
4. The CPU pipeline runs and produces a finite, self-consistent RSF.
5. Requesting a GPU with no CuPy present warns and still computes.

Exits non-zero on the first failure.
"""

from __future__ import annotations

import os
import sys
import warnings
from importlib.metadata import version as installed_version
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
TONE_HZ = 440.0


def _fail(message: str) -> None:
    raise SystemExit(f"smoke_install: FAIL: {message}")


def _ok(message: str) -> None:
    print(f"smoke_install: ok: {message}")


def _tone() -> np.ndarray:
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * TONE_HZ * t)


def main() -> None:
    import pygaborstm as stm

    modules_after_import = set(sys.modules)

    module_file = Path(stm.__file__).resolve()
    installed_dirs = {"site-packages", "dist-packages"}
    if not (installed_dirs & set(module_file.parts)):
        _fail(
            f"pygaborstm was imported from {module_file}, which is not an "
            f"installed distribution. Run this from a directory that is not the "
            f"repository root: from the root, Python picks up ./pygaborstm/ and "
            f"every check below would pass against an empty wheel."
        )
    _ok(f"imported the installed distribution from {module_file.parent}")

    metadata_version = installed_version("pygaborstm")
    if stm.__version__ != metadata_version:
        _fail(
            f"__version__ is {stm.__version__!r} but the installed distribution "
            f"reports {metadata_version!r}"
        )
    expected = os.environ.get("EXPECT_VERSION")
    if expected and metadata_version != expected:
        _fail(f"EXPECT_VERSION is {expected!r} but installed is {metadata_version!r}")
    _ok(f"version {metadata_version}")

    leaked = sorted(
        m for m in modules_after_import if m.split(".")[0] in {"matplotlib", "seaborn"}
    )
    if leaked:
        _fail(f"import pygaborstm eagerly imported the plotting stack: {leaked}")
    _ok("import pygaborstm did not import matplotlib or seaborn")

    model = stm.PyGaborSTM(config=stm.Config(sample_rate=SAMPLE_RATE))
    if model.device.on_gpu:
        _fail("a default Config resolved to a GPU device")

    rsf = model.compute(_tone())
    data = rsf.to_numpy()

    axes = (data.shape[0], len(rsf.rates), len(rsf.scales), len(rsf.freqs))
    if data.ndim != 4 or data.shape != axes:
        _fail(f"RSF shape {data.shape} disagrees with its own axes {axes}")
    if rsf.times.shape != (data.shape[0],):
        _fail(f"time axis {rsf.times.shape} does not match {data.shape[0]} frames")
    if not np.isfinite(data).all():
        _fail("RSF contains NaN or inf")
    if float(np.ptp(data)) <= 0.0:
        _fail("RSF is constant - the pipeline ran but produced no structure")
    _ok(f"CPU pipeline produced a finite RSF of shape {data.shape}")

    # The warning comes from resolve_device during model construction, not from
    # Config, which is a plain dataclass. Wrapping Config would assert on
    # something that never happens.
    if stm.backend.CUPY_AVAILABLE:
        _ok("CuPy is installed; skipping the no-GPU fallback check")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gpu_model = stm.PyGaborSTM(
                config=stm.Config(sample_rate=SAMPLE_RATE, use_gpu=True)
            )
        messages = [str(w.message) for w in caught]
        if not any("Falling back to NumPy" in m for m in messages):
            _fail(f"use_gpu=True did not warn about the fallback; saw {messages}")
        if gpu_model.device.on_gpu:
            _fail("device reports on_gpu=True with no CUDA device present")
        if gpu_model.compute(_tone()).to_numpy().shape != data.shape:
            _fail("the NumPy fallback produced a different shape from the CPU path")
        _ok("use_gpu=True warned, fell back to NumPy, and still computed")

    print("smoke_install: PASS")


if __name__ == "__main__":
    main()
