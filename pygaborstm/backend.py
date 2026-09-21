"""NumPy/CuPy backend selection and array-module helpers.

A thin wrapper around the array-API differences between NumPy and CuPy
so the rest of the package can be written once and run on either CPU
or GPU. Also exposes a few low-level helpers (memory probe, FFT-size
rounding, dtype pairs) that are shared between the spectrogram and
Gabor stages.
"""

import warnings
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_signal = None
    CUPY_AVAILABLE = False

from scipy import signal as sp_signal
from scipy.fft import next_fast_len as _scipy_next_fast_len

_DEVICE_PROBE: bool | None = None
_DEVICE_PROBE_ERROR = ""


def _reset_device_probe() -> None:
    """Clear the cached CUDA device probe. For tests."""
    global _DEVICE_PROBE, _DEVICE_PROBE_ERROR
    _DEVICE_PROBE = None
    _DEVICE_PROBE_ERROR = ""


def _cuda_device_present() -> bool:
    """Return True iff the CUDA driver reports a usable device.

    Importing CuPy proves nothing: the wheels install on any Linux or
    Windows machine, and the failure only surfaces on the first device
    touch. Probing here moves the fallback to construction time.
    """
    global _DEVICE_PROBE, _DEVICE_PROBE_ERROR
    if _DEVICE_PROBE is not None:
        return _DEVICE_PROBE
    try:
        _DEVICE_PROBE = cp.cuda.runtime.getDeviceCount() > 0
        if not _DEVICE_PROBE:
            _DEVICE_PROBE_ERROR = "no CUDA devices reported"
    except Exception as exc:
        _DEVICE_PROBE_ERROR = f"{type(exc).__name__}: {exc}"
        _DEVICE_PROBE = False
    return _DEVICE_PROBE


def _gpu_available(use_gpu: bool) -> bool:
    """Return True iff GPU was requested and actually usable.

    Warns via :class:`UserWarning` on every fallback path so callers know
    they have silently dropped to NumPy.
    """
    if not use_gpu:
        return False
    if not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Falling back to NumPy. "
            "Install with: pip install 'pygaborstm[cuda13]' (CUDA 13.x) "
            "or pip install 'pygaborstm[cuda12]' (CUDA 12.x).",
            UserWarning,
            stacklevel=2,
        )
        return False
    if not _cuda_device_present():
        warnings.warn(
            f"CuPy is installed but no usable CUDA device was found "
            f"({_DEVICE_PROBE_ERROR}). Falling back to NumPy.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


def get_array_module(use_gpu: bool = False):
    """Return the active array module (``numpy`` or ``cupy``).

    Parameters
    ----------
    use_gpu : bool, default False
        If True and CuPy is available, return ``cupy``. Otherwise return
        ``numpy``.

    Returns
    -------
    module
        Either the ``numpy`` or ``cupy`` module object.
    """
    return cp if _gpu_available(use_gpu) else np


def get_signal_module(use_gpu: bool = False):
    """Return the active signal-processing module.

    Parameters
    ----------
    use_gpu : bool, default False
        If True and CuPy is available, return ``cupyx.scipy.signal``.
        Otherwise return ``scipy.signal``.

    Returns
    -------
    module
        Either ``scipy.signal`` or ``cupyx.scipy.signal``.
    """
    return cp_signal if _gpu_available(use_gpu) else sp_signal


def next_fast_len(n: int, use_gpu: bool = False) -> int:
    """Round ``n`` up to an FFT-friendly length.

    CPU uses :func:`scipy.fft.next_fast_len` (product of small primes).
    GPU uses the next power of two, which matches cuFFT's optimal path.

    Parameters
    ----------
    n : int
        Minimum length required.
    use_gpu : bool, default False
        Switches between the CPU and GPU rounding rules.

    Returns
    -------
    int
        The smallest fast length ``>= n``.
    """
    if use_gpu:
        return int(2 ** np.ceil(np.log2(n)))
    return _scipy_next_fast_len(n)


def get_available_memory(use_gpu: bool = False) -> int:
    """Best-effort estimate of free memory in bytes.

    On GPU, reports free VRAM via the current CUDA device. On CPU,
    tries ``/proc/meminfo`` first, then :mod:`psutil`, then falls back
    to a hard-coded 4 GiB so callers always get a usable number.

    Parameters
    ----------
    use_gpu : bool, default False
        If True, report device memory rather than host memory.

    Returns
    -------
    int
        Available memory in bytes.
    """
    if use_gpu and CUPY_AVAILABLE and _cuda_device_present():
        try:
            free, _ = cp.cuda.Device().mem_info
            return int(free)
        except Exception:
            pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass

    return 4 * 1024**3


def get_dtypes(use_float32: bool = True):
    """Return a matching ``(float, complex)`` dtype pair.

    Parameters
    ----------
    use_float32 : bool, default True
        If True, return ``(np.float32, np.complex64)``; otherwise the
        double-precision pair.

    Returns
    -------
    tuple of numpy.dtype
        ``(float_dtype, complex_dtype)``.
    """
    if use_float32:
        return np.float32, np.complex64
    return np.float64, np.complex128


def to_numpy(array) -> np.ndarray:
    """Return ``array`` as a host-side NumPy array, copying from GPU if needed.

    Parameters
    ----------
    array : array_like
        Input array. May be a numpy array, cupy array, or any
        array-like accepted by :func:`numpy.asarray`.

    Returns
    -------
    np.ndarray
        Host-side numpy array.
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


@dataclass(frozen=True)
class Device:
    """The backend actually in use, as opposed to the one requested.

    Built only by :func:`resolve_device`. Callers must not re-derive
    GPU-ness from ``Config.use_gpu``: that flag is the request, this is
    the answer.

    Stores a flag rather than module objects so that models holding a
    ``Device`` stay picklable.
    """

    on_gpu: bool

    @property
    def xp(self):
        """The active array module, ``numpy`` or ``cupy``."""
        return cp if self.on_gpu else np

    @property
    def signal(self):
        """The active signal module, ``scipy.signal`` or ``cupyx.scipy.signal``."""
        return cp_signal if self.on_gpu else sp_signal

    def synchronize(self) -> None:
        """Block until queued device work completes. No-op on CPU."""
        if self.on_gpu:
            cp.cuda.Stream.null.synchronize()

    def next_fast_len(self, n: int) -> int:
        """FFT-friendly length using this device's rounding rule."""
        return next_fast_len(n, self.on_gpu)

    def available_memory(self) -> int:
        """Free memory in bytes: VRAM on GPU, host RAM on CPU."""
        return get_available_memory(self.on_gpu)


def resolve_device(use_gpu: bool = False) -> Device:
    """Resolve a requested backend into the one actually in use.

    Emits at most one :class:`UserWarning` per call when the request
    cannot be honoured.
    """
    return Device(on_gpu=_gpu_available(use_gpu))
