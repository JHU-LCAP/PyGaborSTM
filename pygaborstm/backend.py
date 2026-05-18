"""NumPy/CuPy backend selection and array-module helpers.

A thin wrapper around the array-API differences between NumPy and CuPy
so the rest of the package can be written once and run on either CPU
or GPU. Also exposes a few low-level helpers (memory probe, FFT-size
rounding, dtype pairs) that are shared between the spectrogram and
Gabor stages.
"""

import warnings
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


def _gpu_available(use_gpu: bool) -> bool:
    """Return True iff GPU was requested and CuPy is importable.

    Warns via :class:`UserWarning` when GPU is requested but unavailable
    so callers know they have silently fallen back to NumPy.
    """
    if not use_gpu:
        return False
    if CUPY_AVAILABLE:
        return True
    warnings.warn(
        "CuPy not available. Falling back to NumPy. "
        "Install cupy-cuda13x for GPU acceleration.",
        UserWarning,
    )
    return False


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
    if use_gpu and CUPY_AVAILABLE:
        free, _ = cp.cuda.Device().mem_info
        return free

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
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
