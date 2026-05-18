"""
Backend module for numpy/cupy switching.

Provides a unified interface for array operations that can use either
numpy (CPU) or cupy (GPU) depending on configuration.
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
    """Check if GPU was requested and is available, warn if not."""
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
    """Get numpy or cupy depending on config."""
    return cp if _gpu_available(use_gpu) else np


def get_signal_module(use_gpu: bool = False):
    """Get scipy.signal or cupyx.scipy.signal depending on config."""
    return cp_signal if _gpu_available(use_gpu) else sp_signal


def next_fast_len(n: int, use_gpu: bool = False) -> int:
    """
    Find the next FFT-friendly size.

    CPU: product of small primes (scipy). GPU: next power of 2 (cuFFT).
    """
    if use_gpu:
        return int(2 ** np.ceil(np.log2(n)))
    return _scipy_next_fast_len(n)


def get_available_memory(use_gpu: bool = False) -> int:
    """
    Get available memory in bytes.

    GPU: free VRAM. CPU: /proc/meminfo, then psutil, then 4 GB fallback.
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
    """Get (float_dtype, complex_dtype) pair."""
    if use_float32:
        return np.float32, np.complex64
    return np.float64, np.complex128


def to_numpy(array) -> np.ndarray:
    """Convert array to numpy (transfers from GPU if needed)."""
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
