"""
Backend module for numpy/cupy switching.

Provides a unified interface for array operations that can use either
numpy (CPU) or cupy (GPU) depending on configuration.
"""

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_signal = None
    CUPY_AVAILABLE = False


def get_array_module(use_gpu: bool = False):
    """
    Get the appropriate array module (numpy or cupy).

    Args:
        use_gpu: Whether to use GPU acceleration

    Returns:
        Module (numpy or cupy)
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            import warnings

            warnings.warn(
                "CuPy not available. Falling back to NumPy. "
                "Install cupy-cuda13x for GPU acceleration.",
                UserWarning,
            )
            return np
        return cp
    return np


def get_signal_module(use_gpu: bool = False):
    """
    Get the appropriate signal processing module.

    Args:
        use_gpu: Whether to use GPU acceleration

    Returns:
        Module (scipy.signal or cupyx.scipy.signal)
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            from scipy import signal

            return signal
        return cp_signal
    from scipy import signal

    return signal


def next_fast_len(n: int, use_gpu: bool = False) -> int:
    """
    Find the next FFT-friendly size.
 
    On CPU, uses scipy's optimized lookup (products of small primes).
    On GPU, rounds up to the next power of 2 (optimal for cuFFT).
 
    Args:
        n: Minimum transform size
        use_gpu: Whether GPU backend is active
 
    Returns:
        Optimal FFT length >= n
    """
    if use_gpu:
        return int(2 ** np.ceil(np.log2(n)))
    from scipy.fft import next_fast_len as _scipy_next_fast_len
    return _scipy_next_fast_len(n)


def get_available_memory(use_gpu: bool = False) -> int:
    """
    Get available memory in bytes.
 
    On CPU, returns available system RAM.
    On GPU, returns free VRAM on the current device.
 
    Returns:
        Available memory in bytes
    """
    if use_gpu and CUPY_AVAILABLE:
        free, _ = cp.cuda.Device().mem_info
        return free
 
    # CPU: read from /proc/meminfo (Linux) or fall back to psutil or a safe default
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except (FileNotFoundError, ValueError):
        pass
 
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
 
    # fallback (assume 4GB memory is available) 
    return 4 * 1024**3
 
 
def get_dtypes(use_float32: bool = True):
    """
    Get float and complex dtype pair.
 
    Args:
        use_float32: If True, use float32/complex64. If False, float64/complex128.
 
    Returns:
        (float_dtype, complex_dtype) tuple
    """
    if use_float32:
        return np.float32, np.complex64
    return np.float64, np.complex128


def to_numpy(array) -> np.ndarray:
    """
    Convert array to numpy (transfers from GPU if needed).

    Args:
        array: numpy or cupy array

    Returns:
        numpy array
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
