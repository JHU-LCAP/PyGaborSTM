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
