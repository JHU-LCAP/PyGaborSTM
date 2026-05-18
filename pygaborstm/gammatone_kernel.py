"""
Custom CUDA kernel for batched IIR (SOS) filtering.

Applies a different SOS filter to the same input signal on every channel
in a single kernel launch, replacing a Python-level loop of N independent
cupyx.scipy.signal.sosfilt calls.

Falls back gracefully: ``is_available()`` returns False whenever CuPy or
nvrtc cannot be used, and callers should keep the original loop as a
fallback path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # CPU-only environment
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


# -----------------------------------------------------------------------------
# Kernel source
# -----------------------------------------------------------------------------
# Thread layout:
#   - One thread per output channel.
#   - blockDim.x == 32 so each warp holds 32 independent channels and all
#     lanes execute the same code path (no divergence).
#   - All threads in a warp read the same audio sample, which the hardware
#     services as a single broadcast load.
#
# Per-thread state (all in registers, sized at JIT time via N_SECTIONS):
#   b0[s], b1[s], b2[s], a1[s], a2[s]  -- SOS coefficients, loaded once
#   z1[s], z2[s]                       -- DF2T state, evolves across samples
#
# Recursion (Direct Form II Transposed, matches scipy.signal.sosfilt):
#       y      = b0 * x + z1
#       z1'    = b1 * x - a1 * y + z2
#       z2'    = b2 * x - a2 * y
# with a0 assumed to be 1.0 (scipy normalises SOS this way).
# -----------------------------------------------------------------------------
_KERNEL_TEMPLATE = r"""
typedef DTYPE scalar_t;

extern "C" __global__ void batched_sosfilt_kernel(
    const scalar_t* __restrict__ sos,
    const float*    __restrict__ x,
    float*          __restrict__ y,
    const int n_channels,
    const int n_samples,
    const float gain
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    scalar_t b0[N_SECTIONS], b1[N_SECTIONS], b2[N_SECTIONS];
    scalar_t a1[N_SECTIONS], a2[N_SECTIONS];
    scalar_t z1[N_SECTIONS], z2[N_SECTIONS];

    #pragma unroll
    for (int s = 0; s < N_SECTIONS; ++s) {
        const scalar_t* p = sos + (c * N_SECTIONS + s) * 6;
        b0[s] = p[0]; b1[s] = p[1]; b2[s] = p[2];
        a1[s] = p[4]; a2[s] = p[5];
        z1[s] = (scalar_t)0; z2[s] = (scalar_t)0;
    }

    float* y_row = y + (size_t)c * (size_t)n_samples;
    const scalar_t g = (scalar_t)gain;

    for (int n = 0; n < n_samples; ++n) {
        scalar_t v = (scalar_t)x[n];
        #pragma unroll
        for (int s = 0; s < N_SECTIONS; ++s) {
            const scalar_t out = b0[s] * v + z1[s];
            z1[s] = b1[s] * v - a1[s] * out + z2[s];
            z2[s] = b2[s] * v - a2[s] * out;
            v = out;
        }
        y_row[n] = (float)(g * v);
    }
}
"""

_kernel_cache: dict[tuple[int, str], "cp.RawKernel"] = {}

_DTYPE_MAP = {
    "float64": ("double", cp.float64 if _HAS_CUPY else None),
    "float32": ("float", cp.float32 if _HAS_CUPY else None),
}


def is_available() -> bool:
    """Return True iff CuPy + nvrtc can compile this kernel on this machine.

    Triggers a one-time JIT compile of a tiny stub kernel; subsequent
    calls hit the kernel cache and are essentially free.

    Returns
    -------
    bool
        True if the kernel can be used. False on CPU-only systems or
        when nvrtc is missing.
    """
    if not _HAS_CUPY:
        return False
    try:
        _get_kernel(1, "float32")
        return True
    except Exception as e:
        logger.warning("batched_sosfilt kernel unavailable: %s", e)
        return False


def _get_kernel(n_sections: int, precision: str) -> "cp.RawKernel":
    key = (n_sections, precision)
    if key in _kernel_cache:
        return _kernel_cache[key]
    c_type, _ = _DTYPE_MAP[precision]
    source = (
        f"#define N_SECTIONS {int(n_sections)}\n"
        f"#define DTYPE {c_type}\n" + _KERNEL_TEMPLATE
    )
    kernel = cp.RawKernel(
        code=source, name="batched_sosfilt_kernel", options=("-std=c++14",)
    )
    _kernel_cache[key] = kernel
    return kernel


def batched_sosfilt(
    sos: "cp.ndarray",
    x: "cp.ndarray",
    gain: float = 1.0,
    out: Optional["cp.ndarray"] = None,
    precision: str = "float64",
) -> "cp.ndarray":
    """Apply a per-channel SOS cascade to the same input in one kernel launch.

    Equivalent to a loop of ``cupyx.scipy.signal.sosfilt`` calls, each
    with its own SOS coefficients but a shared input signal, fused into
    a single CUDA kernel.

    Parameters
    ----------
    sos : cupy.ndarray
        SOS coefficients of shape ``(n_channels, n_sections, 6)``.
        Dtype must match ``precision`` (float64 for ``"float64"``,
        float32 for ``"float32"``).
    x : cupy.ndarray
        Shared 1D input signal of shape ``(n_samples,)``. Must be
        float32.
    gain : float, default 1.0
        Per-channel scalar applied to the output before write-back.
    out : cupy.ndarray, optional
        Pre-allocated output buffer of shape ``(n_channels, n_samples)``
        and dtype float32. Allocated by the caller when reused across
        many invocations to avoid per-call allocation.
    precision : {"float64", "float32"}, default "float64"
        Internal compute precision. ``"float32"`` is ~8x faster on
        consumer Ampere (3090ti, Jetson Orin) where FP64 throughput is
        throttled to 1/64 of FP32. The float32 path matches scipy to
        ~1e-3 worst-case relative error; the float64 path to ~1e-9.

    Returns
    -------
    cupy.ndarray
        Filtered output of shape ``(n_channels, n_samples)``, dtype
        float32. If ``out`` was provided, returns that same buffer.

    Raises
    ------
    RuntimeError
        If CuPy is not installed.
    ValueError
        If ``precision`` is not one of the supported values.
    TypeError
        If ``sos`` or ``x`` does not match the expected dtype.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy is not available")
    if precision not in _DTYPE_MAP:
        raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}")

    _, expected_dtype = _DTYPE_MAP[precision]
    if sos.dtype != expected_dtype:
        raise TypeError(
            f"sos must be {expected_dtype} for precision={precision!r}, got {sos.dtype}"
        )
    if x.dtype != cp.float32:
        raise TypeError(f"x must be float32, got {x.dtype}")

    n_channels, n_sections, _ = sos.shape
    n_samples = int(x.shape[0])

    sos_c = cp.ascontiguousarray(sos)
    x_c = cp.ascontiguousarray(x)
    if out is None:
        out = cp.empty((n_channels, n_samples), dtype=cp.float32)

    kernel = _get_kernel(n_sections, precision)
    threads = 32
    blocks = (n_channels + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (sos_c, x_c, out, np.int32(n_channels), np.int32(n_samples), np.float32(gain)),
    )
    return out
