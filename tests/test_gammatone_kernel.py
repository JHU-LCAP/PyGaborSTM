"""Verify batched_sosfilt CUDA kernel matches scipy.signal.sosfilt.

Tests both precision modes against a float64 scipy reference:
  - float64 kernel: tight regression check (~1e-9 expected)
  - float32 kernel: production fast path; ~1e-4 typical, worst channels
    (lowest freqs, narrowest bandwidth) may reach ~1e-3

These tests require CuPy + a working CUDA toolchain. Skipped on
CPU-only machines and in the default CI run (use `pytest -m gpu`).
"""

import numpy as np
import pytest
import scipy.signal

cp = pytest.importorskip("cupy")

from pygaborstm.gammatone_kernel import batched_sosfilt, is_available  # noqa: E402

pytestmark = pytest.mark.gpu


def _build_bank(n_channels=128, n_sections=4, fs=16_000):
    """Gammatone-shaped bandpass bank for testing."""
    centers = np.geomspace(160.0, 5000.0, n_channels)
    return np.stack(
        [
            scipy.signal.butter(
                2 * n_sections,
                [max(20.0, fc * 0.7), min(fs / 2 - 100, fc * 1.3)],
                btype="band",
                fs=fs,
                output="sos",
            )
            for fc in centers
        ]
    ).astype(np.float64)


@pytest.fixture(scope="module")
def reference():
    """scipy.signal.sosfilt reference + matching input + SOS bank."""
    if not is_available():
        pytest.skip("CUDA toolchain not available")

    n_samples = 80_000
    gain = 2.0
    rng = np.random.default_rng(0)

    sos = _build_bank()
    x = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    x64 = x.astype(np.float64)
    ref = (
        gain
        * np.stack([scipy.signal.sosfilt(sos[c], x64) for c in range(sos.shape[0])])
    ).astype(np.float32)

    return {"sos": sos, "x": x, "gain": gain, "ref": ref}


def _max_rel_error(out, ref):
    abs_err = np.abs(out - ref)
    per_ch = abs_err.max(axis=1) / (np.abs(ref).max(axis=1) + 1e-30)
    return per_ch.max()


class TestBatchedSosfilt:
    def test_float64_matches_scipy(self, reference):
        out = cp.asnumpy(
            batched_sosfilt(
                cp.asarray(reference["sos"]),
                cp.asarray(reference["x"]),
                gain=reference["gain"],
                precision="float64",
            )
        )
        assert _max_rel_error(out, reference["ref"]) < 1e-5

    def test_float32_matches_scipy_within_tolerance(self, reference):
        out = cp.asnumpy(
            batched_sosfilt(
                cp.asarray(reference["sos"].astype(np.float32)),
                cp.asarray(reference["x"]),
                gain=reference["gain"],
                precision="float32",
            )
        )
        assert _max_rel_error(out, reference["ref"]) < 1e-3

    def test_rejects_wrong_sos_dtype(self, reference):
        sos_f32 = cp.asarray(reference["sos"].astype(np.float32))
        x = cp.asarray(reference["x"])
        with pytest.raises(TypeError, match="sos must be"):
            batched_sosfilt(sos_f32, x, precision="float64")

    def test_rejects_wrong_input_dtype(self, reference):
        sos = cp.asarray(reference["sos"])
        x_f64 = cp.asarray(reference["x"].astype(np.float64))
        with pytest.raises(TypeError, match="x must be float32"):
            batched_sosfilt(sos, x_f64, precision="float64")

    def test_rejects_invalid_precision(self, reference):
        sos = cp.asarray(reference["sos"])
        x = cp.asarray(reference["x"])
        with pytest.raises(ValueError, match="precision must be"):
            batched_sosfilt(sos, x, precision="bfloat16")
