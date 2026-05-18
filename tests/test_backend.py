import numpy as np
import pytest

from pygaborstm import backend


class TestArrayModule:
    def test_cpu_returns_numpy(self):
        assert backend.get_array_module(use_gpu=False) is np

    def test_signal_module_cpu_is_scipy(self):
        from scipy import signal as sp_signal

        assert backend.get_signal_module(use_gpu=False) is sp_signal

    def test_gpu_request_without_cupy_warns_and_falls_back(self, monkeypatch):
        monkeypatch.setattr(backend, "CUPY_AVAILABLE", False)
        with pytest.warns(UserWarning, match="CuPy not available"):
            xp = backend.get_array_module(use_gpu=True)
        assert xp is np


class TestDtypes:
    def test_float32_default(self):
        f, c = backend.get_dtypes(use_float32=True)
        assert f is np.float32
        assert c is np.complex64

    def test_float64(self):
        f, c = backend.get_dtypes(use_float32=False)
        assert f is np.float64
        assert c is np.complex128


class TestNextFastLen:
    def test_cpu_returns_at_least_n(self):
        assert backend.next_fast_len(100, use_gpu=False) >= 100

    def test_gpu_returns_power_of_two(self):
        # GPU path uses next power of 2 (cuFFT-friendly); doesn't require cupy.
        n = backend.next_fast_len(1000, use_gpu=True)
        assert n >= 1000
        assert (n & (n - 1)) == 0  # power of 2

    def test_exact_power_of_two_returned_unchanged_on_gpu(self):
        assert backend.next_fast_len(1024, use_gpu=True) == 1024


class TestAvailableMemory:
    def test_cpu_returns_positive_int(self):
        mem = backend.get_available_memory(use_gpu=False)
        assert isinstance(mem, int)
        assert mem > 0


class TestToNumpy:
    def test_numpy_passthrough(self):
        arr = np.arange(10)
        result = backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
