import numpy as np
import pytest

import pygaborstm as stm
from pygaborstm.gabor import GaborFilterbank, DEFAULT_PARAM_IDX
from pygaborstm.structs import RSF


class TestGaborFilterbank:
    def test_default_rates(self):
        model = GaborFilterbank()
        assert len(model.rates) == 10  # 5 positive + 5 negative
        assert model.rates[0] < 0
        assert model.rates[-1] > 0

    def test_default_scales(self):
        model = GaborFilterbank()
        assert len(model.scales) == 6
        assert model.scales[0] < model.scales[-1]

    def test_resolution_presets_grow_kernel_count(self):
        low = GaborFilterbank(stm.Config(resolution="low"))
        high = GaborFilterbank(stm.Config(resolution="high"))
        assert len(low.rates) < len(high.rates)
        assert len(low.scales) < len(high.scales)

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            GaborFilterbank(stm.Config(resolution="ultra_mega"))

    def test_high_kernel_count_warns_at_construction(self):
        """Above _KERNEL_COUNT_WARN_THRESHOLD (5000), the cost is independent of
        memory mode, so we warn before the user spends a long time finding out."""
        with pytest.warns(ResourceWarning, match="produces .* kernels"):
            GaborFilterbank(stm.Config(resolution="max"))


class TestRSFComputation:
    def test_output_type(self, spectrogram_from_tone):
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert isinstance(result, RSF)

    def test_output_shape(self, spectrogram_from_tone):
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert result.n_rates == 10
        assert result.n_scales == 6
        assert result.n_freqs == 128
        assert result.n_frames > 0

    def test_no_nans(self, spectrogram_from_tone):
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert not np.any(np.isnan(result.data))

    def test_non_negative(self, spectrogram_from_tone):
        """Magnitude response should be non-negative."""
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert np.all(result.data >= 0)

    def test_rates_axis(self, spectrogram_from_tone):
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert len(result.rates) == result.n_rates
        np.testing.assert_array_equal(result.rates, model.rates)

    def test_scales_axis(self, spectrogram_from_tone):
        model = GaborFilterbank()
        result = model.compute(spectrogram_from_tone)
        assert len(result.scales) == result.n_scales
        np.testing.assert_array_equal(result.scales, model.scales)

    def test_accepts_raw_spectrogram_array(self, spectrogram_from_tone):
        """compute() accepts either a Spectrogram dataclass or a raw 2D array."""
        model = GaborFilterbank()
        from_struct = model.compute(spectrogram_from_tone)
        from_array = model.compute(spectrogram_from_tone.data)
        np.testing.assert_array_equal(from_struct.data, from_array.data)


class TestCachedVsStreaming:
    """Cached and streaming paths must produce equivalent results.

    The cached path pre-FFTs all Gabor kernels into a single tensor; the
    streaming path rebuilds each chunk inline. Same math, different memory
    profile.
    """

    def test_cached_path_used_by_default(self, spectrogram_from_tone):
        model = GaborFilterbank()
        model.compute(spectrogram_from_tone)
        assert model._kernel_ffts is not None  # cache was populated

    def test_streaming_path_taken_when_budget_exceeded(self, spectrogram_from_tone):
        """Setting the cache budget to 0 forces the streaming fallback."""
        model = GaborFilterbank()
        model._CACHE_MEMORY_BUDGET = 0  # instance-level override of class attr
        with pytest.warns(ResourceWarning, match="streaming mode"):
            model.compute(spectrogram_from_tone)
        assert model._kernel_ffts is None  # cache skipped

    def test_streaming_matches_cached(self, spectrogram_from_tone):
        cached = GaborFilterbank()
        streaming = GaborFilterbank()
        streaming._CACHE_MEMORY_BUDGET = 0

        rsf_cached = cached.compute(spectrogram_from_tone)
        with pytest.warns(ResourceWarning):
            rsf_streaming = streaming.compute(spectrogram_from_tone)

        np.testing.assert_allclose(
            rsf_cached.data,
            rsf_streaming.data,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_ga_params_take_streaming_path(self, spectrogram_from_tone):
        """When per-kernel params are supplied (GA tuning), caching has no
        benefit since kernels change per call — must use streaming."""
        model = GaborFilterbank()
        params = np.full((model._n_kernels, 4), DEFAULT_PARAM_IDX, dtype=np.int32)
        model.compute(spectrogram_from_tone, params=params)
        assert model._kernel_ffts is None  # cache never populated

    def test_ga_default_params_match_cached_path(self, spectrogram_from_tone):
        """Passing the default param indices through GA mode should match
        the cached default-params run."""
        cached = GaborFilterbank()
        ga = GaborFilterbank()
        default_params = np.full(
            (cached._n_kernels, 4),
            DEFAULT_PARAM_IDX,
            dtype=np.int32,
        )

        rsf_cached = cached.compute(spectrogram_from_tone)
        rsf_ga = ga.compute(spectrogram_from_tone, params=default_params)
        np.testing.assert_allclose(
            rsf_cached.data,
            rsf_ga.data,
            rtol=1e-5,
            atol=1e-7,
        )


class TestShapeCacheReuse:
    """Shape-dependent caches (meshgrid, FFT pad, frame indices) get rebuilt
    only when the input shape changes."""

    def test_cache_built_on_first_call(self, spectrogram_from_tone):
        model = GaborFilterbank()
        assert model._cached_shape is None
        model.compute(spectrogram_from_tone)
        n_freq, n_time = spectrogram_from_tone.data.shape
        assert model._cached_shape == (n_time, n_freq)

    def test_cache_reused_for_same_shape(self, spectrogram_from_tone):
        model = GaborFilterbank()
        model.compute(spectrogram_from_tone)
        meshgrid_t_first = model._T
        kernel_ffts_first = model._kernel_ffts

        model.compute(spectrogram_from_tone)
        assert model._T is meshgrid_t_first
        assert model._kernel_ffts is kernel_ffts_first
