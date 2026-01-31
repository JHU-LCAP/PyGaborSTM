import numpy as np

import pygaborstm as stm
from pygaborstm.gabor import GaborFilterbank
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

    def test_resolution_presets(self):
        low = GaborFilterbank(stm.GaborConfig(resolution="low"))
        high = GaborFilterbank(stm.GaborConfig(resolution="high"))
        assert len(low.rates) < len(high.rates)
        assert len(low.scales) < len(high.scales)

    def test_invalid_resolution(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid resolution"):
            GaborFilterbank(stm.GaborConfig(resolution="ultra_mega"))


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
