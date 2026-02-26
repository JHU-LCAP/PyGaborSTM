import numpy as np
import pytest

import pygaborstm as stm
from pygaborstm.structs import RSF, Spectrogram


class TestPyGaborSTM:
    def test_default_config(self):
        model = stm.PyGaborSTM()
        assert model.config.use_gpu is False
        assert model.config.resolution == "low"

    def test_custom_config(self):
        cfg = stm.Config(use_gpu=False, resolution="medium", n_filters=64)
        model = stm.PyGaborSTM(config=cfg)
        assert model.config.resolution == "medium"
        assert model.config.n_filters == 64

    def test_spectrogram(self, audio_tone):
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        assert isinstance(spec, Spectrogram)
        assert spec.n_freqs == 128  # default n_filters

    def test_rsf(self, audio_tone):
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert isinstance(rsf, RSF)
        assert rsf.n_rates == 10  # low resolution: 5 positive + 5 negative
        assert rsf.n_scales == 6

    def test_rsf_shapes(self, audio_tone):
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert rsf.n_freqs == spec.n_freqs

    def test_deterministic(self, audio_tone):
        """Same input should produce same output."""
        model = stm.PyGaborSTM()
        spec1 = model.spectrogram(audio_tone)
        spec2 = model.spectrogram(audio_tone)
        np.testing.assert_array_equal(spec1.data, spec2.data)

        rsf1 = model.rsf(spec1)
        rsf2 = model.rsf(spec2)
        np.testing.assert_array_equal(rsf1.data, rsf2.data)


class TestResolutionPresets:
    def test_low_resolution(self, audio_tone):
        model = stm.PyGaborSTM(config=stm.Config(resolution="low"))
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert rsf.n_rates == 10  # 5 positive + 5 negative
        assert rsf.n_scales == 6

    def test_medium_resolution(self, audio_tone):
        model = stm.PyGaborSTM(config=stm.Config(resolution="medium"))
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert rsf.n_rates == 20  # 10 positive + 10 negative
        assert rsf.n_scales == 12

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            cfg = stm.Config(resolution="invalid")
            model = stm.PyGaborSTM(config=cfg)
            # Need to trigger the error by computing something
            audio = np.random.randn(16000)
            spec = model.spectrogram(audio)
            model.rsf(spec)


class TestEndToEnd:
    def test_full_pipeline(self, audio_tone):
        """Verify full pipeline works end-to-end."""
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)

        assert isinstance(spec, Spectrogram)
        assert isinstance(rsf, RSF)
        assert rsf.n_freqs == spec.n_freqs
        assert len(rsf.rates) == rsf.n_rates
        assert len(rsf.scales) == rsf.n_scales

    def test_custom_config_pipeline(self, audio_tone):
        """Test with custom config values."""
        cfg = stm.Config(n_filters=64, resolution="low")
        model = stm.PyGaborSTM(config=cfg)

        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)

        assert spec.n_freqs == 64
        assert rsf.n_freqs == 64
