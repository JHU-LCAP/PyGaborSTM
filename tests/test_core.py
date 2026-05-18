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
        assert spec.n_freqs == 128

    def test_rsf(self, audio_tone):
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert isinstance(rsf, RSF)
        assert rsf.n_rates == 10
        assert rsf.n_scales == 6

    def test_rsf_shapes(self, audio_tone):
        model = stm.PyGaborSTM()
        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)
        assert rsf.n_freqs == spec.n_freqs

    def test_deterministic(self, audio_tone):
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
        rsf = model.rsf(model.spectrogram(audio_tone))
        assert rsf.n_rates == 10
        assert rsf.n_scales == 6

    def test_medium_resolution(self, audio_tone):
        model = stm.PyGaborSTM(config=stm.Config(resolution="medium"))
        rsf = model.rsf(model.spectrogram(audio_tone))
        assert rsf.n_rates == 20
        assert rsf.n_scales == 12

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            stm.PyGaborSTM(config=stm.Config(resolution="invalid"))


class TestFullPipeline:
    def test_compute_returns_rsf(self, audio_tone):
        """compute() chains spectrogram + gabor stages on device."""
        model = stm.PyGaborSTM()
        rsf = model.compute(audio_tone)
        assert isinstance(rsf, RSF)
        assert rsf.n_freqs == 128
        assert rsf.n_rates == 10
        assert rsf.n_scales == 6

    def test_compute_matches_two_stage_call(self, audio_tone):
        """Full-pipeline compute() must match the staged spectrogram() + rsf()."""
        model = stm.PyGaborSTM()
        rsf_chained = model.compute(audio_tone)
        rsf_staged = model.rsf(model.spectrogram(audio_tone))
        np.testing.assert_allclose(
            rsf_chained.data,
            rsf_staged.data,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_compute_device_returns_array(self, audio_tone):
        """compute_device() returns a raw (n_frames, n_rates, n_scales, n_freq)
        array, not an RSF dataclass — caller skips the host transfer."""
        model = stm.PyGaborSTM()
        out = model.compute_device(audio_tone)
        assert isinstance(out, np.ndarray)  # CPU mode → numpy
        assert out.shape == (out.shape[0], 10, 6, 128)

    def test_compute_device_matches_compute(self, audio_tone):
        model = stm.PyGaborSTM()
        device_out = model.compute_device(audio_tone)
        rsf = model.compute(audio_tone)
        np.testing.assert_array_equal(device_out, rsf.data)

    def test_custom_n_filters_pipeline(self, audio_tone):
        cfg = stm.Config(n_filters=64, resolution="low")
        model = stm.PyGaborSTM(config=cfg)

        spec = model.spectrogram(audio_tone)
        rsf = model.rsf(spec)

        assert spec.n_freqs == 64
        assert rsf.n_freqs == 64
