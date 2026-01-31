from pygaborstm.config import SpectrogramConfig, GaborConfig, Config


class TestSpectrogramConfig:
    def test_defaults_match_paper(self):
        cfg = SpectrogramConfig()
        assert cfg.sample_rate == 16000
        assert cfg.n_filters == 128
        assert cfg.f_min == 180.0
        assert cfg.octaves == 5.3
        assert cfg.tau_ms == 8.0
        assert cfg.frmlen_ms == 16.0

    def test_custom_values(self):
        cfg = SpectrogramConfig(n_filters=64, sample_rate=8000)
        assert cfg.n_filters == 64
        assert cfg.sample_rate == 8000


class TestGaborConfig:
    def test_defaults_match_paper(self):
        cfg = GaborConfig()
        assert cfg.sample_rate == 16000
        assert cfg.n_freq_bins == 128
        assert cfg.resolution == "low"
        assert cfg.rsf_frame_size_ms == 500
        assert cfg.rsf_frame_shift_ms == 10

    def test_custom_values(self):
        cfg = GaborConfig(resolution="high", rsf_frame_size_ms=250)
        assert cfg.resolution == "high"
        assert cfg.rsf_frame_size_ms == 250


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.use_gpu is False
        assert isinstance(cfg.spectrogram, SpectrogramConfig)
        assert isinstance(cfg.gabor, GaborConfig)

    def test_nested_override(self):
        cfg = Config(
            spectrogram=SpectrogramConfig(n_filters=64),
            gabor=GaborConfig(resolution="high"),
        )
        assert cfg.spectrogram.n_filters == 64
        assert cfg.gabor.resolution == "high"
