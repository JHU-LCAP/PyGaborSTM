from pygaborstm.config import Config


class TestConfig:
    def test_defaults_match_paper(self):
        cfg = Config()

        # General
        assert cfg.use_gpu is False
        assert cfg.sample_rate == 16000

        # Spectrogram
        assert cfg.n_filters == 128
        assert cfg.f_min == 180.0
        assert cfg.octaves == 5.3
        assert cfg.tau_ms == 8.0
        assert cfg.frmlen_ms == 16.0

        # RSF / Gabor
        assert cfg.resolution == "low"
        assert cfg.rsf_frame_size_ms == 500
        assert cfg.rsf_frame_shift_ms == 10

    def test_custom_values(self):
        cfg = Config(
            use_gpu=True,
            sample_rate=8000,
            n_filters=64,
            resolution="high",
        )
        assert cfg.use_gpu is True
        assert cfg.sample_rate == 8000
        assert cfg.n_filters == 64
        assert cfg.resolution == "high"
