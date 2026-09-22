import numpy as np
import pytest

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

        # Gammatone
        assert cfg.filter_order == 4
        assert cfg.erb_scale == 0.6

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


class TestEquality:
    def test_identical_configs_are_equal(self):
        # The generated __eq__ raised on the ndarray fields.
        assert Config() == Config()

    def test_differing_configs_are_not_equal(self):
        assert Config() != Config(n_filters=64)

    def test_differing_arrays_are_not_equal(self):
        assert Config() != Config(scales=np.array([1.0, 2.0]))

    def test_other_types_are_not_equal(self):
        assert Config() != "not a config"


class TestValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "sample_rate",
            "n_filters",
            "f_min",
            "octaves",
            "tau_ms",
            "frmlen_ms",
            "filter_order",
            "erb_scale",
            "rsf_frame_size_ms",
            "rsf_frame_shift_ms",
        ],
    )
    def test_nonpositive_values_rejected(self, field):
        with pytest.raises(ValueError, match=f"{field} must be > 0"):
            Config(**{field: 0})

    @pytest.mark.parametrize("field", ["f_min", "octaves", "tau_ms", "erb_scale"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_values_rejected(self, field, bad):
        # NaN fails every comparison, so a bare "<= 0" check lets it through
        # and the whole RSF comes back non-finite.
        with pytest.raises(ValueError, match="must be finite"):
            Config(**{field: bad})

    @pytest.mark.parametrize("field", ["rates", "scales"])
    def test_non_finite_arrays_rejected(self, field):
        with pytest.raises(ValueError, match="must all be finite"):
            Config(**{field: np.array([2.0, float("nan")])})

    def test_tau_shorter_than_one_sample_rejected(self):
        with pytest.raises(ValueError, match="integration kernel"):
            Config(tau_ms=0.001, sample_rate=16000)

    def test_window_shorter_than_frame_rejected(self):
        # Used to return an all-NaN RSF with no error at all.
        with pytest.raises(ValueError, match="entirely NaN"):
            Config(rsf_frame_size_ms=5, frmlen_ms=16)

    def test_frame_shorter_than_one_sample_rejected(self):
        with pytest.raises(ValueError, match="samples per frame"):
            Config(frmlen_ms=0.01, sample_rate=8000, octaves=4.0)

    def test_invalid_resolution_rejected(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            Config(resolution="nope")

    def test_nonpositive_scales_rejected(self):
        with pytest.raises(ValueError, match="scales must all be > 0"):
            Config(scales=np.array([0.0, 1.0]))

    def test_rates_without_positive_value_rejected(self):
        with pytest.raises(ValueError, match="at least one positive"):
            Config(rates=np.array([-2.0, -4.0]))

    def test_empty_rates_rejected(self):
        with pytest.raises(ValueError, match="at least one value"):
            Config(rates=np.array([]))

    def test_above_nyquist_warns_but_is_allowed(self):
        with pytest.warns(UserWarning, match="above Nyquist"):
            Config(sample_rate=8000)

    def test_validate_catches_post_construction_mutation(self):
        cfg = Config()
        cfg.frmlen_ms = 0
        with pytest.raises(ValueError, match="frmlen_ms must be > 0"):
            cfg.validate()
