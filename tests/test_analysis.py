"""Matched-filter MTF analysis. Requires the `viz` extra."""

import pytest

import pygaborstm as stm

pytest.importorskip("matplotlib", reason="needs pygaborstm[viz]")

pytestmark = pytest.mark.usefixtures("close_figures")


class TestComputeMatchedFilterMTF:
    def test_returns_expected_keys_and_shapes(self, rsf_dict, rsf_from_tone):
        out = stm.analysis.compute_matched_filter_mtf(rsf_dict)

        assert set(out) == {"upward", "downward", "up_rates", "down_rates", "scales"}
        n_scales = len(rsf_from_tone.scales)
        assert out["upward"].shape == (n_scales, rsf_from_tone.n_rates // 2)
        assert out["downward"].shape == (n_scales, rsf_from_tone.n_rates // 2)

    def test_full_dict_does_not_warn(self, rsf_dict, recwarn):
        stm.analysis.compute_matched_filter_mtf(rsf_dict)
        assert not [w for w in recwarn if "cells" in str(w.message)]

    def test_empty_dict_rejected(self):
        # Used to raise a bare StopIteration.
        with pytest.raises(ValueError, match="rsf_dict is empty"):
            stm.analysis.compute_matched_filter_mtf({})

    def test_key_off_the_grid_names_the_value(self, rsf_from_tone):
        with pytest.raises(ValueError, match="rate 3.3 is not present"):
            stm.analysis.compute_matched_filter_mtf({(3.3, 0.7): rsf_from_tone})

    def test_partial_dict_warns_about_zeros(self, rsf_from_tone):
        # Unfilled cells stay 0.0 and look like a real null response.
        partial = {
            (float(rate), float(rsf_from_tone.scales[0])): rsf_from_tone
            for rate in rsf_from_tone.rates
            if rate > 0
        }
        with pytest.warns(UserWarning, match="are reported as 0.0"):
            stm.analysis.compute_matched_filter_mtf(partial)


class TestMTFPlots:
    @pytest.mark.parametrize(
        "name",
        ["plot_mtf_contour", "plot_mtf_heatmap", "plot_mtf_lines", "plot_mtf_1d"],
    )
    def test_returns_figure_and_axes(self, name, rsf_dict):
        if name == "plot_mtf_heatmap":
            pytest.importorskip("seaborn", reason="needs pygaborstm[viz]")
        fig, axes = getattr(stm.analysis, name)(rsf_dict)
        assert fig is not None
        assert len(axes) == 2

    def test_single_entry_dict(self, rsf_from_tone):
        one = {
            (
                float(rsf_from_tone.rates[-1]),
                float(rsf_from_tone.scales[0]),
            ): rsf_from_tone
        }
        with pytest.warns(UserWarning):
            fig, _ = stm.analysis.plot_mtf_contour(one)
        assert fig is not None
