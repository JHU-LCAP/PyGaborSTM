"""Plotting helpers. Requires the `viz` extra."""

import numpy as np
import pytest

import pygaborstm as stm

pytest.importorskip("matplotlib", reason="needs pygaborstm[viz]")

pytestmark = pytest.mark.usefixtures("close_figures")


class TestPlotSpectrogram:
    def test_accepts_dataclass(self, spectrogram_from_tone):
        ax = stm.plot.plt_spectrogram(spectrogram_from_tone)
        assert ax is not None

    def test_accepts_raw_array(self, spectrogram_from_tone):
        ax = stm.plot.plt_spectrogram(spectrogram_from_tone.data)
        assert ax is not None

    def test_draws_onto_supplied_axes(self, spectrogram_from_tone):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert stm.plot.plt_spectrogram(spectrogram_from_tone, ax=ax) is ax


class TestPlotRSF:
    def test_accepts_dataclass(self, rsf_from_tone):
        assert stm.plot.plt_rsf(rsf_from_tone) is not None

    def test_folded(self, rsf_from_tone):
        assert stm.plot.plt_rsf(rsf_from_tone, fold=True) is not None

    def test_raw_array_needs_axes(self, rsf_from_tone):
        with pytest.raises(ValueError, match="rates and scales must be provided"):
            stm.plot.plt_rsf(rsf_from_tone.data)

    def test_raw_array_with_axes(self, rsf_from_tone):
        ax = stm.plot.plt_rsf(
            rsf_from_tone.data, rates=rsf_from_tone.rates, scales=rsf_from_tone.scales
        )
        assert ax is not None

    def test_all_zero_rsf(self, rsf_from_tone):
        blank = stm.structs.RSF(
            np.zeros_like(rsf_from_tone.data),
            rsf_from_tone.times,
            rsf_from_tone.rates,
            rsf_from_tone.scales,
            rsf_from_tone.freqs,
        )
        assert stm.plot.plt_rsf(blank) is not None


class TestShowGuard:
    # plt.show() on a file-only backend does nothing but emit a warning.
    # Notebooks must keep displaying, so the guard has to let them through.
    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("Agg", False),
            ("pdf", False),
            ("TkAgg", True),
            ("module://ipympl.backend_nbagg", True),
        ],
    )
    def test_only_shows_on_a_backend_that_can_display(self, backend, expected):
        from unittest.mock import patch

        from pygaborstm import plot as plot_module

        with (
            patch.object(plot_module.plt, "show") as show,
            patch.object(plot_module.plt, "get_backend", return_value=backend),
        ):
            plot_module._show()

        assert show.called is expected


class TestGrids:
    def test_spectrogram_grid(self, spectrogram_from_tone):
        stm.plot.plt_spectrogram_grid(
            [{"spectrogram": spectrogram_from_tone, "title": "a"}]
        )

    def test_rsf_grid(self, rsf_from_tone):
        stm.plot.plt_rsf_grid([{"rsf": rsf_from_tone, "title": "a"}])

    def test_empty_grids_are_noops(self):
        stm.plot.plt_spectrogram_grid([])
        stm.plot.plt_rsf_grid([])

    @pytest.mark.parametrize(
        "fn, key",
        [
            (lambda d: stm.plot.plt_spectrogram_grid(d), "spectrogram"),
            (lambda d: stm.plot.plt_rsf_grid(d), "rsf"),
        ],
    )
    def test_missing_key_names_the_entry(self, fn, key):
        with pytest.raises(
            KeyError, match=f"grid entry 0 is missing the required '{key}'"
        ):
            fn([{"title": "no payload"}])
