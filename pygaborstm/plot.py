"""Plotting helpers for :class:`Spectrogram` and :class:`RSF` outputs.

Thin wrappers around :mod:`matplotlib` that handle the per-axis
formatting (log-spaced frequency/rate/scale ticks, the upward/downward
rate split, etc.) so notebook code can stay short.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .structs import Spectrogram, RSF

if TYPE_CHECKING:
    pass


def _get_freq_ticks(freqs: np.ndarray):
    """Get frequency tick positions and labels in Hz."""
    f_min, f_max = freqs[0], freqs[-1]

    nice_freqs = [100, 200, 500, 1000, 2000, 4000, 8000]
    tick_freqs = [f for f in nice_freqs if f_min <= f <= f_max]

    positions = []
    labels = []
    for f in tick_freqs:
        idx = np.argmin(np.abs(freqs - f))
        positions.append(idx)
        labels.append(f"{int(f)}")

    return positions, labels


def plt_spectrogram(
    spectrogram: Spectrogram | np.ndarray,
    title: str = "Auditory Spectrogram",
    figsize: tuple = (12, 6),
    cmap: str = "viridis",
    frmlen_ms: float = 16.0,
    ax: Optional[Axes] = None,
    show_colorbar: bool = True,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 9,
    interpolation: str = "bilinear",
) -> Axes:
    """Plot a single auditory spectrogram.

    Parameters
    ----------
    spectrogram : Spectrogram or np.ndarray
        Spectrogram dataclass or a raw ``(n_freq, n_time)`` array.
    title : str, default "Auditory Spectrogram"
        Plot title.
    figsize : tuple, default (12, 6)
        Figure size. Only used when ``ax`` is None.
    cmap : str, default "viridis"
        Matplotlib colormap.
    frmlen_ms : float, default 16.0
        Frame length in ms, used to compute the time axis when a raw
        array is passed.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created and shown.
    show_colorbar : bool, default True
        Add a colorbar (only when ``ax`` is None).
    title_fontsize : int, default 12
        Title font size.
    label_fontsize : int, default 10
        Axis label font size.
    tick_fontsize : int, default 9
        Tick label font size.
    interpolation : str, default "bilinear"
        Interpolation passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the spectrogram was drawn on.
    """
    # Create figure if no ax provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    if isinstance(spectrogram, Spectrogram):
        data = spectrogram.data
        duration = spectrogram.duration
        freqs = spectrogram.freqs
    else:
        data = spectrogram
        n_frames = data.shape[1]
        duration = n_frames * frmlen_ms / 1000.0
        freqs = None

    n_filters = data.shape[0]

    im = ax.imshow(
        np.abs(data),
        aspect="auto",
        origin="lower",
        extent=(0, duration, 0, n_filters),
        cmap=cmap,
        interpolation=interpolation,
    )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Time (s)", fontsize=label_fontsize)
    ax.set_ylabel("Frequency (Hz)", fontsize=label_fontsize)

    if freqs is not None:
        freq_positions, freq_labels = _get_freq_ticks(freqs)
        ax.set_yticks(freq_positions)
        ax.set_yticklabels(freq_labels)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if show_colorbar and standalone:
        plt.colorbar(im, ax=ax, label="Amplitude")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plt_spectrogram_grid(
    data: List[Dict],
    n_cols: int = 4,
    figsize: tuple | None = None,
    cmap: str = "viridis",
    frmlen_ms: float = 16.0,
    suptitle: str | None = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot multiple spectrograms in a grid.

    Parameters
    ----------
    data : list of dict
        Each entry must have a ``"spectrogram"`` key (a
        :class:`Spectrogram` or raw array) and may have a ``"title"``
        key.
    n_cols : int, default 4
        Number of columns in the grid.
    figsize : tuple, optional
        Figure size. Auto-calculated from ``n_cols`` and the number of
        rows if not given.
    cmap : str, default "viridis"
        Matplotlib colormap.
    frmlen_ms : float, default 16.0
        Frame length in ms, used when raw arrays are passed.
    suptitle : str, optional
        Overall figure title.
    save_path : str, optional
        If given, save the figure to this path at 200 DPI before showing.
    """
    n_plots = len(data)
    if n_plots == 0:
        print("No data to plot.")
        return

    # Calculate grid dimensions based on actual number of plots
    actual_cols = min(n_plots, n_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    if figsize is None:
        figsize = (4 * actual_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, actual_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, item in enumerate(data):
        ax = axes_flat[idx]
        spectrogram = item["spectrogram"]
        title = item.get("title", f"Spectrogram {idx + 1}")

        plt_spectrogram(
            spectrogram=spectrogram,
            title=title,
            cmap=cmap,
            frmlen_ms=frmlen_ms,
            ax=ax,
            show_colorbar=False,
            title_fontsize=9,
            label_fontsize=7,
            tick_fontsize=6,
        )

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to '{save_path}'")

    plt.show()


def plt_rsf(
    rsf: RSF | np.ndarray,
    rates: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    fold: bool = False,
    title: str = "Rate-Scale Representation",
    figsize: tuple = (10, 8),
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
    show_colorbar: bool = True,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 9,
    square: bool = False,
) -> Axes:
    """Plot a single RSF representation as a 2D scale-by-rate heatmap.

    Parameters
    ----------
    rsf : RSF or np.ndarray
        RSF dataclass, or a raw 4D array of shape
        ``(n_frames, n_rates, n_scales, n_freq)`` (in which case
        ``rates`` and ``scales`` must be supplied).
    rates : np.ndarray, optional
        Rate axis values for tick labels. Defaults to ``rsf.rates``
        when ``rsf`` is an :class:`RSF`.
    scales : np.ndarray, optional
        Scale axis values for tick labels. Defaults to ``rsf.scales``
        when ``rsf`` is an :class:`RSF`.
    fold : bool, default False
        If True, average the upward and downward halves and mirror the
        result back out to produce a symmetric plot.
    title : str, default "Rate-Scale Representation"
        Plot title.
    figsize : tuple, default (10, 8)
        Figure size. Only used when ``ax`` is None.
    cmap : str, default "viridis"
        Matplotlib colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created and shown.
    show_colorbar : bool, default True
        Add a colorbar (only when ``ax`` is None).
    title_fontsize : int, default 12
        Title font size.
    label_fontsize : int, default 10
        Axis label font size.
    tick_fontsize : int, default 9
        Tick label font size.
    square : bool, default False
        Force a square aspect ratio on the axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the RSF was drawn on.

    Raises
    ------
    ValueError
        If ``rsf`` is a raw array but ``rates`` or ``scales`` is None.
    """
    # Extract data and rates/scales
    if isinstance(rsf, RSF):
        data = rsf.rate_scale_matrix(fold=fold)
        r_rates = rsf.rates if rates is None else rates
        r_scales = rsf.scales if scales is None else scales
    else:
        if rates is None or scales is None:
            raise ValueError(
                "rates and scales must be provided when rsf is a raw array"
            )
        r_rates = rates
        r_scales = scales
        data = rsf.mean(axis=(0, 3)).T
        if fold:
            n_rates_half = data.shape[1] // 2
            rs_left = np.flip(data[:, :n_rates_half], axis=1)
            rs_right = data[:, n_rates_half:]
            rs_folded = (rs_left + rs_right) / 2
            data = np.concatenate([np.flip(rs_folded, axis=1), rs_folded], axis=1)

    # Create figure if no ax provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )

    if square:
        ax.set_box_aspect(1)

    n_scales, n_rates = data.shape

    # midpoint between negative (upward) and positive (downward) rates
    ax.axvline(x=(n_rates - 1) / 2, color="white", linewidth=1, linestyle="-")

    # Standard tick values (readable at any resolution)
    STANDARD_RATES = [-32, -16, -8, -4, -2, 2, 4, 8, 16, 32]
    STANDARD_SCALES = [0.25, 0.5, 1, 2, 4, 8]

    # Map rate values to pixel positions (log-spaced)
    rate_min, rate_max = np.min(np.abs(r_rates)), np.max(np.abs(r_rates))

    rate_tick_positions = []
    rate_tick_labels = []
    for rate in STANDARD_RATES:
        abs_rate = abs(rate)
        if abs_rate < rate_min or abs_rate > rate_max:
            continue
        log_pos = np.log2(abs_rate / rate_min) / np.log2(rate_max / rate_min)
        if rate < 0:
            pixel_pos = (n_rates / 2 - 1) * (1 - log_pos)
        else:
            pixel_pos = (n_rates / 2) + (n_rates / 2 - 1) * log_pos
        rate_tick_positions.append(pixel_pos)
        rate_tick_labels.append(str(rate))

    # Map scale values to pixel positions (log-spaced)
    scale_min, scale_max = np.min(r_scales), np.max(r_scales)

    scale_tick_positions = []
    scale_tick_labels = []
    for scale in STANDARD_SCALES:
        if scale < scale_min or scale > scale_max:
            continue
        log_pos = np.log2(scale / scale_min) / np.log2(scale_max / scale_min)
        pixel_pos = (n_scales - 1) * log_pos
        scale_tick_positions.append(pixel_pos)
        scale_tick_labels.append(f"{scale:.2f}" if scale < 1 else str(int(scale)))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Rate (Hz)", fontsize=label_fontsize)
    ax.set_ylabel("Scale (cycles/octave)", fontsize=label_fontsize)

    ax.set_xticks(rate_tick_positions)
    ax.set_xticklabels(rate_tick_labels)
    ax.set_yticks(scale_tick_positions)
    ax.set_yticklabels(scale_tick_labels)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if show_colorbar and standalone:
        plt.colorbar(im, ax=ax, label="Modulation Energy")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def plt_rsf_grid(
    data: List[Dict],
    rates: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    fold: bool = False,
    n_cols: int = 6,
    figsize: tuple | None = None,
    cmap: str = "viridis",
    suptitle: str | None = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot multiple RSF representations in a grid.

    Parameters
    ----------
    data : list of dict
        Each entry must have an ``"rsf"`` key (an :class:`RSF` or raw
        4D array) and may have a ``"title"`` key.
    rates : np.ndarray, optional
        Rate axis values. Defaults to ``rates`` from the first
        :class:`RSF` in ``data``.
    scales : np.ndarray, optional
        Scale axis values. Defaults to ``scales`` from the first
        :class:`RSF` in ``data``.
    fold : bool, default False
        Forwarded to :func:`plt_rsf` for each subplot.
    n_cols : int, default 6
        Number of columns in the grid.
    figsize : tuple, optional
        Figure size. Auto-calculated if not given.
    cmap : str, default "viridis"
        Matplotlib colormap.
    suptitle : str, optional
        Overall figure title.
    save_path : str, optional
        If given, save the figure to this path at 200 DPI before showing.

    Raises
    ------
    ValueError
        If raw RSF arrays are passed but ``rates`` or ``scales`` is None.
    """
    n_plots = len(data)
    if n_plots == 0:
        print("No data to plot.")
        return

    # Get rates/scales from first RSF if not provided
    first_rsf = data[0]["rsf"]
    if isinstance(first_rsf, RSF):
        r_rates = rates if rates is not None else first_rsf.rates
        r_scales = scales if scales is not None else first_rsf.scales
    else:
        if rates is None or scales is None:
            raise ValueError(
                "rates and scales must be provided when rsf is a raw array"
            )
        r_rates = rates
        r_scales = scales

    # Calculate grid dimensions based on actual number of plots
    actual_cols = min(n_plots, n_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    if figsize is None:
        figsize = (4 * actual_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, actual_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, item in enumerate(data):
        ax = axes_flat[idx]
        rsf = item["rsf"]
        title = item.get("title", f"RSF {idx + 1}")

        plt_rsf(
            rsf=rsf,
            rates=r_rates,
            scales=r_scales,
            fold=fold,
            title=title,
            cmap=cmap,
            ax=ax,
            show_colorbar=False,
            title_fontsize=9,
            label_fontsize=7,
            tick_fontsize=6,
            square=True,
        )

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to '{save_path}'")

    plt.show()
