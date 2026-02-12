from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .structs import Spectrogram, RSF

if TYPE_CHECKING:
    from .gabor import GaborFilterbank


def plot_spectrogram(
    spectrogram: Spectrogram | np.ndarray | list,
    titles: str | list[str] | None = None,
    frmlen_ms: float = 16.0,
    suptitle: str | None = None,
    figsize: tuple | None = None,
    cmap: str = "turbo",
    max_cols: int = 4,
) -> None:
    """
    Plot one or more auditory spectrograms.

    Args:
        spectrogram: Spectrogram object, array, or list of either
        titles: Title(s) for subplot(s). String for single, list for multiple.
        frmlen_ms: Frame length in ms (used if arrays provided)
        suptitle: Overall figure title (optional)
        figsize: Figure size. If None, auto-calculated.
        cmap: Colormap
        max_cols: Maximum columns for multiple spectrograms
    """
    # Normalize to list
    if isinstance(spectrogram, list):
        specs = spectrogram
    else:
        specs = [spectrogram]

    n = len(specs)

    # Normalize titles
    if titles is None:
        titles = ["Auditory Spectrogram"] if n == 1 else [f"Spectrogram {i+1}" for i in range(n)]
    elif isinstance(titles, str):
        titles = [titles]

    # Calculate layout
    if n == 1:
        n_rows, n_cols = 1, 1
        if figsize is None:
            figsize = (12, 6)
    else:
        n_cols = min(n, max_cols)
        n_rows = (n + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (spec, title) in enumerate(zip(specs, titles)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Extract data
        if isinstance(spec, Spectrogram):
            data = spec.data
            duration = spec.duration
        else:
            data = spec
            n_frames = data.shape[1]
            duration = n_frames * frmlen_ms / 1000.0

        n_filters = data.shape[0]

        im = ax.imshow(
            np.abs(data),
            aspect="auto",
            origin="lower",
            extent=(0, duration, 0, n_filters),
            cmap=cmap,
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("Frequency Channel")

        if n == 1:
            plt.colorbar(im, ax=ax, label="Amplitude")

    # Hide empty subplots
    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_rsf(
    rsf: RSF | np.ndarray,
    rates: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    fold: bool = False,
    title: str = "Rate-Scale Representation",
    figsize: tuple = (10, 8),
    cmap: str = "jet",
) -> None:
    """
    Plot rate-scale representation (averaged over time and frequency).

    Args:
        rsf: RSF object or array [n_frames × n_rates × n_scales × n_freq]
        rates: Rate values (required if array provided)
        scales: Scale values (required if array provided)
        fold: If True, fold positive/negative rates for symmetric visualization
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    if isinstance(rsf, RSF):
        data = rsf.rate_scale_matrix(fold=fold)  # [n_scales × n_rates]
        rates = rsf.rates
        scales = rsf.scales
    else:
        if rates is None or scales is None:
            raise ValueError("rates and scales required when rsf is array")
        data = rsf.mean(axis=(0, 3)).T  # [n_scales × n_rates]
        if fold:
            n_rates_half = data.shape[1] // 2
            rs_left = np.flip(data[:, :n_rates_half], axis=1)
            rs_right = data[:, n_rates_half:]
            rs_folded = (rs_left + rs_right) / 2
            data = np.concatenate([np.flip(rs_folded, axis=1), rs_folded], axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(rates)))
    ax.set_xticklabels([f"{r:.0f}" for r in rates], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(scales)))
    ax.set_yticklabels([f"{s:.2f}" for s in scales])

    ax.set_title(title)
    ax.set_xlabel("Rate (Hz)")
    ax.set_ylabel("Scale (cycles/octave)")

    plt.colorbar(im, ax=ax, label="Modulation Energy")
    plt.tight_layout()
    plt.show()


def plot_filterbank(
    filterbank: "GaborFilterbank",
    figsize: tuple = (12, 4),
) -> None:
    """
    Plot the rate-scale coverage of the filterbank.

    Args:
        filterbank: GaborFilterbank instance
        figsize: Figure size
    """
    rates = filterbank.rates
    scales = filterbank.scales

    fig, ax = plt.subplots(figsize=figsize)

    for r in rates:
        for s in scales:
            ax.plot(r, s, "ko", markersize=4)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log", base=2)

    ax.set_xlabel("Rate (Hz)")
    ax.set_ylabel("Scale (cycles/octave)")
    ax.set_title(f"Gabor Filterbank ({len(rates)} rates × {len(scales)} scales)")

    plt.tight_layout()
    plt.show()
