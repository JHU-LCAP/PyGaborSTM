import numpy as np
import matplotlib.pyplot as plt

from .structs import Spectrogram, RSF


def plot_spectrogram(
    spectrogram: Spectrogram | np.ndarray,
    frmlen_ms: float = 16.0,
    title: str = "Auditory Spectrogram",
    figsize: tuple = (12, 6),
    cmap: str = "turbo",
) -> None:
    """
    Plot auditory spectrogram.

    Args:
        spectrogram: Spectrogram object or array [n_freq × n_time]
        frmlen_ms: Frame length in ms (used if array provided)
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    if isinstance(spectrogram, Spectrogram):
        data = spectrogram.data
        duration = spectrogram.duration
    else:
        data = spectrogram
        n_frames = data.shape[1]
        duration = n_frames * frmlen_ms / 1000.0

    n_filters = data.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_ylabel("Frequency Channel")

    plt.colorbar(im, ax=ax, label="Amplitude")
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