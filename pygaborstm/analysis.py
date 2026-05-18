"""
Analysis and visualization for MTF validation.

Provides matched-filter MTF computation and plotting for validating
Gabor filterbank tuning against Chi, Ru & Shamma (1999).
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_matched_filter_mtf(rsf_dict: dict) -> dict:
    """
    Compute MTF by extracting each filter's response to its matched ripple.

    Args:
        rsf_dict: {(rate, scale): RSF} for each ripple stimulus

    Returns:
        dict with upward/downward matrices (raw magnitudes), rates, scales
    """
    first = next(iter(rsf_dict.values()))
    up_rates = np.array(first.upward_rates())
    down_rates = np.array(first.downward_rates())
    scales = np.array(first.scales)

    matched_up = np.zeros((len(scales), len(up_rates)))
    matched_down = np.zeros((len(scales), len(down_rates)))

    for (rate, scale), rsf in rsf_dict.items():
        rate_idx = np.where(rsf.rates == rate)[0][0]
        scale_idx = np.where(rsf.scales == scale)[0][0]
        response = rsf.data[:, rate_idx, scale_idx, :].mean()

        scale_i = np.where(scales == scale)[0][0]
        if rate < 0:
            rate_i = np.where(up_rates == rate)[0][0]
            matched_up[scale_i, rate_i] = response
        else:
            rate_i = np.where(down_rates == rate)[0][0]
            matched_down[scale_i, rate_i] = response

    return {
        "upward": matched_up,
        "downward": matched_down,
        "up_rates": up_rates,
        "down_rates": down_rates,
        "scales": scales,
    }


def _get_plot_data(rsf_dict: dict, normalize: bool):
    """Get data and limits for plotting."""
    data = compute_matched_filter_mtf(rsf_dict)
    up, down = data["upward"], data["downward"]

    if normalize:
        combined = np.concatenate([up, down], axis=1)
        cmin, cmax = combined.min(), combined.max()
        up = (up - cmin) / (cmax - cmin + 1e-10)
        down = (down - cmin) / (cmax - cmin + 1e-10)
        vmin, vmax = 0, 1
        label = "Normalized Response"
    else:
        vmin = min(up.min(), down.min())
        vmax = max(up.max(), down.max())
        label = "Response Magnitude"

    return {
        "up": up,
        "down": down,
        "up_rates": data["up_rates"],
        "down_rates": data["down_rates"],
        "scales": data["scales"],
        "vmin": vmin,
        "vmax": vmax,
        "label": label,
    }


def _add_colorbar(fig, ax1, ax2, cmap, vmin, vmax, label):
    """Add colorbar between two axes."""
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    p1, p2 = ax1.get_position(), ax2.get_position()
    cbar_ax = fig.add_axes([(p1.x1 + p2.x0) / 2 - 0.01, p1.y0, 0.02, p1.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax, format="%.2f")
    cbar_ax.yaxis.set_ticks_position("right")
    cbar_ax.yaxis.set_label_position("left")
    cbar.set_label(label)


def _format_ax(ax, rates, scales, is_up, xlabel="Rate (Hz)", ylabel="Scale (cyc/oct)"):
    """Format axis with ticks and labels."""
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{int(r)}" for r in rates])
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s:.2f}" for s in scales])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not is_up:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")


def plot_mtf_contour(
    rsf_dict: dict,
    title: str = "Matched Filter MTF",
    figsize: tuple = (11, 5),
    cmap: str = "gray",
    levels: int = 10,
    normalize: bool = True,
    interp_factor: int = 10,
) -> tuple:
    """Plot matched filter MTF as contour plots."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import zoom

    d = _get_plot_data(rsf_dict, normalize)

    if cmap == "gray":
        cmap = LinearSegmentedColormap.from_list(
            "gray_trunc", plt.cm.gray(np.linspace(0.15, 1.0, 256))
        )

    lvls = np.linspace(d["vmin"], d["vmax"], levels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, vals, rates, name, is_up in [
        (ax1, d["up"], d["up_rates"], "Upward", True),
        (ax2, d["down"], d["down_rates"], "Downward", False),
    ]:
        # interpolate data for smoother contours
        vals_smooth = zoom(vals, interp_factor, order=3)
        x_smooth = np.linspace(0, len(rates) - 1, vals_smooth.shape[1])
        y_smooth = np.linspace(0, len(d["scales"]) - 1, vals_smooth.shape[0])

        ax.contourf(
            x_smooth,
            y_smooth,
            vals_smooth,
            levels=lvls,
            cmap=cmap,
            vmin=d["vmin"],
            vmax=d["vmax"],
        )
        cs = ax.contour(
            x_smooth, y_smooth, vals_smooth, levels=lvls, colors="k", linewidths=0.5
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.set_title(name)
        ax.set_box_aspect(1)
        _format_ax(ax, rates, d["scales"], is_up)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    _add_colorbar(fig, ax1, ax2, cmap, d["vmin"], d["vmax"], d["label"])

    return fig, (ax1, ax2)


def plot_mtf_heatmap(
    rsf_dict: dict,
    title: str = "Matched Filter MTF",
    figsize: tuple = (14, 6),
    cmap: str = "viridis",
    normalize: bool = True,
    annot: bool = True,
) -> tuple:
    """Plot matched filter MTF as heatmaps using seaborn."""
    import seaborn as sns

    d = _get_plot_data(rsf_dict, normalize)
    fmt = (
        ".2f"
        if normalize
        else (".1e" if d["vmax"] > 100 or d["vmax"] < 0.01 else ".2f")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, vals, rates, name, is_up in [
        (ax1, d["up"], d["up_rates"], "Upward", True),
        (ax2, d["down"], d["down_rates"], "Downward", False),
    ]:
        sns.heatmap(
            vals,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            xticklabels=[f"{int(r)}" for r in rates],
            yticklabels=[f"{s:.2f}" for s in d["scales"]],
            vmin=d["vmin"],
            vmax=d["vmax"],
            cbar=False,
            annot_kws={"size": 9},
        )
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("Rate (Hz)")
        ax.set_ylabel("Scale (cyc/oct)")
        if not is_up:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    _add_colorbar(fig, ax1, ax2, cmap, d["vmin"], d["vmax"], d["label"])

    return fig, (ax1, ax2)


def plot_mtf_lines(
    rsf_dict: dict,
    title: str = "MTF by Scale",
    figsize: tuple = (12, 5),
    normalize: bool = True,
) -> tuple:
    """Plot response vs rate for each scale."""
    d = _get_plot_data(rsf_dict, normalize)

    n_scales = len(d["scales"])
    colors = plt.cm.rainbow(np.linspace(0, 1, n_scales))[::-1]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "<", ">"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, vals, rates, name in [
        (ax1, d["up"], d["up_rates"], "Upward"),
        (ax2, d["down"], d["down_rates"], "Downward"),
    ]:
        for i, scale in enumerate(d["scales"]):
            ax.plot(
                range(len(rates)),
                vals[i],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=f"{scale:.2f} c/o",
                markersize=6,
            )
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{int(r)}" for r in rates])
        ax.set_xlabel("Rate (Hz)")
        ax.set_ylabel(d["label"])
        ax.set_title(name)
        ax.set_ylim(d["vmin"] - 0.05, d["vmax"] + 0.05)

    ax1.legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_mtf_1d(
    rsf_dict: dict,
    title: str = "1D MTF (SVD)",
    figsize: tuple = (12, 5),
    normalize: bool = True,
) -> tuple:
    """Plot 1D spectral and temporal MTFs via SVD decomposition."""
    d = _get_plot_data(rsf_dict, normalize)

    # SVD decomposition
    U_up, _, Vt_up = np.linalg.svd(d["up"], full_matrices=False)
    U_down, _, Vt_down = np.linalg.svd(d["down"], full_matrices=False)

    spectral = (np.abs(U_up[:, 0]) + np.abs(U_down[:, 0])) / 2
    temp_up = np.abs(Vt_up[0, :])
    temp_down = np.abs(Vt_down[0, :])

    # Normalize each to [0, 1]
    spectral = spectral / (spectral.max() + 1e-10)
    temp_up = temp_up / (temp_up.max() + 1e-10)
    temp_down = temp_down / (temp_down.max() + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Spectral MTF
    ax1.plot(range(len(d["scales"])), spectral, "o-k", markersize=6)
    ax1.set_xticks(range(len(d["scales"])))
    ax1.set_xticklabels([f"{s:.2f}" for s in d["scales"]])
    ax1.set_xlabel("Scale (cyc/oct)")
    ax1.set_ylabel("Modulation Index")
    ax1.set_title("Spectral MTF")
    ax1.set_ylim(0, 1.1)

    # Temporal MTF
    n_up = len(temp_up)
    ax2.plot(range(n_up), temp_up, "*-b", label="upward", markersize=6)
    ax2.plot(
        range(n_up, n_up + len(temp_down)),
        temp_down,
        "v--g",
        label="downward",
        markersize=6,
    )
    all_rates = np.concatenate([d["up_rates"], d["down_rates"]])
    ax2.set_xticks(range(len(all_rates)))
    ax2.set_xticklabels([f"{int(r)}" for r in all_rates])
    ax2.set_xlabel("Rate (Hz)")
    ax2.set_ylabel("Modulation Index")
    ax2.set_title("Temporal MTF")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="best", fontsize=7)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    return fig, (ax1, ax2)
