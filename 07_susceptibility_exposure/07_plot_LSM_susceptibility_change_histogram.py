# -*- coding: utf-8 -*-
"""
Plot a histogram of pixel-level landslide susceptibility probability change.

Input raster:
    LSM_24_00.tif = susceptibility probability in 2020-2024 minus 2000-2019.

The histogram is computed from all valid pixels by streaming raster blocks, so
large rasters do not need to be loaded fully into memory.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# 1. Paths
# ============================================================

INPUT_TIF = (
    r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map"
    r"\LSM_24_00.tif"
)

OUT_DIR = r"H:\Himalaya\RF_susceptibility\susceptibility\c.analysis"
OUT_PNG = os.path.join(
    OUT_DIR,
    "Fig_LSM_susceptibility_change_histogram.png",
)
OUT_PDF = os.path.join(
    OUT_DIR,
    "Fig_LSM_susceptibility_change_histogram.pdf",
)


# ============================================================
# 2. Histogram and style settings
# ============================================================

X_MIN = -0.2
X_MAX = 0.2
N_BINS = 120

DECREASE_COLOR = "#2F6FB0"
INCREASE_COLOR = "#A91522"
ZERO_LINE_COLOR = "#222222"

FIG_W, FIG_H = 3.55, 2.65
DPI = 600


# ============================================================
# 3. Helper functions
# ============================================================

def setup_plot_style():
    """Set compact publication-style figure parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def get_valid_values(block, nodata):
    """Return finite non-nodata values from one raster block."""
    values = np.asarray(block, dtype="float64").ravel()
    mask = np.isfinite(values)

    if nodata is not None and np.isfinite(nodata):
        mask &= values != nodata

    return values[mask]


def compute_streaming_histogram(tif_path, x_min, x_max, n_bins):
    """
    Compute histogram counts and summary statistics from all valid pixels.

    Proportions are normalized by all valid pixels, including values outside
    the plotted x-range. This keeps the y-axis interpretable as total-pixel
    percentage.
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Input raster not found: {tif_path}")

    edges = np.linspace(x_min, x_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)

    total_valid = 0
    n_decrease = 0
    n_increase = 0
    n_zero = 0
    n_below_range = 0
    n_above_range = 0
    value_min = np.inf
    value_max = -np.inf

    with rasterio.open(tif_path) as src:
        print("Input raster:")
        print(f"  {tif_path}")
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width:,} x {src.height:,}")
        print(f"  Nodata: {src.nodata}")

        for _, window in src.block_windows(1):
            block = src.read(1, window=window, masked=False)
            values = get_valid_values(block, src.nodata)

            if values.size == 0:
                continue

            total_valid += values.size
            n_decrease += int(np.sum(values < 0))
            n_increase += int(np.sum(values > 0))
            n_zero += int(np.sum(values == 0))
            n_below_range += int(np.sum(values < x_min))
            n_above_range += int(np.sum(values > x_max))
            value_min = min(value_min, float(np.min(values)))
            value_max = max(value_max, float(np.max(values)))

            in_range = values[(values >= x_min) & (values <= x_max)]
            if in_range.size:
                block_counts, _ = np.histogram(in_range, bins=edges)
                counts += block_counts

    if total_valid == 0:
        raise ValueError("No valid pixels found in the input raster.")

    stats = {
        "total_valid": total_valid,
        "n_decrease": n_decrease,
        "n_increase": n_increase,
        "n_zero": n_zero,
        "n_below_range": n_below_range,
        "n_above_range": n_above_range,
        "value_min": value_min,
        "value_max": value_max,
    }

    return edges, counts, stats


def format_axis(ax):
    """Apply compact axis formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.4,
        width=0.55,
        pad=2,
    )
    ax.grid(False)


def plot_histogram(edges, counts, stats):
    """Create and save the susceptibility-change histogram."""
    setup_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    proportions = counts / stats["total_valid"] * 100.0
    colors = np.where(centers < 0, DECREASE_COLOR, INCREASE_COLOR)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.bar(
        centers,
        proportions,
        width=widths,
        align="center",
        color=colors,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    ax.axvline(
        0,
        color=ZERO_LINE_COLOR,
        linewidth=0.8,
        linestyle="-",
        zorder=4,
    )
    ax.axhline(0, color="0.2", linewidth=0.55, zorder=3)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks([X_MIN, 0, X_MAX])
    ax.set_xticklabels([f"{X_MIN:.2f}", "0", f"{X_MAX:.2f}"])
    ax.set_xlabel(r"$\Delta$ susceptibility")
    ax.set_ylabel("Pixel proportion (%)")

    y_max = float(np.nanmax(proportions)) if proportions.size else 0.0
    ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1)
    format_axis(ax)

    ax.text(
        0.02,
        0.96,
        "Decrease",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=DECREASE_COLOR,
    )
    ax.text(
        0.98,
        0.96,
        "Increase",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=INCREASE_COLOR,
    )

    fig.tight_layout(pad=0.35)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\nSaved figure: {OUT_PNG}")
    print(f"Saved figure: {OUT_PDF}")


def print_summary(stats):
    """Print concise pixel-count diagnostics."""
    total = stats["total_valid"]

    print("\nSusceptibility-change summary:")
    print(f"  Valid pixels: {total:,}")
    print(
        "  Min / max: "
        f"{stats['value_min']:.6f} / {stats['value_max']:.6f}"
    )
    print(
        "  Decrease / zero / increase: "
        f"{stats['n_decrease']:,} / "
        f"{stats['n_zero']:,} / "
        f"{stats['n_increase']:,}"
    )
    print(
        "  Share decrease / zero / increase: "
        f"{stats['n_decrease'] / total * 100:.2f}% / "
        f"{stats['n_zero'] / total * 100:.2f}% / "
        f"{stats['n_increase'] / total * 100:.2f}%"
    )
    print(
        f"  Pixels outside plotted range [{X_MIN}, {X_MAX}]: "
        f"{stats['n_below_range'] + stats['n_above_range']:,} "
        f"({(stats['n_below_range'] + stats['n_above_range']) / total * 100:.2f}%)"
    )


# ============================================================
# 4. Main
# ============================================================

def main():
    edges, counts, stats = compute_streaming_histogram(
        tif_path=INPUT_TIF,
        x_min=X_MIN,
        x_max=X_MAX,
        n_bins=N_BINS,
    )
    print_summary(stats)
    plot_histogram(edges, counts, stats)


if __name__ == "__main__":
    main()
