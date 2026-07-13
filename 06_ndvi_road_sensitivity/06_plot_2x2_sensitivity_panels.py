# plot_environmental_sensitivity_heatmaps.py
# -*- coding: utf-8 -*-

"""
Purpose
-------
Read sensitivity-result CSVs for:

1. elevation
2. rainfall
3. slope
4. aspect

and generate a Nature-style 2×2 figure:
(a) Elevation
(b) Rainfall
(c) Slope
(d) Aspect (polar heatmap)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm


# =========================================================
# 1. INPUT PATHS
# =========================================================

base_dir = r"H:\Himalaya\figure"

folders = {
    "Elevation": os.path.join(
        base_dir,
        "NDVI_Elevation_2D_independent_RF"
    ),

    "Rainfall": os.path.join(
        base_dir,
        "NDVI_Rainfall_2D_independent_RF_with_event_rainfall"
    ),

    "Slope": os.path.join(
        base_dir,
        "NDVI_slope_2D_independent_RF"
    ),

    "Aspect": os.path.join(
        base_dir,
        "NDVI_Aspect_Polar_RF"
    )
}

output_figure = os.path.join(
    base_dir,
    "Environmental_Road_Sensitivity_2x2.png"
)

output_pdf = os.path.join(
    base_dir,
    "Environmental_Road_Sensitivity_2x2.pdf"
)


# =========================================================
# 2. FIND CSV
# =========================================================

def find_csv(folder):

    csvs = glob.glob(
        os.path.join(folder, "*.csv")
    )

    if len(csvs) == 0:

        raise FileNotFoundError(
            f"No CSV found:\n{folder}"
        )

    return csvs[0]


# =========================================================
# 3. LOAD DATA
# =========================================================

dfs = {}

for key, folder in folders.items():

    csv_path = find_csv(folder)

    print(f"Loading:\n{csv_path}")

    dfs[key] = pd.read_csv(csv_path)


# =========================================================
# 4. STYLE
# =========================================================

plt.rcParams.update({

    # Font family and global base font size
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],

    "font.size": 9,

    # Axis border linewidth
    "axes.linewidth": 0.8,

    # Axis tick linewidth
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.direction": "out",
    "ytick.direction": "out",

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =========================================================
# 5. GLOBAL COLOR NORMALIZATION
# =========================================================

all_sens = []

for df in dfs.values():

    if "sensitivity" in df.columns:

        all_sens.extend(
            df["sensitivity"].values
        )

all_sens = np.array(all_sens)

vmax = np.nanmax(np.abs(all_sens))

if vmax == 0:
    vmax = 1e-6

norm = TwoSlopeNorm(
    vmin=-vmax,
    vcenter=0,
    vmax=vmax
)


# =========================================================
# 6. BIN LABEL FORMATTER
# =========================================================

def simplify_interval_labels(labels):

    """
    Convert:
    (-0.001, 0.2]
    ->
    0.2

    (1000.0, 2000.0]
    ->
    2000
    """

    out = []

    for lab in labels:

        s = str(lab)

        try:

            right = (
                s.split(",")[1]
                .replace("]", "")
                .replace(")", "")
                .strip()
            )

            val = float(right)

            if abs(val - round(val)) < 1e-6:

                out.append(
                    str(int(round(val)))
                )

            else:

                out.append(
                    f"{val:.1f}"
                )

        except:

            out.append(s)

    return out


def interval_right_value(label):

    s = str(label)

    try:

        right = (
            s.split(",")[1]
            .replace("]", "")
            .replace(")", "")
            .strip()
        )

        return float(right)

    except:

        return np.nan


# =========================================================
# 7. DRAW RECTANGULAR HEATMAP
# =========================================================

def draw_2d_heatmap(
    ax,
    df,
    env_col,
    title,
    cmap="BrBG",
    sort_y_bins=False
):

    # =====================================================
    # pivot table
    # IMPORTANT:
    # do NOT reverse rows
    # bottom -> top = low -> high
    # =====================================================

    pivot = df.pivot(
        index=env_col,
        columns="ndvi_bin",
        values="sensitivity"
    )

    if sort_y_bins:

        y_order = sorted(
            pivot.index,
            key=interval_right_value
        )

        pivot = pivot.loc[y_order]

    matrix = pivot.values

    nrows, ncols = matrix.shape

    # =====================================================
    # square bins
    # =====================================================

    x_edges = np.arange(ncols + 1)
    y_edges = np.arange(nrows + 1)

    X, Y = np.meshgrid(
        x_edges,
        y_edges
    )

    im = ax.pcolormesh(
        X,
        Y,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        # Inner grid line color
        edgecolors=(0, 0, 0, 0.25),
        # Inner grid line width
        linewidth=0.6
    )

    # =====================================================
    # mark insufficient-sample bins
    # =====================================================

    for iy in range(nrows):

        for ix in range(ncols):

            if np.isnan(matrix[iy, ix]):

                ax.text(
                    ix + 0.5,
                    iy + 0.5,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black"
                )

    # =====================================================
    # ticks at bin edges
    # =====================================================

    ax.set_xticks(
        np.arange(1, ncols + 1)
    )

    ax.set_yticks(
        np.arange(1, nrows + 1)
    )

    # =====================================================
    # simplified labels
    # =====================================================

    xlabels = simplify_interval_labels(
        pivot.columns
    )

    ylabels = simplify_interval_labels(
        pivot.index
    )

    ax.set_xticklabels(
        xlabels,
        rotation=0,
        ha="center",
        # X-axis tick label font size
        fontsize=8
    )

    ax.set_yticklabels(
        ylabels,
        # Y-axis tick label font size
        fontsize=8
    )

    # =====================================================
    # square cells
    # =====================================================

    ax.set_aspect("equal")

    # =====================================================
    # limits
    # =====================================================

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # =====================================================
    # labels
    # =====================================================

    ax.set_xlabel(
        "NDVI",
        # X-axis title font size
        fontsize=9
    )

    ax.set_title(
        title,
        # Subplot title font size
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    # =====================================================
    # ticks outward
    # =====================================================

    ax.tick_params(
        axis="x",
        direction="out",
        width=0.8,
        # X-axis tick label distance from ticks
        pad=3
    )

    ax.tick_params(
        axis="y",
        direction="out",
        width=0.8,
        # Y-axis tick label distance from ticks
        pad=3
    )

    # =====================================================
    # subplot outer border
    # =====================================================

    for spine in ax.spines.values():

        # Rectangular subplot border visibility
        spine.set_visible(True)
        # Rectangular subplot border linewidth
        spine.set_linewidth(0.5)
        spine.set_color("black")

    return im


# =========================================================
# 8. DRAW POLAR ASPECT HEATMAP
# =========================================================

def draw_aspect_polar(
    ax,
    df,
    title,
    cmap="BrBG"
):

    aspect_order = [
        "N", "NE", "E", "SE",
        "S", "SW", "W", "NW"
    ]

    ndvi_bins = sorted(
        df["ndvi_bin"].unique()
    )

    matrix = np.full(
        (len(ndvi_bins), len(aspect_order)),
        np.nan
    )

    for i, ndvi_bin in enumerate(ndvi_bins):

        for j, aspect_bin in enumerate(aspect_order):

            sub = df[
                (df["ndvi_bin"] == ndvi_bin)
                &
                (df["aspect_bin"] == aspect_bin)
            ]

            if len(sub) > 0:

                matrix[i, j] = np.median(
                    sub["sensitivity"]
                )

    theta_edges = np.linspace(
        0,
        2 * np.pi,
        len(aspect_order) + 1
    )

    r_edges = np.arange(
        0,
        len(ndvi_bins) + 1
    )

    Theta, R = np.meshgrid(
        theta_edges,
        r_edges
    )

    pcm = ax.pcolormesh(
        Theta,
        R,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        # Polar inner grid line color
        edgecolors=(0, 0, 0, 0.1),
        # Polar inner grid line width
        linewidth=0.6
    )

    # =====================================================
    # mark insufficient-sample bins
    # =====================================================

    theta_centers = (
        theta_edges[:-1]
        + theta_edges[1:]
    ) / 2

    r_centers = (
        r_edges[:-1]
        + r_edges[1:]
    ) / 2

    for ir, r_center in enumerate(r_centers):

        for it, theta_center in enumerate(theta_centers):

            if np.isnan(matrix[ir, it]):

                ax.text(
                    theta_center,
                    r_center,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black"
                )

    # =====================================================
    # orientation
    # =====================================================

    ax.set_theta_zero_location("N")

    ax.set_theta_direction(-1)

    # =====================================================
    # aspect labels
    # =====================================================

    ax.set_xticks(theta_centers)

    ax.set_xticklabels(
        aspect_order,
        # Polar aspect tick label font size
        fontsize=9
    )

    # =====================================================
    # NDVI labels
    # =====================================================

    ndvi_labels = simplify_interval_labels(
        ndvi_bins
    )

    ax.set_yticks(r_centers)

    ax.set_yticklabels(
        ndvi_labels,
        # Polar NDVI tick label font size
        fontsize=8
    )

    # =====================================================
    # title
    # =====================================================

    ax.set_title(
        title,
        # Polar subplot title font size
        fontsize=12,
        fontweight="bold",
        pad=14
    )

    # =====================================================
    # grid
    # =====================================================

    ax.grid(
        # Polar grid line color
        color="0.65",
        # Polar grid line width
        linewidth=0.5
    )

    # Polar outer border linewidth
    ax.spines["polar"].set_linewidth(0.8)

    return pcm


# =========================================================
# 9. FIGURE LAYOUT
# =========================================================

fig = plt.figure(
    # Overall figure size
    figsize=(10, 8)
)

gs = fig.add_gridspec(
    2,
    2,
    # Horizontal spacing between subplots
    wspace=0.22,
    # Vertical spacing between subplots
    hspace=0.30
)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])

ax3 = fig.add_subplot(gs[1, 0])

ax4 = fig.add_subplot(
    gs[1, 1],
    projection="polar"
)


# =========================================================
# 10. DRAW PANELS
# =========================================================

draw_2d_heatmap(
    ax=ax1,
    df=dfs["Elevation"],
    env_col="ele_bin",
    title="(a) Elevation"
)

draw_2d_heatmap(
    ax=ax2,
    df=dfs["Rainfall"],
    env_col="map_bin",
    title="(b) Rainfall",
    sort_y_bins=True
)

draw_2d_heatmap(
    ax=ax3,
    df=dfs["Slope"],
    env_col="slp_bin",
    title="(c) Slope"
)

draw_aspect_polar(
    ax=ax4,
    df=dfs["Aspect"],
    title="(d) Aspect"
)


# =========================================================
# 11. SHARED COLORBAR
# =========================================================

sm = plt.cm.ScalarMappable(
    cmap="BrBG",
    norm=norm
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=[ax1, ax2, ax3, ax4],
    # Colorbar length
    shrink=0.78,
    # Colorbar distance from subplots
    pad=0.03
)

cbar.set_label(
    r"Road sensitivity"
    "\n"
    r"$(-\Delta P / \Delta \log(1+\mathrm{distance}))$",
    # Colorbar title font size
    fontsize=10
)


# =========================================================
# 12. SAVE
# =========================================================

fig.savefig(
    output_figure,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    output_pdf,
    bbox_inches="tight"
)

print("\nSaved figure:")

print(output_figure)

plt.show()
