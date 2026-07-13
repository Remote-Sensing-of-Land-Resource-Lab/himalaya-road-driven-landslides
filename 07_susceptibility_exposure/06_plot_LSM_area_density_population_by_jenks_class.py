# -*- coding: utf-8 -*-
"""
Create grouped bar plots for:
1. Area by landslide susceptibility class
2. Mean annual landslide density by susceptibility class as point-line plots
3. Population exposure by susceptibility class

Designed as the lower row of a multi-panel figure:
a = susceptibility probability change map
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

ANALYSIS_DIR = r"H:\Himalaya\RF_susceptibility\susceptibility\c.analysis"

INPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "LSM_Landslide_Density_By_Jenks_Class.csv",
)

OUT_PNG = os.path.join(
    ANALYSIS_DIR,
    "Fig_LSM_area_density_population_by_jenks_class_nature_style.png",
)

OUT_PDF = os.path.join(
    ANALYSIS_DIR,
    "Fig_LSM_area_density_population_by_jenks_class_nature_style.pdf",
)


# ============================================================
# 2. Plot settings
# ============================================================

CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]
CLASS_LABELS = ["VL", "L", "M", "H", "VH"]

PERIOD_ORDER = ["2000-2019", "2020-2024"]

PERIOD_COLORS = {
    "2000-2019": "#6E97B8",   # muted blue
    "2020-2024": "#D46A5A",   # muted red
}

ANNOTATE_CLASSES = ["High", "Very High"]

# ============================================================
# 3. functions
# ============================================================

def setup_plot_style():
    """
    Set compact, publication-style figure parameters.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

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


def format_axis(ax):
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

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)


def add_relative_change_annotations(
    ax,
    df,
    value_col,
    scale=1.0,
    x_offset=0.0,
    normalize_by_period_total=False,
):
    x = np.arange(len(CLASS_ORDER), dtype=float)

    value_lookup = (
        df.pivot_table(
            index="Level",
            columns="Period",
            values=value_col,
            aggfunc="first"
        )
        .reindex(CLASS_ORDER)
    )
    if normalize_by_period_total:
        value_lookup = value_lookup.div(value_lookup.sum(axis=0), axis=1) * 100.0

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.12 if y_max > 0 else 1)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_offset = y_range * 0.035

    for class_name in ANNOTATE_CLASSES:
        if class_name not in value_lookup.index:
            continue

        pre = value_lookup.loc[class_name, PERIOD_ORDER[0]]
        post = value_lookup.loc[class_name, PERIOD_ORDER[1]]

        if pd.isna(pre) or pd.isna(post) or pre == 0:
            continue

        rel_change = (post - pre) / pre * 100.0

        class_idx = CLASS_ORDER.index(class_name)
        x_pos = x[class_idx] + x_offset
        y_pos = max(pre, post) / scale

        label = f"{rel_change:+.1f}%"
        label_color = "#A91522" if rel_change > 0 else "#2B6CB0"
        if rel_change == 0:
            label_color = "#555555"

        ax.text(
            x_pos,
            y_pos + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=label_color,
            fontweight="bold",
            clip_on=False,
        )


def add_line_change_annotations(ax, df, value_col, scale=1.0):
    """
    Annotate relative changes close to the 2020-2024 line points.
    """
    x = np.arange(len(CLASS_ORDER), dtype=float)

    value_lookup = (
        df.pivot_table(
            index="Level",
            columns="Period",
            values=value_col,
            aggfunc="first"
        )
        .reindex(CLASS_ORDER)
    )

    scaled_values = value_lookup[PERIOD_ORDER] / scale
    data_min = np.nanmin(scaled_values.to_numpy())
    data_max = np.nanmax(scaled_values.to_numpy())
    data_range = data_max - data_min
    if data_range <= 0:
        data_range = data_max if data_max > 0 else 1.0

    ax.set_ylim(
        max(0, data_min - data_range * 0.20),
        data_max + data_range * 0.26,
    )
    label_offset = data_range * 0.055

    for class_name in ANNOTATE_CLASSES:
        if class_name not in value_lookup.index:
            continue

        class_idx = CLASS_ORDER.index(class_name)
        pre = value_lookup.loc[class_name, PERIOD_ORDER[0]]
        post = value_lookup.loc[class_name, PERIOD_ORDER[1]]

        if pd.isna(pre) or pd.isna(post) or pre == 0:
            continue

        rel_change = (post - pre) / pre * 100.0
        label = f"{rel_change:+.1f}%"
        label_color = "#A91522" if rel_change > 0 else "#2B6CB0"
        if rel_change == 0:
            label_color = "#555555"

        point_y = post / scale
        if rel_change >= 0:
            label_y = point_y + label_offset
            va = "bottom"
        else:
            label_y = point_y - label_offset
            va = "top"

        ax.text(
            x[class_idx],
            label_y,
            label,
            ha="center",
            va=va,
            fontsize=8.6,
            color=label_color,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 0.35,
                "alpha": 0.85,
            },
            clip_on=False,
            zorder=5,
        )


# ============================================================
# 4. Data loading
# ============================================================

def load_results(csv_path):
    """
    Load and check analysis results.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Please run the LSM area-density-population analysis first."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {
        "Period",
        "Level",
        "Area_km2",
        "Annual_LS_Density(per 100km2 yr)",
        "Population_Million",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    df = df[
        df["Period"].isin(PERIOD_ORDER)
        & df["Level"].isin(CLASS_ORDER)
    ].copy()

    df["Level"] = pd.Categorical(
        df["Level"],
        categories=CLASS_ORDER,
        ordered=True,
    )
    df["Period"] = pd.Categorical(
        df["Period"],
        categories=PERIOD_ORDER,
        ordered=True,
    )

    df = df.sort_values(["Level", "Period"]).reset_index(drop=True)

    return df


# ============================================================
# 5. Plotting functions
# ============================================================

def get_values_by_class(
    df,
    period,
    value_col,
    scale=1.0,
    normalize_by_period_total=False,
):
    """
    Return values ordered by susceptibility class.
    """
    period_df = df[df["Period"] == period].set_index("Level")
    period_total = period_df[value_col].sum()

    values = []
    for level in CLASS_ORDER:
        if level in period_df.index:
            value = period_df.loc[level, value_col]
            if normalize_by_period_total:
                if period_total == 0:
                    values.append(np.nan)
                    continue
                value = value / period_total * 100.0
            values.append(value / scale)
        else:
            values.append(np.nan)

    return np.array(values, dtype=float)


def plot_grouped_bars(
    ax,
    df,
    value_col,
    ylabel,
    scale=1.0,
    annotate_change=True,
    normalize_by_period_total=False,
):
    """
    Plot Nature-style grouped bars.
    """
    x = np.arange(len(CLASS_ORDER), dtype=float)
    width = 0.32

    offsets = {
        PERIOD_ORDER[0]: -width / 2,
        PERIOD_ORDER[1]: width / 2,
    }

    for period in PERIOD_ORDER:
        values = get_values_by_class(
            df=df,
            period=period,
            value_col=value_col,
            scale=scale,
            normalize_by_period_total=normalize_by_period_total,
        )

        ax.bar(
            x + offsets[period],
            values,
            width=width,
            color=PERIOD_COLORS[period],
            edgecolor="none",
            linewidth=0,
            label=period,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_xlabel("LSI class")
    ax.set_ylabel(ylabel)

    format_axis(ax)

    if annotate_change:
        add_relative_change_annotations(
            ax=ax,
            df=df,
            value_col=value_col,
            scale=scale,
            x_offset=0.0,
            normalize_by_period_total=normalize_by_period_total,
        )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.8, 0.8),
        frameon=False,
        ncol=1,
        handlelength=1.1,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def plot_point_lines(
    ax,
    df,
    value_col,
    ylabel,
    scale=1.0,
    annotate_change=True,
):
    """
    Plot period values as point-line series by susceptibility class.
    """
    x = np.arange(len(CLASS_ORDER), dtype=float)

    for period in PERIOD_ORDER:
        values = get_values_by_class(
            df=df,
            period=period,
            value_col=value_col,
            scale=scale,
        )

        ax.plot(
            x,
            values,
            color=PERIOD_COLORS[period],
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=PERIOD_COLORS[period],
            markeredgewidth=0.8,
            linewidth=1.05,
            label=period,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_xlabel("LSI class")
    ax.set_ylabel(ylabel)

    format_axis(ax)
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(
        max(0, ymin - y_range * 0.08),
        ymax + y_range * 0.12,
    )
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    if annotate_change:
        add_line_change_annotations(
            ax=ax,
            df=df,
            value_col=value_col,
            scale=scale,
        )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.3, 0.8),
        frameon=False,
        ncol=1,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def create_figure(df):
    """
    Create the three-panel lower-row figure.
    """
    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.7, 2.85),
        sharex=False,
        constrained_layout=False,
    )

    # ------------------------------------------------------------
    # b. Area
    # ------------------------------------------------------------
    plot_grouped_bars(
        axes[0],
        df,
        value_col="Area_km2",
        ylabel="Area proportion (%)",
        scale=1.0,
        annotate_change=True,
        normalize_by_period_total=True,
    )

    # ------------------------------------------------------------
    # c. Mean annual landslide density
    # ------------------------------------------------------------
    plot_point_lines(
        axes[1],
        df,
        value_col="Annual_LS_Density(per 100km2 yr)",
        ylabel="Landslide density\n(events 100 km$^{-2}$ yr$^{-1}$)",
        scale=1.0,
        annotate_change=True,
    )

    # ------------------------------------------------------------
    # d. Population exposure
    # ------------------------------------------------------------
    plot_grouped_bars(
        axes[2],
        df,
        value_col="Population_Million",
        ylabel="Population exposed\n(million)",
        scale=1.0,
        annotate_change=True,
    )

    fig.align_ylabels(axes)

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.24,
        top=0.80,
        wspace=0.45,
    )

    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        OUT_PDF,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved figure: {OUT_PDF}")


# ============================================================
# 6. Main
# ============================================================

def main():
    df = load_results(INPUT_CSV)
    create_figure(df)


if __name__ == "__main__":
    main()
