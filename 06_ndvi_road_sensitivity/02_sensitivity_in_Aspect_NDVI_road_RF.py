# -*- coding: utf-8 -*-

"""
Function:
1. Further stratify by NDVI within different aspect bins;
2. Train separate RFs within each aspect × NDVI subgroup;
3. Calculate road sensitivity using the independent perturbation amount of the current subgroup;
4. Draw a polar heatmap for NDVI × aspect;
5. Keep the RF and sensitivity workflow consistent with the original rainfall/elevation/slope scripts.

Polar heatmap:
- angle  -> aspect
- radius -> NDVI
- color  -> road sensitivity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import TwoSlopeNorm


# =========================================================
# 1. Parameter Settings
# =========================================================

csv_input_path = r"H:\Himalaya\RF_susceptibility\features_all.csv"

output_dir = r"H:\Himalaya\figure\NDVI_Aspect_Polar_RF"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(
    output_dir,
    "NDVI_Aspect_binned_RF_RoadSensitivity.csv"
)

output_png = os.path.join(
    output_dir,
    "NDVI_Aspect_polar_heatmap.png"
)

output_pdf = os.path.join(
    output_dir,
    "NDVI_Aspect_polar_heatmap.pdf"
)

force_recompute = False


col_label = "label"
col_aspect = "aspect"
col_ndvi = "NDVI"

col_road_raw = "dist_to_road"
col_road_loge = "loge_dist_to_road"


feature_cols = [
    "Annual_Mean",
    "i_mm_day",
    "e_mm",
    "aspect",
    "dist_to_fault",
    "dist_to_water",
    "elevation",
    "NDVI",
    "plan_curv",
    "profile_curv",
    "slope",
    "loge_dist_to_road"
]

road_min = 0
road_max = 2000

aspect_bin_edges = np.arange(0, 361, 45)

aspect_labels = [
    "N", "NE", "E", "SE",
    "S", "SW", "W", "NW"
]

ndvi_bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

min_samples_per_bin = 200
min_positive_per_bin = 80
min_negative_per_bin = 80

n_estimators = 500
max_depth = None
min_samples_leaf = 5

random_state = 42
n_jobs = -1

n_resamples = 20


# =========================================================
# 2. Downsampling
# =========================================================

def balance_binary_samples(
    df_in,
    label_col="label",
    random_state=42
):

    df_pos = df_in[df_in[label_col] == 1]
    df_neg = df_in[df_in[label_col] == 0]

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("One class is empty.")

    n_target = min(n_pos, n_neg)

    df_pos_bal = df_pos.sample(
        n=n_target,
        replace=False,
        random_state=random_state
    )

    df_neg_bal = df_neg.sample(
        n=n_target,
        replace=False,
        random_state=random_state
    )

    df_bal = pd.concat([df_pos_bal, df_neg_bal])

    df_bal = df_bal.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return df_bal


# =========================================================
# 3. sensitivity
# =========================================================

def estimate_bin_sensitivity_once(
    sub_df_balanced,
    delta_loge_road,
    rf_seed=42
):

    X = sub_df_balanced[feature_cols].copy()
    y = sub_df_balanced[col_label].values

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=rf_seed,
        n_jobs=n_jobs
    )

    rf.fit(X, y)

    p_base = rf.predict_proba(X)[:, 1]

    X_perturb = X.copy()

    X_perturb[col_road_loge] = (
        X_perturb[col_road_loge] + delta_loge_road
    )

    max_loge_road = np.log(road_max + 1)

    X_perturb[col_road_loge] = np.clip(
        X_perturb[col_road_loge],
        0,
        max_loge_road
    )

    p_perturb = rf.predict_proba(X_perturb)[:, 1]

    delta_p = (
        (p_perturb - p_base)
        / delta_loge_road
    )

    proximity_sens = -delta_p

    return np.median(proximity_sens)


# =========================================================
# 4. repeated resampling
# =========================================================

def estimate_bin_sensitivity_resampled(
    sub_df,
    delta_loge_road,
    n_resamples=20,
    base_seed=42
):

    sens_list = []

    n_total_after_list = []
    n_each_class_after_list = []

    for i in range(n_resamples):

        sample_seed = base_seed + i

        sub_bal = balance_binary_samples(
            sub_df,
            label_col=col_label,
            random_state=sample_seed
        )

        n_total_after = len(sub_bal)

        n_each_class_after = n_total_after // 2

        sens_i = estimate_bin_sensitivity_once(
            sub_df_balanced=sub_bal,
            delta_loge_road=delta_loge_road,
            rf_seed=sample_seed
        )

        sens_list.append(sens_i)

        n_total_after_list.append(n_total_after)
        n_each_class_after_list.append(n_each_class_after)

    return {
        "final_sensitivity": np.median(sens_list),
        "all_sensitivities": sens_list,
        "median_n_total_after": int(np.median(n_total_after_list)),
        "median_n_each_class_after": int(np.median(n_each_class_after_list))
    }


# =========================================================
# 5. Main Analysis
# =========================================================

def run_aspect_analysis():

    if os.path.exists(output_csv) and not force_recompute:

        print("Loading existing CSV...")

        res_df = pd.read_csv(output_csv)

        return res_df

    print("Loading data...")

    df = pd.read_csv(csv_input_path)

    required_cols = (
        [c for c in feature_cols if c != col_road_loge]
        + ["dist_to_road", col_label]
    )

    required_cols = list(dict.fromkeys(required_cols))

    df = df[required_cols].dropna().copy()

    # -------------------------
    # filtering
    # -------------------------

    df = df[
        (df[col_road_raw] >= road_min)
        &
        (df[col_road_raw] <= road_max)
        &
        (df[col_label].isin([0, 1]))
    ].copy()

    # -------------------------
    # log road distance
    # -------------------------

    df[col_road_loge] = np.log(
        df[col_road_raw] + 1
    )

    # =====================================================
    # aspect binning
    # =====================================================

    # ensure aspect within 0-360
    df[col_aspect] = df[col_aspect] % 360

    df["aspect_bin"] = pd.cut(
        df[col_aspect],
        bins=aspect_bin_edges,
        labels=aspect_labels,
        include_lowest=True,
        right=False
    )

    # =====================================================
    # NDVI binning
    # =====================================================

    df["ndvi_bin"] = pd.cut(
        df[col_ndvi],
        bins=ndvi_bin_edges,
        include_lowest=True
    )

    df = df.dropna(
        subset=["aspect_bin", "ndvi_bin"]
    ).copy()

    results = []

    group_cols = [
        "aspect_bin",
        "ndvi_bin"
    ]

    # =====================================================
    # group analysis
    # =====================================================

    for (aspect_bin, ndvi_bin), sub in df.groupby(group_cols):

        n_total_before = len(sub)

        n_pos_before = int(
            (sub[col_label] == 1).sum()
        )

        n_neg_before = int(
            (sub[col_label] == 0).sum()
        )

        # -------------------------
        # sample threshold
        # -------------------------

        if n_total_before < min_samples_per_bin:
            continue

        if (
            n_pos_before < min_positive_per_bin
            or
            n_neg_before < min_negative_per_bin
        ):
            continue

        # -------------------------
        # perturbation
        # -------------------------

        delta_loge_road = sub[col_road_loge].std()

        if (
            pd.isna(delta_loge_road)
            or
            delta_loge_road <= 0
        ):
            continue

        # -------------------------
        # RF sensitivity
        # -------------------------

        out = estimate_bin_sensitivity_resampled(
            sub_df=sub,
            delta_loge_road=delta_loge_road,
            n_resamples=n_resamples,
            base_seed=random_state
        )

        results.append({

            "aspect_bin": str(aspect_bin),

            "ndvi_bin": str(ndvi_bin),

            "sensitivity": out["final_sensitivity"],

            "delta_loge_road": delta_loge_road,

            "n_total_before": n_total_before,
            "n_pos_before": n_pos_before,
            "n_neg_before": n_neg_before,

            "n_total_after": out["median_n_total_after"],
            "n_pos_after": out["median_n_each_class_after"],
            "n_neg_after": out["median_n_each_class_after"]
        })

        print(
            f"{aspect_bin}, {ndvi_bin}: "
            f"sensitivity={out['final_sensitivity']:.6e}"
        )

    res_df = pd.DataFrame(results)

    res_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved CSV:\n{output_csv}")

    return res_df


# =========================================================
# 6. TRUE Polar Heatmap
# =========================================================

def plot_polar_heatmap(
    res_df,
    output_png,
    output_pdf=None,
    cmap="BrBG"
):

    # =====================================================
    # aspect / NDVI order
    # =====================================================

    aspect_order = aspect_labels

    ndvi_text = []

    for i in range(len(ndvi_bin_edges) - 1):

        left = ndvi_bin_edges[i]
        right = ndvi_bin_edges[i + 1]

        ndvi_text.append(
            f"{left:.1f}-{right:.1f}"
        )

    # =====================================================
    # build matrix
    # rows = NDVI
    # cols = aspect
    # =====================================================

    matrix = np.full(
        (len(ndvi_text), len(aspect_order)),
        np.nan
    )

    for i, ndvi_bin in enumerate(sorted(res_df["ndvi_bin"].unique())):

        for j, aspect_bin in enumerate(aspect_order):

            sub = res_df[
                (res_df["ndvi_bin"] == ndvi_bin)
                &
                (res_df["aspect_bin"] == aspect_bin)
            ]

            if len(sub) > 0:

                matrix[i, j] = sub[
                    "sensitivity"
                ].median()

    # =====================================================
    # polar edges
    # =====================================================

    theta_edges = np.linspace(
        0,
        2 * np.pi,
        len(aspect_order) + 1
    )

    r_edges = np.arange(
        0,
        len(ndvi_text) + 1
    )

    # =====================================================
    # meshgrid
    # IMPORTANT
    # =====================================================

    Theta, R = np.meshgrid(
        theta_edges,
        r_edges
    )

    # =====================================================
    # normalization
    # =====================================================

    vmax = np.nanmax(np.abs(matrix))

    if vmax == 0:
        vmax = 1e-6

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    # =====================================================
    # figure
    # =====================================================

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42
    })

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(projection="polar")
    )

    # =====================================================
    # polar orientation
    # =====================================================

    ax.set_theta_zero_location("N")

    ax.set_theta_direction(-1)

    # =====================================================
    # TRUE polar heatmap
    # =====================================================

    pcm = ax.pcolormesh(
        Theta,
        R,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="auto",
        edgecolors="black",
        linewidth=0.8
    )

    # =====================================================
    # aspect labels
    # =====================================================

    theta_centers = (
        theta_edges[:-1]
        + theta_edges[1:]
    ) / 2

    ax.set_xticks(theta_centers)

    ax.set_xticklabels(
        aspect_order,
        fontsize=12,
        fontweight="bold"
    )

    # =====================================================
    # NDVI labels
    # =====================================================

    r_centers = (
        r_edges[:-1]
        + r_edges[1:]
    ) / 2

    ax.set_yticks(r_centers)

    ax.set_yticklabels(
        ndvi_text,
        fontsize=10
    )

    # =====================================================
    # aesthetics
    # =====================================================

    ax.grid(
        color="gray",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6
    )

    ax.spines["polar"].set_linewidth(1.2)

    # =====================================================
    # colorbar
    # =====================================================

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        pad=0.12,
        shrink=0.82
    )

    cbar.set_label(
        r"Road sensitivity"
        "\n"
        r"$(-\Delta P / \Delta \log(1+\mathrm{distance}))$",
        fontsize=12
    )

    # =====================================================
    # save
    # =====================================================

    plt.tight_layout()

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight"
    )

    if output_pdf is not None:

        fig.savefig(
            output_pdf,
            bbox_inches="tight"
        )

    print(f"\nSaved PNG:\n{output_png}")

    plt.show()


# =========================================================
# 7. run
# =========================================================

res_df = run_aspect_analysis()

print("\nValid bins:\n")

print(res_df)

plot_polar_heatmap(
    res_df,
    output_png=output_png,
    output_pdf=output_pdf,
    cmap="BrBG"
)