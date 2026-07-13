# -*- coding: utf-8 -*-

# Features:
# 1. Further stratify by NDVI within different slope bins;
# 2. Train independent RF models within each slope x NDVI subgroup and calculate road sensitivity using that subgroup's specific perturbations;
# 3. If there's already an output CSV, just read the results and plot, avoiding retraining the RF;

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import TwoSlopeNorm


# =========================
# 1. Parameter Settings
# =========================

csv_input_path = r"H:\Himalaya\RF_susceptibility\features_all.csv"

output_dir = r"H:\Himalaya\figure\NDVI_slope_2D_independent_RF"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "NDVI_slope_binned_RF_RoadSensitivity.csv")
output_heatmap_fig = os.path.join(output_dir, "NDVI_slope_RoadSensitivity_heatmap.png")
output_heatmap_pdf = os.path.join(output_dir, "NDVI_slope_RoadSensitivity_heatmap.pdf")

force_recompute = False

col_label = "label"
col_slp = "slope"
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

road_min, road_max = 0, 2000

n_slp_bins = 5
slp_bin_edges = [0, 12, 24, 36, 48, 60]
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


# =========================
# 2. General Helper Functions
# =========================

def extract_interval_numbers(label):
   
    nums = re.findall(r"[-+]?\d*\.?\d+", str(label))

    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), float(nums[0])
    else:
        return np.nan, np.nan


def interval_center(label):
    
    left, right = extract_interval_numbers(label)

    if np.isnan(left) or np.isnan(right):
        return np.nan

    return 0.5 * (left + right)


def sort_result_df(res_df):
   
    res_df = res_df.copy()

    if "slp_center" not in res_df.columns:
        res_df["slp_center"] = res_df["slp_bin"].apply(interval_center)

    if "ndvi_center" not in res_df.columns:
        if "ndvi_bin_left" in res_df.columns and "ndvi_bin_right" in res_df.columns:
            res_df["ndvi_center"] = 0.5 * (
                pd.to_numeric(res_df["ndvi_bin_left"], errors="coerce") +
                pd.to_numeric(res_df["ndvi_bin_right"], errors="coerce")
            )
        else:
            res_df["ndvi_center"] = res_df["ndvi_bin"].apply(interval_center)

    res_df["slp_center"] = pd.to_numeric(res_df["slp_center"], errors="coerce")
    res_df["ndvi_center"] = pd.to_numeric(res_df["ndvi_center"], errors="coerce")
    res_df["sensitivity"] = pd.to_numeric(res_df["sensitivity"], errors="coerce")

    res_df = res_df.dropna(subset=["slp_bin", "ndvi_bin", "ndvi_center"]).copy()

    res_df = res_df.sort_values(
        ["slp_center", "ndvi_center"]
    ).reset_index(drop=True)

    return res_df


# =========================
# 3. Downsampling and sensitivity calculation function
# =========================

def balance_binary_samples(df_in, label_col="label", random_state=42):
    
    df_pos = df_in[df_in[label_col] == 1]
    df_neg = df_in[df_in[label_col] == 0]

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("One class is empty, cannot balance to 1:1.")

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

    df_bal = pd.concat([df_pos_bal, df_neg_bal], axis=0)
    df_bal = df_bal.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return df_bal


def estimate_bin_sensitivity_once(
    sub_df_balanced,
    delta_loge_road,
    rf_seed=42,
    return_partial=False
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
    X_perturb[col_road_loge] = X_perturb[col_road_loge] + delta_loge_road

    max_loge_road = np.log(road_max + 1)

    X_perturb[col_road_loge] = np.clip(
        X_perturb[col_road_loge],
        0,
        max_loge_road
    )

    p_perturb = rf.predict_proba(X_perturb)[:, 1]

    delta_p_over_loge = (p_perturb - p_base) / delta_loge_road

    proximity_sens = -delta_p_over_loge

    sensitivity_value = np.median(proximity_sens)

    if return_partial:
        return sensitivity_value, proximity_sens
    else:
        return sensitivity_value


def estimate_bin_sensitivity_resampled(
    sub_df,
    n_resamples=20,
    base_seed=42,
    delta_loge_road=None
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
            rf_seed=sample_seed,
            delta_loge_road=delta_loge_road,
            return_partial=False
        )

        sens_list.append(sens_i)
        n_total_after_list.append(n_total_after)
        n_each_class_after_list.append(n_each_class_after)

    final_sensitivity = np.median(sens_list)

    return {
        "final_sensitivity": final_sensitivity,
        "all_sensitivities": sens_list,
        "median_n_total_after": int(np.median(n_total_after_list)),
        "median_n_each_class_after": int(np.median(n_each_class_after_list))
    }


# =========================
# 4. Plotting Function: 2D Heatmap
# =========================

def plot_ndvi_slope_sensitivity_heatmap(
    res_df,
    output_png,
    output_pdf=None,
    x_bin_col="ndvi_bin",
    y_bin_col="slp_bin",
    value_col="sensitivity",
    x_edges=None,
    y_edges=None,
    x_label="NDVI",
    y_label="Slope (°)",
    cbar_label=r"Road sensitivity" "\n" r"$(-\Delta P / \Delta \log(1+\mathrm{distance}))$",
    title="NDVI-slope road sensitivity",
    annotate=False,
    cmap="BrBG",
    symmetric_color=True
):

    required_cols = [x_bin_col, y_bin_col, value_col]
    missing_cols = [c for c in required_cols if c not in res_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for heatmap: {missing_cols}")

    df_hm = res_df[required_cols].dropna(subset=[x_bin_col, y_bin_col]).copy()
    if len(df_hm) == 0:
        raise ValueError("No valid rows for heatmap.")

    x_bins = sorted(df_hm[x_bin_col].unique(), key=interval_center)
    y_bins = sorted(df_hm[y_bin_col].unique(), key=interval_center)

    heatmap_matrix = df_hm.pivot_table(
        index=y_bin_col,
        columns=x_bin_col,
        values=value_col,
        aggfunc="median"
    )

    heatmap_matrix = heatmap_matrix.reindex(index=y_bins, columns=x_bins)
    Z = heatmap_matrix.values.astype(float)

    if np.all(np.isnan(Z)):
        raise ValueError("All heatmap values are NaN.")

    Z_masked = np.ma.masked_invalid(Z)

    n_y, n_x = Z.shape

    if x_edges is None:
        x_edges = np.array(ndvi_bin_edges, dtype=float)
    else:
        x_edges = np.array(x_edges, dtype=float)

    if len(x_edges) != n_x + 1:
        raise ValueError(
            f"x_edges length must be n_x + 1 = {n_x + 1}, "
            f"but got {len(x_edges)}"
        )

    if y_edges is None:
        y_edges = np.linspace(0, 100, n_y + 1)
    else:
        y_edges = np.array(y_edges, dtype=float)

    if len(y_edges) != n_y + 1:
        raise ValueError(
            f"y_edges length must be n_y + 1 = {n_y + 1}, "
            f"but got {len(y_edges)}"
        )

    if symmetric_color:
        vmax = np.nanmax(np.abs(Z))
        if vmax == 0:
            vmax = 1e-6
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    X, Y = np.meshgrid(x_edges, y_edges)

    im = ax.pcolormesh(
        X, Y, Z_masked,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=(0, 0, 0, 0.45),
        linewidth=0.45
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.set_xticks(x_edges)
    ax.set_xticklabels([f"{x:.1f}".rstrip("0").rstrip(".") for x in x_edges])

    ax.set_yticks(y_edges)
    ax.set_yticklabels([f"{int(round(y))}" for y in y_edges])

    ax.tick_params(direction="out", length=3.5, width=0.8)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    for iy in range(n_y):
        for ix in range(n_x):
            if np.isnan(Z[iy, ix]):
                x0, x1 = x_edges[ix], x_edges[ix + 1]
                y0, y1 = y_edges[iy], y_edges[iy + 1]

                ax.fill(
                    [x0, x1, x1, x0],
                    [y0, y0, y1, y1],
                    facecolor="white",
                    edgecolor=(0, 0, 0, 0.45),
                    linewidth=0.45,
                    zorder=3
                )

                ax.text(
                    x_centers[ix],
                    y_centers[iy],
                    "×",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    zorder=4
                )

    if annotate:

        for iy, yc in enumerate(y_centers):
            for ix, xc in enumerate(x_centers):
                val = Z[iy, ix]
                if np.isfinite(val):
                    ax.text(
                        xc,
                        yc,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black"
                    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=8, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.035
    )

    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(length=3.5, width=0.8)

    plt.tight_layout()

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")

    print(f"Saved heatmap PNG: {output_png}")
    if output_pdf is not None:
        print(f"Saved heatmap PDF: {output_pdf}")

    plt.show()


# =========================
# 5. slope x NDVI sensitivity calculation function
# =========================

def run_slope_analysis():
   
    print("\n" + "=" * 80)
    print(f"Running analysis: NDVI_slope ({col_slp})")
    print("=" * 80)

    if os.path.exists(output_csv) and not force_recompute:

        print(f"\nDetected existing result CSV:")
        print(output_csv)
        print("Skip RF sensitivity calculation and directly load results for plotting...")

        res_df = pd.read_csv(output_csv)

        required_result_cols = [
            "slp_bin",
            "ndvi_bin",
            "sensitivity"
        ]

        missing_result_cols = [
            c for c in required_result_cols
            if c not in res_df.columns
        ]

        if missing_result_cols:
            raise ValueError(
                f"Existing CSV is missing required columns: {missing_result_cols}. "
                f"Please set force_recompute = True to regenerate the result CSV."
            )

        numeric_cols = [
            "slp_center",
            "ndvi_center",
            "sensitivity",
            "delta_loge_road",
            "n_total_before",
            "n_pos_before",
            "n_neg_before",
            "n_total_after",
            "n_pos_after",
            "n_neg_after"
        ]

        for c in numeric_cols:
            if c in res_df.columns:
                res_df[c] = pd.to_numeric(res_df[c], errors="coerce")

        res_df = sort_result_df(res_df)

        if "slp_bin_left" in res_df.columns and "slp_bin_right" in res_df.columns:
            slp_edges_used = np.array(
                sorted(
                    set(pd.to_numeric(res_df["slp_bin_left"], errors="coerce").dropna().tolist()) |
                    set(pd.to_numeric(res_df["slp_bin_right"], errors="coerce").dropna().tolist())
                ),
                dtype=float
            )
        else:
            slp_edges_used = None

        print(f"Loaded {len(res_df)} valid slope x NDVI sensitivity records.")
        return res_df, slp_edges_used

    print("Loading data...")

    if not os.path.exists(csv_input_path):
        raise FileNotFoundError(f"Missing file: {csv_input_path}")

    df = pd.read_csv(csv_input_path)

    required_cols = [
        c for c in feature_cols
        if c != col_road_loge
    ] + [
        "dist_to_road",
        col_label
    ]

    required_cols = list(dict.fromkeys(required_cols))

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    df = df[required_cols].dropna().copy()

    df = df[
        (df[col_road_raw] >= road_min) &
        (df[col_road_raw] <= road_max) &
        (df[col_label].isin([0, 1]))
    ].copy()

    if len(df) < 300:
        raise ValueError("Too few valid samples after filtering.")

    df[col_road_loge] = np.log(df[col_road_raw] + 1)

    print(f"Remaining samples after filtering: {len(df)}")

    if slp_bin_edges is None:
        df["slp_bin"], slp_bin_edges_used = pd.qcut(
            df[col_slp],
            q=n_slp_bins,
            retbins=True,
            duplicates="drop"
        )

        slp_bin_edges_used = np.array(slp_bin_edges_used, dtype=float)

    else:
        slp_bin_edges_used = np.array(slp_bin_edges, dtype=float)

        if sorted(slp_bin_edges_used) != list(slp_bin_edges_used):
            raise ValueError("ele_bin_edges must be in ascending order.")

        if len(np.unique(slp_bin_edges_used)) != len(slp_bin_edges_used):
            raise ValueError("ele_bin_edges contain duplicated values.")

        df["slp_bin"] = pd.cut(
            df[col_slp],
            bins=slp_bin_edges_used,
            include_lowest=True
        )
 
    actual_slp_bins = df["slp_bin"].nunique()

    if actual_slp_bins < n_slp_bins:
        print(
            f"Warning: requested {n_slp_bins} slope bins, "
            f"but only {actual_slp_bins} bins were created. "
            f"This may be caused by duplicated values near quantile boundaries."
        )

    print("\nslope bin edges:")
    print(slp_bin_edges_used)

    print("\nslope bin sample counts:")
    print(df["slp_bin"].value_counts().sort_index())

    if sorted(ndvi_bin_edges) != list(ndvi_bin_edges):
        raise ValueError("ndvi_bin_edges must be in ascending order.")

    if len(np.unique(ndvi_bin_edges)) != len(ndvi_bin_edges):
        raise ValueError("ndvi_bin_edges contain duplicated values.")

    df["ndvi_bin"] = pd.cut(
        df[col_ndvi],
        bins=ndvi_bin_edges,
        include_lowest=True
    )

    print("\nNDVI bin edges:")
    print(ndvi_bin_edges)

    df = df.dropna(subset=["slp_bin", "ndvi_bin"]).copy()

    print(f"\nSamples after slope and NDVI binning: {len(df)}")

    results = []

    print(
        f"\nTraining Random Forest within each slope x NDVI bin "
        f"(1:1 downsampling repeated {n_resamples} times)..."
    )

    group_cols = ["slp_bin", "ndvi_bin"]

    for (slp_interval, ndvi_interval), sub in df.groupby(group_cols, observed=False):

        n_total_before = len(sub)
        n_pos_before = int((sub[col_label] == 1).sum())
        n_neg_before = int((sub[col_label] == 0).sum())

        if n_total_before < min_samples_per_bin:
            print(
                f"Skip slope {slp_interval}, NDVI {ndvi_interval}: "
                f"too few samples before balancing ({n_total_before})"
            )
            results.append({
                "analysis_name": "NDVI_slope",
                "slp_bin": str(slp_interval),
                "slp_bin_left": slp_interval.left,
                "slp_bin_right": slp_interval.right,
                "slp_center": 0.5 * (slp_interval.left + slp_interval.right),
                "ndvi_bin": str(ndvi_interval),
                "ndvi_bin_left": ndvi_interval.left,
                "ndvi_bin_right": ndvi_interval.right,
                "ndvi_center": 0.5 * (ndvi_interval.left + ndvi_interval.right),
                "sensitivity": np.nan,
                "delta_loge_road": np.nan,
                "n_total_before": n_total_before,
                "n_pos_before": n_pos_before,
                "n_neg_before": n_neg_before,
                "n_total_after": np.nan,
                "n_pos_after": np.nan,
                "n_neg_after": np.nan,
                "n_resamples": n_resamples,
                "all_sensitivities": np.nan,
                "skip_reason": "too_few_total_samples"
            })
            continue

        if n_pos_before < min_positive_per_bin or n_neg_before < min_negative_per_bin:
            print(
                f"Skip slope {slp_interval}, NDVI {ndvi_interval}: "
                f"too few positive/negative before balancing "
                f"({n_pos_before}/{n_neg_before})"
            )
            results.append({
                "analysis_name": "NDVI_slope",
                "slp_bin": str(slp_interval),
                "slp_bin_left": slp_interval.left,
                "slp_bin_right": slp_interval.right,
                "slp_center": 0.5 * (slp_interval.left + slp_interval.right),
                "ndvi_bin": str(ndvi_interval),
                "ndvi_bin_left": ndvi_interval.left,
                "ndvi_bin_right": ndvi_interval.right,
                "ndvi_center": 0.5 * (ndvi_interval.left + ndvi_interval.right),
                "sensitivity": np.nan,
                "delta_loge_road": np.nan,
                "n_total_before": n_total_before,
                "n_pos_before": n_pos_before,
                "n_neg_before": n_neg_before,
                "n_total_after": np.nan,
                "n_pos_after": np.nan,
                "n_neg_after": np.nan,
                "n_resamples": n_resamples,
                "all_sensitivities": np.nan,
                "skip_reason": "too_few_positive_or_negative_samples"
            })
            continue

        try:
            delta_loge_road_bin = sub[col_road_loge].std()

            if pd.isna(delta_loge_road_bin) or delta_loge_road_bin <= 0:
                print(
                    f"Skip slope {slp_interval}, NDVI {ndvi_interval}: "
                    f"invalid delta_loge_road_bin = {delta_loge_road_bin}"
                )
                results.append({
                    "analysis_name": "NDVI_slope",
                    "slp_bin": str(slp_interval),
                    "slp_bin_left": slp_interval.left,
                    "slp_bin_right": slp_interval.right,
                    "slp_center": 0.5 * (slp_interval.left + slp_interval.right),
                    "ndvi_bin": str(ndvi_interval),
                    "ndvi_bin_left": ndvi_interval.left,
                    "ndvi_bin_right": ndvi_interval.right,
                    "ndvi_center": 0.5 * (ndvi_interval.left + ndvi_interval.right),
                    "sensitivity": np.nan,
                    "delta_loge_road": delta_loge_road_bin,
                    "n_total_before": n_total_before,
                    "n_pos_before": n_pos_before,
                    "n_neg_before": n_neg_before,
                    "n_total_after": np.nan,
                    "n_pos_after": np.nan,
                    "n_neg_after": np.nan,
                    "n_resamples": n_resamples,
                    "all_sensitivities": np.nan,
                    "skip_reason": "invalid_delta_loge_road"
                })
                continue

            out = estimate_bin_sensitivity_resampled(
                sub_df=sub,
                n_resamples=n_resamples,
                base_seed=random_state,
                delta_loge_road=delta_loge_road_bin
            )

            slp_center = 0.5 * (slp_interval.left + slp_interval.right)
            ndvi_center = 0.5 * (ndvi_interval.left + ndvi_interval.right)

            results.append({
                "analysis_name": "NDVI_slope",

                "slp_bin": str(slp_interval),
                "slp_bin_left": slp_interval.left,
                "slp_bin_right": slp_interval.right,
                "slp_center": slp_center,

                "ndvi_bin": str(ndvi_interval),
                "ndvi_bin_left": ndvi_interval.left,
                "ndvi_bin_right": ndvi_interval.right,
                "ndvi_center": ndvi_center,

                "sensitivity": out["final_sensitivity"],
                "delta_loge_road": delta_loge_road_bin,

                "n_total_before": n_total_before,
                "n_pos_before": n_pos_before,
                "n_neg_before": n_neg_before,

                "n_total_after": out["median_n_total_after"],
                "n_pos_after": out["median_n_each_class_after"],
                "n_neg_after": out["median_n_each_class_after"],

                "n_resamples": n_resamples,
                "all_sensitivities": out["all_sensitivities"],
                "skip_reason": ""
            })

            print(
                f"slope {slp_interval}, NDVI {ndvi_interval}: "
                f"before={n_total_before} ({n_pos_before}/{n_neg_before}), "
                f"after~={out['median_n_total_after']} "
                f"({out['median_n_each_class_after']}/{out['median_n_each_class_after']}), "
                f"delta_loge_road={delta_loge_road_bin:.4f}, "
                f"sensitivity={out['final_sensitivity']:.6e}, "
                f"repeats={n_resamples}"
            )

        except Exception as e:
            print(
                f"Skipped slope {slp_interval}, NDVI {ndvi_interval} "
                f"due to fitting error: {e}"
            )
            continue

    res_df = pd.DataFrame(results)

    if len(res_df) == 0:
        raise RuntimeError("No valid slope x NDVI bins for analysis.")

    res_df = sort_result_df(res_df)

    res_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved CSV: {output_csv}")

    return res_df, slp_bin_edges_used


# =========================
# 6. Run slope x NDVI analysis and plot a 2D heatmap
# =========================

res_df, slp_edges_used = run_slope_analysis()

print("\nValid slope x NDVI bins used:")
print(res_df[[
    "slp_bin",
    "ndvi_bin",
    "slp_center",
    "ndvi_center",
    "sensitivity",
    "delta_loge_road",
    "n_total_before",
    "n_pos_before",
    "n_neg_before",
    "n_total_after",
    "n_pos_after",
    "n_neg_after",
    "n_resamples"
]])

plot_ndvi_slope_sensitivity_heatmap(
    res_df=res_df,
    output_png=output_heatmap_fig,
    output_pdf=output_heatmap_pdf,
    x_bin_col="ndvi_bin",
    y_bin_col="slp_bin",
    value_col="sensitivity",
    y_edges=slp_edges_used,
    x_label="NDVI",
    y_label="slope (°)",
    title="NDVI-slope road sensitivity",
    annotate=False,
    cmap="BrBG",
    symmetric_color=True
)




