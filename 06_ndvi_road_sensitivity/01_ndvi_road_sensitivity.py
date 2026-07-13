# ============================================================#
# Function summary:
# 1. Load the input feature table and filter samples by NDVI and road-distance ranges.
# 2. Create NDVI-based bins and retain only bins with enough positive/negative samples.
# 3. Within each valid NDVI bin, repeatedly perform 1:1 random downsampling 20 times.
# 4. For each resample, train a Random Forest classifier using the selected features.
# 5. Perturb the log-transformed road-distance variable (loge(dist + 1)) by a bin-specific step,
#    compute the probability change, and derive a road-proximity sensitivity value.
# 6. Aggregate the 20 sensitivity estimates per bin by taking the median and a t-based
#    confidence interval, then plot the NDVI-road sensitivity curve with uncertainty shading.
# ============================================================

import os
import sys
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t


# ============================================================
# 1. Parameter Settings
# ============================================================

csv_input_path = r"H:\Himalaya\RF_susceptibility\features_all.csv"

output_fig = r"H:\Himalaya\figure\NDVI_binned_RF_RoadSensitivity_loge_1to1_20.png"
output_csv = r"H:\Himalaya\figure\NDVI_binned_RF_RoadSensitivity_loge_1to1_20.csv"

# If True, retrain the RF and overwrite the results even if output_csv already exists
# If False, read the existing output_csv first and plot directly
force_recompute = False

col_label = "label"
col_ndvi = "NDVI"
col_road_raw = "dist_to_road"
col_road_loge = "loge_dist_to_road"

# Input features
feature_cols = [
    "Annual_Mean",
    "aspect",
    "dist_to_fault",
    "dist_to_water",
    "elevation",
    "NDVI",
    "plan_curv",
    "profile_curv",
    "slope",
    "loge_dist_to_road",
]

ndvi_min, ndvi_max = 0, 1
road_min, road_max = 0, 2000

n_bins = 9

# Minimum sample requirement for each bin before balancing
min_samples_per_bin = 400
min_positive_per_bin = 200
min_negative_per_bin = 200

n_resamples = 20

smooth_sigma = 1.0


# ============================================================
#2. Helper Functions
# ============================================================

def parse_sens_list(x):
    """
    Convert the all_sensitivities string saved in the CSV back to a list[float].
    """
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    try:
        v = ast.literal_eval(x)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [float(i) for i in v]
        return []
    except Exception:
        return []


# ============================================================
# 3. Plotting Functions
# ============================================================

def calc_t_confidence_interval(values, df=19):
    """
    Calculate mean +/- t(0.975, df) * SE confidence interval.
    """
    sens_arr = np.asarray(values, dtype=float)
    sens_arr = sens_arr[np.isfinite(sens_arr)]

    if len(sens_arr) <= 1:
        return np.nan, np.nan

    sens_mean = np.mean(sens_arr)
    sens_se = np.std(sens_arr, ddof=1) / np.sqrt(len(sens_arr))
    t_crit = t.ppf(0.975, df=df)

    sens_ci_low = sens_mean - t_crit * sens_se
    sens_ci_high = sens_mean + t_crit * sens_se

    return sens_ci_low, sens_ci_high


def plot_results(res_df):

    required_plot_cols = {"ndvi_center", "sensitivity"}
    missing_plot_cols = required_plot_cols - set(res_df.columns)
    if missing_plot_cols:
        raise ValueError(f"Result table is missing columns: {missing_plot_cols}")

    res_df = res_df.copy()
    res_df["ndvi_center"] = pd.to_numeric(res_df["ndvi_center"], errors="coerce")
    res_df["sensitivity"] = pd.to_numeric(res_df["sensitivity"], errors="coerce")
    res_df = res_df.dropna(subset=["ndvi_center", "sensitivity"]).copy()

    if len(res_df) < 4:
        raise RuntimeError("Too few valid NDVI bins for plotting.")

    res_df = res_df.sort_values("ndvi_center").reset_index(drop=True)

    # ------------------------------------------------------------
    # 3.1 Get or calculate the percentile range
    # ------------------------------------------------------------
    if "all_sensitivities" in res_df.columns:
        sens_lists = res_df["all_sensitivities"].apply(parse_sens_list)

        res_df["sens_ci_low"] = sens_lists.apply(
            lambda v: calc_t_confidence_interval(v, df=19)[0]
        )
        res_df["sens_ci_high"] = sens_lists.apply(
            lambda v: calc_t_confidence_interval(v, df=19)[1]
        )

    elif {"sens_ci_low", "sens_ci_high"}.issubset(res_df.columns):
        res_df["sens_ci_low"] = pd.to_numeric(res_df["sens_ci_low"], errors="coerce")
        res_df["sens_ci_high"] = pd.to_numeric(res_df["sens_ci_high"], errors="coerce")

    else:
        res_df["sens_ci_low"] = np.nan
        res_df["sens_ci_high"] = np.nan

    # ------------------------------------------------------------
    # 3.2 Creating smooth curves and shading
    # ------------------------------------------------------------
    x = res_df["ndvi_center"].values
    y = res_df["sensitivity"].values
    y_low = res_df["sens_ci_low"].values
    y_high = res_df["sens_ci_high"].values

    x_fine = np.linspace(x.min(), x.max(), 300)

    y_interp = np.interp(x_fine, x, y)
    y_smooth = gaussian_filter1d(y_interp, sigma=smooth_sigma)

    valid_ci = np.isfinite(y_low) & np.isfinite(y_high)

    if valid_ci.sum() >= 2:
        y_low_interp = np.interp(x_fine, x[valid_ci], y_low[valid_ci])
        y_high_interp = np.interp(x_fine, x[valid_ci], y_high[valid_ci])

        y_low_smooth = gaussian_filter1d(y_low_interp, sigma=smooth_sigma)
        y_high_smooth = gaussian_filter1d(y_high_interp, sigma=smooth_sigma)
    else:
        y_low_smooth = None
        y_high_smooth = None

    # ------------------------------------------------------------
    # 3.3 Drawing style
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 18,
        "axes.titlesize": 13,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    fig, (ax_sens, ax_count) = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [4.0, 1.0],
            "hspace": 0.08,
        },
    )
    main_color = "#C43C35"

    if y_low_smooth is not None and y_high_smooth is not None:
        ax_sens.fill_between(
            x_fine,
            y_low_smooth,
            y_high_smooth,
            color=main_color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    ax_sens.plot(
        x_fine,
        y_smooth,
        linewidth=2.5,
        color=main_color,
        zorder=2,
    )

    ax_sens.scatter(
        res_df["ndvi_center"],
        res_df["sensitivity"],
        s=65,
        alpha=0.9,
        color=main_color,
        edgecolor=main_color,
        linewidth=0.4,
        zorder=3,
    )

    ax_sens.axhline(
        0,
        color="#BBBBBB",
        lw=0.8,
        ls="--",
        alpha=1.0,
        zorder=0,
    )

    ax_sens.set_ylabel(
        r"Road sensitivity" "\n"
        r"$-\Delta P / \Delta \log(1+\mathrm{distance})$",
    )

    for spine in ["top", "right"]:
        ax_sens.spines[spine].set_visible(False)
        ax_count.spines[spine].set_visible(False)

    for axis in [ax_sens, ax_count]:
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        axis.tick_params(axis="both", width=0.8, length=4)

    ax_sens.tick_params(axis="x", labelbottom=False)
    ax_sens.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_sens.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Supporting sample-size strip for each NDVI bin.
    if "n_total_after" in res_df.columns:
        x_bins = res_df["ndvi_center"].to_numpy()
        n_after = pd.to_numeric(res_df["n_total_after"], errors="coerce").to_numpy()
        finite_x = x_bins[np.isfinite(x_bins)]
        if len(finite_x) >= 2:
            bar_width = np.median(np.diff(np.sort(finite_x))) * 0.65
        else:
            bar_width = 0.06

        ax_count.bar(
            x_bins,
            n_after,
            width=bar_width,
            color="#CFCFCF",
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

    ax_count.set_xlabel("NDVI")
    ax_count.set_ylabel("n")
    ax_count.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    legend_handles = [
        Patch(
            facecolor=main_color,
            edgecolor="none",
            alpha=0.16,
            label="95% CI",
        ),
        Line2D(
            [0],
            [0],
            color=main_color,
            marker="o",
            markersize=6,
            linewidth=2.5,
            markerfacecolor=main_color,
            markeredgecolor=main_color,
            label="Median sensitivity",
        ),
    ]

    ax_sens.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=15,
        handlelength=1.8,
        handletextpad=0.6,
        borderaxespad=0.4,
    )

    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0)

    plt.tight_layout(pad=0.8)

    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_fig}")

    plt.show()


# ============================================================
# 4. If there's already a results CSV, go ahead and plot it first
# ============================================================

if os.path.exists(output_csv) and not force_recompute:
    print(f"Existing result CSV found. Skip RF training and plot directly: {output_csv}")
    existing_res_df = pd.read_csv(output_csv, encoding="utf-8-sig")
    plot_results(existing_res_df)
    sys.exit(0)


# ============================================================
# 5. Reading and preprocessing
# ============================================================

print("Loading data...")

if not os.path.exists(csv_input_path):
    raise FileNotFoundError(f"Missing file: {csv_input_path}")

df = pd.read_csv(csv_input_path)

required_cols = [
    "Annual_Mean",
    "aspect",
    "dist_to_fault",
    "dist_to_water",
    "elevation",
    "NDVI",
    "plan_curv",
    "profile_curv",
    "slope",
    "dist_to_road",
    col_label,
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

df = df[required_cols].dropna().copy()

df = df[
    (df[col_ndvi] >= ndvi_min) &
    (df[col_ndvi] <= ndvi_max) &
    (df[col_road_raw] >= road_min) &
    (df[col_road_raw] <= road_max) &
    (df[col_label].isin([0, 1]))
].copy()

if len(df) < 300:
    raise ValueError("Too few valid samples after filtering.")

df[col_road_loge] = np.log(df[col_road_raw] + 1)

print(f"Remaining samples after filtering: {len(df)}")


# ============================================================
# 6. NDVI Binning
# ============================================================

bin_edges = [0.05, 0.15, 0.25, 0.35, 0.45,
             0.55, 0.65, 0.75, 0.85, 0.95]

df["ndvi_bin"] = pd.cut(df[col_ndvi], bins=bin_edges, include_lowest=True)


# ============================================================
# 7. Single 1:1 Downsampling Function
# ============================================================

def balance_binary_samples(df_in, label_col="label", random_state=42):
    """
    Randomly downsample the positive and negative samples to a 1:1 ratio within the current bin.
    """
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
        random_state=random_state,
    )

    df_neg_bal = df_neg.sample(
        n=n_target,
        replace=False,
        random_state=random_state,
    )

    df_bal = pd.concat([df_pos_bal, df_neg_bal], axis=0)
    df_bal = df_bal.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_bal


# ============================================================
# 8. Sensitivity under single resampling
# ============================================================

def estimate_bin_sensitivity_once(
    sub_df_balanced,
    delta_loge_road,
    rf_seed=42,
    return_partial=False,
):
    X = sub_df_balanced[feature_cols].copy()
    y = sub_df_balanced[col_label].values

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features="sqrt",
        random_state=rf_seed,
        n_jobs=-1,
        class_weight=None,
        oob_score=False,
    )

    rf.fit(X, y)

    p_base = rf.predict_proba(X)[:, 1]

    X_perturb = X.copy()
    X_perturb[col_road_loge] = X_perturb[col_road_loge] + delta_loge_road

    max_loge_road = np.log(road_max + 1)

    X_perturb[col_road_loge] = np.clip(
        X_perturb[col_road_loge],
        0,
        max_loge_road,
    )

    p_perturb = rf.predict_proba(X_perturb)[:, 1]

    delta_p_over_loge = (p_perturb - p_base) / delta_loge_road

    proximity_sens = -delta_p_over_loge

    sensitivity_value = np.median(proximity_sens)

    if return_partial:
        return sensitivity_value, proximity_sens

    return sensitivity_value

# ============================================================
# 9. Repeat 1:1 random downsampling 20 times within each NDVI bin
# ============================================================

def estimate_bin_sensitivity_resampled(
    sub_df,
    n_resamples=20,
    base_seed=42,
    delta_loge_road=None,
):
    sens_list = []
    n_total_after_list = []
    n_each_class_after_list = []

    for i in range(n_resamples):
        sample_seed = base_seed + i

        sub_bal = balance_binary_samples(
            sub_df,
            label_col=col_label,
            random_state=sample_seed,
        )

        n_total_after = len(sub_bal)
        n_each_class_after = n_total_after // 2

        sens_i = estimate_bin_sensitivity_once(
            sub_df_balanced=sub_bal,
            rf_seed=sample_seed,
            delta_loge_road=delta_loge_road,
            return_partial=False,
        )

        sens_list.append(sens_i)
        n_total_after_list.append(n_total_after)
        n_each_class_after_list.append(n_each_class_after)

    sens_arr = np.asarray(sens_list, dtype=float)

    final_sensitivity = np.median(sens_arr)
    sens_ci_low, sens_ci_high = calc_t_confidence_interval(sens_arr, df=19)

    return {
        "final_sensitivity": final_sensitivity,
        "sens_ci_low": sens_ci_low,
        "sens_ci_high": sens_ci_high,
        "all_sensitivities": sens_list,
        "median_n_total_after": int(np.median(n_total_after_list)),
        "median_n_each_class_after": int(np.median(n_each_class_after_list)),
    }


# ============================================================
# 10. Estimate sensitivity for each NDVI bin
# ============================================================

results = []

print(
    f"Training Random Forest within each NDVI bin "
    f"(1:1 downsampling repeated {n_resamples} times)..."
)

for bin_interval, sub in df.groupby("ndvi_bin", observed=True):

    n_total_before = len(sub)
    n_pos_before = int((sub[col_label] == 1).sum())
    n_neg_before = int((sub[col_label] == 0).sum())

    if n_total_before < min_samples_per_bin:
        print(f"Skip {bin_interval}: too few samples before balancing ({n_total_before})")
        continue

    if n_pos_before < min_positive_per_bin or n_neg_before < min_negative_per_bin:
        print(
            f"Skip {bin_interval}: too few positive/negative before balancing "
            f"({n_pos_before}/{n_neg_before})"
        )
        continue

    try:
        delta_loge_road_bin = sub[col_road_loge].std()

        if pd.isna(delta_loge_road_bin) or delta_loge_road_bin <= 0:
            print(f"Skip {bin_interval}: invalid delta_loge_road_bin = {delta_loge_road_bin}")
            continue

        out = estimate_bin_sensitivity_resampled(
            sub_df=sub,
            n_resamples=n_resamples,
            base_seed=42,
            delta_loge_road=delta_loge_road_bin,
        )

        ndvi_center = 0.5 * (bin_interval.left + bin_interval.right)

        results.append({
            "ndvi_center": ndvi_center,
            "sensitivity": out["final_sensitivity"],
            "sens_ci_low": out["sens_ci_low"],
            "sens_ci_high": out["sens_ci_high"],
            "n_total_before": n_total_before,
            "n_pos_before": n_pos_before,
            "n_neg_before": n_neg_before,
            "n_total_after": out["median_n_total_after"],
            "n_pos_after": out["median_n_each_class_after"],
            "n_neg_after": out["median_n_each_class_after"],
            "n_resamples": n_resamples,
            "all_sensitivities": out["all_sensitivities"],
            "delta_loge_road": delta_loge_road_bin,
        })

        print(
            f"{bin_interval}: "
            f"before={n_total_before} ({n_pos_before}/{n_neg_before}), "
            f"after~={out['median_n_total_after']} "
            f"({out['median_n_each_class_after']}/{out['median_n_each_class_after']}), "
            f"sensitivity={out['final_sensitivity']:.6e}, "
            f"range=[{out['sens_ci_low']:.6e}, {out['sens_ci_high']:.6e}], "
            f"repeats={n_resamples}"
        )

    except Exception as e:
        print(f"Skipped bin {bin_interval} due to fitting error: {e}")
        continue


# ============================================================
# 11. Save the results and plot
# ============================================================

res_df = pd.DataFrame(results)

if len(res_df) < 4:
    raise RuntimeError("Too few valid NDVI bins for plotting.")

res_df = res_df.sort_values("ndvi_center").reset_index(drop=True)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
res_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"Saved CSV: {output_csv}")

print("\nValid bins used:")
print(res_df[[
    "ndvi_center",
    "sensitivity",
    "sens_ci_low",
    "sens_ci_high",
    "delta_loge_road",
    "n_total_before",
    "n_pos_before",
    "n_neg_before",
    "n_total_after",
    "n_pos_after",
    "n_neg_after",
    "n_resamples",
]])

plot_results(res_df)
sys.exit(0)
