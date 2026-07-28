# -*- coding: utf-8 -*-
"""
Calculate adjusted annual landslide counts with approximate 95% CI directly
from the final landslide inventory, then plot the count time series.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =========================================================
# 1. Input / output paths
# =========================================================
input_csv = r"H:\Himalaya\13w_landslides_list_final.csv"

output_dir = r"H:\Himalaya\figure"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "13w_landslides_count_approx_95CI.csv")
output_fig = os.path.join(output_dir, "frequency_count_CI_trend.png")


# =========================================================
# 2. Validation sample information, same as original logic
# =========================================================
# mapped landslide stratum:
tp = 872   # mapped landslide & reference landslide
fp = 128   # mapped landslide & reference non-landslide

# mapped non-landslide stratum:
fn = 1     # mapped non-landslide & reference landslide
tn = 999   # mapped non-landslide & reference non-landslide


# =========================================================
# 3. Build annual mapped landslide counts directly from final inventory
# =========================================================
df = pd.read_csv(input_csv)

required_cols = {"year"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df.copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

annual = (
    df.groupby("year", as_index=False)
    .size()
    .rename(columns={"size": "annual_landslide_count"})
    .sort_values("year")
    .reset_index(drop=True)
)

mapped_counts = annual["annual_landslide_count"].to_numpy(dtype=float)


# =========================================================
# 4. Sample-based point estimate, same as original logic
# =========================================================
ua = tp / (tp + fp)          # sample-based User's accuracy -- landslide
pa = tp / (tp + fn)          # sample-based Producer's accuracy -- landslide
correction_factor = ua / pa

annual["count_estimate"] = annual["annual_landslide_count"] * correction_factor


# =========================================================
# 5. Bootstrap approximate 95% CI, same as original logic
# =========================================================
B = 10000
rng = np.random.default_rng(42)

# UA bootstrap vector: mapped landslide stratum
# 1 = true landslide, 0 = false landslide
ua_labels = np.array([1] * tp + [0] * fp)

# PA bootstrap vector: reference landslide class
# 1 = correctly detected, 0 = omission
pa_labels = np.array([1] * tp + [0] * fn)

ua_boot = np.empty(B, dtype=float)
pa_boot = np.empty(B, dtype=float)
cf_boot = np.empty(B, dtype=float)

n_ua = len(ua_labels)   # 1000
n_pa = len(pa_labels)   # 873

for b in range(B):
    ua_sample = rng.choice(ua_labels, size=n_ua, replace=True)
    pa_sample = rng.choice(pa_labels, size=n_pa, replace=True)

    ua_b = ua_sample.mean()
    pa_b = pa_sample.mean()

    ua_boot[b] = ua_b
    pa_boot[b] = pa_b

    if pa_b <= 0:
        cf_boot[b] = np.nan
    else:
        cf_boot[b] = ua_b / pa_b

cf_boot = cf_boot[np.isfinite(cf_boot)]

if len(cf_boot) == 0:
    raise RuntimeError("Bootstrap failed: no valid correction factors.")

count_ci_low = []
count_ci_high = []

for n_mapped in mapped_counts:
    n_boot = n_mapped * cf_boot
    low = np.percentile(n_boot, 2.5)
    high = np.percentile(n_boot, 97.5)
    count_ci_low.append(low)
    count_ci_high.append(high)

annual["count_ci_low"] = count_ci_low
annual["count_ci_high"] = count_ci_high

annual["count_estimate_round"] = np.round(annual["count_estimate"]).astype(int)
annual["count_ci_low_round"] = np.round(annual["count_ci_low"]).astype(int)
annual["count_ci_high_round"] = np.round(annual["count_ci_high"]).astype(int)

annual["ua_landslide_unweighted"] = ua
annual["pa_landslide_unweighted"] = pa
annual["count_adjustment_factor"] = correction_factor

annual.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("Done calculating adjusted annual landslide counts.")
print(f"Output saved to:\n{output_csv}")
print(f"Sample-based UA (landslide): {ua:.6f} ({ua * 100:.2f}%)")
print(f"Sample-based PA (landslide): {pa:.6f} ({pa * 100:.2f}%)")
print(f"Count adjustment factor UA/PA: {correction_factor:.6f}")


# =========================================================
# 6. Plot, same visual logic as frequency_count_CL.py
# =========================================================
years = annual["year"].to_numpy(dtype=float)
y = annual["count_estimate"].to_numpy(dtype=float)
lower = annual["count_ci_low"].to_numpy(dtype=float)
upper = annual["count_ci_high"].to_numpy(dtype=float)

y_plot = y / 10000.0
lower_plot = lower / 10000.0
upper_plot = upper / 10000.0
yerr_plot = np.vstack([y_plot - lower_plot, upper_plot - y_plot])

x = years.astype(float)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.2

fig, ax = plt.subplots(figsize=(14, 5.6))

ax.errorbar(
    x,
    y_plot,
    yerr=yerr_plot,
    fmt="none",
    ecolor="gray",
    elinewidth=1.8,
    capsize=6,
    capthick=1.8,
    alpha=0.9,
    zorder=1,
)

ax.scatter(
    x,
    y_plot,
    s=40,
    color="red",
    edgecolor="lightcoral",
    linewidth=0,
    alpha=0.85,
    zorder=1,
)


def segmented_fit(ax, x_seg, y_seg, line_color="red", band_color="red", alpha=0.20, lw=3.0):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_seg, y_seg)
    y_fit = intercept + slope * x_seg

    n = len(x_seg)
    x_mean = np.mean(x_seg)
    ssx = np.sum((x_seg - x_mean) ** 2)
    resid = y_seg - y_fit
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    t_val = stats.t.ppf(0.975, df=n - 2)

    conf = t_val * s_err * np.sqrt(1 / n + (x_seg - x_mean) ** 2 / ssx)
    fit_lower = y_fit - conf
    fit_upper = y_fit + conf

    ax.plot(x_seg, y_fit, color=line_color, linewidth=lw, zorder=4)
    ax.fill_between(x_seg, fit_lower, fit_upper, color=band_color, alpha=alpha, zorder=2)

    slope_count = slope * 10000
    slope_low = (slope - t_val * std_err) * 10000
    slope_high = (slope + t_val * std_err) * 10000

    return {
        "slope": slope_count,
        "low": slope_low,
        "high": slope_high,
        "p": p_value,
    }


mask1 = (x >= 2000) & (x <= 2019)
mask2 = (x >= 2020) & (x <= 2024)

res1 = segmented_fit(
    ax,
    x[mask1],
    y_plot[mask1],
    line_color="#094773",
    band_color="#80b1d3",
    alpha=0.25,
    lw=2.8,
)

res2 = segmented_fit(
    ax,
    x[mask2],
    y_plot[mask2],
    line_color="#c42e1d",
    band_color="#fb8072",
    alpha=0.25,
    lw=3.2,
)


def fmt_slope(v):
    return f"{v:+.1f} yr$^{{-1}}$"


def fmt_p_exact(p):
    if p < 0.001:
        return f"{p:.2e}"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


label1 = f"2000-2019, beta = {fmt_slope(res1['slope'])}, P = {fmt_p_exact(res1['p'])}"
label2 = f"2020-2024, beta = {fmt_slope(res2['slope'])}, P = {fmt_p_exact(res2['p'])}"

x0 = 0.03
y0 = 0.97
line_gap = 0.075

ax.text(
    x0,
    y0,
    "-",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=25,
    color="#094773",
)
ax.text(
    x0 + 0.042,
    y0,
    label1,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    color="black",
)

ax.text(
    x0,
    y0 - line_gap,
    "-",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=25,
    color="#c42e1d",
)
ax.text(
    x0 + 0.042,
    y0 - line_gap,
    label2,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    color="black",
)

ax.set_ylabel("Annual Landslide Count ($\\times 10^4$)", fontsize=20)
ax.set_xlabel("Year", fontsize=20)

ax.set_xticks(np.arange(2000, 2025, 4))
ax.tick_params(axis="both", labelsize=17, width=1.2, length=6)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(1999.3, 2024.7)
ax.set_ylim(lower_plot.min() * 0.90, upper_plot.max() * 1.16)

plt.tight_layout()
plt.savefig(output_fig, dpi=600, bbox_inches="tight")
print(f"Figure saved to:\n{output_fig}")
plt.show()
