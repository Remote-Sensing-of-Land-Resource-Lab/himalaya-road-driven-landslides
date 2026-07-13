# =========================================================
# trend_breakpoint_analysis.py
# Data-driven breakpoint figure for annual landslide counts
# =========================================================

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pymannkendall as mk
from statsmodels.nonparametric.smoothers_lowess import lowess


# =========================================================
# STYLE
# =========================================================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["xtick.major.width"] = 1.0
mpl.rcParams["ytick.major.width"] = 1.0
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# =========================================================
# DATA
# =========================================================
csv_file = (
    r"H:\Himalaya\figure\figure2总统计图"
    r"\13w_landslides_count_approx_95CI.csv"
)

output_dir = r"H:\Himalaya\figure\trend_breakpoint_analysis"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file).sort_values("year")

years = df["year"].to_numpy(dtype=float)
values = df["annual_landslide_count"].to_numpy(dtype=float)


# =========================================================
# PARAMETERS
# =========================================================
min_segment_length = 4
n_bootstrap = 2000
random_seed = 42


# =========================================================
# FUNCTIONS
# =========================================================
def linear_fit(x, y):
    """Fit y = a + bx and return fitted values and RSS."""
    x_centered = x - x.mean()
    design = np.column_stack([np.ones_like(x_centered), x_centered])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    rss = np.sum((y - fitted) ** 2)
    return fitted, rss, coef, x.mean()


def segmented_fit_for_index(x, y, break_index):
    """
    Fit two independent linear segments.

    break_index is the first observation of the second regime, so the
    reported breakpoint year denotes the first year after the shift.
    """
    fit_left, rss_left, coef_left, xmean_left = linear_fit(
        x[:break_index], y[:break_index]
    )
    fit_right, rss_right, coef_right, xmean_right = linear_fit(
        x[break_index:], y[break_index:]
    )

    fitted = np.empty_like(y)
    fitted[:break_index] = fit_left
    fitted[break_index:] = fit_right

    rss = rss_left + rss_right
    return fitted, rss, coef_left, xmean_left, coef_right, xmean_right


def bic_from_rss(rss, n_obs, n_params):
    rss = max(float(rss), 1e-10)
    return n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)


def evaluate_candidate_breaks(x, y, min_size):
    """Evaluate every admissible one-break two-segment linear model."""
    n_obs = len(y)
    rows = []

    for break_index in range(min_size, n_obs - min_size + 1):
        fitted, rss, *_ = segmented_fit_for_index(x, y, break_index)

        # Four regression coefficients plus one searched breakpoint location.
        bic = bic_from_rss(rss, n_obs, n_params=5)
        rows.append(
            {
                "break_index": break_index,
                "break_year": int(x[break_index]),
                "rss": rss,
                "bic": bic,
                "fitted": fitted,
            }
        )

    result = pd.DataFrame(rows)
    result["delta_bic"] = result["bic"] - result["bic"].min()
    return result


def predict_segment(coef, x_mean, x_new):
    return coef[0] + coef[1] * (x_new - x_mean)


def detect_best_break_year(x, y, min_size):
    candidate_df = evaluate_candidate_breaks(x, y, min_size)
    return int(candidate_df.loc[candidate_df["bic"].idxmin(), "break_year"])


# =========================================================
# TREND AND BREAKPOINT ANALYSIS
# =========================================================
mk_result = mk.hamed_rao_modification_test(values)

candidate_breaks = evaluate_candidate_breaks(years, values, min_segment_length)
best_row = candidate_breaks.loc[candidate_breaks["bic"].idxmin()]
best_break_index = int(best_row["break_index"])
best_break_year = int(best_row["break_year"])
best_fitted = np.asarray(best_row["fitted"], dtype=float)

null_fitted, null_rss, *_ = linear_fit(years, values)
null_bic = bic_from_rss(null_rss, len(values), n_params=2)
delta_bic_against_no_break = null_bic - float(best_row["bic"])

_, _, coef_left, xmean_left, coef_right, xmean_right = segmented_fit_for_index(
    years, values, best_break_index
)

x_left = np.linspace(years.min(), years[best_break_index - 1], 200)
x_right = np.linspace(years[best_break_index], years.max(), 200)
y_left = predict_segment(coef_left, xmean_left, x_left)
y_right = predict_segment(coef_right, xmean_right, x_right)

loess_smoothed = lowess(values, years, frac=0.35, return_sorted=True)


# =========================================================
# BOOTSTRAP BREAKPOINT UNCERTAINTY
# =========================================================
rng = np.random.default_rng(random_seed)
residuals = values - best_fitted

bootstrap_years = []
for _ in range(n_bootstrap):
    boot_values = best_fitted + rng.choice(residuals, size=len(residuals), replace=True)
    bootstrap_years.append(detect_best_break_year(years, boot_values, min_segment_length))

bootstrap_years = np.asarray(bootstrap_years, dtype=int)
ci_low, ci_high = np.percentile(bootstrap_years, [2.5, 97.5])
ci_low = int(np.floor(ci_low))
ci_high = int(np.ceil(ci_high))


# =========================================================
# FIGURE
# =========================================================
fig = plt.figure(figsize=(12, 8.5))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[2.2, 1.35],
    width_ratios=[1, 1],
    hspace=0.35,
    wspace=0.28,
)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])


# ---------------------------------------------------------
# (a) Time series and independently detected breakpoint
# ---------------------------------------------------------
ax1.scatter(years, values, s=28, color="black", zorder=4, label="Observed")
ax1.plot(years, values, color="0.75", lw=1.2, zorder=1)
ax1.plot(
    loess_smoothed[:, 0],
    loess_smoothed[:, 1],
    color="black",
    lw=2.4,
    zorder=3,
    label="LOESS",
)
ax1.plot(x_left, y_left, "--", color="firebrick", lw=2.2, label="Segmented fit")
ax1.plot(x_right, y_right, "--", color="firebrick", lw=2.2)

ax1.axvspan(ci_low, ci_high, color="firebrick", alpha=0.12, lw=0)
ax1.axvline(best_break_year, color="firebrick", ls=":", lw=1.8)

ax1.set_ylabel("Annual landslide count")
ax1.set_xlim(years.min() - 0.5, years.max() + 0.5)
ax1.legend(frameon=False, loc="upper left", ncol=3)

ax1.text(
    0.02,
    0.83,
    "Data-driven breakpoint: "
    f"{best_break_year}\n"
    f"Bootstrap 95% CI: {ci_low}-{ci_high}\n"
    f"MK p={mk_result.p:.3f}; Sen slope={mk_result.slope:.1f}",
    transform=ax1.transAxes,
    va="top",
)


# ---------------------------------------------------------
# (b) Candidate breakpoint model comparison
# ---------------------------------------------------------
ax2.plot(
    candidate_breaks["break_year"],
    candidate_breaks["delta_bic"],
    color="black",
    marker="o",
    ms=4,
    lw=1.6,
)
ax2.scatter(best_break_year, 0, color="firebrick", s=60, zorder=4)
ax2.axvline(best_break_year, color="firebrick", ls=":", lw=1.3)

for y_ref in [2, 6, 10]:
    ax2.axhline(y_ref, color="0.82", ls="--", lw=0.9, zorder=0)

ax2.set_xlabel("Candidate breakpoint year")
ax2.set_ylabel("Delta BIC")
ax2.set_title("Model comparison across candidate break years", fontsize=11)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.text(
    0.04,
    0.94,
    f"No-break BIC - best one-break BIC = {delta_bic_against_no_break:.1f}",
    transform=ax2.transAxes,
    va="top",
)


# ---------------------------------------------------------
# (c) Bootstrap distribution of detected breakpoint years
# ---------------------------------------------------------
bins = np.arange(bootstrap_years.min() - 0.5, bootstrap_years.max() + 1.5, 1)
ax3.hist(
    bootstrap_years,
    bins=bins,
    color="0.35",
    edgecolor="white",
    linewidth=0.8,
)
ax3.axvspan(ci_low, ci_high, color="firebrick", alpha=0.12, lw=0)
ax3.axvline(best_break_year, color="firebrick", ls=":", lw=1.8)

ax3.set_xlabel("Detected breakpoint year")
ax3.set_ylabel("Bootstrap frequency")
ax3.set_title("Breakpoint stability from residual bootstrap", fontsize=11)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax3.tick_params(axis="x", labelrotation=0)


# ---------------------------------------------------------
# Panel labels and cleanup
# ---------------------------------------------------------
for label, ax in zip(["a", "b", "c"], [ax1, ax2, ax3]):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )

for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# =========================================================
# SAVE
# =========================================================
plt.savefig(
    os.path.join(output_dir, "breakpoint_evidence_3panel.png"),
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    os.path.join(output_dir, "breakpoint_evidence_3panel.pdf"),
    bbox_inches="tight",
)

plt.show()
