# Y: area
# X: susceptibility
# bin
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib import rcParams

tasks = [
    {
        "name": "2000-2019",
        "tif": r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map\LSM_2000_2019.tif",
        "color": "#2166AC",
    },
    {
        "name": "2020-2024",
        "tif": r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map\LSM_2020_2024.tif",
        "color": "#B2182B",
    }
]

out_dir = r"H:\Himalaya\RF_susceptibility\susceptibility\c.analysis"

# Continuous histogram parameters
value_min = 0.0
value_max = 1.0
n_bins_fine = 100
n_bins_main = 10
break_method = "period_specific_jenks_natural_breaks"

final_png = os.path.join(out_dir, "LSM_Area_Susceptibility_PublicationStyle.png")
final_pdf = os.path.join(out_dir, "LSM_Area_Susceptibility_PublicationStyle.pdf")

use_existing_csv_if_available = True


def set_publication_style():
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["font.size"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["axes.titlesize"] = 9
    rcParams["xtick.labelsize"] = 7
    rcParams["ytick.labelsize"] = 7
    rcParams["legend.fontsize"] = 7
    rcParams["axes.linewidth"] = 0.5
    rcParams["xtick.major.width"] = 0.5
    rcParams["ytick.major.width"] = 0.5
    rcParams["xtick.major.size"] = 3
    rcParams["ytick.major.size"] = 3
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["savefig.dpi"] = 600
    rcParams["figure.dpi"] = 200


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pixel_area_km2(src):
    res_x, res_y = src.res
    crs = src.crs

    if crs is not None and crs.is_projected:
        pixel_area_m2 = abs(res_x * res_y)
        return pixel_area_m2 / 1_000_000

    elif crs is not None and crs.is_geographic:
        print(" [Warning] Geographic CRS detected (lon/lat); area will be approximated.")
        print("           A more rigorous approach is to reproject to a projected CRS first.")
        pixel_area_km2 = abs(res_x * 111.32) * abs(res_y * 111.32)
        return pixel_area_km2

    else:
        raise ValueError("Unable to identify the raster CRS; pixel area cannot be calculated.")


def extract_valid_values_and_area(tif_path):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File does not exist: {tif_path}")

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata

        if nodata is not None:
            valid_mask = (data != nodata) & (~np.isnan(data))
        else:
            valid_mask = ~np.isnan(data)

        values = data[valid_mask]
        values = values[(values >= value_min) & (values <= value_max)]

        pixel_area_km2 = get_pixel_area_km2(src)

        print(f"\n>>> Reading: {tif_path}")
        print(f"    CRS: {src.crs}")
        print(f"    Resolution: {src.res}")
        print(f"    Valid pixel count: {len(values)}")
        print(f"    Pixel area: {pixel_area_km2:.8f} km²")

    return values, pixel_area_km2


def calc_fine_histogram(values, pixel_area_km2, n_bins=1000):
    edges = np.linspace(value_min, value_max, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    area_km2 = counts * pixel_area_km2

    df = pd.DataFrame({
        "bin_id_1000": np.arange(1, n_bins + 1),
        "x_left": edges[:-1],
        "x_right": edges[1:],
        "x_center": centers,
        "count": counts,
        "area_km2": area_km2
    })
    return df


def aggregate_1000_to_200_with_sd(df_fine, group_size=5):
    n = len(df_fine)
    if n % group_size != 0:
        raise ValueError("The number of fine bins cannot be evenly divided by group_size.")

    group_id = np.repeat(np.arange(1, n // group_size + 1), group_size)
    df = df_fine.copy()
    df["bin_id_200"] = group_id

    agg = df.groupby("bin_id_200").agg(
        x_left=("x_left", "min"),
        x_right=("x_right", "max"),
        x_center=("x_center", "mean"),
        count_sum=("count", "sum"),
        area_sum_km2=("area_km2", "sum"),
        area_mean_km2=("area_km2", "mean"),
        area_sd_km2=("area_km2", "std"),
        n_subbins=("area_km2", "count"),
    ).reset_index()

    agg["area_sd_km2"] = agg["area_sd_km2"].fillna(0)
    agg["area_sd_sum_km2"] = np.sqrt(agg["n_subbins"]) * agg["area_sd_km2"]
    agg["sum_lower"] = agg["area_sum_km2"] - agg["area_sd_sum_km2"]
    agg["sum_upper"] = agg["area_sum_km2"] + agg["area_sd_sum_km2"]
    agg["sum_lower"] = agg["sum_lower"].clip(lower=0)

    return agg


def process_one_period(task, out_dir):
    period_name = task["name"]
    tif_path = task["tif"]

    safe_name = period_name.replace("-", "_")
    fine_csv = os.path.join(out_dir, f"{safe_name}_bins1000.csv")
    main_csv = os.path.join(out_dir, f"{safe_name}_bins200_sd.csv")

    values, pixel_area_km2 = extract_valid_values_and_area(tif_path)

    df_fine = calc_fine_histogram(values, pixel_area_km2, n_bins=n_bins_fine)
    df_main = aggregate_1000_to_200_with_sd(df_fine, group_size=n_bins_fine // n_bins_main)

    df_fine.to_csv(fine_csv, index=False, encoding="utf-8-sig")
    df_main.to_csv(main_csv, index=False, encoding="utf-8-sig")

    print(f"    Saved high-resolution CSV: {fine_csv}")
    print(f"    Saved main curve CSV: {main_csv}")

    return fine_csv, main_csv


def load_or_create_period_csv(task, out_dir):
    period_name = task["name"]
    safe_name = period_name.replace("-", "_")
    fine_csv = os.path.join(out_dir, f"{safe_name}_bins1000.csv")
    main_csv = os.path.join(out_dir, f"{safe_name}_bins200_sd.csv")

    if use_existing_csv_if_available and os.path.exists(fine_csv) and os.path.exists(main_csv):
        print(f"\n>>> Existing CSV detected; reading directly: {period_name}")
        df_fine = pd.read_csv(fine_csv)
        df_main = pd.read_csv(main_csv)
        return df_fine, df_main
    else:
        process_one_period(task, out_dir)
        df_fine = pd.read_csv(fine_csv)
        df_main = pd.read_csv(main_csv)
        return df_fine, df_main


# =========================
# Five-class Jenks classification based on the joint distribution of the two rasters, with dashed boundaries in the figure
# =========================

def calc_jenks_breaks_from_values(values, n_classes=5, hist_bins=2000, label=""):
    values = np.asarray(values, dtype=float)
    values = values[
        (~np.isnan(values))
        & (values >= value_min)
        & (values <= value_max)
    ]

    if len(values) == 0:
        raise ValueError(f"No valid LSM values found for Jenks classification: {label}")

    counts, edges = np.histogram(values, bins=hist_bins, range=(value_min, value_max))
    centers = (edges[:-1] + edges[1:]) / 2

    mask = counts > 0
    counts = counts[mask]
    centers = centers[mask]

    print(f"\n>>> {label} valid pixel count: {len(values):,}")
    print(f">>> {label} Jenks approximation uses non-empty histogram bins: {len(centers)}")

    scale = max(1, int(np.ceil(counts.max() / 50)))
    expanded = np.repeat(centers, np.maximum(1, (counts / scale).astype(int)))
    expanded = np.sort(expanded)

    print(f">>> {label} compressed Jenks sample count: {len(expanded):,}")

    data = expanded
    n_data = len(data)

    lower_class_limits = np.zeros((n_data + 1, n_classes + 1), dtype=int)
    variance_combinations = np.full((n_data + 1, n_classes + 1), np.inf, dtype=float)

    for i in range(1, n_classes + 1):
        lower_class_limits[1, i] = 1
        variance_combinations[1, i] = 0.0

    for l in range(2, n_data + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            idx = l - m
            val = data[idx]
            w += 1
            s1 += val
            s2 += val * val
            variance = s2 - (s1 * s1) / w

            if idx != 0:
                for j in range(2, n_classes + 1):
                    test_var = variance + variance_combinations[idx, j - 1]
                    if variance_combinations[l, j] >= test_var:
                        lower_class_limits[l, j] = idx + 1
                        variance_combinations[l, j] = test_var

        lower_class_limits[l, 1] = 1
        variance_combinations[l, 1] = variance

    k = n_data
    kclass = [0.0] * (n_classes + 1)
    kclass[n_classes] = data[-1]
    kclass[0] = data[0]

    count_num = n_classes
    while count_num >= 2:
        idx = int(lower_class_limits[k, count_num] - 2)
        kclass[count_num - 1] = data[idx]
        k = int(lower_class_limits[k, count_num] - 1)
        count_num -= 1

    breaks = np.array(kclass[1:-1], dtype=float)

    print(f"\n>>> {label} period-specific Jenks natural breaks:")
    for i, b in enumerate(breaks, start=1):
        print(f"    Break {i}: {b:.6f}")

    return breaks


def calc_period_jenks_breaks(tasks, n_classes=5, hist_bins=2000):
    breaks_by_period = {}

    for task in tasks:
        period_name = task["name"]
        values, _ = extract_valid_values_and_area(task["tif"])
        breaks_by_period[period_name] = calc_jenks_breaks_from_values(
            values,
            n_classes=n_classes,
            hist_bins=hist_bins,
            label=period_name,
        )

    return breaks_by_period


def calc_joint_jenks_breaks(tasks, n_classes=5, hist_bins=2000):
    """
    Joint Jenks natural-break classification using the combined valid-pixel
    distribution from the two LSM rasters.
    """
    all_values = []

    for task in tasks:
        tif_path = task["tif"]
        values, _ = extract_valid_values_and_area(tif_path)
        all_values.append(values)

    all_values = np.concatenate(all_values).astype(float)
    all_values = all_values[(~np.isnan(all_values)) & (all_values >= value_min) & (all_values <= value_max)]

    print(f"\n>>> Combined valid pixel count: {len(all_values)}")

    counts, edges = np.histogram(all_values, bins=hist_bins, range=(value_min, value_max))
    centers = (edges[:-1] + edges[1:]) / 2

    mask = counts > 0
    counts = counts[mask]
    centers = centers[mask]

    print(f">>> Jenks approximation uses non-empty histogram bins: {len(centers)}")

    scale = max(1, int(np.ceil(counts.max() / 50)))
    expanded = np.repeat(centers, np.maximum(1, (counts / scale).astype(int)))
    expanded = np.sort(expanded)

    print(f">>> Compressed sample count used for Jenks: {len(expanded)}")

    data = expanded
    n_data = len(data)

    lower_class_limits = np.zeros((n_data + 1, n_classes + 1), dtype=int)
    variance_combinations = np.full((n_data + 1, n_classes + 1), np.inf, dtype=float)

    for i in range(1, n_classes + 1):
        lower_class_limits[1, i] = 1
        variance_combinations[1, i] = 0.0

    for l in range(2, n_data + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            idx = l - m
            val = data[idx]
            w += 1
            s1 += val
            s2 += val * val
            variance = s2 - (s1 * s1) / w

            if idx != 0:
                for j in range(2, n_classes + 1):
                    test_var = variance + variance_combinations[idx, j - 1]
                    if variance_combinations[l, j] >= test_var:
                        lower_class_limits[l, j] = idx + 1
                        variance_combinations[l, j] = test_var

        lower_class_limits[l, 1] = 1
        variance_combinations[l, 1] = variance

    k = n_data
    kclass = [0.0] * (n_classes + 1)
    kclass[n_classes] = data[-1]
    kclass[0] = data[0]

    count_num = n_classes
    while count_num >= 2:
        idx = int(lower_class_limits[k, count_num] - 2)
        kclass[count_num - 1] = data[idx]
        k = int(lower_class_limits[k, count_num] - 1)
        count_num -= 1

    breaks = np.array(kclass[1:-1], dtype=float)

    print("\n>>> Joint-distribution Jenks natural break thresholds (quick approximation):")
    for i, b in enumerate(breaks, start=1):
        print(f"    Break {i}: {b:.6f}")

    return breaks


def save_jenks_breaks_csv(breaks, out_dir, filename="joint_jenks_breaks.csv"):
    csv_path = os.path.join(out_dir, filename)

    levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    rows = []

    if isinstance(breaks, dict):
        items = breaks.items()
    else:
        items = [("joint", breaks)]

    for period_name, period_breaks in items:
        rows.extend([
            {
                "Method": break_method,
                "Period": period_name,
                "Level": levels[0],
                "Range": f"[0.000000, {period_breaks[0]:.6f})",
            },
            {
                "Method": break_method,
                "Period": period_name,
                "Level": levels[1],
                "Range": f"[{period_breaks[0]:.6f}, {period_breaks[1]:.6f})",
            },
            {
                "Method": break_method,
                "Period": period_name,
                "Level": levels[2],
                "Range": f"[{period_breaks[1]:.6f}, {period_breaks[2]:.6f})",
            },
            {
                "Method": break_method,
                "Period": period_name,
                "Level": levels[3],
                "Range": f"[{period_breaks[2]:.6f}, {period_breaks[3]:.6f})",
            },
            {
                "Method": break_method,
                "Period": period_name,
                "Level": levels[4],
                "Range": f"[{period_breaks[3]:.6f}, 1.000000]",
            },
        ])

    df_breaks = pd.DataFrame(rows)

    df_breaks.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nJenks classification results saved: {csv_path}")

    return csv_path


def load_jenks_breaks_csv(out_dir, filename="joint_jenks_breaks.csv"):
    csv_path = os.path.join(out_dir, filename)
    df = pd.read_csv(csv_path)

    def parse_breaks(range_series):
        parsed = []
        for r in range_series.iloc[:-1]:
            right_part = r.split(",")[1].strip()
            right_value = right_part.replace(")", "").replace("]", "")
            parsed.append(float(right_value))
        return np.array(parsed, dtype=float)

    if "Period" not in df.columns:
        return {"joint": parse_breaks(df["Range"])}

    breaks_by_period = {}
    for period_name, df_period in df.groupby("Period", sort=False):
        breaks_by_period[str(period_name)] = parse_breaks(df_period["Range"])

    return breaks_by_period


def breaks_csv_is_current(csv_path):
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return False
    return (
        "Method" in df.columns
        and "Period" in df.columns
        and str(df["Method"].iloc[0]) == break_method
    )


def plot_from_csv_with_jenks_breaks(tasks, out_dir, png_path, pdf_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(4.5, 2))

    jenks_csv = os.path.join(out_dir, "joint_jenks_breaks.csv")
    if os.path.exists(jenks_csv) and breaks_csv_is_current(jenks_csv):
        print(f"\n>>> Existing Jenks classification file detected; reading directly: {jenks_csv}")
        breaks_by_period = load_jenks_breaks_csv(out_dir)
    else:
        print("\n>>> joint_jenks_breaks.csv not detected; recalculating Jenks breaks...")
        breaks_by_period = calc_period_jenks_breaks(tasks, n_classes=5, hist_bins=2000)
        save_jenks_breaks_csv(breaks_by_period, out_dir)

    for task in tasks:
        _, df_main = load_or_create_period_csv(task, out_dir)

        x = df_main["x_center"].values
        y = df_main["area_sum_km2"].values / 1e4
        y_min = df_main["sum_lower"].values / 1e4
        y_max = df_main["sum_upper"].values / 1e4

        ax.fill_between(
            x, y_min, y_max,
            color=task["color"],
            alpha=0.22,
            linewidth=0,
            zorder=1
        )

        ax.plot(
            x, y,
            color=task["color"],
            lw=1.2,
            label=task["name"],
            solid_capstyle="round",
            zorder=2
        )

        period_breaks = breaks_by_period.get(task["name"])
        if period_breaks is None:
            period_breaks = breaks_by_period.get("joint")

        if period_breaks is not None:
            for b in period_breaks:
                ax.axvline(
                    x=b,
                    color=task["color"],
                    linestyle="--",
                    linewidth=0.75,
                    alpha=0.55,
                    zorder=0
                )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Susceptibility", fontsize=7)
    ax.set_ylabel("Area ($10^4$ km$^2$)", fontsize=7)
    ax.set_yticks([0, 4, 8, 12])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.tick_params(axis="both", which="major", direction="out", length=2.5, width=0.5, pad=2)

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.2,
        borderpad=0.2,
        labelspacing=0.3
    )

    plt.tight_layout(pad=0.6)

    png_path2 = png_path.replace(".png", "_with_jenks_breaks.png")
    pdf_path2 = pdf_path.replace(".pdf", "_with_jenks_breaks.pdf")

    fig.savefig(png_path2, dpi=600, bbox_inches="tight", transparent=False)
    # fig.savefig(pdf_path2, dpi=600, bbox_inches="tight", transparent=False)
    plt.show()

    print(f"\nFigure with Jenks class dashed boundaries saved as:")
    print(f"  PNG: {png_path2}")
    print(f"  PDF: {pdf_path2}")


def calc_area_by_jenks_classes(tasks, out_dir):
    """
    Based on the five-level Jenks thresholds, calculate the area of each class for both periods and the change magnitude.
    Output:
        df_area_change: DataFrame
    """
    jenks_csv = os.path.join(out_dir, "joint_jenks_breaks.csv")

    if os.path.exists(jenks_csv) and breaks_csv_is_current(jenks_csv):
        print(f"\n>>> Existing Jenks classification file detected; reading directly: {jenks_csv}")
        breaks_by_period = load_jenks_breaks_csv(out_dir)
    else:
        print("\n>>> joint_jenks_breaks.csv not detected; recalculating Jenks breaks...")
        breaks_by_period = calc_period_jenks_breaks(tasks, n_classes=5, hist_bins=2000)
        save_jenks_breaks_csv(breaks_by_period, out_dir)

    labels = ["VL", "L", "M", "H", "VH"]
    full_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]

    results = {}

    for task in tasks:
        period_breaks = breaks_by_period.get(task["name"])
        if period_breaks is None:
            period_breaks = breaks_by_period.get("joint")
        if period_breaks is None:
            raise ValueError(f"No Jenks breaks found for period: {task['name']}")

        bins = np.array([
            value_min,
            period_breaks[0],
            period_breaks[1],
            period_breaks[2],
            period_breaks[3],
            value_max + 1e-9
        ])

        values, pixel_area_km2 = extract_valid_values_and_area(task["tif"])
        class_ids = np.digitize(values, bins, right=False) - 1

        area_dict = {}
        for i, lab in enumerate(labels):
            count = np.sum(class_ids == i)
            area_dict[lab] = count * pixel_area_km2

        results[task["name"]] = area_dict

    early = results["2000-2019"]
    late = results["2020-2024"]

    rows = []
    for short_lab, full_lab in zip(labels, full_labels):
        early_area = early[short_lab]
        late_area = late[short_lab]
        delta_area = late_area - early_area
        pct_change = (delta_area / early_area * 100) if early_area > 0 else np.nan

        rows.append({
            "Class": short_lab,
            "Class_Full": full_lab,
            "Area_2000_2019_km2": round(early_area, 2),
            "Area_2020_2024_km2": round(late_area, 2),
            "Change_km2": round(delta_area, 2),
            "Change_percent": round(pct_change, 2)
        })

    df_area_change = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "jenks_class_area_change.csv")
    df_area_change.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n>>> Areas and changes for each class saved: {out_csv}")
    print(df_area_change)

    return df_area_change

if __name__ == "__main__":
    ensure_dir(out_dir)
    plot_from_csv_with_jenks_breaks(tasks, out_dir, final_png, final_pdf)
    calc_area_by_jenks_classes(tasks, out_dir)
