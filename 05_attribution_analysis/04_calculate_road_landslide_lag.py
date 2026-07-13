import geopandas as gpd
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================
# plotting setup
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.8,
    "legend.title_fontsize": 5.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02
})

# =========================
# Paths
# =========================
road_shp_dir = r"H:/Himalaya/cause/Road/roads_new_build_yearly/newroad_shp"
landslide_shp = r"H:/Himalaya/13w_landslides_list_final.shp"
output_dir = r"H:/Himalaya/figure"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Figure dimensions
# =========================
MM_TO_INCH = 1 / 25.4
FIG_W = 110 * MM_TO_INCH
FIG_H = 64 * MM_TO_INCH


def analyze_landslide_road_relation():
    print("Merging newly added road data from each year...")
    road_files = glob.glob(os.path.join(road_shp_dir, "*.shp"))
    road_list = []

    target_crs = "EPSG:32645"

    for f in road_files:
        year_match = re.search(r'(\d{4})', os.path.basename(f))
        if year_match:
            year = int(year_match.group(1))
            temp_gdf = gpd.read_file(f)

            if temp_gdf.crs is None:
                print(f"Warning: File {f} is missing a coordinate system, skipping...")
                continue

            temp_gdf = temp_gdf.to_crs(target_crs)
            temp_gdf["build_year"] = year
            road_list.append(temp_gdf)
            print(f"Loaded road data for {year}")

    if not road_list:
        print("No valid road data found. Please check the paths and filenames!")
        return

    all_roads = pd.concat(road_list, ignore_index=True)
    all_roads = gpd.GeoDataFrame(all_roads, crs=target_crs)
    print(f"Road data merged, total {len(all_roads)} segments.")

    print("Merging landslide data...")
    gdf_ls = gpd.read_file(landslide_shp)
    gdf_ls = gdf_ls.to_crs(target_crs)

    if "year" not in gdf_ls.columns:
        possible_cols = ["year", "YEAR", "slide_year", "occur_year"]
        for col in possible_cols:
            if col in gdf_ls.columns:
                gdf_ls["year"] = gdf_ls[col]
                break

    if "year" not in gdf_ls.columns:
        raise ValueError("Landslide data is missing a year column. Please check year/YEAR/slide_year/occur_year")

    print("Merging landslide data...")
    gdf_ls = gpd.read_file(landslide_shp)
    gdf_ls = gdf_ls.to_crs(target_crs)

    if "year" not in gdf_ls.columns:
        possible_cols = ["year", "YEAR", "slide_year", "occur_year"]
        for col in possible_cols:
            if col in gdf_ls.columns:
                gdf_ls["year"] = gdf_ls[col]
                break

    if "year" not in gdf_ls.columns:
        raise ValueError("Landslide data is missing a year column. Please check year/YEAR/slide_year/occur_year")

    print("Performing spatial nearest neighbor analysis (this may take some time)...")
    relation = gpd.sjoin_nearest(
        gdf_ls,
        all_roads,
        distance_col="dist_to_road",
        max_distance=500,
        how="left"
    )

    ls_year_col = "year_left" if "year_left" in relation.columns else "year"

    print("Applying time-space judgment logic...")
    relation["delta_t"] = relation[ls_year_col] - relation["build_year"]

    def classify(row):
        if pd.isna(row["delta_t"]):
            return "Natural/Background"

        d = row["dist_to_road"]
        dt = row["delta_t"]

        if 0 <= dt <= 1 and d <= 150:
            return "Construction-Induced"
        if dt > 1 and d <= 300:
            return "Vulnerability-Driven"

        return "Natural/Background"

    relation["type"] = relation.apply(classify, axis=1)

    print("\n--- Statistics ---")
    print(relation["type"].value_counts())

    print("Merging lag time distribution plot...")
    lag_data = relation[
        (relation["delta_t"] >= 0) &
        (relation["type"] != "Natural/Background")
    ].copy()

    if lag_data.empty:
        print("No valid landslide data found for plotting (Delta T >= 0).")
        return

    max_plot_year = 20
    lag_data = lag_data[lag_data["delta_t"] <= max_plot_year].copy()

    lag_counts = lag_data.groupby("delta_t").size().reindex(range(max_plot_year + 1), fill_value=0)
    lag_pct = lag_counts / lag_counts.sum() * 100
    lag_cum = lag_pct.cumsum()

    mean_lag = lag_data["delta_t"].mean()

    x = np.arange(max_plot_year + 1)

    # 50% and 80% cumulative lag years
    lag50 = int(np.argmax(lag_cum.values >= 50))
    lag80 = int(np.argmax(lag_cum.values >= 80))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Annual bars
    ax.bar(
        x,
        lag_pct.values,
        width=0.76,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.45,
        zorder=2
    )

    # Mean line only
    ax.axvline(mean_lag, color="black", linestyle="--", linewidth=0.9, zorder=4)

    # Secondary axis for cumulative percentage
    ax2 = ax.twinx()
    ax2.plot(
        x,
        lag_cum.values,
        color="#D55E00",
        linewidth=1.1,
        marker="o",
        markersize=3.0,
        zorder=5
    )

    # Horizontal guide lines for 50% and 80%
    ax2.axhline(50, color="#D55E00", linestyle=":", linewidth=0.7, alpha=0.8, zorder=1)
    ax2.axhline(80, color="#D55E00", linestyle=":", linewidth=0.7, alpha=0.8, zorder=1)

    # Markers for 50% and 80%
    ax2.scatter([lag50], [lag_cum.iloc[lag50]], color="#D55E00", s=20, zorder=6)
    ax2.scatter([lag80], [lag_cum.iloc[lag80]], color="#D55E00", s=20, zorder=6)

    # Axes
    ax.set_xlim(-0.5, max_plot_year + 0.5)
    ax.set_xticks(x)
    ax.set_xlabel("Lag time (years)")
    ax.set_ylabel("Landslides (%)")

    ax2.set_ylabel("Cumulative proportion (%)")
    ax2.set_ylim(0, 105)

    y_top = max(lag_pct.max() * 1.3, 11)
    ax.set_ylim(0, y_top)

    # Light grid
    ax.yaxis.grid(True, linestyle="-", linewidth=0.3, alpha=0.15)
    ax.xaxis.grid(False)

    # Spines
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tick padding
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=6)
    ax2.tick_params(axis="y", pad=6)

    # 50% and 80% annotations near right axis
    right_x = max_plot_year - 0.1
    ax2.text(
        right_x,
        50 + 1.2,
        f"50% within {lag50} yr",
        color="#D55E00",
        fontsize=5.8,
        ha="right",
        va="bottom"
    )
    ax2.text(
        right_x,
        80 + 1.2,
        f"80% within {lag80} yr",
        color="#D55E00",
        fontsize=5.8,
        ha="right",
        va="bottom"
    )

    # Legend moved to upper left, mean text included in legend
    legend_handles = [
        Patch(facecolor="#4C78A8", edgecolor="black", linewidth=0.45, label="Annual"),
        Line2D([0], [0], color="#D55E00", marker="o", lw=1.1, markersize=3.0, label="Cumulative"),
        Line2D([0], [0], color="black", lw=0.9, linestyle="--", label=f"Mean = {mean_lag:.1f} yr"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        handlelength=2.2,
        borderaxespad=0.2,
        labelspacing=0.45
    )

    plt.tight_layout()

    out_base = os.path.join(output_dir, "lag_time_distribution_final_adjusted")
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".eps")
    fig.savefig(out_base + ".png", dpi=600)

    # Save summary data
    lag_export = pd.DataFrame({
        "lag_year": x,
        "count": lag_counts.values,
        "percentage": lag_pct.values,
        "cumulative_percentage": lag_cum.values
    })
    lag_export.to_csv(os.path.join(output_dir, "lag_distribution_summary_final_adjusted.csv"), index=False)

    save_cols = [ls_year_col, "build_year", "delta_t", "dist_to_road", "type"]
    existing_save_cols = [c for c in save_cols if c in relation.columns]
    relation[existing_save_cols].to_csv(
        os.path.join(output_dir, "classification_results_final_adjusted.csv"),
        index=False
    )

    print(f"The adjusted diagrams and result tables have been successfully saved to {output_dir}")
    plt.show()


if __name__ == "__main__":
    analyze_landslide_road_relation()