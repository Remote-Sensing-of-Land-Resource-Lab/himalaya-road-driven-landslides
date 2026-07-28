import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pymannkendall as mk
import os
import time

# =========================
# Parameter settings
# =========================
P_thre = 0.05
N_thre = 250
ALL_YEARS = list(range(2000, 2025))

# --- Global style settings ---
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})


def compute_and_cache_results(csv_path, shp_path, road_path, eq_points_csv_path, cache_dir):
    """
    Perform the full analysis and save cached results:
    1) Identify hotspot grids for landslides
    2) Extract landslide points within hotspot areas
    3) Count landslides inside/outside road buffers by year (without subtracting earthquake-triggered landslides)
    4) Calculate annual mean precipitation and road density in hotspot areas
    5) Count earthquakes by year in hotspot areas
    """
    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)

    print("Reading landslide points and grids...")
    df = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    grid = gpd.read_file(shp_path)
    if gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)

    print("Performing landslide point-grid spatial join...")
    joined = gpd.sjoin(gdf, grid, how="inner", predicate="within")

    print("Calculating annual landslide frequency...")
    annual_counts = joined.groupby(['index_right', 'year']).size().unstack(fill_value=0)
    annual_counts = annual_counts.reindex(columns=ALL_YEARS, fill_value=0)
    annual_counts = annual_counts.reindex(grid.index, fill_value=0)

    print("Identifying hotspot grids using the Mann-Kendall trend test...")
    total_sums = annual_counts.sum(axis=1).values
    slopes = []
    p_values = []

    for i in range(len(grid)):
        if total_sums[i] > N_thre:
            res = mk.original_test(annual_counts.values[i])
            slopes.append(res.slope)
            p_values.append(res.p)
        else:
            slopes.append(0)
            p_values.append(1.0)

    grid["total_sum"] = total_sums
    grid["slope"] = slopes
    grid["p_value"] = p_values

    sig_increase_mask = (
        (grid["total_sum"] > N_thre) &
        (grid["p_value"] < P_thre) &
        (grid["slope"] > 0)
    )

    target_grid_ids = grid[sig_increase_mask].index
    hotspot_data = joined[joined["index_right"].isin(target_grid_ids)].copy()

    print(f"Number of hotspot grids: {len(target_grid_ids)}")
    print(f"Number of landslide points in hotspot areas: {len(hotspot_data)}")

    print("Reading roads and creating a 1000 m buffer...")
    roads = gpd.read_file(road_path)

    if roads.crs != hotspot_data.crs:
        roads = roads.to_crs(hotspot_data.crs)

    # To buffer in meters, first reproject to a projected CRS
    roads_buffer = roads.to_crs(epsg=3857).buffer(1000)
    roads_buffer_gdf = gpd.GeoDataFrame(
        geometry=roads_buffer,
        crs="EPSG:3857"
    ).to_crs(hotspot_data.crs)

    print("Determining whether hotspot landslide points fall within road buffers...")
    points_near_road = gpd.sjoin(
        hotspot_data,
        roads_buffer_gdf,
        how="left",
        predicate="within",
        rsuffix="road"
    )

    is_near = ~points_near_road["index_road"].isna()
    is_near_cleaned = is_near.groupby(is_near.index).any()
    hotspot_data["is_near_road"] = is_near_cleaned.reindex(hotspot_data.index, fill_value=False)

    print("Counting landslides inside/outside road buffers by year (without subtracting earthquake-triggered landslides)...")
    yearly_road_stats = (
        hotspot_data.groupby(["year", "is_near_road"])
        .size()
        .unstack(fill_value=0)
        .reindex(ALL_YEARS, fill_value=0)
    )

    for col in [True, False]:
        if col not in yearly_road_stats.columns:
            yearly_road_stats[col] = 0

    yearly_road_stats = yearly_road_stats.reindex(columns=[True, False], fill_value=0)
    yearly_road_stats = yearly_road_stats.rename(columns={
        True: "within_buffer",
        False: "outside_buffer"
    })

    print("Calculating annual mean values of environmental factors in hotspot areas...")
    yearly_env_stats = hotspot_data.groupby("year").agg({
        "Annual_Mean": "mean",
        "road_density": "mean"
    }).reindex(ALL_YEARS)

    print("Counting earthquakes by year in hotspot areas...")
    eq_df = pd.read_csv(eq_points_csv_path)

    if "year" not in eq_df.columns:
        raise ValueError("The earthquake CSV is missing the 'year' column.")

    eq_df["year"] = pd.to_numeric(eq_df["year"], errors="coerce")
    eq_df = eq_df.dropna(subset=["year"]).copy()
    eq_df["year"] = eq_df["year"].astype(int)

    yearly_eq_counts = eq_df.groupby("year").size().reindex(ALL_YEARS, fill_value=0)
    yearly_eq_counts = yearly_eq_counts.to_frame(name="earthquake_count")

    hotspot_summary = grid.loc[target_grid_ids, ["total_sum", "slope", "p_value"]].copy()
    hotspot_summary.index.name = "grid_id"

    print("Saving cache results...")
    yearly_road_stats.to_csv(
        os.path.join(cache_dir, "yearly_road_stats.csv"),
        encoding="utf-8-sig"
    )
    yearly_env_stats.to_csv(
        os.path.join(cache_dir, "yearly_env_stats.csv"),
        encoding="utf-8-sig"
    )
    yearly_eq_counts.to_csv(
        os.path.join(cache_dir, "yearly_eq_counts.csv"),
        encoding="utf-8-sig"
    )
    hotspot_summary.to_csv(
        os.path.join(cache_dir, "hotspot_summary.csv"),
        encoding="utf-8-sig"
    )

    # Save the hotspot landslide table as an additional check
    hotspot_data.drop(columns="geometry", errors="ignore").to_csv(
        os.path.join(cache_dir, "hotspot_data_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    t1 = time.time()
    print(f"All computations completed; cache saved to: {cache_dir}")
    print(f"Total elapsed time: {t1 - t0:.2f} seconds")


def plot_from_cache(cache_dir, out_dir,
                    fig_name="Hotspot_Drivers_Road_Buffer1000_with_EQ1.png"):
    """
    Read cached results and quickly generate a plot:
    - Bar chart: landslide counts inside/outside the road buffer
    - Line 1: MAP
    - Line 2: Road Density
    - Line 3: Earthquake Count
    """
    t0 = time.time()

    road_stats_path = os.path.join(cache_dir, "yearly_road_stats.csv")
    env_stats_path = os.path.join(cache_dir, "yearly_env_stats.csv")
    eq_counts_path = os.path.join(cache_dir, "yearly_eq_counts.csv")

    if not (os.path.exists(road_stats_path) and
            os.path.exists(env_stats_path) and
            os.path.exists(eq_counts_path)):
        raise FileNotFoundError("Cache files do not exist; please run the full analysis first.")

    print("Reading cache files and plotting...")

    yearly_road_stats = pd.read_csv(road_stats_path, index_col=0)
    yearly_env_stats = pd.read_csv(env_stats_path, index_col=0)
    yearly_eq_counts = pd.read_csv(eq_counts_path, index_col=0)

    yearly_road_stats.index = yearly_road_stats.index.astype(int)
    yearly_env_stats.index = yearly_env_stats.index.astype(int)
    yearly_eq_counts.index = yearly_eq_counts.index.astype(int)

    os.makedirs(out_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(28, 8), dpi=300)
    fig.subplots_adjust(right=0.86)

    # -------------------------
    # Left axis: bar chart
    # -------------------------
    color_in = "#D28A91"
    color_out = "#D9D9D9"

    ax1.bar(
        yearly_road_stats.index,
        yearly_road_stats["within_buffer"],
        color=color_in,
        alpha=1,
        label="Within 1000m Road Buffer",
        width=0.7
    )

    ax1.bar(
        yearly_road_stats.index,
        yearly_road_stats["outside_buffer"],
        bottom=yearly_road_stats["within_buffer"],
        color=color_out,
        label="Outside Road Buffer",
        width=0.7
    )

    ax1.set_xticks(ALL_YEARS)
    ax1.set_xticklabels(ALL_YEARS, rotation=45, ha="right", fontsize=28)
    ax1.set_xlim(ALL_YEARS[0] - 0.6, ALL_YEARS[-1] + 0.6)

    ax1.set_xlabel("Year", fontsize=32)
    ax1.set_ylabel("Landslide Count", fontsize=32)
    ax1.tick_params(axis="y", labelsize=26)
    ax1.spines["top"].set_visible(False)

    # -------------------------
    # Right axis 1: precipitation
    # -------------------------
    ax2 = ax1.twinx()
    color_rain = "#4C78A8"
    ax2.plot(
        yearly_env_stats.index,
        yearly_env_stats["Annual_Mean"],
        color=color_rain,
        marker="o",
        markersize=8,
        linewidth=3,
        label="MAP"
    )
    ax2.set_ylabel("MAP (mm)", fontsize=30, labelpad=10)
    ax2.tick_params(axis="y", labelsize=26)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 2000)
    ax2.set_yticks([0, 500, 1000, 1500, 2000])

    # -------------------------
    # Right axis 2: road density
    # -------------------------
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 105))
    color_road = "#B04A5A"
    ax3.plot(
        yearly_env_stats.index,
        yearly_env_stats["road_density"],
        color=color_road,
        marker="s",
        markersize=8,
        linewidth=4,
        linestyle="--",
        label="Road Density"
    )
    ax3.set_ylabel("Mean Road Density (km/km$^2$)", fontsize=30, labelpad=12)
    ax3.tick_params(axis="y", labelsize=26)
    ax3.spines["top"].set_visible(False)

    # -------------------------
    # Right axis 3: earthquake count
    # -------------------------
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 195))
    color_eq = "#8E6C8A"
    ax4.plot(
        yearly_eq_counts.index,
        yearly_eq_counts["eq_count_in_hot"],
        color=color_eq,
        marker="^",
        markersize=7,
        linewidth=4,
        linestyle="-.",
        label="Earthquake Count"
    )
    ax4.set_ylabel("Earthquake Count", fontsize=30, labelpad=14)
    ax4.tick_params(axis="y", labelsize=26)
    ax4.spines["top"].set_visible(False)
    ax4.set_ylim(0, 30)
    ax4.set_yticks([0, 10, 20, 30])

    # -------------------------
    # Legend
    # -------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2 + lines3 + lines4,
        labels1 + labels2 + labels3 + labels4,
        loc="upper left",
        fontsize=30,
        bbox_to_anchor=(0.01, 1.04),
        frameon=False,
        ncol=3
    )

    save_path = os.path.join(out_dir, fig_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    t1 = time.time()
    print(f"Plot saved: {save_path}")
    print(f"Plotting time: {t1 - t0:.2f} seconds")


def analyze_hotspots_with_factors(csv_path, shp_path, road_path, eq_points_csv_path,
                                  out_dir, cache_dir,
                                  use_cache=True, force_recompute=False):
    """
    Main controller function
    """
    road_stats_path = os.path.join(cache_dir, "yearly_road_stats.csv")
    env_stats_path = os.path.join(cache_dir, "yearly_env_stats.csv")
    eq_counts_path = os.path.join(cache_dir, "yearly_eq_counts.csv")

    cache_exists = (
        os.path.exists(road_stats_path) and
        os.path.exists(env_stats_path) and
        os.path.exists(eq_counts_path)
    )

    if force_recompute or (not use_cache) or (not cache_exists):
        print("No usable cache detected; starting the full analysis...")
        compute_and_cache_results(
            csv_path, shp_path, road_path, eq_points_csv_path, cache_dir
        )
    else:
        print("Cache detected; skipping the time-consuming analysis step.")

    plot_from_cache(cache_dir, out_dir)


# =========================
# Execution
# =========================
csv_path = r'H:/Himalaya/RF_susceptibility/features_pos.csv'
shp_path = r'H:/Himalaya/grid/Himalaya_hex_1000km2/Himalaya_hex_1000km2.shp'
road_path = r'H:\Himalaya\cause\Road\osm_roads_himalaya\roads_latest\roads_2025_clip_add_Ms.shp'
eq_points_csv_path = r'H:\Himalaya\cause\Earthquake\usgs_shakemap_dual\eq5_00_24.csv'
out_dir = r'H:/Himalaya/figure/'
cache_dir = r'H:\Himalaya\figure\cache_hotspot_road_buffer1000_with_eq/'

analyze_hotspots_with_factors(
    csv_path=csv_path,
    shp_path=shp_path,
    road_path=road_path,
    eq_points_csv_path=eq_points_csv_path,
    out_dir=out_dir,
    cache_dir=cache_dir,
    use_cache=True,
    force_recompute=False
)


