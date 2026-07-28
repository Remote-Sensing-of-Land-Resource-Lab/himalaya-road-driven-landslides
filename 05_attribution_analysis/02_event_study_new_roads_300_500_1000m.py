# -*- coding: utf-8 -*-
# ============================================================
# Purpose:
#   Draw cohort-level landslide density curves around newly built roads.
#
# Design:
#   Road construction cohorts: 2004–2020
#   Event-time window: -4 to +4 years
#   Near-road zones: landslide points within 300 m, 500 m and 1000 m of newly built roads; background 1-3 km
#
# ============================================================

import os
import re
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform

from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring, unary_union


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. INPUT PATHS
# ============================================================

ROAD_FOLDER = r"H:\Himalaya\cause\Road\roads_new_build_yearly\newroad_shp_merge"
LANDSLIDE_POINT_PATH = r"H:\Himalaya\13w_landslides_list_final_points.shp"
HOTSPOT_GRID_PATH = r"H:\Himalaya\grid\hot_grid_20_24\Himalaya_Landslide_Sig_grid.shp"
SLOPE_RASTER_PATH = r"H:\Himalaya\road_event_study\hot_grid_slope.tif"

OUT_DIR = r"H:\Himalaya\road_event_study\road_cohort_density_curves_multizone"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. PARAMETERS
# ============================================================

# Use 2004–2020 to ensure a complete -4 to +4 window within 2000–2024.
COHORT_START_YEAR = 2008
COHORT_END_YEAR = 2016

ALL_ROAD_START_YEAR = 2000
ALL_ROAD_END_YEAR = 2024

TAU_MIN = -8
TAU_MAX = 8
TAU_LIST = list(range(TAU_MIN, TAU_MAX + 1))

PRE_TAUS = [-8,-7,-6,-5,-4, -3, -2, -1]
POST_TAUS = [1, 2, 3, 4,5,6,7,8]

# Cumulative near-road buffers and calendar-year background zone.
# Buffer zones are interpreted as <=300 m, <=500 m and <=1000 m from focal new roads.
# Background is defined for each calendar year as hotspot area between
# BACKGROUND_INNER_DISTANCE_M and BACKGROUND_OUTER_DISTANCE_M from the existing
# road network in that calendar year.
BUFFER_DISTANCES_M = [300.0, 500.0, 1000.0]
BACKGROUND_INNER_DISTANCE_M = 1000.0
BACKGROUND_OUTER_DISTANCE_M = 3000.0

# Preferred source for background exclusion:
# This layer should contain dissolved or non-dissolved existing-road buffers
# by calendar year, with field "exiting_road_year".
# If this file is unavailable, the script falls back to buffers from all_roads
# with build_year <= calendar_year.
BACKGROUND_EXISTING_ROADS_INNER_BUFFER_FILE = os.path.join(
    r"H:\Himalaya\cause\Road\road_buffer_QGIS\projected_road",
    "all_exiting_roads_2007_2023_hotspot_extent_1000m.gpkg"
)
BACKGROUND_EXISTING_ROADS_OUTER_BUFFER_FILE = os.path.join(
    r"H:\Himalaya\cause\Road\road_buffer_QGIS\projected_road",
    "all_exiting_roads_2007_2023_hotspot_extent_3000m.gpkg"
)
BACKGROUND_EXISTING_ROAD_YEAR_FIELD = "exiting_road_year"
BACKGROUND_SOURCE_MODE = "existing_roads_by_calendar_year_annulus_1000_3000m"
BACKGROUND_MIN_SLOPE_DEG = 15.0
BACKGROUND_SLOPE_FILTER = f"slope_gt_{BACKGROUND_MIN_SLOPE_DEG:g}deg"

BUFFER_ZONES = [(f"buffer_{int(d)}m", d) for d in BUFFER_DISTANCES_M]
BACKGROUND_ZONE = (
    f"background_existing_{int(BACKGROUND_INNER_DISTANCE_M)}m_"
    f"to_{int(BACKGROUND_OUTER_DISTANCE_M)}m"
)
ZONE_ORDER = [z for z, _ in BUFFER_ZONES] + [BACKGROUND_ZONE]
BACKGROUND_LABEL = (
    f"Background area ({BACKGROUND_INNER_DISTANCE_M / 1000:g}-"
    f"{BACKGROUND_OUTER_DISTANCE_M / 1000:g} km)"
)
ZONE_LABELS = {
    "buffer_300m": "300 m buffer",
    "buffer_500m": "500 m buffer",
    "buffer_1000m": "1000 m buffer",
    BACKGROUND_ZONE: BACKGROUND_LABEL
}

ZONE_PLOT_STYLE = {
    "buffer_300m": {
        "color": "#B04A5A",
        "marker": "o",
        "linestyle": "-",
        "alpha_fill": 0.12
    },
    "buffer_500m": {
        "color": "#8E6C8A",
        "marker": "o",
        "linestyle": "-",
        "alpha_fill": 0.12
    },
    "buffer_1000m": {
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": "-",
        "alpha_fill": 0.12
    },
    BACKGROUND_ZONE: {
        "color": "#555555",
        "marker": "s",
        "linestyle": "--",
        "alpha_fill": 0.08
    }
}

# Density is reported as landslide count per 100 km² per year.
# The unscaled density per km² is also saved in the output table.
DENSITY_SCALE_KM2 = 100.0

# Kept for backward compatibility in places that need the maximum near-road distance.
NEAR_DISTANCE_M = max(BUFFER_DISTANCES_M)

# Road segmentation
# For cohort-level count curves, segmentation is not strictly required.
# However, splitting very long lines helps nearest search and length calculation.
MAX_SEGMENT_LENGTH_M = 5000.0
MIN_SEGMENT_LENGTH_M = 100.0

# Optional: exclude near-road landslides that are also within the same buffer
# distance of roads built before the focal cohort year. This helps isolate newly
# built road corridors. The background zone is not filtered by this option.
EXCLUDE_OLDER_ROADS = False 

# If input CRS is geographic, set a projected CRS manually.
# Your previous log showed EPSG:32645, so None should work if all inputs match.
TARGET_CRS = None

# Plot
FIG_DPI = 600
 
# ============================================================
# 3. OUTPUT PATHS
# ============================================================

OUT_HOTSPOT = os.path.join(OUT_DIR, "00_hotspot_grid.gpkg")
OUT_EVENT_ROADS = os.path.join(OUT_DIR, "01_event_roads_2004_2020_segments.gpkg")
OUT_LANDSLIDES = os.path.join(OUT_DIR, "02_landslide_points_hotspots.gpkg")

OUT_MATCHES = os.path.join(OUT_DIR, "03_landslide_multizone_matches.csv")
OUT_ZONE_AREAS = os.path.join(OUT_DIR, "03b_cohort_zone_areas.csv")
OUT_BACKGROUND_ZONES = os.path.join(
    OUT_DIR,
    "03c_calendar_year_background_existing_roads.gpkg"
)
OUT_COHORT_COUNTS = os.path.join(OUT_DIR, "04_cohort_tau_landslide_density.csv")
OUT_COHORT_SUMMARY = os.path.join(OUT_DIR, "05_cohort_summary_pre_post_density.csv")
OUT_MEAN_DENSITY = os.path.join(OUT_DIR, "06_mean_density_across_cohorts_by_zone.csv")

OUT_FIG_DENSITY_PDF = os.path.join(OUT_DIR, "Fig1_cohort_landslide_density_by_zone.pdf")
OUT_FIG_DENSITY_PNG = os.path.join(OUT_DIR, "Fig1_cohort_landslide_density_by_zone.png")

OUT_FIG_PRE_NORM_PDF = os.path.join(OUT_DIR, "Fig2_cohort_pre_normalized_density_by_zone.pdf")
OUT_FIG_PRE_NORM_PNG = os.path.join(OUT_DIR, "Fig2_cohort_pre_normalized_density_by_zone.png")

OUT_FIG_MEAN_DENSITY_PDF = os.path.join(OUT_DIR, "Fig3_median_landslide_density_by_zone.pdf")
OUT_FIG_MEAN_DENSITY_PNG = os.path.join(OUT_DIR, "Fig3_median_landslide_density_by_zone.png")

# ============================================================
# 3b. RUN MODE
# ============================================================

# If the density result table already exists, skip spatial processing
# and directly regenerate figures from CSV.
PLOT_FROM_EXISTING_RESULTS = True

# If True, ignore existing CSV and recompute everything from shapefiles.
FORCE_RECOMPUTE = True  # first run after changing background definition; set False after recomputing

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def try_load_existing_density_table():
    """
    Load existing cohort × tau × zone density table if available.

    This allows the script to skip heavy spatial processing and directly
    regenerate figures from previous results. The loader rejects old CSVs
    that do not contain the calendar-year existing-road background definition.
    """
    if FORCE_RECOMPUTE:
        print("FORCE_RECOMPUTE = True. Existing results will be ignored.")
        return None

    if not PLOT_FROM_EXISTING_RESULTS:
        return None

    if not os.path.exists(OUT_COHORT_COUNTS):
        print("No existing density table found. Full computation will be performed.")
        return None

    print("\nExisting density table found.")
    print(f"Loading: {OUT_COHORT_COUNTS}")

    table = pd.read_csv(OUT_COHORT_COUNTS)

    required_cols = [
        "cohort_year",
        "zone",
        "zone_label",
        "tau",
        "calendar_year",
        "landslide_count",
        "area_km2",
        "background_min_slope_deg",
        "background_slope_filter",
        "landslide_density_per_km2",
        "landslide_density_per_100km2",
        "pre_normalized_density"
    ]

    missing_cols = [c for c in required_cols if c not in table.columns]

    if len(missing_cols) > 0:
        print("Existing table is incomplete. Full computation will be performed.")
        print(f"Missing columns: {missing_cols}")
        return None

    # Reject old outputs produced with cohort-specific background zones.
    if BACKGROUND_ZONE not in set(table["zone"].astype(str).unique()):
        print("Existing table uses an old background-zone name.")
        print("Full computation will be performed with calendar-year existing-road background.")
        return None

    if "background_source_mode" not in table.columns:
        print("Existing table has no background_source_mode field.")
        print("Full computation will be performed with calendar-year existing-road background.")
        return None

    bg_mode = (
        table.loc[table["zone"].astype(str) == BACKGROUND_ZONE, "background_source_mode"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(bg_mode) == 0 or not all(v == BACKGROUND_SOURCE_MODE for v in bg_mode):
        print("Existing background results do not match the requested background mode.")
        print("Full computation will be performed.")
        return None

    bg_slope_filter = (
        table.loc[table["zone"].astype(str) == BACKGROUND_ZONE, "background_slope_filter"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(bg_slope_filter) == 0 or not all(v == BACKGROUND_SLOPE_FILTER for v in bg_slope_filter):
        print("Existing background results do not match the requested slope filter.")
        print("Full computation will be performed.")
        return None

    table["cohort_year"] = pd.to_numeric(table["cohort_year"], errors="coerce").astype(int)
    table["tau"] = pd.to_numeric(table["tau"], errors="coerce").astype(int)
    table["calendar_year"] = pd.to_numeric(table["calendar_year"], errors="coerce").astype(int)

    print("Existing table loaded successfully.")
    print("Spatial processing will be skipped. Figures will be regenerated directly.")

    return table

def safe_make_valid(gdf):
    """Remove empty geometries and repair invalid geometries."""
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def ensure_projected(gdf, name):
    """Ensure projected CRS for distance and length calculation."""
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS.")

    if gdf.crs.is_geographic:
        if TARGET_CRS is None:
            raise ValueError(
                f"{name} is in geographic CRS {gdf.crs}. "
                f"Please set TARGET_CRS to a projected CRS with metre units."
            )
        gdf = gdf.to_crs(TARGET_CRS)

    return gdf


def read_hotspot_grid():
    print("Reading hotspot grid...")
    hot = gpd.read_file(HOTSPOT_GRID_PATH)
    hot = safe_make_valid(hot)

    if TARGET_CRS is not None:
        hot = hot.to_crs(TARGET_CRS)
    else:
        hot = ensure_projected(hot, "Hotspot grid")

    hot_union = unary_union(list(hot.geometry))
    hot.to_file(OUT_HOTSPOT, driver="GPKG")

    print(f"Number of hotspot grids: {len(hot)}")
    print(f"Using CRS: {hot.crs}")

    return hot, hot_union


def read_yearly_roads(year_start, year_end, target_crs, read_extent=None):
    """Read roads_new_YEAR.shp files."""
    records = []

    for year in range(year_start, year_end + 1):
        shp_path = os.path.join(ROAD_FOLDER, f"roads_new_{year}.shp")

        if not os.path.exists(shp_path):
            print(f"[Warning] Missing road file: {shp_path}")
            continue

        print(f"Reading roads: {year}")
        gdf = gpd.read_file(shp_path)
        gdf = safe_make_valid(gdf)

        if len(gdf) == 0:
            continue

        if gdf.crs is None:
            raise ValueError(f"Road file has no CRS: {shp_path}")

        gdf = gdf.to_crs(target_crs)
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

        if len(gdf) == 0:
            continue

        if read_extent is not None:
            gdf = gdf[gdf.geometry.intersects(read_extent)].copy()

        if len(gdf) == 0:
            continue

        gdf["build_year"] = year
        records.append(gdf[["build_year", "geometry"]])

    if len(records) == 0:
        raise RuntimeError("No road data were loaded.")

    roads = pd.concat(records, ignore_index=True)
    roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=target_crs)
    roads = safe_make_valid(roads)

    return roads


def explode_lines(gdf):
    """Explode MultiLineString to LineString."""
    rows = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        attrs = row.drop(labels="geometry").to_dict()

        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, LineString):
            rows.append({**attrs, "geometry": geom})

        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if part is not None and not part.is_empty and part.length > 0:
                    rows.append({**attrs, "geometry": part})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def split_line_to_segments(line, max_length_m):
    """Split a LineString into approximately equal-length segments."""
    if line is None or line.is_empty or line.length <= 0:
        return []

    if line.length <= max_length_m:
        return [line]

    n_parts = int(math.ceil(line.length / max_length_m))
    part_len = line.length / n_parts

    segments = []
    for i in range(n_parts):
        start_d = i * part_len
        end_d = min((i + 1) * part_len, line.length)

        try:
            seg = substring(line, start_d, end_d)
            if seg is not None and not seg.is_empty and seg.length > 0:
                segments.append(seg)
        except Exception:
            continue

    return segments


def prepare_event_roads(all_roads, hotspot_union):
    """
    Clip 2004–2020 new roads to hotspot grids and split into standardized segments.
    """
    print("\nPreparing event-road cohorts...")

    event_roads = all_roads[
        (all_roads["build_year"] >= COHORT_START_YEAR) &
        (all_roads["build_year"] <= COHORT_END_YEAR)
    ].copy()

    if len(event_roads) == 0:
        raise RuntimeError("No event roads found for 2004–2020.")

    print("Clipping event roads to hotspot grids...")
    event_roads["geometry"] = event_roads.geometry.intersection(hotspot_union)
    event_roads = safe_make_valid(event_roads)
    event_roads = event_roads[event_roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    print("Exploding event-road geometries...")
    event_roads = explode_lines(event_roads)

    print("Splitting event roads into segments...")
    rows = []
    counter = 0

    for _, row in event_roads.iterrows():
        build_year = int(row["build_year"])
        parts = split_line_to_segments(row.geometry, MAX_SEGMENT_LENGTH_M)

        for part in parts:
            if part.length < MIN_SEGMENT_LENGTH_M:
                continue

            counter += 1
            seg_id = f"R{build_year}_{counter:07d}"
            rows.append({
                "seg_id": seg_id,
                "build_year": build_year,
                "length_m": part.length,
                "length_km": part.length / 1000.0,
                "geometry": part
            })

    segs = gpd.GeoDataFrame(rows, geometry="geometry", crs=all_roads.crs)
    segs = safe_make_valid(segs)

    if len(segs) == 0:
        raise RuntimeError("No valid event-road segments after clipping.")

    segs.to_file(OUT_EVENT_ROADS, driver="GPKG")
    print(f"Number of event-road segments: {len(segs)}")
    print(f"Saved event-road segments: {OUT_EVENT_ROADS}")

    return segs


def read_landslide_points(target_crs, hotspot_union):
    """Read landslide points and keep points within hotspot grids and required years."""
    print("\nReading landslide points...")

    ls = gpd.read_file(LANDSLIDE_POINT_PATH)
    ls = safe_make_valid(ls)

    if ls.crs is None:
        raise ValueError("Landslide point shapefile has no CRS.")

    ls = ls.to_crs(target_crs)

    if "year" not in ls.columns:
        raise ValueError("The landslide point shapefile must contain a 'year' field.")

    ls["landslide_year"] = pd.to_numeric(ls["year"], errors="coerce")
    ls = ls.dropna(subset=["landslide_year"]).copy()
    ls["landslide_year"] = ls["landslide_year"].astype(int)

    min_year = COHORT_START_YEAR + TAU_MIN
    max_year = COHORT_END_YEAR + TAU_MAX

    ls = ls[
        (ls["landslide_year"] >= min_year) &
        (ls["landslide_year"] <= max_year)
    ].copy()

    print("Filtering landslide points to hotspot grids...")
    ls = ls[ls.geometry.intersects(hotspot_union)].copy()

    if "patch_id" not in ls.columns:
        ls["patch_id"] = [f"LS_{i:09d}" for i in range(len(ls))]

    ls = ls[["patch_id", "landslide_year", "geometry"]].copy()
    ls = gpd.GeoDataFrame(ls, geometry="geometry", crs=target_crs)

    ls.to_file(OUT_LANDSLIDES, driver="GPKG")
    print(f"Number of landslide points used: {len(ls)}")
    print(f"Saved landslide points: {OUT_LANDSLIDES}")

    return ls


def nearest_old_road_distance(points, old_roads, max_distance):
    """
    Calculate nearest distance from points to older roads within max_distance.
    If no older road is found within max_distance, distance is NaN.
    """
    if len(points) == 0:
        return pd.DataFrame(columns=["patch_id", "dist_old_m"])

    if old_roads is None or len(old_roads) == 0:
        out = points[["patch_id"]].copy()
        out["dist_old_m"] = np.nan
        return out

    joined = gpd.sjoin_nearest(
        points[["patch_id", "geometry"]],
        old_roads[["geometry"]],
        how="left",
        max_distance=max_distance,
        distance_col="dist_old_m"
    )

    joined = joined[["patch_id", "dist_old_m"]].copy()
    joined = joined.groupby("patch_id", as_index=False)["dist_old_m"].min()

    return joined



def nearest_available_year(year, available_years):
    """
    Pick the closest available source year for calendar years with no
    existing-road background source. Earlier source year is preferred when tied.
    """
    available_years = sorted(int(y) for y in available_years)
    return min(available_years, key=lambda y: (abs(y - int(year)), y))


def build_cohort_buffer_geometries(seg_y, hotspot_union):
    """
    Build cumulative near-road buffer zones for one road-construction cohort.

    Buffer zones:
      - buffer_300m:  hotspot area within 300 m of focal new roads
      - buffer_500m:  hotspot area within 500 m of focal new roads
      - buffer_1000m: hotspot area within 1000 m of focal new roads

    Background zones are NOT built here. They are built separately by
    calendar year from existing roads to avoid contamination from old roads
    and other road-construction years.
    """
    zone_geoms = {}
    area_records = []

    for zone, distance_m in BUFFER_ZONES:
        geom = unary_union(list(seg_y.geometry.buffer(distance_m))).intersection(hotspot_union)
        zone_geoms[zone] = geom
        area_records.append({
            "zone": zone,
            "zone_label": ZONE_LABELS.get(zone, zone),
            "zone_distance_m": distance_m,
            "area_km2": geom.area / 1_000_000.0,
            "background_source_year": np.nan,
            "background_source_mode": np.nan,
            "background_min_slope_deg": np.nan,
            "background_slope_filter": np.nan
        })

    return zone_geoms, pd.DataFrame(area_records)


def read_background_slope_raster(target_crs):
    """
    Open the slope raster in the analysis CRS.

    WarpedVRT keeps reprojection lazy: area and point operations read only the
    requested windows instead of allocating a full reprojected raster in memory.
    """
    if not os.path.exists(SLOPE_RASTER_PATH):
        raise FileNotFoundError(f"Slope raster not found: {SLOPE_RASTER_PATH}")

    target_raster_crs = rasterio.crs.CRS.from_user_input(target_crs)

    print("\nOpening background slope raster:")
    print(SLOPE_RASTER_PATH)
    print(f"Background slope condition: slope > {BACKGROUND_MIN_SLOPE_DEG:g} degrees.")

    src = rasterio.open(SLOPE_RASTER_PATH)

    if src.crs is None:
        src.close()
        raise ValueError("Slope raster has no CRS.")

    if src.crs == target_raster_crs:
        dataset = src
        vrt = None
        print("Slope raster already matches target CRS.")
    else:
        print("Using lazy WarpedVRT for slope raster reprojection to target CRS...")
        vrt = WarpedVRT(
            src,
            crs=target_raster_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            nodata=np.nan
        )
        dataset = vrt

    pixel_area_km2 = abs(dataset.transform.a * dataset.transform.e) / 1_000_000.0

    print(f"Slope raster view shape: {dataset.height} x {dataset.width}")
    print(f"Slope raster pixel area: {pixel_area_km2:.6f} km虏")

    return {
        "src": src,
        "vrt": vrt,
        "dataset": dataset,
        "transform": dataset.transform,
        "crs": dataset.crs,
        "pixel_area_km2": pixel_area_km2,
        "height": dataset.height,
        "width": dataset.width
    }


def close_background_slope_raster(slope_raster):
    """Close raster handles opened for the background slope filter."""
    if slope_raster is None:
        return

    vrt = slope_raster.get("vrt")
    src = slope_raster.get("src")

    if vrt is not None:
        vrt.close()
    if src is not None:
        src.close()


def background_slope_area_km2(background_geom, slope_raster, chunk_size=2048):
    """
    Calculate background area from raster pixels that are inside the road-distance
    background geometry and have slope > BACKGROUND_MIN_SLOPE_DEG.
    """
    if background_geom is None or background_geom.is_empty:
        return 0.0

    try:
        geom = background_geom.buffer(0)
    except Exception:
        geom = background_geom

    dataset = slope_raster["dataset"]

    try:
        raw_window = from_bounds(*geom.bounds, transform=dataset.transform)
    except Exception:
        return 0.0

    col_start = max(0, int(math.floor(raw_window.col_off)))
    row_start = max(0, int(math.floor(raw_window.row_off)))
    col_stop = min(dataset.width, int(math.ceil(raw_window.col_off + raw_window.width)))
    row_stop = min(dataset.height, int(math.ceil(raw_window.row_off + raw_window.height)))

    if col_stop <= col_start or row_stop <= row_start:
        return 0.0

    count = 0
    for row0 in range(row_start, row_stop, chunk_size):
        row1 = min(row0 + chunk_size, row_stop)

        for col0 in range(col_start, col_stop, chunk_size):
            col1 = min(col0 + chunk_size, col_stop)
            window = Window(
                col_off=col0,
                row_off=row0,
                width=col1 - col0,
                height=row1 - row0
            )
            out_shape = (int(window.height), int(window.width))
            transform = window_transform(window, dataset.transform)

            geom_mask = rasterize(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                all_touched=False,
                dtype="uint8"
            ).astype(bool)

            if not geom_mask.any():
                continue

            slope = dataset.read(
                1,
                window=window,
                masked=True,
                out_dtype="float32"
            )
            slope_values = slope.filled(np.nan)
            count += int(np.count_nonzero(geom_mask & np.isfinite(slope_values) & (slope_values > BACKGROUND_MIN_SLOPE_DEG)))

    return float(count * slope_raster["pixel_area_km2"])


def add_slope_values_to_points(points, slope_raster, slope_col="background_slope_deg"):
    """Sample slope raster values at point locations in the analysis CRS."""
    points = points.copy()

    if len(points) == 0:
        points[slope_col] = np.nan
        return points

    dataset = slope_raster["dataset"]
    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    rows, cols = rowcol(dataset.transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) &
        (rows < dataset.height) &
        (cols >= 0) &
        (cols < dataset.width)
    )

    values = np.full(len(points), np.nan, dtype="float64")
    valid_idx = np.where(inside)[0]

    if len(valid_idx) > 0:
        coords = [(float(xs[i]), float(ys[i])) for i in valid_idx]
        sampled = dataset.sample(coords)
        values[valid_idx] = [float(v[0]) if len(v) > 0 else np.nan for v in sampled]

    points[slope_col] = values
    return points


def read_existing_road_background_source(buffer_file, target_crs, source_label):
    """
    Read existing-road buffer source for calendar-year background zones.

    Preferred input:
      buffer_file
      required field: BACKGROUND_EXISTING_ROAD_YEAR_FIELD

    The layer can be dissolved or non-dissolved. It is dissolved by source year
    here before being used as the road-exclusion mask.
    """
    if not os.path.exists(buffer_file):
        print(f"[Warning] Existing-road {source_label} buffer file not found:")
        print(f"          {buffer_file}")
        return None

    print(f"\nReading existing-road {source_label} background buffers:")
    print(buffer_file)

    try:
        gdf = gpd.read_file(buffer_file)
    except UnicodeDecodeError as exc:
        print(f"[Warning] pyogrio failed to decode {source_label} buffer attributes: {exc}")
        print("          Trying Fiona engine...")
        try:
            gdf = gpd.read_file(buffer_file, engine="fiona")
        except Exception as fiona_exc:
            print(f"[Warning] Fiona also failed to read {source_label} buffer: {fiona_exc}")
            print("          This source will be skipped and fallback road buffers will be used.")
            return None
    except Exception as exc:
        print(f"[Warning] Failed to read existing-road {source_label} buffer: {exc}")
        print("          This source will be skipped and fallback road buffers will be used.")
        return None

    if BACKGROUND_EXISTING_ROAD_YEAR_FIELD not in gdf.columns:
        raise ValueError(
            f"Existing-road background source must contain field: "
            f"{BACKGROUND_EXISTING_ROAD_YEAR_FIELD}"
        )

    gdf = gdf[[BACKGROUND_EXISTING_ROAD_YEAR_FIELD, "geometry"]].copy()
    gdf = gdf.dropna(subset=[BACKGROUND_EXISTING_ROAD_YEAR_FIELD]).copy()
    gdf[BACKGROUND_EXISTING_ROAD_YEAR_FIELD] = (
        gdf[BACKGROUND_EXISTING_ROAD_YEAR_FIELD].astype(int)
    )

    gdf = safe_make_valid(gdf)

    if gdf.crs is None:
        raise ValueError("Existing-road background source has no CRS.")

    if gdf.crs != target_crs:
        print("Reprojecting existing-road background source to target CRS...")
        gdf = gdf.to_crs(target_crs)

    dissolved = (
        gdf
        .dissolve(by=BACKGROUND_EXISTING_ROAD_YEAR_FIELD, as_index=False)
        [[BACKGROUND_EXISTING_ROAD_YEAR_FIELD, "geometry"]]
        .copy()
    )
    dissolved = safe_make_valid(dissolved)
    dissolved = dissolved.rename(
        columns={BACKGROUND_EXISTING_ROAD_YEAR_FIELD: "background_source_year"}
    )
    dissolved["background_source_year"] = dissolved["background_source_year"].astype(int)

    print(
        "Existing-road background source years: "
        f"{dissolved['background_source_year'].min()}–"
        f"{dissolved['background_source_year'].max()}"
    )

    return dissolved


def build_background_zones_by_calendar_year(
    all_roads,
    hotspot_union,
    target_crs,
    slope_raster
):
    """
    Build dynamic background zones by real calendar year.

    For each calendar year y:
        background_y = hotspot_extent intersected with
                       buffer(existing_roads_y, BACKGROUND_OUTER_DISTANCE_M)
                       minus buffer(existing_roads_y, BACKGROUND_INNER_DISTANCE_M)
        area_km2 is counted from background_y raster pixels where
        slope > BACKGROUND_MIN_SLOPE_DEG.

    Preferred existing_roads_y source:
        annual existing-road buffer file, selected by source year nearest to y.

    Fallback:
        use all loaded annual new roads with build_year <= y, then buffer them.

    This prevents the background zone from containing roads built before, during
    or in other cohorts of the event-study window.
    """
    print("\nBuilding calendar-year 1-3 km background zones around existing roads...")

    min_calendar_year = COHORT_START_YEAR + TAU_MIN
    max_calendar_year = COHORT_END_YEAR + TAU_MAX
    min_calendar_year = max(min_calendar_year, ALL_ROAD_START_YEAR)
    max_calendar_year = min(max_calendar_year, ALL_ROAD_END_YEAR)

    existing_inner_source = read_existing_road_background_source(
        buffer_file=BACKGROUND_EXISTING_ROADS_INNER_BUFFER_FILE,
        target_crs=target_crs,
        source_label="inner 1 km"
    )
    if existing_inner_source is None or len(existing_inner_source) == 0:
        existing_outer_source = None
        print("[Warning] Inner 1 km source is unavailable.")
        print("          Skipping outer 2 km source and using fallback road buffers.")
    else:
        existing_outer_source = read_existing_road_background_source(
            buffer_file=BACKGROUND_EXISTING_ROADS_OUTER_BUFFER_FILE,
            target_crs=target_crs,
            source_label="outer 3 km"
        )

    rows = []

    if (
        existing_inner_source is not None and len(existing_inner_source) > 0 and
        existing_outer_source is not None and len(existing_outer_source) > 0
    ):
        inner_geom_by_year = dict(zip(
            existing_inner_source["background_source_year"].astype(int),
            existing_inner_source.geometry
        ))
        outer_geom_by_year = dict(zip(
            existing_outer_source["background_source_year"].astype(int),
            existing_outer_source.geometry
        ))
        available_years = sorted(set(inner_geom_by_year) & set(outer_geom_by_year))
        if len(available_years) == 0:
            available_years = sorted(set(inner_geom_by_year) | set(outer_geom_by_year))

        for calendar_year in range(min_calendar_year, max_calendar_year + 1):
            source_year = nearest_available_year(calendar_year, available_years)
            inner_source_year = nearest_available_year(calendar_year, inner_geom_by_year)
            outer_source_year = nearest_available_year(calendar_year, outer_geom_by_year)
            inner_geom = inner_geom_by_year[inner_source_year]
            outer_geom = outer_geom_by_year[outer_source_year]

            try:
                background_geom = outer_geom.difference(inner_geom).intersection(hotspot_union)
            except Exception:
                background_geom = (
                    outer_geom.buffer(0)
                    .difference(inner_geom.buffer(0))
                    .intersection(hotspot_union.buffer(0))
                )

            if background_geom is not None and not background_geom.is_empty:
                try:
                    background_geom = background_geom.buffer(0)
                except Exception:
                    pass
                area_km2 = background_slope_area_km2(background_geom, slope_raster)
            else:
                area_km2 = 0.0

            rows.append({
                "calendar_year": int(calendar_year),
                "zone": BACKGROUND_ZONE,
                "zone_label": ZONE_LABELS.get(BACKGROUND_ZONE, BACKGROUND_ZONE),
                "zone_distance_m": BACKGROUND_OUTER_DISTANCE_M,
                "background_source_year": int(outer_source_year),
                "background_source_mode": BACKGROUND_SOURCE_MODE,
                "background_min_slope_deg": BACKGROUND_MIN_SLOPE_DEG,
                "background_slope_filter": BACKGROUND_SLOPE_FILTER,
                "area_km2": area_km2,
                "geometry": background_geom
            })

    else:
        print("[Warning] Falling back to loaded annual new-road layers.")
        print("          If roads before 2000 are absent, the background may still include")
        print("          pre-2000 existing roads. Prefer the existing-road buffer sources.")

        for calendar_year in range(min_calendar_year, max_calendar_year + 1):
            existing_roads_y = all_roads[all_roads["build_year"] <= calendar_year].copy()

            if len(existing_roads_y) == 0:
                background_geom = None
            else:
                inner_geom = unary_union(
                    list(existing_roads_y.geometry.buffer(BACKGROUND_INNER_DISTANCE_M))
                )
                outer_geom = unary_union(
                    list(existing_roads_y.geometry.buffer(BACKGROUND_OUTER_DISTANCE_M))
                )
                background_geom = outer_geom.difference(inner_geom).intersection(hotspot_union)

            if background_geom is not None and not background_geom.is_empty:
                try:
                    background_geom = background_geom.buffer(0)
                except Exception:
                    pass
                area_km2 = background_slope_area_km2(background_geom, slope_raster)
            else:
                area_km2 = 0.0

            rows.append({
                "calendar_year": int(calendar_year),
                "zone": BACKGROUND_ZONE,
                "zone_label": ZONE_LABELS.get(BACKGROUND_ZONE, BACKGROUND_ZONE),
                "zone_distance_m": BACKGROUND_OUTER_DISTANCE_M,
                "background_source_year": int(calendar_year),
                "background_source_mode": "fallback_new_roads_le_calendar_year_annulus_1000_3000m",
                "background_min_slope_deg": BACKGROUND_MIN_SLOPE_DEG,
                "background_slope_filter": BACKGROUND_SLOPE_FILTER,
                "area_km2": area_km2,
                "geometry": background_geom
            })

    background = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    background = background[background.geometry.notna()].copy()
    background = background[~background.geometry.is_empty].copy()
    background = background[background["area_km2"] > 0].copy()
    background = background.sort_values("calendar_year").reset_index(drop=True)

    if len(background) == 0:
        raise RuntimeError("No valid calendar-year background zones were created.")

    background.to_file(OUT_BACKGROUND_ZONES, driver="GPKG")
    print(f"Saved calendar-year background zones: {OUT_BACKGROUND_ZONES}")
    print(f"Mean background area: {background['area_km2'].mean():.2f} km²")
    print("Background calendar/source year mapping:")
    print(background[[
        "calendar_year",
        "background_source_year",
        "background_source_mode",
        "background_slope_filter",
        "area_km2"
    ]].to_string(index=False))

    return background


def match_landslides_to_road_cohorts(
    event_segments,
    all_roads,
    landslides,
    hotspot_union,
    background_zones,
    slope_raster
):
    """
    For each construction-year cohort, count unique landslide points in multiple
    spatial zones around newly built roads for tau = TAU_MIN to TAU_MAX.

    Near-road zones are cumulative buffers: <=300 m, <=500 m and <=1000 m
    around focal new roads.

    Background zone is dynamic by real calendar year:
        hotspot area 1-3 km from existing roads
        in calendar_year = cohort_year + tau.

    This avoids contaminating the background with old roads or roads from
    other construction years.
    """
    print("\nMatching landslides to multi-distance zones and calendar-year backgrounds...")

    all_matches = []
    all_area_records = []

    bg_by_year = {
        int(row["calendar_year"]): row
        for _, row in background_zones.iterrows()
    }

    for cohort_year in range(COHORT_START_YEAR, COHORT_END_YEAR + 1):
        print(f"Processing cohort year: {cohort_year}")

        seg_y = event_segments[event_segments["build_year"] == cohort_year].copy()

        if len(seg_y) == 0:
            print("  No event-road segments.")
            continue

        zone_geoms, area_y = build_cohort_buffer_geometries(seg_y, hotspot_union)

        # Near-road buffer areas are fixed within a cohort, so repeat them for all tau.
        for _, area_row in area_y.iterrows():
            for tau in TAU_LIST:
                all_area_records.append({
                    "cohort_year": cohort_year,
                    "zone": area_row["zone"],
                    "zone_label": area_row["zone_label"],
                    "zone_distance_m": area_row["zone_distance_m"],
                    "tau": tau,
                    "calendar_year": cohort_year + tau,
                    "area_km2": area_row["area_km2"],
                    "background_source_year": np.nan,
                    "background_source_mode": np.nan,
                    "background_min_slope_deg": np.nan,
                    "background_slope_filter": np.nan
                })

        min_ls_year = cohort_year + TAU_MIN
        max_ls_year = cohort_year + TAU_MAX

        ls_y = landslides[
            (landslides["landslide_year"] >= min_ls_year) &
            (landslides["landslide_year"] <= max_ls_year)
        ].copy()

        if len(ls_y) == 0:
            print("  No landslides in event window.")
            continue

        old_roads = None
        if EXCLUDE_OLDER_ROADS:
            old_roads = all_roads[all_roads["build_year"] < cohort_year].copy()

        # ----------------------------------------------------
        # 1) Near-road cumulative buffers: <=300, <=500, <=1000 m
        # ----------------------------------------------------
        for zone, distance_m in BUFFER_ZONES:
            cand = gpd.sjoin_nearest(
                ls_y[["patch_id", "landslide_year", "geometry"]],
                seg_y[["seg_id", "build_year", "geometry"]],
                how="inner",
                max_distance=distance_m,
                distance_col="dist_event_m"
            )

            if len(cand) == 0:
                print(f"  {zone}: no landslides within {distance_m:.0f} m.")
                continue

            # Keep nearest segment within the same cohort and zone.
            cand = cand.sort_values(["patch_id", "dist_event_m"])
            cand = cand.drop_duplicates(subset=["patch_id"], keep="first")

            cand["cohort_year"] = cohort_year
            cand["tau"] = cand["landslide_year"] - cand["cohort_year"]
            cand = cand[(cand["tau"] >= TAU_MIN) & (cand["tau"] <= TAU_MAX)].copy()

            # Optional: exclude landslides also near older roads within the same distance.
            if EXCLUDE_OLDER_ROADS:
                old_dist = nearest_old_road_distance(
                    points=gpd.GeoDataFrame(
                        cand[["patch_id", "geometry"]].drop_duplicates("patch_id"),
                        geometry="geometry",
                        crs=landslides.crs
                    ),
                    old_roads=old_roads,
                    max_distance=distance_m
                )

                cand = cand.merge(old_dist, on="patch_id", how="left")
                cand = cand[cand["dist_old_m"].isna() | (cand["dist_old_m"] > distance_m)].copy()
            else:
                cand["dist_old_m"] = np.nan

            if len(cand) == 0:
                print(f"  {zone}: all candidates removed by older-road exclusion.")
                continue

            cand["zone"] = zone
            cand["zone_label"] = ZONE_LABELS.get(zone, zone)
            cand["zone_distance_m"] = distance_m
            cand["calendar_year"] = cand["landslide_year"].astype(int)
            cand["background_source_year"] = np.nan
            cand["background_source_mode"] = np.nan
            cand["background_min_slope_deg"] = np.nan
            cand["background_slope_filter"] = np.nan
            cand["background_slope_deg"] = np.nan

            cand = cand[[
                "patch_id",
                "landslide_year",
                "cohort_year",
                "tau",
                "calendar_year",
                "zone",
                "zone_label",
                "zone_distance_m",
                "seg_id",
                "dist_event_m",
                "dist_old_m",
                "background_source_year",
                "background_source_mode",
                "background_min_slope_deg",
                "background_slope_filter",
                "background_slope_deg"
            ]].copy()

            print(f"  {zone}: matched unique landslides = {len(cand)}")
            all_matches.append(cand)

        # ----------------------------------------------------
        # 2) Background: calendar-year existing-road background
        # ----------------------------------------------------
        for tau in TAU_LIST:
            calendar_year = cohort_year + tau

            if calendar_year not in bg_by_year:
                continue

            bg_row = bg_by_year[calendar_year]
            background_geom = bg_row.geometry

            all_area_records.append({
                "cohort_year": cohort_year,
                "zone": BACKGROUND_ZONE,
                "zone_label": ZONE_LABELS.get(BACKGROUND_ZONE, BACKGROUND_ZONE),
                "zone_distance_m": BACKGROUND_OUTER_DISTANCE_M,
                "tau": tau,
                "calendar_year": calendar_year,
                "area_km2": float(bg_row["area_km2"]),
                "background_source_year": int(bg_row["background_source_year"]),
                "background_source_mode": str(bg_row["background_source_mode"]),
                "background_min_slope_deg": float(bg_row["background_min_slope_deg"]),
                "background_slope_filter": str(bg_row["background_slope_filter"])
            })

            if background_geom is None or background_geom.is_empty:
                print(f"  {BACKGROUND_ZONE}, tau={tau}: empty background geometry.")
                continue

            bg = ls_y[ls_y["landslide_year"] == calendar_year].copy()

            if len(bg) == 0:
                continue

            # Points should be inside the calendar-year background zone.
            bg = bg[bg.geometry.within(background_geom)].copy()

            if len(bg) == 0:
                continue

            bg = add_slope_values_to_points(
                bg,
                slope_raster,
                slope_col="background_slope_deg"
            )
            bg = bg[bg["background_slope_deg"] > BACKGROUND_MIN_SLOPE_DEG].copy()

            if len(bg) == 0:
                continue

            bg["cohort_year"] = cohort_year
            bg["tau"] = tau
            bg["calendar_year"] = calendar_year
            bg["zone"] = BACKGROUND_ZONE
            bg["zone_label"] = ZONE_LABELS.get(BACKGROUND_ZONE, BACKGROUND_ZONE)
            bg["zone_distance_m"] = BACKGROUND_OUTER_DISTANCE_M
            bg["seg_id"] = np.nan
            bg["dist_event_m"] = np.nan
            bg["dist_old_m"] = np.nan
            bg["background_source_year"] = int(bg_row["background_source_year"])
            bg["background_source_mode"] = str(bg_row["background_source_mode"])
            bg["background_min_slope_deg"] = float(bg_row["background_min_slope_deg"])
            bg["background_slope_filter"] = str(bg_row["background_slope_filter"])

            bg = bg[[
                "patch_id",
                "landslide_year",
                "cohort_year",
                "tau",
                "calendar_year",
                "zone",
                "zone_label",
                "zone_distance_m",
                "seg_id",
                "dist_event_m",
                "dist_old_m",
                "background_source_year",
                "background_source_mode",
                "background_min_slope_deg",
                "background_slope_filter",
                "background_slope_deg"
            ]].copy()

            all_matches.append(bg)

        print(f"  {BACKGROUND_ZONE}: calendar-year background processed.")

    if len(all_area_records) == 0:
        raise RuntimeError("No cohort-zone area records were created.")

    zone_areas = pd.DataFrame(all_area_records)
    zone_areas = zone_areas[[
        "cohort_year",
        "zone",
        "zone_label",
        "zone_distance_m",
        "tau",
        "calendar_year",
        "area_km2",
        "background_source_year",
        "background_source_mode",
        "background_min_slope_deg",
        "background_slope_filter"
    ]].copy()

    # Avoid duplicate area rows caused by repeated appends.
    zone_areas = zone_areas.drop_duplicates(
        subset=["cohort_year", "zone", "tau", "calendar_year"],
        keep="first"
    )

    zone_areas.to_csv(OUT_ZONE_AREAS, index=False, encoding="utf-8-sig")
    print(f"\nSaved cohort-zone area table: {OUT_ZONE_AREAS}")

    if len(all_matches) == 0:
        raise RuntimeError("No landslide-road/background matches found.")

    matches = pd.concat(all_matches, ignore_index=True)

    # Keep one record per landslide per cohort, spatial zone and tau.
    matches = matches.sort_values(
        ["cohort_year", "zone", "tau", "patch_id", "dist_event_m"],
        na_position="last"
    )
    matches = matches.drop_duplicates(
        subset=["cohort_year", "zone", "tau", "patch_id"],
        keep="first"
    )

    matches.to_csv(OUT_MATCHES, index=False, encoding="utf-8-sig")
    print(f"\nTotal multizone matches: {len(matches)}")
    print(f"Saved match table: {OUT_MATCHES}")

    return matches, zone_areas


def build_cohort_tau_counts(event_segments, matches, zone_areas):
    """
    Build cohort × tau × zone table:
      - landslide_count
      - zone area_km2
      - landslide_density_per_km2
      - landslide_density_per_100km2
      - pre-normalized density value

    Background area is merged by cohort_year × tau × calendar_year because
    it varies with real calendar year under the existing-road background mode.
    """
    print("\nBuilding cohort × tau × zone density table...")

    # Total new-road length for each cohort year, retained for reference only.
    road_length = (
        event_segments
        .groupby("build_year")["length_km"]
        .sum()
        .reset_index()
        .rename(columns={
            "build_year": "cohort_year",
            "length_km": "road_length_km"
        })
    )

    # Count unique landslides by cohort, tau and spatial zone.
    counts = (
        matches
        .groupby(["cohort_year", "zone", "tau"])["patch_id"]
        .nunique()
        .reset_index(name="landslide_count")
    )

    # Complete cohort × tau × zone table.
    rows = []
    for cohort_year in range(COHORT_START_YEAR, COHORT_END_YEAR + 1):
        for zone in ZONE_ORDER:
            for tau in TAU_LIST:
                rows.append({
                    "cohort_year": cohort_year,
                    "zone": zone,
                    "tau": tau,
                    "calendar_year": cohort_year + tau
                })

    table = pd.DataFrame(rows)

    table = table.merge(road_length, on="cohort_year", how="left")
    table = table.merge(
        zone_areas,
        on=["cohort_year", "zone", "tau", "calendar_year"],
        how="left"
    )
    table = table.merge(counts, on=["cohort_year", "zone", "tau"], how="left")

    table["landslide_count"] = table["landslide_count"].fillna(0).astype(int)
    table = table.dropna(subset=["road_length_km", "area_km2"]).copy()

    table["landslide_density_per_km2"] = np.where(
        table["area_km2"] > 0,
        table["landslide_count"] / table["area_km2"],
        np.nan
    )
    table["landslide_density_per_100km2"] = (
        table["landslide_density_per_km2"] * DENSITY_SCALE_KM2
    )

    # Pre-mean normalization based on PRE_TAUS.
    pre_mean = (
        table[table["tau"].isin(PRE_TAUS)]
        .groupby(["cohort_year", "zone"])["landslide_density_per_100km2"]
        .mean()
        .reset_index(name="pre_mean_density_per_100km2")
    )

    table = table.merge(pre_mean, on=["cohort_year", "zone"], how="left")

    table["pre_normalized_density"] = np.where(
        table["pre_mean_density_per_100km2"] > 0,
        table["landslide_density_per_100km2"] / table["pre_mean_density_per_100km2"],
        np.nan
    )

    table.to_csv(OUT_COHORT_COUNTS, index=False, encoding="utf-8-sig")
    print(f"Saved cohort density table: {OUT_COHORT_COUNTS}")

    return table

def summarize_pre_post(table):
    """
    Summarize Pre/Post density values for each cohort year and spatial zone.
    tau = 0 is excluded from Post.
    """
    print("\nSummarizing Pre/Post density values...")

    pre = (
        table[table["tau"].isin(PRE_TAUS)]
        .groupby(["cohort_year", "zone", "zone_label"])["landslide_density_per_100km2"]
        .mean()
        .reset_index(name="pre_density_per_100km2")
    )

    post = (
        table[table["tau"].isin(POST_TAUS)]
        .groupby(["cohort_year", "zone", "zone_label"])["landslide_density_per_100km2"]
        .mean()
        .reset_index(name="post_density_per_100km2")
    )

    out = pre.merge(post, on=["cohort_year", "zone", "zone_label"], how="outer")
    out["post_minus_pre_density_per_100km2"] = out["post_density_per_100km2"] - out["pre_density_per_100km2"]
    out["post_div_pre_density"] = np.where(
        out["pre_density_per_100km2"] > 0,
        out["post_density_per_100km2"] / out["pre_density_per_100km2"],
        np.nan
    )

    area_table = (
        table[["cohort_year", "zone", "zone_label", "zone_distance_m", "area_km2", "road_length_km"]]
        .drop_duplicates()
        .copy()
    )

    out = out.merge(area_table, on=["cohort_year", "zone", "zone_label"], how="left")

    raw_pre = (
        table[table["tau"].isin(PRE_TAUS)]
        .groupby(["cohort_year", "zone"])["landslide_count"]
        .mean()
        .reset_index(name="pre_mean_raw_count")
    )

    raw_post = (
        table[table["tau"].isin(POST_TAUS)]
        .groupby(["cohort_year", "zone"])["landslide_count"]
        .mean()
        .reset_index(name="post_mean_raw_count")
    )

    out = out.merge(raw_pre, on=["cohort_year", "zone"], how="left")
    out = out.merge(raw_post, on=["cohort_year", "zone"], how="left")
    out["post_minus_pre_raw_count"] = out["post_mean_raw_count"] - out["pre_mean_raw_count"]

    out.to_csv(OUT_COHORT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"Saved cohort summary table: {OUT_COHORT_SUMMARY}")
    print("\n========== Overall cohort summary by zone ==========")
    for zone in ZONE_ORDER:
        sub = out[out["zone"] == zone]
        if len(sub) == 0:
            continue
        label = ZONE_LABELS.get(zone, zone)
        print(
            f"{label}: mean post - pre density = "
            f"{sub['post_minus_pre_density_per_100km2'].mean():.6f}; "
            f"median post/pre = {sub['post_div_pre_density'].median():.3f}"
        )

    return out


# ============================================================
# 5. PLOTTING FUNCTIONS
# ============================================================

def setup_nature_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })


def plot_version1_density_curves(table):
    """
    Version 1:
    Each panel is one spatial zone.
    Each line is one construction-year cohort.
    y = landslide density, expressed as count per 100 km² per year.
    """
    print("\nPlotting Version 1: density curves by zone...")

    setup_nature_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True)
    axes = axes.ravel()

    years = sorted(table["cohort_year"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(years), max(years))

    for ax, zone in zip(axes, ZONE_ORDER):
        ztab = table[table["zone"] == zone].copy()

        for year in years:
            sub = ztab[ztab["cohort_year"] == year].sort_values("tau")
            if len(sub) == 0:
                continue

            ax.plot(
                sub["tau"],
                sub["landslide_density_per_100km2"],
                linewidth=0.9,
                alpha=0.75,
                color=cmap(norm(year))
            )

        ax.axvline(0, color="0.35", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.75", linestyle=":", linewidth=0.7)
        ax.set_title(ZONE_LABELS.get(zone, zone))
        ax.set_xticks(TAU_LIST)
        ax.set_xlim(TAU_MIN - 0.2, TAU_MAX + 0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].set_xlabel("Years relative to road construction")
    axes[3].set_xlabel("Years relative to road construction")
    axes[0].set_ylabel("Landslide density\n(count per 100 km² yr⁻¹)")
    axes[2].set_ylabel("Landslide density\n(count per 100 km² yr⁻¹)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.035)
    cbar.set_label("Road construction year")

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    # fig.savefig(OUT_FIG_DENSITY_PDF, dpi=FIG_DPI)
    fig.savefig(OUT_FIG_DENSITY_PNG, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved: {OUT_FIG_DENSITY_PNG}")


def plot_version2_pre_normalized(table):
    """
    Version 2:
    Each panel is one spatial zone.
    Each line is one construction-year cohort.
    y = density / mean(pre-construction density).
    """
    print("\nPlotting Version 2: pre-normalized density curves by zone...")

    setup_nature_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True)
    axes = axes.ravel()

    years = sorted(table["cohort_year"].unique())
    cmap = plt.cm.plasma
    norm = plt.Normalize(min(years), max(years))

    for ax, zone in zip(axes, ZONE_ORDER):
        ztab = table[table["zone"] == zone].copy()

        for year in years:
            sub = ztab[ztab["cohort_year"] == year].sort_values("tau")
            if len(sub) == 0:
                continue

            # Skip cohorts with zero pre-mean.
            if sub["pre_normalized_density"].isna().all():
                continue

            ax.plot(
                sub["tau"],
                sub["pre_normalized_density"],
                linewidth=0.9,
                alpha=0.75,
                color=cmap(norm(year))
            )

        ax.axvline(0, color="0.35", linestyle="--", linewidth=0.8)
        ax.axhline(1, color="0.75", linestyle=":", linewidth=0.7)
        ax.set_title(ZONE_LABELS.get(zone, zone))
        ax.set_xticks(TAU_LIST)
        ax.set_xlim(TAU_MIN - 0.2, TAU_MAX + 0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].set_xlabel("Years relative to road construction")
    axes[3].set_xlabel("Years relative to road construction")
    axes[0].set_ylabel("Relative landslide density\n(normalized by pre-construction mean)")
    axes[2].set_ylabel("Relative landslide density\n(normalized by pre-construction mean)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.035)
    cbar.set_label("Road construction year")

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    # fig.savefig(OUT_FIG_PRE_NORM_PDF, dpi=FIG_DPI)
    fig.savefig(OUT_FIG_PRE_NORM_PNG, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved: {OUT_FIG_PRE_NORM_PNG}")


def plot_mean_curve_optional(table):
    """
    Generate mean density curves across cohorts for each spatial zone.
    """
    print("\nSaving and plotting mean density curves across cohorts...")

    summary = (
        table
        .groupby(["zone", "zone_label", "tau"])
        .agg(
            n_cohorts=("cohort_year", "nunique"),
            mean_landslide_count=("landslide_count", "mean"),
            median_landslide_count=("landslide_count", "median"),
            mean_area_km2=("area_km2", "mean"),
            mean_density_per_km2=("landslide_density_per_km2", "mean"),
            median_density_per_km2=("landslide_density_per_km2", "median"),
            mean_density_per_100km2=("landslide_density_per_100km2", "mean"),
            median_density_per_100km2=("landslide_density_per_100km2", "median"),
            q25_density_per_100km2=("landslide_density_per_100km2", lambda s: s.quantile(0.25)),
            q75_density_per_100km2=("landslide_density_per_100km2", lambda s: s.quantile(0.75)),
            sd_density_per_100km2=("landslide_density_per_100km2", "std"),
            mean_pre_normalized_density=("pre_normalized_density", "mean"),
            median_pre_normalized_density=("pre_normalized_density", "median")
        )
        .reset_index()
    )

    summary["sem_density_per_100km2"] = summary["sd_density_per_100km2"] / np.sqrt(summary["n_cohorts"])

    # 95% confidence interval across road-construction cohorts.
    # If you prefer a wider variability envelope, replace this with:
    # summary["ci95_density_per_100km2"] = summary["sd_density_per_100km2"]
    summary["ci95_density_per_100km2"] = 1.96 * summary["sem_density_per_100km2"]

    summary.to_csv(OUT_MEAN_DENSITY, index=False, encoding="utf-8-sig")
    print(f"Saved density summary table: {OUT_MEAN_DENSITY}")

    setup_nature_style()

    fig, ax = plt.subplots(figsize=(7.6, 4.3))

    for zone in ZONE_ORDER:
        sub = summary[summary["zone"] == zone].sort_values("tau")
        if len(sub) == 0:
            continue

        style = ZONE_PLOT_STYLE.get(zone, {
            "color": "0.3",
            "marker": "o",
            "linestyle": "-",
            "alpha_fill": 0.10
        })

        x = sub["tau"].to_numpy()
        y = sub["median_density_per_100km2"].to_numpy()
        ci = sub["ci95_density_per_100km2"].fillna(0).to_numpy()

        y_lower = np.maximum(y - ci, 0)
        y_upper = y + ci

        # Shaded uncertainty envelope
        ax.fill_between(
            x,
            y_lower,
            y_upper,
            color=style["color"],
            alpha=style["alpha_fill"],
            linewidth=0
        )

        # Mean density curve
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            label=ZONE_LABELS.get(zone, zone)
        )

    # Road construction year
    ax.axvline(
        0,
        color="0.35",
        linestyle=":",
        linewidth=1.3,
        zorder=0
    )

    ax.set_xlabel("Years relative to road construction")
    ax.set_ylabel("Landslide density\n(polygons per 100 km$^2$ yr$^{-1}$)")

    ax.set_xticks(TAU_LIST)
    ax.set_xlim(TAU_MIN - 0.8, TAU_MAX + 0.8)

    # Optional: adjust y-axis lower bound to make the background curve clearer.
    ax.set_ylim(0,7)

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=2.2,
        handletextpad=0.8,
        borderaxespad=0.6
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    # fig.savefig(OUT_FIG_MEAN_DENSITY_PDF, dpi=FIG_DPI)
    fig.savefig(OUT_FIG_MEAN_DENSITY_PNG, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# 6. MAIN WORKFLOW
# ============================================================

def main():
    print("\n============================================================")
    print("Cohort-level landslide density curves around new roads")
    print(f"Cohorts: {COHORT_START_YEAR}–{COHORT_END_YEAR}")
    print(f"Event window: {TAU_MIN} to {TAU_MAX}")
    print("Near-road zones: 300 m, 500 m, 1000 m")
    print(
        f"Background zone: {BACKGROUND_INNER_DISTANCE_M / 1000:g}-"
        f"{BACKGROUND_OUTER_DISTANCE_M / 1000:g} km from existing roads "
        f"by calendar year and slope > {BACKGROUND_MIN_SLOPE_DEG:g} degrees"
    )
    print("Density unit: landslide count per 100 km² per year")
    print("============================================================\n")
    
        # --------------------------------------------------------
    # Fast mode: if previous result table exists, directly plot
    # --------------------------------------------------------
    existing_table = try_load_existing_density_table()

    if existing_table is not None:
        plot_mean_curve_optional(existing_table)

        print("\n============================================================")
        print("Completed successfully using existing results.")
        print("Regenerated figures:")
        print(f"  {OUT_FIG_MEAN_DENSITY_PNG}")
        print("============================================================\n")

        return

    # --------------------------------------------------------
    # Step 1. Read hotspot grid
    # --------------------------------------------------------
    hot, hotspot_union = read_hotspot_grid()
    target_crs = hot.crs

    # Background areas and points are restricted to slope > BACKGROUND_MIN_SLOPE_DEG.
    slope_raster = read_background_slope_raster(target_crs=target_crs)

    # Read road data slightly outside hotspot grids to support older-road exclusion.
    read_extent = hotspot_union.buffer(NEAR_DISTANCE_M + 1000.0)

    # --------------------------------------------------------
    # Step 2. Read annual roads
    # --------------------------------------------------------
    all_roads = read_yearly_roads(
        year_start=ALL_ROAD_START_YEAR,
        year_end=ALL_ROAD_END_YEAR,
        target_crs=target_crs,
        read_extent=read_extent
    )

    all_roads = explode_lines(all_roads)
    all_roads = safe_make_valid(all_roads)

    # Optional geometry simplification for speed
    # 5–10 m simplification usually has negligible effect for 300–1000 m buffer logic.
    all_roads["geometry"] = all_roads.geometry.simplify(
        tolerance=5,
        preserve_topology=True
    )

    # --------------------------------------------------------
    # Step 3. Prepare event-road segments
    # --------------------------------------------------------
    event_segments = prepare_event_roads(
        all_roads=all_roads,
        hotspot_union=hotspot_union
    )

    # --------------------------------------------------------
    # Step 4. Read landslide points
    # --------------------------------------------------------
    landslides = read_landslide_points(
        target_crs=target_crs,
        hotspot_union=hotspot_union
    )

    # --------------------------------------------------------
    # Step 5. Build calendar-year background zones
    # --------------------------------------------------------
    background_zones = build_background_zones_by_calendar_year(
        all_roads=all_roads,
        hotspot_union=hotspot_union,
        target_crs=target_crs,
        slope_raster=slope_raster
    )

    # --------------------------------------------------------
    # Step 6. Match landslides to multi-distance zones
    # --------------------------------------------------------
    matches, zone_areas = match_landslides_to_road_cohorts(
        event_segments=event_segments,
        all_roads=all_roads,
        landslides=landslides,
        hotspot_union=hotspot_union,
        background_zones=background_zones,
        slope_raster=slope_raster
    )

    # --------------------------------------------------------
    # Step 6. Build cohort × tau × zone density table
    # --------------------------------------------------------
    table = build_cohort_tau_counts(
        event_segments=event_segments,
        matches=matches,
        zone_areas=zone_areas
    )

    # --------------------------------------------------------
    # Step 7. Summarize Pre/Post density
    # --------------------------------------------------------
    summarize_pre_post(table)

    # --------------------------------------------------------
    # Step 8. Plot median density figure and save average density table
    # --------------------------------------------------------
    plot_mean_curve_optional(table)

    close_background_slope_raster(slope_raster)

    print("\n============================================================")
    print("Completed successfully.")
    print("Main outputs:")
    print(f"  {OUT_ZONE_AREAS}")
    print(f"  {OUT_BACKGROUND_ZONES}")
    print(f"  {OUT_COHORT_COUNTS}")
    print(f"  {OUT_COHORT_SUMMARY}")
    print(f"  {OUT_MEAN_DENSITY}")
    print(f"  {OUT_FIG_MEAN_DENSITY_PNG}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
