# Himalaya Landslide Hex Variation Map (ΔDensity, MK Significant Increase Black Dots)
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymannkendall as mk

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.ops import unary_union
from shapely.geometry import box, LineString
from matplotlib.patches import Rectangle

# =========================================================
# 1. USER SETTINGS
# =========================================================

POINT_SHP = r"H:/Himalaya/13w_landslides_list_final_points.shp"
GRID_SHP  = r"H:/Himalaya/grid/Himalaya_hex_1000km2/Himalaya_hex_1000km2.shp"

OUT_PNG = r"H:/Himalaya/figure/himalaya_hex_delta_density_mk.png"
OUT_PDF = r"H:/Himalaya/figure/himalaya_hex_delta_density_mk.pdf"

YEAR_FIELD = "year"

LON_MIN, LON_MAX = 74.0, 97.5
LAT_MIN, LAT_MAX = 25.0, 36.0

START_YEAR = 2000
END_YEAR = 2024
ALL_YEARS = list(range(START_YEAR, END_YEAR + 1))

PERIOD1 = list(range(2000, 2020))   # 2000-2019
PERIOD2 = list(range(2020, 2025))   # 2020-2024

# MK significance thresholds
N_thre = 250
P_thre = 0.05

USE_WEIGHT_FIELD = False
WEIGHT_FIELD = None
POINT_WEIGHT = 1.0

FIG_W, FIG_H = 13, 6
DPI = 600

DATA_CRS = "EPSG:4326"
MAP_PROJ = ccrs.PlateCarree()

MAP_CMAP = plt.cm.RdBu_r

class EsriTerrainTiles(GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def ensure_wgs84(gdf):
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS.")
    if gdf.crs.to_string() != DATA_CRS:
        gdf = gdf.to_crs(DATA_CRS)
    return gdf


def crop_to_extent(gdf, lon_min, lon_max, lat_min, lat_max):
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    return gdf[gdf.geometry.intersects(bbox)].copy()


def add_locator_map(ax_main, extent_main, inset_loc=(0.73, 0.73, 0.23, 0.22),
                    region_extent=(68, 105, 5, 40)):
    ax_inset = ax_main.inset_axes(inset_loc, projection=ccrs.PlateCarree())
    ax_inset.set_extent(region_extent, crs=ccrs.PlateCarree())
    ax_inset.set_facecolor("white")

    ax_inset.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.96", edgecolor="none", zorder=0)
    ax_inset.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
    ax_inset.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.35, edgecolor="0.5", zorder=1)
    ax_inset.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.6", zorder=1)

    xmin, xmax, ymin, ymax = extent_main
    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        facecolor=(1, 0, 0, 0.18),
        edgecolor="#b22222",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=3
    )
    ax_inset.add_patch(rect)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")

    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.35")

    return ax_inset


def add_scalebar_lonlat(ax, total_km=300, divisions=3, location=(0.82, 0.11),
                        linewidth=1.2, tick_height=0.10, fontsize=11):
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())

    lat_bar = lat0 + (lat1 - lat0) * location[1]
    lon_bar_start = lon0 + (lon1 - lon0) * location[0]

    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_bar))
    total_deg = total_km / km_per_deg_lon
    seg_deg = total_deg / divisions
    seg_km = total_km / divisions

    for i in range(divisions):
        x0 = lon_bar_start + i * seg_deg
        x1 = x0 + seg_deg
        color = "black" if i % 2 == 0 else "white"
        ax.fill(
            [x0, x1, x1, x0],
            [lat_bar, lat_bar, lat_bar + tick_height, lat_bar + tick_height],
            transform=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=10
        )

    for i in range(divisions + 1):
        x = lon_bar_start + i * seg_deg
        ax.text(
            x, lat_bar + tick_height + 0.12,
            f"{int(i * seg_km)}",
            transform=ccrs.PlateCarree(),
            ha="center", va="bottom",
            fontsize=fontsize, color="0.15", zorder=10
        )

    ax.text(
        lon_bar_start + total_deg + seg_deg * 0.25,
        lat_bar + tick_height + 0.12,
        "km",
        transform=ccrs.PlateCarree(),
        ha="left", va="bottom",
        fontsize=fontsize, color="0.15", zorder=10
    )


def add_north_arrow(ax, location=(0.04, 0.90), size=0.08):
    x, y = location
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="0.15",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.15"),
        zorder=11
    )


def build_hex_mk_and_delta(grid_gdf, point_gdf, year_field, all_years,
                           period1, period2, n_thre=250, p_thre=0.05):
    """
    Statistically count the annual landslide numbers for each hex and calculate:
    1) MK slope / p_value / sig_increase
    2) ΔDensity = mean(2020-2024) - mean(2000-2019)

    Returns
    -------
    grid : GeoDataFrame
    annual_counts : DataFrame
    """
    grid = grid_gdf.copy().reset_index(drop=True)
    grid["grid_id"] = np.arange(len(grid))

    pts = point_gdf.copy()

    joined = gpd.sjoin(
        pts[[year_field, "_weight_", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within"
    )

    joined = joined.dropna(subset=["grid_id"]).copy()
    joined["grid_id"] = joined["grid_id"].astype(int)

    if USE_WEIGHT_FIELD:
        annual_counts = (
            joined.groupby(["grid_id", year_field])["_weight_"]
            .sum()
            .unstack(fill_value=0)
        )
    else:
        annual_counts = (
            joined.groupby(["grid_id", year_field])
            .size()
            .unstack(fill_value=0)
        )

    # Fill in the years and grids
    annual_counts = annual_counts.reindex(columns=all_years, fill_value=0)
    annual_counts = annual_counts.reindex(grid["grid_id"], fill_value=0)

    # -------- MK --------
    total_sums = annual_counts.sum(axis=1).values
    slopes = []
    p_values = []

    for i in range(len(grid)):
        series = annual_counts.iloc[i].values.astype(float)

        if total_sums[i] > n_thre:
            try:
                res = mk.original_test(series)
                slopes.append(res.slope)
                p_values.append(res.p)
            except:
                slopes.append(0.0)
                p_values.append(1.0)
        else:
            slopes.append(0.0)
            p_values.append(1.0)

    grid["total_sum"] = total_sums
    grid["slope"] = slopes
    grid["p_value"] = p_values

    grid["sig_increase"] = (
        (grid["total_sum"] > n_thre) &
        (grid["p_value"] < p_thre) &
        (grid["slope"] > 0)
    )

    # -------- ΔDensity --------
    mean_2000_2019 = annual_counts[period1].mean(axis=1)
    mean_2020_2024 = annual_counts[period2].mean(axis=1)

    grid["mean_2000_2019"] = mean_2000_2019.values
    grid["mean_2020_2024"] = mean_2020_2024.values
    grid["delta_density"] = grid["mean_2020_2024"] - grid["mean_2000_2019"]

    return grid, annual_counts


# =========================================================
# 3. READ DATA
# =========================================================

gdf = gpd.read_file(POINT_SHP)
gdf = ensure_wgs84(gdf)

if YEAR_FIELD not in gdf.columns:
    raise ValueError(f"Field '{YEAR_FIELD}' not found in point shapefile.")

gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[gdf.is_valid].copy()
gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
gdf = gdf.explode(index_parts=False).reset_index(drop=True)

gdf = gdf[gdf[YEAR_FIELD].notna()].copy()
gdf[YEAR_FIELD] = pd.to_numeric(gdf[YEAR_FIELD], errors="coerce")
gdf = gdf[gdf[YEAR_FIELD].notna()].copy()
gdf[YEAR_FIELD] = gdf[YEAR_FIELD].astype(int)

gdf = gdf[(gdf[YEAR_FIELD] >= START_YEAR) & (gdf[YEAR_FIELD] <= END_YEAR)].copy()
gdf = crop_to_extent(gdf, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

if len(gdf) == 0:
    raise ValueError("No points remain after cropping to study area extent.")

if USE_WEIGHT_FIELD:
    if WEIGHT_FIELD is None or WEIGHT_FIELD not in gdf.columns:
        raise ValueError("USE_WEIGHT_FIELD=True but WEIGHT_FIELD is invalid.")
    gdf["_weight_"] = pd.to_numeric(gdf[WEIGHT_FIELD], errors="coerce").fillna(0.0)
else:
    gdf["_weight_"] = float(POINT_WEIGHT)

grid_gdf = gpd.read_file(GRID_SHP)
grid_gdf = ensure_wgs84(grid_gdf)
grid_gdf = crop_to_extent(grid_gdf, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

if len(grid_gdf) == 0:
    raise ValueError("No grid polygons remain after cropping to study area extent.")

study_union = unary_union(grid_gdf.geometry).buffer(0)

if study_union.geom_type == "Polygon":
    study_outline_geoms = [LineString(study_union.exterior.coords)]
elif study_union.geom_type == "MultiPolygon":
    study_outline_geoms = [LineString(poly.exterior.coords) for poly in study_union.geoms]
else:
    study_outline_geoms = []

# MK + ΔDensity
grid_plot, annual_counts = build_hex_mk_and_delta(
    grid_gdf, gdf, YEAR_FIELD, ALL_YEARS,
    PERIOD1, PERIOD2,
    n_thre=N_thre, p_thre=P_thre
)

sig_hex = grid_plot[grid_plot["sig_increase"]].copy()
sig_count = len(sig_hex)

print(f"Number of hexagons with significant increasing trend: {sig_count}")

if sig_count > 0:
    sig_hex["marker_geom"] = sig_hex.geometry.representative_point()

# =========================================================
# 4. FIGURE
# =========================================================

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax_map = fig.add_subplot(1, 1, 1, projection=MAP_PROJ)

FRAME_PAD_X = 0.8   
FRAME_PAD_Y = 1.5   

ax_map.set_extent(
    [LON_MIN - FRAME_PAD_X, LON_MAX + FRAME_PAD_X,
     LAT_MIN - FRAME_PAD_Y, LAT_MAX + FRAME_PAD_Y],
    crs=ccrs.PlateCarree()
)

ax_map.set_facecolor("white")


BASEMAP_ZOOM = 7 

terrain_tiles = EsriTerrainTiles()
ax_map.add_image(terrain_tiles, BASEMAP_ZOOM, zorder=0)

COAST_LW = 0.25
BORDER_LW = 0.30

ax_map.add_feature(
    cfeature.COASTLINE.with_scale("50m"),
    linewidth=COAST_LW, edgecolor="0.6", zorder=1
)
ax_map.add_feature(
    cfeature.BORDERS.with_scale("50m"),
    linewidth=BORDER_LW, edgecolor="0.6", zorder=1
)

# =========================================================
# 5. PLOT ΔDensity
# =========================================================

valid_vals = grid_plot["delta_density"].replace([np.inf, -np.inf], np.nan).dropna()

if len(valid_vals) > 0:
    raw_min = float(valid_vals.min())
    raw_max = float(valid_vals.max())

    clip_min = float(np.nanpercentile(valid_vals, 2))
    clip_max = float(np.nanpercentile(valid_vals, 98))
else:
    raw_min, raw_max = -1.0, 1.0
    clip_min, clip_max = -1.0, 1.0

if clip_min >= 0:
    clip_min = -1e-6
if clip_max <= 0:
    clip_max = 1e-6

norm_map = mpl.colors.TwoSlopeNorm(vmin=clip_min, vcenter=0.0, vmax=clip_max)

HEX_LINEWIDTH = 0.35
HEX_EDGECOLOR = "0.78"

grid_plot.plot(
    ax=ax_map,
    column="delta_density",
    cmap=MAP_CMAP,
    norm=norm_map,
    linewidth=HEX_LINEWIDTH,
    edgecolor=HEX_EDGECOLOR,
    transform=ccrs.PlateCarree(),
    zorder=3
)

OUTLINE_LW = 0.9
OUTLINE_COLOR = "0.35"

ax_map.add_geometries(
    study_outline_geoms,
    crs=ccrs.PlateCarree(),
    facecolor="none",
    edgecolor=OUTLINE_COLOR,
    linewidth=OUTLINE_LW,
    zorder=4
)

SIG_DOT_SIZE = 8
SIG_DOT_COLOR = "black"
SIG_DOT_ZORDER = 6

if sig_count > 0:
    ax_map.scatter(
        sig_hex["marker_geom"].x,
        sig_hex["marker_geom"].y,
        s=SIG_DOT_SIZE,
        c=SIG_DOT_COLOR,
        marker="o",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=SIG_DOT_ZORDER
    )

xmin, ymin, xmax, ymax = grid_gdf.total_bounds
# add_locator_map(
#     ax_map,
#     extent_main=(xmin, xmax, ymin, ymax),
#     inset_loc=(0.69, 0.68, 0.28, 0.26),
#     region_extent=(68, 105, 5, 40)
# )

GRIDLINE_LW = 0.35
GRIDLINE_COLOR = "0.82"
GRIDLINE_ALPHA = 0.6
GRIDLINE_STYLE = "--"

GRID_LABEL_SIZE = 13

gl = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=GRIDLINE_LW,
    color=GRIDLINE_COLOR,
    alpha=GRIDLINE_ALPHA,
    linestyle=GRIDLINE_STYLE
)
gl.top_labels = True
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = True
gl.xlabel_style = {"size": GRID_LABEL_SIZE}
gl.ylabel_style = {"size": GRID_LABEL_SIZE}
gl.xlocator = mticker.FixedLocator(np.arange(75, 100, 5))
gl.ylocator = mticker.FixedLocator(np.arange(25, 40, 5))
gl.xformatter = LongitudeFormatter(number_format=".0f", degree_symbol="°")
gl.yformatter = LatitudeFormatter(number_format=".0f", degree_symbol="°")

TEXT_X = 0.06
TEXT_Y = 0.25
TEXT_FONTSIZE = 13
TEXT_COLOR = "0.15"

ax_map.text(
    TEXT_X, TEXT_Y,
    "· Mann-Kendall test (p < 0.05)",
    transform=ax_map.transAxes,
    ha="left", va="center",
    fontsize=TEXT_FONTSIZE, color=TEXT_COLOR,
    zorder=10
)

# =========================================================
# 6. COLORBAR
# =========================================================

sm = mpl.cm.ScalarMappable(cmap=MAP_CMAP, norm=norm_map)
sm.set_array([])

CBAR_WIDTH = "28%"    
CBAR_HEIGHT = "4%"   
CBAR_LOC = "lower left"

CBAR_BBOX_X = 0.08      
CBAR_BBOX_Y = 0.15     
CBAR_BBOX_W = 1.0
CBAR_BBOX_H = 1.0

cax = inset_axes(
    ax_map,
    width=CBAR_WIDTH,
    height=CBAR_HEIGHT,
    loc=CBAR_LOC,
    bbox_to_anchor=(CBAR_BBOX_X, CBAR_BBOX_Y, CBAR_BBOX_W, CBAR_BBOX_H),
    bbox_transform=ax_map.transAxes,
    borderpad=0
)

cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")

CBAR_OUTLINE_LW = 0.6
CBAR_TICK_SIZE = 11
CBAR_TICK_LEN = 2
CBAR_LABEL_SIZE = 12
CBAR_LABEL_PAD = 2

cbar.outline.set_linewidth(CBAR_OUTLINE_LW)
cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, length=CBAR_TICK_LEN)

tick_positions = [clip_min, 0.0, clip_max]
tick_labels = [f"{raw_min:.1f}", "0.0", f"{raw_max:.1f}"]

cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)

cbar.set_label(
    "Δ Density ( Events / 1000 km$^2$. yr)",
    fontsize=CBAR_LABEL_SIZE,
    labelpad=CBAR_LABEL_PAD
)

# =========================================================
# 7. SAVE
# =========================================================

plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
plt.show()