import os
import numpy as np
import rasterio
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

# ==============================
# Input paths
# ==============================
tif_path = r'H:\Himalaya\RF_two_model\areas_may_slide\may_slide_mask.tif'
shp_path = r'H:\Himalaya\13w_landslides_list_final.shp'

# WGS84 ellipsoid
geod = Geod(ellps="WGS84")


# ==============================
# Calculate the geodesic area of a longitude/latitude polygon (m²)
# ==============================
def geodesic_area_geom(geom):
    """
    Calculate the true area of a shapely geometry on the WGS84 ellipsoid (m²).
    Supports Polygon / MultiPolygon / GeometryCollection.
    """
    if geom is None or geom.is_empty:
        return 0.0

    geom_type = geom.geom_type

    if geom_type in ["Polygon", "MultiPolygon"]:
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(area)

    elif geom_type == "GeometryCollection":
        total = 0.0
        for g in geom.geoms:
            total += geodesic_area_geom(g)
        return total

    else:
        return 0.0


# ==============================
# 1. Calculate the total TIFF area and the area where value = 1
# ==============================
with rasterio.open(tif_path) as src:
    if src.crs is None:
        raise ValueError("The TIFF has no coordinate reference system information.")
    if str(src.crs).upper() not in ["EPSG:4326", "OGC:CRS84"]:
        print(f"Warning: the current TIFF CRS is {src.crs}; the code treats coordinates as longitude/latitude, so please confirm.")

    transform = src.transform
    width = src.width
    height = src.height
    nodata = src.nodata

    # Check whether there is a rotation term
    if not np.isclose(transform.b, 0) or not np.isclose(transform.d, 0):
        raise ValueError("The raster has a rotation term; the current code does not support rotated rasters.")

    # Read the first band
    band1 = src.read(1)

    # ------------------------------
    # 1.1 Total TIFF coverage area (based on the bounding rectangle)
    # ------------------------------
    bounds = src.bounds
    bbox_poly = Polygon([
        (bounds.left,  bounds.top),
        (bounds.right, bounds.top),
        (bounds.right, bounds.bottom),
        (bounds.left,  bounds.bottom)
    ])
    tif_total_area_m2 = geodesic_area_geom(bbox_poly)

    # ------------------------------
    # 1.2 Calculate the area where value = 1
    # Pixel area varies with latitude in EPSG:4326, so calculate the true area of each pixel row by row.
    # ------------------------------
    pixel_width_deg = transform.a
    pixel_height_deg = abs(transform.e)
    x_left = transform.c

    value1_area_m2 = 0.0
    valid_area_m2 = 0.0  # Optional: valid pixel area (excluding nodata)

    for row in range(height):
        # Latitude boundaries of the current row of pixels
        y_top = transform.f + row * transform.e
        y_bottom = y_top + transform.e  # transform.e is usually negative

        north = max(y_top, y_bottom)
        south = min(y_top, y_bottom)

        # Construct a pixel polygon from the first column of the row to estimate the single-pixel area
        west = x_left
        east = x_left + pixel_width_deg

        pixel_poly = Polygon([
            (west, north),
            (east, north),
            (east, south),
            (west, south)
        ])
        pixel_area_m2 = geodesic_area_geom(pixel_poly)

        row_data = band1[row, :]

        # Valid pixel statistics
        if nodata is not None:
            valid_mask = row_data != nodata
        else:
            valid_mask = np.ones_like(row_data, dtype=bool)

        valid_count = np.count_nonzero(valid_mask)
        valid_area_m2 += valid_count * pixel_area_m2

        # Statistics for value = 1 (only within valid pixels)
        value1_count = np.count_nonzero((row_data == 1) & valid_mask)
        value1_area_m2 += value1_count * pixel_area_m2


# ==============================
# 2. Calculate the total area of the landslide shapefile
# ==============================
gdf = gpd.read_file(shp_path)

if gdf.empty:
    raise ValueError("The shapefile is empty.")

if gdf.crs is None:
    raise ValueError("The shapefile has no coordinate reference system information.")

# Reproject to EPSG:4326 before calculating geodesic area
if gdf.crs.to_string().upper() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# Fix any invalid geometries that may exist
gdf["geometry"] = gdf["geometry"].buffer(0)

landslide_area_m2 = gdf["geometry"].apply(geodesic_area_geom).sum()


# ==============================
# 3. Output results
# ==============================
print("=" * 60)
print("Area statistics results (unit: km²)")
print("=" * 60)
print(f"TIFF total coverage area                : {tif_total_area_m2 / 1e6:,.3f} km²")
print(f"TIFF valid pixel area (excluding NoData) : {valid_area_m2 / 1e6:,.3f} km²")
print(f"Area with value = 1 in TIFF             : {value1_area_m2 / 1e6:,.3f} km²")
print(f"Total area of landslide shapefile      : {landslide_area_m2 / 1e6:,.3f} km²")
print("=" * 60)