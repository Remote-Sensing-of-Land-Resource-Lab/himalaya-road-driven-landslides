import os
import random
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from rasterio.transform import xy

def random_points_from_binary_raster_with_csv(
    raster_path,
    output_point_path,
    output_csv_path,
    sample_size=1000,
    target_value=1,
    random_seed=42
):
    print("--- Starting processing ---")
    print(f"Reading raster file: {os.path.basename(raster_path)}")

    random.seed(random_seed)
    np.random.seed(random_seed)

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        print(f"Raster size: {band.shape}")
        print(f"Coordinate system: {crs}")
        print(f"NoData value: {nodata}")

        # 1. Find all pixels with the value target_value
        print(f"Extracting pixels with value {target_value}...")
        valid_mask = (band == target_value)

        if nodata is not None:
            valid_mask = valid_mask & (band != nodata)

        rows, cols = np.where(valid_mask)
        total_candidates = len(rows)

        print(f"Total candidate pixels with value {target_value}: {total_candidates}")

        if total_candidates == 0:
            raise ValueError(f"Raster does not contain any pixels with value {target_value}, sampling not possible.")

        # 2. Check if there's enough sampling
        if total_candidates >= sample_size:
            sampled_idx = np.random.choice(total_candidates, size=sample_size, replace=False)
            print(f"Candidate pixels are sufficient, randomly sampling {sample_size} points.")
        else:
            sampled_idx = np.random.choice(total_candidates, size=sample_size, replace=True)
            print(f"⚠️ Warning: Insufficient candidate pixels ({total_candidates} < {sample_size}), performing sampling with replacement.")

        sampled_rows = rows[sampled_idx]
        sampled_cols = cols[sampled_idx]

        # 3. Convert pixel row and column numbers to coordinates
        print("Generating point coordinates...")
        points = []
        records = []

        for i, (r, c) in enumerate(zip(sampled_rows, sampled_cols), start=1):
            x, y = xy(transform, r, c, offset='center')  
            pt = Point(x, y)
            points.append(pt)

            records.append({
                "point_id": i,
                "row": int(r),
                "col": int(c),
                "raster_value": int(target_value)
            })

    # 4. Generate GeoDataFrame
    gdf = gpd.GeoDataFrame(records, geometry=points, crs=crs)

    # 5. 保存 Shapefile
    print("Generating Shapefile...")
    os.makedirs(os.path.dirname(output_point_path), exist_ok=True)
    gdf.to_file(output_point_path, encoding='utf-8')

    # 6. Export CSV
    print("Generating CSV output...")
    csv_gdf = gdf.copy()

    if csv_gdf.crs is not None and csv_gdf.crs.to_string() != "EPSG:4326":
        csv_gdf = csv_gdf.to_crs("EPSG:4326")

    csv_gdf["lon"] = csv_gdf.geometry.x
    csv_gdf["lat"] = csv_gdf.geometry.y
    csv_gdf["lat_lon"] = csv_gdf.apply(lambda row: f"{row['lat']}, {row['lon']}", axis=1)

    csv_df = pd.DataFrame(csv_gdf.drop(columns="geometry"))
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    csv_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print("Processing completed! Results saved:")
    print(f"  Point SHP: {output_point_path}")
    print(f"  Table CSV: {output_csv_path}")


# --- Parameter configuration ---
config = {
    "raster_path": r"H:\Himalaya\RF_two_model\areas_may_slide\may_slide_mask.tif",
    "output_point": r"H:\Himalaya\1000_non_ls_samples\Accuracy\non_ls_1000_Point.shp",
    "output_csv": r"H:\Himalaya\1000_non_ls_samples\Accuracy\non_ls_1000_Point.csv"
}

# --- Run ---
random_points_from_binary_raster_with_csv(
    raster_path=config["raster_path"],
    output_point_path=config["output_point"],
    output_csv_path=config["output_csv"],
    sample_size=1000,
    target_value=1,
    random_seed=42
)