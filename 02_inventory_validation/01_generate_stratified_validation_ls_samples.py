# Generate 1000 random points from landslide samples
import os
import geopandas as gpd
import pandas as pd


def sample_landslide_features(master_path, output_poly_path, output_point_path, output_csv_path, sample_size=1000, random_seed=42):
    print("--- Starting processing ---")

    # 1. Read the input landslide dataset
    print(f"Reading input file: {os.path.basename(master_path)}")
    gdf = gpd.read_file(master_path)

    if gdf.empty:
        raise ValueError("The input file contains no features.")

    if len(gdf) < sample_size:
        raise ValueError(f"The input file contains only {len(gdf)} features, which is fewer than the requested sample size of {sample_size}.")

    # 2. Randomly sample 1000 landslide features
    sampled_gdf = gdf.sample(n=sample_size, random_state=random_seed).copy()
    sampled_gdf["src_type"] = "Landslide"
    print(f"Randomly selected {len(sampled_gdf)} landslide samples from {len(gdf)} records.")

    # 3. Create representative points for the sampled polygons
    sampled_points = sampled_gdf.copy()
    sampled_points["geometry"] = sampled_points.geometry.representative_point()

    # 4. Save shapefiles and CSV
    os.makedirs(os.path.dirname(output_poly_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_point_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    sampled_gdf.to_file(output_poly_path, encoding="utf-8")
    sampled_points.to_file(output_point_path, encoding="utf-8")

    print("Preparing CSV output...")
    csv_gdf = sampled_points.copy()

    if csv_gdf.crs is not None and csv_gdf.crs.to_string() != "EPSG:4326":
        csv_gdf = csv_gdf.to_crs("EPSG:4326")

    csv_gdf["lon"] = csv_gdf.geometry.x
    csv_gdf["lat"] = csv_gdf.geometry.y
    csv_gdf["lat_lon"] = csv_gdf.apply(lambda row: f"{row['lat']}, {row['lon']}", axis=1)

    csv_df = pd.DataFrame(csv_gdf.drop(columns="geometry"))
    csv_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("Processing completed successfully.")
    print(f"  Polygon SHP: {output_poly_path}")
    print(f"  Point SHP: {output_point_path}")
    print(f"  CSV: {output_csv_path}")


# --- Parameter configuration ---
config = {
    "master_path": r"H:\Himalaya\13w_landslides_list_final.shp",
    "output_poly": r"H:\Himalaya\Accuracy\1000samples\Sampled_1000_Poly.shp",
    "output_point": r"H:\Himalaya\Accuracy\1000samples\Sampled_1000_Point.shp",
    "output_csv": r"H:\Himalaya\Accuracy\1000samples\Sampled_1000_Point.csv"
}

# Run the sampling workflow
sample_landslide_features(
    master_path=config["master_path"],
    output_poly_path=config["output_poly"],
    output_point_path=config["output_point"],
    output_csv_path=config["output_csv"],
    sample_size=1000,
    random_seed=42
)





