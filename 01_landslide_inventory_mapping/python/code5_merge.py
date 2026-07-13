import os
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
import fiona
import numpy as np
from rasterio.merge import merge

def merge_year(year, scene_dir, output_dir, clip_shp):
    try:
        # Collect all TIFF files for the given year across scenes
        tiff_files = []
        for scene_id in os.listdir(scene_dir):
            tif_path = os.path.join(scene_dir, scene_id, f"filter2_{year}.tif")
            if os.path.isfile(tif_path):
                tiff_files.append(tif_path)

        if not tiff_files:
            print(f"{year} did not find any TIFF files")
            return

        print(f"Starting processing for year: {year} ({len(tiff_files)} TIFF files)")

        # Open all TIFF files
        src_files_to_mosaic = [rasterio.open(tif) for tif in tiff_files]

        # Merge all rasters
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Get metadata from the first raster
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 1,
            "compress": "lzw"
        })

        # Save the merged raster temporarily
        merged_path = os.path.join(output_dir, f"{year}_merged.tif")
        os.makedirs(output_dir, exist_ok=True)
        with rasterio.open(merged_path, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)

        # Read the clipping extent
        with fiona.open(clip_shp, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        # Clip the merged raster
        with rasterio.open(merged_path) as src:
            clipped_data, clipped_transform = mask(src, shapes, crop=True, nodata=src.nodata)

            out_meta.update({
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform
            })

            out_path = os.path.join(output_dir, f"{year}.tif")
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(clipped_data[0], 1)

        # Delete the temporary merged file
        os.remove(merged_path)

        print(f"{year} merge and clip completed: {out_path}")

    except Exception as e:
        print(f"Error processing year {year}: {e}")



if __name__ == "__main__":
    scene_dir = r"H:\Himalaya\Landsat_density/output/select_2"
    output_dir = r"H:\Himalaya\Landsat_density/output/result"
    clip_shp = r"H:\Himalaya\grid\Him_1\grid_1x1.shp"  # Clipping extent

    years = list(range(2000, 2025))

    for year in tqdm(years, desc="Processing years", unit="year"):
        merge_year(year, scene_dir, output_dir, clip_shp)


