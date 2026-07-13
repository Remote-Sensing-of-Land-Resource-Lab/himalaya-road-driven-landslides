# Batch-compute deviation angles for aspect direction
# Yes

import os
import glob
import time
import math
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# ------------------------
# Step 1: Reproject aspect data globally to a unified CRS
# ------------------------
def reproject_aspect_once(aspect_path, output_path, ref_path):
    """Reproject aspect data to the reference CRS (e.g., EPSG:4326) and do it only once."""
    if os.path.exists(output_path):
        print(f"[Skip] Unified reprojected aspect data already exists: {output_path}")
        return output_path

    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs

    with rasterio.open(aspect_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32"
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

    print(f"[Done] Aspect data reprojected to a unified CRS: {output_path}")
    return output_path


# ------------------------
# Step 2: Process a single landslide file
# ------------------------
def process_landslide_file(landslide_path, output_folder, aspect_wgs84_path, year):
    """Process a single year's landslide file."""
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"deviation_{year}.tif")
    
    if os.path.exists(output_path):
        return f"Skip {year}"

    with rasterio.open(landslide_path) as landslide_src, rasterio.open(aspect_wgs84_path) as aspect_src:
        landslide_array = landslide_src.read(1)
        landslide_profile = landslide_src.profile.copy()

        # Use a window to clip the aspect data
        slope_window = from_bounds(*landslide_src.bounds, transform=aspect_src.transform)
        aspect_array = aspect_src.read(1, window=slope_window)
        aspect_nodata = aspect_src.nodata if aspect_src.nodata is not None else -9999

    # Label landslide regions
    labeled_array = label(landslide_array > 0)
    regions = regionprops(labeled_array)

    aspect_avg_out = np.zeros_like(landslide_array, dtype=np.float32)
    axis_angle_out = np.zeros_like(landslide_array, dtype=np.float32)
    deviation_out = np.zeros_like(landslide_array, dtype=np.float32)
    shape_index_out = np.zeros_like(landslide_array, dtype=np.float32)

    for region in regions:
        coords = region.coords
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr

        # Back-scarp area
        if height > 0:
            back_scarp_coords = [coord for coord in coords if coord[0] <= minr + height * 0.33]
        else:
            back_scarp_coords = coords

        vals = []
        for r, c in back_scarp_coords:
            if 0 <= r < aspect_array.shape[0] and 0 <= c < aspect_array.shape[1]:
                v = aspect_array[r, c]
                if v != aspect_nodata and not np.isnan(v):
                    vals.append(v)

        if vals:
            rads = np.radians(vals)
            mean_x = np.mean(np.sin(rads))
            mean_y = np.mean(np.cos(rads))
            aspect_avg = np.degrees(np.arctan2(mean_x, mean_y)) % 360
        else:
            aspect_avg = 0.0

        # Long-axis orientation
        orientation = region.orientation
        dx, dy = math.cos(orientation), math.sin(orientation)
        axis_angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360

        # Deviation angle
        diff = abs(axis_angle - aspect_avg)
        deviation_angle = min(diff, 360 - diff)

        # Shape index
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        shape_index = major_axis / minor_axis if minor_axis > 0 else 0

        # Vectorized assignment
        rr, cc = coords[:, 0], coords[:, 1]
        aspect_avg_out[rr, cc] = aspect_avg
        axis_angle_out[rr, cc] = axis_angle
        deviation_out[rr, cc] = deviation_angle
        shape_index_out[rr, cc] = shape_index

    # Write the output file
    profile = landslide_profile.copy()
    profile.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": 0,
        "count": 4
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(aspect_avg_out, 1)
        dst.set_band_description(1, "Aspect_Avg")
        dst.write(axis_angle_out, 2)
        dst.set_band_description(2, "Axis_Angle")
        dst.write(deviation_out, 3)
        dst.set_band_description(3, "Deviation_Angle")
        dst.write(shape_index_out, 4)
        dst.set_band_description(4, "SI")

    elapsed = time.time() - start_time


# ------------------------
# Step 3: Batch process a single scene
# ------------------------
def process_scene(scene_id, aspect_wgs84_path):
    base_path = r"H:\Himalaya\Landsat_density"
    base_path_2 = r"G:/Him"
    input_folder = os.path.join(base_path, "output", "select_1", scene_id)
    output_folder = os.path.join(base_path, "output", "deviation_angle", scene_id)
    os.makedirs(output_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(input_folder, "filter1_*.tif"))
    print(f"Scene {scene_id} contains {len(input_files)} input files")
    if not input_files:
        print(f"⚠ Scene {scene_id} does not contain any filter1_*.tif files; skipping")
        return scene_id

    for input_file in input_files:
        filename = os.path.basename(input_file)
        try:
            year = int(filename.split("_")[-1].split(".")[0])
        except:
            print(f"⚠ Skipping {filename}: unable to parse the year")
            continue

        process_landslide_file(input_file, output_folder, aspect_wgs84_path, year)

    return scene_id


# ------------------------
# Step 4: Main workflow
# ------------------------
def main():
    base_path = r"H:\Himalaya\Landsat_density"

    dataset_path = r"D:/喜马拉雅/dataset/喜马拉雅山区30m坡向分布数据-数据实体"
    aspect_path = os.path.join(dataset_path, "喜马拉雅山区30m坡向分布数据.tif")
    aspect_wgs84_path = os.path.join(dataset_path, "aspect_wgs84.tif")

    ref_candidates = glob.glob(os.path.join(base_path, "output", "select_1", "*", "filter1_*.tif"))
    if not ref_candidates:
        raise FileNotFoundError("❌ No filter1_*.tif files were found; please check the path and file naming")
    ref_path = ref_candidates[0]

    reproject_aspect_once(aspect_path, aspect_wgs84_path, ref_path)

    scene_ids = [os.path.basename(p) for p in glob.glob(os.path.join(base_path, "output", "select_1", "*"))]
    print(f"✅ A total of {len(scene_ids)} scenes will be processed: {scene_ids[:10]}{'...' if len(scene_ids) > 10 else ''}")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(partial(process_scene, aspect_wgs84_path=aspect_wgs84_path), scene_ids)

    elapsed_time = time.time() - start_time
    print(f"🎯 All scenes processed; total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()


