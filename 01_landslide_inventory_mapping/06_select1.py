# Slope

import os
import numpy as np
import rasterio
from skimage.measure import regionprops
from rasterio.warp import reproject, Resampling, calculate_default_transform  # === MODIFIED: Added calculate_default_transform
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def reproject_global_slope_to_wgs84(slope_path, slope_wgs84_path):
    if os.path.exists(slope_wgs84_path):
        print(f"Reprojected slope file already exists: {slope_wgs84_path}")
        return slope_wgs84_path

    print("Reprojecting slope data to EPSG:4326...")
    with rasterio.open(slope_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.meta.copy()
        profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": rasterio.float32,
            "count": 1,
            "nodata": np.nan
        })

        os.makedirs(os.path.dirname(slope_wgs84_path), exist_ok=True)
        with rasterio.open(slope_wgs84_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    print(f"Reprojection completed: {slope_wgs84_path}")
    return slope_wgs84_path


def reproject_and_crop_slope(slope_wgs84_path, ref_meta, ref_bounds, out_path):
    """
    Resample slope data to the reference raster and crop it to the reference extent.
    """
    if os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            return src.read(1)

    with rasterio.open(slope_wgs84_path) as src:
        window = rasterio.windows.from_bounds(
            ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top,
            transform=src.transform
        )
        slope_data = src.read(1, window=window)
        slope_transform = src.window_transform(window)

        aligned_data = np.zeros((ref_meta['height'], ref_meta['width']), dtype=np.float32)

        reproject(
            source=slope_data,
            destination=aligned_data,
            src_transform=slope_transform,
            src_crs=src.crs,
            dst_transform=ref_meta['transform'],
            dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear
        )

        meta = ref_meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(aligned_data.astype(np.float32), 1)

    return aligned_data




def filter_by_slope(label_array, tbreak_array, slope_array, slope_thre):
    """
    Filter patches by slope threshold:
    - Compute the mean slope for each patch
    - Keep the patch if mean slope > slope_thre
    - Otherwise remove it
    """
    regions = regionprops(label_array, intensity_image=tbreak_array)
    valid_labels = []

    for region in regions:
        mask = (label_array == region.label)
        region_slope = slope_array[mask]
        valid_slope = region_slope[~np.isnan(region_slope)]

        if len(valid_slope) == 0:
            continue

        slope_mean = np.mean(valid_slope)

        if slope_mean > slope_thre:
            valid_labels.append(region.label)

    filtered_labels = np.where(np.isin(label_array, valid_labels), label_array, 0)
    return filtered_labels


def process_single_year(label_path, tbreak_path, slope_array, slope_thre, output_path):
    """Process a single year"""
    with rasterio.open(label_path) as src:
        label_array = src.read(1)
        profile = src.meta.copy()

    with rasterio.open(tbreak_path) as src:
        tbreak_array = src.read(1)

    filtered_array = filter_by_slope(label_array, tbreak_array, slope_array, slope_thre=slope_thre)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile.update(dtype=rasterio.uint32, count=1, nodata=0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filtered_array.astype(rasterio.uint32), 1)
    return True


def preprocess_slope_for_regions(label_base, slope_wgs84_path, slope_base):
    """
    Preprocess: crop and save one slope raster per region as slope_region.tif.
    """
    region_ids = [f for f in os.listdir(label_base) if os.path.isdir(os.path.join(label_base, f))]
    region_ids.sort()

    for region in tqdm(region_ids, desc="Preprocessing slope data for all regions"):
        label_folder = os.path.join(label_base, region)
        slope_region_folder = os.path.join(slope_base, region)
        slope_region_path = os.path.join(slope_region_folder, f"slope_{region}.tif")

        if os.path.exists(slope_region_path):
            continue

        # Select one label raster as the reference for alignment
        sample_file = next((os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(".tif")), None)
        if sample_file is None:
            print(f"Region {region} has no valid label file; skipping")
            continue

        with rasterio.open(sample_file) as ref_img:
            ref_meta = ref_img.meta.copy()
            ref_bounds = ref_img.bounds

        reproject_and_crop_slope(slope_wgs84_path, ref_meta, ref_bounds, slope_region_path)


def process_region(region, label_base, tbreak_base, slope_base, output_dir,
                   slope_thre, max_workers):
    """Processing logic for a single region"""
    label_folder = os.path.join(label_base, region)
    tbreak_folder = os.path.join(tbreak_base, region)
    slope_region_path = os.path.join(slope_base, region, f"slope_{region}.tif")
    output_region_path = os.path.join(output_dir, region)

    os.makedirs(output_region_path, exist_ok=True)

    if not os.path.exists(slope_region_path):
        print(f"Region {region} slope data was not generated; skipping")
        return region

    with rasterio.open(slope_region_path) as src:
        slope_array = src.read(1)

    # Collect all available years
    years = []
    for fname in os.listdir(label_folder):
        if fname.startswith("labeled_") and fname.endswith(".tif"):
            year = int(fname.split("_")[1].split(".")[0])
            years.append(year)
    years.sort()

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for year in years:
            label_path = os.path.join(label_folder, f"labeled_{year}.tif")
            tbreak_path = os.path.join(tbreak_folder, f"tbreak_{year}.tif")
            output_path = os.path.join(output_region_path, f"filter1_{year}.tif")
            futures.append(executor.submit(
                process_single_year,
                label_path, tbreak_path, slope_array, slope_thre, output_path
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing years for Region {region}"):
            try:
                future.result()
            except Exception as e:
                print(f"Year processing failed for Region {region}: {e}")

    return region


def process_all_regions(base_output_dir, slope_path, output_dir, slope_thre=15):
    """
    Step 1: Filter patches for all regions by slope threshold.
    - base_output_dir: contains label_img / tbreak_yearly / slope_cut
    - slope_path: original slope data
    - output_dir: output directory (select_1)
    - slope_thre: slope threshold
    """
    label_base = os.path.join(base_output_dir, "label_img")
    tbreak_base = os.path.join(base_output_dir, "tbreak_yearly")
    slope_base = os.path.join(base_output_dir, "slope_cut")

    os.makedirs(output_dir, exist_ok=True)
    
    slope_dir = os.path.dirname(slope_path)
    slope_fname = os.path.splitext(os.path.basename(slope_path))[0]
    slope_wgs84_path = os.path.join(slope_dir, f"{slope_fname}_wgs84.tif")

    # === Reproject globally once first ===
    reproject_global_slope_to_wgs84(slope_path, slope_wgs84_path)

    # Preprocess slope data for all regions first
    preprocess_slope_for_regions(label_base, slope_wgs84_path, slope_base)

    region_ids = [f for f in os.listdir(label_base) if os.path.isdir(os.path.join(label_base, f))]
    region_ids.sort()
    print(f"Found {len(region_ids)} regions: {region_ids}")

    # Dynamically assign parallelism
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory().available / (1024 ** 3)
    # max_workers = min(cpu_count, max(1, int(mem // 2)))
    max_workers = 22
    print(f"Year-level parallelism per region: {max_workers} (CPU={cpu_count}, available memory≈{mem:.1f}GB)")

    # Process each region sequentially, with parallel year processing inside each region
    for region in region_ids:
        print(f"\n=== Processing Region {region} ===")
        process_region(region, label_base, tbreak_base, slope_base,
                       output_dir, slope_thre, max_workers)

    print(f"\nselect1 processing completed for {output_dir}")
    return True


if __name__ == "__main__":
    base_output_dir = r"H:\Himalaya\Landsat_density\output"
    slope_path = r"D:/dataset/aspect/30m_aspect.tif"
    output_dir = r"H:\Himalaya\Landsat_density/output/select_1"
    slope_thre = 15
    process_all_regions(base_output_dir, slope_path, output_dir, slope_thre)







