# water \ road \ building \ LULC

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil


def align_raster_to_reference(src_path, ref_meta):
    """Align the raster at src_path to the reference metadata in ref_meta"""
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        aligned_data = np.zeros((ref_meta['height'], ref_meta['width']), dtype=src_data.dtype)

        reproject(
            source=src_data,
            destination=aligned_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_meta['transform'],
            dst_crs=ref_meta['crs'],
            resampling=Resampling.nearest
        )

    return aligned_data


def filter_patches_by_overlap(patch_array, road_array, water_array, building_array, landuse_array, threshold_road, threshold_water):
    """
    Filter patches based on overlap ratios from multiple rasters.
    - patch_array: patch ID raster
    - road_array: road binary raster (road=1, otherwise=0)
    - water_array: water binary raster (water=1, otherwise=0)
    - building_array: building binary raster (building=1, otherwise=0)
    - threshold_road: remove patches with road overlap above this ratio
    - threshold_water: remove patches with water overlap above this ratio
    """
    patch_array = patch_array.astype(np.int32)
    max_id = patch_array.max()
    if max_id == 0:
        return patch_array

    # Create masks
    road_mask = (road_array == 1)
    water_mask = (water_array == 1)
    building_mask = (building_array == 1)
    landuse_invalid_mask = (landuse_array != 1)
    
    counts_total = np.bincount(patch_array.ravel(), minlength=max_id + 1)
    valid_ids_mask = (counts_total > 0)

    counts_road = np.bincount(patch_array[road_mask].ravel(), minlength=max_id + 1)
    counts_water = np.bincount(patch_array[water_mask].ravel(), minlength=max_id + 1)
    counts_building = np.bincount(patch_array[building_mask].ravel(), minlength=max_id + 1)
    counts_landuse_invalid = np.bincount(patch_array[landuse_invalid_mask].ravel(), minlength=max_id + 1)

    # Calculate overlap ratios
    overlap_ratio_road = np.zeros_like(counts_road, dtype=np.float32)
    overlap_ratio_water = np.zeros_like(counts_water, dtype=np.float32)
    
    overlap_ratio_road[valid_ids_mask] = counts_road[valid_ids_mask] / counts_total[valid_ids_mask]
    overlap_ratio_water[valid_ids_mask] = counts_water[valid_ids_mask] / counts_total[valid_ids_mask]
    
    remove_mask = (
        (counts_building > 0) |
        ((counts_road > 1) & (overlap_ratio_road > threshold_road)) |
        ((counts_water > 1) & (overlap_ratio_water > threshold_water)) |
        (counts_landuse_invalid > 0)
    )

    remove_ids = np.where(remove_mask)[0]

    if remove_ids.size > 0:
        patch_array[np.isin(patch_array, remove_ids)] = 0

    return patch_array

def prepare_region_cache(region, base_dir, road_raster, water_raster, building_raster, landuse_raster, cache_dir):
    """Align the base rasters once for each region and cache them"""
    os.makedirs(cache_dir, exist_ok=True)

    # Find the first filter2_xxxx.tif in this region as the reference
    ref_path = None
    for year in range(2000, 2025):
        candidate = os.path.join(base_dir, region, f"filter2_{year}.tif")
        if os.path.exists(candidate):
            ref_path = candidate
            break
    if ref_path is None:
        return False

    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()

    # Align the 4 base rasters
    road_aligned = align_raster_to_reference(road_raster, ref_meta)
    water_aligned = align_raster_to_reference(water_raster, ref_meta)
    building_aligned = align_raster_to_reference(building_raster, ref_meta)
    landuse_aligned = align_raster_to_reference(landuse_raster, ref_meta)

    # Save cache files (NPY format for speed)
    np.save(os.path.join(cache_dir, f"{region}_road.npy"), road_aligned)
    np.save(os.path.join(cache_dir, f"{region}_water.npy"), water_aligned)
    np.save(os.path.join(cache_dir, f"{region}_building.npy"), building_aligned)
    np.save(os.path.join(cache_dir, f"{region}_landuse.npy"), landuse_aligned)

    return True


def process_single_year(region, year, base_dir, cache_dir,
                        output_dir, threshold_road, threshold_water):
    
    region_path = os.path.join(base_dir, region)
    output_region_path = os.path.join(output_dir, region)
    os.makedirs(output_region_path, exist_ok=True)

    file_path = os.path.join(region_path, f"filter2_{year}.tif")
    if not os.path.exists(file_path):
        return False

    try:
        with rasterio.open(file_path) as src:
            patch_array = src.read(1)
            patch_meta = src.meta.copy()
            
        road_aligned = np.load(os.path.join(cache_dir, f"{region}_road.npy"))
        water_aligned = np.load(os.path.join(cache_dir, f"{region}_water.npy"))
        building_aligned = np.load(os.path.join(cache_dir, f"{region}_building.npy"))
        landuse_aligned = np.load(os.path.join(cache_dir, f"{region}_landuse.npy"))

        # Ensure dimensions match
        if patch_array.shape != road_aligned.shape:
            raise ValueError(f"{region} {year} dimension mismatch: patch={patch_array.shape}, road={road_aligned.shape}")

        filtered_array = filter_patches_by_overlap(
            patch_array, road_aligned, water_aligned,
            building_aligned, landuse_aligned,
            threshold_road, threshold_water
        )

        out_path = os.path.join(output_region_path, f"filter3_{year}.tif")
        patch_meta.update(dtype=rasterio.uint32, count=1, nodata=0)
        with rasterio.open(out_path, 'w', **patch_meta) as dst:
            dst.write(filtered_array.astype(rasterio.uint32), 1)

        return True

    except Exception as e:
        print(f"{region} {year} processing error: {e}")
        return False
    

def process_all_regions(base_dir, road_raster, water_raster, buildings_raster,landuse_raster, 
                        output_dir, threshold_road, threshold_water, max_limit=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    region_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Preparing alignment cache for each region...")
    for region in tqdm(region_folders):
        prepare_region_cache(region, base_dir, road_raster, water_raster, buildings_raster, landuse_raster, cache_dir)

    # Automatically detect CPU / memory
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory().available / (1024 ** 3)
    auto_workers = min(cpu_count, max(1, int(mem // 2)))
    max_workers = min(auto_workers, max_limit)
    print(f"Parallel worker limit: {max_workers} (CPU={cpu_count}, available memory≈{mem:.1f}GB)")


    futures_list = []
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for region in region_folders:
            for year in range(2000, 2025):
                futures_list.append(
                    executor.submit(
                        process_single_year, region, year, base_dir,
                        cache_dir, output_dir, threshold_road, threshold_water
                    )
                )
        for f in tqdm(as_completed(futures_list), total=len(futures_list), desc="Step3 Processing regions"):
            results.append(f.result())

    print(f"select3 processing completed for {output_dir}")
    return results


if __name__ == "__main__":
    base_dir = r"H:/Himalaya/output/select_2_S1out" 
    road_raster = r"D:/喜马拉雅/input/roads/him_roads_wgs84.tif"
    water_raster = r"D:/喜马拉雅/input/water/him_water_wgs84.tif"
    buildings_raster = r"D:/喜马拉雅/input/building/him_building_wgs84.tif"
    landuse_raster = r"D:/喜马拉雅/dataset/喜马拉雅山区30m土地覆盖(2020)/Reclass_Him2020.tif"
    output_dir = r"H:/Himalaya/output/select_3_S1out"
    threshold_road = 0.5  
    threshold_water = 0.8 

    process_all_regions(base_dir, road_raster, water_raster, buildings_raster,
                        landuse_raster, output_dir, threshold_road, threshold_water, max_limit=22)

