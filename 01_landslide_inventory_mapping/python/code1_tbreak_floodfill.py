# tbreak yearly 
# Yes


import os
import numpy as np
import rasterio
import cv2
from skimage.measure import regionprops
from typing import Tuple
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

FLOODFILL_DATE_INTERVAL = 64
NAN_VAL = -9999

def segmentation_floodfill_dateonly2(
    break_date_array: np.ndarray,
    date_interval: int = FLOODFILL_DATE_INTERVAL  
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    date_array_processed = np.where(np.isnan(break_date_array), NAN_VAL, break_date_array)
    
    valid_points = (break_date_array > 0) & (break_date_array != NAN_VAL)
    seed_index = np.where(valid_points)
    
    # Ensure seed_index is a tuple of (row_indices, col_indices)
    if len(seed_index) != 2:
        seed_index = np.where(break_date_array > 0)
        if len(seed_index) != 2:
            raise ValueError("seed_index should contain exactly two arrays (rows and cols)")
    
    # Get the seed point coordinate pairs
    seed_points = list(zip(seed_index[0], seed_index[1]))
    if not seed_points:
        # If there are no valid seed points, return an empty result
        empty_label = np.zeros_like(break_date_array, dtype=np.int32)
        empty_df = pd.DataFrame(columns=['label', 'area', 'mean_date', 'std_date'])
        return empty_label, date_array_processed, empty_label, empty_df
    
    # Sort seed points in descending order of date
    seed_points.sort(key=lambda p: break_date_array[p[0], p[1]], reverse=True)
    
    n_rows, n_cols = break_date_array.shape
    mask_s1 = np.zeros((n_rows + 2, n_cols + 2), dtype=np.uint8)
    mask_label_s1 = np.zeros((n_rows + 2, n_cols + 2))
    floodflags_base = 8 | cv2.FLOODFILL_MASK_ONLY
    
    cm_stack = np.dstack([break_date_array]*3).astype(np.float32)
    
    for i, (y, x) in enumerate(seed_points):
        remainder = i % 255
        # print(remainder)
        floodflags = floodflags_base | ((remainder + 1) << 8)
        
        # Run floodFill
        num, im, mask_s1, rect = cv2.floodFill(
            cm_stack,
            mask_s1,
            (x, y),  # OpenCV uses (x, y) coordinates
            0,
            loDiff=(date_interval, date_interval, date_interval),
            upDiff=(date_interval, date_interval, date_interval),
            flags=floodflags,
        )

        # Update labels once every 255 objects
        if remainder == 254:
            no = i // 255
            mask_label_s1[(mask_label_s1 == 0) & (mask_s1 > 0)] = (
                mask_s1[(mask_label_s1 == 0) & (mask_s1 > 0)].astype(np.float64) + no * 255
            )
            new_n = len(mask_label_s1[mask_label_s1==6])

    # Process the remaining unassigned labels
    if len(seed_points) > 0:
        no = len(seed_points) // 255
        mask_label_s1[(mask_label_s1 == 0) & (mask_s1 > 0)] = (
            mask_s1[(mask_label_s1 == 0) & (mask_s1 > 0)] + no * 255
        )

    # Crop off the extra boundary
    object_map_s1 = mask_label_s1[1:-1, 1:-1].astype(np.int32)
    
    # Compute region properties
    valid_dates = np.where(~np.isnan(break_date_array), break_date_array, 0)
    regions = regionprops(object_map_s1, intensity_image=valid_dates)
    
    region_info = pd.DataFrame([{
        'label': r.label,
        'area': r.area,
        'mean_date': r.mean_intensity,
        'std_date': np.std(valid_dates[r.coords[:, 0], r.coords[:, 1]])
    } for r in regions if r.area > 0])
    return mask_label_s1[1:-1, 1:-1], date_array_processed, object_map_s1, region_info



def process_tbreak_yearly_data(region: str) -> str:
    """Process breakpoint coefficients and generate yearly TIFF files"""
    nsegment = 6
    start_year = 2000        
    end_year = 2024
    
    coefs_dir = r'H:\Himalaya\Landsat_density\input\coef'
    breaks_dir = r'H:\Himalaya\Landsat_density\input\break'
    region_dir = os.path.join(coefs_dir, region)
    output_dir = os.path.join(r'H:\Himalaya\Landsat_density\output/tbreak_yearly', region)
    os.makedirs(output_dir, exist_ok=True)
    
    breaks_path = os.path.join(breaks_dir, f'{region}_break_monthly.tif')
    with rasterio.open(breaks_path) as src:
        bands = src.read()
        profile = src.profile

    height, width = bands.shape[1], bands.shape[2]
    output_layers = [np.full((height, width), np.nan, dtype=np.float32) 
                    for _ in range(end_year - start_year + 1)]
    
    # Process each segment
    for seg_idx in range(nsegment):
        time_band = nsegment * 2 + seg_idx     # Bands 12-17
        chgpro_band = nsegment * 3 + seg_idx   # Bands 18-23
        
        time_values = bands[time_band]
        chgpro_values = bands[chgpro_band]

        # Pixel-level processing
        for y in range(height):
            for x in range(width):
                time_value = time_values[y, x]
                if np.isnan(time_value) or chgpro_values[y, x] < 1:
                    continue
                
                year = int(np.floor(time_value))
                if start_year <= year <= end_year:
                    layer_index = year - start_year
                    output_layers[layer_index][y, x] = (time_value - year) * 365
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_dir, f'tbreak_{year}.tif')
        with rasterio.open(output_path, 'w', 
                         driver='GTiff',
                         height=height,
                         width=width,
                         count=1,
                         dtype='float32',
                         crs=profile['crs'],
                         transform=profile['transform']) as dst:
            dst.write(output_layers[year - start_year], 1)
    
    return output_dir


def process_labeled_images(input_folder: str, region: str, min_pixels: int):
    """Process generated TIFFs using flood fill segmentation with slope and time std filtering"""
    start_year = 2000         
    end_year = 2024

    labeled_folder = os.path.join(r'H:\Himalaya\Landsat_density\output/label_img', region)
    os.makedirs(labeled_folder, exist_ok=True)


    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue
            
        year = filename.split('_')[-1].split('.')[0]
        if not year.isdigit() or not (start_year <= int(year) <= end_year):
            continue

        input_path = os.path.join(input_folder, filename)
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            date_array = src.read(1)
            
            print(f"\nProcessing {region} year {year}")
            _, _, object_labels, region_info = segmentation_floodfill_dateonly2(
                date_array, date_interval=FLOODFILL_DATE_INTERVAL
            )
            
            filtered_labels = np.zeros_like(object_labels)
            for _, region_row in region_info.iterrows():
                if region_row['area'] >= min_pixels:
                    label = int(region_row['label'])
                    filtered_labels[object_labels == label] = label

            labeled_path = os.path.join(labeled_folder, f'labeled_{year}.tif')
            with rasterio.open(
                labeled_path,
                'w',
                **{**profile, 'dtype': 'uint32', 'count': 1}
            ) as dst:
                dst.write(filtered_labels, 1)

            print(f"{region} year {year} processed")

def process_region(region: str):
    print(f"\n{'='*40}\nStarting processing for Region {region}\n{'='*40}")
    yearly_dir = process_tbreak_yearly_data(region)
    process_labeled_images(yearly_dir, region, min_pixels=4)
    print(f"Region {region} completed")
    return region


def process_all_regions():
    """Automatically detect CPU/memory availability and process all regions in parallel"""
    coefs_dir = r"H:\Himalaya\Landsat_density\input\coef"
    regions = [d for d in os.listdir(coefs_dir) if os.path.isdir(os.path.join(coefs_dir, d))]
    regions.sort()
    print(f"Found {len(regions)} regions: {regions}")

    # CPU / memory detection
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory().available / (1024**3)  # GB
    max_workers = min(cpu_count, int(mem // 5))  # Assume each worker needs 2GB
    max_workers = max(1, max_workers)  # At least 1
    print(f"Dynamically allocated parallel workers: {max_workers} (CPU={cpu_count}, available memory≈{mem:.1f}GB)")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_region, r): r for r in regions}
        for future in as_completed(futures):
            region = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Region {region} processing failed: {e}")

    print("\nAll regions processing completed")
    return results

    # for region in regions:
    #     print(f"\n{'='*40}\nProcessing Region {region}\n{'='*40}")
    #     yearly_dir = process_tbreak_yearly_data(region)
    #     process_labeled_images(yearly_dir, region, min_pixels=4)
    #     print(f"Region {region} completed")

if __name__ == "__main__":
    process_all_regions()


