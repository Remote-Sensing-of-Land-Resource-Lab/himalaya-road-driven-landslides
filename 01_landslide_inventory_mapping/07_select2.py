# SWIR1 reflectance > 200
# Brightness range: 200 ~ 2000
# When shape index (SI) > 3, deviation angle < 60°
# In each patch, 80% of pixels have numObs > 100


import numpy as np
import rasterio
from tqdm import tqdm
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil

base_path = r"H:\Himalaya\Landsat_density"
base_path_2 = r"H:\Himalaya"
target_bands = ['SWIR1']   
n_segments = 6  

dright_thre = 200   
ref_thre = 200
deviation_thre = 60
SI_thre = 3.0
obs_thre = 100
obs_ratio_thre = 0.5


def is_valid_tif(path):
    """Check whether a TIFF file is complete and valid"""
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)  # Try reading the first band
            if arr is None or arr.size == 0:
                return False
        return True
    except Exception:
        return False
    

def process_single_year(region, year):
    paths = {
        "breaks": rf"{base_path}/input/break/{region}_break_monthly.tif",
        "brightness": rf"{base_path}/output/del_brightness/del_bri_{region}.tif",
        "delta_ref": rf"{base_path}/output/del_reflectance/del_ref_{region}.tif", 
        "select1": rf"{base_path}/output/select_1/{region}/filter1_{year}.tif",
        "deviation_angle": rf"{base_path}/output/deviation_angle/{region}/deviation_{year}.tif",  
        "output": rf"{base_path}/output/select_2/{region}/filter2_{year}.tif"
    }
    os.makedirs(os.path.dirname(paths["output"]), exist_ok=True)
    
    final_out = paths["output"]
    tmp_out = final_out + ".tmp"

    # ===== Skip logic =====
    if is_valid_tif(final_out):
        print(f"Already exists and is valid: {region}-{year}; skipping")
        return True
    
    if os.path.exists(tmp_out):
        print(f"Residual temporary file detected; deleting and reprocessing: {tmp_out}")
        os.remove(tmp_out)

    if is_valid_tif(final_out):
        print(f"Already completed: {region}-{year}; skipping")
        return True
    elif os.path.exists(final_out):  # File exists but is corrupted
        print(f"Corrupt file detected; deleting and reprocessing: {final_out}")
        os.remove(final_out)


    try:
        with rasterio.open(paths["select1"]) as src:
            patch_ids = src.read(1)
            profile = src.profile.copy()
            height, width = patch_ids.shape

        with rasterio.open(paths["breaks"]) as src:
            tbreak_bands = src.read(list(range(13, 19)))
            obs_bands = src.read(list(range(25, 31))) 

        with rasterio.open(paths["brightness"]) as src:
            bright_data = src.read()  
            
        with rasterio.open(paths["delta_ref"]) as src:
            delta_ref_data = src.read()

        with rasterio.open(paths["deviation_angle"]) as src:
            deviation_angle = src.read(3)  
            shape_index = src.read(4)   

        # ========== Patch processing ==========
        valid_labels = []
        unique_labels = np.unique(patch_ids)
        unique_labels = unique_labels[unique_labels != 0]

        for label_id in tqdm(unique_labels, desc=f"Region {region} - processing patches for {year}"):
            mask = (patch_ids == label_id)
            yy, xx = np.where(mask)
            total_pixels = len(yy)
            
            if total_pixels == 0:
                continue
         
            bright_values = []
            del_ref_values = {b: [] for b in target_bands}
            
            y0, x0 = yy[0], xx[0]
            avg_deviation = deviation_angle[y0, x0]
            avg_si = shape_index[y0, x0]
            
            patch_obs = obs_bands[:, yy, xx]  
            add_numObs = np.sum(patch_obs, axis=0)  
            
            obs_over_thre = np.sum(add_numObs > obs_thre)
            obs_ratio = obs_over_thre / total_pixels
            condition_obs = obs_ratio >= obs_ratio_thre

            for y, x in zip(yy, xx):
                seg_index = -1
                for seg in range(n_segments):
                    tbreak = tbreak_bands[seg, y, x]
                    if (tbreak >= year) and (tbreak < year + 1):
                        seg_index = seg
                        break
                
                if seg_index == -1:
                    continue  

                try:
                    swir1_band = seg_index * 6 + 4
                    bright_val = bright_data[seg_index, y, x]
                    swir1_del = delta_ref_data[swir1_band, y, x]

                    bright_values.append(bright_val)
                    del_ref_values['SWIR1'].append(swir1_del)
                    
                except IndexError:
                    continue

            avg_bright = np.nanmean(bright_values) if bright_values else 0
            avg_delref = {
                b: np.nanmean(del_ref_values[b]) if del_ref_values[b] else 0 
                for b in target_bands
            }
            
            
            # ========== Selection criteria ==========
            # Criterion 1: SWIR1 reflectance change and brightness change
            condition_ref_bri = (
                all(v > ref_thre for v in avg_delref.values()) and (dright_thre < avg_bright < 2000))
            
            # condition_ref_bri = (dright_thre < avg_bright < 2000)
            
            # Criterion 2: If shape index SI > 3, deviation angle < 60°
            condition_SI = (avg_si <= SI_thre) or (avg_si > SI_thre and avg_deviation < deviation_thre)
            
            # Combined selection criteria
            if condition_ref_bri and condition_SI and condition_obs:
                valid_labels.append(label_id)

        # ========== Output results ==========
    #     result = np.where(np.isin(patch_ids, valid_labels), patch_ids, 0)
    #     with rasterio.open(paths["output"], 'w',
    #                       driver='GTiff',
    #                       height=height,
    #                       width=width,
    #                       count=1,
    #                       dtype=np.int32,
    #                       crs=profile['crs'],
    #                       transform=profile['transform'],
    #                       compress='lzw',
    #                       nodata=0) as dst:
    #         dst.write(result.astype(np.int32), 1)
            
    #     return True

    # except Exception as e:
    #     print(f"Error while processing year {year}: {str(e)}")
    #     return False
    
        result = np.where(np.isin(patch_ids, valid_labels), patch_ids, 0)
        with rasterio.open(tmp_out, 'w',
                          driver='GTiff',
                          height=height,
                          width=width,
                          count=1,
                          dtype=np.int32,
                          crs=profile['crs'],
                          transform=profile['transform'],
                          compress='lzw',
                          nodata=0) as dst:
            dst.write(result.astype(np.int32), 1)

        # Atomic replacement: ensure only a complete file becomes final_out
        os.replace(tmp_out, final_out)
        print(f"Completed {region}-{year} → {final_out}")

        return True

    except Exception as e:
        print(f"Error while processing {region}-{year}: {str(e)}")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)  # Clean up the bad temporary file
        return False


# ====================== Batch processing functions ======================
def process_all_regions(max_limit=12):
    """Iterate over all regions and process available years"""
    select1_base = rf"{base_path}/output/select_1"
    output_base = rf"{base_path}/output/select_2"
    os.makedirs(output_base, exist_ok=True)
    region_ids = [f for f in os.listdir(select1_base) if os.path.isdir(os.path.join(select1_base, f))]
    
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory().available / (1024 ** 3)
    dynamic_workers = max(1, int(mem // 2))
    max_workers = min(max_limit, cpu_count, dynamic_workers)   
    print(f"Parallel worker limit: {max_workers} (CPU={cpu_count}, available memory≈{mem:.1f}GB)")

    futures_list = []
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for region in region_ids:
            input_dir = os.path.join(select1_base, region)
            year_pattern = re.compile(r"filter1_(\d{4})\.tif$")
            years = []
            for fname in os.listdir(input_dir):
                match = year_pattern.match(fname)
                if match:
                    years.append(int(match.group(1)))
            for year in years:
                futures_list.append(executor.submit(process_single_year, region, year))

        for f in tqdm(as_completed(futures_list), total=len(futures_list), desc="Step2 processing regions"):
            results.append(f.result())

    print("select2 processing completed")
    return results


if __name__ == "__main__":
    process_all_regions(max_limit=3)
