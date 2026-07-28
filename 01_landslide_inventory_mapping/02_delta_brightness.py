# Input a folder and compute brightness differences for all TIFF images within it
# Coefficients are stored by band
# Compute brightness differences at a fixed date (July 15 of the previous and following year)

import rasterio
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ====================== Input parameter configuration ======================
base_path = r"H:\Himalaya\Landsat_density" 
target_month = 7
target_day = 15
max_segments = 6
output_dir = rf"{base_path}/output/del_brightness"
os.makedirs(output_dir, exist_ok=True) 

target_bands = ['RED', 'GREEN', 'BLUE']
weights = {'RED':0.299, 'GREEN':0.587, 'BLUE':0.114}

WORKERS = 1  

# ====================== Utility functions ======================

def date_to_fraction(year, month, day):
    """Convert a date to a fractional year value (handling leap years correctly)."""
    date_obj = datetime(year, month, day)
    year_start = datetime(year, 1, 1)
    next_year_start = datetime(year + 1, 1, 1)
    return year + (date_obj - year_start).days / (next_year_start - year_start).days

def build_coef_indices(coefs_data, max_segments, target_bands):
    """Build a mapping from coefficient names to their storage positions."""
    index_map = {}
    for seg_id in [f"S{i}" for i in range(1, max_segments+1)]:
        for band in target_bands:
            band_descriptions = coefs_data[band]['descriptions']
            intp_name = f"{seg_id}_{band}_coef_INTP"
            slp_name = f"{seg_id}_{band}_coef_SLP"
            
            if intp_name in band_descriptions:
                index_map[intp_name] = (band, band_descriptions.index(intp_name))
            if slp_name in band_descriptions:
                index_map[slp_name] = (band, band_descriptions.index(slp_name))
    return index_map

# ====================== Main processing function ======================
def process_scene(scene_id):
    print(f"\n=== Processing Scene {scene_id} ===")

    # Input paths
    breaks_path = rf"{base_path}/input/break/{scene_id}_break_monthly.tif"
    coef_dir = rf"{base_path}/input/coef/{scene_id}"
    output_path = rf"{output_dir}/del_bri_{scene_id}.tif"

    # Check whether the inputs exist
    if not os.path.exists(breaks_path):
        print(f"⚠️ Skipping {scene_id}: break file does not exist")
        return False
    if not os.path.exists(coef_dir):
        print(f"⚠️ Skipping {scene_id}: coef directory does not exist")
        return False

    # Read breaks
    with rasterio.open(breaks_path) as src_breaks:
        breaks_data = src_breaks.read()
        breaks_meta = src_breaks.meta
        breaks_bands = src_breaks.descriptions

    # Read coefficient data
    coefs_data = {}
    for band in target_bands:
        coef_path = rf"{coef_dir}/{scene_id}_{band}_monthly.tif"
        if not os.path.exists(coef_path):
            print(f"⚠️ Skipping {scene_id}: missing {band} coefficient file")
            return False
        with rasterio.open(coef_path) as src:
            coefs_data[band] = {
                'data': src.read(),
                'descriptions': src.descriptions
            }
            
            
    coef_indices = build_coef_indices(coefs_data, max_segments, target_bands)
    height, width = breaks_meta['height'], breaks_meta['width']
    delta_bright = np.zeros((max_segments, height, width), dtype=np.float32)

    # Iterate over each segment
    for seg_num in range(1, max_segments + 1):
        seg_id = f"S{seg_num}"
        tbreak_band_name = f"{seg_id}_tBreak"
        
        if tbreak_band_name not in breaks_bands:
            continue
        tbreak_idx = breaks_bands.index(tbreak_band_name)
        tbreak_data = breaks_data[tbreak_idx]
        valid_mask = tbreak_data > 0

        if seg_num == max_segments or not np.any(valid_mask):
            continue

        t_pre = np.zeros_like(tbreak_data, dtype=np.float32)
        t_post = np.zeros_like(tbreak_data, dtype=np.float32)
        
        valid_t = tbreak_data[valid_mask]
        if valid_t.size > 0:
            t_pre_valid = np.vectorize(lambda t: date_to_fraction(int(t)-1, target_month, target_day))(valid_t)
            t_post_valid = np.vectorize(lambda t: date_to_fraction(int(t)+1, target_month, target_day))(valid_t)
            
            t_pre[valid_mask] = t_pre_valid
            t_post[valid_mask] = t_post_valid

        # Iterate over the three bands to compute the difference
        for band in target_bands:
            # Get the current segment coefficients
            current_intp = coefs_data[band]['data'][coef_indices[f"{seg_id}_{band}_coef_INTP"][1]]
            current_slp = coefs_data[band]['data'][coef_indices[f"{seg_id}_{band}_coef_SLP"][1]]
            
            # Get the next segment coefficients
            next_seg_id = f"S{seg_num+1}"
            next_intp = coefs_data[band]['data'][coef_indices[f"{next_seg_id}_{band}_coef_INTP"][1]]
            next_slp = coefs_data[band]['data'][coef_indices[f"{next_seg_id}_{band}_coef_SLP"][1]]

            # Compute the reflectance difference
            y_pre = current_slp * t_pre + current_intp
            y_post = next_slp * t_post + next_intp
            delta = np.where(valid_mask, y_post - y_pre, 0)

            delta_bright[seg_num-1] += weights[band] * delta

    # Write the output
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=max_segments,
        dtype=np.float32,
        crs=breaks_meta['crs'],
        transform=breaks_meta['transform']
    ) as dst:
        for i in range(max_segments):
            dst.write(delta_bright[i], i+1)
            dst.set_band_description(i+1, f"S{i+1}_delta_bright")
            
    print(f"✅ Scene {scene_id} processed successfully; result saved to {output_path}")
    return True

# ====================== Execute processing ======================
if __name__ == "__main__":
    break_files = [f for f in os.listdir(rf"{base_path}/input/break") if f.endswith("_break_monthly.tif")]
    scene_ids = [f.split('_')[0] for f in break_files]
    success_list = []
    fail_list = []
    
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(process_scene, sid): sid for sid in scene_ids}
    for future in as_completed(futures):
        sid = futures[future]
        try:
            ok = future.result()
            if ok:
                success_list.append(sid)
            else:
                fail_list.append(sid)
        except Exception as e:
            print(f"❌ Scene {sid} error: {e}")
            fail_list.append(sid)
            
    
    
    # ====================== Run report ======================
    print("\n===== Run report =====")
    print(f"✅ Success: {len(success_list)} -> {success_list}")
    print(f"❌ Failed: {len(fail_list)} -> {fail_list}")
    print("====================")
