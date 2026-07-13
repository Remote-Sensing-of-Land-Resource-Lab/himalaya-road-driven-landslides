# Compute reflectance differences for target bands at a fixed date one year before and after a breakpoint (time represented as 2022.567)
# Process regions in parallel
# Yes

import rasterio
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed # Use a process pool to avoid memory contention
from rasterio.windows import Window
from contextlib import ExitStack # Used to manage multiple file handles

# ====================== Input parameter configuration ======================
base_path = r"H:\Himalaya\Landsat_density"
max_segments = 6
target_month = 7
target_day = 1

output_dir = rf"{base_path}/output/del_reflectance"
os.makedirs(output_dir, exist_ok=True)

# Restore the full band list
target_bands = ['RED', 'GREEN', 'BLUE', 'NIR', 'SWIR1', 'SWIR2']
# Define bands that do not need to be computed
skip_bands = ['NIR', 'SWIR2']

BLOCK_SIZE = 1024 
WORKERS = 4 

# ====================== Utility functions ======================
def date_to_year_fraction(year, month, day):
    try:
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        current = datetime(year, month, day)
        return year + (current - start).total_seconds() / (end - start).total_seconds()
    except:
        return np.nan

# Precompute a lookup table for time values to improve performance
LOOKUP_TABLE = {
    y: date_to_year_fraction(y, target_month, target_day) 
    for y in range(1985, 2030)
}

def get_coef_idx(descriptions, seg_id, band, type_suffix):
    """Helper function: safely find the coefficient index."""
    try:
        name = f"{seg_id}_{band}_coef_{type_suffix}"
        return descriptions.index(name)
    except ValueError:
        return None

# ====================== Core processing function (block-based mode) ======================
def process_region(region_id):
    print(f"🚀 Start processing Region {region_id} (block mode)...")
    
    break_path = rf"{base_path}/input/break/{region_id}_break_monthly.tif"
    output_path = rf"{output_dir}/del_ref_{region_id}.tif"

    if not os.path.exists(break_path):
        return f"❌ Region {region_id}: break file does not exist"

    try:
        with ExitStack() as stack:
            # 1. Open the break file
            src_brk = stack.enter_context(rasterio.open(break_path))
            profile = src_brk.profile.copy()
            height, width = profile['height'], profile['width']

            # 2. Prepare coefficient file handles (skip NIR/SWIR2)
            src_coefs = {}
            for band in target_bands:
                if band in skip_bands: 
                    continue
                
                coef_path = rf"{base_path}/input/coef/{region_id}/{region_id}_{band}_monthly.tif"
                if os.path.exists(coef_path):
                    src_coefs[band] = stack.enter_context(rasterio.open(coef_path))
                else:
                    print(f"⚠️ Region {region_id}: missing {band} coefficient file")

            # 3. Configure the output file
            num_output_bands = max_segments * len(target_bands)
            profile.update(count=num_output_bands, dtype='float32', nodata=np.nan, compress='lzw')
            
            dst = stack.enter_context(rasterio.open(output_path, 'w', **profile))

            # 4. Set band descriptions (metadata)
            band_cursor = 1
            desc_map = {} # Record (seg, band) -> output_band_index (0-based)
            for seg in range(1, max_segments + 1):
                for band in target_bands:
                    desc = f"Seg{seg}_{band}_del"
                    if band in skip_bands: desc += "_skipped"
                    dst.set_band_description(band_cursor, desc)
                    desc_map[(seg, band)] = band_cursor - 1
                    band_cursor += 1

            # 5. Generate block windows
            windows = [
                Window(col_off, row_off, min(BLOCK_SIZE, width - col_off), min(BLOCK_SIZE, height - row_off))
                for row_off in range(0, height, BLOCK_SIZE)
                for col_off in range(0, width, BLOCK_SIZE)
            ]

            # 6. Process blocks in a loop
            for win in windows:
                # [Read] Only read the current window's 13-18 bands (S1-S6 tBreak)
                # range(13, 19) ensures only these six bands are read, avoiding Region 31 index 26 errors
                try:
                    tbreaks = src_brk.read(list(range(13, 19)), window=win)
                except Exception as e:
                    # If the file does not even contain 18 bands, this exception will be caught here
                    print(f"❌ Failed to read break bands: {e}")
                    continue

                # Initialize the output block buffer (bands, win_height, win_width)
                out_block = np.full((num_output_bands, win.height, win.width), np.nan, dtype=np.float32)

                # Compute segment by segment
                for seg_idx in range(max_segments):
                    seg_num = seg_idx + 1
                    t_vals = tbreaks[seg_idx] # Time values for the current segment
                    
                    # Mask: valid time values
                    valid_mask = (t_vals > 0) & (~np.isnan(t_vals))
                    if not np.any(valid_mask):
                        continue

                    # Convert time to year indices (vectorized lookup)
                    years = t_vals[valid_mask].astype(int)
                    # Get float year values from LOOKUP_TABLE; fill with nan if the year is missing
                    t_pre = np.array([LOOKUP_TABLE.get(y-1, np.nan) for y in years], dtype=np.float32)
                    t_post = np.array([LOOKUP_TABLE.get(y+1, np.nan) for y in years], dtype=np.float32)

                    # Process band by band
                    for band in target_bands:
                        out_idx = desc_map[(seg_num, band)]

                        # [Skip logic] NIR and SWIR2 are skipped directly; the corresponding out_block positions remain NaN
                        if band in skip_bands:
                            continue
                        
                        # If the coefficient file is missing, skip it
                        if band not in src_coefs:
                            continue

                        try:
                            # Get coefficient descriptions to find indices
                            descs = src_coefs[band].descriptions
                            
                            # Find the indices of the four coefficients (SLP, INTP)
                            idx_slp = get_coef_idx(descs, f"S{seg_num}", band, "SLP")
                            idx_intp = get_coef_idx(descs, f"S{seg_num}", band, "INTP")
                            idx_n_slp = get_coef_idx(descs, f"S{seg_num+1}", band, "SLP")
                            idx_n_intp = get_coef_idx(descs, f"S{seg_num+1}", band, "INTP")

                            if None in [idx_slp, idx_intp, idx_n_slp, idx_n_intp]:
                                continue

                            # [Read] Only read the coefficients for the current window (rasterio indices start at 1)
                            c_slp = src_coefs[band].read(idx_slp + 1, window=win)[valid_mask]
                            c_intp = src_coefs[band].read(idx_intp + 1, window=win)[valid_mask]
                            c_n_slp = src_coefs[band].read(idx_n_slp + 1, window=win)[valid_mask]
                            c_n_intp = src_coefs[band].read(idx_n_intp + 1, window=win)[valid_mask]

                            # [Calculate]
                            y_pre = c_slp * t_pre + c_intp
                            y_post = c_n_slp * t_post + c_n_intp
                            delta = y_post - y_pre

                            # [Fill]
                            out_block[out_idx, valid_mask] = delta

                        except Exception:
                            continue # An error in one band does not affect the whole process

                # 7. Write the computed block to disk
                dst.write(out_block, window=win)

        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Region {region_id} crashed: {str(e)}"

# ====================== Main program ======================
if __name__ == "__main__":
    break_files = [f for f in os.listdir(rf"{base_path}/input/break") if f.endswith("_break_monthly.tif")]
    region_ids = [f.split('_')[0] for f in break_files]
    
    print(f"Detected {len(region_ids)} regions; using {WORKERS} processes for parallel processing...")

    success_list = []
    fail_list = []

    # Use ProcessPoolExecutor (more stable for Rasterio)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_region, rid): rid for rid in region_ids}
        
        for future in as_completed(futures):
            rid = futures[future]
            try:
                result = future.result()
                if result is True:
                    print(f"✅ Region {rid} completed")
                    success_list.append(rid)
                else:
                    print(result) # Print the error information
                    fail_list.append(rid)
            except Exception as e:
                print(f"❌ Region {rid} encountered an unknown error: {e}")
                fail_list.append(rid)

    print("\n===== Run report =====")
    print(f"✅ Success: {len(success_list)} -> {success_list}")
    print(f"❌ Failed: {len(fail_list)} -> {fail_list}")
    print("====================")


