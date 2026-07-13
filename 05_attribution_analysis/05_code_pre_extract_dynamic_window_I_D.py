#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic-window extraction of landslide-related rainfall events:

window_start = CCDC_date - mean_doy_difference
window_end = CCDC_date

Among all continuous rainfall segments:
R_end > window_start AND R_start < window_end defines candidate events.
Among candidate events, select the event whose rainfall peak is closest to the landslide reference time (window_end).
"""

import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm

# -----------------------------
# Parameter settings
# -----------------------------
LANDSLIDE_CSV = r"H:/Himalaya/RF_model/negative_sample_new2.csv"
SHAPEFILE = r"H:/Himalaya/grid/Him/Himalaya.shp"
CHIRPS_DIR = r"H:/Himalaya/cause/Precipitation/CHIRPS_download_1999_2025"
CLIP_NC = os.path.join(CHIRPS_DIR, "CHIRPS_Himalaya_1999_2025_p05.nc")
OUT_CSV = r"H:/Himalaya/RF_model/negative_sample_with_pre.csv"

THRESHOLD = 1.0  # mm/day
GAP_DAYS = 2
MAX_PROCS_EXTRACT = max(1, cpu_count() - 1)

# -----------------------------
# 1. Clip CHIRPS data to the shapefile region
# -----------------------------
def clip_chirps_files_to_shape(nc_files, shapefile, out_nc):
    if os.path.exists(out_nc):
        print("✅ Found existing clipped file:", out_nc)
        return xr.open_dataset(out_nc)

    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    shapes = [mapping(geom) for geom in gdf.geometry]
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]

    clipped_list = []
    for nc_path in tqdm(nc_files, desc="Clipping CHIRPS by shapefile"):
        try:
            ds = xr.open_dataset(nc_path)
            varname = "precip" if "precip" in ds else "precipitation"

            ds_sel = ds.sel(longitude=slice(bbox[0], bbox[2]),
                            latitude=slice(bbox[3], bbox[1]))

            data = ds_sel[varname].to_numpy()
            lon = ds_sel["longitude"].values
            lat = ds_sel["latitude"].values
            time = ds_sel["time"].values

            # Clip using a raster mask.
            masked = np.zeros_like(data)
            from rasterio.transform import from_bounds
            import rasterio

            transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                                    len(lon), len(lat))
            dummy_profile = {
                "driver": "GTiff",
                "height": len(lat),
                "width": len(lon),
                "count": 1,
                "dtype": "float32",
                "transform": transform,
                "crs": "EPSG:4326"
            }

            for i in range(data.shape[0]):
                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(**dummy_profile) as tmp:
                        tmp.write(data[i, :, :].astype("float32"), 1)
                        out_img, _ = mask(tmp, shapes, crop=True, nodata=np.nan)
                masked[i, :, :] = out_img[0]

            ds_masked = xr.Dataset(
                {"precip": (("time", "latitude", "longitude"), masked)},
                coords={"time": time, "latitude": lat, "longitude": lon}
            )
            clipped_list.append(ds_masked)
        except Exception as e:
            print("⚠️ Clip failed:", nc_path, e)

    if not clipped_list:
        raise RuntimeError("No CHIRPS files successfully clipped.")

    merged = xr.concat(clipped_list, dim="time")
    merged.to_netcdf(out_nc)
    print(f"Saved clipped CHIRPS: {out_nc}")
    return merged

# -----------------------------
# 2. Load landslide points
# -----------------------------


def load_points(csv_path):
    with open(csv_path, "r", encoding="gbk", errors="ignore") as f:
        df = pd.read_csv(f)

    required = ("lon", "lat", "year_neg", "DOY", "mean_doy_difference")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["CCDC_date"] = df.apply(
        lambda r: datetime(int(r["year_neg"]), 1, 1) + timedelta(days=int(r["DOY"]) - 1),
        axis=1,
    )
    return df

# -----------------------------
# 3. Extract rainfall event D/E/I using a dynamic window
# -----------------------------
ds_cached = None  

def init_worker(ds):
    """Initialize the cached dataset in each worker process."""
    global ds_cached
    ds_cached = ds

def extract_point_event(args):

    idx, lon, lat, ccdc_date, diff, zone = args
    global ds_cached
    # debug = True

    try:
        window_start = ccdc_date - timedelta(days=int(diff))
        window_end = ccdc_date
        
        sub = ds_cached.sel(longitude=lon, latitude=lat, method="nearest") \
                       .sortby("time") \
                       .sel(time=slice(window_start - timedelta(days=30), window_end))  # Start earlier to avoid truncating rainfall segments.

        rain = sub["precip"].values
        times = sub["time"].values

        if rain is None or len(rain) == 0:
            return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, np.nan, np.nan, np.nan, np.nan)

        rain = np.nan_to_num(rain, nan=0.0)

        # ----------------------
        # 1. Identify continuous rainfall segments.
        # ----------------------
        segments = []
        start = None
        gap = 0

        for i in range(len(rain)):
            if rain[i] >= THRESHOLD:
                if start is None:
                    start = i
                gap = 0
            else:
                gap += 1
                if start is not None and gap >= GAP_DAYS:
                    segments.append((start, i - gap))
                    start = None
                    gap = 0
        if start is not None:
            segments.append((start, len(rain) - 1))
            
        # if debug:
        #     print(f"\nDEBUG idx={idx}, lon={lon}, lat={lat}, zone={zone}")
        #     print(f"Window: {window_start} ~ {window_end}")
        #     print("Times:", times[:50])
        #     print("Rain (first 100 days):", rain[:100])
        #     print("Segments (up to 10):", segments[:10])


        if len(segments) == 0:
            return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, np.nan, np.nan, np.nan, np.nan)

        # ----------------------
        # 2. Filter events that overlap the dynamic window.
        #    R_end > window_start AND R_start < window_end.
        # ----------------------
        candidates = []
        for s, e in segments:
            R_start_time = times[s]
            R_end_time = times[e]
            if (R_end_time > np.datetime64(window_start)) and (R_start_time < np.datetime64(window_end)):
                candidates.append((s, e))
        
        # if debug:
        #     print("Candidate segments (up to 10):", candidates[:10])

        if not candidates:
            return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, np.nan, np.nan, np.nan, np.nan)

        # ----------------------
        # 3. Select the event whose rainfall peak is closest to the landslide reference date.
        # ----------------------
        best_event = None
        min_diff = np.inf

        for s, e in candidates:
            event_rain = rain[s:e+1]
            event_time = times[s:e+1]

            peak_idx = np.argmax(event_rain)
            peak_time = event_time[peak_idx]

            diff_days = abs((peak_time - np.datetime64(window_end)).astype('timedelta64[D]').astype(int))

            # if debug:
            #     print(f"Candidate {s}-{e}, peak_time={peak_time}, diff_days={diff_days}, sum_rain={event_rain.sum()}")

            if diff_days < min_diff:
                min_diff = diff_days
                best_event = (s, e)
        
        # if debug:
        #     print("Best event selected:", best_event)


        if best_event is None:
            return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, np.nan, np.nan, np.nan, np.nan)

        best_s, best_e = best_event
        event = rain[best_s:best_e+1]
        D = len(event)
        E = event.sum()
        I = E / D if D > 0 else np.nan

        return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, D, E, I, min_diff)

    except Exception:
        return (idx, lon, lat, ccdc_date.strftime("%Y-%m-%d"), zone, np.nan, np.nan, np.nan, np.nan)

# -----------------------------
# Main routine
# -----------------------------
if __name__ == "__main__":
    print("STEP 1: Reading CHIRPS files")
    nc_files = sorted(
        [os.path.join(CHIRPS_DIR, f) for f in os.listdir(CHIRPS_DIR)
         if f.endswith(".nc") and "chirps" in f.lower()]
    )
    print("CHIRPS files found:", len(nc_files))

    print("STEP 2: Clipping to the study area")
    ds_cached = clip_chirps_files_to_shape(nc_files, SHAPEFILE, CLIP_NC)
    print("Time range:", ds_cached.time.values[0], "to", ds_cached.time.values[-1])

    print("STEP 3: Loading landslide points")
    df = load_points(LANDSLIDE_CSV)
    print("Loaded landslides:", len(df))
    
    import numpy as np
    df["mean_doy_difference"] = np.ceil(df["mean_doy_difference"]).astype(int)

    args_list = [
    (i, r["lon"], r["lat"], r["CCDC_date"], r["mean_doy_difference"], r.get("zone", np.nan))
    for i, r in df.iterrows()
    ]

    print("STEP 4: Extracting rainfall events with multiprocessing")
    results = []
    with Pool(processes=MAX_PROCS_EXTRACT, initializer=init_worker, initargs=(ds_cached,)) as pool:
        for res in tqdm(pool.imap_unordered(extract_point_event, args_list), total=len(args_list)):
            results.append(res)
    
    # for args in args_list[:1]:  # First landslide point for debugging.
    #     res = extract_point_event(args)
    #     print(res)
    

    print("STEP 5: Saving results")
    df_out = pd.DataFrame(results, columns=["idx", "lon", "lat", "date", "zone", "D_days", "E_mm", "I_mm_day", "diff_days"])
    df_out = df_out.sort_values("idx").reset_index(drop=True)

    merged = pd.concat([df.reset_index(drop=True), df_out[["D_days", "E_mm", "I_mm_day", "diff_days"]]], axis=1)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Results saved to: {OUT_CSV}")
