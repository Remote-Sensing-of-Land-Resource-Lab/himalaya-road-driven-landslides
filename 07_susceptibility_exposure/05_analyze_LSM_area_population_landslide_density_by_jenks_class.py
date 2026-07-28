
import rasterio
import numpy as np
import pandas as pd
import os
import re
from rasterio.warp import transform

tasks = [
    {
        "name": "2000-2019",
        "tif": r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map\LSM_2000_2019.tif",
        "pop_tif": r"D:\dataset\Population_Landscan_1km_yearly\Himalaya_Pop_00_19_mean_30.tif",
        "start_year": 2000,
        "end_year": 2019
    },
    {
        "name": "2020-2024",
        "tif": r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map\LSM_2020_2024.tif",
        "pop_tif": r"D:\dataset\Population_Landscan_1km_yearly\Himalaya_Pop_20_24_mean_30.tif",
        "start_year": 2020,
        "end_year": 2024
    }
]

landslide_csv = r"H:\Himalaya\RF_susceptibility\features_pos.csv"
analysis_dir = r"H:\Himalaya\RF_susceptibility\susceptibility\c.analysis"
jenks_breaks_csv = os.path.join(analysis_dir, "joint_jenks_breaks.csv")

output_csv = os.path.join(analysis_dir, "LSM_Landslide_Density_By_Jenks_Class.csv")

CLASS_NAMES = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']


from rasterio.warp import transform

def load_jenks_bins(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Jenks threshold table not found: {csv_path}\n"
            "Please run 04_analyze_LSM_area_distribution_and_class_change.py first."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Range" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'Range' column.")

    def parse_edges(range_series):
        edges = []
        for range_text in range_series.astype(str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", range_text)
            if len(nums) < 2:
                raise ValueError(f"Cannot parse threshold range: {range_text}")
            left, right = float(nums[0]), float(nums[1])
            if not edges:
                edges.append(left)
            edges.append(right)

        if len(edges) != 6:
            raise ValueError(
                f"Expected 5 Jenks classes / 6 bin edges, got {len(edges)} edges."
            )

        edges[0] = 0.0
        edges[-1] = 1.000000001
        return np.array(edges, dtype=float)

    bins_by_period = {}
    if "Period" in df.columns:
        for period_name, df_period in df.groupby("Period", sort=False):
            bins_by_period[str(period_name)] = parse_edges(df_period["Range"])
    else:
        bins_by_period["joint"] = parse_edges(df["Range"])

    print(f">>> Loaded Jenks class bins from 04 result table: {csv_path}")
    for period_name, edges in bins_by_period.items():
        print(f"    {period_name}: " + ", ".join(f"{v:.6f}" for v in edges))

    return bins_by_period


def get_pixel_area_km2(src):
    res_x, res_y = src.res
    if src.crs is not None and src.crs.is_projected:
        return abs(res_x * res_y) / 1_000_000.0
    if src.crs is not None and src.crs.is_geographic:
        print(" [Warning] Geographic CRS detected; pixel area is approximated.")
        return abs(res_x * 111.32) * abs(res_y * 111.32)
    raise ValueError("Cannot determine raster CRS for pixel-area calculation.")


def analyze_period(task, df_all_points, class_bins):
    period_name = task['name']
    tif_path = task['tif']
    pop_path = task.get('pop_tif')
    start_year = task['start_year']
    end_year = task['end_year']
    n_years = end_year - start_year + 1    

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs        
        data = src.read(1)
        if src.nodata is not None:
            valid_mask = data != src.nodata
        else:
            valid_mask = ~np.isnan(data)
            
        valid_class_mask = (
            valid_mask
            & (data >= class_bins[0])
            & (data <= class_bins[-1])
        )
        valid_pixels = data[valid_class_mask]
        total_pixels = valid_pixels.size
        pixel_area_km2 = get_pixel_area_km2(src)
        
        pixel_classes = np.digitize(valid_pixels, class_bins, right=False) - 1
        pixel_classes = np.clip(pixel_classes, 0, len(CLASS_NAMES) - 1)
        class_map = np.full(data.shape, -1, dtype=np.int8)
        class_map[valid_class_mask] = pixel_classes.astype(np.int8)
        unique, counts = np.unique(pixel_classes, return_counts=True)
        area_stats = dict(zip(unique, counts))

        pop_stats = {i: 0.0 for i in range(len(CLASS_NAMES))}
        if pop_path is None:
            print(" [Warning] No population raster configured for this period.")
        elif not os.path.exists(pop_path):
            print(f" [Warning] Population raster not found: {pop_path}")
        else:
            with rasterio.open(pop_path) as pop_src:
                pop_data = pop_src.read(1).astype(float)
                if pop_src.nodata is not None:
                    pop_data[pop_data == pop_src.nodata] = 0
                pop_data[~np.isfinite(pop_data)] = 0
                pop_data[pop_data < 0] = 0

                pop_class_map = class_map
                if pop_data.shape != pop_class_map.shape:
                    print(
                        " [Warning] LSM and population raster shapes do not match; "
                        "using their overlapping rows/columns."
                    )
                    rows = min(pop_data.shape[0], pop_class_map.shape[0])
                    cols = min(pop_data.shape[1], pop_class_map.shape[1])
                    pop_data = pop_data[:rows, :cols]
                    pop_class_map = pop_class_map[:rows, :cols]

                for cls_id in range(len(CLASS_NAMES)):
                    pop_stats[cls_id] = float(np.sum(pop_data[pop_class_map == cls_id]))

        df_period = df_all_points[
            (df_all_points['year'] >= start_year) & 
            (df_all_points['year'] <= end_year) & 
            (df_all_points['label'] == 1)
        ]
        total_points = len(df_period)
        
        point_stats = {}
        if total_points > 0:
            lons = df_period['lon'].values
            lats = df_period['lat'].values
            
            if raster_crs != 'EPSG:4326':
                # source crs: EPSG:4326 (WGS84)
                xs, ys = transform('EPSG:4326', raster_crs, lons, lats)
                coords = list(zip(xs, ys))
            else:
                coords = list(zip(lons, lats))
            
            point_values = []
            for val in src.sample(coords):
                point_values.append(val[0])
            point_values = np.array(point_values)
            
            valid_pt_mask = ~np.isnan(point_values) & (point_values != src.nodata)
            if src.nodata is None:
                 valid_pt_mask = valid_pt_mask & (point_values != -9999)
            
            point_values = point_values[valid_pt_mask]
            point_values = point_values[
                (point_values >= class_bins[0])
                & (point_values <= class_bins[-1])
            ]
            
            pt_classes = np.digitize(point_values, class_bins, right=False) - 1
            pt_classes = np.clip(pt_classes, 0, len(CLASS_NAMES) - 1)
            u_pt, c_pt = np.unique(pt_classes, return_counts=True)
            point_stats = dict(zip(u_pt, c_pt))

    results = []
    for i in range(5):
        t_min = class_bins[i]
        t_max = class_bins[i + 1]
        if i == 4:
            t_max = 1.0
        
        n_pixels = area_stats.get(i, 0)
        area_pct = (n_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        n_points = point_stats.get(i, 0)
        annual_n_points = n_points / n_years if n_years > 0 else np.nan
    
        actual_total_points = len(point_values) if total_points > 0 else 0
        point_pct = (n_points / actual_total_points) * 100 if actual_total_points > 0 else 0
        
        fr = point_pct / area_pct if area_pct > 0 else 0
        
        class_area_km2 = n_pixels * pixel_area_km2
        population = pop_stats.get(i, 0.0)
        annual_density = (
            (annual_n_points / class_area_km2) * 100
            if class_area_km2 > 0
            else 0
        )
        
        results.append({
            'Period': period_name,
            'Years': n_years,
            'Level': CLASS_NAMES[i],
            'Range': f"{t_min:.6f} - {t_max:.6f}",
            'Area_km2': round(class_area_km2, 2),
            'Area_Pct(%)': round(area_pct, 2),
            'Population': round(population, 2),
            'Population_Million': round(population / 1_000_000.0, 4),
            'Landslide_Pct(%)': round(point_pct, 2),
            'Frequency_Ratio': round(fr, 2),
            'Landslide_Count': n_points,
            'Annual_Landslide_Count': round(annual_n_points, 2),
            'Pixel_Count': n_pixels,
            'Annual_LS_Density(per 100km2 yr)': round(annual_density, 4)
        })
        
    return results


if __name__ == "__main__":
    df_all = pd.read_csv(landslide_csv)
    bins_by_period = load_jenks_bins(jenks_breaks_csv)
    
    all_results = []
    
    for task in tasks:
        class_bins = bins_by_period.get(task["name"])
        if class_bins is None:
            class_bins = bins_by_period.get("joint")
        if class_bins is None:
            raise ValueError(f"No Jenks bins found for period: {task['name']}")

        task_res = analyze_period(task, df_all, class_bins)
        all_results.extend(task_res)
        
    df_final = pd.DataFrame(all_results)
    
    cols = [
        'Period',
        'Years',
        'Level',
        'Range',
        'Area_km2',
        'Area_Pct(%)',
        'Population',
        'Population_Million',
        'Landslide_Pct(%)',
        'Frequency_Ratio',
        'Landslide_Count',
        'Annual_Landslide_Count',
        'Pixel_Count',
        'Annual_LS_Density(per 100km2 yr)',
    ]
    df_final = df_final[cols]
    
    print("\n" + "="*80)
    print(df_final)
    print("="*80)
    
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_final.to_csv(output_csv, index=False)



