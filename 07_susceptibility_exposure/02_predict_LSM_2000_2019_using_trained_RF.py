import os
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import gc
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

model_path = r"H:\Himalaya\RF_susceptibility\RF_model_all_samples_5foldcv.pkl"
feature_dir = r"H:\Himalaya\RF_susceptibility\susceptibility\a.features_tiffs_input"
output_tif = r"H:\Himalaya\RF_susceptibility\susceptibility\b.result_map\LSM_2000_2019_5foldcv_new.tif"

BLOCK_SIZE = 4096 

feature_names = [
    'dist_to_fault', 'dist_to_road', 'dist_to_water', 'elevation', 'NDVI', 
    'slope', 'Annual_Mean', 'aspect_sin', 'aspect_cos', 
    'Litho_1.0', 'Litho_2.0', 'Litho_3.0', 'Litho_4.0', 'Litho_5.0', 'Litho_6.0', 'Litho_7.0', 
    'Litho_8.0', 'Litho_9.0', 'Litho_10.0', 'Litho_11.0', 'Litho_12.0', 'Litho_13.0', 'Litho_14.0', 
    'plan_curv', 'profile_curv', 
    'LC_10', 'LC_20', 'LC_30', 'LC_40', 'LC_50', 'LC_60', 'LC_80', 'LC_90', 'LC_100'
]


file_mapping = {
    'dist_to_fault': 'dist_to_fault_new.tif', 
    'dist_to_road':  'dist_to_road_2019_new.tif', 
    'dist_to_water': 'dist_to_water_new.tif',
    'elevation':     'D:/dataset/30mDEM.tif',
    
    'NDVI':          'medium_NDVI_2019.tif',  
    
    'slope':         'D:/dataset/slope_from_dem.tif',
    
    'Annual_Mean':   'aligned_pre_2019_new.tif',  
    
    'aspect_sin':    'D:/aspect_from_dem_sin.tif',
    'aspect_cos':    'D:/aspect_from_dem_cos.tif',
    'plan_curv':     'D:/plan_curvature.tif',
    'profile_curv':  'D:/profile_curvature.tif',
    
    'LC_Raw':     'D:/LULC_2020_LSM.tif', 
    'Litho_Raw':        'D:/GLiM_LSM.tif'  
}

def predict_large_raster_optimized():
    rf_model = joblib.load(model_path)
    rf_model.n_jobs = -1 

    src_dict = {}
    src_lith = None
    src_lc = None

    try:
        for feat in feature_names:
            if "Litho" in feat or "LC_" in feat: continue 
            fname = file_mapping.get(feat)
            if not fname: raise ValueError(f"Missing mapping for {feat}")
            path = fname if os.path.isabs(fname) else os.path.join(feature_dir, fname)
            if not os.path.exists(path): raise FileNotFoundError(f"Missing file: {path}")
            src_dict[feat] = rasterio.open(path)

        path_lith = file_mapping['Litho_Raw']
        path_lc = file_mapping['LC_Raw']
        src_lith = rasterio.open(path_lith)
        src_lc = rasterio.open(path_lc)
        
        ref_src = src_dict['elevation']
        profile = ref_src.profile.copy()
        profile.update(
            dtype=rasterio.float32, 
            count=1, 
            compress='lzw', 
            nodata=-9999, 
            BIGTIFF='YES', 
            tiled=True,
            blockxsize=512,
            blockysize=512
        )
        
        width = ref_src.width
        height = ref_src.height
        n_features = len(feature_names)
        
        with rasterio.open(output_tif, 'w', **profile) as dst:
            
            windows = []
            for col_off in range(0, width, BLOCK_SIZE):
                for row_off in range(0, height, BLOCK_SIZE):
                    w = min(BLOCK_SIZE, width - col_off)
                    h = min(BLOCK_SIZE, height - row_off)
                    windows.append(Window(col_off, row_off, w, h))

            for window in tqdm(windows, desc="Processing"):
                rows = window.height
                cols = window.width
                n_pixels = rows * cols
                
                data_lith = src_lith.read(1, window=window).reshape(-1)
                data_lc = src_lc.read(1, window=window).reshape(-1)

                X_block = np.empty((n_pixels, n_features), dtype=np.float32)

                for i, fname in enumerate(feature_names):
                    if fname.startswith("Litho_"):
                        cid = int(float(fname.split("_")[1]))
                      
                        X_block[:, i] = (data_lith == cid)
                        
                    elif fname.startswith("LC_"):
                        cid = int(fname.split("_")[1])
                        X_block[:, i] = (data_lc == cid)
                        
                    else:
                        X_block[:, i] = src_dict[fname].read(1, window=window).reshape(-1)

                valid_mask = ~np.isnan(X_block).any(axis=1)
                
                y_pred = np.full(n_pixels, -9999, dtype=np.float32)
                
                if np.sum(valid_mask) > 0:
                    probs = rf_model.predict_proba(X_block[valid_mask])[:, 1]
                    y_pred[valid_mask] = probs
                
                dst.write(y_pred.reshape(rows, cols), 1, window=window)
                
                del X_block, y_pred

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        for src in src_dict.values(): src.close()
        if src_lith: src_lith.close()
        if src_lc: src_lc.close()

if __name__ == "__main__":
    predict_large_raster_optimized()
