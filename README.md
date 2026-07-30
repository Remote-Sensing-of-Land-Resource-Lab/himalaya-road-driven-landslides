# Rapid road expansion drives a recent surge in Himalayan landsliding

This repository contains the analysis workflow and code used for a study investigating how rapid road-network expansion has reshaped landslide regimes across the Himalaya.

By combining dense Landsat time series, annual road-growth records, and statistical and machine-learning analyses, the project reconstructs annual 30-m landslide activity from 2000 to 2024 and evaluates the role of roads in driving recent increases in landslide occurrence.

## Research Summary

Rapid infrastructure expansion is increasingly transforming mountain landscapes, and road construction is widely recognized as a local driver of slope instability. Yet it remains poorly constrained whether the cumulative expansion of road networks can reorganize landslide regimes at regional scales by changing when, where, and under what terrain conditions landslides occur.

This study reconstructs annual 30-m landslide activity across the Himalaya from 2000 to 2024 and identifies a marked transition around 2020. Mean annual landslide occurrence increased from 3,855 events during 2000–2019 to 7,490 events during 2020–2024. The surge was accompanied by a redistribution of failures toward lower elevations, gentler slopes, and denser vegetation cover. A large share of the increase occurred within 1 km of roads, while precipitation and recent earthquake activity did not show changes consistent with the timing or geography of the surge.

These findings indicate an emerging infrastructure-associated reorganization of Himalayan landsliding and highlight the need to account for delayed road impacts as transport networks expand into increasingly sensitive landscapes.

## Project Structure

```text
himalaya-road-driven-landslides/
├── 01_landslide_inventory_mapping/
│   ├── gee/
│   └── python/
├── 02_inventory_validation/
├── 03_temporal_trend_and_breakpoint/
├── 04_spatial_hotspots_and_geomorphic_shift/
├── 05_attribution_analysis/
├── 06_ndvi_road_sensitivity/
└── 07_susceptibility_exposure/
```

## Workflow

```text
Remote sensing data
  → Landslide inventory mapping
  → Validation and accuracy assessment
  → Temporal trend and breakpoint analysis
  → Spatial hotspot analysis
  → Road attribution and causality analysis
  → NDVI-road sensitivity analysis
  → Landslide susceptibility and exposure assessment
```

## Module Descriptions

### 01_landslide_inventory_mapping

This module is used to construct landslide inventories and change-detection products, and it forms the foundation of the entire analysis workflow.

- 01_GEE_run_ccdc_break_detection.py
  - Uses Google Earth Engine and the CCDC (Continuous Change Detection) method for time-series change detection.
  - Mainly used to identify change points and potential landslide areas from Landsat image sequences.

- 02_delta_brightness.py
  - Calculates brightness differences between Landsat images from different years.
  - Helps identify surface reflectance and brightness anomalies caused by landslides.

- 03_delta_reflectance.py
  - Calculates reflectance changes across different bands.

- 04_deviation_angle.py
  - Calculates spectral deviation angles to assist in identifying surface change types.

- 05_tbreak_floodfill.py
  - Performs floodfill segmentation based on temporal breakpoints.
  - Can be used to extract spatially connected landslide patches from change-detection results.

- 06_select1.py, 07_select2.py, and 08_select3.py
  - Perform multi-stage filtering and quality control.
  - Mainly used to remove misclassified patches and improve the quality of the landslide inventory.

- 09_merge.py
  - Merges multi-year results to generate final landslide mapping products or annual result sets.

### 02_inventory_validation

This module is used to validate landslide mapping results and assess accuracy.

- 01_generate_stratified_validation_ls_samples.py
  - Generates stratified validation samples for landslide points.

- 02_generate_stratified_validation_nonls_samples.py
  - Generates stratified validation samples for non-landslide points.

- 03_area_calculation.py
  - Calculates landslide polygon areas with geodesic accuracy.

- 04_area_adjusted_accuracy_assessment.py
  - Conducts accuracy assessment with area-adjusted correction.

### 03_temporal_trend_and_breakpoint

This module analyzes the temporal trend of landslide counts and detects breakpoints.

- 01_annual_landslide_count_with_ci.py
  - Computes annual landslide counts and estimates confidence intervals.

- 02_detect_annual_landslide_count_breakpoint.py
  - Detects significant breakpoints in annual landslide count changes.
  - Often used together with statistical methods such as Mann-Kendall.

### 04_spatial_hotspots_and_geomorphic_shift

This module identifies spatial landslide hotspots and related geomorphic changes.

- 01_identify_significant_hotspot_grids.py
  - Identifies grid cells with significantly increasing landslide density.

- 02_calculate_density_change_by_grid.py
  - Calculates landslide density change trends for different grid cells.

- 03_compare_elevation_slope_ndvi_aspect.py
  - Compares elevation, slope, NDVI, and aspect differences between hotspot and non-hotspot areas.

### 05_attribution_analysis

This module explores the causal relationship and driving mechanism between roads and landslides.

- 01_calculate_landslides_inside_outside_1km_roads.py
  - Compares landslide density or counts inside and outside a 1 km road buffer.
- 02_event_study_new_roads_300_500_1000m.py
  - Conducts an event study for newly built roads and analyzes landslide changes before and after road construction.
  - Compares several buffer zones (300 m, 500 m, 1000 m) against background areas.
- 03_RF_feature_importance_compare.py
  - Evaluates the relative importance of terrain, climate, and road variables for landslide occurrence.
- 04_calculate_road_landslide_lag.py
  - Calculates the temporal lag between road construction and landslide occurrence.
- 05_code_pre_extract_dynamic_window_I_D.py
  - Used for preprocessing and feature extraction within dynamic time windows.

### 06_ndvi_road_sensitivity

This module analyzes the sensitivity of landslides to road distance under different vegetation conditions.

- 01_ndvi_road_sensitivity.py
  - Analyzes sensitivity curves between road distance and landslide occurrence by NDVI bins.
  - Uses random forest models and repeated resampling to estimate uncertainty.
- 02_sensitivity_in_Aspect_NDVI_road_RF.py
- 03_sensitivity_in_Ele_NDVI_road_RF.py
- 04_sensitivity_in_MAP_NDVI_road_RF.py
- 05_sensitivity_in_Slope_NDVI_road_RF.py
  - Analyze the interaction between NDVI and road effects from the perspectives of aspect, elevation, precipitation, and slope.
- 06_plot_2x2_sensitivity_panels.py
  - Produces Nature-style sensitivity analysis figures.

### 07_susceptibility_exposure

This module builds landslide susceptibility models and evaluates changes in exposure.

- 01_train_RF_landslide_susceptibility_5fold.py
  - Trains a random forest landslide susceptibility model.
  - Uses 5-fold cross-validation for model training and evaluation.

- 02_predict_LSM_2000_2019_using_trained_RF.py
  - Generates susceptibility maps for 2000–2019 based on the trained model.

- 03_predict_LSM_2020_2024_using_trained_RF.py
  - Generates susceptibility maps for 2020–2024 based on the trained model.

- 04_analyze_LSM_area_distribution_and_class_change.py
  - Analyzes the spatial area distribution and temporal changes of different susceptibility classes.

- 05_analyze_LSM_area_population_landslide_density_by_jenks_class.py
  - Uses Jenks natural breaks classification to analyze area, population, and landslide density changes.

- 06_plot_LSM_area_density_population_by_jenks_class.py
  - Plots area, density, and population distribution by susceptibility class.

- 07_plot_LSM_susceptibility_change_histogram.py
  - Plots histograms of susceptibility changes across time periods.

## Environment

The workflow primarily relies on the following Python packages:

- pandas
- numpy
- geopandas
- rasterio
- scikit-learn
- matplotlib
- scipy
- pymannkendall
- cartopy
- shapely
- joblib

Some scripts also depend on Google Earth Engine (GEE) or other remote sensing processing tools.
