# Changes in Feature Importance of Hotspot Areas (12 Variables)
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pymannkendall as mk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D 

P_thre = 0.05
N_thre = 250

# ==========================================
#1. Spatial Hotspot Selection (Lock Analysis Area)
# ==========================================
def get_hotspot_ids(csv_path, shp_path):
    print("🔹 Identifying spatial hotspot grids...")
    df_raw = pd.read_csv(csv_path)
    gdf_raw = gpd.GeoDataFrame(df_raw, geometry=gpd.points_from_xy(df_raw.lon, df_raw.lat), crs="EPSG:4326")
    grid = gpd.read_file(shp_path)
    if gdf_raw.crs != grid.crs:
        gdf_raw = gdf_raw.to_crs(grid.crs)
    
    joined = gpd.sjoin(gdf_raw, grid, how="inner", predicate="within")
    
    annual_counts = joined.groupby(['index_right', 'year']).size().unstack(fill_value=0)
    all_years = list(range(2000, 2025))
    annual_counts = annual_counts.reindex(columns=all_years, fill_value=0).reindex(grid.index, fill_value=0)
    
    total_sums = annual_counts.sum(axis=1).values
    hotspot_ids = []
    
    for i in range(len(grid)):
        if total_sums[i] > N_thre:
            try:
                res = mk.original_test(annual_counts.values[i])
                if res.p < P_thre and res.slope > 0:
                    hotspot_ids.append(grid.index[i])
            except:
                pass

    print(f"✅ Found significant increasing hotspot grids: {len(hotspot_ids)}")
    return hotspot_ids


# 设置文件路径
csv_input = "H:/Himalaya/RF_susceptibility/features_pos.csv"
shp_input = "H:/Himalaya/grid/Himalaya_hex_1000km2/Himalaya_hex_1000km2.shp"
model_input = "H:/Himalaya/RF_susceptibility/features_all.csv"

# 获取热点 ID
hotspot_indices = get_hotspot_ids(csv_input, shp_input)

# ==========================================
# 2. Load modeling data and apply spatial filtering
# ==========================================
print("🔹 Identifying spatial hotspot grids...")
df_full = pd.read_csv(model_input)
gdf_model = gpd.GeoDataFrame(df_full, geometry=gpd.points_from_xy(df_full.lon, df_full.lat), crs="EPSG:4326")
grid_shp = gpd.read_file(shp_input)
if gdf_model.crs != grid_shp.crs:
    gdf_model = gdf_model.to_crs(grid_shp.crs)

df_spatial = gpd.sjoin(gdf_model, grid_shp, how="inner", predicate="within")
df_hotspots = df_spatial[df_spatial['index_right'].isin(hotspot_indices)].copy()

# ==========================================
# 3. Data Cleaning and Modeling Variable Definitions
# ==========================================

# Continuous Variables
continuous_features = [
    'elevation', 'slope', 'aspect', 'dist_to_road',
    'plan_curv', 'profile_curv', 'NDVI',
    'Annual_Mean', 'dist_to_fault', 'dist_to_water'
]

# Lithology column after one-hot encoding
lithology_features = [
    'Litho_10', 'Litho_100', 'Litho_110', 'Litho_120', 'Litho_130', 'Litho_140',
    'Litho_20', 'Litho_30', 'Litho_40', 'Litho_50', 'Litho_60', 'Litho_70',
    'Litho_80', 'Litho_90'
]

# landcover column after one-hot encoding
landcover_features = [
    'LC_10', 'LC_100', 'LC_20', 'LC_30', 'LC_40', 'LC_50', 'LC_60', 'LC_80', 'LC_90'
]

# Actual input variables of the model
model_features = continuous_features + lithology_features + landcover_features

grouped_features = [
    'elevation', 'slope', 'aspect', 'dist_to_road',
    'plan_curv', 'profile_curv', 'NDVI',
    'Annual_Mean', 'dist_to_fault', 'dist_to_water',
    'lithology', 'landcover'
]

for col in continuous_features:
    df_hotspots[col] = pd.to_numeric(df_hotspots[col], errors='coerce')

for col in lithology_features + landcover_features:
    df_hotspots[col] = df_hotspots[col].astype(str).str.strip().replace({
        'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0
    })
    df_hotspots[col] = pd.to_numeric(df_hotspots[col], errors='coerce').fillna(0).astype(int)

feature_medians = df_hotspots[continuous_features].median()
df_hotspots[continuous_features] = df_hotspots[continuous_features].fillna(feature_medians)

data_alpha = df_hotspots[(df_hotspots['year'] >= 2000) & (df_hotspots['year'] <= 2014)]
data_beta = df_hotspots[(df_hotspots['year'] >= 2015) & (df_hotspots['year'] <= 2024)]


# ==========================================
# 4. Train the model
# ==========================================
def aggregate_importances(feature_names, importances):
    """
    Summarize the feature importance for one-hot encoded features:
- All Litho_* -> lithology
- All LC_* -> landcover
- Keep the original names for the remaining continuous variables
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    def map_group(feat):
        if feat.startswith('Litho_'):
            return 'lithology'
        elif feat.startswith('LC_'):
            return 'landcover'
        else:
            return feat

    importance_df['Group'] = importance_df['Feature'].apply(map_group)
    grouped = importance_df.groupby('Group', as_index=False)['Importance'].sum()

    grouped['Group'] = pd.Categorical(grouped['Group'], categories=grouped_features, ordered=True)
    grouped = grouped.sort_values('Group')

    return grouped['Importance'].values


def preprocess_and_train(data, name):
    X = data[model_features].copy()
    y = data['label']

    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X[continuous_features])
    X_scaled = X.copy()
    X_scaled[continuous_features] = X_scaled_values

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"--- {name} AUC: {auc:.4f} ---")

    # 汇总 one-hot 重要性
    grouped_importances = aggregate_importances(X_scaled.columns.tolist(), rf.feature_importances_)

    return rf, grouped_importances, X_scaled


rf_alpha, importances_alpha, X_alpha_all = preprocess_and_train(data_alpha, "Period Alpha")
rf_beta, importances_beta, X_beta_all = preprocess_and_train(data_beta, "Period Beta")

print("🔹 Export the feature importance results for the two periods as CSV...")

importance_df = pd.DataFrame({
    'Feature': grouped_features,
    'Importance_1': importances_alpha,
    'Importance_2': importances_beta
})

importance_df['Difference'] = importance_df['Importance_2'] - importance_df['Importance_1']
importance_df = importance_df.sort_values(by='Importance_2', ascending=False)

csv_save_path = "H:/Himalaya/RF_susceptibility/feature_importance_comparison.csv"
importance_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

print(f"Feature importance CSV has been saved: {csv_save_path}")
print(importance_df)


from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

def plot_importance_change_signed(features, imp_alpha, imp_beta):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    name_mapping = {
        'elevation': 'Ele.',
        'slope': 'Slope',
        'aspect': 'Aspect',
        'lithology': 'Litho.',
        'dist_to_road': 'Dist. to road',
        'Annual_Mean': 'MAP',
        'dist_to_fault': 'Dist. to fault',
        'dist_to_water': 'Dist. to water',
        'NDVI': 'NDVI',
        'plan_curv': 'Plan curv.',
        'profile_curv': 'Profile curv.',
        'landcover': 'LULC'
    }

    df_plot = pd.DataFrame({
        'Raw_Feature': features,
        'Historical': imp_alpha,
        'Recent': imp_beta
    })

    df_plot['Difference'] = df_plot['Recent'] - df_plot['Historical']
    df_plot['Feature'] = df_plot['Raw_Feature'].map(name_mapping).fillna(df_plot['Raw_Feature'])

    df_plot = df_plot.sort_values(by='Difference', ascending=False).reset_index(drop=True)

    increase_cmap = LinearSegmentedColormap.from_list(
        'increase_teal_gradient',
        ['#F3D8DA', '#D28A91', '#B04A5A']   
    )

    decrease_cmap = LinearSegmentedColormap.from_list(
        'decrease_blue_gradient',
       ['#DCE7F1', '#8AA9C8', '#4C78A8']   
    )

    c_zero = '#8E8E8E'
    c_text = '#222222'

    pos_abs = df_plot.loc[df_plot['Difference'] > 0, 'Difference']
    neg_abs = -df_plot.loc[df_plot['Difference'] < 0, 'Difference']

    pos_norm = Normalize(
        vmin=0,
        vmax=pos_abs.max() if len(pos_abs) > 0 else 1
    )

    neg_norm = Normalize(
        vmin=0,
        vmax=neg_abs.max() if len(neg_abs) > 0 else 1
    )

    def assign_gradient_color(v):
        if v > 0:
            return increase_cmap(pos_norm(v))
        elif v < 0:
            return decrease_cmap(neg_norm(abs(v)))
        else:
            return '#D9D9D9'

    df_plot['Color'] = df_plot['Difference'].apply(assign_gradient_color)

    plt.rcParams.update({
        'font.size': 34,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })

    fig, ax = plt.subplots(figsize=(13, 10), dpi=300)

    y = np.arange(len(df_plot))

    ax.barh(
        y,
        df_plot['Difference'],
        color=df_plot['Color'],
        edgecolor='none',
        height=0.62,
        zorder=3
    )

    ax.axvline(0, color=c_zero, lw=1.5, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot['Feature'], fontsize=30)
    ax.tick_params(axis='y', pad=12)
    ax.invert_yaxis()

    ax.set_xlabel('Change in importance', fontsize=31, labelpad=12)
    ax.tick_params(axis='x', labelsize=29, pad=7)
    ax.tick_params(axis='both', which='major', length=8)

    xlim = 0.045
    ax.set_xlim(-xlim, xlim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    for i, v in enumerate(df_plot['Difference']):
        offset = xlim * 0.025
        if v >= 0:
            ax.text(
                v + offset, i, f'{v:+.3f}',
                va='center', ha='left',
                fontsize=25, color=c_text
            )
        else:
            ax.text(
                v - offset, i, f'{v:+.3f}',
                va='center', ha='right',
                fontsize=25, color=c_text
            )

    legend_elements = [
        Patch(facecolor='#B04A5A', edgecolor='none', label='Increased'),
        Patch(facecolor='#4C78A8', edgecolor='none', label='Decreased')
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=30,
        loc='lower right'
    )

    plt.tight_layout()

    save_path = "H:/Himalaya/figure/HOT_Importance_Change_Signed_Nature_Gradient.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()

    print(f"Feature importance change plot has been saved: {save_path}")


plot_importance_change_signed(grouped_features, importances_alpha, importances_beta)

