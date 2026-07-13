# Slope, Aspect, Elevation, NDVI Transition Map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# --- 1. Global Style Settings ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'axes.linewidth': 5,   
    'font.size': 28,
    'axes.titlesize': 24,
    'axes.labelsize': 28,        
    'xtick.labelsize': 26,       
    'ytick.labelsize': 26,        
    'axes.titleweight': 'bold',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 5,   
    'ytick.major.width': 1.4,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2
})

# --- 2. Data Reading and Preprocessing ---
file_path = r'H:/Himalaya/RF_susceptibility/features_pos.csv'
df = pd.read_csv(file_path)

df['evt_date'] = pd.to_datetime(df['evt_date'])
df['year'] = df['evt_date'].dt.year
df['period'] = np.where(df['year'] < 2020, "2000–2019", "2020–2024")
df = df[(df['year'] >= 2000) & (df['year'] <= 2024)]

if 'NDVI' in df.columns:
    df['NDVI'] = df['NDVI'].clip(0, 1)


def plot_migration_analysis(
    df,
    output_path,
    aspect_label_pos=(0.5, -0.12),   
    aspect_p_pos=(0.70, 0.92)       
):
    colors = {"2000–2019": "#80b1d3", "2020–2024": "#fb8072"}
    periods = ["2000–2019", "2020–2024"]
    
    features = ['elevation', 'slope', 'aspect', 'NDVI']
    units = ['m', '°', '°', '']

    fig = plt.figure(figsize=(24, 7), dpi=300)
    
    ax1 = fig.add_subplot(141)  # Elevation
    ax2 = fig.add_subplot(142)  # Slope
    ax3 = fig.add_subplot(144, projection='polar')  # Aspect 
    ax4 = fig.add_subplot(143)  # NDVI
    axes = [ax1, ax2, ax3, ax4]

    for i, feat in enumerate(features):
        ax = axes[i]
        
        data1 = df[df['period'] == periods[0]][feat].dropna()
        data2 = df[df['period'] == periods[1]][feat].dropna()
        
        # Statistical Test (K-S test)
        stat, p_val = stats.ks_2samp(data1, data2)

        if feat != 'aspect':
            for period in periods:
                sns.kdeplot(
                    data=df[df['period'] == period],
                    x=feat,
                    fill=True,
                    alpha=0.4,
                    linewidth=1.2,
                    color=colors[period],
                    label=period,
                    ax=ax
                )
            
            ax.set_ylabel('Density')
            label_suffix = f" ({units[i]})" if units[i] else ""
            ax.set_xlabel(f'{feat.capitalize()}{label_suffix}')
            sns.despine(ax=ax)

            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
            ax.tick_params(axis='both', which='major', width=1.4, length=6)
            ax.tick_params(axis='both', which='minor', width=1.2, length=3)

            if feat.upper() == 'NDVI':
                ax.set_xlabel('NDVI')
                ax.set_xlim(-0.1, 1.05)
                ax.set_xticks([0, 0.5, 1])

            ax.set_title('')

        else:
            rad1 = np.deg2rad(data1)
            rad2 = np.deg2rad(data2)
            
            bins = np.linspace(0, 2 * np.pi, 24)  
            for data, period in zip([rad1, rad2], periods):
                counts, bin_edges = np.histogram(data, bins=bins, density=True)
                ax.fill(bin_edges[:-1], counts, color=colors[period], alpha=0.4, label=period)
                ax.plot(bin_edges[:-1], counts, color=colors[period], linewidth=0.8)
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_thetagrids(
                np.arange(0, 360, 45),
                labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            )
            ax.set_yticklabels([])

            ax.spines['polar'].set_linewidth(1.4)
            ax.tick_params(axis='both', which='major', width=1.4, length=6)
            ax.tick_params(axis='both', which='minor', width=1.2, length=3)

            ax.set_title('')

            ax.text(
                aspect_label_pos[0], aspect_label_pos[1], 'Aspect',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=28
            )

        if p_val < 0.001:
            sig_text = "p < 0.001"
        else:
            sig_text = f"p = {p_val:.3f}"
            
        if feat == 'NDVI':
            ax.text(
                0.60, 0.95, sig_text,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=26, fontweight='bold'
            )

        elif feat != 'aspect':
            ax.text(
                0.98, 0.95, sig_text,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=26, fontweight='bold'
            )
        else:
            ax.text(
                aspect_p_pos[0], aspect_p_pos[1], sig_text,
                transform=ax.transAxes,
                ha='left', va='center',
                fontsize=26, fontweight='bold'
            )
        
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    pos4 = ax4.get_position()
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, pos4.y0, pos4.width, pos4.height])

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=2,
        frameon=False,
        fontsize=30,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()


# --- 3. Execution ---
output_img = r'H:/Himalaya/figure/Landslide_Features_Migration.png'
os.makedirs(os.path.dirname(output_img), exist_ok=True)

plot_migration_analysis(
    df,
    output_img,
    aspect_label_pos=(0.5, -0.15),   
    aspect_p_pos=(0.75, 0.98)       
)

#--- 4. Calculate statistics and merge output into a single CSV ---
def degree_to_direction(deg):
    """
    Convert angles to 8 directional categories
    N, NE, E, SE, S, SW, W, NW
    """
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    deg = deg % 360
    idx = int(((deg + 22.5) % 360) // 45)
    return directions[idx]


def calculate_combined_statistics_by_period(df, linear_features, circular_feature, output_csv):
    results = []
    periods = ["2000–2019", "2020–2024"]

    # 1) Continuous variables: mean and 95% CI
    for feat in linear_features:
        for period in periods:
            data = df[df["period"] == period][feat].dropna()
            n = len(data)

            if n > 0:
                mean_val = data.mean()
            else:
                mean_val = np.nan

            if n > 1:
                sem = stats.sem(data)
                ci_low, ci_high = stats.t.interval(
                    confidence=0.95,
                    df=n - 1,
                    loc=mean_val,
                    scale=sem
                )
            else:
                ci_low, ci_high = np.nan, np.nan

            results.append({
                "feature": feat,
                "period": period,
                "stat_type": "linear",
                "n": n,
                "mean": mean_val,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "circular_mean_deg": np.nan,
                "resultant_length_r": np.nan,
                "dominant_direction": np.nan
            })

    # 2) Circular variables: circular mean / resultant length / dominant direction of aspect
    for period in periods:
        data = df[df["period"] == period][circular_feature].dropna()
        data = np.mod(data, 360)   # normalize to [0, 360)
        n = len(data)

        if n > 0:
            rad = np.deg2rad(data)

            # circular mean
            circular_mean_rad = stats.circmean(rad, high=2 * np.pi, low=0)
            circular_mean_deg = np.mod(np.rad2deg(circular_mean_rad), 360)

            # mean resultant length r
            C = np.mean(np.cos(rad))
            S = np.mean(np.sin(rad))
            resultant_length_r = np.sqrt(C**2 + S**2)

            # dominant direction
            dominant_direction = degree_to_direction(circular_mean_deg)
        else:
            circular_mean_deg = np.nan
            resultant_length_r = np.nan
            dominant_direction = np.nan

        results.append({
            "feature": circular_feature,
            "period": period,
            "stat_type": "circular",
            "n": n,
            "mean": np.nan,
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "circular_mean_deg": circular_mean_deg,
            "resultant_length_r": resultant_length_r,
            "dominant_direction": dominant_direction
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result_df


# --- 5. Execution and Export of Combined CSV ---
output_summary_csv = r'H:/Himalaya/figure/Landslide_Features_Summary_Combined.csv'
os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)

summary_df = calculate_combined_statistics_by_period(
    df,
    linear_features=['elevation', 'slope', 'NDVI'],
    circular_feature='aspect',
    output_csv=output_summary_csv
)

print(summary_df)



