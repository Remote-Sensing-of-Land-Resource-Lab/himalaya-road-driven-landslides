import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

input_csv_path = r'H:\Himalaya\RF_susceptibility\features_all.csv'
model_save_path = r'H:\Himalaya\RF_susceptibility\RF_model_all_samples_5foldcv.pkl'

figure_save_dir = r'H:\Himalaya\RF_susceptibility'
os.makedirs(figure_save_dir, exist_ok=True)

# =================1. Data loading and cleaning=================
print(f"Reading data: {input_csv_path} ...")
df = pd.read_csv(input_csv_path)
print(f"Original sample count: {len(df)}")

# =================2. Data cleaning and invalid value filtering =================
print("\n>>> Starting data cleaning and outlier handling...")
initial_count = len(df)

# 2.1 Remove missing values (NaN)
df.dropna(inplace=True)
print(f"  - Remaining after removing missing values: {len(df)}")

# 2.2 Filter NDVI valid values (0 < NDVI < 1)
df = df[(df['NDVI'] > 0) & (df['NDVI'] < 1)]
print(f"  - Remaining after removing NDVI outliers ((0,1) excluded): {len(df)}")

# 2.3 Handle curvature outliers (retain 1% - 99% quantile range)
curvature_cols = ['plan_curv', 'profile_curv']

for col in curvature_cols:
    if col in df.columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"  - {col} retained range [{lower_bound:.4f}, {upper_bound:.4f}], remaining samples: {len(df)}")
    else:
        print(f"  Warning: column {col} was not found in the data; skipping this filter.")

cleaned_count = len(df)
loss_rate = (initial_count - cleaned_count) / initial_count * 100
print(f">>> Data cleaning completed. Removed {initial_count - cleaned_count} abnormal samples (removal rate: {loss_rate:.2f}%)")
print(f"Final sample count used for modeling: {cleaned_count}")

# =================3. Feature selection=================
exclude_cols = [
    'label', 'year', 'lon', 'lat',
    'aspect', 'aspect_rad',
    'lithology',
    'landcover',
    'road_age', 'road_density', 'R20mm', 'Rx1day',
    'SDII', 'evt_date', 'DOY', 'd_days', 'e_mm', 'i_mm_day'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeature count: {len(feature_cols)}")
print("Feature list used:", feature_cols)

# =================4. Model training using all samples=================
X_all = df[feature_cols]
y_all = df['label']

print(f"\nModeling dataset preparation completed:")
print(f"Total sample count: {len(X_all)}")
print(f"Landslide sample proportion: {y_all.mean():.2%}")
print(f"Non-landslide sample proportion: {(1 - y_all.mean()):.2%}")

# =================5. Define the random forest model=================
print("\nStarting to build the random forest model...")

rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    oob_score=False  
)

# =================6. Five-fold cross-validation evaluation=================
print("\nPerforming five-fold cross-validation evaluation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6.1 Compute fold-wise AUC / Accuracy statistics
cv_auc_scores = cross_val_score(
    rf_model, X_all, y_all,
    cv=cv, scoring='roc_auc', n_jobs=-1
)

cv_acc_scores = cross_val_score(
    rf_model, X_all, y_all,
    cv=cv, scoring='accuracy', n_jobs=-1
)

# 6.2 Obtain out-of-fold predicted probabilities for all samples
y_all_pred_prob = cross_val_predict(
    rf_model, X_all, y_all,
    cv=cv, method='predict_proba', n_jobs=-1
)[:, 1]

# 6.3 Plot the ROC curve based on the five-fold cross-validation results and compute the overall AUC
all_fpr, all_tpr, all_thresholds = roc_curve(y_all, y_all_pred_prob)
all_auc_score = roc_auc_score(y_all, y_all_pred_prob)

# 6.4 Find the best threshold from cross-validated predicted probabilities (Youden's J statistic)
youden = all_tpr - all_fpr
best_idx = np.argmax(youden)
best_threshold = all_thresholds[best_idx]

y_all_pred = (y_all_pred_prob >= best_threshold).astype(int)
all_accuracy = accuracy_score(y_all, y_all_pred)
cm_all = confusion_matrix(y_all, y_all_pred)

print("-" * 50)
print("[Core evaluation metrics from five-fold cross-validation]")
print(f"5-fold CV AUC (mean ± std): {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"5-fold CV Acc (mean ± std): {cv_acc_scores.mean():.4f} ± {cv_acc_scores.std():.4f}")
print(f"OOF Overall AUC           : {all_auc_score:.4f}")
print(f"OOF Overall Accuracy      : {all_accuracy:.4f}")
print(f"Best threshold (Youden)   : {best_threshold:.4f}")
print("-" * 50)

print("\nDetailed classification report from five-fold cross-validation:")
print(classification_report(y_all, y_all_pred, target_names=['Non-landslide', 'Landslide']))

print("Confusion matrix from five-fold cross-validation:")
print(cm_all)

# =================7. Train the final model using all samples and save it=================
print("\nStarting to train the final model using all samples...")
rf_model.fit(X_all, y_all)
joblib.dump(rf_model, model_save_path)
print(f"The final unified model has been saved to: {model_save_path}")

# =================8. Plot and save results=================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out'
})

# ===== Layout: ROC on the left, confusion matrix on the right =====
fig = plt.figure(figsize=(7.4, 3.6))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.2, 1],
    wspace=0.38
)

ax_roc = fig.add_subplot(gs[0, 0])
ax_cm = fig.add_subplot(gs[0, 1])

# ===== Left panel: 5-fold CV ROC =====
roc_color = '#4C72B0'
diag_color = '#BFBFBF'

ax_roc.plot(
    all_fpr, all_tpr,
    color=roc_color, lw=1.8,
    label=f'5-fold CV (AUC = {all_auc_score:.3f})'
)
ax_roc.plot([0, 1], [0, 1], color=diag_color, lw=1.0, linestyle='--')

ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1)
ax_roc.set_aspect('equal', adjustable='box')
ax_roc.set_xlabel('False positive rate')
ax_roc.set_ylabel('True positive rate')
ax_roc.set_title('ROC curve', pad=6, fontsize=10)
ax_roc.legend(frameon=False, loc='lower right', fontsize=8, handlelength=2.2)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)

ax_roc.text(
    -0.14, 1.03, 'a',
    transform=ax_roc.transAxes,
    fontsize=11, fontweight='bold',
    va='bottom', ha='left'
)

# ===== Right panel: 5-fold CV confusion matrix =====
im = ax_cm.imshow(cm_all, cmap='Blues', vmin=0, vmax=cm_all.max())

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Non-landslide', 'Landslide'])
ax_cm.set_yticklabels(['Non-landslide', 'Landslide'])
ax_cm.set_xlabel('Predicted label')
ax_cm.set_ylabel('True label')
ax_cm.set_title('5-fold CV', pad=4, fontsize=10)

cm_all_pct = cm_all / cm_all.sum(axis=1, keepdims=True) * 100
threshold_cm = cm_all.max() / 2.0

for i in range(cm_all.shape[0]):
    for j in range(cm_all.shape[1]):
        ax_cm.text(
            j, i, f'{cm_all[i, j]}\n({cm_all_pct[i, j]:.1f}%)',
            ha='center', va='center',
            color='white' if cm_all[i, j] > threshold_cm else '#1A1A1A',
            fontsize=8.5
        )

for spine in ax_cm.spines.values():
    spine.set_linewidth(0.8)

ax_cm.text(
    -0.20, 1.03, 'b',
    transform=ax_cm.transAxes,
    fontsize=11, fontweight='bold',
    va='bottom', ha='left'
)

plt.tight_layout()

fig_save_path = os.path.join(figure_save_dir, 'ROC_Curve_and_ConfusionMatrix_5foldCV.png')
plt.savefig(fig_save_path, dpi=600, bbox_inches='tight')
print(f"\nThe 5-fold CV ROC + confusion matrix figure has been saved to: {fig_save_path}")
plt.close()

print("\nProcessing finished.")