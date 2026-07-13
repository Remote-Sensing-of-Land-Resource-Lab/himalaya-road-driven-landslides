import numpy as np
from statsmodels.stats.proportion import proportion_confint

# -------------------------
# Validation sample sizes
# -------------------------
n_ls = 1000
n_nls = 1000

# mapped landslide stratum
tp = 872
fp = 128

# mapped non-landslide stratum
fn = 1
tn = 999

# area weights
A_total = 420794.252
A_ls = 1257.921
A_nls = 419536.331

W_ls = A_ls / A_total
W_nls = A_nls / A_total

# -------------------------
# 1) User's accuracy -- landslide
# -------------------------
ua_ls = tp / n_ls

# Wilson 95% CI
ua_ci_low, ua_ci_high = proportion_confint(tp, n_ls, alpha=0.05, method="wilson")

print("User's accuracy -- landslide")
print(f"Estimate: {ua_ls*100:.2f}%")
print(f"95% CI: {ua_ci_low*100:.2f}% - {ua_ci_high*100:.2f}%")

# -------------------------
# 2) Producer's accuracy -- landslide
#    stratified bootstrap
# -------------------------
B = 10000
pa_boot = np.empty(B)

rng = np.random.default_rng(42)

# Original label vector
labels_ls = np.array([1]*tp + [0]*fp)      # mapped landslide stratum
labels_nls = np.array([1]*fn + [0]*tn)     # mapped non-landslide stratum

for b in range(B):
    sample_ls = rng.choice(labels_ls, size=n_ls, replace=True)
    sample_nls = rng.choice(labels_nls, size=n_nls, replace=True)

    p11 = W_ls * sample_ls.mean()
    p21 = W_nls * sample_nls.mean()

    pa_boot[b] = p11 / (p11 + p21)

pa_est = (W_ls * (tp/n_ls)) / ((W_ls * (tp/n_ls)) + (W_nls * (fn/n_nls)))
pa_ci_low, pa_ci_high = np.percentile(pa_boot, [2.5, 97.5])

print("\nProducer's accuracy -- landslide")
print(f"Estimate: {pa_est*100:.2f}%")
print(f"95% CI: {pa_ci_low*100:.2f}% - {pa_ci_high*100:.2f}%")



