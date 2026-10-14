"""Run QuickReduct and GBNRS on spambase dataset only."""

import sys
sys.path.insert(0, '~/Documents/2026/Core base-NRS/code')

import csv
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines.gbnrs import gbnrs_reduction
from baselines.quick_reduct_nrs import quick_reduct_nrs
from experiments.datasets import BENCHMARK_DATASETS

RANDOM_STATE = 42
N_SPLITS = 5
DELTA_DEFAULT = 0.1

# Find spambase dataset
spambase_info = None
for info in BENCHMARK_DATASETS:
    if info.key == "spambase":
        spambase_info = info
        break

if not spambase_info:
    print("Spambase dataset not found!")
    sys.exit(1)

print(f"Running spambase: {spambase_info.display_name}")

X, y = spambase_info.loader()
print(f"X shape: {X.shape}, y shape: {y.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(X, y))

results = []

for algo_name, algo_func, algo_label in [
    ("quick_reduct", quick_reduct_nrs, "QuickReduct"),
    ("gbnrs", lambda X, y, delta: gbnrs_reduction(X, y, delta, purity_threshold=0.9), "GBNRS"),
]:
    fold_reduct = []
    fold_runtime = []
    fold_knn = []
    fold_svm = []
    
    for train_idx, test_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        t0 = time.perf_counter()
        reduct = algo_func(X_tr, y_tr, DELTA_DEFAULT)
        rt = time.perf_counter() - t0
        
        fold_reduct.append(len(reduct))
        fold_runtime.append(rt)
        
        # Test scores
        if reduct:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[:, reduct])
            X_te_s = scaler.transform(X_te[:, reduct])
            knn = KNeighborsClassifier(n_neighbors=3).fit(X_tr_s, y_tr).score(X_te_s, y_te)
            svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_tr_s, y_tr).score(X_te_s, y_te)
        else:
            knn, svm = 0.0, 0.0
        
        fold_knn.append(knn)
        fold_svm.append(svm)
    
    results.append({
        "dataset": "spambase",
        "dataset_name": spambase_info.display_name,
        "n_samples": X.shape[0],
        "n_attrs": X.shape[1],
        "algorithm": algo_name,
        "algorithm_label": algo_label,
        "n_reduct_mean": float(np.mean(fold_reduct)),
        "n_reduct_std": float(np.std(fold_reduct)),
        "runtime_mean": float(np.mean(fold_runtime)),
        "runtime_std": float(np.std(fold_runtime)),
        "knn_mean": float(np.mean(fold_knn)),
        "knn_std": float(np.std(fold_knn)),
        "svm_mean": float(np.mean(fold_svm)),
        "svm_std": float(np.std(fold_svm)),
    })
    
    print(f"{algo_label}: |R|={np.mean(fold_reduct):.1f}, runtime={np.mean(fold_runtime):.3f}s")

# Append to existing CSV
results_dir = Path(__file__).resolve().parents[2] / "results"
new_file = results_dir / "benchmark_new_baselines_20260526_073243.csv"

with new_file.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    for r in results:
        w.writerow(r)

print(f"\nAppended {len(results)} rows to {new_file}")
