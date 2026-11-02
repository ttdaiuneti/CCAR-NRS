"""GBNRS as in Algorithm 1 of Xia et al., on the benchmark folds. Writes results_v12/gbnrs_xia_folds.csv."""
from __future__ import annotations

import csv
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from baselines.gbnrs_xia import gbnrs_xia_runs
from ccar.nrs import dependency_degree, normalize_features
from experiments.datasets import BENCHMARK_DATASETS
from experiments.run_benchmark import fold_test_scores

DELTA = 0.1
RUNS = 10
OUT = ROOT / "results_v12" / "gbnrs_xia_folds.csv"
FIELDS = ["dataset", "fold", "n_train", "reduct_set", "n_reduct", "n_mean_runs", "n_min_runs", "n_max_runs",
          "time_ten_runs_sec", "knn", "svm", "gamma_C", "gamma_R", "valid", "minimal"]


def gamma(Xn, y, attrs):
    return dependency_degree(Xn, y, np.array(attrs, dtype=int), DELTA) if len(attrs) else 0.0


def main():
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for info in BENCHMARK_DATASETS:
            X, y = info.loader()
            for fi, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y), start=1):
                Xn = normalize_features(X[tr])
                t0 = time.perf_counter()
                reducts = gbnrs_xia_runs(Xn, y[tr], RUNS)
                elapsed = time.perf_counter() - t0
                R = min(reducts, key=len)
                sizes = [len(r) for r in reducts]
                target, g = gamma(Xn, y[tr], list(range(X.shape[1]))), gamma(Xn, y[tr], R)
                valid = g >= target - 1e-9
                minimal = valid and all(gamma(Xn, y[tr], [b for b in R if b != a]) < target - 1e-9 for a in R)
                sc = fold_test_scores(X[tr], y[tr], X[te], y[te], R)
                writer.writerow(dict(dataset=info.key, fold=fi, n_train=len(tr), reduct_set=";".join(map(str, R)),
                                     n_reduct=len(R), n_mean_runs=np.mean(sizes), n_min_runs=min(sizes), n_max_runs=max(sizes),
                                     time_ten_runs_sec=elapsed, knn=sc["knn"], svm=sc["svm"], gamma_C=target, gamma_R=g,
                                     valid=valid, minimal=minimal))
                fh.flush()
                print(info.key, fi, len(R), f"{elapsed:.2f}s", valid, flush=True)


if __name__ == "__main__":
    main()
