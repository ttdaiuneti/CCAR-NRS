"""One backward pass added to QuickReduct and Core-init, on the benchmark folds and Covertype.

The pass tries the attributes in reverse order of selection and keeps every attribute whose
removal lowers the dependency. Writes results_v11/pruned_folds.csv, one row per table, fold and method.
"""
from __future__ import annotations

import csv
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from baselines.greedy_core_init import core_init_greedy
from baselines.quick_reduct_nrs import quick_reduct_nrs
from ccar.nrs import dependency_degree, normalize_features
from experiments.datasets import BENCHMARK_DATASETS
from experiments.run_scalability import subsample

DELTA = 0.1
OUT = ROOT / "results_v11" / "pruned_folds.csv"
FIELDS = ["table", "fold", "n", "method", "n_before", "n_after", "prune_time", "minimal_before", "valid"]
METHODS = {"QuickReduct": quick_reduct_nrs, "Core-init": core_init_greedy}


def gamma(Xn, y, attrs):
    return dependency_degree(Xn, y, np.array(attrs, dtype=int), DELTA) if attrs else 0.0


def run(writer, table, fold, X, y):
    Xn = normalize_features(X)
    target = gamma(Xn, y, list(range(X.shape[1])))
    for name, fn in METHODS.items():
        before = [int(a) for a in fn(X, y, DELTA)]
        kept = list(before)
        t0 = time.perf_counter()
        for a in reversed(before):
            rest = [b for b in kept if b != a]
            if gamma(Xn, y, rest) >= target - 1e-9:
                kept = rest
        writer.writerow(dict(table=table, fold=fold, n=X.shape[0], method=name, n_before=len(before), n_after=len(kept),
                             prune_time=time.perf_counter() - t0, minimal_before=len(kept) == len(before),
                             valid=gamma(Xn, y, kept) >= target - 1e-9))


def main():
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for info in BENCHMARK_DATASETS:
            X, y = info.loader()
            for fi, (tr, _) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y), start=1):
                run(writer, info.key, fi, X[tr], y[tr])
            fh.flush()
        b = fetch_covtype()
        Xa, ya = b.data.astype(float), (b.target - 1).astype(int)
        for n in (1000, 2000, 5000):
            X, y = subsample(Xa, ya, n, seed=42)
            run(writer, f"covertype_{n}", 42, X, y)
            fh.flush()


if __name__ == "__main__":
    main()
