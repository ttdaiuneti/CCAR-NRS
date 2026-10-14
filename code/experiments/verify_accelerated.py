"""Check that the accelerated greedy loops return the v7 reducts, fold by fold.

Usage: python -m experiments.verify_accelerated [--ref ../results_v7] [dataset ...]
Reads the stored reduct_set of every (dataset, algorithm, fold) from the
reference run and re-runs the current reducers on the same stratified folds.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.datasets import BENCHMARK_DATASETS
from experiments.run_benchmark import N_SPLITS, RANDOM_STATE, REDUCERS

ALGOS = ["quick_reduct", "core_init_greedy", "wnrs", "nicp", "ccar_h", "ccar_exact"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[2] / "results_v7"))
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("datasets", nargs="*")
    a = ap.parse_args()
    ref = Path(a.ref)
    infos = [d for d in BENCHMARK_DATASETS if not a.datasets or d.key in a.datasets]
    bad = 0
    for info in infos:
        stored: dict[tuple[str, int], tuple[set[int], float]] = {}
        with open(ref / f"{info.key}_perfold.csv") as f:
            for r in csv.DictReader(f):
                rs = {int(v) for v in r["reduct_set"].split(";") if v != ""}
                stored[(r["algorithm"], int(r["fold"]))] = (rs, float(r["time_reduct_sec"]))
        X, y = info.loader()
        folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y))
        for algo in ALGOS:
            t_new = t_old = 0.0
            ok = True
            for fi, (tr, _) in enumerate(folds, start=1):
                t0 = time.perf_counter()
                out = REDUCERS[algo](X[tr], y[tr], a.delta)
                t_new += time.perf_counter() - t0
                red = out.reduct_indices if hasattr(out, "reduct_indices") else out
                ref_set, t_ref = stored[(algo, fi)]
                t_old += t_ref
                if set(int(v) for v in red) != ref_set:
                    ok = False
                    bad += 1
            print(f"{info.key:12s} {algo:17s} same={ok!s:5s} t_v7={t_old:8.3f}s t_now={t_new:8.3f}s ratio={t_old / max(t_new, 1e-9):5.2f}", flush=True)
    print("MISMATCHES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
