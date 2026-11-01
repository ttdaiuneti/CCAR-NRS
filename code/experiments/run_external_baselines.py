"""FNC and FHARA against CCAR and QuickReduct, timed in one session on one thread.

Same 18 tables, folds and delta as run_benchmark; Covertype sizes and wide tables as in
run_scalability and run_wide_table_stress. Results are written after every row, and finished
rows are skipped on a restart. Writes results_v11/external_{folds,scalability,wide,memory}.csv. Covertype uses five subsamples per size, three at n = 10,000 (FHARA one).
"""
from __future__ import annotations

import csv
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from baselines.fast_nrs_external import core_init_pairs, fhara_reduct, fnc_reduct
from baselines.greedy_core_init import core_init_greedy
from baselines.quick_reduct_nrs import quick_reduct_nrs
from ccar.ccar_exact import ccar_nrs_exact
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.core import compute_core
from ccar.nrs import dependency_degree, normalize_features
from experiments.datasets import BENCHMARK_DATASETS
from experiments.run_scalability import SIZES, subsample
from experiments.run_wide_table_stress import make_table

DELTA = 0.1
WIDE_N, WIDE_INF = 400, 10
WIDE = [(p, 42 + p) for p in (20, 40, 60, 80, 100, 150, 200)]
OUT = ROOT / "results_v11"
SEEDS = (42, 1, 2, 3, 4)
FIELDS = ["table", "fold", "n", "p", "n_core", "method", "runtime", "n_reduct", "reduct_set",
          "valid", "stop_prefix", "valid_published_stop", "n_pairs", "error"]
METHODS = {
    "CCAR-E": lambda X, y: (ccar_nrs_exact(X, y, DELTA).reduct_indices, {}),
    "CCAR-H": lambda X, y: (ccar_nrs_heuristic(X, y, DELTA).reduct_indices, {}),
    "Core-init": lambda X, y: (core_init_greedy(X, y, DELTA), {}),
    "Core-init-P": lambda X, y: core_init_pairs(X, y, DELTA),
    "QuickReduct": lambda X, y: (quick_reduct_nrs(X, y, DELTA), {}),
    "FNC": lambda X, y: fnc_reduct(X, y, DELTA),
    "FHARA": lambda X, y: fhara_reduct(X, y, DELTA),
}


def one(X, y, name, table, fold, n_core, gamma_c, Xn):
    row = dict.fromkeys(FIELDS, "")
    row.update(table=table, fold=fold, n=X.shape[0], p=X.shape[1], n_core=n_core, method=name)
    try:
        t0 = time.perf_counter()
        R, meta = METHODS[name](X, y)
        row["runtime"] = time.perf_counter() - t0
    except MemoryError as exc:
        row["error"] = repr(exc)
        return row
    R = [int(a) for a in R]
    gamma = lambda A: dependency_degree(Xn, y, np.array(A, dtype=int), DELTA) if A else 0.0
    row.update(n_reduct=len(R), reduct_set=";".join(map(str, sorted(R))),
               valid=gamma(R) >= gamma_c - 1e-9)
    row["n_pairs"] = meta.get("n_pairs", "")
    if "stop_prefix" in meta:
        row["stop_prefix"] = meta["stop_prefix"]
        row["valid_published_stop"] = gamma(R[: meta["stop_prefix"]]) >= gamma_c - 1e-9
    return row


class Sink:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if path.exists():
            with path.open() as fh:
                self.done = {(r["table"], r["fold"], r["method"]) for r in csv.DictReader(fh)}
        else:
            with path.open("w", newline="") as fh:
                csv.DictWriter(fh, FIELDS).writeheader()

    def add(self, row):
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writerow(row)
        print(row["table"], row["fold"], row["method"], row["runtime"], row["n_reduct"], row["valid"], flush=True)


def run(sink, table, fold, X, y, names):
    todo = [m for m in names if (table, str(fold), m) not in sink.done]
    if not todo:
        return
    Xn = normalize_features(X)
    core, gamma_c = compute_core(Xn, y, DELTA)
    for m in todo:
        sink.add(one(X, y, m, table, fold, len(core), gamma_c, Xn))


def memory_run(name, n=10000, seed=42):
    """Peak resident memory of one reduction, in a fresh process."""
    path = OUT / "external_memory.csv"
    code = (
        "import resource\n"
        "from experiments.run_external_baselines import *\n"
        "b = fetch_covtype()\n"
        f"X, y = subsample(b.data.astype(float), (b.target - 1).astype(int), {n}, seed={seed})\n"
        "base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        f"R, _ = METHODS[{name!r}](X, y)\n"
        "print(base, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
    )
    base, peak = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "code", capture_output=True,
                                text=True, check=True).stdout.split()
    new = not path.exists()
    with path.open("a") as fh:
        if new:
            fh.write("n,seed,method,rss_before_bytes,rss_peak_bytes\n")
        fh.write(f"{n},{seed},{name},{base},{peak}\n")
    print("memory", name, base, peak, flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    names = list(METHODS)
    bench = Sink(OUT / "external_folds.csv")
    for info in BENCHMARK_DATASETS:
        X, y = info.loader()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fi, (tr, _) in enumerate(skf.split(X, y), start=1):
            run(bench, info.key, fi, X[tr], y[tr], names)
    scal = Sink(OUT / "external_scalability.csv")
    bunch = fetch_covtype()
    Xa, ya = bunch.data.astype(float), (bunch.target - 1).astype(int)
    for n in SIZES:
        for seed in SEEDS[:3] if n == 10000 else SEEDS:
            X, y = subsample(Xa, ya, n, seed=seed)
            names_s = ["CCAR-H", "Core-init", "Core-init-P", "QuickReduct", "FNC"]
            if n < 10000 or seed == 42:
                names_s.append("FHARA")
            run(scal, f"covertype_{n}", seed, X, y, names_s)
    for name in ("CCAR-H", "Core-init-P", "QuickReduct", "FNC"):
        memory_run(name)
    wide = Sink(OUT / "external_wide.csv")
    for p, seed in WIDE:
        X, y = make_table(WIDE_N, p, WIDE_INF, seed)
        run(wide, f"wide_{p}", 0, X, y, ["CCAR-H", "QuickReduct", "FNC", "FHARA"])


if __name__ == "__main__":
    main()
