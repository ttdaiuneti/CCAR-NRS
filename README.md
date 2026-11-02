# CCAR-NRS reproducibility package

Code and results for "Core-Centric Attribute Reduction for Neighborhood Rough Sets: Minimum-Size Reducts from the Core and
Residual Conflicting Pairs" (Tran, Nguyen, Nguyen).

## Layout
- `code/` — `ccar/` (CCAR-E, CCAR-H, surrogate seed, core and conflicting-pair routines), `baselines/` (QuickReduct, Core-init,
  NIP-NRS, W-greedy, GBNRS), `experiments/` (benchmark and analysis runners), `scripts/` (table and number generators), `tests/`.
- `results_v9/` — fold-level results (`*_perfold.csv`: one row per dataset, method and fold with the reduct, runtimes,
  accuracies and validity checks), per-dataset summaries (`*_folds.csv`), `benchmark_20datasets.csv`, Covertype runs
  (`scalability_v9.csv`), synthetic wide tables (`wide_table_stress_v9.csv`), `manuscript_numbers_v9.txt` (every statistic in the paper),
  `RUN_INFO.txt` (hardware, versions, command).
- `results_v11/` — comparison with FNC and FHARA and the pair-filtered Core-init variant: UCI folds, Covertype subsamples, wide tables, peak memory, the generated table and macros, and `pruned_folds.csv` (one backward pass on QuickReduct and Core-init).
- `results_v12/` — GBNRS as in Algorithm 1 of Xia et al., on the benchmark folds (`experiments/run_gbnrs_xia.py`). This column replaces the GBNRS column of `results_v9` in the manuscript; `baselines/gbnrs.py` is the earlier greedy search and not that algorithm.
- `results_v7_unchanged/` — radius sweep and fold stability of the core. These involve only CCAR-E/CCAR-H, whose code did not change
  between v7 and v8, so they were not re-run.

## What changed from v7
Every greedy method (QuickReduct, Core-init, WNRS, NIP-NRS and the significance tie-break of CCAR-H) uses positive-region
sample deletion and caches gamma(B). The maximum distance over the selected set is computed once per block of objects and reused for
every candidate. CCAR-H prunes on the residual family when the core is non-empty (equivalent to the gamma test).
Reducts are identical to v7 on all 630 dataset-method-fold cells; only the times change.

## Reproduce
```
cd code
pip install -r requirements.txt
python3 -m experiments.run_benchmark --jobs 1 --results-dir ../results_v9     # 18 datasets, about 30 min on an Apple M4 Pro
python3 -m experiments.run_scalability                                        # Covertype, about 50 min
python3 -m experiments.run_wide_table_stress
python3 scripts/manuscript_tables.py <out_dir>   # every table and quoted number
python3 scripts/manuscript_numbers.py ../results_v9/benchmark_20datasets.csv
python3 -m experiments.run_external_baselines                                # about 45 min
python3 experiments/external_report.py
```
Timings depend on the machine; the reducts and the validity and minimality counts do not.

## Licence
MIT (see `LICENSE`).
