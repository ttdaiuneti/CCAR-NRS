import numpy as np
import pytest

from baselines.fast_nrs_external import core_init_pairs, fhara_reduct, fnc_reduct
from baselines.greedy_core_init import core_init_greedy
from baselines.quick_reduct_nrs import quick_reduct_nrs
from ccar.core import compute_core
from ccar.nrs import dependency_degree, normalize_features


def _table(seed, grid):
    rng = np.random.default_rng(seed)
    n, d = int(rng.integers(30, 200)), int(rng.integers(2, 12))
    X = rng.random((n, d))
    if grid:
        X = np.round(X * 10) / 10  # many distances fall exactly on delta
    y = rng.integers(0, int(rng.integers(2, 5)), n)
    return X, y


@pytest.mark.parametrize("fn", [fnc_reduct, fhara_reduct])
@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2, 0.3])
def test_same_selections_as_quickreduct(fn, grid, delta):
    for seed in range(40):
        X, y = _table(seed, grid)
        R, meta = fn(X, y, delta)
        assert R == quick_reduct_nrs(X, y, delta)
        assert 0 <= meta["stop_prefix"] <= len(R)
        Xn = normalize_features(X)
        assert dependency_degree(Xn, y, np.array(R, int), delta) >= compute_core(Xn, y, delta)[1] - 1e-9


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2, 0.3])
def test_pair_filter_keeps_core_init(grid, delta):
    nonempty = 0
    for seed in range(60):
        X, y = _table(seed, grid)
        R, meta = core_init_pairs(X, y, delta)
        assert R == core_init_greedy(X, y, delta)
        nonempty += meta["n_pairs"] is not None
    assert nonempty > 0
