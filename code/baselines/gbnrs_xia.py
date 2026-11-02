"""GBNRS attribute reduction following Algorithm 1 of Xia et al. (IEEE TKDE 34(3), 2022).

Stage 1: 2-means splitting until every ball is pure (threshold 1), k-means started at the
ball centers to fine-tune them, impure balls discarded. Stage 2: drop one attribute, assign
each object to the nearest center, and delete the attribute if every ball stays pure.

The paper leaves three points open, and we resolve them as follows.
- Step 2 removes objects "not contained in any ball". Taken as objects of balls that cannot
  be split to purity (duplicates with different labels). Objects beyond the mean radius are
  kept in the partition, because the purity test of step 7 assigns every object to a center.
- Step 6 is run on the objects of the balls kept in step 4.
- "Best of ten runs" is the smallest reduct over ten seeds, chosen on the training fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _pure(labels: np.ndarray) -> bool:
    return labels.size > 0 and bool((labels == labels[0]).all())


def _generate_balls(X: np.ndarray, y: np.ndarray, seed: int, start: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """2-means splitting to purity 1 (GBC). Unsplittable impure groups are dropped as outliers."""
    balls, stack, step = [], list(start) if start else [np.arange(len(X))], 0
    while stack:
        idx = stack.pop()
        if _pure(y[idx]):
            balls.append(idx)
            continue
        if np.ptp(X[idx], axis=0).max() < 1e-14:
            continue
        step += 1
        lab = KMeans(n_clusters=2, n_init=1, random_state=seed + step).fit_predict(X[idx])
        stack.extend([idx[lab == 0], idx[lab == 1]])
    return balls


def _fine_tune(X: np.ndarray, y: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K-means from the given centers; returns the pure-ball centers and their objects (step 3-4)."""
    centers = np.unique(centers, axis=0)
    lab = KMeans(n_clusters=len(centers), init=centers, n_init=1).fit_predict(X)
    keep_c, keep_i = [], []
    for j in range(len(centers)):
        idx = np.flatnonzero(lab == j)
        if idx.size and _pure(y[idx]):
            keep_c.append(X[idx].mean(axis=0))
            keep_i.append(idx)
    return np.array(keep_c), np.concatenate(keep_i) if keep_i else np.array([], dtype=int)


def _partition_pure(X: np.ndarray, y: np.ndarray, centers: np.ndarray) -> bool:
    """Assign each object to its nearest center and test that every ball is pure (steps 6-8)."""
    near = np.empty(len(X), dtype=int)
    for s in range(0, len(X), 512):
        near[s:s + 512] = ((X[s:s + 512, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return all(_pure(y[near == j]) for j in np.unique(near))


def _one_run(X: np.ndarray, y: np.ndarray, seed: int) -> list[int]:
    kept = list(range(X.shape[1]))
    balls = _generate_balls(X, y, seed)
    if not balls:
        return kept
    centers, objs = _fine_tune(X, y, np.array([X[b].mean(axis=0) for b in balls]))
    a_pos = 0
    while a_pos < len(kept) and len(kept) > 1 and len(centers):
        trial = kept[:a_pos] + kept[a_pos + 1:]
        cols = [kept.index(a) for a in trial]
        if _partition_pure(X[objs][:, trial], y[objs], centers[:, cols]):
            # Step 11: k-means from the current centers, then split until every ball is pure.
            kept = trial
            Xk, start = X[:, kept], np.unique(centers[:, cols], axis=0)
            lab = KMeans(n_clusters=len(start), init=start, n_init=1).fit_predict(Xk)
            groups = [np.flatnonzero(lab == j) for j in range(len(start)) if (lab == j).any()]
            step_balls = _generate_balls(Xk, y, seed + 1000 * len(kept), groups)
            if not step_balls:
                break
            centers, objs = _fine_tune(Xk, y, np.array([Xk[b].mean(axis=0) for b in step_balls]))
        else:
            a_pos += 1
    return kept


def gbnrs_xia_runs(X: np.ndarray, y: np.ndarray, runs: int = 10, seed: int = 42) -> list[list[int]]:
    """Reducts of the independent runs, in seed order. X must already be scaled."""
    return [_one_run(X, y, seed + 7919 * r) for r in range(runs)]


def gbnrs_xia_reduction(X: np.ndarray, y: np.ndarray, runs: int = 10, seed: int = 42) -> list[int]:
    """Smallest reduct over the runs; ties go to the earliest run."""
    return min(gbnrs_xia_runs(X, y, runs, seed), key=len)
