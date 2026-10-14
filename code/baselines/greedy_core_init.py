"""Core-initialized greedy baseline (not Core-Centric)."""

from __future__ import annotations

import numpy as np

from ccar.core import compute_core
from ccar.nrs import (
    active_rows,
    attribute_significance,
    dependency_degree,
    gain_counts,
    normalize_features,
    rows_in_pos,
)


def core_init_greedy(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    normalize: bool = True,
    accelerated: bool = True,
) -> list[int]:
    """
    B <- CORE; repeatedly add argmax sig(a|B) until gamma(B)=gamma(C).
    """
    X_work = normalize_features(X) if normalize else X.astype(float)
    y = np.asarray(y)
    n_attr = X_work.shape[1]
    all_idx = np.arange(n_attr)

    core, gamma_c = compute_core(X_work, y, delta)
    selected = list(core)
    sel_idx = np.array(selected, dtype=int) if selected else np.array([], dtype=int)

    if accelerated:
        # Objects in POS_B are never re-tested (Lemma 3.2) and gamma(B) is
        # computed once per step, not once per candidate.
        n = X_work.shape[0]
        active = active_rows(X_work, y, sel_idx, delta)
        while (n - active.size) / n < gamma_c - 1e-12:
            remaining = [a for a in range(n_attr) if a not in selected]
            if not remaining:
                break
            counts = gain_counts(X_work, y, sel_idx, remaining, delta, active)
            selected.append(remaining[int(np.argmax(counts))])
            sel_idx = np.array(selected, dtype=int)
            active = active[~rows_in_pos(X_work, y, active, sel_idx, delta)]
        return sorted(selected)

    while dependency_degree(X_work, y, sel_idx, delta) < gamma_c - 1e-12:
        remaining = [a for a in range(n_attr) if a not in selected]
        if not remaining:
            break
        best_a = max(
            remaining,
            key=lambda a: attribute_significance(X_work, y, sel_idx, a, delta),
        )
        selected.append(best_a)
        sel_idx = np.array(selected, dtype=int)

    return sorted(selected)
