"""Published fast forward reducers for NRS, on the shared max-metric definition.

FNC    Peng et al., Knowl.-Based Syst. 252 (2022) 109394: sorted window on the first
       attribute, then stored neighbour pairs filtered as attributes are added.
FHARA  Liu et al., Inf. Sci. 271 (2014) 65-81: F-POS buckets by distance to the minimum
       point, so that neighbours of an object lie in adjacent buckets only.

Adapted from research-commons exact-nrs-accelerators v1 (Euclidean, verified 28/28 against
the official FNC code). Both papers state their results for a general metric. Selections
follow QuickReduct: first maximum of the POS gain, continue until gamma(R) = gamma(C).
Meta `stop_prefix` is the reduct length under the published rule (stop at zero gain).
FNC stores cross-class pairs only, which favours it over the published all-neighbour store.

core_init_pairs: the Core-init ablation run on conflicting pairs under the core.
"""
from __future__ import annotations

import numpy as np

from baselines.greedy_core_init import core_init_greedy
from ccar.core import compute_core
from ccar.nrs import normalize_features

PAIR_LIMIT = 300_000_000
CHUNK = 4_000_000
WIN = 1e-9
EXIT_CHUNK = 100_000


class MemOut(Exception):
    pass


def _expand(lo, hi):
    cnt = hi - lo
    rows = np.repeat(np.arange(len(lo)), cnt)
    start = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
    return rows, np.arange(int(cnt.sum())) + start


def _chunks(cnt, limit=CHUNK):
    cs = np.concatenate(([0], np.cumsum(cnt)))
    i, n = 0, len(cnt)
    while i < n:
        j = max(i + 1, int(np.searchsorted(cs, cs[i] + limit, "right")) - 1)
        yield i, j
        i = j


def _first_gains(X, y, delta):
    """POS gain of each single attribute, by a sorted window with exact boundaries."""
    n, d = X.shape
    classes = np.unique(y)
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    gains = np.zeros(d, dtype=np.int64)
    for a in range(d):
        o = np.argsort(X[:, a], kind="stable")
        v = X[o, a]
        P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
        lo = np.searchsorted(v, v - delta - WIN, "left")
        hi = np.searchsorted(v, v + delta + WIN, "right")
        while True:
            m = np.abs(v - v[lo]) > delta
            if not m.any():
                break
            lo[m] += 1
        while True:
            m = np.abs(v[hi - 1] - v) > delta
            if not m.any():
                break
            hi[m] -= 1
        own = np.searchsorted(classes, y[o])
        gains[a] = int(((hi - lo) - (P[hi, own] - P[lo, own]) == 0).sum())
    return gains


def fnc_reduct(X, y, delta):
    X = normalize_features(X)
    y = np.asarray(y)
    n, d = X.shape
    classes = np.unique(y)
    gains = _first_gains(X, y, delta)
    a1 = int(np.argmax(gains))
    stop_prefix = None if gains[a1] > 0 else 0
    Q, E = [], []
    tot = 0
    for k in classes:
        q = np.flatnonzero(y == k)
        e = np.flatnonzero(y != k)
        e = e[np.argsort(X[e, a1], kind="stable")]
        ve = X[e, a1]
        lo = np.searchsorted(ve, X[q, a1] - delta - WIN, "left")
        hi = np.searchsorted(ve, X[q, a1] + delta + WIN, "right")
        tot += int((hi - lo).sum())
        if tot > PAIR_LIMIT:
            raise MemOut(f"{tot} pairs")
        for i, j in _chunks(hi - lo):
            r, p = _expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]
            m = np.abs(X[qq, a1] - X[ee, a1]) <= delta
            Q.append(qq[m].astype(np.int32))
            E.append(ee[m].astype(np.int32))
    Q = np.concatenate(Q) if Q else np.empty(0, np.int32)
    E = np.concatenate(E) if E else np.empty(0, np.int32)

    n_stored = len(Q)
    # blocked objects under all of C, from the same pairs: the stopping target
    Qc, Ec = Q, E
    for a in range(d):
        if a != a1 and len(Qc):
            m = np.abs(X[Qc, a] - X[Ec, a]) <= delta
            Qc, Ec = Qc[m], Ec[m]
    mark = np.zeros(n, bool)
    mark[Qc] = True
    target = int(mark.sum())
    del Qc, Ec
    if target == n:
        return [], {"stop_prefix": 0, "n_pairs": n_stored}

    R = [a1]
    rem = [a for a in range(d) if a != a1]
    mark[:] = False
    mark[Q] = True
    n_act = int(mark.sum())
    while rem and n_act > target:
        best_gain, best = -1, None
        for a in rem:
            mark[:] = False
            mark[Q[np.abs(X[Q, a] - X[E, a]) <= delta]] = True
            g = n_act - int(mark.sum())
            if g > best_gain:
                best_gain, best = g, a
        if best_gain == 0 and stop_prefix is None:
            stop_prefix = len(R)
        R.append(best)
        rem.remove(best)
        m = np.abs(X[Q, best] - X[E, best]) <= delta
        Q, E = Q[m], E[m]
        mark[:] = False
        mark[Q] = True
        n_act = int(mark.sum())
    return R, {"stop_prefix": len(R) if stop_prefix is None else stop_prefix, "n_pairs": n_stored}


def _fpos_blocked(X, y, classes, B, D0, act, delta):
    """F-POS on `act` (Alg. 1-2): enemies come from the remaining objects Q only, in the window
    of adjacent buckets around D0; an object found blocked is not tested again."""
    mark = np.zeros(X.shape[0], bool)
    for k in classes:
        q = act[y[act] == k]
        if not len(q):
            continue
        e = act[y[act] != k]
        e = e[np.argsort(D0[e], kind="stable")]
        de = D0[e]
        lo = np.searchsorted(de, D0[q] - delta - WIN, "left")
        hi = np.searchsorted(de, D0[q] + delta + WIN, "right")
        for i, j in _chunks(hi - lo, EXIT_CHUNK):
            r, p = _expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]
            m = ~mark[qq]
            qq, ee = qq[m], ee[m]
            for c in B:
                m = np.abs(X[qq, c] - X[ee, c]) <= delta
                qq, ee = qq[m], ee[m]
                if not len(qq):
                    break
            mark[qq] = True
    return mark[act]


def fhara_reduct(X, y, delta):
    X = normalize_features(X)
    y = np.asarray(y)
    n, d = X.shape
    classes = np.unique(y)
    Xr = X - X.min(0)
    full = list(range(d))
    target = int(_fpos_blocked(X, y, classes, full, Xr.max(1), np.arange(n), delta).sum())
    R, rem, act = [], full.copy(), np.arange(n)
    D_R = np.zeros(n)
    stop_prefix = None
    while rem and len(act) > target:
        best_gain, best, best_mask = -1, None, None
        for a in rem:
            mask = _fpos_blocked(X, y, classes, R + [a], np.maximum(D_R, Xr[:, a]), act, delta)
            g = len(act) - int(mask.sum())
            if g > best_gain:
                best_gain, best, best_mask = g, a, mask
        if best_gain == 0 and stop_prefix is None:
            stop_prefix = len(R)
        R.append(best)
        rem.remove(best)
        D_R = np.maximum(D_R, Xr[:, best])
        act = act[best_mask]
    return R, {"stop_prefix": len(R) if stop_prefix is None else stop_prefix}


def core_init_pairs(X, y, delta):
    """Same selections as Core-init; candidates are scored on conflicting pairs under the core."""
    Xn = normalize_features(X)
    y = np.asarray(y)
    n, d = Xn.shape
    core, gamma_c = compute_core(Xn, y, delta)
    R = [int(a) for a in core]
    if not R:
        return core_init_greedy(X, y, delta), {"n_pairs": None}
    a1 = R[0]
    Q, E = [], []
    for k in np.unique(y):
        q = np.flatnonzero(y == k)
        e = np.flatnonzero(y != k)
        e = e[np.argsort(Xn[e, a1], kind="stable")]
        ve = Xn[e, a1]
        lo = np.searchsorted(ve, Xn[q, a1] - delta - WIN, "left")
        hi = np.searchsorted(ve, Xn[q, a1] + delta + WIN, "right")
        for i, j in _chunks(hi - lo):
            r, p = _expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]
            for c in R:
                m = np.abs(Xn[qq, c] - Xn[ee, c]) <= delta
                qq, ee = qq[m], ee[m]
            Q.append(qq.astype(np.int32))
            E.append(ee.astype(np.int32))
    Q = np.concatenate(Q) if Q else np.empty(0, np.int32)
    E = np.concatenate(E) if E else np.empty(0, np.int32)
    n_pairs = len(Q)
    rem = [a for a in range(d) if a not in R]
    mark = np.zeros(n, bool)
    mark[Q] = True
    n_act = int(mark.sum())
    while (n - n_act) / n < gamma_c - 1e-12 and rem:
        best_g, best = -1, None
        for a in rem:
            mark[:] = False
            mark[Q[np.abs(Xn[Q, a] - Xn[E, a]) <= delta]] = True
            g = n_act - int(mark.sum())
            if g > best_g:
                best_g, best = g, a
        R.append(best)
        rem.remove(best)
        m = np.abs(Xn[Q, best] - Xn[E, best]) <= delta
        Q, E = Q[m], E[m]
        mark[:] = False
        mark[Q] = True
        n_act = int(mark.sum())
    return sorted(R), {"n_pairs": n_pairs}
