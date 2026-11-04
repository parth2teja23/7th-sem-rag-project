from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 42) -> float:
    """p-value that mean(b) > mean(a) by chance (fraction of resampled mean diffs <= 0)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if d.size == 0 or not d.any():
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return float((means <= 0).mean())


def paired_ttest(a: list[float], b: list[float]) -> float:
    return float(stats.ttest_rel(b, a).pvalue)
