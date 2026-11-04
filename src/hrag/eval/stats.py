from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 42) -> float:
    """One-sided p-value that mean(b) > mean(a) by chance (fraction of resampled mean diffs <= 0)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if d.size == 0 or not d.any():
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return float((means <= 0).mean())


def paired_ttest(a: list[float], b: list[float]) -> float:
    """One-sided paired t-test p-value that mean(b) > mean(a) (same direction as paired_bootstrap)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if d.size == 0 or not d.any():
        return 1.0
    if np.all(d == d[0]):  # zero variance: t-test undefined
        return 0.0 if d[0] > 0 else 1.0
    return float(stats.ttest_rel(b, a, alternative="greater").pvalue)
