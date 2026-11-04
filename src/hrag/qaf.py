"""Query-Adaptive Fusion (QAF): predict the dense/sparse fusion weight per query."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hrag.eval.metrics import ndcg_at_k
from hrag.fusion import weighted_rrf
from hrag.types import Ranking

FEATURE_NAMES = [
    "id_ratio", "num_ratio", "caps_ratio", "len_tokens", "mean_idf", "max_idf",
    "oov_ratio", "sparse_top", "sparse_margin", "dense_top", "dense_margin", "overlap_at10",
]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/:-][A-Za-z0-9]+)*")


def _is_caps(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    return len(tok) >= 2 and bool(letters) and all(c.isupper() for c in letters) and tok.isalnum()


def _is_num(tok: str) -> bool:
    return tok.isdigit()


def _is_id(tok: str) -> bool:
    if _is_num(tok):
        return False
    has_d = any(c.isdigit() for c in tok)
    has_a = any(c.isalpha() for c in tok)
    return (has_d and has_a) or any(c in tok for c in "_-./") or _is_caps(tok)


def _top_margin(r: Ranking) -> tuple[float, float]:
    if not r:
        return 0.0, 0.0
    top = float(r[0][1])
    return top, (top - float(r[1][1])) if len(r) > 1 else 0.0


def query_features(
    query: str,
    dense: Ranking,
    sparse: Ranking,
    idf: Callable[[str], float],
    tokenizer_vocab: set[str],
) -> np.ndarray:
    toks = _TOKEN_RE.findall(query)
    n = len(toks)
    d_top, d_margin = _top_margin(dense)
    s_top, s_margin = _top_margin(sparse)
    overlap = len({d for d, _ in dense[:10]} & {d for d, _ in sparse[:10]}) / 10.0
    if n:
        low = [t.lower() for t in toks]
        idfs = [float(idf(t)) for t in low]
        id_r = sum(_is_id(t) for t in toks) / n
        num_r = sum(_is_num(t) for t in toks) / n
        caps_r = sum(_is_caps(t) for t in toks) / n
        mean_idf, max_idf = sum(idfs) / n, max(idfs)
        oov = sum(t not in tokenizer_vocab for t in low) / n
    else:
        id_r = num_r = caps_r = mean_idf = max_idf = oov = 0.0
    return np.array(
        [id_r, num_r, caps_r, float(n), mean_idf, max_idf, oov,
         s_top, s_margin, d_top, d_margin, overlap],
        dtype=float,
    )


def _plateau_center(opt: list[int]) -> float:
    """opt: sorted grid indices that are optimal. Centre of the longest contiguous run."""
    runs: list[list[int]] = []
    for i in opt:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    means = [(len(r), sum(ALPHA_GRID[i] for i in r) / len(r)) for r in runs]
    _, mean = min(means, key=lambda lm: (-lm[0], abs(lm[1] - 0.5), lm[1]))
    return round(mean, 2)


def oracle_alpha(dense: Ranking, sparse: Ranking, rel: dict[str, int], k_eval: int = 10) -> float:
    """Oracle fusion weight: the centre of the longest plateau of grid alphas maximising nDCG@k.

    Rank-based fusion (RRF) is piecewise constant in alpha, so the optimum is usually a
    plateau of equally good weights rather than a single point. Taking the centre of the
    longest contiguous plateau (ties between equal runs go to the one nearest 0.5) gives a
    stable, unbiased regression target instead of an arbitrary or 0.5-collapsed grid point.
    Returns 0.5 when nothing relevant is retrieved.
    """
    if not any(v > 0 for v in rel.values()):
        return 0.5
    scores = [ndcg_at_k(weighted_rrf(dense, sparse, a), rel, k_eval) for a in ALPHA_GRID]
    best = max(scores)
    if best <= 0.0:
        return 0.5
    return _plateau_center([i for i, sc in enumerate(scores) if abs(sc - best) < 1e-12])


class QAFModel:
    def __init__(self) -> None:
        self.pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QAFModel":
        self.pipe.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.clip(self.pipe.predict(X), 0.0, 1.0)

    def coef_table(self) -> list[tuple[str, float]]:
        coefs = self.pipe.named_steps["ridge"].coef_
        return [(n, float(c)) for n, c in zip(FEATURE_NAMES, coefs)]

    def save(self, path) -> None:
        joblib.dump(self.pipe, Path(path))

    @classmethod
    def load(cls, path) -> "QAFModel":
        m = cls()
        m.pipe = joblib.load(Path(path))
        return m
