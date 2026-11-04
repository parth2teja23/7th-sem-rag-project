from __future__ import annotations

from hrag.types import Ranking, sort_ranking


def _positions(r: Ranking) -> dict[str, int]:
    pos: dict[str, int] = {}
    for i, (doc_id, _) in enumerate(r, start=1):
        pos.setdefault(doc_id, i)  # first (best) position wins
    return pos


def weighted_rrf(
    dense: Ranking, sparse: Ranking, alpha: float, k: int = 60, top_n: int = 100
) -> Ranking:
    """score(d) = alpha/(k+rank_dense) + (1-alpha)/(k+rank_sparse); missing list -> 0."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    dp, sp = _positions(dense), _positions(sparse)
    scores: Ranking = []
    for d in dp.keys() | sp.keys():
        s = 0.0
        if d in dp:
            s += alpha / (k + dp[d])
        if d in sp:
            s += (1.0 - alpha) / (k + sp[d])
        scores.append((d, s))
    return sort_ranking(scores)[:top_n]
