from __future__ import annotations

from dataclasses import dataclass, field

Ranking = list[tuple[str, float]]  # (doc_id, score), sorted by score desc
Qrels = dict[str, dict[str, int]]  # qid -> doc_id -> relevance


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    final: Ranking
    stages: dict[str, Ranking]
    alpha: float | None
    timings_ms: dict[str, float]


def sort_ranking(r: Ranking) -> Ranking:
    """Sort by score descending; ties broken by doc_id ascending."""
    return sorted(r, key=lambda x: (-x[1], x[0]))
