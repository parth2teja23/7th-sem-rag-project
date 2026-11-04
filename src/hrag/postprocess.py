from __future__ import annotations

from typing import Callable

from hrag.types import Doc, Ranking

_SHINGLE = 5


def _shingles(text: str) -> frozenset:
    words = tuple(text.lower().split())
    if len(words) < _SHINGLE:
        return frozenset([words])
    return frozenset(words[i : i + _SHINGLE] for i in range(len(words) - _SHINGLE + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def dedupe(ranking: Ranking, corpus: dict[str, Doc], threshold: float = 0.9) -> Ranking:
    kept: Ranking = []
    kept_sh: list[frozenset] = []
    for doc_id, score in ranking:
        doc = corpus.get(doc_id)
        if doc is None:
            kept.append((doc_id, score))
            continue
        sh = _shingles(doc.text)
        if any(_jaccard(sh, o) >= threshold for o in kept_sh):
            continue
        kept.append((doc_id, score))
        kept_sh.append(sh)
    return kept


def metadata_filter(
    ranking: Ranking,
    corpus: dict[str, Doc],
    predicate: Callable[[dict], bool] | None,
) -> Ranking:
    if predicate is None:
        return ranking
    return [(d, s) for d, s in ranking if d in corpus and predicate(corpus[d].meta)]
