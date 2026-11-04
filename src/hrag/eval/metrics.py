from __future__ import annotations

import math
import re
import string
from collections import Counter

from hrag.types import Qrels, Ranking


def recall_at_k(ranking: Ranking, rel: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for d, _ in ranking[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranking: Ranking, rel: dict[str, int], k: int) -> float:
    for i, (d, _) in enumerate(ranking[:k], start=1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Ranking, rel: dict[str, int], k: int) -> float:
    dcg = _dcg([max(rel.get(d, 0), 0) for d, _ in ranking[:k]])
    ideal = _dcg(sorted((g for g in rel.values() if g > 0), reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_run(
    run: dict[str, Ranking], qrels: Qrels, ks=(5, 10, 20)
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Mean and per-query metrics over every query in qrels.

    Queries without qrels in `run` are skipped; queries in qrels but missing
    from `run` count as an empty ranking (all metrics 0).
    """
    per_query: dict[str, dict[str, float]] = {}
    for qid, rel in qrels.items():
        if not rel:
            continue
        ranking = run.get(qid, [])
        m: dict[str, float] = {}
        for k in ks:
            m[f"recall@{k}"] = recall_at_k(ranking, rel, k)
            m[f"mrr@{k}"] = mrr_at_k(ranking, rel, k)
            m[f"ndcg@{k}"] = ndcg_at_k(ranking, rel, k)
        per_query[qid] = m
    mean: dict[str, float] = {}
    if per_query:
        for name in next(iter(per_query.values())):
            mean[name] = sum(m[name] for m in per_query.values()) / len(per_query)
    return mean, per_query


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, golds: list[str]) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)
