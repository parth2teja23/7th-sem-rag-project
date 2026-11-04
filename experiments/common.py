"""Shared helpers for the Tier-A retrieval experiments.

Offline evaluation path (used by train_qaf.py and run_retrieval.py):
  1. For every (dataset, split, encoder) the dense top-100 and BM25 top-100 of every query are
     computed ONCE with ``DenseIndex.search`` / ``SparseIndex.search`` (exactly the calls
     ``Pipeline.retrieve`` makes) and cached under ``results/cache/rankings/`` (gitignored).
  2. Every config is derived from those lists: ``weighted_rrf(k=60, top_n=100)`` for fusion,
     then the cross-encoder ``Reranker`` at depth 50 for ``+rr`` configs.
  3. ``dedupe`` is NOT applied in the experiments (Pipeline dedupe=False equivalent); the
     metadata filter is a no-op (no predicate).
``tests/test_experiments_smoke.py`` proves this path reproduces ``Pipeline.retrieve``.

Result file contract: ``results/<exp>/<dataset>__<config>.json`` =
``{"config", "metrics", "per_query": {qid: {"ndcg@10", "alpha", ...}}, "latency_ms": {stage: {"mean", "p95"}}}``.
"""
from __future__ import annotations

import functools
import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from hrag.eval.metrics import evaluate_run
from hrag.fusion import weighted_rrf
from hrag.types import Doc, Qrels, Ranking, sort_ranking

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "indexes"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
SUBSETS_DIR = ROOT / "data" / "subsets"

SEED = 42
RRF_K = 60
TOP_N = 100
RERANK_DEPTH = 50
KS = (5, 10, 20)
FIQA_MAX_DOCS = 20000
SANITY_SCIFACT_BM25 = 0.665  # published BEIR BM25 (Anserini) nDCG@10 on SciFact test
SANITY_TOL = 0.05

DATASETS = ("scifact", "fiqa", "hotpot", "rfc")
SPLITS: dict[str, tuple[str, ...]] = {
    "scifact": ("dev", "test"),
    "fiqa": ("dev", "test"),
    "hotpot": ("test",),
    "rfc": ("test",),
}
DEV_DATASETS: tuple[str, ...] = ("scifact", "fiqa")  # datasets with a dev split (QAF training, alpha*)
ENCODER_NAMES = ("bge-small", "gemma")


def log(*a) -> None:
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- datasets

def _load_scifact(split: str):
    from hrag.data.beir import load_beir

    corpus, queries, qrels = load_beir("scifact", split)  # full corpus for every split
    return corpus, queries, qrels, None


@functools.lru_cache(maxsize=None)
def _fiqa_full(split: str):
    from hrag.data.beir import load_beir

    return load_beir("fiqa", split)


def fiqa_subset_ids() -> list[str]:
    """20k-doc FiQA corpus shared by dev and test: every qrel-positive doc of dev AND test,
    filled with a seeded (42) random sample of the remaining docs. Persisted to
    ``data/subsets/fiqa_corpus_20000.json``; if that file exists it must match."""
    corpus, _, qrels_dev = _fiqa_full("dev")
    _, _, qrels_test = _fiqa_full("test")
    pos = {d for qr in (qrels_dev, qrels_test) for rel in qr.values() for d, s in rel.items()
           if s > 0 and d in corpus}
    others = sorted(d for d in corpus if d not in pos)
    fill = random.Random(SEED).sample(others, min(max(0, FIQA_MAX_DOCS - len(pos)), len(others)))
    ids = sorted(pos | set(fill))
    path = SUBSETS_DIR / f"fiqa_corpus_{FIQA_MAX_DOCS}.json"
    if path.exists():
        saved = json.loads(path.read_text())["doc_ids"]
        if saved != ids:
            raise RuntimeError(f"{path} does not match the recomputed FiQA subset (data drift?)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dataset": "fiqa", "max_docs": FIQA_MAX_DOCS, "seed": SEED,
            "positives_from_splits": ["dev", "test"], "n_positive": len(pos),
            "n_full_corpus": len(corpus), "doc_ids": ids}))
    return ids


def _load_fiqa(split: str):
    corpus, queries, qrels = _fiqa_full(split)
    keep = fiqa_subset_ids()
    sub = {d: corpus[d] for d in keep}
    qrels = {q: {d: s for d, s in rel.items() if d in sub} for q, rel in qrels.items()}
    return sub, dict(queries), qrels, None


def _load_hotpot(split: str):
    from hrag.data.hotpot import load_hotpot

    return load_hotpot(500)


def _load_rfc(split: str):
    from hrag.data.rfc import load_rfc_corpus

    return load_rfc_corpus("semantic")


LOADERS: dict[str, Callable] = {
    "scifact": _load_scifact, "fiqa": _load_fiqa, "hotpot": _load_hotpot, "rfc": _load_rfc,
}


def dataset_available(name: str) -> bool:
    """False (with a reason logged) for RFC until the human-verified questions.csv exists."""
    if name == "rfc":
        try:
            from hrag.data.rfc import RFC_DIR
        except ImportError as e:
            log(f"SKIP rfc: hrag.data.rfc not importable ({e})")
            return False
        if not (Path(RFC_DIR) / "questions.csv").exists():
            log(f"SKIP rfc: {Path(RFC_DIR) / 'questions.csv'} missing "
                "(human verification of RFC questions pending); will be run later")
            return False
    return name in LOADERS


def load_dataset(name: str, split: str = "test"):
    """-> (corpus, queries, qrels, answers|None). Corpus is identical across a dataset's splits."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}")
    if split not in SPLITS[name]:
        raise ValueError(f"{name} has no {split!r} split (available: {SPLITS[name]})")
    return LOADERS[name](split)


def eval_qids(qrels: Qrels) -> list[str]:
    return sorted(q for q, rel in qrels.items() if any(v > 0 for v in rel.values()))


# ----------------------------------------------------------------------------- models / indexes

def make_encoder(name: str, dim: int | None = None):
    from hrag.dense import Encoder

    return Encoder(name, dim)


def make_reranker():
    from hrag.rerank import Reranker

    return Reranker()


def encoder_key(encoder: str, dim: int | None = None) -> str:
    return encoder if dim is None else f"{encoder}_{dim}"


def index_dir(dataset: str, kind: str) -> Path:
    return INDEX_DIR / dataset / kind


def load_indexes(name: str, encoder, dim: int | None = None):
    """-> (SparseIndex, DenseIndex|None). ``encoder`` is an encoder name (loaded via
    make_encoder), an already-loaded encoder object, or None for sparse only."""
    from hrag.dense import DenseIndex
    from hrag.sparse import SparseIndex

    sparse = SparseIndex.load(index_dir(name, "bm25"))
    if encoder is None:
        return sparse, None
    enc = make_encoder(encoder, dim) if isinstance(encoder, str) else encoder
    key = encoder_key(enc.name, dim if isinstance(encoder, str) else None)
    dense = DenseIndex.load(index_dir(name, key), enc)
    return sparse, dense


def check_index_ids(dataset: str, corpus: dict[str, Doc], dense=None) -> None:
    """Fail loudly if an index on disk was built from a different corpus."""
    meta = json.loads((index_dir(dataset, "bm25") / "meta.json").read_text())
    if set(meta["ids"]) != set(corpus):
        raise RuntimeError(f"{dataset}/bm25 index ids do not match the corpus; rebuild with --force")
    if dense is not None and set(dense.ids) != set(corpus):
        raise RuntimeError(f"{dataset} dense index ids do not match the corpus; rebuild with --force")


def qaf_path(encoder: str) -> Path:
    return INDEX_DIR / f"qaf_{encoder}.joblib"


def load_qaf(encoder: str):
    from hrag.qaf import QAFModel

    return QAFModel.load(qaf_path(encoder))


def encoder_vocab(encoder) -> set[str]:
    from hrag.pipeline import vocab_from_encoder

    return vocab_from_encoder(encoder)


# ----------------------------------------------------------------------------- offline retrieval

def _as_ranking(r) -> Ranking:
    return [(str(d), float(s)) for d, s in r]


def get_rankings(dataset: str, split: str, encoder: str, queries: dict[str, str],
                 sparse, dense, force: bool = False) -> dict[str, dict[str, Ranking]]:
    """{qid: {"dense": top-100, "sparse": top-100}} for every query, computed once per
    (dataset, split, encoder) with the same per-query search calls Pipeline.retrieve makes."""
    path = CACHE_DIR / "rankings" / f"{dataset}__{split}__{encoder}.json"
    cached: dict = {}
    if path.exists() and not force:
        blob = json.loads(path.read_text())
        if blob.get("n_docs") == len(dense.ids) and blob.get("k") == TOP_N:
            cached = blob["rankings"]
    missing = [q for q in queries if q not in cached or cached[q]["query"] != queries[q]]
    if missing:
        log(f"rankings {dataset}/{split}/{encoder}: computing {len(missing)} queries")
        t0 = time.perf_counter()
        for i, qid in enumerate(missing, 1):
            q = queries[qid]
            cached[qid] = {"query": q, "dense": dense.search(q, k=TOP_N),
                           "sparse": sparse.search(q, k=TOP_N)}
            if i % 200 == 0:
                log(f"  {i}/{len(missing)}")
        log(f"  done in {time.perf_counter() - t0:.1f}s")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n_docs": len(dense.ids), "k": TOP_N, "rankings": cached}))
    return {q: {"dense": _as_ranking(cached[q]["dense"]), "sparse": _as_ranking(cached[q]["sparse"])}
            for q in queries}


def fuse(dense: Ranking, sparse: Ranking, alpha: float) -> Ranking:
    return weighted_rrf(dense, sparse, alpha, k=RRF_K, top_n=TOP_N)


def qaf_alpha(qaf, query: str, dense: Ranking, sparse: Ranking, sparse_index, vocab) -> float:
    from hrag.qaf import query_features

    return float(qaf.predict(query_features(query, dense, sparse, sparse_index.idf, vocab))[0])


class RerankCache:
    """Cross-encoder scores cached per (qid, doc_id) so overlapping candidate lists (fixed vs
    qaf, bge vs gemma) are scored once. Missing pairs are scored through the public
    ``Reranker.rerank``; output follows its contract (top-``depth`` ids present in the corpus,
    sorted by score desc, ties by id)."""

    def __init__(self, dataset: str, reranker):
        self.reranker = reranker
        name = str(getattr(reranker, "name", "reranker")).replace("/", "_")
        self.path = CACHE_DIR / "rerank" / f"{dataset}__{name}.json"
        self.scores: dict[str, float] = {}
        self.queries: dict[str, str] = {}
        self._dirty = 0
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            self.scores, self.queries = blob["scores"], blob["queries"]

    def rerank(self, qid: str, query: str, ranking: Ranking, corpus: dict[str, Doc],
               depth: int = RERANK_DEPTH) -> Ranking:
        if depth <= 0 or not ranking:
            return []
        if self.queries.setdefault(qid, query) != query:
            raise RuntimeError(f"query text changed for {qid}; clear {self.path}")
        ids = [i for i, _ in ranking[:depth] if i in corpus]
        missing = [i for i in ids if f"{qid}\t{i}" not in self.scores]
        if missing:
            scored = self.reranker.rerank(query, [(i, 0.0) for i in missing], corpus,
                                          depth=len(missing))
            for i, s in scored:
                self.scores[f"{qid}\t{i}"] = float(s)
            self._dirty += len(missing)
            if self._dirty >= 5000:
                self.save()
        return sort_ranking([(i, self.scores[f"{qid}\t{i}"]) for i in ids])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"scores": self.scores, "queries": self.queries}))
        self._dirty = 0


# ----------------------------------------------------------------------------- evaluation / results

def evaluate(run: dict[str, Ranking], qrels: Qrels):
    return evaluate_run(run, qrels, ks=KS)


def per_query_with_alpha(per_query: dict, alphas: dict[str, float | None]) -> dict:
    return {q: {**m, "alpha": alphas.get(q)} for q, m in per_query.items()}


def alpha_star_path(encoder: str) -> Path:
    return RESULTS_DIR / "qaf" / f"alpha_star_{encoder}.json"


def select_alpha(curve: dict[float, float]) -> float:
    """Argmax of mean nDCG@10 over the grid; ties -> closest to 0.5, then smaller alpha."""
    best = max(curve.values())
    tied = [a for a, v in curve.items() if abs(v - best) < 1e-12]
    return min(tied, key=lambda a: (abs(a - 0.5), a))


def alpha_star(encoder: str) -> float:
    """Best fixed alpha on pooled dev (written by train_qaf.py)."""
    p = alpha_star_path(encoder)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run experiments/train_qaf.py first")
    return float(json.loads(p.read_text())["alpha_star"])


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return path


def write_result(exp: str, dataset: str, config_name: str, metrics: dict, per_query: dict,
                 latency: dict | None, config: dict | None = None, **extra) -> Path:
    """Write ``results/<exp>/<dataset>__<config_name>.json`` (result file contract)."""
    obj = {"config": {"name": config_name, **(config or {})}, "metrics": metrics,
           "per_query": per_query, "latency_ms": latency or {}}
    obj.update(extra)
    return write_json(RESULTS_DIR / exp / f"{dataset}__{config_name}.json", obj)


def latency_stats(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean / p95 per stage over the sampled queries; stages that never ran are dropped."""
    out: dict[str, dict[str, float]] = {}
    for k in samples[0]:
        v = np.asarray([s[k] for s in samples], dtype=float)
        if k != "total" and not v.any():
            continue
        out[k] = {"mean": float(v.mean()), "p95": float(np.percentile(v, 95))}
    return out
