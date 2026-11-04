"""End-to-end hybrid retrieval pipeline with per-stage timing."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

from hrag.fusion import weighted_rrf
from hrag.generate import Answer, generate_answer
from hrag.postprocess import dedupe, metadata_filter
from hrag.rewrite import expand_query
from hrag.types import Doc, Ranking, RetrievalResult

RETRIEVE_K = 100
TIMING_KEYS = ("expand", "sparse", "dense", "features", "fusion", "rerank", "post")


@dataclass
class PipelineConfig:
    encoder: str = "bge-small"
    dim: int | None = None
    use_dense: bool = True
    use_sparse: bool = True
    alpha_mode: Literal["fixed", "qaf", "dat"] = "fixed"
    alpha: float = 0.5
    expand: bool = False
    rerank: bool = True
    rerank_depth: int = 50
    final_k: int = 5
    dedupe: bool = True


def vocab_from_encoder(encoder) -> set[str]:
    """Lowercased whole-word entries of the encoder's HF tokenizer vocab."""
    tok = encoder.model.tokenizer
    out: set[str] = set()
    for w in tok.get_vocab():
        for p in ("##", "▁", "Ġ"):
            if w.startswith(p):
                w = w[len(p):]
        if len(w) >= 2 and w.isalnum():
            out.add(w.lower())
    return out


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class Pipeline:
    def __init__(self, cfg: PipelineConfig, corpus: dict[str, Doc], sparse, dense,
                 reranker=None, qaf=None, filter_fn: Callable[[dict], bool] | None = None):
        if not (cfg.use_dense or cfg.use_sparse):
            raise ValueError("at least one of use_dense/use_sparse must be enabled")
        if cfg.use_dense and dense is None:
            raise ValueError("use_dense requires a dense index")
        if cfg.use_sparse and sparse is None:
            raise ValueError("use_sparse requires a sparse index")
        if cfg.rerank and reranker is None:
            raise ValueError("rerank requires a reranker")
        if cfg.alpha_mode == "qaf" and qaf is None:
            raise ValueError("alpha_mode='qaf' requires a QAFModel")
        self.cfg, self.corpus = cfg, corpus
        self.sparse = sparse if cfg.use_sparse else None
        self.dense = dense if cfg.use_dense else None
        self.reranker, self.qaf, self.filter_fn = reranker, qaf, filter_fn
        self._vocab: set[str] = set()
        if cfg.alpha_mode == "qaf" and self.dense is not None:
            enc = getattr(self.dense, "encoder", None)
            if enc is not None:
                self._vocab = vocab_from_encoder(enc)

    def _alpha(self, q: str, dense: Ranking, sparse: Ranking, timings: dict) -> float:
        mode = self.cfg.alpha_mode
        if mode == "fixed":
            return self.cfg.alpha
        if mode == "qaf":
            from hrag.qaf import query_features

            t = time.perf_counter()
            x = query_features(q, dense, sparse, self.sparse.idf, self._vocab)
            timings["features"] = _ms(t)
            return float(self.qaf.predict(x)[0])
        try:
            from hrag.baselines.dat import dat_alpha  # noqa: F401
        except ImportError as e:
            raise NotImplementedError("alpha_mode='dat' needs hrag.baselines.dat") from e
        raise NotImplementedError("alpha_mode='dat' is not wired up yet")

    def retrieve(self, query: str) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("empty query")
        cfg = self.cfg
        timings = {k: 0.0 for k in TIMING_KEYS}
        stages: dict[str, Ranking] = {}

        q = query
        if cfg.expand:
            t = time.perf_counter()
            q = expand_query(query)
            timings["expand"] = _ms(t)

        s: Ranking = []
        d: Ranking = []
        if self.sparse is not None:
            t = time.perf_counter()
            s = self.sparse.search(q, k=RETRIEVE_K)
            timings["sparse"] = _ms(t)
            stages["sparse"] = s
        if self.dense is not None:
            t = time.perf_counter()
            d = self.dense.search(q, k=RETRIEVE_K)
            timings["dense"] = _ms(t)
            stages["dense"] = d

        alpha: float | None = None
        if self.sparse is not None and self.dense is not None:
            alpha = self._alpha(q, d, s, timings)
            t = time.perf_counter()
            cur = weighted_rrf(d, s, alpha, top_n=RETRIEVE_K)
            timings["fusion"] = _ms(t)
            stages["fused"] = cur
        else:
            cur = d if self.dense is not None else s

        t = time.perf_counter()
        if cfg.dedupe:
            cur = dedupe(cur, self.corpus)
        cur = metadata_filter(cur, self.corpus, self.filter_fn)
        timings["post"] = _ms(t)

        if cfg.rerank:
            t = time.perf_counter()
            cur = self.reranker.rerank(q, cur, self.corpus, depth=cfg.rerank_depth)
            timings["rerank"] = _ms(t)
            stages["reranked"] = cur

        return RetrievalResult(query=query, final=cur[: cfg.final_k], stages=stages,
                               alpha=alpha, timings_ms=timings)

    def answer(self, query: str) -> tuple[RetrievalResult, Answer]:
        result = self.retrieve(query)
        passages = [self.corpus[d] for d, _ in result.final]
        return result, generate_answer(query, passages)
