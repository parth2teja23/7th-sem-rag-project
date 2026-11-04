"""Matryoshka truncation of EmbeddingGemma (Task 14, Tier B) on the TEST split.

Per dim in {768, 512, 256, 128} x config in {dense, qaf+rr}:
  * doc vectors: the stored 768-d vectors are reconstructed from the existing FAISS index
    (indexes/<ds>/gemma), truncated to the first d dims and L2-renormalised, then indexed
    with the same HNSW settings -> indexes/<ds>/gemma_<d>. Nothing is re-encoded.
  * query vectors: the full encoder output truncated + renormalised the same way
    (TruncatedEncoder), i.e. sentence-transformers' truncate_dim + normalize.
  * QAF retrained per reduced dim on pooled SciFact-dev + FiQA-dev (train_qaf._dev_block /
    _fit, so FiQA-dev's gemma index is truncated too) and alpha* re-selected on the same
    pooled dev curve -> indexes/qaf_gemma_<d>.joblib, results/matryoshka/qaf/. Never tuned
    on test. The full dim reuses indexes/qaf_gemma.joblib and alpha* from train_qaf.py
    (the identical procedure).
Writes results/matryoshka/<dataset>__<dim>__<config>.json with config.dim, top-level
index_mb (size of index.faiss, MiB) and latency_ms (Pipeline.retrieve, 100 seeded test
queries after one warm-up).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

MAT_DATASETS = ("scifact", "rfc")
DIMS = (768, 512, 256, 128)


def truncate_rows(x: np.ndarray, dim: int) -> np.ndarray:
    v = np.ascontiguousarray(np.asarray(x, dtype=np.float32)[:, :dim])
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v / np.where(n == 0, 1.0, n), dtype=np.float32)


class TruncatedEncoder:
    """Wraps a loaded encoder: embeddings truncated to the first ``dim`` dims and
    L2-renormalised. Named ``<base>_<dim>`` so common.load_indexes/get_rankings/qaf paths
    address indexes/<ds>/<base>_<dim> and its own caches."""

    def __init__(self, base, dim: int):
        if dim > base.dim:
            raise ValueError(f"dim {dim} > native {base.dim}")
        self.base, self.dim = base, dim
        self.name = common.encoder_key(base.name, dim)
        self.model = base.model
        self.device = getattr(base, "device", None)

    def encode_queries(self, texts):
        return truncate_rows(self.base.encode_queries(texts), self.dim)

    def encode_docs(self, docs, batch_size: int = 16):
        return truncate_rows(self.base.encode_docs(docs, batch_size), self.dim)


def build_truncated_index(ds: str, base, enc: TruncatedEncoder, force: bool = False) -> Path:
    """indexes/<ds>/<base>_<dim> from the stored full-dim vectors of indexes/<ds>/<base>."""
    import faiss

    from hrag.dense import DenseIndex

    out = common.index_dir(ds, enc.name)
    if (out / "meta.json").exists() and not force:
        return out
    src = DenseIndex.load(common.index_dir(ds, base.name), base)
    full = src.index.reconstruct_n(0, src.index.ntotal)
    vecs = truncate_rows(full, enc.dim)
    index = faiss.IndexHNSWFlat(enc.dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    index.hnsw.efSearch = 128
    di = DenseIndex()
    di.index, di.ids, di.encoder = index, list(src.ids), enc
    di.save(out)
    log(f"built {ds}/{enc.name}: {len(di.ids)} vectors truncated from {base.name}")
    return out


def index_mb(ds: str, key: str) -> float:
    return (common.index_dir(ds, key) / "index.faiss").stat().st_size / 2 ** 20


def train_qaf_for(enc, vocab) -> tuple:
    """QAF + alpha* on pooled dev for a truncated encoder (train_qaf.py's functions)."""
    import train_qaf

    blocks = {}
    for ds in common.DEV_DATASETS:
        if common.dataset_available(ds):
            blocks[ds] = train_qaf._dev_block(ds, enc.name, enc, vocab)
    pooled = train_qaf._curve(np.vstack([b["curves"] for b in blocks.values()]))
    a_star = common.select_alpha(pooled)
    model, X, y = train_qaf._fit(list(blocks.values()))
    model.save(common.qaf_path(enc.name))
    common.write_json(common.RESULTS_DIR / "matryoshka" / "qaf" / f"alpha_star_{enc.name}.json", {
        "encoder": enc.name, "dim": enc.dim, "alpha_star": a_star,
        "selection": "argmax of pooled (query-level micro-averaged) dev nDCG@10 of "
                     "weighted_rrf(k=60, top_n=100) over ALPHA_GRID; ties -> closest to 0.5",
        "n_dev": {ds: len(b["qids"]) for ds, b in blocks.items()},
        "pooled_dev_ndcg@10": {str(a): v for a, v in pooled.items()},
        "qaf_model": f"indexes/qaf_{enc.name}.joblib",
        "qaf_coefficients": dict(model.coef_table()),
        "qaf_train_mae_clipped": float(np.abs(y - model.predict(X)).mean()),
    })
    log(f"[{enc.name}] alpha* = {a_star} (pooled dev nDCG@10 {pooled[a_star]:.4f}); QAF saved")
    return model, a_star


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(MAT_DATASETS))
    ap.add_argument("--encoder", default="gemma")
    ap.add_argument("--dims", nargs="+", type=int, default=list(DIMS))
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild truncated indexes")
    args = ap.parse_args(argv)

    from hrag.pipeline import Pipeline, PipelineConfig
    import run_ablation
    import run_retrieval

    base = common.make_encoder(args.encoder)
    vocab = common.encoder_vocab(base)
    reranker = common.make_reranker()
    native = base.dim
    for dim in sorted(args.dims, reverse=True):
        if dim == native:
            enc, qaf, a_star = base, common.load_qaf(base.name), common.alpha_star(base.name)
            qaf_src = f"indexes/qaf_{base.name}.joblib (train_qaf.py)"
        else:
            enc = TruncatedEncoder(base, dim)
            for ds in dict.fromkeys([*args.datasets, *common.DEV_DATASETS]):
                if common.dataset_available(ds):
                    build_truncated_index(ds, base, enc, args.force)
            qaf, a_star = train_qaf_for(enc, vocab)
            qaf_src = f"indexes/qaf_{enc.name}.joblib (retrained on pooled dev at d={dim})"

        for ds in args.datasets:
            if not common.dataset_available(ds):
                continue
            corpus, queries, qrels_all, _ = common.load_dataset(ds, "test")
            qids = [q for q in common.eval_qids(qrels_all) if q in queries]
            qrels = {q: qrels_all[q] for q in qids}
            sparse, dense = common.load_indexes(ds, enc)
            common.check_index_ids(ds, corpus, dense)
            rk = common.get_rankings(ds, "test", enc.name, {q: queries[q] for q in qids},
                                     sparse, dense)
            rr = common.RerankCache(ds, reranker)
            runs = {"dense": {}, "qaf+rr": {}}
            alphas = {"dense": {}, "qaf+rr": {}}
            for q in qids:
                d, s = rk[q]["dense"], rk[q]["sparse"]
                runs["dense"][q] = d
                aq = common.qaf_alpha(qaf, queries[q], d, s, sparse, vocab)
                alphas["qaf+rr"][q] = aq
                runs["qaf+rr"][q] = rr.rerank(q, queries[q], common.fuse(d, s, aq), corpus)
            rr.save()
            mb = index_mb(ds, enc.name)
            sample, warm = run_ablation.latency_sample(qids, args.latency_n)
            specs = {"dense": dict(use_sparse=False, rerank=False),
                     "qaf+rr": dict(alpha_mode="qaf", rerank=True)}
            for name, kw in specs.items():
                lat, live = {}, {}
                if not args.no_latency and qids:
                    pcfg = PipelineConfig(encoder=base.name, dim=dim, expand=False, dedupe=False,
                                          rerank_depth=common.RERANK_DEPTH,
                                          final_k=common.TOP_N, **kw)
                    pipe = Pipeline(pcfg, corpus, sparse, dense, reranker,
                                    qaf if name == "qaf+rr" else None)
                    lat, live = run_retrieval._measure_latency(
                        pipe, sample, warm, queries, runs[name],
                        alphas[name] if name == "qaf+rr" else None)
                m, pq = common.evaluate(runs[name], qrels)
                cfg = {"name": name, "dataset": ds, "split": "test", "encoder": base.name,
                       "dim": dim, "index": f"indexes/{ds}/{enc.name}",
                       "truncation": None if dim == native else
                       f"first {dim} of the stored {native}-d vectors, L2-renormalised",
                       "rrf_k": common.RRF_K, "retrieve_k": common.TOP_N, "dedupe": False,
                       "expand": False}
                if name == "qaf+rr":
                    cfg.update(alpha_mode="qaf", qaf_model=qaf_src, alpha_star=a_star,
                               rerank=True, rerank_depth=common.RERANK_DEPTH,
                               reranker=getattr(reranker, "name", None))
                extra = {"live_consistency": live} if live else {}
                common.write_result("matryoshka", ds, f"{dim}__{name}", m,
                                    common.per_query_with_alpha(pq, alphas[name]), lat, cfg,
                                    index_mb=mb, n_vectors=len(dense.ids),
                                    vectors_mb=len(dense.ids) * dim * 4 / 2 ** 20, **extra)
                tot = lat.get("total", {}).get("mean") if lat else None
                log(f"{ds} d={dim} {name:7s} nDCG@10={m['ndcg@10']:.4f} index {mb:.1f} MiB"
                    + (f" latency {tot:.1f} ms" if tot is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
