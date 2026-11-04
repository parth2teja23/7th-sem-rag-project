"""Rerank depth N in {20, 50, 100} for qaf+rr (Task 14, Tier C) on the TEST split.

Better encoder (make_paper_assets.headline_encoder unless --encoder), QAF alpha, the same
offline path as run_retrieval.py; the cross-encoder scores the top-N of the fused list
(RerankCache persists the new pairs N=51..100 needs). The returned list is the reranked
top-N, exactly as Pipeline.retrieve(rerank_depth=N). Latency: Pipeline.retrieve on 100
seeded test queries after one warm-up.
Writes results/rerank_depth/<dataset>__<encoder>__qaf+rr_depth<N>.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

DEPTH_DATASETS = ("scifact", "rfc")
DEPTHS = (20, 50, 100)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DEPTH_DATASETS))
    ap.add_argument("--encoder", default=None, help="default: headline (better) encoder")
    ap.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args(argv)

    from hrag.pipeline import Pipeline, PipelineConfig
    import run_ablation
    import run_retrieval

    enc_name = run_ablation.resolve_encoder(args.encoder)
    encoder = common.make_encoder(enc_name)
    vocab = common.encoder_vocab(encoder)
    reranker = common.make_reranker()
    qaf = common.load_qaf(enc_name)

    for ds in args.datasets:
        if not common.dataset_available(ds):
            continue
        corpus, queries, qrels_all, _ = common.load_dataset(ds, "test")
        qids = [q for q in common.eval_qids(qrels_all) if q in queries]
        qrels = {q: qrels_all[q] for q in qids}
        sparse, dense = common.load_indexes(ds, encoder)
        common.check_index_ids(ds, corpus, dense)
        rk = common.get_rankings(ds, "test", enc_name, {q: queries[q] for q in qids}, sparse, dense)
        rr = common.RerankCache(ds, reranker)
        alphas, fused = {}, {}
        for q in qids:
            d, s = rk[q]["dense"], rk[q]["sparse"]
            alphas[q] = common.qaf_alpha(qaf, queries[q], d, s, sparse, vocab)
            fused[q] = common.fuse(d, s, alphas[q])
        sample, warm = run_ablation.latency_sample(qids, args.latency_n)
        for n in sorted(args.depths):
            run = {q: rr.rerank(q, queries[q], fused[q], corpus, depth=n) for q in qids}
            rr.save()
            lat, live = {}, {}
            if not args.no_latency and qids:
                pcfg = PipelineConfig(encoder=enc_name, alpha_mode="qaf", rerank=True,
                                      rerank_depth=n, expand=False, dedupe=False,
                                      final_k=common.TOP_N)
                pipe = Pipeline(pcfg, corpus, sparse, dense, reranker, qaf)
                lat, live = run_retrieval._measure_latency(pipe, sample, warm, queries, run, alphas)
            m, pq = common.evaluate(run, qrels)
            cfg = {"name": f"qaf+rr_depth{n}", "dataset": ds, "split": "test", "encoder": enc_name,
                   "alpha_mode": "qaf", "qaf_model": f"indexes/qaf_{enc_name}.joblib",
                   "rrf_k": common.RRF_K, "retrieve_k": common.TOP_N, "dedupe": False,
                   "expand": False, "rerank": True, "rerank_depth": n,
                   "reranker": getattr(reranker, "name", None),
                   "note": "returned list = CE-reranked top-N of the fused QAF list"}
            extra = {"live_consistency": live} if live else {}
            common.write_result("rerank_depth", ds, f"{enc_name}__qaf+rr_depth{n}", m,
                                common.per_query_with_alpha(pq, alphas), lat, cfg, **extra)
            tot = lat.get("total", {}).get("mean") if lat else None
            log(f"{ds} depth {n:3d} nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f}"
                + (f" latency {tot:.1f} ms" if tot is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
