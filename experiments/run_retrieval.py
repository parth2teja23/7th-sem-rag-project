"""Core retrieval experiments (Tier A) on the TEST split of each dataset, per encoder.

Configs (all derived offline from cached dense/BM25 top-100; dedupe NOT applied):
  bm25, dense, hybrid_fixed (alpha* from pooled dev), hybrid_fixed+rr, qaf, qaf+rr,
  oracle_upper_bound (per-query oracle_alpha on TEST qrels - not a deployable method).
Fusion: weighted_rrf(k=60, top_n=100). Rerank: cross-encoder, depth 50.
Metrics on the full returned list (<=100, or <=50 for +rr) at k = 5, 10, 20.

Writes results/retrieval/<dataset>__[<encoder>__]<config>.json, results/alpha_sweep/
<dataset>__<encoder>.json, results/retrieval/summary.json and results/significance.json.
Latency: Pipeline.retrieve (expand=False, dedupe=False) on 100 sampled test queries
(seed 42) after one warm-up query, for dense, hybrid_fixed+rr and qaf+rr.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

HEADLINE = ("ndcg@10", "recall@10", "mrr@10")
COMPARISONS = (("qaf+rr", "hybrid_fixed+rr"), ("qaf+rr", "dense"), ("qaf", "hybrid_fixed"))


def _measure_latency(pipe, sample: list[str], warmup: str, queries: dict[str, str],
                     offline: dict | None, offline_alpha: dict | None):
    pipe.retrieve(queries[warmup])
    samples, top10_match, alpha_match = [], 0, 0
    for q in sample:
        t0 = time.perf_counter()
        res = pipe.retrieve(queries[q])
        total = (time.perf_counter() - t0) * 1000.0
        samples.append({**res.timings_ms, "total": total})
        if offline is not None:
            top10_match += [d for d, _ in res.final[:10]] == [d for d, _ in offline[q][:10]]
        if offline_alpha is not None and res.alpha is not None:
            alpha_match += abs(res.alpha - offline_alpha[q]) < 1e-9
    check = {"n": len(sample), "top10_identical_rate": top10_match / len(sample)}
    if offline_alpha is not None and pipe.cfg.use_dense and pipe.cfg.use_sparse:
        check["alpha_identical_rate"] = alpha_match / len(sample)
    return common.latency_stats(samples), check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(common.DATASETS))
    ap.add_argument("--encoders", nargs="+", default=list(common.ENCODER_NAMES))
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--skip-sanity-gate", action="store_true")
    args = ap.parse_args(argv)

    from hrag.pipeline import Pipeline, PipelineConfig
    from hrag.eval.stats import paired_bootstrap, paired_ttest
    from hrag.qaf import ALPHA_GRID, oracle_alpha
    from hrag.eval.metrics import ndcg_at_k

    sig_path = common.RESULTS_DIR / "significance.json"
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    sum_path = common.RESULTS_DIR / "retrieval" / "summary.json"
    summary = json.loads(sum_path.read_text()) if sum_path.exists() else {}
    encoders, vocabs, reranker = {}, {}, None

    for ds in args.datasets:
        if not common.dataset_available(ds):
            continue
        corpus, queries, qrels_all, _ = common.load_dataset(ds, "test")
        qids = [q for q in common.eval_qids(qrels_all) if q in queries]
        qrels = {q: qrels_all[q] for q in qids}
        log(f"{ds}: {len(corpus)} docs, {len(qids)} test queries with qrels")
        if reranker is None:
            reranker = common.make_reranker()
        rr = common.RerankCache(ds, reranker)
        bm25_done = False

        for enc_name in args.encoders:
            if enc_name not in encoders:
                encoders[enc_name] = common.make_encoder(enc_name)
                vocabs[enc_name] = common.encoder_vocab(encoders[enc_name])
            sparse, dense = common.load_indexes(ds, encoders[enc_name])
            common.check_index_ids(ds, corpus, dense)
            rk = common.get_rankings(ds, "test", enc_name, {q: queries[q] for q in qids}, sparse, dense)
            a_star = common.alpha_star(enc_name)
            qaf = common.load_qaf(enc_name)
            base_cfg = {"dataset": ds, "split": "test", "encoder": enc_name, "rrf_k": common.RRF_K,
                        "retrieve_k": common.TOP_N, "dedupe": False, "expand": False}

            if not bm25_done:
                run = {q: rk[q]["sparse"] for q in qids}
                m, pq = common.evaluate(run, qrels)
                common.write_result("retrieval", ds, "bm25", m, common.per_query_with_alpha(pq, {}),
                                    {}, {"dataset": ds, "split": "test", "retrieve_k": common.TOP_N,
                                         "note": "encoder-independent"})
                summary.setdefault(ds, {})["bm25"] = {k: m[k] for k in HEADLINE}
                bm25_done = True
                log(f"{ds} bm25 nDCG@10={m['ndcg@10']:.4f}")
                if ds == "scifact":
                    dev = abs(m["ndcg@10"] - common.SANITY_SCIFACT_BM25)
                    ok = dev <= common.SANITY_TOL
                    log(f"SANITY GATE scifact bm25 nDCG@10 {m['ndcg@10']:.4f} vs published "
                        f"{common.SANITY_SCIFACT_BM25} (|diff|={dev:.4f}) -> {'PASS' if ok else 'FAIL'}")
                    if not ok and not args.skip_sanity_gate:
                        log("stopping: investigate BM25 before continuing")
                        return 2

            # derive every config offline
            alphas = {"dense": {}, "hybrid_fixed": {}, "qaf": {}, "oracle_upper_bound": {}}
            runs: dict[str, dict] = {"dense": {}, "hybrid_fixed": {}, "hybrid_fixed+rr": {},
                                     "qaf": {}, "qaf+rr": {}, "oracle_upper_bound": {}}
            sweep = {a: [] for a in ALPHA_GRID}
            grid_max = []
            t0 = time.perf_counter()
            for q in qids:
                d, s = rk[q]["dense"], rk[q]["sparse"]
                runs["dense"][q] = d
                runs["hybrid_fixed"][q] = common.fuse(d, s, a_star)
                alphas["hybrid_fixed"][q] = a_star
                aq = common.qaf_alpha(qaf, queries[q], d, s, sparse, vocabs[enc_name])
                alphas["qaf"][q] = aq
                runs["qaf"][q] = common.fuse(d, s, aq)
                ao = oracle_alpha(d, s, qrels[q])
                alphas["oracle_upper_bound"][q] = ao
                runs["oracle_upper_bound"][q] = common.fuse(d, s, ao)
                per_a = [ndcg_at_k(common.fuse(d, s, a), qrels[q], 10) for a in ALPHA_GRID]
                for a, v in zip(ALPHA_GRID, per_a):
                    sweep[a].append(v)
                grid_max.append(max(per_a))
                runs["hybrid_fixed+rr"][q] = rr.rerank(q, queries[q], runs["hybrid_fixed"][q], corpus)
                runs["qaf+rr"][q] = rr.rerank(q, queries[q], runs["qaf"][q], corpus)
            rr.save()
            alphas["hybrid_fixed+rr"] = alphas["hybrid_fixed"]
            alphas["qaf+rr"] = alphas["qaf"]
            log(f"{ds}/{enc_name}: configs derived in {time.perf_counter() - t0:.1f}s (alpha*={a_star})")

            # latency + live consistency check against Pipeline.retrieve
            lat: dict[str, dict] = {}
            live: dict[str, dict] = {}
            if not args.no_latency and qids:
                rng = random.Random(common.SEED)
                sample = rng.sample(qids, min(args.latency_n, len(qids)))
                rest = [q for q in qids if q not in set(sample)]
                warm = rest[0] if rest else sample[0]
                specs = {
                    "dense": dict(use_sparse=False, rerank=False),
                    "hybrid_fixed+rr": dict(alpha_mode="fixed", alpha=a_star, rerank=True),
                    "qaf+rr": dict(alpha_mode="qaf", rerank=True),
                }
                for name, kw in specs.items():
                    cfg = PipelineConfig(encoder=enc_name, expand=False, dedupe=False,
                                         rerank_depth=common.RERANK_DEPTH, final_k=common.TOP_N, **kw)
                    pipe = Pipeline(cfg, corpus, sparse, dense, reranker,
                                    qaf if cfg.alpha_mode == "qaf" else None)
                    lat[name], live[name] = _measure_latency(
                        pipe, sample, warm, queries, runs[name],
                        alphas.get(name) if name != "dense" else None)
                    log(f"{ds}/{enc_name} latency {name}: total mean "
                        f"{lat[name]['total']['mean']:.1f} ms, p95 {lat[name]['total']['p95']:.1f} ms; "
                        f"live top-10 identical {live[name]['top10_identical_rate']:.2%}")

            per_query_ndcg = {}
            sm = summary.setdefault(ds, {}).setdefault(enc_name, {})
            for name, run in runs.items():
                m, pq = common.evaluate(run, qrels)
                per_query_ndcg[name] = [pq[q]["ndcg@10"] for q in qids]
                cfg = {"name": name, **base_cfg}
                if name == "dense":
                    pass
                elif name.startswith("hybrid_fixed"):
                    cfg.update(alpha=a_star, alpha_selected_on="pooled dev (scifact-dev + fiqa-dev)",
                               label=f"hybrid_fixed@{a_star}" + ("+rr" if name.endswith("+rr") else ""))
                elif name.startswith("qaf"):
                    a = np.asarray(list(alphas["qaf"].values()))
                    cfg.update(alpha_mode="qaf", qaf_model=f"indexes/qaf_{enc_name}.joblib",
                               alpha_pred_mean=float(a.mean()), alpha_pred_std=float(a.std()))
                else:
                    cfg.update(alpha_mode="oracle", upper_bound=True,
                               note="per-query oracle_alpha computed on TEST qrels; NOT a "
                                    "deployable method, upper bound only",
                               **{"grid_max_ndcg@10": float(np.mean(grid_max))})
                if name.endswith("+rr"):
                    cfg.update(rerank=True, rerank_depth=common.RERANK_DEPTH,
                               reranker=getattr(reranker, "name", None))
                extra = {"live_consistency": live[name]} if name in live else {}
                common.write_result("retrieval", ds, f"{enc_name}__{name}", m,
                                    common.per_query_with_alpha(pq, alphas.get(name, {})),
                                    lat.get(name, {}), cfg, **extra)
                sm[name] = {k: m[k] for k in HEADLINE}
                log(f"{ds}/{enc_name} {name:20s} nDCG@10={m['ndcg@10']:.4f} "
                    f"R@10={m['recall@10']:.4f} MRR@10={m['mrr@10']:.4f}")

            h, b, dn = sm["hybrid_fixed"]["ndcg@10"], summary[ds]["bm25"]["ndcg@10"], sm["dense"]["ndcg@10"]
            log(f"CHECK {ds}/{enc_name}: hybrid {h:.4f} >= min(bm25 {b:.4f}, dense {dn:.4f}) -> "
                f"{'OK' if h >= min(b, dn) else 'VIOLATED'}")

            dev_curves = None
            ap_ = common.alpha_star_path(enc_name)
            if ap_.exists():
                dev_curves = json.loads(ap_.read_text())["per_dataset_dev_ndcg@10"].get(ds)
            common.write_json(common.RESULTS_DIR / "alpha_sweep" / f"{ds}__{enc_name}.json", {
                "dataset": ds, "encoder": enc_name, "metric": "ndcg@10", "alpha_star": a_star,
                "note": "alpha = weight on dense in weighted_rrf(k=60); no rerank",
                "n_test": len(qids),
                "test": {str(a): float(np.mean(v)) for a, v in sweep.items()},
                "dev": dev_curves})

            sd = sig.setdefault(ds, {}).setdefault(enc_name, {})
            for bname, aname in COMPARISONS:
                a_v, b_v = per_query_ndcg[aname], per_query_ndcg[bname]
                sd[f"{bname}_vs_{aname}"] = {
                    "metric": "ndcg@10", "a": aname, "b": bname, "n": len(qids),
                    "mean_a": float(np.mean(a_v)), "mean_b": float(np.mean(b_v)),
                    "mean_diff": float(np.mean(b_v) - np.mean(a_v)),
                    "p_bootstrap": paired_bootstrap(a_v, b_v),
                    "p_ttest": paired_ttest(a_v, b_v),
                    "alternative": "one-sided, mean(b) > mean(a); bootstrap n=10000 seed 42"}
            common.write_json(sig_path, sig)
            common.write_json(sum_path, summary)
    common.write_json(sig_path, sig)
    common.write_json(sum_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
