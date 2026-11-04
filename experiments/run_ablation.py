"""Component ablation of the full system on the TEST split (Task 14), better encoder only.

Variants (each removes ONE component from ``full``):
  full        structural chunking + QAF alpha + cross-encoder rerank (depth 50)
              [+ LLM query expansion only when --expansion actually ran]
  -chunking   fixed 256-word chunks instead of structural chunks (datasets in CHUNK_VARIANTS:
              RFC only). Qrels are re-derived by answer-span overlap (hrag.data.rfc.span_relevant
              via load_rfc_corpus("fixed")): every fixed chunk overlapping the gold answer
              [answer_start, answer_end) of the parent RFC is relevant.
  -expansion  full without query expansion (only written when --expansion ran)
  -qaf        fixed alpha* (pooled dev) instead of QAF
  -rerank     no cross-encoder; the fused QAF top-100

Same offline path as run_retrieval.py (cached dense/BM25 top-100 -> weighted_rrf(k=60) ->
RerankCache depth 50; dedupe off). Latency: Pipeline.retrieve on 100 seeded (42) test
queries after one warm-up, per variant. Writes results/ablation/<dataset>__<variant>.json and
results/latency_summary.json (mean / p95 per stage of ``full``).

The "better encoder" defaults to make_paper_assets.headline_encoder (pooled DEV nDCG@10).
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

ABL_DATASETS = ("scifact", "rfc")
FIXED_CHUNK_WORDS = 256  # hrag.chunking.fixed_chunks default (plan, Task 4)


def _rfc_fixed(split: str):
    from hrag.data.rfc import load_rfc_corpus

    return load_rfc_corpus("fixed")


# dataset -> loader(split) of its fixed-size-chunk variant (same queries, span-remapped qrels)
CHUNK_VARIANTS: dict[str, Callable] = {"rfc": _rfc_fixed}


def _generation_running() -> bool:
    """True while experiments/run_generation.py (an Ollama job) is running: expansion
    must not start a concurrent Ollama job."""
    r = subprocess.run(["pgrep", "-f", "run_generation.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def resolve_encoder(name: str | None) -> str:
    if name:
        return name
    import make_paper_assets as mpa

    enc, how = mpa.headline_encoder(common.RESULTS_DIR, mpa.load_retrieval(common.RESULTS_DIR))
    if enc is None:
        raise RuntimeError("no retrieval results to pick the better encoder from; pass --encoder")
    log(f"better encoder = {enc} (chosen by {how})")
    return enc


def latency_sample(qids: list[str], n: int) -> tuple[list[str], str]:
    """Same sampling as run_retrieval.py: seed-42 sample of n test qids + a warm-up qid."""
    sample = random.Random(common.SEED).sample(qids, min(n, len(qids)))
    rest = [q for q in qids if q not in set(sample)]
    return sample, (rest[0] if rest else sample[0])


def ensure_indexes(ds: str, corpus: dict, encoder, batch_size: int) -> dict:
    """Build BM25 + dense indexes for a derived corpus (e.g. rfc_fixed) when missing or stale.
    Build times are returned (kept out of results/index_build_times.json)."""
    import json

    from hrag.dense import DenseIndex
    from hrag.sparse import SparseIndex

    times = {}
    bdir, ddir = common.index_dir(ds, "bm25"), common.index_dir(ds, encoder.name)

    def ok(d: Path) -> bool:
        m = d / "meta.json"
        return m.exists() and set(json.loads(m.read_text())["ids"]) == set(corpus)

    if not ok(bdir):
        t0 = time.perf_counter()
        sp = SparseIndex()
        sp.build(corpus)
        sp.save(bdir)
        times["bm25_s"] = round(time.perf_counter() - t0, 2)
    if not ok(ddir):
        log(f"encoding {len(corpus)} {ds} docs with {encoder.name} (batch {batch_size})")
        t0 = time.perf_counter()
        di = DenseIndex()
        di.build(corpus, encoder, batch_size=batch_size)
        di.save(ddir)
        times[f"{encoder.name}_s"] = round(time.perf_counter() - t0, 2)
    return times


def _expanded_queries(queries: dict[str, str], qids: list[str]) -> dict[str, str]:
    import hrag.rewrite

    return {q: hrag.rewrite.expand_query(queries[q]) for q in qids}


def _derive(qids, qtext, rk, sparse, vocab, qaf, a_star, rr, corpus) -> tuple[dict, dict]:
    """runs/alphas for full (qaf+rr), -qaf (fixed+rr), -rerank (qaf). ``qtext`` is the text
    the pipeline would retrieve/rerank with (the expanded query when expansion is on)."""
    runs = {"full": {}, "-qaf": {}, "-rerank": {}}
    alphas = {"full": {}, "-qaf": {}, "-rerank": {}}
    for q in qids:
        d, s = rk[q]["dense"], rk[q]["sparse"]
        aq = common.qaf_alpha(qaf, qtext[q], d, s, sparse, vocab)
        fq = common.fuse(d, s, aq)
        runs["-rerank"][q] = fq
        runs["full"][q] = rr.rerank(q, qtext[q], fq, corpus)
        runs["-qaf"][q] = rr.rerank(q, qtext[q], common.fuse(d, s, a_star), corpus)
        alphas["full"][q] = alphas["-rerank"][q] = aq
        alphas["-qaf"][q] = a_star
    rr.save()
    return runs, alphas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(ABL_DATASETS))
    ap.add_argument("--encoder", default=None, help="default: headline (better) encoder")
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16, help="encode batch for new indexes")
    ap.add_argument("--expansion", action="store_true",
                    help="also run the Tier-B expansion ablation (Ollama); refused while "
                         "run_generation.py is running")
    args = ap.parse_args(argv)

    from hrag.pipeline import Pipeline, PipelineConfig
    import run_retrieval

    enc_name = resolve_encoder(args.encoder)
    expand = False
    exp_note = ("query expansion NOT run (Tier B, --expansion not given): full = structural "
                "chunking + QAF + rerank, expand=False")
    if args.expansion:
        if _generation_running():
            exp_note = ("query expansion SKIPPED: run_generation.py (Ollama) was running and no "
                        "concurrent Ollama job is allowed; full = structural chunking + QAF + "
                        "rerank, expand=False")
            log(exp_note)
        else:
            expand = True
            exp_note = "full includes LLM query expansion (hrag.rewrite.expand_query, cached)"

    encoder = common.make_encoder(enc_name)
    vocab = common.encoder_vocab(encoder)
    reranker = common.make_reranker()
    qaf = common.load_qaf(enc_name)
    a_star = common.alpha_star(enc_name)
    full_lat: dict[str, dict] = {}

    for ds in args.datasets:
        if not common.dataset_available(ds):
            continue
        corpus, queries, qrels_all, _ = common.load_dataset(ds, "test")
        qids = [q for q in common.eval_qids(qrels_all) if q in queries]
        qrels = {q: qrels_all[q] for q in qids}
        sparse, dense = common.load_indexes(ds, encoder)
        common.check_index_ids(ds, corpus, dense)
        log(f"{ds}: {len(corpus)} docs, {len(qids)} test queries; encoder {enc_name}, "
            f"alpha*={a_star}, expand={expand}")
        sample, warm = latency_sample(qids, args.latency_n)
        base_cfg = {"dataset": ds, "split": "test", "encoder": enc_name, "rrf_k": common.RRF_K,
                    "retrieve_k": common.TOP_N, "dedupe": False, "chunker": "structural",
                    "note": exp_note}

        # (variant) -> (runs, alphas, corpus, sparse, dense, cfg-overrides)
        variants: dict[str, tuple] = {}
        plain_q = {q: queries[q] for q in qids}
        rk = common.get_rankings(ds, "test", enc_name, plain_q, sparse, dense)
        rr = common.RerankCache(ds, reranker)
        if expand:
            xq = _expanded_queries(queries, qids)
            rkx = common.get_rankings(ds, "test-expanded", enc_name, xq, sparse, dense)
            runs, alphas = _derive(qids, xq, rkx, sparse, vocab, qaf, a_star,
                                   common.RerankCache(f"{ds}-expanded", reranker), corpus)
            r0, a0 = _derive(qids, plain_q, rk, sparse, vocab, qaf, a_star, rr, corpus)
            variants["-expansion"] = (r0["full"], a0["full"], {"expand": False})
        else:
            runs, alphas = _derive(qids, plain_q, rk, sparse, vocab, qaf, a_star, rr, corpus)
        for v in ("full", "-qaf", "-rerank"):
            variants[v] = (runs[v], alphas[v], {"expand": expand})

        # -chunking: same pipeline over the fixed-size-chunk corpus
        fixed_env = None
        if ds in CHUNK_VARIANTS:
            vds = f"{ds}_fixed"
            fcorpus, fqueries, fqrels_all, _ = CHUNK_VARIANTS[ds]("test")
            build_t = ensure_indexes(vds, fcorpus, encoder, args.batch_size)
            fsparse, fdense = common.load_indexes(vds, encoder)
            fq_ids = [q for q in common.eval_qids(fqrels_all) if q in fqueries]
            if set(fq_ids) != set(qids):
                raise RuntimeError(f"{vds}: evaluated queries differ from {ds} "
                                   f"({len(fq_ids)} vs {len(qids)}); qrel remap lost a span?")
            fqt = {q: (xq if expand else plain_q)[q] for q in qids}
            frk = common.get_rankings(vds, "test-expanded" if expand else "test", enc_name, fqt,
                                      fsparse, fdense)
            frr = common.RerankCache(vds + ("-expanded" if expand else ""), reranker)
            fruns, falphas = _derive(qids, fqt, frk, fsparse, vocab, qaf, a_star, frr, fcorpus)
            n_rel = [sum(1 for v in fqrels_all[q].values() if v > 0) for q in qids]
            variants["-chunking"] = (fruns["full"], falphas["full"], {
                "expand": expand, "chunker": "fixed", "chunk_words": FIXED_CHUNK_WORDS,
                "chunk_overlap": 0, "n_chunks": len(fcorpus),
                "mean_rel_chunks_per_query": float(np.mean(n_rel)),
                "qrels": "fixed chunks overlapping the gold answer span [answer_start, "
                         "answer_end) of the parent RFC (hrag.data.rfc.span_relevant)",
                **({"index_build_s": build_t} if build_t else {})})
            fixed_env = (fcorpus, fsparse, fdense, {q: fqrels_all[q] for q in qids})

        # latency via the live pipeline
        specs = {"full": dict(alpha_mode="qaf", rerank=True),
                 "-qaf": dict(alpha_mode="fixed", alpha=a_star, rerank=True),
                 "-rerank": dict(alpha_mode="qaf", rerank=False),
                 "-chunking": dict(alpha_mode="qaf", rerank=True),
                 "-expansion": dict(alpha_mode="qaf", rerank=True)}
        for v, (run, al, over) in variants.items():
            cor, sp, dn, qr = ((fixed_env if v == "-chunking" else None)
                               or (corpus, sparse, dense, qrels))
            lat, live = {}, {}
            if not args.no_latency and qids:
                kw = specs[v]
                pcfg = PipelineConfig(encoder=enc_name, expand=over["expand"], dedupe=False,
                                      rerank_depth=common.RERANK_DEPTH, final_k=common.TOP_N, **kw)
                pipe = Pipeline(pcfg, cor, sp, dn, reranker,
                                qaf if pcfg.alpha_mode == "qaf" else None)
                lat, live = run_retrieval._measure_latency(pipe, sample, warm, queries, run, al)
            m, pq = common.evaluate(run, qr)
            cfg = {**base_cfg, "variant": v, **over,
                   "alpha_mode": specs[v]["alpha_mode"],
                   "rerank": specs[v]["rerank"]}
            if specs[v]["alpha_mode"] == "fixed":
                cfg["alpha"] = a_star
            else:
                cfg["qaf_model"] = f"indexes/qaf_{enc_name}.joblib"
            if specs[v]["rerank"]:
                cfg.update(rerank_depth=common.RERANK_DEPTH, reranker=getattr(reranker, "name", None))
            extra = {"live_consistency": live} if live else {}
            common.write_result("ablation", ds, v, m, common.per_query_with_alpha(pq, al), lat,
                                cfg, **extra)
            if v == "full" and lat:
                full_lat[ds] = lat
            tot = lat.get("total", {}).get("mean") if lat else None
            log(f"{ds} {v:11s} nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f}"
                + (f"  latency {tot:.1f} ms (live top-10 identical "
                   f"{live['top10_identical_rate']:.0%})" if tot is not None else ""))
        if "-rerank" in variants and ds == "scifact":
            a = common.evaluate(variants["full"][0], qrels)[0]["ndcg@10"]
            b = common.evaluate(variants["-rerank"][0], qrels)[0]["ndcg@10"]
            log(f"SANITY scifact: removing rerank {'lowers' if b < a else 'does NOT lower'} "
                f"nDCG@10 ({a:.4f} -> {b:.4f})")

    if full_lat:
        stages = sorted({s for v in full_lat.values() for s in v})
        common.write_json(common.RESULTS_DIR / "latency_summary.json", {
            "config": "full", "encoder": enc_name, "expand": expand,
            "components": "structural chunks + QAF alpha + CE rerank depth 50"
                          + (" + query expansion" if expand else ""),
            "n_queries": args.latency_n, "unit": "ms",
            "protocol": "Pipeline.retrieve (dedupe=False) on a seed-42 sample of test queries "
                        "after one warm-up query; per-stage mean / p95",
            "datasets": full_lat,
            "mean_over_datasets": {s: {k: float(np.mean([full_lat[d][s][k] for d in full_lat
                                                          if s in full_lat[d]]))
                                       for k in ("mean", "p95")} for s in stages},
            "note": exp_note})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
