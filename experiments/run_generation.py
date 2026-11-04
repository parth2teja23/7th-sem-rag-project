"""End-to-end answer quality (EM / F1 / citation_hit) per retrieval config.

For each dataset the final top-5 passages of every config are rebuilt OFFLINE from the cached
dense/BM25 top-100 rankings, weighted RRF and cached cross-encoder scores (no corpus is
re-encoded), then ``hrag.generate.generate_answer`` (Ollama qwen2.5:7b-instruct, temperature 0,
seed 42) answers from them.  LLM outputs are cached on disk in results/cache/llm.sqlite (keyed by
model + prompt), so an interrupted run resumes without repeating generations.

Configs: bm25, dense, hybrid_fixed+rr, qaf+rr.  One encoder (``headline_encoder`` of
make_paper_assets, chosen by pooled DEV nDCG, never test) serves dense / hybrid / qaf.
Question set: ``--n`` questions sampled with seed 42 from the evaluated test questions; a dataset
with fewer questions than ``--n`` (RFC) uses all of them.  The sampled hotpot qids are written to
data/subsets/hotpot_generation_<n>.json.

Writes results/generation/<dataset>__<config>.json (config, metrics{em,f1,citation_hit},
per_query{qid: {em, f1, citation_hit, answer, cited, retrieved}}) and
results/generation/human_check.csv (30 random rows, empty faithful(0/1) column, for the team).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

GEN_CONFIGS = ("bm25", "dense", "hybrid_fixed+rr", "qaf+rr")
FINAL_K = 5
RERANKER_NAME = "BAAI/bge-reranker-base"
MODEL = "qwen2.5:7b-instruct"
HUMAN_CHECK_N = 30
_CITE = re.compile(r"\[[\d,\s]+\]")


class _LazyReranker:
    """Loads the real cross-encoder only if the rerank cache lacks a needed (qid, doc) pair."""

    def __init__(self):
        self.name = RERANKER_NAME
        self._real = None

    def rerank(self, query, ranking, corpus, depth=50):
        if self._real is None:
            log("rerank cache miss: loading cross-encoder")
            self._real = common.make_reranker()
        return self._real.rerank(query, ranking, corpus, depth=depth)


def select_qids(ds: str, qids: list[str], n: int) -> list[str]:
    """Seeded sample of n qids; all of them (sorted) when the dataset has <= n."""
    qids = sorted(qids)
    if len(qids) <= n:
        if len(qids) < n:
            log(f"{ds}: only {len(qids)} questions < n={n}; using ALL {len(qids)}")
        return qids
    return sorted(random.Random(common.SEED).sample(qids, n))


def final_context(cfg: str, qid: str, query: str, rk: dict, corpus, rr, a_star: float,
                  qaf, sparse, vocab) -> list[str]:
    d, s = rk["dense"], rk["sparse"]
    if cfg == "bm25":
        ranking = s
    elif cfg == "dense":
        ranking = d
    elif cfg == "hybrid_fixed+rr":
        ranking = rr.rerank(qid, query, common.fuse(d, s, a_star), corpus)
    elif cfg == "qaf+rr":
        a = common.qaf_alpha(qaf, query, d, s, sparse, vocab)
        ranking = rr.rerank(qid, query, common.fuse(d, s, a), corpus)
    else:
        raise ValueError(f"unknown config {cfg!r}")
    return [i for i, _ in ranking[:FINAL_K]]


def score_answer(ans, golds: list[str], rel: dict[str, int]) -> dict:
    from hrag.eval.metrics import exact_match, token_f1

    # "[1][2]" citation markers are not part of the answer: strip them before EM/F1 (the raw
    # text, markers included, is still stored under "answer").
    pred = " ".join(_CITE.sub(" ", ans.text).split())
    return {"em": exact_match(pred, golds), "f1": token_f1(pred, golds),
            "citation_hit": float(any(rel.get(c, 0) > 0 for c in ans.cited)),
            "answer": ans.text, "cited": ans.cited}


def headline() -> str:
    import make_paper_assets as mpa

    R = mpa.load_retrieval(common.RESULTS_DIR)
    enc, how = mpa.headline_encoder(common.RESULTS_DIR, R)
    if enc is None:
        raise RuntimeError("no retrieval results; cannot choose the headline encoder")
    log(f"headline encoder: {enc} (selected by {how} nDCG, never test)")
    return enc


def write_human_check(out_dir: Path, rows: list[dict]) -> None:
    rng = random.Random(common.SEED)
    pick = rng.sample(rows, min(HUMAN_CHECK_N, len(rows)))
    path = out_dir / "human_check.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "config", "answer", "faithful(0/1)"])
        for r in pick:
            w.writerow([r["qid"], r["config"], r["answer"], ""])


def ragas_pass(datasets: list[str], configs: list[str], n: int) -> None:
    """Tier B: add per_query[qid]["faithfulness"] (+ context relevance) to existing generation
    files for the first ``n`` qids (sorted).  Scoring runs in a subprocess of $RAGAS_PYTHON
    (default: this interpreter) because ragas needs older langchain pins than the main venv."""
    import os
    import subprocess
    import tempfile

    py = os.environ.get("RAGAS_PYTHON", sys.executable)
    for ds in datasets:
        corpus, queries, _, _ = common.load_dataset(ds, "test")
        for cfg in configs:
            path = common.RESULTS_DIR / "generation" / f"{ds}__{cfg}.json"
            if not path.exists():
                log(f"ragas: {path.name} missing, skipped")
                continue
            res = json.loads(path.read_text())
            qids = sorted(res["per_query"])[:n]
            samples = [{"question": queries[q], "answer": res["per_query"][q]["answer"],
                        "contexts": [corpus[d].text for d in res["per_query"][q]["retrieved"]]}
                       for q in qids]
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / "in.json", Path(td) / "out.json"
                src.write_text(json.dumps(samples))
                env = {**os.environ, "PYTHONPATH": str(common.ROOT / "src")}
                subprocess.run([py, "-m", "hrag.eval.ragas_eval", str(src), str(dst)], check=True, env=env)
                rows = json.loads(dst.read_text())
            names = sorted({k for r in rows for k in r})
            if len(rows) != len(qids):
                raise RuntimeError(f"ragas returned {len(rows)} rows for {len(qids)} queries")
            for q, r in zip(qids, rows):
                res["per_query"][q].update({k: r.get(k) for k in names})
            means = {}
            for k in names:
                v = [r[k] for r in rows if r.get(k) is not None]
                means[k] = sum(v) / len(v) if v else None
            res["ragas"] = {"n": len(qids), "judge": MODEL, "means": means,
                            "note": "faithfulness NaN/failed samples are null and excluded from the mean"}
            common.write_json(path, res)
            log(f"ragas {ds}/{cfg}: {means}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", nargs="+", default=["hotpot", "rfc"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--configs", nargs="+", default=list(GEN_CONFIGS))
    ap.add_argument("--encoder", default=None, help="override the headline encoder")
    ap.add_argument("--force", action="store_true", help="recompute existing result files")
    ap.add_argument("--ragas", action="store_true",
                    help="only run the RAGAS pass over existing generation files (Tier B)")
    ap.add_argument("--ragas-n", type=int, default=100)
    args = ap.parse_args(argv)
    if args.ragas:
        ragas_pass(args.dataset, args.configs, args.ragas_n)
        return 0

    from hrag.generate import generate_answer

    enc_name = args.encoder or headline()
    out_dir = common.RESULTS_DIR / "generation"
    encoder = common.make_encoder(enc_name)
    vocab = common.encoder_vocab(encoder)
    qaf = common.load_qaf(enc_name)
    a_star = common.alpha_star(enc_name)
    human_rows: list[dict] = []

    for ds in args.dataset:
        if not common.dataset_available(ds):
            continue
        corpus, queries, qrels, answers = common.load_dataset(ds, "test")
        if not answers:
            raise RuntimeError(f"{ds}: dataset provides no gold answers")
        qids = select_qids(ds, [q for q in common.eval_qids(qrels) if q in queries and q in answers],
                           args.n)
        if ds == "hotpot":
            common.write_json(common.SUBSETS_DIR / f"hotpot_generation_{args.n}.json",
                              {"seed": common.SEED, "n": len(qids), "qids": qids})
        log(f"{ds}: generating for {len(qids)} questions, encoder={enc_name}")
        sparse, dense = common.load_indexes(ds, encoder)
        common.check_index_ids(ds, corpus, dense)
        rk = common.get_rankings(ds, "test", enc_name, {q: queries[q] for q in qids}, sparse, dense)
        rr = common.RerankCache(ds, _LazyReranker())

        for cfg in args.configs:
            path = out_dir / f"{ds}__{cfg}.json"
            if path.exists() and not args.force:
                log(f"skip {path.name} (exists)")
                prev = json.loads(path.read_text())["per_query"]
                human_rows += [{"qid": q, "config": cfg, "answer": v["answer"]} for q, v in prev.items()]
                continue
            per_query: dict[str, dict] = {}
            for i, q in enumerate(qids, 1):
                ids = final_context(cfg, q, queries[q], rk[q], corpus, rr, a_star, qaf, sparse, vocab)
                ans = generate_answer(queries[q], [corpus[d] for d in ids])
                per_query[q] = {**score_answer(ans, answers[q], qrels[q]), "retrieved": ids}
                if i % 25 == 0:
                    log(f"{ds}/{cfg}: {i}/{len(qids)}")
            rr.save()
            metrics = {m: sum(v[m] for v in per_query.values()) / len(per_query)
                       for m in ("em", "f1", "citation_hit")}
            common.write_json(path, {
                "config": {"name": cfg, "dataset": ds, "split": "test", "encoder": enc_name,
                           "encoder_selected_by": "pooled dev nDCG@10 (headline_encoder)",
                           "llm": MODEL, "temperature": 0, "seed": 42, "final_k": FINAL_K,
                           "n": len(qids), "n_requested": args.n,
                           "sampling": ("seed 42 sample of evaluated test questions"
                                        if len(qids) == args.n else "all evaluated questions"),
                           "alpha_star": a_star if cfg == "hybrid_fixed+rr" else None},
                "metrics": metrics, "per_query": per_query, "latency_ms": {}})
            log(f"{ds}/{cfg}: EM={metrics['em']:.3f} F1={metrics['f1']:.3f} "
                f"cite={metrics['citation_hit']:.3f}")
            human_rows += [{"qid": q, "config": cfg, "answer": v["answer"]} for q, v in per_query.items()]
    if human_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_human_check(out_dir, human_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
