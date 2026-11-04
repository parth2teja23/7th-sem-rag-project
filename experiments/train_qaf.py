"""Train Query-Adaptive Fusion models and pick the best fixed alpha (alpha*) on pooled dev.

Per encoder:
  * features: query_features(query, dense_top100, sparse_top100, bm25.idf, vocab_from_encoder)
  * target:   oracle_alpha(dense_top100, sparse_top100, dev qrels)
  * pooled model on SciFact-dev + FiQA-dev   -> indexes/qaf_<encoder>.joblib
  * alpha*: argmax over ALPHA_GRID of pooled (query-level micro) mean nDCG@10 of fused
    dev rankings                              -> results/qaf/alpha_star_<encoder>.json
  * coefficients                              -> results/qaf/coefficients_<encoder>.json
  * oracle-target distribution                -> results/qaf/targets_<encoder>.json
  * leave-one-dataset-out: train on one dev set, evaluate on the other dataset's TEST split
                                              -> results/qaf/lodo_<encoder>.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402


def _dev_block(ds: str, enc_name: str, encoder, vocab):
    from hrag.eval.metrics import ndcg_at_k
    from hrag.qaf import ALPHA_GRID, oracle_alpha, query_features

    corpus, queries, qrels, _ = common.load_dataset(ds, "dev")
    sparse, dense = common.load_indexes(ds, encoder)
    common.check_index_ids(ds, corpus, dense)
    rk = common.get_rankings(ds, "dev", enc_name, queries, sparse, dense)
    qids = [q for q in common.eval_qids(qrels) if q in queries]
    X, y, curves = [], [], []
    for q in qids:
        d, s = rk[q]["dense"], rk[q]["sparse"]
        X.append(query_features(queries[q], d, s, sparse.idf, vocab))
        y.append(oracle_alpha(d, s, qrels[q]))
        curves.append([ndcg_at_k(common.fuse(d, s, a), qrels[q], 10) for a in ALPHA_GRID])
    return {"qids": qids, "X": np.asarray(X), "y": np.asarray(y), "curves": np.asarray(curves)}


def _hist(y) -> dict[str, int]:
    c = Counter(round(float(v), 2) for v in y)
    return {f"{k:.2f}": c[k] for k in sorted(c)}


def _target_summary(b) -> dict:
    curves = b["curves"]
    flat = (curves.max(axis=1) - curves.min(axis=1)) < 1e-12
    return {"n": int(len(b["y"])), "histogram": _hist(b["y"]),
            "mean": float(np.mean(b["y"])), "std": float(np.std(b["y"])),
            "n_alpha_insensitive": int(flat.sum()),
            "n_zero_ndcg_all_alphas": int((curves.max(axis=1) <= 0).sum()),
            "note": "n_alpha_insensitive: nDCG@10 identical for every grid alpha (target "
                    "falls back to the plateau centre 0.5)"}


def _curve(curves: np.ndarray) -> dict[float, float]:
    from hrag.qaf import ALPHA_GRID

    m = curves.mean(axis=0)
    return {a: float(v) for a, v in zip(ALPHA_GRID, m)}


def _fit(blocks):
    from hrag.qaf import QAFModel

    X = np.vstack([b["X"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    return QAFModel().fit(X, y), X, y


def _lodo_eval(held: str, enc_name: str, encoder, vocab, models: dict, alphas: dict,
               reranker_box: list) -> dict:
    """Evaluate fixed-alpha and QAF variants on the held-out dataset's TEST split."""
    corpus, queries, qrels, _ = common.load_dataset(held, "test")
    sparse, dense = common.load_indexes(held, encoder)
    common.check_index_ids(held, corpus, dense)
    rk = common.get_rankings(held, "test", enc_name, queries, sparse, dense)
    qids = [q for q in common.eval_qids(qrels) if q in queries]
    if not reranker_box:
        reranker_box.append(common.make_reranker())
    rr = common.RerankCache(held, reranker_box[0])
    out, preds = {}, {}
    variants = [(f"hybrid_fixed@{alphas[k]}[{k}]", "fixed", k) for k in alphas] + \
               [(f"qaf[{k}]", "qaf", k) for k in models]
    for label, kind, k in variants:
        run, run_rr, al = {}, {}, []
        for q in qids:
            d, s = rk[q]["dense"], rk[q]["sparse"]
            a = alphas[k] if kind == "fixed" else common.qaf_alpha(models[k], queries[q], d, s, sparse, vocab)
            al.append(a)
            run[q] = common.fuse(d, s, a)
            run_rr[q] = rr.rerank(q, queries[q], run[q], corpus)
        for suffix, r in (("", run), ("+rr", run_rr)):
            m, _ = common.evaluate(r, {q: qrels[q] for q in qids})
            out[label + suffix] = {x: m[x] for x in ("ndcg@10", "recall@10", "mrr@10")}
        if kind == "qaf":
            preds[label] = {"mean": float(np.mean(al)), "std": float(np.std(al))}
    rr.save()
    return {"n_test": len(qids), "results": out, "predicted_alpha": preds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="+", default=list(common.ENCODER_NAMES))
    args = ap.parse_args(argv)

    from hrag.qaf import FEATURE_NAMES

    reranker_box: list = []
    for enc_name in args.encoders:
        encoder = common.make_encoder(enc_name)
        vocab = common.encoder_vocab(encoder)
        blocks = {}
        for ds in common.DEV_DATASETS:
            if not common.dataset_available(ds):
                continue
            log(f"[{enc_name}] dev features/targets: {ds}")
            blocks[ds] = _dev_block(ds, enc_name, encoder, vocab)
        if not blocks:
            log(f"[{enc_name}] no dev data; nothing to train")
            continue

        # alpha* on pooled dev (query-level micro mean)
        pooled_curves = np.vstack([b["curves"] for b in blocks.values()])
        pooled = _curve(pooled_curves)
        a_star = common.select_alpha(pooled)
        common.write_json(common.alpha_star_path(enc_name), {
            "encoder": enc_name, "alpha_star": a_star,
            "selection": "argmax of pooled (query-level micro-averaged) dev nDCG@10 of "
                         "weighted_rrf(k=60, top_n=100) over ALPHA_GRID; ties -> closest to 0.5",
            "n_dev": {ds: len(b["qids"]) for ds, b in blocks.items()},
            "pooled_dev_ndcg@10": {str(a): v for a, v in pooled.items()},
            "per_dataset_dev_ndcg@10": {ds: {str(a): v for a, v in _curve(b["curves"]).items()}
                                        for ds, b in blocks.items()},
            "per_dataset_best_alpha": {ds: common.select_alpha(_curve(b["curves"]))
                                       for ds, b in blocks.items()},
        })
        log(f"[{enc_name}] alpha* = {a_star}  (pooled dev nDCG@10 {pooled[a_star]:.4f})")

        # oracle target distribution
        common.write_json(common.RESULTS_DIR / "qaf" / f"targets_{enc_name}.json", {
            "encoder": enc_name,
            "pooled": _target_summary({"y": np.concatenate([b["y"] for b in blocks.values()]),
                                       "curves": pooled_curves}),
            **{ds: _target_summary(b) for ds, b in blocks.items()},
        })

        # pooled model
        model, X, y = _fit(list(blocks.values()))
        model.save(common.qaf_path(enc_name))
        pred = model.predict(X)
        ridge = model.pipe.named_steps["ridge"]
        scaler = model.pipe.named_steps["scale"]
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        common.write_json(common.RESULTS_DIR / "qaf" / f"coefficients_{enc_name}.json", {
            "encoder": enc_name, "trained_on": {f"{ds}-dev": len(b["qids"]) for ds, b in blocks.items()},
            "model": "StandardScaler + Ridge(alpha=1.0); predictions clipped to [0, 1]",
            "coefficients": dict(model.coef_table()),
            "coefficients_note": "standardized-feature coefficients (change in predicted alpha "
                                 "per 1 SD of the feature); alpha = weight on DENSE",
            "intercept": float(ridge.intercept_),
            "feature_mean": dict(zip(FEATURE_NAMES, map(float, scaler.mean_))),
            "feature_std": dict(zip(FEATURE_NAMES, map(float, scaler.scale_))),
            "train_r2_clipped": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "train_mae_clipped": float(np.abs(y - pred).mean()),
            "target_mean": float(y.mean()), "target_std": float(y.std()),
            "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        })
        log(f"[{enc_name}] QAF saved -> {common.qaf_path(enc_name)}; "
            f"train R2 {1 - ss_res / ss_tot if ss_tot > 0 else float('nan'):.3f}")

        # leave-one-dataset-out
        if len(blocks) >= 2:
            lodo = {"encoder": enc_name,
                    "protocol": "QAF and fixed alpha fitted on the OTHER dataset's dev split only; "
                                "evaluated on the held-out dataset's test split. '[pooled]' = "
                                "in-domain pooled-dev model / alpha* for reference.",
                    "heldout": {}}
            for held in blocks:
                train = [ds for ds in blocks if ds != held]
                m_lodo, _, _ = _fit([blocks[ds] for ds in train])
                a_lodo = common.select_alpha(_curve(np.vstack([blocks[ds]["curves"] for ds in train])))
                log(f"[{enc_name}] LODO held-out {held} (train {train}, alpha*={a_lodo})")
                res = _lodo_eval(held, enc_name, encoder, vocab,
                                 {"lodo": m_lodo, "pooled": model},
                                 {"lodo": a_lodo, "pooled": a_star}, reranker_box)
                lodo["heldout"][held] = {"trained_on": [f"{t}-dev" for t in train],
                                         "alpha_star_lodo": a_lodo, "alpha_star_pooled": a_star,
                                         "lodo_coefficients": dict(m_lodo.coef_table()), **res}
            common.write_json(common.RESULTS_DIR / "qaf" / f"lodo_{enc_name}.json", lodo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
