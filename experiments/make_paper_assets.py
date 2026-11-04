"""Generate the paper's LaTeX tables, PDF figures and number macros from result JSON files.

    .venv/bin/python experiments/make_paper_assets.py --results results/ --out paper/

Reads the result-file contract of experiments/common.py
(``results/<exp>/<dataset>__[<encoder>__]<config>.json``) and writes
  <out>/tables/{main_results,significance,ablation,generation,latency,qaf_coefficients,
                dataset_stats,matryoshka,dat,pageindex}.tex
  <out>/figures/{alpha_sweep,qaf_alpha_hist,latency_breakdown,matryoshka}.pdf
  <out>/numbers.tex
A table / figure whose inputs are absent is skipped silently; a macro whose input is absent
keeps the ``\\TBD`` placeholder.  ``--dataset-stats`` additionally loads the datasets through
experiments/common.py and (re)writes results/dataset_stats.json and results/qaf/query_types.json.

Statistics conventions: significance is a one-sided paired bootstrap (10,000 resamples, seed 42)
and one-sided paired t-test that the *treatment* beats the *baseline*; "dagger" marks p < 0.05
of the one-sided bootstrap versus hybrid_fixed+rr.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

DATASETS = ("scifact", "fiqa", "hotpot", "rfc")
ENCODERS = ("bge-small", "gemma")
DS_LABEL = {"scifact": "SciFact", "fiqa": "FiQA", "hotpot": "HotpotQA", "rfc": "RFC"}
DS_MACRO = {"scifact": "SciFact", "fiqa": "FiQA", "hotpot": "Hotpot", "rfc": "RFC"}
ENC_LABEL = {"bge-small": "BGE-small", "gemma": "Gemma"}
CONFIGS = ("bm25", "dense", "hybrid_fixed", "hybrid_fixed+rr", "qaf", "qaf+rr", "oracle_upper_bound")
CFG_LABEL = {"bm25": "BM25", "dense": "Dense", "hybrid_fixed": "Hybrid-fixed",
             "hybrid_fixed+rr": "Hybrid-fixed+RR", "qaf": "QAF", "qaf+rr": "QAF+RR",
             "oracle_upper_bound": "Oracle$^{\\ddagger}$"}
REF = "hybrid_fixed+rr"
METRICS = (("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"))
STAGES = ("expand", "sparse", "dense", "features", "fusion", "rerank", "post", "total")
STAGE_LABEL = {"expand": "Expansion", "sparse": "BM25", "dense": "Dense search",
               "features": "$\\alpha$ decision", "fusion": "Fusion", "rerank": "Reranking",
               "post": "Post-processing", "total": "Total"}
ALPHA = 0.05
ID_HEAVY_THRESHOLD = 0.2  # fraction of identifier-like tokens (hrag.qaf id_ratio) -> "identifier-heavy"
DAG = "$^{\\dagger}$"
N_RESAMPLES, SEED = 10000, 42


# ----------------------------------------------------------------------------- loading

def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_retrieval(results: Path) -> dict:
    """{dataset: {encoder|None: {config: result}}}; encoder None = encoder independent (bm25)."""
    out: dict = {}
    d = Path(results) / "retrieval"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        parts = f.stem.split("__")
        if len(parts) not in (2, 3) or parts[0] not in DATASETS:
            continue
        data = _read(f)
        if not isinstance(data, dict) or "metrics" not in data:
            continue
        ds, enc, cfg = (parts[0], None, parts[1]) if len(parts) == 2 else parts
        out.setdefault(ds, {}).setdefault(enc, {})[cfg] = data
    return out


def encoders_of(R: dict) -> list[str]:
    seen = {e for ds in R.values() for e in ds if e}
    return [e for e in ENCODERS if e in seen] + sorted(seen - set(ENCODERS))


def get(R: dict, ds: str, enc: str | None, cfg: str):
    by = R.get(ds, {})
    return by.get(enc, {}).get(cfg) or by.get(None, {}).get(cfg)


def metric(R, ds, enc, cfg, m="ndcg@10"):
    d = get(R, ds, enc, cfg)
    return None if d is None else d["metrics"].get(m)


def headline_encoder(results: Path, R: dict) -> tuple[str | None, str]:
    """Encoder used for every headline number, chosen WITHOUT test results: the one with the
    higher pooled dev nDCG@10 at alpha* (hybrid_fixed, results/qaf/alpha_star_<enc>.json).
    Falls back to bge-small when any encoder lacks that dev score. -> (encoder, how)."""
    encs = encoders_of(R)
    if not encs:
        return None, "none"
    sc = {}
    for e in encs:
        a = _read(Path(results) / "qaf" / f"alpha_star_{e}.json")
        try:
            sc[e] = float(max(a["pooled_dev_ndcg@10"].values()))
        except (TypeError, KeyError, ValueError):
            pass
    if len(sc) == len(encs):
        return max(encs, key=lambda e: (sc[e], -encs.index(e))), "dev"
    return ("bge-small" if "bge-small" in encs else encs[0]), "fixed"


def _pq(data: dict, m: str) -> dict:
    return {q: v[m] for q, v in data.get("per_query", {}).items() if isinstance(v, dict) and v.get(m) is not None}


def paired(a: dict | None, b: dict | None, m: str = "ndcg@10"):
    """Aligned per-query metric lists (a, b), or None unless both runs cover exactly the same
    qids and every value is a finite number (never intersect: a mismatch is a bug upstream)."""
    if a is None or b is None:
        return None
    qa, qb = a.get("per_query") or {}, b.get("per_query") or {}
    if not qa or set(qa) != set(qb):
        return None
    ids = sorted(qa)
    x, y = [], []
    for q in ids:
        va = qa[q].get(m) if isinstance(qa[q], dict) else None
        vb = qb[q].get(m) if isinstance(qb[q], dict) else None
        if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
            return None
        x.append(va)
        y.append(vb)
    return x, y


def compare(a: dict | None, b: dict | None, m: str = "ndcg@10") -> dict | None:
    """Treatment b vs baseline a: delta, 95% bootstrap CI, one-sided p (bootstrap, t-test), n."""
    from hrag.eval.stats import paired_bootstrap, paired_bootstrap_ci, paired_ttest

    pr = paired(a, b, m)
    if pr is None:
        return None
    x, y = pr
    delta, lo, hi = paired_bootstrap_ci(x, y, N_RESAMPLES, SEED)
    p_t = paired_ttest(x, y)
    if np.isnan(p_t):  # (numerically) constant differences: t-test undefined
        p_t = 0.0 if delta > 0 else 1.0
    return {"delta": delta, "lo": lo, "hi": hi,
            "p_boot": paired_bootstrap(x, y, N_RESAMPLES, SEED), "p_t": p_t, "n": len(x)}


# ----------------------------------------------------------------------------- formatting

def tex(s: str) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("&", "\\&")


def f3(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def sgn(x: float, nd: int = 3) -> str:
    return f"${'+' if x >= 0 else '-'}{abs(x):.{nd}f}$"


def fmt_ci(lo: float, hi: float) -> str:
    return f"$[{lo:.3f},\\,{hi:.3f}]$"


def fmt_p(p: float) -> str:
    return "$<\\!0.0001$" if p < 1e-4 else f"{p:.4f}"


def fmt_ms(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def cell(v, best: bool = False, dag: bool = False, fmt=f3) -> str:
    if v is None:
        return "--"
    s = fmt(v)
    if best:
        s = f"\\textbf{{{s}}}"
    return s + (DAG if dag else "")


def best_flags(vals: list, mode: str = "max", tol: float = 1e-12) -> list[bool]:
    ok = [v for v in vals if v is not None]
    if not ok:
        return [False] * len(vals)
    t = max(ok) if mode == "max" else min(ok)
    return [v is not None and abs(v - t) <= tol for v in vals]


def table(name: str, caption: str, colspec: str, header: list[str], body: list[str],
          resize: bool = False, note: str = "") -> str:
    lines = [f"\\begin{{table}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
             "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}"]
    if resize:
        lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule", *body,
              "\\bottomrule", "\\end{tabular}" + ("}" if resize else "")]
    if note:
        lines.append(note)
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s)


# ----------------------------------------------------------------------------- tables

def main_results_table(R: dict) -> str | None:
    encs = encoders_of(R)
    dss = [d for d in DATASETS if d in R]
    if not encs or not dss:
        return None
    ncol = len(METRICS) * len(encs)
    head1 = "Config & " + " & ".join(f"\\multicolumn{{{len(encs)}}}{{c}}{{{lab}}}" for _, lab in METRICS) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + i * len(encs)}-{1 + (i + 1) * len(encs)}}}" for i in range(len(METRICS)))
    head2 = " & " + " & ".join(ENC_LABEL.get(e, tex(e)) for _ in METRICS for e in encs) + " \\\\"
    body: list[str] = []
    for di, ds in enumerate(dss):
        if di:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textit{{{DS_LABEL[ds]}}}}} \\\\")
        cfgs = [c for c in CONFIGS if any(get(R, ds, e, c) for e in encs)]
        # columns of values: for each metric, each encoder
        cols = [(m, e) for m, _ in METRICS for e in encs]
        vals = {c: [metric(R, ds, e, c, m) for m, e in cols] for c in cfgs}
        flags = {}
        for j in range(len(cols)):
            col = [vals[c][j] if c != "oracle_upper_bound" else None for c in cfgs]
            for c, fl in zip(cfgs, best_flags(col)):
                flags[(c, j)] = fl
        for c in cfgs:
            cells = []
            for j, (m, e) in enumerate(cols):
                dag = False
                if c != REF and c != "oracle_upper_bound":
                    pr = paired(get(R, ds, e, REF), get(R, ds, e, c), m)
                    if pr is not None:
                        from hrag.eval.stats import paired_bootstrap
                        dag = paired_bootstrap(pr[0], pr[1], N_RESAMPLES, SEED) < ALPHA
                cells.append(cell(vals[c][j], flags[(c, j)], dag))
            body.append(f"{CFG_LABEL[c]} & " + " & ".join(cells) + " \\\\")
    cap = ("Retrieval effectiveness on the test queries. Best deployable value per column in bold; "
           "$\\dagger$: significantly better than Hybrid-fixed+RR (one-sided paired bootstrap, $p<0.05$). "
           "$\\ddagger$: per-query oracle $\\alpha$ on test qrels, a headroom reference, not a strict upper bound. "
           "BM25 does not depend on the encoder.")
    return table("main_results", cap, "l" + "c" * ncol, [head1, cmid, head2], body, resize=True)


COMPARISONS = (
    ("qaf_rr_vs_fixed_rr", "QAF+RR vs.\\ Hybrid-fixed+RR"),
    ("hybrid_vs_single", "Hybrid-fixed vs.\\ best single retriever"),
    ("rr_vs_norr", "+RR vs.\\ no RR (QAF)"),
    ("gemma_vs_bge", "Gemma vs.\\ BGE (QAF+RR)"),
)


def significance_stats(R: dict) -> dict:
    """{(comparison_key, dataset, encoder): {delta, lo, hi, p_boot, p_t, n}}; comparisons 1-3 are
    computed for every encoder, the encoder comparison has encoder None."""
    out: dict = {}
    for ds in DATASETS:
        if ds not in R:
            continue
        g = lambda enc, cfg: get(R, ds, enc, cfg)  # noqa: E731
        trials = {}
        for e in encoders_of(R):
            trials[("qaf_rr_vs_fixed_rr", e)] = (g(e, REF), g(e, "qaf+rr"))
            single = [(metric(R, ds, e, c), c) for c in ("dense", "bm25")]
            single = [(v, c) for v, c in single if v is not None]
            if single:
                trials[("hybrid_vs_single", e)] = (g(e, max(single)[1]), g(e, "hybrid_fixed"))
            trials[("rr_vs_norr", e)] = (g(e, "qaf"), g(e, "qaf+rr"))
        trials[("gemma_vs_bge", None)] = (g("bge-small", "qaf+rr"), g("gemma", "qaf+rr"))
        for (key, enc), (a, b) in trials.items():
            r = compare(a, b)
            if r:
                out[(key, ds, enc)] = r
    return out


def _head_note(head: tuple | None) -> str:
    if not head or not head[0]:
        return ""
    enc = ENC_LABEL.get(head[0], tex(head[0]))
    if head[1] == "dev":
        return (f"Headline numbers in the text use {enc}, chosen by pooled dev nDCG@10 of Hybrid-fixed "
                "(not on test results).")
    return f"Headline numbers in the text use {enc}, fixed a priori (no dev score available)."


def significance_table(R: dict, head: tuple | None = None) -> str | None:
    sig = significance_stats(R)
    if not sig:
        return None
    body: list[str] = []
    first = True
    for key, label in COMPARISONS:
        rows = [(ds, e, sig[(key, ds, e)]) for ds in DATASETS for e in [*encoders_of(R), None]
                if (key, ds, e) in sig]
        if not rows:
            continue
        if not first:
            body.append("\\midrule")
        first = False
        body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for ds, e, r in rows:
            enc = f" ({ENC_LABEL.get(e, tex(e))})" if e else ""
            d = sgn(r["delta"]) + (DAG if r["p_boot"] < ALPHA else "")
            body.append(f"{DS_LABEL[ds]}{enc} & {d} & {fmt_ci(r['lo'], r['hi'])} & "
                        f"{fmt_p(r['p_boot'])} & {fmt_p(r['p_t'])} & {r['n']} \\\\")
    header = ["Dataset & $\\Delta$nDCG@10 & 95\\% CI & $p$ (boot.) & $p$ ($t$-test) & $n$ \\\\"]
    cap = ("Paired significance of the main comparisons (treatment minus baseline, nDCG@10). CI: 95\\% "
           "paired-bootstrap interval (10{,}000 resamples, seed 42); $p$: one-sided, treatment better than "
           "baseline, floored at $10^{-4}$ by the 10{,}000 resamples. $\\dagger$: bootstrap $p<0.05$. "
           "Comparisons 1--3 are shown for each encoder. " + _head_note(head))
    return table("significance", cap, "lccccr", header, body, resize=True)


def _ablation_key(name: str) -> str | None:
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    if n in ("full", "fullsystem"):
        return "full"
    for k, sub in (("chunking", "chunk"), ("expansion", "expan"), ("qaf", "qaf"), ("rerank", "rerank")):
        if sub in n:
            return k
    return None


ABL_ORDER = ("full", "chunking", "expansion", "qaf", "rerank")
ABL_LABEL = {"full": "Full", "chunking": "$-$chunking", "expansion": "$-$expansion", "qaf": "$-$QAF",
             "rerank": "$-$rerank"}


def _load_exp(results: Path, exp: str) -> dict:
    """{dataset: {config_name: result}} from results/<exp>/<dataset>__[<enc>__]<config>.json."""
    out: dict = {}
    d = Path(results) / exp
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        parts = f.stem.split("__")
        if len(parts) < 2:
            continue
        data = _read(f)
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            out.setdefault(parts[0], {})[parts[-1]] = data
    return out


def ablation_table(results: Path) -> str | None:
    A = _load_exp(results, "ablation")
    dss = [d for d in DATASETS if d in A]
    rows: dict = {}
    for ds in dss:
        for name, data in A[ds].items():
            k = _ablation_key(name)
            if k:
                rows.setdefault(k, {})[ds] = data
    keys = [k for k in ABL_ORDER if k in rows]
    if not dss or not keys:
        return None
    ms = (("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"))
    cols = [(ds, m) for ds in dss for m, _ in ms]
    vals = {k: [(rows[k].get(ds) or {}).get("metrics", {}).get(m) for ds, m in cols] for k in keys}
    flags = [best_flags([vals[k][j] for k in keys]) for j in range(len(cols))]
    head1 = "Variant & " + " & ".join(f"\\multicolumn{{{len(ms)}}}{{c}}{{{DS_LABEL[d]}}}" for d in dss) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + i * len(ms)}-{1 + (i + 1) * len(ms)}}}" for i in range(len(dss)))
    head2 = " & " + " & ".join(lab for _ in dss for _, lab in ms) + " \\\\"
    body = [f"{ABL_LABEL[k]} & " + " & ".join(cell(vals[k][j], flags[j][i]) for j in range(len(cols))) + " \\\\"
            for i, k in enumerate(keys)]
    cap = "Ablation of the full system (better encoder); best value per column in bold."
    return table("ablation", cap, "l" + "c" * len(cols), [head1, cmid, head2], body, resize=True)


GEN_CFGS = ("bm25", "dense", "hybrid_fixed+rr", "qaf+rr")
GEN_METRICS = (("em", "EM"), ("f1", "F1"), ("citation_hit", "Cite"))


def generation_table(results: Path) -> str | None:
    G = _load_exp(results, "generation")
    dss = [d for d in DATASETS if d in G]
    cfgs = [c for c in GEN_CFGS if any(c in G[d] for d in dss)]
    if not dss or not cfgs:
        return None
    cols = [(ds, m) for ds in dss for m, _ in GEN_METRICS]
    vals = {c: [(G[ds].get(c) or {}).get("metrics", {}).get(m) for ds, m in cols] for c in cfgs}
    flags = [best_flags([vals[c][j] for c in cfgs]) for j in range(len(cols))]
    head1 = "Config & " + " & ".join(f"\\multicolumn{{{len(GEN_METRICS)}}}{{c}}{{{DS_LABEL[d]}}}" for d in dss) + " \\\\"
    nm = len(GEN_METRICS)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + i * nm}-{1 + (i + 1) * nm}}}" for i in range(len(dss)))
    head2 = " & " + " & ".join(lab for _ in dss for _, lab in GEN_METRICS) + " \\\\"
    body = []
    for i, c in enumerate(cfgs):
        cells = []
        for j, (ds, m) in enumerate(cols):
            dag = False
            if c != REF and REF in G[ds] and c in G[ds]:
                pr = paired(G[ds][REF], G[ds][c], m)
                if pr is not None:
                    from hrag.eval.stats import paired_bootstrap
                    dag = paired_bootstrap(pr[0], pr[1], N_RESAMPLES, SEED) < ALPHA
            cells.append(cell(vals[c][j], flags[j][i], dag))
        body.append(f"{CFG_LABEL[c]} & " + " & ".join(cells) + " \\\\")
    cap = ("Answer quality. EM/F1 against the gold answer; Cite: a cited passage is relevant. Best per "
           "column in bold; $\\dagger$: significantly better than Hybrid-fixed+RR (one-sided paired bootstrap, "
           "$p<0.05$).")
    return table("generation", cap, "l" + "c" * len(cols), [head1, cmid, head2], body, resize=True)


def latency_stats(R: dict, enc: str | None) -> dict:
    """{config: {stage: (mean_ms, p95_ms)}} for one encoder, averaged over the datasets on which
    ALL compared configs have latency (a common dataset set, so configs stay comparable)."""
    cfgs = ("dense", REF, "qaf+rr")
    have = {c: {ds for ds in R if (get(R, ds, enc, c) or {}).get("latency_ms")} for c in cfgs}
    present = [c for c in cfgs if have[c]]
    if not present:
        return {}
    common = set.intersection(*(have[c] for c in present))
    out: dict = {}
    for cfg in present:
        acc: dict = {}
        for ds in common:
            for st, v in get(R, ds, enc, cfg)["latency_ms"].items():
                if isinstance(v, dict) and "mean" in v:
                    acc.setdefault(st, []).append((v["mean"], v.get("p95", v["mean"])))
        if acc:
            out[cfg] = {st: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
                        for st, v in acc.items()}
    return out


def latency_table(R: dict, enc: str | None) -> str | None:
    L = latency_stats(R, enc)
    cfgs = [c for c in ("dense", REF, "qaf+rr") if c in L]
    if not cfgs:
        return None
    stages = [s for s in STAGES if any(s in L[c] and (s == "total" or L[c][s][0] > 0) for c in cfgs)]
    head1 = "Stage & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{CFG_LABEL[c]}}}" for c in cfgs) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cfgs)))
    head2 = " & " + " & ".join("mean & p95" for _ in cfgs) + " \\\\"
    body = []
    for s in stages:
        means = [L[c][s][0] if s in L[c] else None for c in cfgs]
        fl = best_flags(means, "min") if s == "total" else [False] * len(cfgs)
        cells = []
        for c, b in zip(cfgs, fl):
            if s in L[c]:
                cells += [cell(L[c][s][0], b, fmt=fmt_ms), cell(L[c][s][1], b, fmt=fmt_ms)]
            else:
                cells += ["--", "--"]
        body.append(f"{STAGE_LABEL[s]} & " + " & ".join(cells) + " \\\\")
    cap = (f"Per-stage latency in ms (mean and 95th percentile over sampled test queries, averaged over "
           f"collections; encoder: {ENC_LABEL.get(enc, enc)}). Lowest total in bold.")
    return table("latency", cap, "l" + "cc" * len(cfgs), [head1, cmid, head2], body)


def qaf_coef_table(results: Path) -> str | None:
    C = {}
    for e in ENCODERS:
        d = _read(Path(results) / "qaf" / f"coefficients_{e}.json")
        if isinstance(d, dict) and isinstance(d.get("coefficients"), dict):
            C[e] = d
    if not C:
        return None
    encs = list(C)
    feats = list(next(iter(C.values()))["coefficients"])
    for e in encs[1:]:
        feats += [f for f in C[e]["coefficients"] if f not in feats]
    best = {e: max(C[e]["coefficients"].items(), key=lambda kv: abs(kv[1]))[0] for e in encs}
    body = []
    for f in feats:
        cells = []
        for e in encs:
            v = C[e]["coefficients"].get(f)
            cells.append(cell(v, best[e] == f, fmt=lambda x: f"{x:+.3f}"))
        body.append(f"\\texttt{{{tex(f)}}} & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")
    body.append("Train $R^2$ & " + " & ".join(f3(C[e].get("train_r2_clipped")) for e in encs) + " \\\\")
    body.append("Intercept & " + " & ".join(f3(C[e].get("intercept")) for e in encs) + " \\\\")
    head = ["Feature & " + " & ".join(ENC_LABEL.get(e, tex(e)) for e in encs) + " \\\\"]
    cap = ("Standardised ridge coefficients of the QAF model (change in predicted dense weight $\\alpha$ per "
           "1 SD of the feature); largest $|$coefficient$|$ per encoder in bold.")
    return table("qaf_coefficients", cap, "l" + "c" * len(encs), head, body)


def dataset_stats_table(results: Path) -> str | None:
    S = _read(Path(results) / "dataset_stats.json")
    if not isinstance(S, dict):
        return None
    dss = [d for d in DATASETS if d in S]
    if not dss:
        return None
    body = []
    for ds in dss:
        s = S[ds]
        nd = s.get("n_dev_queries")
        body.append(f"{DS_LABEL[ds]} & {f'{s['mean_doc_words']:.1f}' if s.get('mean_doc_words') is not None else '--'} & "
                    f"{f'{s['mean_query_words']:.1f}' if s.get('mean_query_words') is not None else '--'} & "
                    f"{f'{s['mean_rel_per_query']:.2f}' if s.get('mean_rel_per_query') is not None else '--'} & "
                    f"{nd if nd is not None else '--'} \\\\")
    head = ["Dataset & Doc.\\ words & Query words & Rel./query & Dev queries \\\\"]
    cap = "Collection statistics (mean lengths in whitespace-separated words; relevant passages per evaluated test query)."
    return table("dataset_stats", cap, "lcccc", head, body)


def _dim_of(name: str, data: dict) -> int | None:
    d = (data.get("config") or {}).get("dim") or data.get("dim")
    if d:
        return int(d)
    m = re.search(r"(?:^|[_\-])(\d{2,4})(?:$|[_\-])", name)
    return int(m.group(1)) if m else None


def load_matryoshka(results: Path) -> list[dict]:
    """[{dataset, dim, config, ndcg, index_mb, query_ms}] from results/matryoshka/*.json."""
    d = Path(results) / "matryoshka"
    rows = []
    if not d.is_dir():
        return rows
    for f in sorted(d.glob("*.json")):
        data = _read(f)
        parts = f.stem.split("__")
        if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict) or parts[0] not in DATASETS:
            continue
        dim = _dim_of("_".join(parts[1:]), data)
        cfg = (data.get("config") or {}).get("name")
        if cfg not in ("dense", "qaf+rr"):
            cfg = parts[-1]
        if dim is None or cfg not in ("dense", "qaf+rr"):
            continue
        mb = data.get("index_mb", data.get("index_size_mb", (data.get("config") or {}).get("index_mb")))
        lat = (data.get("latency_ms") or {}).get("total", {})
        rows.append({"dataset": parts[0], "dim": dim, "config": cfg, "ndcg": data["metrics"].get("ndcg@10"),
                     "index_mb": mb, "query_ms": lat.get("mean") if isinstance(lat, dict) else None})
    return rows


def matryoshka_table(results: Path) -> str | None:
    rows = load_matryoshka(results)
    if not rows:
        return None
    dims = sorted({r["dim"] for r in rows}, reverse=True)
    pairs = sorted({(r["dataset"], r["config"]) for r in rows},
                   key=lambda p: (DATASETS.index(p[0]), p[1]))
    vals = {(dm, p): next((r["ndcg"] for r in rows if r["dim"] == dm and (r["dataset"], r["config"]) == p), None)
            for dm in dims for p in pairs}
    flags = {p: dict(zip(dims, best_flags([vals[(dm, p)] for dm in dims]))) for p in pairs}

    def avg(dm, key):
        v = [r[key] for r in rows if r["dim"] == dm and r[key] is not None]
        return float(np.mean(v)) if v else None

    body = []
    for dm in dims:
        cells = [cell(vals[(dm, p)], flags[p][dm]) for p in pairs]
        cells += [cell(avg(dm, "index_mb"), fmt=lambda x: f"{x:.1f}"), cell(avg(dm, "query_ms"), fmt=fmt_ms)]
        body.append(f"{dm} & " + " & ".join(cells) + " \\\\")
    head = ["Dims & " + " & ".join(f"{DS_LABEL[d]} {CFG_LABEL.get(c, tex(c))}" for d, c in pairs)
            + " & Index MB & Query ms \\\\"]
    cap = "EmbeddingGemma 2 with Matryoshka truncation: nDCG@10 (best per column in bold), mean dense-index size (FAISS HNSW file) and query latency."
    return table("matryoshka", cap, "l" + "c" * (len(pairs) + 2), head, body, resize=True)


def dat_table(results: Path) -> str | None:
    D = _load_exp(results, "dat")
    dss = [d for d in DATASETS if d in D]
    if not dss:
        return None
    rows = [(ds, c, D[ds][c]) for ds in dss for c in sorted(D[ds])]
    cols = [m for m, _ in METRICS]
    vals = [[r[2]["metrics"].get(m) for m in cols] + [((r[2].get("latency_ms") or {}).get("total") or {}).get("mean")]
            for r in rows]
    flags = [best_flags([v[j] for v in vals]) for j in range(len(cols))]
    body = []
    for i, (ds, c, _) in enumerate(rows):
        cells = [cell(vals[i][j], flags[j][i]) for j in range(len(cols))] + [cell(vals[i][-1], fmt=fmt_ms)]
        body.append(f"{DS_LABEL[ds]} & {tex(c.upper())} & " + " & ".join(cells) + " \\\\")
    head = ["Dataset & Method & " + " & ".join(lab for _, lab in METRICS) + " & Latency ms \\\\"]
    cap = "LLM-based dynamic alpha tuning (DAT): effectiveness and per-query latency; best value per column in bold."
    return table("dat", cap, "llcccc", head, body)


def pageindex_table(results: Path) -> str | None:
    d = Path(results) / "pageindex"
    rows = []
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            data = _read(f)
            if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
                rows.append((f.stem, data))
    if not rows:
        return None
    keys = [("recall@5", "Recall@5"), ("f1", "Answer F1")]
    vals = [[data["metrics"].get(k) for k, _ in keys] + [((data.get("latency_ms") or {}).get("total") or {}).get("mean")]
            for _, data in rows]
    flags = [best_flags([v[j] for v in vals]) for j in range(len(keys))]
    body = []
    for i, (name, _) in enumerate(rows):
        cells = [cell(vals[i][j], flags[j][i]) for j in range(len(keys))] + [cell(vals[i][-1], fmt=fmt_ms)]
        body.append(f"{tex(name.split('__')[0].upper())} & " + " & ".join(cells) + " \\\\")
    head = ["Collection & " + " & ".join(lab for _, lab in keys) + " & Latency ms \\\\"]
    cap = "PageIndex-style reasoning-based retrieval; best value per column in bold."
    return table("pageindex", cap, "lccc", head, body)


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif",
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 100})
    return plt


GRAYS = ("0.0", "0.35", "0.55", "0.7")
STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")


def fig_alpha_sweep(results: Path, R: dict, out: Path) -> bool:
    enc = headline_encoder(results, R)[0]
    d = Path(results) / "alpha_sweep"
    if not d.is_dir():
        return False
    curves = {}
    for ds in DATASETS:
        cands = sorted(d.glob(f"{ds}__*.json"))
        pick = [f for f in cands if enc and f.stem == f"{ds}__{enc}"] or cands
        if pick:
            j = _read(pick[0])
            if isinstance(j, dict) and isinstance(j.get("test"), dict):
                curves[ds] = j
    if not curves:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, (ds, j) in enumerate(curves.items()):
        xs = sorted(float(a) for a in j["test"])
        ax.plot(xs, [j["test"][str(a)] if str(a) in j["test"] else j["test"][next(k for k in j["test"] if float(k) == a)]
                     for a in xs], color=GRAYS[i % 4], linestyle=STYLES[i % 4], marker=MARKERS[i % 4],
                markersize=3, label=DS_LABEL[ds])
    ax.set_xlabel(r"dense weight $\alpha$ (0 = BM25 only, 1 = dense only)")
    ax.set_ylabel("nDCG@10")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False, ncol=2)
    _save(fig, out / "figures" / "alpha_sweep.pdf")
    plt.close(fig)
    return True


def query_types(results: Path, ds: str, per_query: dict) -> dict[str, bool]:
    """{qid: is_identifier_heavy} from per_query["query_type"/"id_ratio"] or results/qaf/query_types.json."""
    out: dict = {}
    cache = _read(Path(results) / "qaf" / "query_types.json") or {}
    for q, v in per_query.items():
        if isinstance(v, dict) and v.get("query_type") is not None:
            out[q] = str(v["query_type"]).lower().startswith("id")
        elif isinstance(v, dict) and v.get("id_ratio") is not None:
            out[q] = float(v["id_ratio"]) >= ID_HEAVY_THRESHOLD
        elif isinstance(cache.get(ds), dict) and q in cache[ds]:
            out[q] = float(cache[ds][q]) >= ID_HEAVY_THRESHOLD
    return out


def predicted_alphas(results: Path, R: dict):
    """-> (id_heavy_alphas, natural_alphas, grouping) with grouping 'query' or 'dataset'."""
    enc = headline_encoder(results, R)[0]
    idh, nat, ds_alpha = [], [], {}
    typed = False
    for ds in DATASETS:
        data = get(R, ds, enc, "qaf")
        if not data:
            continue
        pq = data.get("per_query", {})
        al = {q: v["alpha"] for q, v in pq.items() if isinstance(v, dict) and v.get("alpha") is not None}
        if not al:
            continue
        ds_alpha[ds] = list(al.values())
        t = query_types(results, ds, pq)
        for q, a in al.items():
            if q in t:
                typed = True
                (idh if t[q] else nat).append(a)
    if typed:
        return idh, nat, "query"
    if "rfc" in ds_alpha and len(ds_alpha) > 1:  # fallback: RFC is the identifier-heavy collection
        return ds_alpha["rfc"], [a for d, v in ds_alpha.items() if d != "rfc" for a in v], "dataset"
    return [], [], "none"


def fig_qaf_hist(results: Path, R: dict, out: Path) -> bool:
    idh, nat, grouping = predicted_alphas(results, R)
    if not idh or not nat:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bins = np.linspace(0, 1, 11)
    lab = ("identifier-heavy", "natural-language") if grouping == "query" else ("RFC (identifier-heavy)", "other collections")
    ax.hist(nat, bins=bins, weights=np.full(len(nat), 1 / len(nat)), color="0.75", edgecolor="0.3",
            hatch="//", linewidth=0.5, label=f"{lab[1]} (n={len(nat)})", alpha=0.9)
    ax.hist(idh, bins=bins, weights=np.full(len(idh), 1 / len(idh)), histtype="step", color="black",
            linewidth=1.3, label=f"{lab[0]} (n={len(idh)})")
    ax.set_xlabel(r"predicted dense weight $\alpha$")
    ax.set_ylabel("fraction of queries")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    _save(fig, out / "figures" / "qaf_alpha_hist.pdf")
    plt.close(fig)
    return True


def fig_latency(R: dict, enc: str | None, out: Path) -> bool:
    L = latency_stats(R, enc)
    cfgs = [c for c in ("dense", REF, "qaf+rr") if c in L]
    if not cfgs:
        return False
    stages = [s for s in STAGES[:-1] if any(L[c].get(s, (0, 0))[0] > 0 for c in cfgs)]
    if not stages:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    shades = np.linspace(0.15, 0.9, len(stages))
    hatches = ["", "//", "\\\\", "xx", "..", "--", "++"]
    left = np.zeros(len(cfgs))
    for i, s in enumerate(stages):
        v = np.array([L[c].get(s, (0.0, 0.0))[0] for c in cfgs])
        ax.barh(range(len(cfgs)), v, left=left, color=str(shades[i]), edgecolor="black", linewidth=0.4,
                hatch=hatches[i % len(hatches)], label=STAGE_LABEL[s].replace("$\\alpha$", "alpha"))
        left += v
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([CFG_LABEL[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlabel("mean latency per query (ms)")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, out / "figures" / "latency_breakdown.pdf")
    plt.close(fig)
    return True


def fig_matryoshka(results: Path, out: Path) -> bool:
    rows = [r for r in load_matryoshka(results) if r["ndcg"] is not None]
    if not rows:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    pairs = sorted({(r["dataset"], r["config"]) for r in rows}, key=lambda p: (DATASETS.index(p[0]), p[1]))
    for i, p in enumerate(pairs):
        pts = sorted((r["dim"], r["ndcg"]) for r in rows if (r["dataset"], r["config"]) == p)
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=GRAYS[i % 4], linestyle=STYLES[i % 4],
                marker=MARKERS[i % 4], markersize=3, label=f"{DS_LABEL[p[0]]} {CFG_LABEL.get(p[1], p[1])}".replace("$^{\\ddagger}$", ""))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["dim"] for r in rows}))
    ax.set_xticklabels([str(d) for d in sorted({r["dim"] for r in rows})])
    ax.set_xlabel("embedding dimensions")
    ax.set_ylabel("nDCG@10")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False, fontsize=6)
    _save(fig, out / "figures" / "matryoshka.pdf")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- macros

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _range(vals: list[float], nd: int) -> str | None:
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return a if a == b else f"{a}--{b}"


def compute_macros(results: Path, R: dict) -> dict[str, str]:
    """{macro name: LaTeX value}; only macros whose inputs exist."""
    M: dict[str, str] = {}
    results = Path(results)
    H = headline_encoder(results, R)[0] if R else None
    gen_enc = H

    # alpha* and QAF training
    for key, enc in (("Bge", "bge-small"), ("Gemma", "gemma")):
        a = _read(results / "qaf" / f"alpha_star_{enc}.json")
        if isinstance(a, dict) and a.get("alpha_star") is not None:
            M[f"AlphaStar{key}"] = f"{a['alpha_star']:.1f}"
    pick = H if H and (results / "qaf" / f"coefficients_{H}.json").exists() else None
    pick = pick or next((e for e in ENCODERS if (results / "qaf" / f"coefficients_{e}.json").exists()), None)
    c = _read(results / "qaf" / f"coefficients_{pick}.json") if pick else None
    if isinstance(c, dict):
        td = c.get("trained_on")
        if isinstance(td, dict) and td:
            M["QAFnDev"] = f"{sum(td.values()):,}".replace(",", "{,}")
        if c.get("train_r2_clipped") is not None:
            M["QAFtrainRsq"] = f"{c['train_r2_clipped']:.2f}"
        if isinstance(c.get("coefficients"), dict) and c["coefficients"]:
            top = max(c["coefficients"].items(), key=lambda kv: abs(kv[1]))[0]
            M["QAFtopFeature"] = f"\\texttt{{{tex(top)}}}"
    tg = _read(results / "qaf" / f"targets_{pick}.json") if pick else None
    pooled = tg.get("pooled") if isinstance(tg, dict) else None
    if isinstance(pooled, dict) and pooled.get("n") and pooled.get("n_alpha_insensitive") is not None:
        M["QAFinsensitivePct"] = f"{100 * pooled['n_alpha_insensitive'] / pooled['n']:.0f}"
    # predicted alpha spread of the qaf config on test, headline encoder, pooled over datasets
    if H:
        alphas = []
        for ds in DATASETS:
            d = get(R, ds, H, "qaf")
            pq = d.get("per_query") if isinstance(d, dict) else None
            if isinstance(pq, dict):
                alphas += [v["alpha"] for v in pq.values() if isinstance(v, dict) and v.get("alpha") is not None]
        if len(alphas) > 1:
            M["QAFpredMean"] = f"{float(np.mean(alphas)):.2f}"
            M["QAFpredSd"] = f"{float(np.std(alphas, ddof=1)):.2f}"

    # latency (headline encoder, common dataset set)
    lat = latency_stats(R, H) if R else {}
    if "qaf+rr" in lat:
        if "rerank" in lat["qaf+rr"]:
            M["RerankLatencyMs"] = fmt_ms(lat["qaf+rr"]["rerank"][0])
        if "total" in lat["qaf+rr"]:
            M["TotalLatencyMs"] = fmt_ms(lat["qaf+rr"]["total"][0])
    # added alpha-decision latency: only on datasets where qaf+rr has the stage AND the fixed
    # baseline has latency recorded.  The pipeline leaves "features" at 0 for fixed alpha and
    # common.latency_stats drops all-zero stages, so a recorded baseline without the stage = 0.
    diffs = []
    for ds in R:
        q, f = get(R, ds, H, "qaf+rr"), get(R, ds, H, REF)
        ql, fl = (q or {}).get("latency_ms") or {}, (f or {}).get("latency_ms") or {}
        if "features" in ql and fl:
            diffs.append(ql["features"]["mean"] - fl.get("features", {"mean": 0.0})["mean"])
    if diffs:
        M["QAFlatencyMs"] = f"{max(float(np.mean(diffs)), 0.0):.1f}"

    sig = significance_stats(R) if R else {}
    for ds in DATASETS:
        k = DS_MACRO[ds]
        s_ = sig.get(("qaf_rr_vs_fixed_rr", ds, H))
        if s_:
            M[f"QAFgain{k}"] = sgn(s_["delta"])
            M[f"QAFp{k}"] = fmt_p(s_["p_boot"])
            M[f"QAFgainCI{k}"] = fmt_ci(s_["lo"], s_["hi"])
        data = get(R, ds, H, "qaf+rr") if ds in R else None
        if data:
            v = [x.get("ndcg@10") for x in (data.get("per_query") or {}).values() if isinstance(x, dict)]
            if len(v) > 1 and all(x is not None and np.isfinite(x) for x in v):
                M[f"NDCGsd{k}"] = f"{float(np.std(v, ddof=1)):.3f}"
    for ds in ("rfc",):
        if ds in R:
            h, d, b = (metric(R, ds, H, x) for x in ("hybrid_fixed", "dense", "bm25"))
            if h is not None and d is not None:
                M["HybridGainOverDenseRFC"] = sgn(h - d)
            if h is not None and b is not None:
                M["HybridGainOverBMRFC"] = sgn(h - b)
    main_cfgs = [c for c in CONFIGS if c != "oracle_upper_bound"]
    for ds, k in (("rfc", "RFC"), ("scifact", "SciFact")):
        # every main config must be present for every encoder of the dataset, else TBD
        allv = [[metric(R, ds, e, cfg) for cfg in main_cfgs] for e in encoders_of(R)]
        if allv and all(v is not None for row in allv for v in row):
            M[f"BestNDCG{k}"] = f"{max(v for row in allv for v in row):.3f}"
        if ds in R:
            o, h = metric(R, ds, H, "oracle_upper_bound"), metric(R, ds, H, "hybrid_fixed")
            if o is not None and h is not None:
                M[f"OracleHeadroom{k}"] = sgn(o - h)
    bm = metric(R, "scifact", None, "bm25") if "scifact" in R else None
    if bm is not None:
        M["BMndcgSciFact"] = f"{bm:.3f}"
    s_ = sig.get(("rr_vs_norr", "scifact", H))
    if s_:
        M["RerankGainSciFact"] = sgn(s_["delta"])

    # Matryoshka
    mrows = load_matryoshka(results)
    if mrows:
        # dense-only: truncation affects the embedding, so measure it without fusion/rerank masking it
        cfg = "dense" if any(r["config"] == "dense" for r in mrows) else "qaf+rr"
        drops = []
        for ds in {r["dataset"] for r in mrows}:
            at = {r["dim"]: r["ndcg"] for r in mrows if r["dataset"] == ds and r["config"] == cfg and r["ndcg"] is not None}
            if 768 in at and 256 in at:
                drops.append(at[768] - at[256])
        if drops:
            M["MatryoshkaDropTwoFiftySix"] = f"{float(np.mean(drops)):.3f}"
        mb = [r for r in mrows if r["dim"] == 128 and r["index_mb"] is not None]
        mb = [r for r in mb if r["dataset"] == "scifact"] or mb
        if mb:
            M["MatryoshkaIndexMBOneTwentyEight"] = f"{float(np.mean([r['index_mb'] for r in mb])):.1f}"

    # build times
    bt = _read(results / "index_build_times.json")
    if isinstance(bt, dict):
        def col(enc, key):
            return [v[enc][key] for v in bt.values() if isinstance(v, dict) and isinstance(v.get(enc), dict) and key in v[enc]]
        r = _range(col("gemma", "docs_per_s"), 1)
        if r:
            M["GemmaDocsPerSec"] = r
        r = _range(col("bge-small", "docs_per_s"), 0)
        if r:
            M["BGEDocsPerSec"] = r
        sec = ((bt.get("fiqa") or {}).get("gemma") or {}).get("seconds")
        if sec is not None:
            M["FiQAEncodeMinGemma"] = f"{sec / 60:.0f}"

    # generation
    G = _load_exp(results, "generation")
    if G:
        n = [len(d.get("per_query") or {}) for ds in G.values() for d in ds.values()]
        n = [x for x in n if x]
        if n:
            M["GenN"] = str(max(n))
        fa: dict = {}
        cnt = []
        for ds in G.values():
            for cfg, d in ds.items():
                v = [q["faithfulness"] for q in (d.get("per_query") or {}).values()
                     if isinstance(q, dict) and q.get("faithfulness") is not None]
                if v:
                    fa.setdefault(cfg, []).extend(v)
                    cnt.append(len(v))
        if cnt:
            M["RagasN"] = str(max(cnt))
        for cfg, k in (("qaf+rr", "QAF"), ("dense", "Dense")):
            if cfg in fa:
                M[f"Faith{k}"] = f"{float(np.mean(fa[cfg])):.2f}"
        both = [ds for ds in G.values() if "qaf+rr" in ds and "dense" in ds]  # same datasets for both
        if both:
            def avg(cfg, m):
                v = [ds[cfg]["metrics"].get(m) for ds in both]
                return float(np.mean(v)) if all(x is not None for x in v) else None
            for cfg, k in (("qaf+rr", "QAF"), ("dense", "Dense")):
                f1 = avg(cfg, "f1")
                if f1 is not None:
                    M[f"GenFone{k}"] = f"{f1:.3f}"
            em = avg("qaf+rr", "em")
            if em is not None:
                M["GenEMQAF"] = f"{em:.3f}"
    return M


MACRO_NAMES = (
    "AlphaStarBge AlphaStarGemma QAFnDev QAFtrainRsq QAFtopFeature QAFlatencyMs QAFinsensitivePct QAFpredMean QAFpredSd "
    "QAFgainSciFact QAFgainFiQA QAFgainHotpot QAFgainRFC QAFpSciFact QAFpFiQA QAFpHotpot QAFpRFC "
    "QAFgainCISciFact QAFgainCIFiQA QAFgainCIHotpot QAFgainCIRFC NDCGsdSciFact NDCGsdFiQA NDCGsdHotpot NDCGsdRFC "
    "HybridGainOverDenseRFC HybridGainOverBMRFC BestNDCGRFC BestNDCGSciFact BMndcgSciFact "
    "OracleHeadroomSciFact OracleHeadroomRFC RerankGainSciFact RerankLatencyMs TotalLatencyMs "
    "MatryoshkaDropTwoFiftySix MatryoshkaIndexMBOneTwentyEight GemmaDocsPerSec BGEDocsPerSec FiQAEncodeMinGemma "
    "GenN RagasN GenFoneQAF GenFoneDense GenEMQAF FaithQAF FaithDense").split()

HEADER = ("% Number macros. Generated by experiments/make_paper_assets.py from results/; do not edit by hand.\n"
          "% A macro whose input result file is missing stays \\TBD.\n"
          "\\newcommand{\\TBD}{\\textbf{[TBD]}}\n")
_NEWCMD = re.compile(r"\\newcommand\{\\(\w+)\}\{")


def _split_newcommand(line: str):
    """(name, value, trailing) for ``\\newcommand{\\Name}{value}trailing`` (brace matched), else None."""
    m = _NEWCMD.match(line)
    if not m:
        return None
    depth, i = 1, m.end()
    while i < len(line) and depth:
        depth += {"{": 1, "}": -1}.get(line[i], 0)
        i += 1
    if depth:
        return None
    return m.group(1), line[m.end():i - 1], line[i:]


def render_numbers(template: str | None, macros: dict[str, str]) -> str:
    """Rewrite a numbers.tex template (comments and order preserved) with computed values."""
    seen: set[str] = set()
    out: list[str] = []
    for line in (template or "").splitlines():
        sp = _split_newcommand(line)
        if sp is None:
            out.append(line)
            continue
        name, _, trail = sp
        if name == "TBD":
            out.append(line)
            continue
        seen.add(name)
        out.append(f"\\newcommand{{\\{name}}}{{{macros.get(name, chr(92) + 'TBD')}}}{trail}")
    if not template:
        out = HEADER.splitlines()
    out.extend(f"\\newcommand{{\\{n}}}{{{macros.get(n, chr(92) + 'TBD')}}}" for n in MACRO_NAMES if n not in seen)
    out.extend(f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in macros.items() if n not in seen and n not in MACRO_NAMES)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- dataset stats (opt-in)

def compute_dataset_stats(results: Path) -> dict:
    """Load datasets through experiments/common.py (slow; only behind --dataset-stats)."""
    import common
    from hrag.qaf import _TOKEN_RE, _is_id

    stats, types = {}, {}
    for ds in common.DATASETS:
        if not common.dataset_available(ds):
            continue
        corpus, queries, qrels, _ = common.load_dataset(ds, "test")
        qids = [q for q in common.eval_qids(qrels) if q in queries]
        words = lambda s: len(s.split())  # noqa: E731
        n_dev = None
        if "dev" in common.SPLITS[ds]:
            _, dq, dqrels, _ = common.load_dataset(ds, "dev")
            n_dev = len([q for q in common.eval_qids(dqrels) if q in dq])
        stats[ds] = {
            "mean_doc_words": float(np.mean([words((d.title + " " + d.text).strip()) for d in corpus.values()])),
            "mean_query_words": float(np.mean([words(queries[q]) for q in qids])),
            "mean_rel_per_query": float(np.mean([sum(v > 0 for v in qrels[q].values()) for q in qids])),
            "n_dev_queries": n_dev}
        qt = {}
        for q in qids:
            toks = _TOKEN_RE.findall(queries[q])
            qt[q] = sum(_is_id(t) for t in toks) / len(toks) if toks else 0.0
        types[ds] = qt
    common.write_json(Path(results) / "dataset_stats.json", stats)
    common.write_json(Path(results) / "qaf" / "query_types.json", types)
    return stats


# ----------------------------------------------------------------------------- driver

def generate(results: Path, out: Path) -> dict[str, list[str]]:
    results, out = Path(results), Path(out)
    R = load_retrieval(results)
    head = headline_encoder(results, R)
    written = {"tables": [], "figures": [], "numbers": []}
    tables = {
        "main_results": lambda: main_results_table(R), "significance": lambda: significance_table(R, head),
        "ablation": lambda: ablation_table(results), "generation": lambda: generation_table(results),
        "latency": lambda: latency_table(R, head[0]), "qaf_coefficients": lambda: qaf_coef_table(results),
        "dataset_stats": lambda: dataset_stats_table(results), "matryoshka": lambda: matryoshka_table(results),
        "dat": lambda: dat_table(results), "pageindex": lambda: pageindex_table(results)}
    for name, fn in tables.items():
        s = fn()
        if s:
            write_text(out / "tables" / f"{name}.tex", s)
            written["tables"].append(name)
    figs = {"alpha_sweep": lambda: fig_alpha_sweep(results, R, out), "qaf_alpha_hist": lambda: fig_qaf_hist(results, R, out),
            "latency_breakdown": lambda: fig_latency(R, head[0], out), "matryoshka": lambda: fig_matryoshka(results, out)}
    for name, fn in figs.items():
        if fn():
            written["figures"].append(name)
    num = out / "numbers.tex"
    template = num.read_text() if num.exists() else (ROOT / "paper" / "numbers.tex").read_text() \
        if (ROOT / "paper" / "numbers.tex").exists() else None
    macros = compute_macros(results, R)
    write_text(num, render_numbers(template, macros))
    written["numbers"] = sorted(macros)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    ap.add_argument("--dataset-stats", action="store_true",
                    help="load the datasets via experiments/common.py and (re)write results/dataset_stats.json")
    args = ap.parse_args(argv)
    if args.dataset_stats:
        compute_dataset_stats(Path(args.results))
    w = generate(Path(args.results), Path(args.out))
    print(f"tables: {', '.join(w['tables']) or '-'}")
    print(f"figures: {', '.join(w['figures']) or '-'}")
    print(f"macros filled: {len(w['numbers'])}/{len(MACRO_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
