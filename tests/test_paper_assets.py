"""Tests for experiments/make_paper_assets.py on toy result JSONs (never touches paper/ or results/)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import make_paper_assets as mpa  # noqa: E402

from hrag.eval.stats import paired_bootstrap, paired_bootstrap_ci, paired_ttest  # noqa: E402

N = 40
QIDS = [f"q{i}" for i in range(N)]


def _rng(seed):
    return np.random.default_rng(seed)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _result(vals, alpha=None, lat=None, qtype=None, name="x"):
    pq = {}
    for q, v in zip(QIDS, vals):
        pq[q] = {"ndcg@10": float(v), "recall@10": float(min(1.0, v + 0.1)), "mrr@10": float(v)}
        if alpha is not None:
            pq[q]["alpha"] = alpha[q] if isinstance(alpha, dict) else alpha
        if qtype is not None:
            pq[q]["query_type"] = qtype[q]
    m = {k: float(np.mean([p[k] for p in pq.values()])) for k in ("ndcg@10", "recall@10", "mrr@10")}
    return {"config": {"name": name}, "metrics": m, "per_query": pq, "latency_ms": lat or {}}


_ORDER = [QIDS.index(q) for q in sorted(QIDS)]  # the generator aligns queries by sorted qid


def srt(v):
    """Per-query vector in the generator's (sorted-qid) order."""
    return np.asarray(v)[_ORDER]


def make_toy(results: Path) -> dict:
    """scifact + rfc, two encoders. Returns the per-config nDCG vectors for hand checks."""
    vecs = {}
    for di, ds in enumerate(("scifact", "rfc")):
        base = _rng(di).uniform(0.3, 0.7, N)
        vecs[(ds, None, "bm25")] = base - 0.10
        for ei, enc in enumerate(("bge-small", "gemma")):
            b = base + 0.05 * ei  # gemma better
            noise = _rng(10 + di).normal(0, 0.02, N)  # same noise for both encoders
            cfgs = {
                "dense": b - 0.05, "hybrid_fixed": b, "hybrid_fixed+rr": b + 0.02,
                "qaf": b + 0.01 + noise, "qaf+rr": b + 0.06 + noise,   # clearly > ref
                "oracle_upper_bound": b + 0.15}
            for cfg, v in cfgs.items():
                vecs[(ds, enc, cfg)] = v
    lat_stage = lambda m: {"mean": m, "p95": m * 2}  # noqa: E731
    for (ds, enc, cfg), v in vecs.items():
        qalpha = {q: (0.9 if i % 2 == 0 else 0.2) for i, q in enumerate(QIDS)} if "qaf" in cfg else (0.5 if "hybrid" in cfg else None)
        qtype = {q: ("id" if i % 2 else "natural") for i, q in enumerate(QIDS)}
        lat = {}
        if cfg in ("dense", "hybrid_fixed+rr", "qaf+rr"):
            lat = {"dense": lat_stage(10.0), "features": lat_stage(3.0 if cfg == "qaf+rr" else 0.0),
                   "rerank": lat_stage(100.0 if "rr" in cfg else 0.0), "total": lat_stage(150.0)}
            lat = {k: x for k, x in lat.items() if k == "total" or x["mean"] > 0}
        r = _result(v, qalpha, lat, qtype if cfg == "qaf" else None, cfg)
        name = f"{ds}__{cfg}" if enc is None else f"{ds}__{enc}__{cfg}"
        _write(results / "retrieval" / f"{name}.json", r)
    return vecs


@pytest.fixture()
def toy(tmp_path):
    res = tmp_path / "results"
    vecs = make_toy(res)
    return res, vecs


def test_main_results_bold_on_max_and_dagger(toy, tmp_path):
    res, vecs = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    tex = (out / "tables" / "main_results.tex").read_text()
    assert "\\label{tab:main_results}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    # first block = scifact; per column the best deployable cell is qaf+rr and must be bold
    qrow = [l for l in tex.splitlines() if l.startswith("QAF+RR &")][0]
    assert qrow.split(" & ")[1].startswith("\\textbf{")
    exp = np.mean(vecs[("scifact", "bge-small", "qaf+rr")])
    assert f"\\textbf{{{exp:.3f}}}" in qrow
    # oracle is higher in value but never bold
    orow = [l for l in tex.splitlines() if l.startswith("Oracle")][0]
    assert "\\textbf" not in orow
    # exactly one bold per nDCG@10 BGE column per dataset: count bold cells in the first data column
    first_col = [l.split(" & ")[1] for l in tex.splitlines() if re.match(r"^(BM25|Dense|Hybrid|QAF|Oracle)", l)]
    assert sum("\\textbf" in c for c in first_col) == 2  # one per dataset block


def test_dagger_placement(toy, tmp_path):
    res, _ = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    lines = (out / "tables" / "main_results.tex").read_text().splitlines()
    by = lambda prefix: [l for l in lines if l.startswith(prefix)]  # noqa: E731
    # qaf+rr beats the reference significantly -> dagger on every cell
    assert all("\\dagger" in c for c in by("QAF+RR &")[0].rstrip(" \\\\").split(" & ")[1:])
    # reference row itself, and the worse rows, never carry a dagger
    for prefix in ("Hybrid-fixed+RR &", "Hybrid-fixed &", "Dense &", "BM25 &", "Oracle"):
        assert all("\\dagger" not in l for l in by(prefix) if not l.startswith("Hybrid-fixed+RR") or prefix.startswith("Hybrid-fixed+RR"))
    assert "\\dagger" not in by("Hybrid-fixed+RR &")[0]
    assert "\\dagger" not in by("Hybrid-fixed &")[0]


def test_dagger_absent_when_not_significant(tmp_path):
    res = tmp_path / "results"
    a = _rng(3).uniform(0.3, 0.7, N)
    noise = _rng(4).normal(0, 0.1, N)
    _write(res / "retrieval" / "scifact__bge-small__hybrid_fixed+rr.json", _result(a, name="r"))
    _write(res / "retrieval" / "scifact__bge-small__qaf+rr.json", _result(a + noise - noise.mean(), name="q"))
    mpa.generate(res, tmp_path / "paper")
    tex = (tmp_path / "paper" / "tables" / "main_results.tex").read_text()
    assert "\\dagger" not in [l for l in tex.splitlines() if l.startswith("QAF+RR &")][0]


def test_missing_inputs_skipped_silently(tmp_path, capsys):
    res = tmp_path / "empty"
    res.mkdir()
    out = tmp_path / "paper"
    w = mpa.generate(res, out)
    assert w["tables"] == [] and w["figures"] == []
    assert not list((out / "tables").glob("*.tex")) if (out / "tables").exists() else True
    assert not (out / "figures").exists() or not list((out / "figures").glob("*.pdf"))
    assert capsys.readouterr().err == ""
    # nonexistent results dir too
    assert mpa.generate(tmp_path / "nope", tmp_path / "paper2")["tables"] == []
    # numbers.tex still written with all macros as TBD
    assert "\\TBD" in (out / "numbers.tex").read_text()


def test_partial_inputs_only_write_what_exists(toy, tmp_path):
    res, _ = toy
    out = tmp_path / "paper"
    w = mpa.generate(res, out)
    assert {"main_results", "significance", "latency"} <= set(w["tables"])
    for absent in ("ablation", "generation", "qaf_coefficients", "dataset_stats", "matryoshka", "dat", "pageindex"):
        assert absent not in w["tables"] and not (out / "tables" / f"{absent}.tex").exists()
    assert not (out / "figures" / "matryoshka.pdf").exists()


def test_numbers_tex_keeps_every_macro(toy, tmp_path):
    res, _ = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    names_now = set(re.findall(r"\\newcommand\{\\(\w+)\}", (ROOT / "paper" / "numbers.tex").read_text()))
    new = (out / "numbers.tex").read_text()
    names_new = set(re.findall(r"\\newcommand\{\\(\w+)\}", new))
    assert names_now <= names_new, names_now - names_new
    assert set(mpa.MACRO_NAMES) <= names_new
    # no input for these -> placeholder retained
    assert "\\newcommand{\\GemmaDocsPerSec}{\\TBD}" in new
    assert "\\newcommand{\\FiQAEncodeMinGemma}{\\TBD}" in new


def test_numbers_values_and_build_times(toy, tmp_path):
    res, vecs = toy
    _write(res / "indexes" / "build_times.json", {
        "scifact": {"gemma": {"seconds": 900.0, "docs_per_s": 5.75}, "bge-small": {"seconds": 80, "docs_per_s": 65.34}},
        "fiqa": {"gemma": {"seconds": 1230.0, "docs_per_s": 16.26}, "bge-small": {"seconds": 179, "docs_per_s": 111.63}},
        "hotpot": {"bge-small": {"seconds": 29, "docs_per_s": 168.63}}})
    _write(res / "qaf" / "coefficients_gemma.json", {
        "trained_on": {"scifact-dev": 162, "fiqa-dev": 500}, "train_r2_clipped": 0.1234,
        "coefficients": {"id_ratio": 0.01, "dense_top": -0.09, "len_tokens": 0.03}, "intercept": 0.6})
    _write(res / "qaf" / "alpha_star_gemma.json", {"alpha_star": 0.7, "pooled_dev_ndcg@10": {"0.6": 0.5, "0.7": 0.6}})
    _write(res / "qaf" / "alpha_star_bge-small.json", {"alpha_star": 0.9, "pooled_dev_ndcg@10": {"0.9": 0.55}})
    out = tmp_path / "paper"
    mpa.generate(res, out)
    num = (out / "numbers.tex").read_text()
    val = lambda n: re.search(r"\\newcommand\{\\" + n + r"\}\{(.*?)\}(?:\s*%.*)?$", num, re.M).group(1)  # noqa: E731
    assert val("FiQAEncodeMinGemma") == "20"
    assert val("GemmaDocsPerSec") == "5.8--16.3"
    assert val("BGEDocsPerSec") == "65--169"
    assert val("AlphaStarGemma") == "0.7" and val("AlphaStarBge") == "0.9"
    assert val("QAFnDev") == "662" and val("QAFtrainRsq") == "0.12"
    assert val("QAFtopFeature") == "\\texttt{dense\\_top}"
    assert val("QAFlatencyMs") == "3.0"          # features(qaf+rr) - features(hybrid_fixed+rr)
    assert val("RerankLatencyMs") == "100" and val("TotalLatencyMs") == "150"
    # NDCGsd = sample std of per-query nDCG@10 of qaf+rr of the better encoder (gemma)
    sd = float(np.std(vecs[("scifact", "gemma", "qaf+rr")], ddof=1))
    assert val("NDCGsdSciFact") == f"{sd:.3f}"
    assert val("NDCGsdFiQA") == "\\TBD"
    # gain/CI/p equal hand computation
    a, b = srt(vecs[("scifact", "gemma", "hybrid_fixed+rr")]), srt(vecs[("scifact", "gemma", "qaf+rr")])
    d, lo, hi = paired_bootstrap_ci(a.tolist(), b.tolist())
    assert val("QAFgainSciFact") == f"$+{d:.3f}$"
    assert val("QAFgainCISciFact") == f"$[{lo:.3f},\\,{hi:.3f}]$"
    assert val("BMndcgSciFact") == f"{np.mean(vecs[('scifact', None, 'bm25')]):.3f}"
    assert val("BestNDCGRFC") == f"{max(np.mean(v) for (ds, e, c), v in vecs.items() if ds == 'rfc' and c != 'oracle_upper_bound'):.3f}"
    assert val("RerankGainSciFact") == f"$+{np.mean(b - vecs[('scifact', 'gemma', 'qaf')]):.3f}$"
    assert val("OracleHeadroomRFC") == "$+0.150$"


def test_significance_values(toy, tmp_path):
    res, vecs = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    tex = (out / "tables" / "significance.tex").read_text()
    assert "\\label{tab:significance}" in tex
    sig = mpa.significance_stats(mpa.load_retrieval(res))
    G = "gemma"

    # 1: qaf+rr vs hybrid_fixed+rr (gemma is the better encoder)
    a = srt(vecs[("scifact", "gemma", "hybrid_fixed+rr")]).tolist()
    b = srt(vecs[("scifact", "gemma", "qaf+rr")]).tolist()
    s = sig[("qaf_rr_vs_fixed_rr", "scifact", G)]
    d, lo, hi = paired_bootstrap_ci(a, b)
    assert (s["delta"], s["lo"], s["hi"]) == pytest.approx((d, lo, hi))
    assert s["delta"] == pytest.approx(np.mean(b) - np.mean(a))
    assert s["p_boot"] == paired_bootstrap(a, b) and s["p_t"] == paired_ttest(a, b) and s["n"] == N
    assert ("qaf_rr_vs_fixed_rr", "scifact", "bge-small") in sig  # both encoders shown
    # 2: hybrid_fixed vs best single retriever: dense (b-0.05) beats bm25 (base-0.10) for gemma
    #    gemma b = base+0.05 -> dense = base; bm25 = base - 0.10 -> best single = dense
    s2 = sig[("hybrid_vs_single", "rfc", G)]
    assert s2["delta"] == pytest.approx(0.05)
    # 3: +rr vs no-rr (qaf)
    s3 = sig[("rr_vs_norr", "rfc", G)]
    assert s3["delta"] == pytest.approx(0.05)
    # 4: Gemma vs BGE: constant shift 0.05 -> degenerate CI, p ~ 0
    s4 = sig[("gemma_vs_bge", "scifact", None)]
    assert (s4["delta"], s4["lo"], s4["hi"]) == pytest.approx((0.05, 0.05, 0.05))
    assert s4["p_boot"] == 0.0 and s4["p_t"] == 0.0
    # the rendered row carries delta, CI, p and n, with the dagger on significant deltas
    row = [l for l in tex.splitlines() if l.startswith("SciFact (Gemma)")][0]
    assert f"$+{d:.3f}$" in row and "\\dagger" in row and f"[{lo:.3f}" in row and row.rstrip(" \\\\").endswith(str(N))
    assert tex.count("\\multicolumn{6}{l}") == 4 and "SciFact (BGE-small)" in tex and "fixed a priori" in tex and "10^{-4}" in tex
    assert "$<\\!0.0001$" in tex or "0.0000" not in tex


def test_significance_skips_missing_comparisons(tmp_path):
    res = tmp_path / "results"
    v = _rng(1).uniform(0.3, 0.7, N)
    _write(res / "retrieval" / "scifact__bge-small__hybrid_fixed+rr.json", _result(v))
    _write(res / "retrieval" / "scifact__bge-small__qaf+rr.json", _result(v + 0.05))
    sig = mpa.significance_stats(mpa.load_retrieval(res))
    assert set(sig) == {("qaf_rr_vs_fixed_rr", "scifact", "bge-small")}


def _extra_inputs(res: Path):
    _write(res / "qaf" / "coefficients_bge-small.json", {
        "coefficients": {"id_ratio": 0.011, "dense_top": 0.048, "oov_ratio": -0.028},
        "train_r2_clipped": 0.2, "intercept": 0.6, "trained_on": {"scifact-dev": 2, "fiqa-dev": 3}})
    for ds in ("scifact", "rfc"):
        for cfg, d in (("full", 0.7), ("no_chunking", 0.6), ("no_expansion", 0.65), ("no_qaf", 0.62), ("no_rerank", 0.55)):
            _write(res / "ablation" / f"{ds}__{cfg}.json", {"config": {"name": cfg}, "metrics": {"ndcg@10": d, "recall@10": d + 0.1}})
    for ds in ("hotpot", "rfc"):
        for i, cfg in enumerate(("bm25", "dense", "hybrid_fixed+rr", "qaf+rr")):
            _write(res / "generation" / f"{ds}__{cfg}.json", {
                "config": {"name": cfg}, "metrics": {"em": 0.2 + 0.05 * i, "f1": 0.3 + 0.05 * i, "citation_hit": 0.5},
                "per_query": {"a": {"f1": 0.1, "em": 0, "faithfulness": 1}, "b": {"f1": 0.2, "em": 0, "faithfulness": 0}}})
    for dim, v in ((768, 0.6), (512, 0.59), (256, 0.57), (128, 0.5)):
        _write(res / "matryoshka" / f"scifact__{dim}__qaf+rr.json", {
            "config": {"name": "qaf+rr", "dim": dim}, "metrics": {"ndcg@10": v}, "index_mb": dim / 10,
            "per_query": {}, "latency_ms": {"total": {"mean": 20.0, "p95": 30.0}}})
    _write(res / "dat" / "scifact__dat.json", {"config": {"name": "dat"}, "metrics": {"ndcg@10": 0.5, "recall@10": 0.6, "mrr@10": 0.4},
                                              "latency_ms": {"total": {"mean": 900.0, "p95": 1200.0}}})
    _write(res / "pageindex" / "rfc.json", {"metrics": {"recall@5": 0.4, "f1": 0.3}, "latency_ms": {"total": {"mean": 5000.0}}})
    _write(res / "dataset_stats.json", {"scifact": {"mean_doc_words": 214.5, "mean_query_words": 12.3, "mean_rel_per_query": 1.13, "n_dev_queries": 162},
                                       "rfc": {"mean_doc_words": 180.0, "mean_query_words": 15.0, "mean_rel_per_query": 1.0, "n_dev_queries": None}})


def test_all_optional_tables_and_figures(toy, tmp_path):
    res, _ = toy
    _extra_inputs(res)
    out = tmp_path / "paper"
    w = mpa.generate(res, out)
    assert set(w["tables"]) == {"main_results", "significance", "ablation", "generation", "latency", "qaf_coefficients",
                                "dataset_stats", "matryoshka", "dat", "pageindex"}
    for t in w["tables"]:
        s = (out / "tables" / f"{t}.tex").read_text()
        assert f"\\label{{tab:{t}}}" in s and "\\toprule" in s
    assert set(w["figures"]) == {"qaf_alpha_hist", "latency_breakdown", "matryoshka"}
    ds = (out / "tables" / "dataset_stats.tex").read_text()
    assert "214.5" in ds and "162" in ds and "passages" not in ds.lower().split("\\begin{tabular}")[1]
    abl = (out / "tables" / "ablation.tex").read_text()
    assert abl.count("\\textbf{0.700}") == 2 and "$-$rerank" in abl
    coef = (out / "tables" / "qaf_coefficients.tex").read_text()
    assert "\\textbf{+0.048}" in coef and "dense\\_top" in coef
    gen = (out / "tables" / "generation.tex").read_text()
    assert "\\textbf{0.450}" in gen  # best F1 = qaf+rr
    for f in ("latency_breakdown", "qaf_alpha_hist", "matryoshka"):
        assert (out / "figures" / f"{f}.pdf").read_bytes()[:4] == b"%PDF"
    assert w["figures"].count("matryoshka") == 1


def test_alpha_sweep_and_hist_figures(toy, tmp_path):
    res, _ = toy
    for ds in ("scifact", "rfc"):
        _write(res / "alpha_sweep" / f"{ds}__gemma.json", {"test": {str(round(0.1 * i, 1)): 0.5 + 0.01 * i for i in range(11)}})
    out = tmp_path / "paper"
    w = mpa.generate(res, out)
    assert {"alpha_sweep", "qaf_alpha_hist", "latency_breakdown"} <= set(w["figures"])
    for f in ("alpha_sweep", "qaf_alpha_hist", "latency_breakdown"):
        data = (out / "figures" / f"{f}.pdf").read_bytes()
        assert data[:4] == b"%PDF"
    idh, nat, how = mpa.predicted_alphas(res, mpa.load_retrieval(res))
    assert how == "query" and len(idh) == len(nat) == 2 * N // 2 and set(idh) == {0.2}


def test_dataset_stats_not_computed_without_flag(toy, tmp_path):
    res, _ = toy
    mpa.generate(res, tmp_path / "paper")
    assert not (res / "dataset_stats.json").exists()


def _tectonic_doc(out: Path) -> str:
    tabs = sorted((out / "tables").glob("*.tex"))
    inputs = "\n".join(f"\\input{{tables/{t.stem}}}" for t in tabs)
    return ("\\documentclass{article}\n\\usepackage{booktabs,graphicx,amssymb}\n"
            f"\\begin{{document}}\n{inputs}\n\\end{{document}}\n")


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_tables_compile_with_tectonic(toy, tmp_path):
    res, _ = toy
    _extra_inputs(res)
    out = tmp_path / "paper"
    mpa.generate(res, out)
    (out / "doc.tex").write_text(_tectonic_doc(out))
    (out / "numbers_check.tex").write_text("")
    r = subprocess.run(["tectonic", "--keep-logs", "doc.tex"], cwd=out, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "doc.pdf").exists()


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_numbers_tex_compiles(toy, tmp_path):
    res, _ = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    (out / "n.tex").write_text("\\documentclass{article}\n\\begin{document}\n\\input{numbers}\n"
                               "\\QAFgainSciFact{} \\QAFgainCISciFact{} \\QAFpSciFact{} \\QAFtopFeature{} \\GemmaDocsPerSec{}\n"
                               "\\end{document}\n")
    r = subprocess.run(["tectonic", "n.tex"], cwd=out, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def _run(vals, qids=QIDS):
    return {"metrics": {}, "per_query": {q: {"ndcg@10": float(v)} for q, v in zip(qids, vals)}}


def test_paired_rejects_misaligned_or_bad_values():
    v = _rng(5).uniform(0.2, 0.8, N)
    a = _run(v)
    assert mpa.paired(a, _run(v + 0.1)) is not None
    b = _run(v + 0.1)
    del b["per_query"]["q3"]
    assert mpa.paired(a, b) is None and mpa.compare(a, b) is None  # never intersect
    c = _run(v + 0.1)
    c["per_query"]["q4"]["ndcg@10"] = float("nan")
    assert mpa.paired(a, c) is None
    d = _run(v + 0.1)
    d["per_query"]["q5"]["ndcg@10"] = None
    assert mpa.paired(a, d) is None


def test_misaligned_pair_gives_dashes_and_tbd(tmp_path):
    res = tmp_path / "results"
    v = _rng(6).uniform(0.3, 0.7, N)
    r = _result(v + 0.05, name="q")
    ref = _result(v, name="r")
    del r["per_query"]["q7"]
    _write(res / "retrieval" / "scifact__bge-small__hybrid_fixed+rr.json", ref)
    _write(res / "retrieval" / "scifact__bge-small__qaf+rr.json", r)
    w = mpa.generate(res, tmp_path / "paper")
    assert "significance" not in w["tables"]
    tex = (tmp_path / "paper" / "tables" / "main_results.tex").read_text()
    assert "\\dagger" not in [l for l in tex.splitlines() if l.startswith("QAF+RR &")][0]
    num = (tmp_path / "paper" / "numbers.tex").read_text()
    assert "\\newcommand{\\QAFgainSciFact}{\\TBD}" in num


def test_headline_encoder_chosen_on_dev_not_test(toy):
    res, _ = toy  # test results favour gemma
    R = mpa.load_retrieval(res)
    assert mpa.headline_encoder(res, R) == ("bge-small", "fixed")
    _write(res / "qaf" / "alpha_star_bge-small.json", {"alpha_star": 0.9, "pooled_dev_ndcg@10": {"0.9": 0.9}})
    _write(res / "qaf" / "alpha_star_gemma.json", {"alpha_star": 0.5, "pooled_dev_ndcg@10": {"0.5": 0.4}})
    assert mpa.headline_encoder(res, R) == ("bge-small", "dev")  # dev says BGE even though test says Gemma


def test_headline_macros_use_headline_encoder(toy, tmp_path):
    res, vecs = toy
    out = tmp_path / "paper"
    mpa.generate(res, out)
    num = (out / "numbers.tex").read_text()
    a, b = srt(vecs[("scifact", "bge-small", "hybrid_fixed+rr")]), srt(vecs[("scifact", "bge-small", "qaf+rr")])
    d, _, _ = paired_bootstrap_ci(a.tolist(), b.tolist())
    assert f"\\newcommand{{\\QAFgainSciFact}}{{$+{d:.3f}$}}" in num


def test_ci_percentile_hand_computed_and_narrow_for_correlated():
    a = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7, 0.6]
    b = [0.2, 0.3, 0.5, 1.0, 0.4, 0.6, 0.9, 0.5]
    mean, lo, hi = paired_bootstrap_ci(a, b, n_resamples=2000, seed=7)
    d = np.asarray(b) - np.asarray(a)
    idx = np.random.default_rng(7).integers(0, d.size, size=(2000, d.size))
    m = np.sort(d[idx].mean(axis=1))
    assert (lo, hi) == pytest.approx((np.percentile(m, 2.5), np.percentile(m, 97.5)))
    assert mean == pytest.approx(d.mean())
    # strongly correlated pairs (huge between-query spread, tiny paired noise): narrow CI
    base = _rng(8).uniform(0, 1, 200)
    _, lo2, hi2 = paired_bootstrap_ci(base.tolist(), (base + 0.05 + _rng(9).normal(0, 0.005, 200)).tolist())
    assert hi2 - lo2 < 0.005


def test_treatment_clearly_loses():
    from hrag.eval.stats import paired_bootstrap as pb
    a = _run(_rng(2).uniform(0.4, 0.8, N))
    b = {"metrics": {}, "per_query": {q: {"ndcg@10": v["ndcg@10"] - 0.1} for q, v in a["per_query"].items()}}
    r = mpa.compare(a, b)
    assert r["p_boot"] > 0.95 and r["p_t"] > 0.95 and r["delta"] == pytest.approx(-0.1)
    assert pb([0.0] * 20, [-0.1] * 20) > 0.95


def test_tie_bolds_both_cells(tmp_path):
    res = tmp_path / "results"
    v = _rng(11).uniform(0.3, 0.7, N)
    _write(res / "retrieval" / "scifact__bge-small__hybrid_fixed+rr.json", _result(v, name="r"))
    _write(res / "retrieval" / "scifact__bge-small__qaf+rr.json", _result(v, name="q"))
    mpa.generate(res, tmp_path / "paper")
    lines = (tmp_path / "paper" / "tables" / "main_results.tex").read_text().splitlines()
    rows = [l for l in lines if l.startswith(("QAF+RR &", "Hybrid-fixed+RR &"))]
    assert len(rows) == 2 and all(r.split(" & ")[1].startswith("\\textbf{") for r in rows)


def test_partial_macros_stay_tbd(toy, tmp_path):
    res, _ = toy
    # drop one main config of one encoder -> BestNDCGRFC must be TBD
    (res / "retrieval" / "rfc__gemma__qaf.json").unlink()
    # latency missing on one dataset for the reference -> compared only on the common set
    r = json.loads((res / "retrieval" / "rfc__bge-small__hybrid_fixed+rr.json").read_text())
    r["latency_ms"] = {}
    _write(res / "retrieval" / "rfc__bge-small__hybrid_fixed+rr.json", r)
    R = mpa.load_retrieval(res)
    L = mpa.latency_stats(R, "bge-small")
    assert set(L) == {"dense", "hybrid_fixed+rr", "qaf+rr"}
    out = tmp_path / "paper"
    mpa.generate(res, out)
    num = (out / "numbers.tex").read_text()
    assert "\\newcommand{\\BestNDCGRFC}{\\TBD}" in num
    assert "\\newcommand{\\BestNDCGSciFact}{\\TBD}" not in num
    # QAFlatencyMs: only scifact has both -> value from scifact alone (3.0), not rfc
    assert "\\newcommand{\\QAFlatencyMs}{3.0}" in num


def test_qaf_latency_tbd_without_baseline_latency(toy, tmp_path):
    res, _ = toy
    for ds in ("scifact", "rfc"):
        f = res / "retrieval" / f"{ds}__bge-small__hybrid_fixed+rr.json"
        r = json.loads(f.read_text())
        r["latency_ms"] = {}
        _write(f, r)
    mpa.generate(res, tmp_path / "paper")
    assert "\\newcommand{\\QAFlatencyMs}{\\TBD}" in (tmp_path / "paper" / "numbers.tex").read_text()


def test_generation_macros_use_common_datasets_and_per_query_faith(tmp_path):
    res = tmp_path / "results"
    def gen(ds, cfg, f1, faith=None):
        pq = {"a": {"f1": 0.1}} if faith is None else {"a": {"f1": 0.1, "faithfulness": faith}}
        _write(res / "generation" / f"{ds}__{cfg}.json", {"config": {"name": cfg}, "metrics": {"em": 0.1, "f1": f1, "faithfulness": 0.99}, "per_query": pq})
    gen("hotpot", "qaf+rr", 0.5, 1.0); gen("hotpot", "dense", 0.3, 0.0)
    gen("rfc", "qaf+rr", 0.9)  # no dense on rfc -> excluded from both averages
    mpa.generate(res, tmp_path / "paper")
    num = (tmp_path / "paper" / "numbers.tex").read_text()
    assert "\\newcommand{\\GenFoneQAF}{0.500}" in num and "\\newcommand{\\GenFoneDense}{0.300}" in num
    assert "\\newcommand{\\FaithQAF}{1.00}" in num and "\\newcommand{\\FaithDense}{0.00}" in num  # not 0.99
    # no dataset has both -> TBD
    (res / "generation" / "hotpot__dense.json").unlink()
    mpa.generate(res, tmp_path / "paper")
    assert "\\newcommand{\\GenFoneQAF}{\\TBD}" in (tmp_path / "paper" / "numbers.tex").read_text()


def test_matryoshka_exact_config_names(tmp_path):
    res = tmp_path / "results"
    _write(res / "matryoshka" / "scifact__256__qaf+rr.json", {"config": {"name": "qaf+rr", "dim": 256}, "metrics": {"ndcg@10": 0.5}})
    _write(res / "matryoshka" / "scifact__256__undense.json", {"config": {"name": "undense", "dim": 256}, "metrics": {"ndcg@10": 0.9}})
    rows = mpa.load_matryoshka(res)
    assert [r["config"] for r in rows] == ["qaf+rr"]
