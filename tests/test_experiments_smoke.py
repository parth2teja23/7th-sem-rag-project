"""Smoke test for the Tier-A experiment chain (build_indexes -> train_qaf -> run_retrieval)
on two toy 20-doc / 5-query datasets with a stub encoder and a stub reranker.

Also proves that the offline path used by the experiments (cached dense/sparse top-100 ->
weighted_rrf(k=60, top_n=100) -> rerank depth 50) yields exactly the same final ranking as
``Pipeline.retrieve`` (dedupe=False) for fixed and qaf alpha modes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP_DIR))

import build_indexes  # noqa: E402
import common  # noqa: E402
import run_retrieval  # noqa: E402
import train_qaf  # noqa: E402

from hrag.pipeline import Pipeline, PipelineConfig  # noqa: E402
from hrag.types import Doc, sort_ranking  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")
TOPICS = ["protein", "market", "river", "engine", "galaxy", "virus", "tariff", "neuron",
          "glacier", "router", "enzyme", "bond", "volcano", "kernel", "planet", "vaccine",
          "dividend", "synapse", "desert", "socket"]


def _toy(prefix: str):
    """20 docs, 5 dev + 5 test queries. Docs mix a topic word, an identifier and filler."""
    corpus = {}
    for i, t in enumerate(TOPICS):
        did = f"{prefix}{i}"
        corpus[did] = Doc(id=did, title=f"{t} notes",
                          text=f"{t} {t} study of {TOPICS[(i + 3) % 20]} with code ID-{i}{prefix} "
                               f"and general background on {TOPICS[(i + 7) % 20]} systems")
    splits = {}
    for split, offs in (("dev", range(0, 5)), ("test", range(5, 10))):
        queries, qrels = {}, {}
        for j in offs:
            i = (j * 3 + (1 if prefix == "b" else 0)) % 20
            qid = f"{prefix}-{split}-q{j}"
            # alternate identifier-heavy (sparse-friendly) and descriptive queries
            queries[qid] = (f"ID-{i}{prefix} code" if j % 2 else f"what about {TOPICS[i]} study")
            qrels[qid] = {f"{prefix}{i}": 1, f"{prefix}{(i + 4) % 20}": (1 if j % 3 == 0 else 0)}
            qrels[qid] = {d: r for d, r in qrels[qid].items() if r > 0}
        splits[split] = (queries, qrels)
    return corpus, splits


class _Tok:
    def get_vocab(self):
        return {w: i for i, w in enumerate(TOPICS + ["study", "what", "about", "code"])}


class _Model:
    tokenizer = _Tok()


class StubEncoder:
    """Deterministic hashed bag-of-words embeddings (dim 16)."""

    def __init__(self, name="stub", dim=None):
        self.name, self.dim, self.model = "stub", 16, _Model()

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 + (h >> 8) % 3
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_docs(self, docs, batch_size=64):
        return np.stack([self._vec(f"{d.title} {d.text}") for d in docs]).astype(np.float32)


class StubReranker:
    """Same contract as hrag.rerank.Reranker.rerank; score = word overlap + id-hash jitter."""

    name = "stub-ce"

    def rerank(self, query, ranking, corpus, depth=50):
        if depth <= 0 or not ranking:
            return []
        q = set(_WORD.findall(query.lower()))
        out = []
        for i, _ in ranking[:depth]:
            if i not in corpus:
                continue
            d = set(_WORD.findall(f"{corpus[i].title} {corpus[i].text}".lower()))
            jitter = int(hashlib.md5(i.encode()).hexdigest(), 16) % 1000 / 1e5
            out.append((i, float(len(q & d)) + jitter))
        return sort_ranking(out)


@pytest.fixture()
def toy_env(tmp_path, monkeypatch):
    data = {"toya": _toy("a"), "toyb": _toy("b")}

    def loader(name):
        def load(split):
            corpus, splits = data[name]
            q, r = splits[split]
            return dict(corpus), dict(q), {k: dict(v) for k, v in r.items()}, None
        return load

    monkeypatch.setattr(common, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path / "results" / "cache")
    monkeypatch.setitem(common.LOADERS, "toya", loader("toya"))
    monkeypatch.setitem(common.LOADERS, "toyb", loader("toyb"))
    monkeypatch.setitem(common.SPLITS, "toya", ("dev", "test"))
    monkeypatch.setitem(common.SPLITS, "toyb", ("dev", "test"))
    monkeypatch.setattr(common, "DEV_DATASETS", ("toya", "toyb"))
    monkeypatch.setattr(common, "make_encoder", lambda name, dim=None: StubEncoder(name, dim))
    monkeypatch.setattr(common, "make_reranker", lambda: StubReranker())
    return tmp_path


CONFIGS = ["dense", "hybrid_fixed", "hybrid_fixed+rr", "qaf", "qaf+rr", "oracle_upper_bound"]


def _run_chain():
    args = ["--datasets", "toya", "toyb", "--encoders", "stub"]
    assert build_indexes.main(args) == 0
    assert train_qaf.main(["--encoders", "stub"]) == 0
    assert run_retrieval.main(args + ["--latency-n", "3", "--skip-sanity-gate"]) == 0


def test_full_chain_writes_contract_files(toy_env):
    root = toy_env
    _run_chain()
    for ds in ("toya", "toyb"):
        assert (root / "indexes" / ds / "bm25" / "meta.json").exists()
        assert (root / "indexes" / ds / "stub" / "meta.json").exists()
        files = [root / "results" / "retrieval" / f"{ds}__bm25.json"] + [
            root / "results" / "retrieval" / f"{ds}__stub__{c}.json" for c in CONFIGS]
        for f in files:
            res = json.loads(f.read_text())
            assert set(res) >= {"config", "metrics", "per_query", "latency_ms"}
            assert set(res["metrics"]) >= {"recall@10", "mrr@10", "ndcg@10", "ndcg@5", "ndcg@20"}
            assert len(res["per_query"]) == 5
            for pq in res["per_query"].values():
                assert "ndcg@10" in pq and "alpha" in pq
        orc = json.loads((root / "results" / "retrieval" / f"{ds}__stub__oracle_upper_bound.json").read_text())
        assert orc["config"]["name"] == "oracle_upper_bound"
        hyb = json.loads((root / "results" / "retrieval" / f"{ds}__stub__hybrid_fixed+rr.json").read_text())
        assert hyb["config"]["alpha"] in [round(0.1 * i, 1) for i in range(11)]
        assert hyb["config"]["rerank_depth"] == 50 and hyb["config"]["dedupe"] is False
        for c in ("dense", "hybrid_fixed+rr", "qaf+rr"):
            lat = json.loads((root / "results" / "retrieval" / f"{ds}__stub__{c}.json").read_text())["latency_ms"]
            assert "total" in lat and set(lat["total"]) == {"mean", "p95"}
        sweep = json.loads((root / "results" / "alpha_sweep" / f"{ds}__stub.json").read_text())
        assert set(sweep["test"]) == {str(a) for a in [round(0.1 * i, 1) for i in range(11)]}
    assert (root / "indexes" / "qaf_stub.joblib").exists()
    coef = json.loads((root / "results" / "qaf" / "coefficients_stub.json").read_text())
    assert len(coef["coefficients"]) == 12
    lodo = json.loads((root / "results" / "qaf" / "lodo_stub.json").read_text())
    assert set(lodo["heldout"]) == {"toya", "toyb"}
    tg = json.loads((root / "results" / "qaf" / "targets_stub.json").read_text())
    assert sum(tg["pooled"]["histogram"].values()) == 10
    sig = json.loads((root / "results" / "significance.json").read_text())
    for ds in ("toya", "toyb"):
        cmp = sig[ds]["stub"]
        assert set(cmp) == {"qaf+rr_vs_hybrid_fixed+rr", "qaf+rr_vs_dense", "qaf_vs_hybrid_fixed"}
        for v in cmp.values():
            assert set(v) >= {"n", "mean_diff", "p_bootstrap", "p_ttest"}
            assert v["n"] == 5


def test_build_skips_existing_unless_force(toy_env, capsys):
    args = ["--datasets", "toya", "--encoders", "stub"]
    assert build_indexes.main(args) == 0
    capsys.readouterr()
    assert build_indexes.main(args) == 0
    assert "skip" in capsys.readouterr().out.lower()
    assert build_indexes.main(args + ["--force"]) == 0
    assert "built" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("mode", ["fixed", "qaf"])
@pytest.mark.parametrize("rerank", [False, True])
def test_offline_path_matches_pipeline(toy_env, mode, rerank):
    _run_chain()
    corpus, queries, qrels, _ = common.load_dataset("toya", "test")
    sparse, dense = common.load_indexes("toya", "stub")
    rankings = common.get_rankings("toya", "test", "stub", queries, sparse, dense)
    qaf = common.load_qaf("stub")
    vocab = common.encoder_vocab(dense.encoder)
    alpha_star = common.alpha_star("stub")
    reranker = common.make_reranker()
    rr_cache = common.RerankCache("toya", reranker)
    cfg = PipelineConfig(encoder="stub", alpha_mode=mode, alpha=alpha_star, rerank=rerank,
                         rerank_depth=50, dedupe=False, expand=False, final_k=100)
    pipe = Pipeline(cfg, corpus, sparse, dense, reranker, qaf if mode == "qaf" else None)
    for qid, q in queries.items():
        d, s = rankings[qid]["dense"], rankings[qid]["sparse"]
        a = alpha_star if mode == "fixed" else common.qaf_alpha(qaf, q, d, s, sparse, vocab)
        offline = common.fuse(d, s, a)
        if rerank:
            offline = rr_cache.rerank(qid, q, offline, corpus)
        live = pipe.retrieve(q)
        assert live.alpha == pytest.approx(a, abs=0)
        assert [x for x, _ in live.final] == [x for x, _ in offline]
        assert [y for _, y in live.final] == pytest.approx([y for _, y in offline])


# ----------------------------------------------------------------------------- Task 14:
# ablation, Matryoshka and rerank-depth scripts on the same toy fixture.

import run_ablation  # noqa: E402
import run_matryoshka  # noqa: E402
import run_rerank_depth  # noqa: E402
import make_paper_assets as mpa  # noqa: E402


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def _toy_fixed(name: str):
    """Fixed-size-chunk variant of a toy dataset; a chunk is relevant iff its parent is."""
    from hrag.chunking import fixed_chunks

    def load(split):
        corpus, queries, qrels, _ = common.load_dataset(name, "test")
        chunks = {c.id: c for d in corpus.values() for c in fixed_chunks(d, size=8)}
        qr = {q: {c: 1 for c, ch in chunks.items() if rel.get(ch.meta["parent"], 0) > 0}
              for q, rel in qrels.items()}
        return chunks, queries, qr, None
    return load


def _same_metrics(a: dict, b: dict) -> None:
    assert a["metrics"] == pytest.approx(b["metrics"])
    assert {q: v["ndcg@10"] for q, v in a["per_query"].items()} == \
        pytest.approx({q: v["ndcg@10"] for q, v in b["per_query"].items()})


def test_ablation_writes_contract_files(toy_env, monkeypatch):
    root = toy_env
    _run_chain()
    monkeypatch.setitem(run_ablation.CHUNK_VARIANTS, "toya", _toy_fixed("toya"))
    monkeypatch.setattr(run_ablation, "_generation_running", lambda: False)
    assert run_ablation.main(["--datasets", "toya", "toyb", "--encoder", "stub",
                              "--latency-n", "3"]) == 0
    abl, ret = root / "results" / "ablation", root / "results" / "retrieval"
    assert sorted(f.name for f in abl.glob("toya__*")) == [
        "toya__-chunking.json", "toya__-qaf.json", "toya__-rerank.json", "toya__full.json"]
    assert sorted(f.name for f in abl.glob("toyb__*")) == [
        "toyb__-qaf.json", "toyb__-rerank.json", "toyb__full.json"]
    for ds in ("toya", "toyb"):
        full = _read(abl / f"{ds}__full.json")
        # each ablation row is the matching core-retrieval config of the same encoder
        _same_metrics(full, _read(ret / f"{ds}__stub__qaf+rr.json"))
        _same_metrics(_read(abl / f"{ds}__-qaf.json"), _read(ret / f"{ds}__stub__hybrid_fixed+rr.json"))
        _same_metrics(_read(abl / f"{ds}__-rerank.json"), _read(ret / f"{ds}__stub__qaf.json"))
        assert full["config"]["expand"] is False and "expansion" in full["config"]["note"].lower()
        assert full["config"]["chunker"] == "structural"
        for f in abl.glob(f"{ds}__*.json"):
            res = _read(f)
            assert set(res) >= {"config", "metrics", "per_query", "latency_ms"}
            assert set(res["latency_ms"]["total"]) == {"mean", "p95"}
            assert set(res["per_query"]) == set(full["per_query"])
    ch = _read(abl / "toya__-chunking.json")
    assert ch["config"]["chunker"] == "fixed" and ch["config"]["n_chunks"] > 20
    assert (root / "indexes" / "toya_fixed" / "stub" / "meta.json").exists()
    names = [f.stem.split("__")[-1] for f in abl.glob("toya__*.json")]
    assert {mpa._ablation_key(n) for n in names} == {"full", "chunking", "qaf", "rerank"}
    summ = _read(root / "results" / "latency_summary.json")
    assert summ["config"] == "full" and set(summ["datasets"]) == {"toya", "toyb"}
    for st in summ["datasets"].values():
        assert set(st["total"]) == {"mean", "p95"}
    assert set(summ["mean_over_datasets"]["total"]) == {"mean", "p95"}


def test_ablation_expansion(toy_env, monkeypatch):
    root = toy_env
    _run_chain()
    import hrag.pipeline
    import hrag.rewrite

    fake = lambda q: q + " study protein"  # noqa: E731
    monkeypatch.setattr(hrag.rewrite, "expand_query", fake)
    monkeypatch.setattr(hrag.pipeline, "expand_query", fake)
    monkeypatch.setattr(run_ablation, "_generation_running", lambda: False)
    assert run_ablation.main(["--datasets", "toya", "--encoder", "stub", "--latency-n", "3",
                              "--expansion"]) == 0
    abl = root / "results" / "ablation"
    full, noexp = _read(abl / "toya__full.json"), _read(abl / "toya__-expansion.json")
    assert full["config"]["expand"] is True and noexp["config"]["expand"] is False
    _same_metrics(noexp, _read(root / "results" / "retrieval" / "toya__stub__qaf+rr.json"))
    assert _read(abl / "toya__-qaf.json")["config"]["expand"] is True


def test_ablation_expansion_skipped_while_generation_runs(toy_env, monkeypatch):
    root = toy_env
    _run_chain()
    monkeypatch.setattr(run_ablation, "_generation_running", lambda: True)
    assert run_ablation.main(["--datasets", "toya", "--encoder", "stub", "--latency-n", "3",
                              "--expansion"]) == 0
    abl = root / "results" / "ablation"
    assert not (abl / "toya__-expansion.json").exists()
    full = _read(abl / "toya__full.json")
    assert full["config"]["expand"] is False and "skipped" in full["config"]["note"].lower()


def test_matryoshka_writes_contract_files(toy_env, monkeypatch):
    root = toy_env
    _run_chain()
    assert run_matryoshka.main(["--datasets", "toya", "toyb", "--encoder", "stub",
                                "--dims", "16", "8", "4", "--latency-n", "3"]) == 0
    mdir = root / "results" / "matryoshka"
    for ds in ("toya", "toyb"):
        for dim in (16, 8, 4):
            for cfg in ("dense", "qaf+rr"):
                res = _read(mdir / f"{ds}__{dim}__{cfg}.json")
                assert res["config"]["name"] == cfg and res["config"]["dim"] == dim
                assert res["index_mb"] > 0
                assert res["latency_ms"]["total"]["mean"] > 0
                assert len(res["per_query"]) == 5
        # full dimension == the core retrieval run
        _same_metrics(_read(mdir / f"{ds}__16__dense.json"),
                      _read(root / "results" / "retrieval" / f"{ds}__stub__dense.json"))
        _same_metrics(_read(mdir / f"{ds}__16__qaf+rr.json"),
                      _read(root / "results" / "retrieval" / f"{ds}__stub__qaf+rr.json"))
    # QAF retrained and alpha* re-selected per reduced dim (pooled dev only)
    for dim in (8, 4):
        assert (root / "indexes" / f"qaf_stub_{dim}.joblib").exists()
        a = _read(mdir / "qaf" / f"alpha_star_stub_{dim}.json")
        assert set(a["n_dev"]) == {"toya", "toyb"}
        assert _read(mdir / f"toya__{dim}__qaf+rr.json")["config"]["alpha_star"] == a["alpha_star"]
    # truncated index = first d dims of the stored vectors, L2-renormalised
    import faiss

    full = faiss.read_index(str(root / "indexes" / "toya" / "stub" / "index.faiss"))
    small = faiss.read_index(str(root / "indexes" / "toya" / "stub_4" / "index.faiss"))
    v, w = full.reconstruct_n(0, full.ntotal)[:, :4], small.reconstruct_n(0, small.ntotal)
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    assert np.allclose(w, v / np.where(nv == 0, 1, nv), atol=1e-6)
    # loads through the paper-asset loader
    monkeypatch.setattr(mpa, "DATASETS", (*mpa.DATASETS, "toya", "toyb"))
    rows = mpa.load_matryoshka(root / "results")
    assert len(rows) == 12
    assert all(r["ndcg"] is not None and r["index_mb"] and r["query_ms"] for r in rows)


def test_rerank_depth_writes_files(toy_env):
    root = toy_env
    _run_chain()
    assert run_rerank_depth.main(["--datasets", "toya", "--encoder", "stub", "--depths", "3", "5",
                                  "50", "--latency-n", "3"]) == 0
    rdir = root / "results" / "rerank_depth"
    for n in (3, 5, 50):
        res = _read(rdir / f"toya__stub__qaf+rr_depth{n}.json")
        assert res["config"]["rerank_depth"] == n
        assert set(res["latency_ms"]["total"]) == {"mean", "p95"}
    _same_metrics(_read(rdir / "toya__stub__qaf+rr_depth50.json"),
                  _read(root / "results" / "retrieval" / "toya__stub__qaf+rr.json"))
