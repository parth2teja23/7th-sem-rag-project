import numpy as np
import pytest

import hrag.pipeline as pl
from hrag.pipeline import Pipeline, PipelineConfig
from hrag.types import Doc

CORPUS = {f"d{i}": Doc(id=f"d{i}", text=f"unique text number {i} " + "w%d " % i * 5) for i in range(8)}


class StubSparse:
    def __init__(self):
        self.queries = []

    def search(self, query, k=100):
        self.queries.append(query)
        return [(f"d{i}", 10.0 - i) for i in range(6)]

    def idf(self, term):
        return 1.0


class StubDense:
    encoder = None

    def __init__(self):
        self.queries = []

    def search(self, query, k=100):
        self.queries.append(query)
        return [(f"d{i}", 1.0 - 0.1 * i) for i in (5, 4, 3, 2, 1, 0)]


class StubReranker:
    def rerank(self, query, ranking, corpus, depth=50):
        return sorted(ranking[:depth], key=lambda x: x[0])


class StubQAF:
    def predict(self, X):
        return np.array([0.2])


def make(**kw):
    qaf = kw.pop("qaf", None)
    sparse = kw.pop("sparse", StubSparse())
    dense = kw.pop("dense", StubDense())
    reranker = kw.pop("reranker", StubReranker())
    return Pipeline(PipelineConfig(**kw), CORPUS, sparse, dense, reranker, qaf)


@pytest.mark.parametrize("q", ["", "   "])
def test_empty_query(q):
    with pytest.raises(ValueError, match="empty query"):
        make().retrieve(q)


def test_qaf_alpha():
    r = make(alpha_mode="qaf", qaf=StubQAF()).retrieve("hello world")
    assert r.alpha == 0.2


def test_qaf_requires_model():
    with pytest.raises(ValueError):
        make(alpha_mode="qaf")


def test_no_retrievers():
    with pytest.raises(ValueError):
        make(use_dense=False, use_sparse=False)


def test_dat_not_implemented():
    with pytest.raises(NotImplementedError):
        make(alpha_mode="dat").retrieve("hello")


def test_no_rerank_stage():
    r = make(rerank=False).retrieve("hello")
    assert "reranked" not in r.stages
    assert {"sparse", "dense", "fused"} <= set(r.stages)
    assert r.alpha == 0.5


def test_rerank_stage_and_final_k():
    r = make(final_k=3).retrieve("hello")
    assert "reranked" in r.stages
    assert len(r.final) <= 3


def test_timings():
    r = make().retrieve("hello")
    assert set(r.timings_ms) == {"expand", "sparse", "dense", "features", "fusion", "rerank", "post"}
    assert all(v >= 0 for v in r.timings_ms.values())
    assert r.timings_ms["expand"] == 0.0 and r.timings_ms["features"] == 0.0


def test_dense_only():
    r = make(use_sparse=False, rerank=False, final_k=3, sparse=None).retrieve("hello")
    assert [d for d, _ in r.final] == ["d5", "d4", "d3"]
    assert r.alpha is None
    assert "fused" not in r.stages and "sparse" not in r.stages
    assert r.timings_ms["fusion"] == 0.0 and r.timings_ms["sparse"] == 0.0


def test_expansion_uses_expanded_text(monkeypatch):
    monkeypatch.setattr(pl, "expand_query", lambda q: q + " extra")
    sp, de = StubSparse(), StubDense()
    r = make(expand=True, sparse=sp, dense=de).retrieve("hello")
    assert sp.queries == ["hello extra"] and de.queries == ["hello extra"]
    assert r.query == "hello"


def test_answer(monkeypatch):
    seen = {}

    def fake(q, passages):
        seen["p"] = passages
        return "ANS"

    monkeypatch.setattr(pl, "generate_answer", fake)
    r, a = make(final_k=2).answer("hello")
    assert a == "ANS" and [p.id for p in seen["p"]] == [d for d, _ in r.final]


def test_dat_mode_calls_dat_alpha(monkeypatch):
    import sys
    import types

    calls = {}

    def fake_dat(query, dense_top, sparse_top):
        calls["a"] = (query, dense_top, sparse_top)
        return 0.7

    fake = types.ModuleType("hrag.baselines.dat")
    fake.dat_alpha = fake_dat
    monkeypatch.setitem(sys.modules, "hrag.baselines.dat", fake)
    r = make(alpha_mode="dat").retrieve("hello")
    assert r.alpha == 0.7
    assert calls["a"] == ("hello", CORPUS["d5"], CORPUS["d0"])


def test_vocab_from_encoder():
    from types import SimpleNamespace as NS

    vocab = {"##ing": 1, "▁The": 2, "Ġcat": 3, "a": 4, "[CLS]": 5, "x-y": 6}
    enc = NS(model=NS(tokenizer=NS(get_vocab=lambda: vocab)))
    assert pl.vocab_from_encoder(enc) == {"ing", "the", "cat"}
