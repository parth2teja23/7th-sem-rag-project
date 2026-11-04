import pytest

from hrag.types import Doc

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reranker():
    from hrag.rerank import Reranker

    return Reranker()


CORPUS = {
    "bad": Doc("bad", "Bananas are yellow"),
    "good": Doc("good", "Paris is the capital of France"),
}


def test_rerank_puts_relevant_first(reranker):
    out = reranker.rerank("capital of France", [("bad", 2.0), ("good", 1.0)], CORPUS)
    assert [d for d, _ in out] == ["good", "bad"]


def test_rerank_depth_limits(reranker):
    out = reranker.rerank("capital of France", [("bad", 2.0), ("good", 1.0)], CORPUS, depth=1)
    assert len(out) == 1


def test_rerank_edge_cases(reranker):
    assert reranker.rerank("q", [], CORPUS) == []
    assert reranker.rerank("q", [("bad", 1.0)], CORPUS, depth=0) == []
    assert reranker.rerank("q", [("missing", 1.0)], CORPUS) == []
