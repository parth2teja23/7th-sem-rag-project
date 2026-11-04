from hrag.postprocess import dedupe, metadata_filter
from hrag.types import Doc

T = "the quick brown fox jumps over the lazy dog again and again"


def test_dedupe_identical_removes_second():
    corpus = {"a": Doc("a", T), "b": Doc("b", T), "c": Doc("c", "totally different words about something else entirely here")}
    r = dedupe([("a", 3.0), ("b", 2.0), ("c", 1.0)], corpus)
    assert r == [("a", 3.0), ("c", 1.0)]


def test_dedupe_short_docs_and_missing_ids():
    corpus = {"a": Doc("a", "Hi there"), "b": Doc("b", "hi  THERE")}
    r = dedupe([("a", 2.0), ("b", 1.0), ("zz", 0.5)], corpus)
    assert r == [("a", 2.0), ("zz", 0.5)]


def test_metadata_filter():
    corpus = {"a": Doc("a", "x", meta={"year": 2020}), "b": Doc("b", "y", meta={"year": 2019})}
    r = [("a", 2.0), ("b", 1.0)]
    assert metadata_filter(r, corpus, lambda m: m.get("year") == 2020) == [("a", 2.0)]
    assert metadata_filter(r, corpus, None) == r
