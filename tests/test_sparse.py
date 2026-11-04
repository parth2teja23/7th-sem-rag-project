from hrag.sparse import SparseIndex
from hrag.types import Doc


def _docs():
    return {
        "a": Doc("a", "The IETF published RFC 9110 describing HTTP semantics.", title="HTTP"),
        "b": Doc("b", "Cats sleep most of the day and hunt at night.", title="Cats"),
        "c": Doc("c", "Bread is baked from flour, water and yeast.", title="Bread"),
    }


def _idx():
    i = SparseIndex()
    i.build(_docs())
    return i


def test_exact_token_match_ranks_first():
    r = _idx().search("RFC 9110")
    assert r[0][0] == "a"
    assert all(s > 0 for _, s in r)


def test_title_is_indexed():
    assert _idx().search("bread")[0][0] == "c"


def test_all_stopword_query_empty():
    assert _idx().search("the of and") == []


def test_unknown_and_empty_query_empty():
    assert _idx().search("zzzxqv") == []
    assert _idx().search("") == []


def test_k_limits():
    assert len(_idx().search("cats bread rfc", k=2)) <= 2


def test_save_load_roundtrip(tmp_path):
    i = _idx()
    i.save(tmp_path / "idx")
    j = SparseIndex.load(tmp_path / "idx")
    for q in ["RFC 9110", "cats night", "bread yeast"]:
        assert j.search(q) == i.search(q)
    assert j.idf("rfc") == i.idf("rfc")


def test_idf():
    i = _idx()
    assert i.idf("rfc") > 0
    assert i.idf("the") == 0.0
    assert i.idf("zzzxqv") == 0.0
    assert i.idf("rfc") > i.idf("the")
