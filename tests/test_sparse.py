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


def _ident():
    i = SparseIndex()
    i.build({
        "h2": Doc("h2", "Support for HTTP/2 multiplexing."),
        "h3": Doc("h3", "Support for HTTP/3 over QUIC."),
        "tls": Doc("tls", "Handshake in TLS1.3 is faster."),
        "old": Doc("old", "Handshake in SSL is slow."),
        "s155": Doc("s155", "See section 15.5.1 for redirects."),
        "s154": Doc("s154", "See section 15.4 for other things."),
    })
    return i


def test_http3_beats_http2():
    r = _ident().search("HTTP/3")
    assert r[0][0] == "h3"


def test_tls_spacing_variants():
    i = _ident()
    assert "tls" in [d for d, _ in i.search("TLS 1.3")]
    assert i.search("TLS1.3")[0][0] == "tls"


def test_section_number():
    r = _ident().search("section 15.5.1")
    ids = [d for d, _ in r]
    assert ids[0] == "s155" and ids.index("s155") < ids.index("s154")


def test_build_first_error(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        SparseIndex().save(tmp_path / "x")
