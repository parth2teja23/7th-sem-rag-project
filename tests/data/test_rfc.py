import json

import pytest

from hrag.chunking import fixed_chunks, semantic_chunks
from hrag.data import questions as Q
from hrag.data import rfc
from hrag.types import Doc

RAW = (
    "Request for Comments: 9110                                    R. Fielding\n"
    "                                                                   Adobe\n"
    "                                                               June 2022\n"
    "\n"
    "\n"
    "                            HTTP Semantics\n"
    "\n"
    "Abstract\n"
    "\n"
    "   The Hypertext Transfer Protocol (HTTP) is a stateless protocol.\n"
    "\n"
    "Fielding, et al.             Standards Track                    [Page 1]\n"
    "\f\n"
    "RFC 9110                     HTTP Semantics                    June 2022\n"
    "\n"
    "\n"
    "1.  Introduction\n"
    "\n"
    "   Status code 404 means not found.\n"
)


def test_clean_strips_headers_footers_formfeeds():
    out = rfc.clean_rfc_text(RAW)
    assert "\f" not in out
    assert "[Page 1]" not in out
    assert "RFC 9110                     HTTP Semantics" not in out
    assert "\n\n\n" not in out
    assert "   Status code 404 means not found." in out
    assert "1.  Introduction" in out


def test_title_and_year():
    assert rfc.parse_title(RAW) == "HTTP Semantics"
    assert rfc.parse_year(RAW) == 2022


def test_fetch_rfc_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rfc, "RAW_DIR", tmp_path)
    (tmp_path / "rfc9110.txt").write_text(RAW)
    d = rfc.fetch_rfc(9110)
    assert d.id == "rfc9110" and d.title == "HTTP Semantics"
    assert d.meta == {"rfc": 9110, "year": 2022}


CHUNK = Doc(
    id="rfc1#0",
    text="Section 4.2.1 says the HTTP/1.1 header Content-Length must match the RFC 9110 value.",
    meta={"parent": "rfc1", "section": "4.2.1.  Foo", "char_start": 100, "char_end": 183},
)


def fake(answer, question="What header is it?"):
    return lambda prompt, **kw: json.dumps({"question": question, "answer": answer})


def test_identifier_count():
    assert Q.count_identifiers(CHUNK.text) >= 2
    assert Q.count_identifiers("plain words only here") == 0


def test_accepts_valid_and_span_is_absolute():
    qs = Q.generate_questions([CHUNK], n=1, chat=fake("Content-Length"))
    assert len(qs) == 1
    q = qs[0]
    assert q["qid"] == "rfcq001" and q["verified"] == "" and q["gold_chunk_id"] == "rfc1#0"
    assert "Content-Length" in q["gold_text_snippet"]
    assert q["span_start"] >= 100 and q["span_end"] <= 183
    assert CHUNK.text[q["span_start"] - 100:q["span_end"] - 100] == q["gold_text_snippet"]


@pytest.mark.parametrize("ans,qn", [
    ("X-Not-There", "What?"),                      # not in chunk
    ("one two three four five six seven eight nine", "What?"),  # >8 words (also not in chunk)
    ("Content-Length", "Is Content-Length right?"),  # question contains answer
])
def test_rejections(ans, qn):
    assert Q.generate_questions([CHUNK], n=1, chat=fake(ans, qn)) == []


def test_rejects_bad_json_and_long_answer_in_chunk():
    assert Q.generate_questions([CHUNK], n=1, chat=lambda p, **k: "not json") == []
    c = Doc("rfc1#1", "A1 B2 " + "w " * 10 + "C3", meta={"parent": "rfc1", "section": "s", "char_start": 0, "char_end": 30})
    long_ans = "w w w w w w w w w"
    assert Q.generate_questions([c], n=1, chat=fake(long_ans)) == []


def test_span_qrels_only_overlapping_chunks():
    text = " ".join(f"w{i}" for i in range(100))
    doc = Doc("rfc1", text)
    chunks = fixed_chunks(doc, size=10)
    s = text.index("w25"); e = s + 3
    rel = rfc.span_relevant([c for c in chunks], "rfc1", s, e)
    assert rel == ["rfc1#2"]
    # span straddling chunks 2 and 3
    s = text.index("w29"); e = text.index("w31") + 3
    rel = rfc.span_relevant(chunks, "rfc1", s, e)
    assert rel == ["rfc1#2", "rfc1#3"]
    for c in chunks:
        ov = c.meta["char_start"] < e and s < c.meta["char_end"]
        assert (c.id in rel) == ov


def test_load_requires_verified_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(rfc, "RFC_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="questions_draft.csv"):
        rfc.load_rfc_corpus()


def test_load_corpus_verified_only(tmp_path, monkeypatch):
    monkeypatch.setattr(rfc, "RFC_DIR", tmp_path)
    text = "\n\n".join(f"Para {i} " + " ".join(f"w{j}" for j in range(30)) for i in range(4))
    doc = Doc("rfc1", text, "T", {"rfc": 1, "year": 2000})
    (tmp_path / "docs.jsonl").write_text(json.dumps(
        {"id": "rfc1", "title": "T", "text": text, "meta": doc.meta}) + "\n")
    s = text.index("w5 "); e = s + 2
    rows = [
        {"qid": "rfcq001", "question": "q1", "answer": "w5", "gold_chunk_id": "rfc1#0",
         "gold_text_snippet": "w5", "verified": "1", "notes": "", "span_start": s, "span_end": e},
        {"qid": "rfcq002", "question": "q2", "answer": "w5", "gold_chunk_id": "rfc1#0",
         "gold_text_snippet": "w5", "verified": "0", "notes": "", "span_start": s, "span_end": e},
    ]
    Q.write_csv(rows, tmp_path / "questions.csv")
    corpus, queries, qrels, answers = rfc.load_rfc_corpus("fixed")
    assert set(queries) == {"rfcq001"} and answers == {"rfcq001": ["w5"]}
    assert qrels["rfcq001"] and all(c in corpus for c in qrels["rfcq001"])
    assert all(v == 1 for v in qrels["rfcq001"].values())
