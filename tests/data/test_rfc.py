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
    assert CHUNK.text[q["answer_start"] - 100:q["answer_end"] - 100] == "Content-Length"
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
         "gold_text_snippet": "w5", "verified": "1", "notes": "", "span_start": s, "span_end": e,
         "answer_start": s, "answer_end": e},
        {"qid": "rfcq002", "question": "q2", "answer": "w5", "gold_chunk_id": "rfc1#0",
         "gold_text_snippet": "w5", "verified": "0", "notes": "", "span_start": s, "span_end": e,
         "answer_start": s, "answer_end": e},
    ]
    Q.write_csv(rows, tmp_path / "questions.csv")
    corpus, queries, qrels, answers = rfc.load_rfc_corpus("fixed")
    assert set(queries) == {"rfcq001"} and answers == {"rfcq001": ["w5"]}
    assert qrels["rfcq001"] and all(c in corpus for c in qrels["rfcq001"])
    assert all(v == 1 for v in qrels["rfcq001"].values())


OLD = (
    "RFC:  791\n\n\n                         INTERNET PROTOCOL\n\n"
    "                             September 1981\n\n"
    "1.  Introduction\n\n   Body text one.\n\n"
    "Postel                                                          [Page 1]\n"
    "\n[Page ii]    \n\n\n"
    "September 1981                                                          \n"
    "                                                         Internet Protocol\n\n"
    "   Body text two.\n\n"
    "                                                              [Page iii]\n"
    "\n\n\nRFC:  791\n\n   Body text three.\n"
)


def test_clean_old_style_1981_headers():
    out = rfc.clean_rfc_text(OLD)
    assert "September 1981" not in out and "RFC:" not in out and "[Page" not in out
    assert "Internet Protocol" not in out.split("Introduction")[1]
    assert "Body text one." in out and "Body text two." in out and "Body text three." in out
    assert "\n\n\n" not in out


def mk(answer, question="What is it?"):
    return fake(answer, question)


def gen(answer, question="What header is it?", text=None, docs=None, **kw):
    c = CHUNK if text is None else Doc("rfc1#0", text, meta=CHUNK.meta)
    return Q.generate_questions([c], n=1, chat=fake(answer, question), docs=docs, **kw)


def test_shape_and_ambiguity_filters():
    t = "The RFC 9110 value (see 4.2.1 and HTTP/1.1) uses Content-Length; value 10 appears; foo-bar foo-bar."
    assert gen("(see", text=t) == []                       # unbalanced
    assert gen("HTTP/1.1)", text=t) == []                  # unbalanced
    assert gen("10", text=t) == []                         # too short
    assert gen("foo-bar", text=t) == []                    # twice in chunk
    assert gen("Content-Length", "What does this document say?", text=t) == []  # deictic
    assert gen("Content-Length", text=t) != []
    assert gen("value (see", text=t) == []                 # unbalanced paren
    d = Doc("rfc1", "Content-Length " * 5)
    assert gen("Content-Length", text=t, docs={"rfc1": d}) == []  # >3 in parent doc


def test_truncated_and_duplicates_and_stats():
    t = "The RFC 9110 value (e.g., 4.2.1 and Content-Length) is used."
    assert gen("e.g.,", text=t) == []
    stats = {}
    chunks = [Doc(f"rfc1#{i}", CHUNK.text, meta={**CHUNK.meta, "section": f"s{i}"}) for i in range(2)]
    qs = Q.generate_questions(chunks, n=5, chat=fake("Content-Length"), stats=stats)
    assert len(qs) == 1 and stats["accepted"] == 1 and stats["rej_duplicate_question"] == 1


def test_per_rfc_cap_and_round_robin():
    import re
    cs = [Doc(f"rfc{p}#{i}", f"Use X{p}{i}-Foo and RFC{p}{i}-Bar now.",
              meta={"parent": f"rfc{p}", "section": f"s{i}", "char_start": 0, "char_end": 40})
          for p in (1, 2, 3) for i in range(8)]

    def chat(prompt, **kw):
        tok = re.search(r"X\d+-Foo", prompt).group(0)
        return json.dumps({"question": f"What is {tok[:-4]} called?", "answer": tok})

    qs = Q.generate_questions(cs, n=100, chat=chat, per_rfc=3)
    from collections import Counter
    assert Counter(q["gold_chunk_id"].split("#")[0] for q in qs) == {"rfc1": 3, "rfc2": 3, "rfc3": 3}
    first3 = Q.generate_questions(cs, n=3, chat=chat, per_rfc=3)
    assert len({q["gold_chunk_id"].split("#")[0] for q in first3}) == 3  # round-robin


def test_load_rejects_blank_offsets_and_unknown_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(rfc, "RFC_DIR", tmp_path)
    text = "Para one w1 w2.\n\nPara two."
    (tmp_path / "docs.jsonl").write_text(json.dumps(
        {"id": "rfc1", "title": "T", "text": text, "meta": {}}) + "\n")
    base = {"qid": "rfcq001", "question": "q", "answer": "w1", "gold_chunk_id": "rfc1#0",
            "gold_text_snippet": "w1", "verified": "1", "notes": "", "span_start": 0,
            "span_end": 5, "answer_start": "", "answer_end": ""}
    Q.write_csv([base], tmp_path / "questions.csv")
    with pytest.raises(ValueError, match="rfcq001"):
        rfc.load_rfc_corpus("fixed")
    Q.write_csv([{**base, "answer_start": 1, "answer_end": 3, "gold_chunk_id": "rfc99#0"}],
                tmp_path / "questions.csv")
    with pytest.raises(ValueError, match="rfcq001"):
        rfc.load_rfc_corpus("fixed")
    Q.write_csv([{**base, "verified": "0"}], tmp_path / "questions.csv")
    with pytest.raises(ValueError, match="no verified rows"):
        rfc.load_rfc_corpus("fixed")
    (tmp_path / "questions.csv").write_text("qid,question\nrfcq001,q\n")
    with pytest.raises(ValueError, match="missing columns"):
        rfc.load_rfc_corpus("fixed")


def test_llm_error_skips_chunk():
    def boom(prompt, **kw):
        raise RuntimeError("token repeat limit")
    stats = {}
    assert Q.generate_questions([CHUNK], n=1, chat=boom, stats=stats) == []
    assert stats["rej_llm_error"] == 1
