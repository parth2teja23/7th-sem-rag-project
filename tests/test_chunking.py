import numpy as np

from hrag.chunking import fixed_chunks, semantic_chunks
from hrag.types import Doc


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_headings_start_new_chunks_and_set_section():
    doc = Doc(id="d", text="1. Intro\n\npara a\n\n2. Methods\n\npara b", meta={"k": 1})
    chunks = semantic_chunks(doc)
    assert len(chunks) == 2
    assert [c.meta["section"] for c in chunks] == ["1. Intro", "2. Methods"]
    assert [c.id for c in chunks] == ["d#0", "d#1"]
    assert all(c.meta["parent"] == "d" and c.meta["k"] == 1 for c in chunks)
    assert chunks[0].text.startswith("1. Intro") and chunks[1].text.startswith("2. Methods")
    assert "para a" in chunks[0].text and "para b" in chunks[1].text
    assert doc.meta == {"k": 1}


def test_long_paragraph_respects_max_words_plus_overlap():
    doc = Doc(id="d", text=words(1000))
    chunks = semantic_chunks(doc, target=300, max_words=380, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 380 + 50 for c in chunks)
    # all original words are covered
    assert chunks[-1].text.split()[-1] == "w999"


def test_overlap_prefix_from_previous_chunk_same_section():
    doc = Doc(id="d", text=words(1000))
    chunks = semantic_chunks(doc, target=300, max_words=380, overlap=50)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.text.split()[:50] == prev.text.split()[-50:]


def test_no_overlap_across_sections():
    doc = Doc(id="d", text="1. Intro\n\n" + words(20) + "\n\n2. Methods\n\n" + words(20, "x"))
    chunks = semantic_chunks(doc)
    assert chunks[1].text.startswith("2. Methods")
    assert "w19" not in chunks[1].text


def test_sentence_split_for_long_paragraph():
    sent = lambda i: " ".join(f"s{i}_{j}" for j in range(99)) + "."
    doc = Doc(id="d", text=" ".join(sent(i) for i in range(10)))
    chunks = semantic_chunks(doc, target=300, max_words=380, overlap=0)
    assert all(len(c.text.split()) <= 380 for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_embed_break_between_dissimilar_paragraphs():
    doc = Doc(id="d", text="alpha one\n\nbeta two\n\ngamma three")
    vecs = {"alpha one": [1, 0], "beta two": [0, 1], "gamma three": [0, 1]}
    embed = lambda paras: np.array([vecs[p] for p in paras], dtype=float)
    chunks = semantic_chunks(doc, overlap=0, embed=embed)
    assert [c.text for c in chunks] == ["alpha one", "beta two\n\ngamma three"]
    # without embed: merged
    assert len(semantic_chunks(doc, overlap=0)) == 1


def test_fixed_chunks():
    doc = Doc(id="d", text=words(600), meta={"a": 2})
    chunks = fixed_chunks(doc, size=256)
    assert [len(c.text.split()) for c in chunks] == [256, 256, 88]
    assert chunks[0].id == "d#0" and chunks[0].meta["parent"] == "d"


def test_fixed_chunks_overlap():
    doc = Doc(id="d", text=words(600))
    chunks = fixed_chunks(doc, size=256, overlap=50)
    assert chunks[1].text.split()[:50] == chunks[0].text.split()[-50:]
    assert chunks[-1].text.split()[-1] == "w599"


RFC = (
    "Network Working Group\n\n"
    "3. Connection Setup\n\n"
    "The handshake proceeds as follows. The sender\n"
    "The sender MUST NOT transmit data\n"
    "until the peer replies. Steps:\n"
    "1.  Send the SYN segment\n"
    "2.  Await the SYN-ACK segment\n"
    "Network Working Group rules apply here.\n\n"
    "4. Teardown\n\n"
    "Either side may close."
)


def test_rfc_body_not_misparsed_and_no_words_lost():
    doc = Doc(id="r", text=RFC)
    chunks = semantic_chunks(doc, overlap=0)
    sections = [c.meta["section"] for c in chunks]
    assert sections == ["Network Working Group", "3. Connection Setup", "4. Teardown"]
    covered = set(" ".join(c.text for c in chunks).split())
    assert set(doc.text.split()) <= covered


def test_char_offsets_are_verbatim_slices():
    docs = [Doc(id="r", text=RFC), Doc(id="w", text=words(1000)),
            Doc(id="s", text="1. Intro\n\n" + words(500) + "\n\n2. Methods\n\n" + words(40, "x"))]
    for doc in docs:
        for c in semantic_chunks(doc) + fixed_chunks(doc, size=100, overlap=10):
            assert doc.text[c.meta["char_start"]:c.meta["char_end"]] == c.text
