import numpy as np
import pytest

from hrag.dense import ENCODERS, DenseIndex, Encoder
from hrag.types import Doc

pytestmark = pytest.mark.slow


def _docs():
    return {
        "cat": Doc("cat", "cats purr", title="Cats"),
        "stock": Doc("stock", "stock market crash", title="Finance"),
        "feline": Doc("feline", "feline sounds", title=""),
    }


@pytest.fixture(scope="module", params=list(ENCODERS))
def enc(request):
    return Encoder(request.param)


def test_ranks_cat_doc_first(enc):
    idx = DenseIndex()
    idx.build(_docs(), enc)
    top = idx.search("why do cats make noise", k=3)
    assert top[0][0] in {"cat", "feline"}
    assert len(top) == 3
    batch = idx.search_batch(["why do cats make noise", "stock prices fell"], k=2)
    assert batch[0][0][0] in {"cat", "feline"}
    assert batch[1][0][0] == "stock"


def test_unit_norm_float32(enc):
    docs = list(_docs().values())
    for e in (enc.encode_docs(docs), enc.encode_queries(["hello", "world"])):
        assert e.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    assert enc.max_len > 0


def test_gemma_matryoshka_dim():
    e = Encoder("gemma", dim=128)
    out = e.encode_docs(list(_docs().values()))
    assert out.shape == (3, 128)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_bge_rejects_dim():
    with pytest.raises(ValueError):
        Encoder("bge-small", dim=128)


def test_save_load_roundtrip(enc, tmp_path):
    idx = DenseIndex()
    idx.build(_docs(), enc)
    before = idx.search("why do cats make noise", k=3)
    idx.save(tmp_path)
    idx2 = DenseIndex.load(tmp_path, enc)
    assert idx2.search("why do cats make noise", k=3)[0][0] == before[0][0]


def test_load_mismatch_raises(tmp_path):
    e1 = Encoder("bge-small")
    idx = DenseIndex()
    idx.build(_docs(), e1)
    idx.save(tmp_path)
    with pytest.raises(ValueError):
        DenseIndex.load(tmp_path, Encoder("gemma", dim=256))


def test_k_larger_than_ntotal(enc):
    idx = DenseIndex()
    idx.build(_docs(), enc)
    res = idx.search("cats", k=50)
    assert len(res) <= 3
    assert all(i in _docs() for i, _ in res)


def test_gemma_dim256_roundtrip(tmp_path):
    e = Encoder("gemma", dim=256)
    idx = DenseIndex()
    idx.build(_docs(), e)
    idx.save(tmp_path)
    idx2 = DenseIndex.load(tmp_path, e)
    q = "why do cats make noise"
    assert idx2.search(q, k=3)[0][0] == idx.search(q, k=3)[0][0]
    assert idx2.search(q, k=3)[0][0] in {"cat", "feline"}
