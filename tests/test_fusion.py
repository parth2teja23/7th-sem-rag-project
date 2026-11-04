import pytest

from hrag.fusion import weighted_rrf


def ids(r):
    return [d for d, _ in r]


def test_basic_order_and_score():
    r = weighted_rrf([("a", 3.0), ("b", 2.0)], [("b", 9.0), ("c", 1.0)], alpha=0.5)
    assert ids(r) == ["b", "a", "c"]
    assert dict(r)["b"] == pytest.approx(0.5 / 61 + 0.5 / 62)


def test_alpha_extremes():
    dense = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
    sparse = [("c", 5.0), ("d", 4.0), ("a", 1.0)]
    assert ids(weighted_rrf(dense, sparse, alpha=1.0))[:3] == ["a", "b", "c"]
    assert ids(weighted_rrf(dense, sparse, alpha=0.0))[:3] == ["c", "d", "a"]


def test_empty_sparse_gives_dense_order():
    dense = [("a", 1.0), ("b", 0.5)]
    assert ids(weighted_rrf(dense, [], alpha=0.5)) == ["a", "b"]
    assert ids(weighted_rrf([], dense, alpha=0.5)) == ["a", "b"]


def test_doc_appears_once():
    r = weighted_rrf([("a", 1), ("b", 1)], [("a", 1), ("b", 1)], alpha=0.5)
    assert sorted(ids(r)) == ["a", "b"]


def test_duplicate_in_list_uses_first_position():
    r = weighted_rrf([("a", 1), ("b", 1), ("a", 1)], [], alpha=1.0)
    assert dict(r)["a"] == pytest.approx(1 / 61)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError):
        weighted_rrf([("a", 1.0)], [], alpha=alpha)


def test_top_n_and_tie_break():
    r = weighted_rrf([("b", 1), ("a", 1)], [("a", 1), ("b", 1)], alpha=0.5, top_n=1)
    assert ids(r) == ["a"]
