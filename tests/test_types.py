from hrag.device import pick_device
from hrag.types import sort_ranking


def test_ranking_sorted_helper():
    assert sort_ranking([("a", 0.1), ("b", 0.9)]) == [("b", 0.9), ("a", 0.1)]


def test_sort_ranking_ties_by_doc_id():
    assert sort_ranking([("b", 1.0), ("a", 1.0)]) == [("a", 1.0), ("b", 1.0)]


def test_pick_device_returns_known():
    assert pick_device() in {"mps", "cpu"}
