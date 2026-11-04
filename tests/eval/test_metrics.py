import math

import pytest

from hrag.eval.metrics import (
    evaluate_run,
    exact_match,
    mrr_at_k,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    token_f1,
)

RANK = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]


def test_recall():
    assert recall_at_k(RANK, {"d2": 1}, 1) == 0.0
    assert recall_at_k(RANK, {"d2": 1}, 2) == 1.0


def test_mrr():
    assert mrr_at_k(RANK, {"d2": 1}, 10) == 0.5


def test_ndcg_binary():
    assert ndcg_at_k(RANK, {"d2": 1}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_graded():
    expected = (3 / 1 + 1 / 2) / (3 / 1 + 1 / math.log2(3))
    assert ndcg_at_k(RANK, {"d1": 2, "d3": 1}, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.9639, abs=1e-4)


def test_exact_match_and_f1():
    assert normalize_answer("The Eiffel  Tower!") == "eiffel tower"
    assert exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert exact_match("Paris", ["eiffel tower"]) == 0.0
    assert token_f1("big red dog", ["red dog"]) == pytest.approx(0.8)


def test_evaluate_run_skips_missing_qrels():
    run = {"q1": RANK, "q2": RANK}
    qrels = {"q1": {"d2": 1}}
    mean, per_q = evaluate_run(run, qrels, ks=(1, 2))
    assert set(per_q) == {"q1"}
    assert mean["recall@2"] == 1.0
    assert mean["recall@1"] == 0.0
    assert "mrr@2" in mean and "ndcg@1" in mean
