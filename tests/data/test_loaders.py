import pytest

from hrag.data.beir import load_beir
from hrag.data.hotpot import load_hotpot

pytestmark = pytest.mark.slow


def test_scifact_test_split():
    corpus, queries, qrels = load_beir("scifact", "test")
    assert len(queries) == 300
    assert len(corpus) >= 5000
    assert set(qrels) == set(queries)
    for rel in qrels.values():
        assert rel
        for did, score in rel.items():
            assert score > 0
            assert did in corpus


def test_scifact_dev_train_disjoint():
    _, q_dev, _ = load_beir("scifact", "dev")
    _, q_train, _ = load_beir("scifact", "train")
    assert q_dev and q_train
    assert set(q_dev).isdisjoint(q_train)


def test_hotpot_50():
    corpus, queries, qrels, answers = load_hotpot(50)
    assert len(queries) == 50
    assert set(qrels) == set(queries) == set(answers)
    for qid in queries:
        assert len(qrels[qid]) == 2
        assert all(d in corpus for d in qrels[qid])
        assert answers[qid] and answers[qid][0]


def test_hotpot_deterministic():
    _, q1, _, _ = load_hotpot(50)
    _, q2, _, _ = load_hotpot(50)
    assert list(q1) == list(q2)
