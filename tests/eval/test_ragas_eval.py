import pytest

from hrag.eval.ragas_eval import ragas_scores

GOOD = {"question": "q?", "answer": "a", "contexts": ["ctx one", "ctx two"]}


def stub(samples):
    return {"faithfulness": 0.5, "context_precision": 1.0, "n": len(samples)}


def test_returns_evaluator_scores():
    out = ragas_scores([GOOD, dict(GOOD)], evaluator=stub)
    assert out["faithfulness"] == 0.5 and out["n"] == 2


@pytest.mark.parametrize("bad", [
    {**GOOD, "contexts": []},
    {**GOOD, "contexts": ["  "]},
    {**GOOD, "contexts": "not a list"},
    {"question": "q", "answer": "a"},
    {**GOOD, "answer": ""},
    {**GOOD, "question": " "},
])
def test_invalid_sample_raises(bad):
    with pytest.raises(ValueError):
        ragas_scores([GOOD, bad], evaluator=stub)


def test_empty_samples_raises():
    with pytest.raises(ValueError):
        ragas_scores([], evaluator=stub)
