"""run_generation on the toy fixtures with a stubbed LLM: file contract + offline context."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_experiments_smoke import _run_chain, toy_env  # noqa: E402,F401

import common  # noqa: E402
import run_generation  # noqa: E402

from hrag import generate  # noqa: E402


def test_run_generation_contract(toy_env, monkeypatch):
    _run_chain()
    orig = common.LOADERS["toya"]

    def with_answers(split):
        c, q, r, _ = orig(split)
        return c, q, r, {k: [f"{k[-2:]} study"] for k in q}

    monkeypatch.setitem(common.LOADERS, "toya", with_answers)
    monkeypatch.setattr(common, "SUBSETS_DIR", toy_env / "subsets")
    monkeypatch.setattr(run_generation, "RERANKER_NAME", "stub-ce")
    calls = []

    def fake_chat(prompt, system="", **kw):
        calls.append(prompt)
        return "an answer [1]"

    monkeypatch.setattr(generate, "chat", fake_chat)
    assert run_generation.main(["--dataset", "toya", "--n", "3", "--encoder", "stub"]) == 0
    gen = toy_env / "results" / "generation"
    for cfg in run_generation.GEN_CONFIGS:
        res = json.loads((gen / f"toya__{cfg}.json").read_text())
        assert set(res["metrics"]) == {"em", "f1", "citation_hit"}
        assert res["config"]["encoder"] == "stub" and res["config"]["n"] == 3
        assert len(res["per_query"]) == 3
        for pq in res["per_query"].values():
            assert {"em", "f1", "citation_hit", "answer", "cited", "retrieved"} <= set(pq)
            assert len(pq["retrieved"]) == 5
            assert pq["cited"] == [pq["retrieved"][0]]
    assert len(calls) == 12
    rows = list(csv.reader((gen / "human_check.csv").open()))
    assert rows[0] == ["qid", "config", "answer", "faithful(0/1)"]
    assert len(rows) == 1 + 12 and all(r[3] == "" for r in rows[1:])
    # rerun skips existing files (no new LLM calls)
    assert run_generation.main(["--dataset", "toya", "--n", "3", "--encoder", "stub"]) == 0
    assert len(calls) == 12


def test_score_answer_strips_citation_markers():
    from hrag.generate import Answer

    r = run_generation.score_answer(Answer("The Peach [1][2, 3]", ["d1", "d2"]), ["peach"], {"d2": 1})
    assert r["em"] == 1.0 and r["f1"] == 1.0 and r["citation_hit"] == 1.0
    assert r["answer"] == "The Peach [1][2, 3]"
    assert run_generation.score_answer(Answer("x", []), ["peach"], {"d2": 1})["citation_hit"] == 0.0
