"""AppTest smoke test of the Streamlit demo on a toy fixture (no real models/indexes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import backend  # noqa: E402

from hrag.llm import OllamaUnavailable  # noqa: E402
from hrag.qaf import FEATURE_NAMES, QAFModel  # noqa: E402
from hrag.types import Doc  # noqa: E402

CORPUS = {f"d{i}": Doc(id=f"d{i}", title=f"Title {i}", text=f"toy text {i} " * 5) for i in range(8)}


class Sparse:
    def search(self, q, k=100):
        return [(f"d{i}", 10.0 - i) for i in range(6)]

    def idf(self, t):
        return 1.0


class Dense:
    encoder = None

    def search(self, q, k=100):
        return [(f"d{i}", 1.0 - 0.1 * i) for i in (5, 4, 3, 2, 1, 0)]


class Reranker:
    def rerank(self, q, ranking, corpus, depth=50):
        return sorted(ranking[:depth], key=lambda x: x[0])


def toy_backend(dataset, encoder):
    rng = np.random.default_rng(0)
    qaf = QAFModel().fit(rng.normal(size=(20, len(FEATURE_NAMES))), rng.uniform(size=20))
    return backend.Backend(CORPUS, {"q1": "toy question one"}, {"q1": {"d0": 1}},
                           Sparse(), Dense(), Reranker(), qaf, set())


@pytest.fixture
def app(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr(backend, "load_backend", toy_backend)
    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=30)


def test_empty_query_shows_info(app):
    app.run()
    assert not app.exception
    assert [i.value for i in app.info] == ["Type a question"]


def test_ollama_down_warns_and_shows_four_columns(app, monkeypatch):
    def down(*a, **k):
        raise OllamaUnavailable("down")

    monkeypatch.setattr("hrag.generate.chat", down)
    app.run()
    app.text_input(key="query").set_value("toy question one").run()
    assert not app.exception
    assert len(app.warning) == 1 and "ollama" in app.warning[0].value.lower()
    headers = [s.value for s in app.subheader]
    assert headers == ["BM25", "Dense", "Fused", "Reranked"]
    assert any("✅" in m.value for m in app.markdown)  # relevant doc marked


def test_qaf_mode_and_answer(app, monkeypatch):
    from hrag.generate import Answer

    monkeypatch.setattr("hrag.generate.chat", lambda *a, **k: "An answer [1].")
    app.run()
    app.radio[0].set_value("QAF")
    app.text_input(key="query").set_value("toy question one").run()
    assert not app.exception
    assert not app.warning
    assert any(h.value == "QAF" for h in app.header)
    assert len(app.dataframe) == 1
