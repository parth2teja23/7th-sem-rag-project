import httpx
import ollama
import pytest

from hrag import generate, llm, rewrite
from hrag.types import Doc


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "cache" / "llm.sqlite")


class FakeClient:
    def __init__(self, reply="hello", exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def chat(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return {"message": {"content": self.reply}}


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(llm, "_client", lambda: client)
    return client


def test_chat_connection_error_maps_to_unavailable(monkeypatch):
    _patch_client(monkeypatch, FakeClient(exc=httpx.ConnectError("refused")))
    with pytest.raises(llm.OllamaUnavailable, match="ollama pull"):
        llm.chat("hi")


def test_chat_model_not_found_maps_to_unavailable(monkeypatch):
    _patch_client(monkeypatch, FakeClient(exc=ollama.ResponseError("model not found", 404)))
    with pytest.raises(llm.OllamaUnavailable, match="ollama pull"):
        llm.chat("hi")


def test_chat_other_errors_propagate(monkeypatch):
    _patch_client(monkeypatch, FakeClient(exc=ValueError("boom")))
    with pytest.raises(ValueError):
        llm.chat("hi")


def test_chat_options_and_cache(monkeypatch):
    c = _patch_client(monkeypatch, FakeClient("out"))
    assert llm.chat("p", system="s") == "out"
    assert llm.chat("p", system="s") == "out"
    assert len(c.calls) == 1
    assert c.calls[0]["options"] == {"temperature": 0, "seed": 42}
    assert c.calls[0]["format"] is None
    assert [m["role"] for m in c.calls[0]["messages"]] == ["system", "user"]
    llm.chat("p", system="s", json_mode=True)
    assert len(c.calls) == 2 and c.calls[1]["format"] == "json"


def test_expand_query_fallback(monkeypatch):
    def boom(*a, **k):
        raise llm.OllamaUnavailable("x")

    monkeypatch.setattr(rewrite, "chat", boom)
    assert rewrite.expand_query("who wrote it") == "who wrote it"


def test_expand_query_appends_and_truncates(monkeypatch):
    monkeypatch.setattr(rewrite, "chat", lambda *a, **k: ", ".join(f"k{i}" for i in range(30)))
    out = rewrite.expand_query("q")
    assert out.split() == ["q"] + [f"k{i}," .rstrip(",") for i in range(20)]
    monkeypatch.setattr(rewrite, "chat", lambda *a, **k: "  ")
    assert rewrite.expand_query("q") == "q"


def test_generate_answer_cites(monkeypatch):
    seen = {}

    def fake(prompt, system="", **k):
        seen["prompt"], seen["system"] = prompt, system
        return " Paris [2][2][9][1] "

    monkeypatch.setattr(generate, "chat", fake)
    ps = [Doc("a", "Lyon text", title="T1"), Doc("b", "Paris text", title="T2")]
    ans = generate.generate_answer("capital?", ps)
    assert ans.text == "Paris [2][2][9][1]"
    assert ans.cited == ["b", "a"]
    assert "[1] (T1) Lyon text" in seen["prompt"] and "[2] (T2) Paris text" in seen["prompt"]
    assert "not in context" in seen["system"]


@pytest.mark.slow
def test_generate_answer_real_ollama():
    import time

    t = time.time()
    ans = generate.generate_answer(
        "What is the capital of France?", [Doc("d1", "Paris is the capital of France.")]
    )
    print("latency_s", time.time() - t)
    assert "Paris" in ans.text
    assert ans.cited == ["d1"]


def test_empty_reply_not_cached(monkeypatch):
    c = _patch_client(monkeypatch, FakeClient("  "))
    llm.chat("p")
    llm.chat("p")
    assert len(c.calls) == 2


def test_stored_empty_is_miss(monkeypatch):
    c = _patch_client(monkeypatch, FakeClient("ok"))
    llm._cache_put(llm._key(llm.DEFAULT_MODEL, "", "p", False), "")
    assert llm.chat("p") == "ok"
    assert len(c.calls) == 1


def test_timeout_maps_to_unavailable(monkeypatch):
    _patch_client(monkeypatch, FakeClient(exc=httpx.ConnectTimeout("t")))
    with pytest.raises(llm.OllamaUnavailable):
        llm.chat("hi")


def test_unrelated_404_not_mapped(monkeypatch):
    _patch_client(monkeypatch, FakeClient(exc=ollama.ResponseError("boom", 500)))
    with pytest.raises(ollama.ResponseError):
        llm.chat("hi")


def test_cache_path_anchored():
    from pathlib import Path
    import hrag.llm as m
    assert Path(m.__file__).resolve().parents[2].joinpath("results/cache/llm.sqlite").name == "llm.sqlite"


def test_generate_answer_grouped_citations(monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda *a, **k: "X [1, 2] and [2,3] [7]")
    ps = [Doc("a", "t"), Doc("b", "t"), Doc("c", "t")]
    assert generate.generate_answer("q", ps).cited == ["a", "b", "c"]
