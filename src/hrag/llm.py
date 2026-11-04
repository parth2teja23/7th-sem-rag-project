from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import httpx
import ollama

DEFAULT_MODEL = "qwen2.5:7b-instruct"
CACHE_PATH = Path(__file__).resolve().parents[2] / "results" / "cache" / "llm.sqlite"
_HINT = "Start Ollama with `ollama serve` and run `ollama pull qwen2.5:7b-instruct`"


class OllamaUnavailable(RuntimeError):
    pass


def _client() -> ollama.Client:
    return ollama.Client()


def _key(model: str, system: str, prompt: str, json_mode: bool) -> str:
    raw = "\x00".join([model, system, prompt, str(json_mode)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
    return con


def _cache_get(key: str) -> str | None:
    con = _connect()
    try:
        row = con.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    finally:
        con.close()
    return row[0] if row and row[0].strip() else None


def _cache_put(key: str, response: str) -> None:
    con = _connect()
    try:
        with con:
            con.execute("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", (key, response))
    finally:
        con.close()


def chat(prompt: str, system: str = "", model: str = DEFAULT_MODEL, json_mode: bool = False) -> str:
    key = _key(model, system, prompt, json_mode)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        resp = _client().chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "seed": 42},
            format="json" if json_mode else None,
        )
    except (httpx.TransportError, ConnectionError) as e:
        raise OllamaUnavailable(_HINT) from e
    except ollama.ResponseError as e:
        msg = str(e).lower()
        if e.status_code == 404 or ("model" in msg and "not found" in msg):
            raise OllamaUnavailable(_HINT) from e
        raise
    out = resp["message"]["content"]
    if out.strip():
        _cache_put(key, out)
    return out
