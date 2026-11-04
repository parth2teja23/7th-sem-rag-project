"""RAGAS scoring (Tier B): faithfulness + context relevance/precision with a local Ollama judge.

``ragas_scores`` validates its input first (cheap, no model needed) and then hands the samples
to an evaluator.  The default evaluator imports ragas lazily, so the rest of the package works
even when ragas is not importable.
"""
from __future__ import annotations

from typing import Callable

JUDGE_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"


def _validate(samples: list[dict]) -> None:
    if not samples:
        raise ValueError("ragas_scores: no samples")
    for i, s in enumerate(samples):
        for key in ("question", "answer"):
            v = s.get(key) if isinstance(s, dict) else None
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"sample {i}: {key!r} must be a non-empty string")
        ctx = s.get("contexts")
        if (not isinstance(ctx, (list, tuple)) or not ctx
                or any(not isinstance(c, str) or not c.strip() for c in ctx)):
            raise ValueError(f"sample {i}: 'contexts' must be a non-empty list of non-empty strings")


def _ragas_rows(samples: list[dict]) -> list[dict[str, float]]:
    """Per-sample ragas scores (faithfulness + context relevance / precision)."""
    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    import ragas.metrics as M

    llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, seed=42))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    metrics = [M.faithfulness]
    for name in ("ContextRelevance", "LLMContextPrecisionWithoutReference"):
        if hasattr(M, name):
            metrics.append(getattr(M, name)())
            break
    ds = Dataset.from_list([
        {"user_input": s["question"], "response": s["answer"], "retrieved_contexts": list(s["contexts"])}
        for s in samples])
    res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, show_progress=False,
                   run_config=RunConfig(timeout=600, max_workers=1))
    df = res.to_pandas()
    cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts")]
    return [{c: (None if v != v else float(v)) for c, v in row.items()} for row in df[cols].to_dict("records")]


def ragas_per_sample(samples: list[dict],
                     evaluator: Callable[[list[dict]], list[dict]] | None = None) -> list[dict]:
    """Validated per-sample scores [{metric: value|None}], aligned with ``samples``."""
    _validate(samples)
    return (evaluator or _ragas_rows)(samples)


def _ragas_evaluator(samples: list[dict]) -> dict[str, float]:
    rows = _ragas_rows(samples)
    out: dict[str, float] = {}
    for c in rows[0]:
        v = [r[c] for r in rows if r.get(c) is not None]
        if v:
            out[c] = sum(v) / len(v)
    return out


def ragas_scores(samples: list[dict],
                 evaluator: Callable[[list[dict]], dict[str, float]] | None = None) -> dict[str, float]:
    """samples: [{question, answer, contexts: [str, ...]}] -> {metric: mean score}."""
    _validate(samples)
    return (evaluator or _ragas_evaluator)(samples)


if __name__ == "__main__":  # python -m hrag.eval.ragas_eval samples.json out.json  (per-sample rows)
    import json
    import sys

    rows = ragas_per_sample(json.load(open(sys.argv[1])))
    json.dump(rows, open(sys.argv[2], "w"))
