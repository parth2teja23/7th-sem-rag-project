"""BEIR dataset loaders (SciFact, FiQA) from the Hugging Face Hub."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from datasets import load_dataset

from hrag.types import Doc, Qrels

SEED = 42
SUBSETS_DIR = Path(__file__).resolve().parents[3] / "data" / "subsets"

# HF split names of the BeIR/<name>-qrels repos
_HF_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


def _qrels_for_split(name: str, split: str) -> Qrels:
    """Positive-relevance qrels for the requested split (SciFact dev carved from train)."""
    carve = name == "scifact" and split in ("train", "dev")
    hf_split = "train" if carve else _HF_SPLIT[split]
    ds = load_dataset(f"BeIR/{name}-qrels", split=hf_split)
    qrels: Qrels = {}
    for row in ds:
        score = int(row["score"])
        if score > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    if carve:
        qids = sorted(qrels)
        dev = set(random.Random(SEED).sample(qids, round(0.2 * len(qids))))
        keep = dev if split == "dev" else set(qids) - dev
        qrels = {q: r for q, r in qrels.items() if q in keep}
    return qrels


def load_beir(
    name: Literal["scifact", "fiqa"],
    split: Literal["train", "dev", "test"],
    max_docs: int | None = None,
) -> tuple[dict[str, Doc], dict[str, str], Qrels]:
    """Load a BEIR dataset: (corpus, queries, qrels).

    Sources: ``BeIR/<name>`` and ``BeIR/<name>-qrels``. Only qrels with relevance > 0
    are kept; queries are restricted to those having qrels in ``split``. SciFact has no
    dev split, so dev = 20% of train queries (seed 42) and train = the remaining 80%.

    If ``max_docs`` is set, the corpus is subsampled deterministically (seed 42): all
    qrel-positive docs for the loaded split are kept (even if that exceeds ``max_docs``),
    then random fill up to ``max_docs``. Chosen doc IDs are written to
    ``data/subsets/<name>_docs.json`` (note: overwritten per call/split).
    """
    qrels = _qrels_for_split(name, split)

    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    corpus = {
        str(r["_id"]): Doc(id=str(r["_id"]), text=r["text"], title=r.get("title") or "")
        for r in corpus_ds
    }
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    queries = {str(r["_id"]): r["text"] for r in queries_ds if str(r["_id"]) in qrels}

    if max_docs is not None:
        positives = sorted({d for rel in qrels.values() for d in rel if d in corpus})
        pos_set = set(positives)
        others = sorted(d for d in corpus if d not in pos_set)
        n_fill = max(0, max_docs - len(positives))
        fill = random.Random(SEED).sample(others, min(n_fill, len(others)))
        chosen = sorted(pos_set | set(fill))
        corpus = {d: corpus[d] for d in chosen}
        SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
        (SUBSETS_DIR / f"{name}_docs.json").write_text(json.dumps(chosen))

    qrels = {q: {d: s for d, s in rel.items() if d in corpus} for q, rel in qrels.items()}
    return corpus, queries, qrels
