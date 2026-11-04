"""HotpotQA (distractor setting) loader with a seeded question subset."""
from __future__ import annotations

import json
import random

from datasets import load_dataset

from hrag.data.beir import SEED, SUBSETS_DIR  # noqa: F401 (SUBSETS_DIR patchable here)
from hrag.types import Doc, Qrels


def load_hotpot(
    n_questions: int = 500,
) -> tuple[dict[str, Doc], dict[str, str], Qrels, dict[str, list[str]]]:
    """Load ``n_questions`` seeded-random HotpotQA validation questions (distractor).

    Returns (corpus, queries, qrels, answers). Corpus is the union of the 10 paragraphs
    of every sampled question; doc id is ``"<title>::<para_idx>"`` and docs are
    deduplicated by title (first occurrence wins). Qrels are the supporting-fact
    paragraphs (relevance 1). Question and doc IDs are saved to ``data/subsets/hotpot_n<n>.json``.
    Note: the id of a doc whose title is shared by several questions depends on which
    question first contributes it, so ids (para_idx suffix) can vary with ``n``.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = sorted(random.Random(SEED).sample(range(len(ds)), n_questions))

    corpus: dict[str, Doc] = {}
    title_to_id: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: Qrels = {}
    answers: dict[str, list[str]] = {}
    for i in idx:
        row = ds[i]
        qid = row["id"]
        ctx = row["context"]
        for p, (title, sents) in enumerate(zip(ctx["title"], ctx["sentences"])):
            if title not in title_to_id:
                did = f"{title}::{p}"
                title_to_id[title] = did
                corpus[did] = Doc(id=did, text=" ".join(s.strip() for s in sents), title=title)
        queries[qid] = row["question"]
        answers[qid] = [row["answer"]]
        qrels[qid] = {
            title_to_id[t]: 1 for t in set(row["supporting_facts"]["title"]) if t in title_to_id
        }

    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    (SUBSETS_DIR / f"hotpot_n{n_questions}.json").write_text(
        json.dumps({"qids": list(queries), "doc_ids": list(corpus)})
    )
    return corpus, queries, qrels, answers
