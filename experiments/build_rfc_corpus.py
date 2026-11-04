"""Fetch the RFC corpus, write docs.jsonl, and generate questions_draft.csv (human verifies next)."""
from __future__ import annotations

import json
import time

from hrag.chunking import semantic_chunks
from hrag.data import questions as Q
from hrag.data.rfc import RFC_DIR, RFC_IDS, fetch_rfc


def main(n_questions: int = 150) -> None:
    t0 = time.time()
    docs = [fetch_rfc(n) for n in RFC_IDS]
    RFC_DIR.mkdir(parents=True, exist_ok=True)
    with (RFC_DIR / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "title": d.title, "text": d.text, "meta": d.meta}) + "\n")
    chunks = [c for d in docs for c in semantic_chunks(d)]
    stats: dict = {}
    rows = Q.generate_questions(chunks, n=n_questions, seed=42, stats=stats)
    Q.write_csv(rows, RFC_DIR / "questions_draft.csv")
    print(f"docs={len(docs)} chunks={len(chunks)} accepted={stats['accepted']} "
          f"rejected={stats['rejected']} rows={len(rows)} time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
