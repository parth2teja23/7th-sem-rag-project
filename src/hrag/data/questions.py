"""Generate identifier-heavy questions with exact-substring answers from RFC chunks."""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from hrag.types import Doc

CSV_COLUMNS = ["qid", "question", "answer", "gold_chunk_id", "gold_text_snippet", "verified",
               "notes", "span_start", "span_end"]
CONTEXT_CHARS = 20
MAX_PER_SECTION = 2

_TOKEN = re.compile(r"\S+")
_ALLCAPS = re.compile(r"^[A-Z]{2,}$")
_DOTTED = re.compile(r"^\d+(\.\d+)+\.?$")

SYSTEM = "You write retrieval test questions from technical specification text."
PROMPT = """Below is an excerpt from an RFC. Write ONE question that can only be answered with an exact \
identifier, number, header name, status code, field name, or acronym found in the excerpt. The question \
must not contain the answer. The answer must be copied verbatim from the excerpt and be at most 8 words.

Excerpt:
\"\"\"
{chunk}
\"\"\"

Respond with JSON only: {{"question": "...", "answer": "..."}}"""


def _is_identifier(tok: str) -> bool:
    tok = tok.strip(".,;:()[]{}\"'<>")
    if len(tok) < 2:
        return False
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return True
    if _ALLCAPS.match(tok) or _DOTTED.match(tok):
        return True
    return any(ch in tok[1:-1] for ch in "/-_")


def count_identifiers(text: str) -> int:
    return sum(_is_identifier(t) for t in _TOKEN.findall(text))


def _default_chat(prompt: str, **kw) -> str:
    from hrag import llm
    return llm.chat(prompt, **kw)


def _validate(raw: str, chunk_text: str) -> tuple[str, str] | str:
    """Return (question, answer) or a rejection reason."""
    try:
        d = json.loads(raw)
        q, a = str(d["question"]).strip(), str(d["answer"]).strip()
    except (ValueError, KeyError, TypeError):
        return "bad_json"
    if not q or not a:
        return "empty"
    if len(a.split()) > 8:
        return "answer_too_long"
    if a.lower() not in chunk_text.lower():
        return "answer_not_in_chunk"
    if a.lower() in q.lower():
        return "question_contains_answer"
    return q, a


def generate_questions(chunks: list[Doc], n: int = 150, seed: int = 42,
                       chat: Callable[..., str] | None = None,
                       stats: dict | None = None) -> list[dict]:
    chat = chat or _default_chat
    stats = stats if stats is not None else {}
    stats.update(accepted=0, rejected=0)
    cands = [c for c in chunks if count_identifiers(c.text) >= 2]
    random.Random(seed).shuffle(cands)
    per_section: dict[tuple, int] = defaultdict(int)
    out: list[dict] = []
    for c in cands:
        if len(out) >= n:
            break
        key = (c.meta.get("parent"), c.meta.get("section"))
        if per_section[key] >= MAX_PER_SECTION:
            continue
        res = _validate(chat(PROMPT.format(chunk=c.text), system=SYSTEM, json_mode=True), c.text)
        if isinstance(res, str):
            stats["rejected"] += 1
            continue
        q, a = res
        i = c.text.lower().index(a.lower())
        ans = c.text[i:i + len(a)]
        s, e = max(0, i - CONTEXT_CHARS), min(len(c.text), i + len(a) + CONTEXT_CHARS)
        base = c.meta["char_start"]
        per_section[key] += 1
        stats["accepted"] += 1
        out.append({
            "qid": f"rfcq{len(out) + 1:03d}", "question": q, "answer": ans,
            "gold_chunk_id": c.id, "gold_text_snippet": c.text[s:e], "verified": "",
            "notes": "", "span_start": base + s, "span_end": base + e,
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
