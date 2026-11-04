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
               "notes", "span_start", "span_end", "answer_start", "answer_end"]
MAX_PER_RFC = 12
MAX_DOC_OCCURRENCES = 3
_DEICTIC = ("this document", "the document", "this specification", "this memo")
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


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _bad_shape(a: str) -> str | None:
    if a.endswith(("(", ",")):
        return "answer_truncated"
    if (a.count("(") != a.count(")") or a.count("[") != a.count("]")
            or a.count("{") != a.count("}") or a.count('"') % 2):
        return "answer_unbalanced"
    if not any(ch.isalnum() for ch in a) or len(a) <= 2:
        return "answer_too_short"
    if "\n" in a or re.search(r"\s\s", a):
        return "answer_whitespace"
    return None


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
    bad = _bad_shape(a)
    if bad:
        return bad
    if any(d_ in q.lower() for d_ in _DEICTIC):
        return "question_deictic"
    if chunk_text.lower().count(a.lower()) > 1:
        return "answer_ambiguous_in_chunk"
    return q, a


def _round_robin(cands: list[Doc], rng: random.Random):
    groups: dict = defaultdict(list)
    for c in cands:
        groups[c.meta.get("parent")].append(c)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    while any(groups.values()):
        for k in keys:
            if groups[k]:
                yield groups[k].pop()


def generate_questions(chunks: list[Doc], n: int = 180, seed: int = 42,
                       chat: Callable[..., str] | None = None,
                       stats: dict | None = None, docs: dict[str, Doc] | None = None,
                       per_rfc: int = MAX_PER_RFC) -> list[dict]:
    """Sample identifier chunks round-robin over parents (<= ``per_rfc`` each, <= 2 per section)."""
    chat = chat or _default_chat
    stats = stats if stats is not None else {}
    stats.clear()
    stats.update(accepted=0, rejected=0)

    def reject(reason: str) -> None:
        stats["rejected"] += 1
        stats[f"rej_{reason}"] = stats.get(f"rej_{reason}", 0) + 1

    cands = [c for c in chunks if count_identifiers(c.text) >= 2]
    per_section: dict[tuple, int] = defaultdict(int)
    per_parent: dict = defaultdict(int)
    seen_q: set[str] = set()
    seen_a: set[str] = set()
    out: list[dict] = []
    for c in _round_robin(cands, random.Random(seed)):
        if len(out) >= n:
            break
        parent = c.meta.get("parent")
        key = (parent, c.meta.get("section"))
        if per_section[key] >= MAX_PER_SECTION or per_parent[parent] >= per_rfc:
            continue
        try:
            raw = chat(PROMPT.format(chunk=c.text), system=SYSTEM, json_mode=True)
        except Exception as exc:  # degenerate generations (e.g. Ollama repeat-limit 500) skip the chunk
            if type(exc).__name__ == "OllamaUnavailable":
                raise
            reject("llm_error")
            continue
        res = _validate(raw, c.text)
        if isinstance(res, str):
            reject(res)
            continue
        q, a = res
        if docs and parent in docs and docs[parent].text.lower().count(a.lower()) > MAX_DOC_OCCURRENCES:
            reject("answer_ambiguous_in_doc")
            continue
        nq, na = _norm(q), _norm(a)
        if nq in seen_q:
            reject("duplicate_question")
            continue
        if na in seen_a:
            reject("duplicate_answer")
            continue
        seen_q.add(nq)
        seen_a.add(na)
        i = c.text.lower().index(a.lower())
        ans = c.text[i:i + len(a)]
        s, e = max(0, i - CONTEXT_CHARS), min(len(c.text), i + len(a) + CONTEXT_CHARS)
        base = c.meta["char_start"]
        per_section[key] += 1
        per_parent[parent] += 1
        stats["accepted"] += 1
        out.append({
            "qid": f"rfcq{len(out) + 1:03d}", "question": q, "answer": ans,
            "gold_chunk_id": c.id, "gold_text_snippet": c.text[s:e], "verified": "",
            "notes": "", "span_start": base + s, "span_end": base + e,
            "answer_start": base + i, "answer_end": base + i + len(a),
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
