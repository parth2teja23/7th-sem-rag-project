from __future__ import annotations

import re
from typing import Callable

import numpy as np

from hrag.types import Doc

_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S|^[A-Z][A-Za-z ]{2,60}$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _chunk_doc(doc: Doc, i: int, text: str, section: str = "") -> Doc:
    meta = {**doc.meta, "parent": doc.id, "section": section}
    return Doc(id=f"{doc.id}#{i}", text=text, title=doc.title, meta=meta)


def fixed_chunks(doc: Doc, size: int = 256, overlap: int = 0) -> list[Doc]:
    """Split into word windows of `size`, consecutive windows sharing `overlap` words."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    toks = doc.text.split()
    step = size - overlap
    out: list[Doc] = []
    start = 0
    while start < len(toks):
        end = start + size
        out.append(_chunk_doc(doc, len(out), " ".join(toks[start:end])))
        if end >= len(toks):
            break
        start += step
    return out


def _parse(text: str) -> list[tuple[str, str | int]]:
    """Return events: ("head", heading) or ("para", paragraph_text)."""
    events: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text):
        buf: list[str] = []
        for line in block.splitlines():
            s = line.strip()
            if not s:
                continue
            if _HEADING_RE.match(s):
                if buf:
                    events.append(("para", "\n".join(buf)))
                    buf = []
                events.append(("head", s))
            else:
                buf.append(s)
        if buf:
            events.append(("para", "\n".join(buf)))
    return events


def _split_long(par: str, target: int, max_words: int) -> list[str]:
    """Split an over-long paragraph into units of <= max_words (sentence-aware)."""
    units: list[str] = []
    for sent in _SENT_SPLIT.split(par):
        w = sent.split()
        if len(w) > max_words:
            units.extend(" ".join(w[k:k + target]) for k in range(0, len(w), target))
        elif w:
            units.append(" ".join(w))
    return units


def semantic_chunks(
    doc: Doc,
    target: int = 300,
    max_words: int = 380,
    overlap: int = 50,
    embed: Callable[[list[str]], np.ndarray] | None = None,
    break_threshold: float = 0.55,
) -> list[Doc]:
    events = _parse(doc.text)
    paras = [p for kind, p in events if kind == "para"]
    emb = None
    if embed is not None and paras:
        emb = np.asarray(embed(paras), dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1.0, norms)

    # groups: (section, [unit strings]) -- each group becomes one chunk body
    groups: list[tuple[str, list[str]]] = []
    section = ""
    cur: list[str] = []
    cur_words = 0
    prev_idx: int | None = None  # index of previous paragraph within the section

    def flush() -> None:
        nonlocal cur, cur_words
        if cur:
            groups.append((section, cur))
        cur, cur_words = [], 0

    pi = -1
    for kind, val in events:
        if kind == "head":
            flush()
            section = val
            prev_idx = None
            continue
        pi += 1
        units = [val] if len(val.split()) <= max_words else _split_long(val, target, max_words)
        topical_break = (
            emb is not None and prev_idx is not None
            and float(emb[prev_idx] @ emb[pi]) < break_threshold
        )
        if topical_break:
            flush()
        for u in units:
            n = len(u.split())
            if cur and cur_words + n > target:
                flush()
            cur.append(u)
            cur_words += n
        prev_idx = pi
    flush()

    out: list[Doc] = []
    prev_body: list[str] = []
    prev_section: str | None = None
    for sec, units in groups:
        body = "\n\n".join(units)
        text = body
        if overlap > 0 and prev_body and prev_section == sec:
            text = " ".join(prev_body[-overlap:]) + " " + body
        out.append(_chunk_doc(doc, len(out), text, sec))
        prev_body, prev_section = body.split(), sec
    return out
