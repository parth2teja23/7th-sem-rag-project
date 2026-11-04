from __future__ import annotations

import re
from typing import Callable

import numpy as np

from hrag.types import Doc

_NUM_HEAD = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]\S*")
_WORD_HEAD = re.compile(r"^[A-Z][A-Za-z ]{2,60}$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")

Span = tuple[int, int]


def _is_heading(line: str) -> bool:
    n = len(line.split())
    if _NUM_HEAD.match(line) and n <= 12 and line[-1] not in ".:,":
        return True
    return bool(_WORD_HEAD.match(line)) and n <= 8


def _make(doc: Doc, i: int, start: int, end: int, section: str = "") -> Doc:
    meta = {
        **doc.meta,
        "parent": doc.id,
        "section": section,
        "char_start": start,
        "char_end": end,
    }
    return Doc(id=f"{doc.id}#{i}", text=doc.text[start:end], title=doc.title, meta=meta)


def _word_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    end = len(text) if end is None else end
    return [(start + m.start(), start + m.end()) for m in _WORD.finditer(text[start:end])]


def fixed_chunks(doc: Doc, size: int = 256, overlap: int = 0) -> list[Doc]:
    """Word windows of `size`, consecutive windows sharing `overlap` words; verbatim slices."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    ws = _word_spans(doc.text)
    step = size - overlap
    out: list[Doc] = []
    start = 0
    while start < len(ws):
        end = min(start + size, len(ws))
        out.append(_make(doc, len(out), ws[start][0], ws[end - 1][1]))
        if end >= len(ws):
            break
        start += step
    return out


def _parse(text: str) -> list[tuple[str, int, int]]:
    """Events ("head"|"para", start, end) as verbatim char spans, in document order."""
    events: list[tuple[str, int, int]] = []
    block: list[tuple[int, int]] = []  # stripped line spans of the current block

    def close() -> None:
        if block:
            kind = "head" if len(block) == 1 and _is_heading(text[block[0][0]:block[0][1]]) else "para"
            events.append((kind, block[0][0], block[-1][1]))
            block.clear()

    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            s = pos + (len(line) - len(line.lstrip()))
            block.append((s, s + len(stripped)))
        else:
            close()
        pos += len(line)
    close()
    return events


def _split_long(text: str, start: int, end: int, target: int) -> list[Span]:
    """Split an over-long paragraph span into sentence spans; over-long sentences by words."""
    s = text[start:end]
    sents: list[Span] = []
    pos = 0
    for m in _SENT_SPLIT.finditer(s):
        sents.append((start + pos, start + m.start()))
        pos = m.end()
    sents.append((start + pos, end))
    return sents


def semantic_chunks(
    doc: Doc,
    target: int = 300,
    max_words: int = 380,
    overlap: int = 50,
    embed: Callable[[list[str]], np.ndarray] | None = None,
    break_threshold: float = 0.55,
) -> list[Doc]:
    text = doc.text
    events = _parse(text)
    paras = [text[s:e] for k, s, e in events if k == "para"]
    emb = None
    if embed is not None and paras:
        emb = np.asarray(embed(paras), dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1.0, norms)

    groups: list[tuple[str, int, int]] = []  # (section, char_start, char_end)
    section = ""
    start = end = -1
    words = 0
    has_body = False
    prev_idx: int | None = None

    def flush() -> None:
        nonlocal start, end, words, has_body
        if start >= 0:
            groups.append((section, start, end))
        start = end = -1
        words, has_body = 0, False

    def add(s: int, e: int, body: bool) -> None:
        nonlocal start, end, words, has_body
        n = len(text[s:e].split())
        if start >= 0 and ((has_body and words + n > target) or words + n > max_words):
            flush()
        if start < 0:
            start = s
        end = e
        words += n
        has_body = has_body or body

    pi = -1
    for kind, s, e in events:
        if kind == "head":
            flush()
            section = text[s:e]
            prev_idx = None
            add(s, e, False)
            continue
        pi += 1
        if emb is not None and prev_idx is not None and float(emb[prev_idx] @ emb[pi]) < break_threshold:
            flush()
        if len(text[s:e].split()) <= max_words:
            units = [(s, e)]
        else:
            units = []
            for us, ue in _split_long(text, s, e, target):
                ws = _word_spans(text, us, ue)
                if len(ws) > max_words:
                    units.extend((ws[k][0], ws[min(k + target, len(ws)) - 1][1])
                                 for k in range(0, len(ws), target))
                elif ws:
                    units.append((us, ue))
        for us, ue in units:
            add(us, ue, True)
        prev_idx = pi
    flush()

    out: list[Doc] = []
    prev: tuple[str, int, int] | None = None
    for sec, s, e in groups:
        cs = s
        if overlap > 0 and prev is not None and prev[0] == sec:
            pws = _word_spans(text, prev[1], prev[2])
            if pws:
                cs = pws[-overlap:][0][0]
        out.append(_make(doc, len(out), cs, e, sec))
        prev = (sec, s, e)
    return out
