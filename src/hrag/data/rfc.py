"""Custom identifier-heavy RFC corpus: fetch, clean, and span-based qrels."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import requests

from hrag.types import Doc, Qrels

RFC_IDS = [791, 793, 1035, 2616, 3986, 4291, 5321, 6455, 6749, 7230, 7519, 7540, 8200, 8259,
           8446, 8484, 9000, 9110, 9112, 9113]
URL = "https://www.rfc-editor.org/rfc/rfc{n}.txt"
USER_AGENT = "hrag-research-corpus-builder/0.1 (academic RAG benchmark)"

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "rfc"
RFC_DIR = ROOT / "data" / "rfc"

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_HEADER = re.compile(rf"^RFC \d+ .* ({_MONTHS}) \d{{4}}$")
_FOOTER = re.compile(r"^.*\[Page \d+\]$")
_DATE = re.compile(rf"\b({_MONTHS}) (\d{{4}})\b")


def clean_rfc_text(raw: str) -> str:
    """Drop BOM, form feeds, page headers/footers; collapse 3+ blank lines to one."""
    text = raw.replace("﻿", "").replace("\f", "")
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept = [ln for ln in lines if not _HEADER.match(ln) and not _FOOTER.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n") + "\n"


def parse_year(raw: str) -> int:
    for ln in raw.replace("﻿", "").splitlines()[:60]:
        m = _DATE.search(ln)
        if m:
            return int(m.group(2))
    raise ValueError("no date line found")


def parse_title(raw: str) -> str:
    """First block of the first page whose lines are all centred (indented), not a date."""
    lines = [ln.rstrip() for ln in raw.replace("﻿", "").replace("\f", "").splitlines()[:60]]
    block: list[str] = []
    for ln in lines + [""]:
        if ln.strip():
            block.append(ln)
            continue
        if block and all(len(b) - len(b.lstrip()) >= 3 for b in block) \
                and not any(_DATE.search(b) for b in block):
            return " ".join(b.strip() for b in block)
        block = []
    return ""


def fetch_rfc(n: int) -> Doc:
    path = RAW_DIR / f"rfc{n}.txt"
    if not path.exists():
        r = requests.get(URL.format(n=n), headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    raw = path.read_text(encoding="utf-8")
    return Doc(id=f"rfc{n}", text=clean_rfc_text(raw), title=parse_title(raw),
               meta={"rfc": n, "year": parse_year(raw)})


def span_relevant(chunks: list[Doc], parent: str, start: int, end: int) -> list[str]:
    """Ids of chunks of ``parent`` whose [char_start, char_end) overlaps [start, end)."""
    return [c.id for c in chunks
            if c.meta.get("parent") == parent
            and c.meta["char_start"] < end and start < c.meta["char_end"]]


def load_docs() -> dict[str, Doc]:
    path = RFC_DIR / "docs.jsonl"
    if path.exists():
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out[d["id"]] = Doc(d["id"], d["text"], d["title"], d["meta"])
        return out
    return {f"rfc{n}": fetch_rfc(n) for n in RFC_IDS}


def load_rfc_corpus(chunker: str = "semantic") -> tuple[
        dict[str, Doc], dict[str, str], Qrels, dict[str, list[str]]]:
    """Return (chunk corpus, queries, span-based qrels, answers) for verified questions only."""
    from hrag.chunking import fixed_chunks, semantic_chunks

    if chunker not in ("semantic", "fixed"):
        raise ValueError("chunker must be 'semantic' or 'fixed'")
    qpath = RFC_DIR / "questions.csv"
    if not qpath.exists():
        raise FileNotFoundError(
            f"{qpath} not found: manually verify {RFC_DIR / 'questions_draft.csv'} "
            "(set verified=1 on correct rows) and save it as questions.csv")
    chunk = semantic_chunks if chunker == "semantic" else fixed_chunks
    corpus: dict[str, Doc] = {}
    by_parent: dict[str, list[Doc]] = {}
    for did, doc in load_docs().items():
        cs = chunk(doc)
        by_parent[did] = cs
        corpus.update({c.id: c for c in cs})
    queries: dict[str, str] = {}
    qrels: Qrels = {}
    answers: dict[str, list[str]] = {}
    with qpath.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["verified"].strip() != "1":
                continue
            parent = row["gold_chunk_id"].split("#")[0]
            rel = span_relevant(by_parent[parent], parent, int(row["span_start"]), int(row["span_end"]))
            queries[row["qid"]] = row["question"]
            qrels[row["qid"]] = {c: 1 for c in rel}
            answers[row["qid"]] = [row["answer"]]
    return corpus, queries, qrels, answers
