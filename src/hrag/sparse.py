from __future__ import annotations

import json
import math
import re
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from bm25s.stopwords import STOPWORDS_EN

from hrag.types import Doc, Ranking, sort_ranking

_K1, _B = 1.2, 0.75
_COMPOUND = re.compile(r"[a-z0-9]+(?:[._/:-][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")
_STOP = frozenset(STOPWORDS_EN)


class SparseIndex:
    """BM25 (Lucene variant) over ``title + " " + text``.

    English stopwords + PyStemmer English stemming, applied identically to docs,
    queries and ``idf`` terms.
    """

    def __init__(self) -> None:
        self._stemmer = Stemmer.Stemmer("english")
        self._retriever: bm25s.BM25 | None = None
        self._ids: list[str] = []
        self._df: dict[str, int] = {}

    def _term(self, t: str) -> str | None:
        if t.isalpha():
            return None if t in _STOP else self._stemmer.stemWord(t)
        return t  # digits/separators: never stemmed or stop-filtered

    def _tok_one(self, text: str) -> list[str]:
        out: list[str] = []
        for m in _COMPOUND.findall(text.lower()):
            whole = self._term(m)
            if whole is not None:
                out.append(whole)
            if m != whole or not m.isalnum():
                parts = _PART.findall(m)
                if len(parts) > 1:
                    for part in parts:
                        t = self._term(part)
                        if t is not None:
                            out.append(t)
        return out

    def _tok(self, texts: list[str]) -> list[list[str]]:
        return [self._tok_one(t) for t in texts]

    def build(self, docs: dict[str, Doc]) -> None:
        self._ids = list(docs)
        texts = [f"{docs[i].title} {docs[i].text}" for i in self._ids]
        tokens = self._tok(texts)
        df: dict[str, int] = {}
        for toks in tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        r = bm25s.BM25(k1=_K1, b=_B, method="lucene")
        r.index(tokens, show_progress=False)  # pre-tokenised lists
        self._retriever = r

    def search(self, query: str, k: int = 100) -> Ranking:
        if self._retriever is None or not self._ids or k <= 0:
            return []
        qtoks = [t for t in self._tok([query])[0] if t in self._df]
        if not qtoks:
            return []
        scores = np.asarray(self._retriever.get_scores(qtoks))
        pos = np.flatnonzero(scores > 0)
        if pos.size == 0:
            return []
        if pos.size > k:
            # keep a tie-safe candidate set: everything >= k-th best score
            kth = np.partition(scores[pos], -k)[-k]
            pos = pos[scores[pos] >= kth]
        out = [(self._ids[i], float(scores[i])) for i in pos]
        return sort_ranking(out)[:k]

    def idf(self, term: str) -> float:
        """IDF of a single raw term (tokenised the same way; only the first token is used)."""
        toks = self._tok([term])[0]
        if not toks:
            return 0.0
        df = self._df.get(toks[0], 0)
        n = len(self._ids)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def save(self, path) -> None:
        if self._retriever is None:
            raise RuntimeError("build() first")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(p / "bm25"))
        (p / "meta.json").write_text(json.dumps({"ids": self._ids, "df": self._df}))

    @classmethod
    def load(cls, path) -> "SparseIndex":
        p = Path(path)
        self = cls()
        self._retriever = bm25s.BM25.load(str(p / "bm25"))
        meta = json.loads((p / "meta.json").read_text())
        self._ids, self._df = meta["ids"], meta["df"]
        return self
