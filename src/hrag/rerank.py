from __future__ import annotations

from hrag.device import pick_device
from hrag.types import Doc, Ranking, sort_ranking


class Reranker:
    """Cross-encoder reranker (default BAAI/bge-reranker-base)."""

    def __init__(self, name: str = "BAAI/bge-reranker-base", device: str | None = None,
                 batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(name, max_length=512, device=self.device)

    def rerank(self, query: str, ranking: Ranking, corpus: dict[str, Doc],
               depth: int = 50) -> Ranking:
        """Score the top-`depth` docs of `ranking` with the cross-encoder.

        Returns (doc_id, score) sorted by score desc; scores are raw CE logits
        (no sigmoid). Docs beyond `depth` are dropped; ids absent from
        `corpus` are skipped.
        """
        if depth <= 0 or not ranking:
            return []
        ids = [i for i, _ in ranking[:depth] if i in corpus]
        if not ids:
            return []
        pairs = []
        for i in ids:
            d = corpus[i]
            pairs.append((query, f"{d.title}\n{d.text}" if d.title else d.text))
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return sort_ranking([(i, float(s)) for i, s in zip(ids, scores)])
