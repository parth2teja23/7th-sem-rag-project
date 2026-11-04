"""Dense encoders (bge-small, EmbeddingGemma 2) and a FAISS HNSW index.

Notes:
- EmbeddingGemma 2 is loaded text-only (vision/audio configs dropped) in float32
  (never float16). The processor import still needs ``torchvision`` installed.
- Gemma ``max_seq_length`` is set to 2048 to bound memory on long chunks; our
  chunks are <= ~430 words so this never truncates real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hrag.types import Doc, Ranking

ENCODERS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "gemma": "google/embeddinggemma-2",
}
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_GEMMA_QUERY_PREFIX = "task: search result | query: "
_GEMMA_DIMS = {768, 512, 256, 128}
_GEMMA_MAX_LEN = 2048


def _single_thread_faiss() -> None:
    """faiss-cpu and torch each bundle an OpenMP runtime on macOS; multi-threaded
    faiss calls after torch is loaded segfault. HNSW search is cheap; stay single-threaded."""
    import faiss

    faiss.omp_set_num_threads(1)


class Encoder:
    def __init__(self, name: str, dim: int | None = None, device: str | None = None):
        if name not in ENCODERS:
            raise ValueError(f"unknown encoder {name!r}; choose from {list(ENCODERS)}")
        if dim is not None and name != "gemma":
            raise ValueError("dim (Matryoshka truncation) is only supported for 'gemma'")
        if dim is not None and dim not in _GEMMA_DIMS:
            raise ValueError(f"gemma dim must be one of {sorted(_GEMMA_DIMS)}")
        import torch
        from sentence_transformers import SentenceTransformer

        from hrag.device import pick_device

        self.name = name
        self.device = device or pick_device()
        if name == "gemma":
            self.model = SentenceTransformer(
                ENCODERS[name],
                device=self.device,
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"torch_dtype": torch.float32},
            )
            self.model.max_seq_length = _GEMMA_MAX_LEN
            native = 768
        else:
            self.model = SentenceTransformer(ENCODERS[name], device=self.device)
            native = self.model.get_embedding_dimension()
        self.dim = dim or native
        self._truncate = dim if (dim is not None and dim != native) else None
        self.max_len = int(self.model.max_seq_length)

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            truncate_dim=self._truncate,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(out, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefix = _GEMMA_QUERY_PREFIX if self.name == "gemma" else _BGE_QUERY_PREFIX
        return self._encode([prefix + t for t in texts])

    def encode_docs(self, docs: list[Doc], batch_size: int = 64) -> np.ndarray:
        if self.name == "gemma":
            texts = [f"title: {d.title or 'none'} | text: {d.text}" for d in docs]
        else:
            texts = [d.text for d in docs]
        return self._encode(texts, batch_size)


class DenseIndex:
    def __init__(self):
        self.index = None
        self.ids: list[str] = []
        self.encoder: Encoder | None = None

    def build(self, docs: dict[str, Doc], encoder: Encoder, batch_size: int = 64) -> None:
        import faiss

        _single_thread_faiss()
        self.encoder = encoder
        self.ids = list(docs)
        embs = encoder.encode_docs([docs[i] for i in self.ids], batch_size)
        index = faiss.IndexHNSWFlat(encoder.dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(embs)
        index.hnsw.efSearch = 128
        self.index = index

    def _rank(self, scores: np.ndarray, idxs: np.ndarray) -> Ranking:
        return [(self.ids[i], float(s)) for s, i in zip(scores, idxs) if i >= 0]

    def search_batch(self, queries: list[str], k: int = 100) -> list[Ranking]:
        q = self.encoder.encode_queries(list(queries))
        k = min(k, len(self.ids))
        scores, idxs = self.index.search(q, k)
        return [self._rank(s, i) for s, i in zip(scores, idxs)]

    def search(self, query: str, k: int = 100) -> Ranking:
        return self.search_batch([query], k)[0]

    def save(self, dir) -> None:
        import faiss

        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        meta = {"ids": self.ids, "encoder": self.encoder.name, "dim": self.encoder.dim}
        (d / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, dir, encoder: Encoder) -> "DenseIndex":
        import faiss

        d = Path(dir)
        meta = json.loads((d / "meta.json").read_text())
        if meta["encoder"] != encoder.name or meta["dim"] != encoder.dim:
            raise ValueError(
                f"index built with {meta['encoder']}/{meta['dim']}, "
                f"got encoder {encoder.name}/{encoder.dim}"
            )
        _single_thread_faiss()
        obj = cls()
        obj.index = faiss.read_index(str(d / "index.faiss"))
        obj.index.hnsw.efSearch = 128
        obj.ids = meta["ids"]
        obj.encoder = encoder
        return obj
