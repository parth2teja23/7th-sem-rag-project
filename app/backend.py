"""Heavy loading for the Streamlit demo, isolated so tests can swap it for a toy fixture."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "experiments", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATASETS = ["scifact", "rfc", "hotpot"]
ENCODERS = ["bge-small", "gemma"]


@dataclass
class Backend:
    corpus: dict
    queries: dict
    qrels: dict
    sparse: object
    dense: object
    reranker: object
    qaf: object | None = None  # None when no trained QAF exists for this encoder
    vocab: set = field(default_factory=set)


def load_backend(dataset: str, encoder: str) -> Backend:
    """Load dataset, indexes, reranker and (if trained) the QAF model for one dataset+encoder."""
    import common

    corpus, queries, qrels, _ = common.load_dataset(dataset, "test")
    sparse, dense = common.load_indexes(dataset, encoder)
    reranker = common.make_reranker()
    qaf, vocab = None, set()
    try:
        qaf = common.load_qaf(encoder)
        vocab = common.encoder_vocab(dense.encoder)
    except (FileNotFoundError, OSError):
        qaf = None
    return Backend(corpus, queries, qrels, sparse, dense, reranker, qaf, vocab)
