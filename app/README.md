# Streamlit demo

1. Start Ollama: `ollama serve` (the answer and query expansion need it; retrieval works without).
2. Run: `.venv/bin/streamlit run app/streamlit_app.py`

Sidebar: dataset (scifact / rfc / hotpot), encoder (bge-small / gemma), alpha mode (fixed slider or QAF, offered only if a trained QAF exists for the encoder), rerank on/off, query expansion on/off.

Panels:
- Example queries: test-set questions; picking one marks its relevant docs with a check mark.
- Four columns: BM25, Dense, Fused, Reranked, each with title, snippet and score.
- Answer: LLM answer with citations that expand to the cited passage. If Ollama is down, a warning is shown and retrieval still renders.
- QAF (QAF mode only): the 12 query features and the chosen alpha.
- Latency: per-stage milliseconds bar chart.

Indexes and models load once via `st.cache_resource` (`app/backend.py`).

Limitations: only datasets and encoders whose indexes exist on disk are offered (FiQA is not offered by the demo). The first load of gemma is slow (large model plus index); later reruns reuse the cache.
