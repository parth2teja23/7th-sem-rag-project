# Hybrid Retrieval RAG with Query-Adaptive Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local hybrid RAG system (BM25 + two dense encoders + weighted RRF + cross-encoder rerank + LLM generation) with a new query-adaptive fusion (QAF) method, run the experiments the paper needs, ship a Streamlit demo, and draft the IEEE paper from the generated results.

**Architecture:** A Python package `hrag` with one module per pipeline stage, all exchanging plain types (`Doc`, `Ranking`). Experiment scripts in `experiments/` call the package, write JSON results to `results/`, and one script turns `results/` into LaTeX tables and PDF figures under `paper/`. The demo in `app/` loads prebuilt indexes and calls the same `Pipeline`.

**Tech Stack:** Python 3.11, uv, sentence-transformers, torch (MPS), faiss-cpu, bm25s, scikit-learn, scipy, HF `datasets`, ollama (Qwen2.5-7B-Instruct), ragas, streamlit, matplotlib, pytest, LaTeX (IEEEtran).

**Spec:** `docs/specs/2026-11-04-hybrid-rag-design.md`

## Global Constraints

- Everything runs locally on an Apple M5 / 32 GB: torch device `mps` when available else `cpu`; no paid APIs.
- LLM: `qwen2.5:7b-instruct` via Ollama, `temperature=0`, fixed `seed=42`.
- Encoders: `BAAI/bge-small-en-v1.5` and `google/embeddinggemma-2`; reranker `BAAI/bge-reranker-base`.
- RRF constant `k=60`; default over-retrieval depth `N=50`; final context `k=5`.
- Fusion formula: `score(d) = α·1/(60+rank_dense(d)) + (1−α)·1/(60+rank_sparse(d))`, ranks 1-based; a doc missing from a list contributes 0 from that list.
- All random sampling uses seed 42; subset IDs written to `data/subsets/*.json`.
- No number in the paper is typed by hand: every table/figure comes from `experiments/make_paper_assets.py`.
- Paper prose: first-person plural, varied sentence length, concrete numbers; banned phrases: "delve", "in today's rapidly evolving", "it is worth noting", "plays a crucial role", "landscape", "seamless", "leverage", "furthermore," at sentence start more than once per page.
- Priority tiers from spec §4.7: Tier A must finish before paper drafting; B if time; C post-submission.

## Review Focus

1. Query whose terms are all stopwords/unknown to BM25 → sparse ranking empty; fusion must return the dense ranking (not crash, not empty). Test in Task 7.
2. Same document at different ranks in both lists, and ties → appears once, scores summed. Test in Task 7.
3. Ollama not running or model not pulled → `OllamaUnavailable` raised with a fix hint; demo still shows retrieval columns. Tests in Tasks 10 and 19.
4. Passage longer than the encoder max length (512 tokens bge, 8192 gemma) → truncated silently by encoder, chunker keeps chunks under limit. Test in Task 4.
5. Empty or whitespace-only query in the demo/pipeline → `ValueError("empty query")`, demo shows a message. Test in Task 11.

---

## File Structure

```
pyproject.toml
src/hrag/
  types.py          Doc, Ranking, Qrels, RetrievalResult dataclasses/aliases
  device.py         pick_device()
  data/beir.py      load BEIR datasets (scifact, fiqa) + test/dev splits
  data/hotpot.py    HotpotQA QA set with gold paragraphs and answers
  data/rfc.py       custom RFC corpus fetch + structural parse
  data/questions.py LLM question generation + review CSV I/O
  chunking.py       semantic_chunks(), fixed_chunks()
  sparse.py         SparseIndex (bm25s)
  dense.py          Encoder, DenseIndex (FAISS HNSW)
  fusion.py         weighted_rrf()
  postprocess.py    dedupe(), metadata_filter()
  rerank.py         Reranker
  qaf.py            query_features(), oracle_alpha(), QAFModel
  llm.py            Ollama client wrapper, OllamaUnavailable
  rewrite.py        expand_query()
  generate.py       generate_answer()
  pipeline.py       PipelineConfig, Pipeline
  baselines/dat.py       DAT-style LLM α (Tier B)
  baselines/pageindex.py tree-navigation retrieval (Tier C)
  eval/metrics.py   retrieval + QA metrics
  eval/stats.py     paired bootstrap / t-test
  eval/ragas_eval.py RAGAS wrapper with local judge
experiments/
  build_indexes.py  run_retrieval.py  train_qaf.py  run_ablation.py
  run_matryoshka.py run_rerank_depth.py run_generation.py run_dat.py
  run_pageindex.py  make_paper_assets.py
app/streamlit_app.py
paper/main.tex, sections/*.tex, refs.bib, tables/, figures/
tests/  (mirrors src/hrag)
data/  results/  indexes/  evidence/   (gitignored except data/subsets and evidence/)
```

Result file contract (every experiment): `results/<exp>/<dataset>__<config>.json` =
`{"config": {...}, "metrics": {"recall@10":…, "mrr@10":…, "ndcg@10":…}, "per_query": {qid: {"ndcg@10":…, "alpha":…}}, "latency_ms": {stage: {"mean":…, "p95":…}}}`.

---

### Task 1: Project scaffold, environment, core types

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/hrag/__init__.py`, `src/hrag/types.py`, `src/hrag/device.py`, `tests/test_types.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Doc(id: str, text: str, title: str = "", meta: dict = field(default_factory=dict))`
  - `Ranking = list[tuple[str, float]]` (doc_id, score), sorted by score desc
  - `Qrels = dict[str, dict[str, int]]` (qid → doc_id → relevance)
  - `@dataclass RetrievalResult(query: str, final: Ranking, stages: dict[str, Ranking], alpha: float | None, timings_ms: dict[str, float])`
  - `pick_device() -> str` returning `"mps"` / `"cpu"`

- [ ] **Step 1:** (repo already initialised on branch `feat/hybrid-rag`; append to existing `.gitignore`) create `pyproject.toml` (package `hrag`, src layout, deps from Tech Stack, dev dep pytest; pytest marker `slow`); `.gitignore` with `data/raw indexes/ results/cache .venv __pycache__ *.pyc`.
- [ ] **Step 2:** `uv venv --python 3.12 && uv pip install -e ".[dev]"`; `ollama pull qwen2.5:7b-instruct`. Expected: `python -c "import hrag, faiss, bm25s, sentence_transformers"` exits 0.
- [ ] **Step 3: Write failing test** `test_ranking_sorted_helper`: `sort_ranking([("a",0.1),("b",0.9)]) == [("b",0.9),("a",0.1)]`; `test_pick_device_returns_known`: result in `{"mps","cpu"}`.
- [ ] **Step 4:** Run `pytest tests/test_types.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement types, `sort_ranking(r: Ranking) -> Ranking` (stable, desc, ties by doc_id asc), `pick_device()`.
- [ ] **Step 6:** Run tests → PASS.
- [ ] **Step 7:** Commit `chore: scaffold hrag package and core types`.

### Task 2: Evaluation metrics and significance testing

**Files:**
- Create: `src/hrag/eval/metrics.py`, `src/hrag/eval/stats.py`, `tests/eval/test_metrics.py`, `tests/eval/test_stats.py`

**Interfaces:**
- Produces:
  - `recall_at_k(ranking: Ranking, rel: dict[str,int], k: int) -> float`
  - `mrr_at_k(ranking, rel, k) -> float`
  - `ndcg_at_k(ranking, rel, k) -> float` (graded gains, log2 discount, ideal from rel)
  - `evaluate_run(run: dict[str, Ranking], qrels: Qrels, ks=(5,10,20)) -> tuple[dict[str,float], dict[str,dict[str,float]]]` (mean metrics, per-query) — queries with no qrels are skipped
  - `normalize_answer(s) -> str` (SQuAD rules), `exact_match(pred, golds: list[str]) -> float`, `token_f1(pred, golds) -> float`
  - `paired_bootstrap(a: list[float], b: list[float], n=10000, seed=42) -> float` (p-value that mean(b) > mean(a) by chance), `paired_ttest(a, b) -> float`

- [ ] **Step 1: Write failing tests:**
  - ranking `[("d1",3),("d2",2),("d3",1)]`, rel `{"d2":1}` → recall@1=0, recall@2=1, mrr@10=0.5, ndcg@10 = `1/log2(3)` ≈ 0.6309
  - rel `{"d1":2,"d3":1}` → ndcg@3 = `(3/1 + 1/2)/(3/1 + 1/log2(3))` ≈ 0.9639 (gain `2^rel−1`)
  - `exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0`; `token_f1("big red dog", ["red dog"]) == 0.8`
  - `paired_bootstrap([0]*50, [1]*50) < 0.01`; identical lists → `> 0.4`
- [ ] **Step 2:** Run `pytest tests/eval -v` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(eval): retrieval/QA metrics and significance tests`.

### Task 3: Dataset loaders (BEIR, HotpotQA) with fixed subsets

**Files:**
- Create: `src/hrag/data/beir.py`, `src/hrag/data/hotpot.py`, `tests/data/test_loaders.py`

**Interfaces:**
- Produces:
  - `load_beir(name: Literal["scifact","fiqa"], split: Literal["train","dev","test"], max_docs: int | None = None) -> tuple[dict[str, Doc], dict[str, str], Qrels]` — source HF `BeIR/<name>` + `BeIR/<name>-qrels`; SciFact has no dev: carve dev = 20% of train queries (seed 42). When `max_docs` set, keep all qrel-positive docs for the loaded split plus random fill to `max_docs`; write chosen IDs to `data/subsets/<name>_docs.json`.
  - `load_hotpot(n_questions: int = 500) -> tuple[dict[str, Doc], dict[str, str], Qrels, dict[str, list[str]]]` — HF `hotpotqa/hotpot_qa` config `distractor` split `validation`; corpus = union of the 10 paragraphs of each sampled question (doc id `"<title>::<para_idx>"`, deduplicated by title); qrels = supporting-fact paragraphs; answers dict qid → `[answer]`. IDs saved to `data/subsets/hotpot.json`.

- [ ] **Step 1: Write failing tests** (marked `slow`, need network once, cached by HF):
  - `load_beir("scifact","test")` → 300 queries, ≥5000 docs, every qrels doc in corpus
  - `load_beir("scifact","dev")` and `"train"` query IDs disjoint
  - `load_hotpot(50)` → 50 queries, every qid has 2 relevant docs present in corpus, answers non-empty
  - calling `load_hotpot(50)` twice gives identical qids (seed fixed)
- [ ] **Step 2:** Run `pytest tests/data -v -m slow` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(data): BEIR and HotpotQA loaders with seeded subsets`.

### Task 4: Semantic and fixed-size chunking

**Files:**
- Create: `src/hrag/chunking.py`, `tests/test_chunking.py`

**Interfaces:**
- Produces:
  - `fixed_chunks(doc: Doc, size: int = 256, overlap: int = 0) -> list[Doc]` (word-based)
  - `semantic_chunks(doc: Doc, target: int = 300, max_words: int = 380, overlap: int = 50, embed: Callable[[list[str]], np.ndarray] | None = None, break_threshold: float = 0.55) -> list[Doc]`
  - Chunk ids `"<doc.id>#<i>"`; chunk `meta` copies `doc.meta` plus `{"parent": doc.id, "section": <heading or "">}`.

Algorithm for `semantic_chunks`: split into paragraphs on blank lines; a line matching `^\d+(\.\d+)*\.?\s+\S` or `^[A-Z][A-Za-z ]{2,60}$` is a heading and always starts a new chunk and sets `section`. Within a section, greedily append paragraphs while total ≤ `target`; if `embed` is given, also start a new chunk when cosine(prev para, next para) < `break_threshold`. A single paragraph over `max_words` is split by sentences. Prepend the last `overlap` words of the previous chunk in the same document.

- [ ] **Step 1: Write failing tests:**
  - text with headings `"1. Intro\n\npara a\n\n2. Methods\n\npara b"` → 2 chunks, sections `"1. Intro"`, `"2. Methods"`
  - a 1000-word single paragraph → every chunk ≤ `max_words + overlap` words (Review Focus 4)
  - chunk i>0 starts with the last 50 words of chunk i−1 when same section
  - fake `embed` returning orthogonal vectors for para 2 → break inserted between para 1 and 2
  - `fixed_chunks` of 600 words, size 256 → 3 chunks
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: semantic and fixed-size chunkers`.

### Task 5: Sparse BM25 index

**Files:**
- Create: `src/hrag/sparse.py`, `tests/test_sparse.py`

**Interfaces:**
- Produces: `class SparseIndex: build(docs: dict[str, Doc]) -> None; search(query: str, k: int = 100) -> Ranking; idf(term: str) -> float; save(path) / load(path) (classmethod)`. Indexes `title + " " + text`; bm25s with English stopwords, `k1=1.2, b=0.75`; returns only docs with score > 0.

- [ ] **Step 1: Write failing tests:** corpus of 3 docs, one containing `"RFC 9110"` → `search("RFC 9110")[0][0]` is that doc; `search("the of and")` → `[]`; save/load round-trip gives same ranking; `idf("rfc") > idf("the")` or `idf("the")==0`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: BM25 sparse index`.

### Task 6: Dense encoders and FAISS index

**Files:**
- Create: `src/hrag/dense.py`, `tests/test_dense.py`

**Interfaces:**
- Produces:
  - `class Encoder(name: str, dim: int | None = None, device: str | None = None)`; `encode_queries(texts) -> np.ndarray`, `encode_docs(docs: list[Doc]) -> np.ndarray` (float32, L2-normalised, shape `(n, dim or native)`); `max_len` attribute.
    - bge: query prefix `"Represent this sentence for searching relevant passages: "`, docs raw.
    - EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0, already in local HF cache, requires sentence-transformers>=6.1.0): load text-only with `config_kwargs={"vision_config": None, "audio_config": None}` and `model_kwargs={"torch_dtype": torch.float32}` (never float16). Queries prefixed `"task: search result | query: "`; docs formatted `f"title: {doc.title or 'none'} | text: {doc.text}"`. Matryoshka: `truncate_dim=dim` ∈ {768,512,256,128} with `normalize_embeddings=True`.
  - `ENCODERS = {"bge-small": "BAAI/bge-small-en-v1.5", "gemma": "google/embeddinggemma-2"}`
  - `class DenseIndex: build(docs: dict[str,Doc], encoder: Encoder, batch_size=64) -> None; search(query: str, k=100) -> Ranking; search_batch(queries: list[str], k) -> list[Ranking]; save(dir)/load(dir, encoder)`. FAISS `IndexHNSWFlat(dim, 32, METRIC_INNER_PRODUCT)`, `efConstruction=200`, `efSearch=128`.
- [ ] **Step 1: Write failing tests** (`slow`): for each encoder, 3 docs ("cats purr", "stock market crash", "feline sounds") → query "why do cats make noise" ranks a cat doc first; embeddings have unit norm; gemma with `dim=128` → shape `(3,128)`; save/load round-trip identical top-1.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: dense encoders (bge, EmbeddingGemma+Matryoshka) and FAISS HNSW index`.

### Task 7: Weighted RRF fusion and post-processing

**Files:**
- Create: `src/hrag/fusion.py`, `src/hrag/postprocess.py`, `tests/test_fusion.py`, `tests/test_postprocess.py`

**Interfaces:**
- Produces:
  - `weighted_rrf(dense: Ranking, sparse: Ranking, alpha: float, k: int = 60, top_n: int = 100) -> Ranking` (formula in Global Constraints)
  - `dedupe(ranking: Ranking, corpus: dict[str,Doc], threshold: float = 0.9) -> Ranking` — drop a doc whose word-5-shingle Jaccard with a higher-ranked kept doc ≥ threshold
  - `metadata_filter(ranking, corpus, predicate: Callable[[dict], bool] | None) -> Ranking`

- [ ] **Step 1: Write failing tests:**
  - dense `[a,b]`, sparse `[b,c]`, α=0.5 → order `b, a, c`; b's score = `0.5/61 + 0.5/62`
  - α=1 → order equals dense; α=0 → equals sparse
  - sparse `[]` → result equals dense order (Review Focus 1)
  - doc in both lists appears exactly once (Review Focus 2)
  - `alpha` outside [0,1] → `ValueError`
  - dedupe: two identical texts → second removed; filter `lambda m: m.get("year")==2020` keeps only matching
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: weighted RRF fusion, dedupe and metadata filter`.

### Task 8: Cross-encoder reranker

**Files:**
- Create: `src/hrag/rerank.py`, `tests/test_rerank.py`

**Interfaces:**
- Produces: `class Reranker(name="BAAI/bge-reranker-base", device=None, batch_size=32)`; `rerank(query: str, ranking: Ranking, corpus: dict[str,Doc], depth: int = 50) -> Ranking` — scores top-`depth` (title + text, max_length 512), returns them sorted by CE score; docs beyond depth dropped.

- [ ] **Step 1: Write failing test** (`slow`): ranking `[irrelevant, relevant]` for "capital of France" with docs "Bananas are yellow" / "Paris is the capital of France" → relevant first; `depth=1` returns 1 item.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: cross-encoder reranker`.

### Task 9: Query-Adaptive Fusion (our method)

**Files:**
- Create: `src/hrag/qaf.py`, `tests/test_qaf.py`

**Interfaces:**
- Consumes: `weighted_rrf` (Task 7), `ndcg_at_k` (Task 2), `SparseIndex.idf` (Task 5)
- Produces:
  - `FEATURE_NAMES = ["id_ratio","num_ratio","caps_ratio","len_tokens","mean_idf","max_idf","oov_ratio","sparse_top","sparse_margin","dense_top","dense_margin","overlap_at10"]`
  - `query_features(query: str, dense: Ranking, sparse: Ranking, idf: Callable[[str], float], tokenizer_vocab: set[str]) -> np.ndarray` (length 12, order as names). Definitions: identifier token = contains a digit and a letter, or `_`/`-`/`.` inside, or ALL-CAPS len≥2; `num_ratio` = pure numeric tokens; `oov_ratio` = tokens not in encoder wordpiece vocab as whole words; `*_top` = top score (0 if empty); `*_margin` = top1 − top2; `overlap_at10` = |top10 dense ∩ top10 sparse|/10.
  - `ALPHA_GRID = [round(0.1*i,1) for i in range(11)]`
  - `oracle_alpha(dense, sparse, rel: dict[str,int], k_eval=10) -> float` — α in grid maximising nDCG@10 of fused ranking; ties broken toward 0.5.
  - `class QAFModel`: `fit(X: np.ndarray, y: np.ndarray) -> self` (StandardScaler + Ridge(alpha=1.0)); `predict(X) -> np.ndarray` clipped to [0,1]; `coef_table() -> list[tuple[str,float]]` for the paper; `save(path)/load(path)` via joblib.

- [ ] **Step 1: Write failing tests:**
  - `query_features("RFC 9110 section 15.5.1", [], [], idf=lambda t:1.0, tokenizer_vocab=set())` → `id_ratio > 0.4`, `sparse_top == 0`
  - `"how do cats communicate"` → `id_ratio == 0`
  - `oracle_alpha` where only sparse holds the relevant doc → `0.0`; only dense → `1.0`
  - `QAFModel().fit` on synthetic X where y = clip(0.8 − 0.6·x0) → predictions within 0.1 of y; predict output always in [0,1]
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: query-adaptive fusion features, oracle alpha and model`.

### Task 10: LLM wrapper, query expansion, answer generation

**Files:**
- Create: `src/hrag/llm.py`, `src/hrag/rewrite.py`, `src/hrag/generate.py`, `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `class OllamaUnavailable(RuntimeError)`; `chat(prompt: str, system: str = "", model="qwen2.5:7b-instruct", json_mode=False) -> str` (temperature 0, seed 42); raises `OllamaUnavailable("Start Ollama with `ollama serve` and run `ollama pull qwen2.5:7b-instruct`")` on connection error or missing model. Disk cache keyed by sha256(model+system+prompt) in `results/cache/llm.sqlite`.
  - `expand_query(q: str) -> str` — returns `q + " " + <≤20 words of keywords/synonyms>`; on `OllamaUnavailable` returns `q`.
  - `@dataclass Answer(text: str, cited: list[str])`; `generate_answer(question: str, passages: list[Doc]) -> Answer` — passages numbered `[1]..[k]`, instruct short answer with `[n]` citations, "say 'not in context' if unsupported"; `cited` maps `[n]` back to doc ids.

- [ ] **Step 1: Write failing tests:** with `chat` monkeypatched to raise connection error → `OllamaUnavailable` with "ollama pull" in message (Review Focus 3); `expand_query` falls back to original; `generate_answer` with fake chat returning `"Paris [2]"` → `cited == [passages[1].id]`; cache: second identical call doesn't invoke client.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: Ollama wrapper, query expansion and cited answer generation`.

### Task 11: End-to-end pipeline with per-stage timing

**Files:**
- Create: `src/hrag/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 5–10.
- Produces:
  - `@dataclass PipelineConfig(encoder: str = "bge-small", dim: int | None = None, use_dense=True, use_sparse=True, alpha_mode: Literal["fixed","qaf","dat"] = "fixed", alpha: float = 0.5, expand=False, rerank=True, rerank_depth=50, final_k=5, dedupe=True)`
  - `class Pipeline(cfg, corpus, sparse: SparseIndex | None, dense: DenseIndex | None, reranker: Reranker | None, qaf: QAFModel | None)`; `retrieve(query: str) -> RetrievalResult` with `stages` keys among `"sparse","dense","fused","reranked"` and timings for `expand, sparse, dense, features, fusion, rerank, post`; `answer(query) -> tuple[RetrievalResult, Answer]`.
  - Dense-only (`use_sparse=False`) / sparse-only skip fusion; `alpha` in result is `None` then.
  - `alpha_mode="dat"` imports `hrag.baselines.dat.dat_alpha` lazily (Task 16); until Task 16 exists it raises `NotImplementedError`.

- [ ] **Step 1: Write failing tests** with tiny in-memory corpus and stub dense index (return fixed ranking): `retrieve("")` and `retrieve("   ")` → `ValueError("empty query")` (Review Focus 5); `alpha_mode="qaf"` with a stub model returning 0.2 → `result.alpha == 0.2`; `rerank=False` → `"reranked" not in stages`; `len(final) <= final_k`; all timing values ≥ 0.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: configurable hybrid pipeline with stage timings`.

### Task 12: Custom identifier-heavy RFC corpus and question set

**Files:**
- Create: `src/hrag/data/rfc.py`, `src/hrag/data/questions.py`, `experiments/build_rfc_corpus.py`, `tests/data/test_rfc.py`
- Output: `data/rfc/docs.jsonl`, `data/rfc/questions_draft.csv`, `data/rfc/questions.csv` (verified)

**Interfaces:**
- Produces:
  - `RFC_IDS = [791, 793, 1035, 2616, 3986, 4291, 5321, 6455, 6749, 7230, 7519, 7540, 8200, 8259, 8446, 8484, 9000, 9110, 9112, 9113]`
  - `fetch_rfc(n: int) -> Doc` from `https://www.rfc-editor.org/rfc/rfc{n}.txt`, strip page headers/footers and form feeds, `meta={"rfc": n, "year": ...}`; cached under `data/raw/rfc/`.
  - `load_rfc_corpus(chunker="semantic"|"fixed") -> tuple[dict[str,Doc], dict[str,str], Qrels, dict[str,list[str]]]` — reads verified CSV only.
  - `generate_questions(chunks, n=150, seed=42) -> list[dict]` — sample chunks containing identifiers; prompt LLM (json_mode) for one question that needs an exact identifier/number/acronym and a ≤8-word answer copied from the chunk; reject if answer not a substring of the chunk.
  - CSV columns: `qid,question,answer,gold_chunk_id,gold_text_snippet,verified,notes`.
  - Gold labels are by **parent span**: qrels for the fixed-chunk variant are all fixed chunks overlapping the gold snippet's character span (so chunking ablation is fair).

- [ ] **Step 1: Write failing tests:** header/footer stripping on a fixture text with `RFC 9110   HTTP Semantics   June 2022` lines removed; `generate_questions` with fake chat returning an answer not in the chunk → rejected; span-based qrels maps one gold span to ≥1 fixed chunk.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Run `python experiments/build_rfc_corpus.py` → `questions_draft.csv` with ~150 rows.
- [ ] **Step 6 (HUMAN — Parth):** open `questions_draft.csv`, set `verified=1` on correct rows, fix or set `0` on bad ones, save as `questions.csv`. Target ≥100 verified. Record time spent in `evidence/log.md`.
- [ ] **Step 7:** Commit `feat(data): custom RFC corpus and verified question set`.

### Task 13: Index building and core retrieval experiments (Tier A)

**Files:**
- Create: `experiments/common.py`, `experiments/build_indexes.py`, `experiments/run_retrieval.py`, `experiments/train_qaf.py`, `tests/test_experiments_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `common.load_dataset(name: Literal["scifact","fiqa","hotpot","rfc"], split="test")` → `(corpus, queries, qrels, answers|None)`; `common.load_indexes(name, encoder, dim=None)`; `common.write_result(exp, dataset, config_name, metrics, per_query, latency)` (result file contract).
  - Indexes under `indexes/<dataset>/{bm25, <encoder>[_<dim>]}`.
  - Configs run per dataset × encoder: `bm25`, `dense`, `hybrid_fixed@α*` (α* = best fixed α on **dev**), `hybrid_fixed+rr`, `qaf`, `qaf+rr`, `oracle` (upper bound, test-oracle α, labelled as such).
  - `train_qaf.py`: features + oracle α on pooled SciFact-dev + FiQA-dev (FiQA Tier B; SciFact-only if FiQA skipped), save `indexes/qaf_<encoder>.joblib`; also leave-one-dataset-out models; writes `results/qaf/coefficients_<encoder>.json`.
  - α sweep: `results/alpha_sweep/<dataset>__<encoder>.json` with nDCG@10 per α in grid.
  - Significance: `results/significance.json` — paired bootstrap p for `qaf+rr` vs `hybrid_fixed+rr` and vs `dense` per dataset/encoder.

- [ ] **Step 1: Write failing smoke test** running the full experiment chain on a 20-doc / 5-query toy dataset fixture with stub encoder → result JSONs exist and match the contract keys.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Run `python experiments/build_indexes.py --datasets scifact rfc hotpot --encoders bge-small gemma` (then `fiqa` if Tier B). Expected: index dirs exist; log build time.
- [ ] **Step 6:** Run `python experiments/train_qaf.py` then `python experiments/run_retrieval.py`. Expected: results files for every config; sanity check: hybrid nDCG@10 ≥ min(bm25, dense) on SciFact. Published SciFact BM25 nDCG@10 ≈ 0.665 — ours should be within ±0.05, else investigate before continuing.
- [ ] **Step 7:** Commit `feat(exp): index build, QAF training and core retrieval experiments` (commit code + `results/*.json`, not indexes).

### Task 14: Ablation, Matryoshka, rerank depth, latency

**Files:**
- Create: `experiments/run_ablation.py`, `experiments/run_matryoshka.py`, `experiments/run_rerank_depth.py`

**Interfaces:**
- Consumes: `common.*`, `Pipeline`.
- Produces:
  - Ablation configs on SciFact + RFC with the better encoder: `full` (semantic chunk + expand + QAF + rerank), `−chunking` (fixed chunks; RFC only), `−expansion`, `−qaf` (fixed α*), `−rerank` → `results/ablation/`. Expansion ablation is Tier B (skip `expand` in `full` if not run, and say so).
  - Matryoshka (Tier B): gemma dims 768/512/256/128 × {dense, qaf+rr} on SciFact + RFC; record index size MB and query latency → `results/matryoshka/`.
  - Rerank depth (Tier C): N ∈ {20,50,100} → `results/rerank_depth/`.
  - Latency: every result already carries `latency_ms`; this task adds `results/latency_summary.json` (mean, p95 per stage for `full`).

- [ ] **Step 1:** Extend the smoke test to cover the three scripts on the toy fixture → FAIL → implement → PASS.
- [ ] **Step 2:** Run the three scripts (Tier A ablation first). Expected: files written; removing rerank lowers nDCG@10 on SciFact (sanity).
- [ ] **Step 3:** Commit `feat(exp): ablation, Matryoshka, rerank-depth and latency runs`.

### Task 15: Generation evaluation (EM/F1, RAGAS)

**Files:**
- Create: `src/hrag/eval/ragas_eval.py`, `experiments/run_generation.py`, `tests/eval/test_ragas_eval.py`

**Interfaces:**
- Produces:
  - `run_generation.py --dataset hotpot rfc --n 200` over configs `bm25`, `dense`, `hybrid_fixed+rr`, `qaf+rr` → `results/generation/<dataset>__<config>.json` with `metrics: {em, f1, citation_hit}` (citation_hit = any cited doc is relevant) and per-query answers.
  - `ragas_scores(samples: list[dict(question, answer, contexts)]) -> dict[str, float]` using ragas `faithfulness` and `context_relevance`/`context_precision` (whichever the installed version provides) with Ollama LLM + local embeddings; 100 samples, Tier B.
  - `results/generation/human_check.csv`: 30 random (qid, config, answer) rows with empty `faithful(0/1)` column for Parth.

- [ ] **Step 1:** Test `ragas_scores` input validation with a stubbed evaluator (empty contexts → ValueError) → FAIL → implement → PASS.
- [ ] **Step 2:** Run generation (run in background; ~3–4 h). Expected: 8 result files.
- [ ] **Step 3:** Run RAGAS if time (Tier B).
- [ ] **Step 4 (HUMAN — Parth):** fill `human_check.csv` (~45 min); record in `evidence/log.md`.
- [ ] **Step 5:** Commit `feat(exp): generation evaluation and human-check sheet`.

### Task 16: DAT-style LLM α baseline (Tier B)

**Files:**
- Create: `src/hrag/baselines/dat.py`, `experiments/run_dat.py`, `tests/baselines/test_dat.py`

**Interfaces:**
- Produces: `dat_alpha(query: str, dense_top: Doc | None, sparse_top: Doc | None) -> float` — LLM rates each top-1 doc's relevance 0–5 (json_mode); α = s_d/(s_d+s_s), 0.5 when both 0; if one list empty, α = 1 or 0. Pipeline `alpha_mode="dat"` uses it. Results `results/dat/` on SciFact + RFC with latency.

- [ ] **Step 1: Tests:** fake chat scores (4,1) → α=0.8; (0,0) → 0.5; sparse_top None → 1.0. FAIL → implement → PASS.
- [ ] **Step 2:** Run `experiments/run_dat.py`. **Step 3:** Commit `feat: DAT-style LLM alpha baseline`.

### Task 17: PageIndex-style vectorless retrieval (Tier C)

**Files:**
- Create: `src/hrag/baselines/pageindex.py`, `experiments/run_pageindex.py`, `tests/baselines/test_pageindex.py`

**Interfaces:**
- Produces:
  - `build_tree(doc: Doc) -> Node` where `Node(title: str, summary: str, span: tuple[int,int], children: list[Node])` from RFC section numbering; leaf summaries by LLM (≤30 words), cached.
  - `navigate(query: str, forest: list[Node], max_steps=4) -> list[tuple[int,int]]` — LLM picks the document, then descends child-by-child by title+summary, returns chosen leaf spans.
  - `PageIndexRetriever.search(query, k=5) -> Ranking` mapping spans to RFC semantic chunk ids (overlap). Results `results/pageindex/rfc.json` with recall@5, answer F1 (via `generate_answer`), latency.

- [ ] **Step 1: Tests:** `build_tree` on fixture with sections `1`, `1.1`, `2` → root has 2 children, `1` has 1 child; `navigate` with fake chat always choosing first child returns a leaf span. FAIL → implement → PASS.
- [ ] **Step 2:** Run on 50–100 RFC questions. **Step 3:** Commit `feat: PageIndex-style vectorless baseline`.

### Task 18: Paper assets generator

**Files:**
- Create: `experiments/make_paper_assets.py`, `tests/test_paper_assets.py`

**Interfaces:**
- Consumes: result file contract.
- Produces in `paper/tables/`: `main_results.tex` (datasets × configs × encoders, nDCG@10/Recall@10/MRR@10, best bold, `†` where p<0.05 vs hybrid_fixed+rr), `ablation.tex`, `generation.tex`, `latency.tex`, `qaf_coefficients.tex`, optional `matryoshka.tex`, `dat.tex`, `pageindex.tex` (skipped silently if results absent). In `paper/figures/`: `alpha_sweep.pdf`, `qaf_alpha_hist.pdf` (predicted α by query type id-heavy vs natural), `latency_breakdown.pdf`, `architecture.pdf` (drawn with matplotlib boxes or TikZ in `sections/`), optional `matryoshka.pdf`. Also `paper/numbers.tex` with `\newcommand` macros (e.g. `\QAFgainSciFact`) for every number quoted in prose.

- [ ] **Step 1:** Test on toy result JSONs → `main_results.tex` contains `\textbf` on the max cell and compiles inside a minimal `\documentclass{article}` (`latexmk` if installed, else just snapshot string). FAIL → implement → PASS.
- [ ] **Step 2:** Run on real results. **Step 3:** Commit `feat: generate paper tables, figures and number macros`.

### Task 19: Streamlit demo

**Files:**
- Create: `app/streamlit_app.py`, `app/README.md`, `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `common.load_dataset`, `common.load_indexes`, `Pipeline`.
- Behaviour: sidebar selects dataset (scifact/rfc/hotpot), encoder, α mode (fixed slider / QAF), rerank on/off, expansion on/off; `st.cache_resource` for indexes/models. Main: query box + example queries per dataset; four columns BM25 / Dense / Fused / Reranked (title, snippet, score; relevant docs marked when query is from the test set); QAF panel (feature table + chosen α gauge); answer with clickable citations; per-stage latency bar chart. Empty query → `st.info("Type a question")`; `OllamaUnavailable` → `st.warning(<hint>)` and retrieval still shown.

- [ ] **Step 1:** Smoke test with `streamlit.testing.v1.AppTest` on the toy fixture: empty query shows info; with chat stub raising `OllamaUnavailable` the warning appears and four columns render (Review Focus 3, 5). FAIL → implement → PASS.
- [ ] **Step 2:** Run `streamlit run app/streamlit_app.py`; take screenshots of SciFact and RFC queries into `evidence/screenshots/`.
- [ ] **Step 3:** Commit `feat: Streamlit demo`.

### Task 20: IEEE paper draft

**Files:**
- Create: `paper/main.tex` (IEEEtran conference), `paper/sections/{abstract,introduction,related,method,setup,results,discussion,conclusion}.tex`, `paper/refs.bib`, `paper/Makefile`

**Interfaces:**
- Consumes: `paper/tables/*`, `paper/figures/*`, `paper/numbers.tex`, `results/qaf/coefficients_*.json`.

- [ ] **Step 1:** Template + authors block (three students, Chitkara University CSE, supervisor acknowledged per team decision), `\input` sections and tables; `make` builds `main.pdf`.
- [ ] **Step 2:** `refs.bib` with 25–35 references, each verified by web lookup (title/authors/venue/year/DOI or arXiv id). Must include: Lewis et al. 2020 (RAG), Karpukhin et al. 2020 (DPR), Robertson & Zaragoza 2009 (BM25), Cormack et al. 2009 (RRF), Thakur et al. 2021 (BEIR), Khattab & Zaharia 2020 (ColBERT), Nogueira & Cho 2019 (BERT rerank), Xiao et al. 2023 (BGE/C-Pack), EmbeddingGemma 2025 tech report, Kusupati et al. 2022 (Matryoshka), Bruch et al. 2023 (fusion functions), DAT 2025, PageIndex (VectifyAI), Es et al. 2024 (RAGAS), Yang et al. 2018 (HotpotQA), Wadden et al. 2020 (SciFact), Maia et al. 2018 (FiQA), Ma et al. 2023 (query rewriting), Gao et al. 2023 (RAG survey), Ji et al. 2023 (hallucination survey), Qwen2.5 report, Chen et al. 2017 (DrQA), Lin et al. 2021 (Pyserini/hybrid), Kuzi et al. 2020 (hybrid lexical-semantic).
- [ ] **Step 3:** Write sections in order method → setup → results → discussion → related → introduction → abstract → conclusion. Every number via `numbers.tex` macros or `\input` tables. Contributions list (3–4 bullets) in the introduction; Limitations paragraph (local 7B judge, subsets, LLM-assisted question generation verified by one person). Abstract stays consistent with the approved one, adding QAF and EmbeddingGemma.
- [ ] **Step 4:** Style pass against the banned-phrase list (`grep -niE "delve|landscape|seamless|crucial role|worth noting|rapidly evolving|leverag" paper/sections/*.tex` → no matches) and IEEE checks: ≤6 pages + refs, figures referenced before they appear, captions above tables / below figures, all `\cite` keys resolve (no `?` in PDF).
- [ ] **Step 5:** Commit `docs(paper): IEEE draft generated from results`.

### Task 21: Reproducibility README and evidence folder

**Files:**
- Create: `README.md`, `evidence/log.md`, `evidence/screenshots/.gitkeep`, `Makefile`

- [ ] **Step 1:** `Makefile` targets: `setup`, `test`, `indexes`, `experiments-a`, `experiments-b`, `generation`, `assets`, `paper`, `demo`. README: hardware, setup, gated-model login, Ollama pull, `make` sequence, expected runtimes, tier explanation, results layout.
- [ ] **Step 2:** `evidence/log.md` template: date, meeting with supervisor, attendees, decisions, progress; pre-filled with today's design/plan entries.
- [ ] **Step 3:** Verify from clean shell: `make test` passes (non-slow) and `make assets paper` rebuilds `paper/main.pdf`.
- [ ] **Step 4:** Commit `docs: README, Makefile and evidence log`.
