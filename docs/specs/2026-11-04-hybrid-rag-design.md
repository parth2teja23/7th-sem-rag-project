# Hybrid Retrieval RAG — Project & Paper Design

**Date:** 2026-11-04
**Team:** Parth Tuteja (2310991172), Reetika (2310991267), Rishabh Puri (2310991269)
**Supervisor:** Dr Rajat Takkar — Chitkara University, CSE, COOP-1 (22CS037), Sem VII
**Source of truth for scope:** `Team 39 - Abstract Hybrid Retrieval RAG.pdf` (the approved abstract) and `rubrics-coop1.docx`.

---

## 1. Intent — why this exists

The team must produce, for the college's "Research Paper" pathway:

1. **A research paper in IEEE conference format**, written from scratch around the approved abstract, and **submitted to a reputed (IEEE / Scopus-indexed) venue** so that a *submission acknowledgement* can be shown as **Proof of Communication between 12–15 Nov 2026**.
2. **A working, demonstrable system** for the external evaluation (Demonstration 20 marks, Viva 20 marks, File work 20 marks), expected 1st/2nd week of December 2026.

The goal is the **"Excellent" band of every rubric row**. The rubric rows that drive design decisions:

| Rubric row | What it forces |
|---|---|
| Novelty & Originality | A named contribution of our own, beyond "dense + BM25 + RRF + rerank" (which is well published). |
| Technical content | A detailed, reproducible methodology section. |
| Prior art | A literature review that explicitly differentiates from DPR, BM25, ColBERT, RRF, existing hybrid-RAG, EmbeddingGemma, PageIndex. |
| Results | Real tables/figures from real runs, against baselines, with ablations. |
| Formatting & integrity | IEEE template, original prose, all claims cited, no plagiarism. |
| Submission status / platform quality / proof | Submitted to a reputed venue with acknowledgement email saved. |
| Working model / validation | Stable demo UI; tests; reproducible experiment scripts. |
| Evidence & file work | Screenshots, logs, submission proof collected in one place. |

**Hard deadline:** the paper must be complete and submitted by **~11 Nov 2026** (7 days from today) to have proof for the 12–15 Nov window.

## 2. Goals, non-goals, anti-goals

**Goals**
- A hybrid retrieval pipeline exactly as the abstract describes: semantic chunking with overlap → dual indexing (dense + BM25) → optional query rewriting/expansion → parallel retrieval → weighted Reciprocal Rank Fusion → cross-encoder reranking → metadata filtering + dedup → top-k into an instruction-tuned LLM.
- **Our contribution — Query-Adaptive Fusion (QAF):** per-query fusion weight α(q) predicted from cheap query features, compared against the best *fixed* α found by grid search.
- **Two dense encoders** compared throughout: `bge-small-en-v1.5` and Google **EmbeddingGemma** (`google/embeddinggemma-300m`), including its Matryoshka truncation (768/512/256/128 dims).
- Full evaluation: retrieval (Recall@k, MRR@10, nDCG@10), generation (EM, token F1, RAGAS faithfulness & context relevance with a local judge), latency per stage, ablation study.
- A Streamlit demo showing all retrievers side by side, the α chosen by QAF, and the cited answer.
- An IEEE-format LaTeX paper (~6 pages + refs) whose every number is generated from experiment output files.

**Stretch goal (only if core is done)**
- **PageIndex-style vectorless retrieval** (LLM builds a table-of-contents tree per long document and reasons down it) evaluated **only on the long-document corpus**, ~50–100 questions, as a reasoning-based comparison point with latency cost. The paper must be complete and submittable without it.

**Non-goals**
- Training or fine-tuning embedding models or the LLM.
- Full-size benchmark runs (we use documented subsets).
- Production deployment, auth, multi-user serving.
- Paid APIs — everything runs locally.

**Anti-goals (failure even if it "works")**
- Any fabricated, estimated or hand-typed result number in the paper.
- Copied text from sources or other GitHub projects (rubric: plagiarism; guidelines: don't replicate GitHub projects).
- Claiming novelty for plain hybrid RRF.
- A demo that crashes or needs live debugging during evaluation.
- Missing the 12–15 Nov proof window because experiments over-ran.

## 3. Constraints

- **Hardware:** single Apple M5 MacBook, 32 GB RAM. PyTorch on MPS for encoders/reranker; Ollama for the LLM. No cloud GPU, no paid API.
- **Time:** ~7 days to submission. Experiments must fit within a few hours of total compute; generation evaluation dominates cost.
- **Language/stack:** Python 3.11+. Not a git repo today — will be initialised.
- **Integrity:** all prose written fresh; all references real and verified (title, authors, venue, year).

## 4. Decided design (the "how")

### 4.1 Data
- **BEIR SciFact** (5k docs, 300 test queries) — scientific claims, terminology-heavy.
- **BEIR FiQA-2018** (57k docs, 648 test queries) — finance, natural language; subsample corpus if index time is excessive, documenting the method.
- **QA set for generation metrics:** a 300–500 question subset of **Natural Questions** or **HotpotQA** (BEIR versions have gold passages; answers from original dataset) — builder picks whichever has cleaner answer strings.
- **Custom identifier-heavy long-document corpus:** public technical documents with dense identifiers/acronyms/codes (e.g., IETF RFCs). ~20–50 long structured documents, chunked. ~100–150 questions with gold chunk and short answer, generated with LLM assistance then **manually verified by the team** (documented as such). This corpus is also where PageIndex runs.
- All subsets fixed by seed; exact IDs saved to disk for reproducibility.

### 4.2 Pipeline components
| Stage | Choice |
|---|---|
| Chunking | Semantic/structural: split on headings/paragraphs, merge to a target token size, configurable overlap. Baseline for ablation: fixed-size chunks. (BEIR passages are already short — chunking ablation runs on the custom corpus.) |
| Dense | `bge-small-en-v1.5` and `embeddinggemma-300m` via sentence-transformers; FAISS (HNSW or flat IP — flat is fine at this scale; the abstract says ANN, so use HNSW and report it). |
| Sparse | BM25 (`bm25s` or `rank_bm25`), standard tokenisation, k1/b defaults. |
| Query rewriting/expansion | Optional LLM rewrite (Qwen2.5-7B) producing an expanded query; toggled in ablation. |
| Fusion | Weighted RRF: score(d) = α·1/(k+rank_dense) + (1−α)·1/(k+rank_sparse), k=60. |
| QAF (ours) | Features of q: share of tokens that are identifiers/numbers/ALL-CAPS/mixed alnum, query length, avg IDF of query terms, out-of-vocabulary rate vs encoder tokenizer, BM25 top-score margin, dense top-score margin. Map features → α via (a) a transparent rule/logistic model, trained on a held-out dev split by choosing per-query oracle α. Must report against best fixed α and against oracle α (upper bound). |
| Rerank | `BAAI/bge-reranker-base` (or v2-m3 if speed OK) on the over-retrieved fused top-N (N=50 default, sweep 20/50/100). |
| Post-process | Metadata filter hook + near-duplicate removal, then top-k (k=5) into prompt. |
| Generation | Qwen2.5-7B-Instruct via Ollama; prompt requires citing chunk IDs; temperature 0. |

### 4.3 Experiments
1. **Baselines:** BM25-only, Dense-only (each encoder), Hybrid fixed-α RRF, Hybrid + rerank.
2. **Ours:** QAF hybrid, QAF + rerank — per encoder.
3. **α sweep:** fixed α ∈ {0, 0.1, …, 1} per dataset → figure.
4. **Ablation:** remove one at a time — semantic chunking (custom corpus), query expansion, QAF (→ fixed α), reranker.
5. **EmbeddingGemma Matryoshka:** 768/512/256/128 dims vs nDCG and index size/latency.
6. **Rerank depth:** N ∈ {20, 50, 100} vs nDCG and latency.
7. **Generation:** EM/F1 on QA subset; RAGAS faithfulness & context relevance on a sampled subset (~100) with local judge; small human spot-check (team, ~30 items) documented.
8. **Latency:** per-stage timings (mean, p95) for each configuration.
9. **(Stretch) PageIndex** on custom long-doc corpus vs best hybrid: Recall/answer F1/latency.
10. **Statistical check:** paired bootstrap or paired t-test on per-query nDCG for headline comparisons.

All outputs written as JSON/CSV under `results/`; one script regenerates every paper table and figure.

### 4.4 Demo
Streamlit app: pick corpus + encoder, type a question → four columns (BM25, Dense, Fused, Reranked) with scores; shows QAF features and chosen α; generated answer with clickable citations; per-stage latency bar. Pre-built indexes loaded at start so it's fast and stable.

### 4.5 Paper
- IEEE conference LaTeX template (`IEEEtran`), ~6 pages + references.
- Sections: Abstract, Introduction (problem, gap, contributions list), Related Work (sparse, dense, hybrid/fusion, rerankers, RAG, embedding models incl. EmbeddingGemma, vectorless/reasoning retrieval incl. PageIndex), Methodology (architecture figure, chunking, indexing, QAF, fusion, rerank, generation), Experimental Setup, Results & Discussion, Ablation, Latency, Limitations & Threats to Validity, Conclusion & Future Work, References (~25–35 real, verified citations).
- Abstract revised from the approved one to name QAF and EmbeddingGemma while staying consistent with the internally submitted abstract (rubric: "fully consistent with internal submission").
- Tables/figures auto-generated (`.tex` tables, PDF plots).

### 4.6 Supporting deliverables
- `evidence/` folder: meeting log template, screenshots of demo, submission acknowledgement.
- Shortlist of reputed venues with open/rolling deadlines accepting submissions by ~11 Nov (IEEE-sponsored or Scopus-indexed conferences); team chooses and submits.
- README with reproduction steps.

## 4.7 Calls made after builder review (Claude's calls — team to check)

**Novelty framing.** Per-query/dynamic fusion weighting has prior work (e.g. DAT — Dynamic Alpha Tuning, 2025, LLM-scored; Bruch et al. on fusion functions). QAF is therefore framed narrowly and honestly: *a training-light, LLM-free, interpretable per-query α predictor from cheap lexical/score features, targeted at identifier-heavy queries, with near-zero added latency*. Related Work cites and contrasts DAT explicitly. A **DAT-style LLM-scored α baseline** is run on SciFact + custom corpus if time allows (priority tier B) — the comparison point is accuracy vs latency.

**QAF training protocol.** Train one QAF model on pooled dev queries from SciFact + FiQA (BEIR train/dev splits, disjoint from test), using per-query oracle α as targets. Evaluate on all test sets, including the custom corpus as an *unseen-domain* test. Also report leave-one-dataset-out. If gain on the custom corpus is small, say so and analyse by query type (identifier vs natural-language buckets).

**Generation-evaluation scope.** To support the abstract's claim that hybrid yields more faithful answers than single strategies, generation (EM/F1) runs on 4 configs: BM25-only, Dense-only (better encoder), Hybrid fixed-α + rerank, QAF + rerank — on a 200-question QA subset. RAGAS faithfulness/context relevance on 100 items for the same 4 configs if compute allows, else for Dense-only vs QAF+rerank only.

**Priority tiers (cut from the bottom up if compute/time runs over):**
- **Tier A — must have for submission:** SciFact + custom corpus; BM25, Dense (both encoders), fixed-α hybrid, QAF, rerank (N=50); α sweep; generation EM/F1 (4 configs); ablation (QAF, rerank, chunking); per-stage latency; significance test.
- **Tier B — include if time:** FiQA; RAGAS; query-expansion ablation; EmbeddingGemma Matryoshka dims; DAT-style baseline.
- **Tier C — post-submission / demo only:** rerank-depth sweep; PageIndex; human spot-check beyond 30 items.

## 4.8 Team answers (2026-11-04)

- **Venue:** none chosen; no external deadline imposed. Write to the generic IEEE conference standard (IEEEtran, two-column, ~6 pages + refs). Venue selection and submission are outside this build. (Note: rubric Component 3 still needs a submission acknowledgement by 12–15 Nov — team's responsibility.)
- **Prose style:** Claude drafts the full paper. Prose must read as human academic writing: varied sentence length, concrete specifics, first-person plural ("we"), no stock AI phrasing (no "delve", "in today's rapidly evolving", "it is worth noting", "plays a crucial role", "landscape", "seamless", triplet lists everywhere, em-dash chains). The team will then rewrite in their own voice.
- **Roles:** Parth Tuteja is Lead and performs all manual tasks (verifying custom-corpus questions, human spot-check, final edit). Reetika and Rishabh Puri's roles to be recorded by the team for the rubric file.

## 5. Left to the builder
- Exact libraries (bm25s vs rank_bm25, FAISS index params), file layout, test granularity.
- NQ vs HotpotQA choice, FiQA corpus subsample size.
- QAF model form (rules vs logistic regression) — pick whichever gives clearer, more explainable results; report honestly even if gain is modest.
- Choice of RFCs/technical docs for the custom corpus.
- RAGAS version/judge config given a local LLM.

## 6. Risks
- **QAF gives little or no gain.** Report honestly; frame as analysis (when does adaptive weighting help — identifier-heavy queries) with the oracle-α upper bound showing headroom. Do not cherry-pick.
- **Generation eval too slow.** Reduce sample sizes; document.
- **Venue deadline.** Venue shortlist is produced on day 1, in parallel with code.
