# Project evidence log

| Date | Who | Activity | Time |
|---|---|---|---|
| 2026-11-05 | Claude (AI assistant) | First-pass screening of the 180 auto-generated RFC questions: proposed 94 keeps, 7 rewordings, 79 rejections, with a reason per row. | — |
| 2026-11-05 | Parth Tuteja, Reetika, Rishabh Puri | Team verification: checked every row of `data/rfc/questions.csv` against the RFC source text and confirmed the final set of 101 verified questions. | _fill in_ |

Note for the paper: the RFC question set was produced by LLM generation (qwen2.5:7b-instruct) and automatic filtering. An AI-assisted screening pass followed, and the team then verified every row.

## Supervisor meeting template

| Field | Value |
|---|---|
| Date | _fill in_ |
| Meeting with supervisor (Dr Rajat Takkar) | _fill in_ |
| Attendees | _fill in_ |
| Decisions | _fill in_ |
| Progress | _fill in_ |

## Dated entries

### 2026-11-04 - design and plan

- Design: hybrid retrieval (BM25 + dense, weighted RRF) with a query-adaptive fusion model, cross-encoder rerank, local generation with Ollama; two encoders (bge-small, EmbeddingGemma); datasets SciFact, FiQA subset, HotpotQA (500 q) and a custom RFC set.
- Plan: tiered experiments (A core retrieval, B ablation/Matryoshka/generation/RAGAS, C rerank depth), then paper and demo.
- Meeting with supervisor: _fill in_

### 2026-11-05 - experiments and paper draft

- Ran the retrieval, QAF, ablation, Matryoshka, rerank-depth and generation experiments; results are in `results/`.
- Generated paper tables, figures and number macros with `experiments/make_paper_assets.py` and drafted the paper in `paper/`.
- The query-expansion ablation was not run.
- Meeting with supervisor: _fill in_

## Pending human steps

- [ ] Fill the `faithful` column in `results/generation/human_check.csv` (30 rows).
- [ ] Fill the time columns in this log.
- [ ] Take demo screenshots into `evidence/screenshots/` if missing.
- [ ] Add supervisor meeting notes (fields above).
