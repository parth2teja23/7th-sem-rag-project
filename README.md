# Hybrid-retrieval RAG with query-adaptive fusion

Team 39, 7th semester project, Chitkara University.

Authors: Parth Tuteja (2310991172), Reetika (2310991267), Rishabh Puri (2310991269).
Supervisor: Dr Rajat Takkar.

The project compares BM25, dense retrieval and hybrid (weighted reciprocal rank fusion) retrieval, adds a small model that picks the fusion weight per query (QAF, query-adaptive fusion), and tests a cross-encoder reranker on top. It also measures answer quality with a local LLM. The paper is in `paper/`, the demo is in `app/`.

## Hardware used

All numbers were produced on one Apple M5 laptop with 32 GB RAM (encoders on the MPS GPU). Other machines should work but runtimes will differ, and exact scores can shift slightly with a different PyTorch / device.

## Setup

Python 3.12 or newer.

```
make setup
```

This creates `.venv` and installs the package (`pip install -e ".[dev]"`).

Models:

- `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base` download automatically.
- `google/embeddinggemma-2` is gated on Hugging Face. Accept the licence on its model page, then log in once:
  ```
  .venv/bin/huggingface-cli login
  ```
- The LLM runs in Ollama. Install Ollama, then:
  ```
  ollama pull qwen2.5:7b-instruct
  ollama pull nomic-embed-text   # embeddings used by RAGAS
  ollama serve
  ```
- RAGAS needs its own environment (see below).

Tectonic builds the paper: `brew install tectonic`. Without it, upload `paper/` to Overleaf and compile `main.tex` with pdfLaTeX and BibTeX.

### RAGAS environment

`ragas` fails to import in the main venv because of a `langchain_community` version clash. It was run from a second venv with `ragas==0.3.9`, `langchain-community<0.4` and `langchain-ollama<1`. The exact pins are in `requirements-ragas.txt`:

```
python3 -m venv .venv-ragas
.venv-ragas/bin/pip install -r requirements-ragas.txt
export RAGAS_PYTHON=$PWD/.venv-ragas/bin/python
```

`experiments/run_generation.py --ragas` reads the environment variable `RAGAS_PYTHON` and runs `hrag.eval.ragas_eval` as a subprocess of that interpreter (with `PYTHONPATH=src`). If the variable is not set it uses the interpreter running the script, which fails in the main venv.

## Running everything

```
make test           # unit tests; tests marked slow are skipped by default
make indexes        # BM25 and dense indexes (indexes/, not in git)
make experiments-a  # QAF training + core retrieval (Tier A)
make experiments-b  # ablation, Matryoshka, rerank depth (Tier B/C)
make generation     # EM/F1 generation, then RAGAS (needs RAGAS_PYTHON)
make assets         # tables, figures and number macros for the paper
make paper          # builds paper/main.pdf
make demo           # Streamlit demo (see app/README.md)
```

Run them in this order, since later steps read the cached rankings and indexes from earlier ones. Experiments are resumable: finished indexes are skipped, and LLM outputs and reranker scores are cached in `results/cache` (not in git), so an interrupted run continues where it stopped.

`make indexes` uses encoding batch size 16 (`make indexes BATCH=8` to change). Batch 64 in fp32 with Gemma used about 37 GB of memory on our machine and swapped heavily.

The RFC corpus and the 101 verified questions are committed in `data/rfc/`. `experiments/build_rfc_corpus.py` regenerates the draft questions, but you do not need it to reproduce the results. `make assets` also reloads the datasets to refresh `results/dataset_stats.json`; this needs network or a populated Hugging Face cache.

## Expected runtimes (Apple M5, 32 GB)

| Step | Approx. time |
|---|---|
| Gemma encoding | 3 to 9 docs/s (FiQA 20k subset about 38 min) |
| BGE-small encoding | 65 to 169 docs/s |
| `make indexes` (all datasets, both encoders) | see `results/index_build_times.json` |
| Core retrieval experiments incl. index builds (Tier A) | about 4 h |
| Ablation + Matryoshka | about 2 h |
| Generation (EM/F1) | about 2.5 h |
| RAGAS | about 75 min |

Measured index build times (seconds, from `results/index_build_times.json`):

| Dataset | BM25 | bge-small | gemma |
|---|---|---|---|
| SciFact (5,183 docs) | 0.8 | 79 | 901 |
| FiQA (20k subset) | 1.6 | 179 | 2,253 |
| HotpotQA | 0.4 | 29 | 593 |
| RFC | 0.3 | 28 | 980 |

## Tiers

- **Tier A, core retrieval.** BM25, dense, hybrid with a fixed alpha (alpha*, picked on pooled dev data), QAF, and each with the cross-encoder reranker. Datasets: SciFact, FiQA, HotpotQA (500 questions) and RFC (custom: 20 RFCs, 101 team-verified questions). FiQA uses a 20k-document subset that keeps all dev and test positives, so only relative comparisons are meaningful there.
- **Tier B.** Component ablation, Matryoshka truncation of the Gemma embedding, generation EM/F1, and RAGAS.
- **Tier C.** Reranker depth (20, 50, 100).

The query-expansion ablation was not run.

## Results layout

```
results/retrieval/        per dataset/encoder/config metrics
results/qaf/              QAF coefficients, alpha*, oracle targets, leave-one-dataset-out
results/alpha_sweep/      nDCG against alpha
results/ablation/         component ablation
results/matryoshka/       embedding size vs quality
results/rerank_depth/     rerank depth 20/50/100
results/generation/        EM/F1 per config, RAGAS means, human_check.csv
results/latency_summary.json
results/significance.json
results/index_build_times.json
results/dataset_stats.json
results/cache/            gitignored caches (LLM outputs, reranker scores, rankings)
indexes/                  gitignored indexes and trained QAF models
```

`make assets` turns these into `paper/tables/`, `paper/figures/` and `paper/numbers.tex`. `make -C paper check` greps the sections for phrases we decided not to use.

## Evidence

`evidence/log.md` is the project log (supervisor meetings, who did what, how long it took). `evidence/screenshots/` holds demo screenshots.

Still to be done by the team:

- Fill the `faithful` column (0 or 1) in `results/generation/human_check.csv` (30 rows).
- Fill the time columns in `evidence/log.md`.
- Take demo screenshots into `evidence/screenshots/` if missing.
- Add supervisor meeting notes to `evidence/log.md`.
