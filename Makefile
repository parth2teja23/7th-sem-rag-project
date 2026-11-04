# Reproduction targets. Run from the repository root.
# See README.md for the order, expected runtimes and the Hugging Face / Ollama prerequisites.

PY ?= .venv/bin/python
BATCH ?= 16

.PHONY: setup test indexes experiments-a experiments-b generation assets paper demo

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

# BM25 + dense indexes for every dataset and both encoders (about 1.5 h on an M-series Mac).
indexes:
	$(PY) experiments/build_indexes.py --batch-size $(BATCH)

# Tier A: train QAF + alpha*, then the core retrieval comparison.
experiments-a:
	$(PY) experiments/train_qaf.py
	$(PY) experiments/run_retrieval.py

# Tier B and C: ablation, Matryoshka, rerank depth.
experiments-b:
	$(PY) experiments/run_ablation.py
	$(PY) experiments/run_matryoshka.py
	$(PY) experiments/run_rerank_depth.py

# Tier B: EM / F1 generation (Ollama), then RAGAS (needs RAGAS_PYTHON, see README).
generation:
	$(PY) experiments/run_generation.py
	$(PY) experiments/run_generation.py --ragas

assets:
	$(PY) experiments/make_paper_assets.py --results results/ --out paper/ --dataset-stats

paper:
	$(MAKE) -C paper

demo:
	.venv/bin/streamlit run app/streamlit_app.py
