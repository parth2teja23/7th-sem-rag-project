"""Interactive demo of the hybrid RAG pipeline: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import backend  # noqa: E402
from hrag.llm import OllamaUnavailable  # noqa: E402
from hrag.pipeline import Pipeline, PipelineConfig  # noqa: E402
from hrag.qaf import FEATURE_NAMES, query_features  # noqa: E402

OLLAMA_HINT = ("Ollama is not reachable, so no generated answer. Start it with `ollama serve` "
               "(and pull the model); retrieval results are still shown below.")
STAGES = [("sparse", "BM25"), ("dense", "Dense"), ("fused", "Fused"), ("reranked", "Reranked")]


@st.cache_resource(show_spinner="Loading indexes and models...")
def get_backend(dataset: str, encoder: str):
    # Looked up on the module at call time so tests can replace backend.load_backend.
    return backend.load_backend(dataset, encoder)


def render_column(label, ranking, be, rel, top=5):
    st.subheader(label)
    if not ranking:
        st.caption("stage not active")
        return
    for did, score in ranking[:top]:
        doc = be.corpus[did]
        mark = "✅ " if rel.get(did, 0) > 0 else ""
        st.markdown(f"**{mark}{doc.title or did}**")
        st.caption(f"`{did}` score {score:.4f}")
        st.write(doc.text[:200] + ("..." if len(doc.text) > 200 else ""))


def main():
    st.set_page_config(page_title="Hybrid RAG demo", layout="wide")
    st.title("Hybrid retrieval RAG demo")

    with st.sidebar:
        dataset = st.selectbox("Dataset", backend.DATASETS)
        encoder = st.selectbox("Encoder", backend.ENCODERS)
        be = get_backend(dataset, encoder)
        modes = ["Fixed alpha"] + (["QAF"] if be.qaf is not None else [])
        mode = st.radio("Alpha mode", modes)
        if be.qaf is None:
            st.caption(f"No trained QAF model for {encoder}.")
        alpha = st.slider("alpha (dense weight)", 0.0, 1.0, 0.5, 0.05,
                          disabled=(mode != "Fixed alpha"))
        rerank = st.checkbox("Rerank", value=True)
        expand = st.checkbox("Query expansion (LLM)", value=False)

    examples = list(be.queries.items())[:4]
    st.caption("Example queries (from the test set; relevant docs are marked ✅)")
    cols = st.columns(len(examples) or 1)
    for c, (qid, qtext) in zip(cols, examples):
        if c.button(qtext[:60], key=f"ex_{dataset}_{qid}"):
            st.session_state["query"] = qtext
    query = st.text_input("Question", key="query")

    if not query.strip():
        st.info("Type a question")
        return

    cfg = PipelineConfig(encoder=encoder, alpha_mode="qaf" if mode == "QAF" else "fixed",
                         alpha=alpha, expand=expand, rerank=rerank)
    pipe = Pipeline(cfg, be.corpus, be.sparse, be.dense, be.reranker, be.qaf)
    pipe._vocab = be.vocab or pipe._vocab
    try:
        result, answer = pipe.answer(query)
    except OllamaUnavailable:
        st.warning(OLLAMA_HINT)
        result, answer = pipe.retrieve(query), None

    # qid -> relevance only when the query is a test-set query
    qid = next((i for i, t in be.queries.items() if t == query), None)
    rel = be.qrels.get(qid, {}) if qid else {}

    cols = st.columns(4)
    for col, (key, label) in zip(cols, STAGES):
        with col:
            render_column(label, result.stages.get(key, []), be, rel)

    st.header("Answer")
    if answer is not None:
        st.write(answer.text)
        if answer.cited:
            st.markdown("Citations: " + " ".join(
                f"[{d}](#{d})" for d in answer.cited))
            for d in answer.cited:
                with st.expander(f"{d}: {be.corpus[d].title}"):
                    st.write(be.corpus[d].text)

    if mode == "QAF" and be.qaf is not None:
        st.header("QAF")
        feats = query_features(query, result.stages.get("dense", []), result.stages.get("sparse", []),
                               be.sparse.idf, be.vocab)
        st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES, "value": feats}), hide_index=True)
        st.metric("Chosen alpha", f"{result.alpha:.2f}")
        st.progress(float(result.alpha), text="alpha: 0 = sparse only, 1 = dense only")

    st.header("Latency (ms)")
    lat = pd.Series({k: v for k, v in result.timings_ms.items() if v > 0}, name="ms")
    if len(lat):
        st.bar_chart(lat)


main()
