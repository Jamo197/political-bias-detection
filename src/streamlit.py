"""
Political RAG Bias Detector — Streamlit Application
=====================================================
Pipeline:
  1. Provide / load text  (+ optional metadata from DB)
  2. Choose retrieval method (Simple / TwoStage / HyDE) — retriever is
     initialised eagerly at sidebar render time so HF model downloads
     happen once on startup, not on first button click.
  3. Click "Retrieve Chunks" to fetch and inspect relevant excerpts.
  4. Click "Run Prediction" — auto-retrieves first if step 3 was skipped,
     then asks the LLM to predict the bias score (0-10) with justification.
  5. Compare against CHES ground truth.
  6. Results are logged to disk.
  Use "Start New" to reset and analyse a different text.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Make project root importable when launched via `streamlit run src/streamlit.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging.log_run import log_evaluation_run
from rag.evaluator import BaselineEvaluator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_MODES = ["Simple", "TwoStage", "HyDE"]

LLM_MODELS = ["Mistral", "OpenAI", "Gemini"]

DATA_PATH = (
    "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset"
    "_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bundestag_speeches_chunks"

_SESSION_RESET_KEYS = [
    "input_text",
    "meta_party",
    "meta_speaker",
    "meta_year",
    "retrieved_chunks",
    "hyde_docs",
    "prediction",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_bias_label(score: float) -> str:
    """Maps a continuous 0-10 bias score to a human-readable category."""
    if score is None:
        return "Unknown"
    score = max(0.0, min(10.0, float(score)))
    if score < 1.5:
        return "Extreme Left"
    elif score < 3.5:
        return "Left"
    elif score < 4.5:
        return "Center-Left"
    elif score <= 5.5:
        return "Centrist / Moderate"
    elif score <= 6.5:
        return "Center-Right"
    elif score <= 8.5:
        return "Right"
    else:
        return "Extreme Right"


@st.cache_data
def load_test_dataset() -> pd.DataFrame:
    """Loads and caches the evaluation dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        df["full_text"] = df["full_text"].astype(str).str.replace("/comma", ",", regex=False)
        rename_dict = {
            "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
            "Linke": "DIE LINKE"
        }
        df["party"] = df["party"].replace(rename_dict)
        return df
    except FileNotFoundError:
        st.error(f"Test dataset not found at `{DATA_PATH}`")
        return pd.DataFrame()


@st.cache_resource
def get_evaluator() -> BaselineEvaluator:
    """Singleton evaluator (holds API clients + CHES DB)."""
    return BaselineEvaluator()


@st.cache_resource
def get_retriever(mode: str):
    """
    Constructs and caches a PoliticalRAGRetriever for the chosen mode.
    Called eagerly after the sidebar selectbox so HF model downloads happen
    at app init rather than on first button click.
    Returns None if Qdrant is unavailable.
    """
    _MODE_MAP = {
        "simple": "simple",
        "twostage": "two_stage",
        "hyde": "hyde",
    }
    internal_mode = _MODE_MAP.get(mode.lower().replace(" ", ""), mode.lower())

    try:
        from rag.retrieval import PoliticalRAGRetriever
        return PoliticalRAGRetriever(
            qdrant_url=QDRANT_URL,
            chunk_collection=COLLECTION_NAME,
            retrieval_mode=internal_mode,
        )
    except Exception as e:
        st.warning(f"Could not connect to Qdrant ({e}). RAG disabled.")
        return None


def chunks_to_context_dicts(points) -> List[Dict[str, Any]]:
    """Converts raw Qdrant PointStruct list to plain dicts for the LLM context."""
    result = []
    for p in points:
        payload = p.payload or {}
        result.append(
            {
                "text": payload.get("text", ""),
                "party": payload.get("party", ""),
                "country": payload.get("country", ""),
                "speaker": payload.get("speaker", ""),
                "date": payload.get("date", ""),
                "speech_id": payload.get("speech_id", ""),
                "score": round(float(p.score), 4),
            }
        )
    return result


def reset_session():
    """Clears all run-specific session state keys and triggers a rerun."""
    for key in _SESSION_RESET_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def run_streamlit_app():
    st.set_page_config(page_title="Political RAG Bias Detector", layout="wide")
    st.title("Political Bias Detection")
    st.caption("RAG-augmented LLM pipeline validated against CHES expert scores.")

    evaluator = get_evaluator()
    df = load_test_dataset()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Pipeline Configuration")

        llm_choice = st.selectbox("LLM", LLM_MODELS)

        st.divider()
        st.subheader("RAG Parameters")
        retrieval_mode = st.selectbox("Retrieval Method", RETRIEVAL_MODES)
        k_chunks = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)

    # Eagerly init retriever for the selected mode — this warms up HF model
    # downloads on app start (and on mode switch) rather than on first click.
    retriever = get_retriever(retrieval_mode)

    # -----------------------------------------------------------------------
    # Step 1 — Text Input
    # -----------------------------------------------------------------------
    st.subheader("1. Text Input")

    col_load, col_reset, col_info = st.columns([1, 1, 4])
    with col_load:
        if st.button("Load Random Row", disabled=df.empty):
            sample = df.sample(1).iloc[0]
            st.session_state["input_text"] = str(sample.get("full_text", ""))
            st.session_state["meta_party"] = str(sample.get("party", ""))
            st.session_state["meta_speaker"] = str(
                sample.get("twitter_handle", sample.get("author", ""))
            )
            raw_year = sample.get("year", sample.get("date", "2021"))
            try:
                st.session_state["meta_year"] = int(str(raw_year)[:4])
            except (ValueError, TypeError):
                st.session_state["meta_year"] = 2021
            # Clear stale results from a previous run
            for key in ("retrieved_chunks", "hyde_docs", "prediction"):
                st.session_state.pop(key, None)

    with col_reset:
        if st.button("Start New"):
            reset_session()
            st.rerun()

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    input_text = st.text_area(
        "Text to Analyse",
        height=160,
        key="input_text",
    )

    # -----------------------------------------------------------------------
    # Step 1a — Metadata (party / speaker / year)
    # -----------------------------------------------------------------------
    st.subheader("1a. Metadata")
    st.caption(
        "Pre-filled when loading from dataset or retrieved from the vector DB. "
        "Used for CHES ground truth lookup."
    )
    if "meta_party" not in st.session_state:
        st.session_state["meta_party"] = ""
    if "meta_speaker" not in st.session_state:
        st.session_state["meta_speaker"] = ""
    if "meta_year" not in st.session_state:
        st.session_state["meta_year"] = 2021

    m1, m2, m3 = st.columns(3)
    meta_party = m1.text_input("Party", key="meta_party")
    meta_speaker = m2.text_input("Speaker", key="meta_speaker")
    meta_year = m3.number_input(
        "Year",
        min_value=1990,
        max_value=2030,
        step=1,
        key="meta_year",
    )

    # -----------------------------------------------------------------------
    # Step 2 — Retrieval
    # -----------------------------------------------------------------------
    st.subheader("2. Retrieval")

    def _do_retrieve() -> tuple:
        """Run retrieval and return (retrieved_chunks, hyde_docs). Writes to session state."""
        chunks: List[Dict[str, Any]] = []
        docs: List[str] = []

        if retriever is None:
            st.error("Retriever unavailable — check Qdrant connection.")
            return chunks, docs

        if retrieval_mode == "HyDE":
            strategy = retriever.retrieval_strategy
            if getattr(strategy, "llm", None) is None:
                st.warning(
                    "HyDE selected but Ollama is unavailable "
                    "(is `ollama serve` running with the `gemma3` model?). "
                    "Falling back to plain vector search without hypothetical documents."
                )

        with st.spinner(f"Running {retrieval_mode} retrieval..."):
            try:
                if retrieval_mode == "HyDE":
                    strategy = retriever.retrieval_strategy
                    docs = strategy._generate_hypothetical_docs(input_text, num_docs=3)

                points = retriever.search(query=input_text, limit=k_chunks)
                chunks = chunks_to_context_dicts(points)

                # Back-fill metadata from top-1 chunk if not yet set
                if chunks and not st.session_state.get("meta_party"):
                    top = chunks[0]
                    st.session_state["meta_party"] = top.get("party", "")
                    st.session_state["meta_speaker"] = top.get("speaker", "")
                    date_str = top.get("date", "")
                    if date_str:
                        try:
                            st.session_state["meta_year"] = int(date_str[:4])
                        except ValueError:
                            pass

            except Exception as e:
                st.error(f"Retrieval failed: {e}")

        st.session_state["retrieved_chunks"] = chunks
        st.session_state["hyde_docs"] = docs
        return chunks, docs

    if st.button("Retrieve Chunks", disabled=not input_text.strip() or retriever is None):
        _do_retrieve()
        st.rerun()

    # Persist state across reruns
    retrieved_chunks: List[Dict[str, Any]] = st.session_state.get("retrieved_chunks", [])
    hyde_docs: List[str] = st.session_state.get("hyde_docs", [])

    # 2a — HyDE hypothetical docs
    if retrieval_mode == "HyDE" and hyde_docs:
        with st.expander("HyDE: Generated Hypothetical Documents", expanded=False):
            for i, doc in enumerate(hyde_docs, 1):
                st.markdown(f"**Excerpt {i}:** {doc}")

    # -----------------------------------------------------------------------
    # Step 3 — Retrieved Chunks
    # -----------------------------------------------------------------------
    if retrieved_chunks:
        st.subheader("3. Retrieved Chunks")
        for i, chunk in enumerate(retrieved_chunks, 1):
            with st.expander(
                f"Chunk {i} — {chunk.get('party', 'Unknown Party')} | "
                f"{chunk.get('speaker', 'Unknown Speaker')} | "
                f"{chunk.get('date', '')[:10]}  (score: {chunk.get('score', 0):.4f})",
                expanded=(i == 1),
            ):
                st.write(chunk.get("text", ""))
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"Party: **{chunk.get('party', '—')}**")
                meta_cols[1].caption(f"Speaker: **{chunk.get('speaker', '—')}**")
                meta_cols[2].caption(f"Date: **{chunk.get('date', '—')}**")

    # -----------------------------------------------------------------------
    # Step 4 — Bias Prediction
    # -----------------------------------------------------------------------
    st.subheader("4. Bias Prediction")

    if st.button("Run Prediction", type="primary", disabled=not input_text.strip()):
        # Auto-retrieve if the user skipped step 2
        if not retrieved_chunks:
            retrieved_chunks, hyde_docs = _do_retrieve()

        with st.spinner(f"Predicting with {llm_choice}..."):
            prediction = evaluator.predict_bias(
                text=input_text,
                model_provider=llm_choice,
                context_chunks=retrieved_chunks if retrieved_chunks else None,
            )
        st.session_state["prediction"] = prediction
        st.rerun()

    prediction = st.session_state.get("prediction")

    if prediction:
        predicted_score = prediction.get("bias_score")

        p_col, _ = st.columns([2, 3])
        if isinstance(predicted_score, (int, float)):
            p_col.metric(
                label=f"{llm_choice} Predicted Bias",
                value=f"{predicted_score:.1f} / 10",
                delta=get_bias_label(predicted_score),
                delta_color="off",
            )
        else:
            p_col.error("Prediction failed — see justification below.")

        with st.expander("Show Justification", expanded=False):
            st.info(prediction.get("justification", "No justification returned."))

        # -------------------------------------------------------------------
        # Step 5 — CHES Ground Truth Comparison
        # -------------------------------------------------------------------
        st.subheader("5. CHES Ground Truth Comparison")

        current_party = st.session_state.get("meta_party", "")
        current_year = int(st.session_state.get("meta_year", 2021))

        lrecon, galtan, lrgen = evaluator._get_closest_ches_score(
            current_party, current_year
        )

        gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
        gt_col1.metric("CHES lrgen (input text)", f"{lrgen:.2f}" if lrgen is not None else "N/A")
        gt_col2.metric("CHES lrecon", f"{lrecon:.2f}" if lrecon is not None else "N/A")
        gt_col3.metric("CHES galtan", f"{galtan:.2f}" if galtan is not None else "N/A")

        if isinstance(predicted_score, (int, float)) and lrgen is not None:
            delta = abs(predicted_score - lrgen)
            gt_col4.metric("Absolute Error (lrgen)", f"{delta:.2f}")

        if retrieved_chunks:
            st.markdown("**CHES scores of retrieved chunk parties:**")
            chunk_ches_rows = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                c_party = chunk.get("party", "")
                date_str = chunk.get("date", "")
                c_year = int(date_str[:4]) if date_str and len(date_str) >= 4 else current_year
                c_lrecon, c_galtan, c_lrgen = evaluator._get_closest_ches_score(c_party, c_year)
                chunk_ches_rows.append(
                    {
                        "Chunk": i,
                        "Party": c_party,
                        "Speaker": chunk.get("speaker", ""),
                        "Year": c_year,
                        "lrgen": round(c_lrgen, 2) if c_lrgen is not None else None,
                        "lrecon": round(c_lrecon, 2) if c_lrecon is not None else None,
                        "galtan": round(c_galtan, 2) if c_galtan is not None else None,
                        "Sim Score": chunk.get("score", 0),
                    }
                )
            st.dataframe(
                pd.DataFrame(chunk_ches_rows),
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------------------
        # Step 6 — Logging
        # -------------------------------------------------------------------
        log_evaluation_run(
            input_text=input_text,
            llm_choice=llm_choice,
            retrieval_mode=retrieval_mode,
            k_chunks=k_chunks,
            hyde_docs=hyde_docs,
            retrieved_chunks=retrieved_chunks,
            meta_party=current_party,
            meta_speaker=st.session_state.get("meta_speaker", ""),
            meta_year=current_year,
            output_score=predicted_score,
            output_justification=prediction.get("justification"),
            ches_lrgen=lrgen,
            ches_lrecon=lrecon,
            ches_galtan=galtan,
        )
        st.success("Run logged to `logs/evaluation_logs.jsonl`.")


if __name__ == "__main__":
    run_streamlit_app()
else:
    run_streamlit_app()


# TODO:
# Check if the process is running correctly with the evaluator and retriever 
# Check if the parties are mapped correctly for the CHES lookup
# Check the score, it looked weird for HyDE
# Also add that the process can run without streamlit, e.g. via a CLI interface or as a pure function call, for easier testing and integration into other pipelines.

# HyDE retrieval are in german and english:
# Excerpt 1: Here are three hypothetical parliamentary speech excerpts, mirroring the style of the German parliament, responding to the provided statement:

# Excerpt 2: “Meine Damen und Herren, diese Rhetorik der Ausgrenzung ist nicht nur moralisch verwerflich, sondern eine katastrophale Strategie. To blame Eastern Europeans for logistical challenges is a deliberate attempt to deflect responsibility from systemic failures – a dangerous game indeed.”

# Excerpt 3: “Wir dürfen uns nicht von nationalistischen Simplizitäten blenden lassen. The consequences of such policies are precisely those we now witness: shortages, soaring prices, and a destabilized economy. This demands a sober and pragmatic response, not inflammatory accusations.”