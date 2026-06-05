import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging.log_run import log_evaluation_run
from rag.evaluator import BiasPredictor
from rag.retrieval import PoliticalRAGRetriever

# ---------------------------------------------------------------------------
# Global Configurations & Constants
# ---------------------------------------------------------------------------
RETRIEVAL_MODES = ["Simple", "TwoStage", "HyDE"]

LLM_MODELS = {
    # --- European Model ---
    "Mistral Small 2603": {"id": "mistralai/mistral-small-2603", "region": "Europe"},
    
    # --- American Models ---
    "GPT-5.4 Mini": {"id": "openai/gpt-5.4-mini", "region": "Americas"},
    "Claude Sonnet 4.6": {"id": "anthropic/claude-sonnet-4.6", "region": "Americas"},
    "Grok 4.3": {"id": "x-ai/grok-4.3", "region": "Americas"},
    "Gemini 3.5 Flash": {"id": "google/gemini-3.5-flash", "region": "Americas"},
    "Llama 4 Maverick": {"id": "meta-llama/llama-4-maverick", "region": "Americas"},
    
    # --- Chinese Models ---
    "DeepSeek V4 Flash": {"id": "deepseek/deepseek-v4-flash", "region": "China"},
    "Qwen 3.7 Plus": {"id": "qwen/qwen3.7-plus", "region": "China"}
}

DATA_PATH = (
    "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset"
    "_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bundestag_speeches_chunks"

SESSION_STATE_DEFAULTS = {
    "input_text": "",
    "meta_party": "",
    "meta_speaker": "",
    "meta_year": 2021,
    "retrieved_chunks": [],
    "hyde_docs": [],
    "prediction": None,
    "active_model_id": "",
    "active_model_region": "",
    "active_retrieval_mode": "",
}

# ---------------------------------------------------------------------------
# Core Business Logic Caching & Formatting Helpers
# ---------------------------------------------------------------------------
def get_bias_label(score: Optional[float]) -> str:
    """Maps a continuous 0-10 CHES bias score to human-readable text categories."""
    if score is None:
        return "Unknown"
    score = max(0.0, min(10.0, float(score)))
    if score < 1.5: return "Extreme Left"
    elif score < 3.5: return "Left"
    elif score < 4.5: return "Center-Left"
    elif score <= 5.5: return "Centrist / Moderate"
    elif score <= 6.5: return "Center-Right"
    elif score <= 8.5: return "Right"
    return "Extreme Right"


@st.cache_data
def load_test_dataset() -> pd.DataFrame:
    """Loads, sanitizes, and caches the primary evaluation data file."""
    try:
        df = pd.read_csv(DATA_PATH)
        df["full_text"] = df["full_text"].astype(str).str.replace("/comma", ",", regex=False)
        rename_dict = {"B90Grune": "BÜNDNIS 90/DIE GRÜNEN", "Linke": "DIE LINKE"}
        df["party"] = df["party"].replace(rename_dict)
        return df
    except FileNotFoundError:
        st.error(f"Critical execution error: Test dataset not found at `{DATA_PATH}`")
        return pd.DataFrame()


@st.cache_resource
def get_evaluator() -> BiasPredictor:
    """Instantiates and persistent-caches our OpenRouter evaluation engine."""
    return BiasPredictor()


@st.cache_resource
def get_retriever(mode: str) -> Optional[PoliticalRAGRetriever]:
    """Warms up and caches the cross-encoder/bi-encoder pipelines instantly upon setup selection."""
    _MODE_MAP = {"simple": "simple", "twostage": "two_stage", "hyde": "hyde"}
    internal_mode = _MODE_MAP.get(mode.lower().replace(" ", ""), mode.lower())
    try:
        return PoliticalRAGRetriever(
            qdrant_url=QDRANT_URL,
            chunk_collection=COLLECTION_NAME,
            retrieval_mode=internal_mode,
        )
    except Exception as e:
        st.warning(f"Could not connect to Qdrant cluster instance ({e}). Vector routing deactivated.")
        return None


def chunks_to_context_dicts(points: List[Any]) -> List[Dict[str, Any]]:
    """Transforms structural Qdrant PointStruct objects directly into pure standard dictionary payloads."""
    return [
        {
            "text": (p.payload or {}).get("text", ""),
            "party": (p.payload or {}).get("party", ""),
            "country": (p.payload or {}).get("country", ""),
            "speaker": (p.payload or {}).get("speaker", ""),
            "date": (p.payload or {}).get("date", ""),
            "speech_id": (p.payload or {}).get("speech_id", ""),
            "score": round(float(p.score), 4) if hasattr(p, "score") and p.score is not None else 0.0,
        }
        for p in points
    ]


def run_pipeline_retrieval(retriever: Optional[PoliticalRAGRetriever], text: str, mode: str, k: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Handles vector index scanning independently of application rendering contexts."""
    if not retriever:
        return [], []
    
    chunks, docs = [], []
    try:
        if mode == "HyDE":
            strategy = retriever.retrieval_strategy
            if getattr(strategy, "llm", None) is None:
                st.warning("Ollama instance unreachable. Reverting back to basic vector paths.")
            docs = strategy._generate_hypothetical_docs(text, num_docs=3)

        points = retriever.search(query=text, limit=k)
        chunks = chunks_to_context_dicts(points)
    except Exception as e:
        st.error(f"Pipeline processing execution interrupted: {e}")
        
    return chunks, docs


# ---------------------------------------------------------------------------
# UI View Modules
# ---------------------------------------------------------------------------
def init_session_state():
    """Guarantees default structural consistency across all execution states."""
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> Tuple[str, str, str, int]:
    """Renders layout selection bars and manages configuration state variables."""
    with st.sidebar:
        st.header("Pipeline Configuration")
        
        # Display regional badges inside the dropdown selections for better clarity
        llm_label = st.selectbox(
            "LLM Provider Target", 
            list(LLM_MODELS.keys()),
            format_func=lambda x: f"{x} ({LLM_MODELS[x]['region']})"
        )
        selected_model_id = LLM_MODELS[llm_label]["id"]
        selected_model_region = LLM_MODELS[llm_label]["region"]

        st.divider()
        st.subheader("RAG Parameters")
        retrieval_mode = st.selectbox("Retrieval Method", RETRIEVAL_MODES)
        k_chunks = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)
        
    return selected_model_id, selected_model_region, retrieval_mode, k_chunks


def render_input_section(df: pd.DataFrame):
    """Manages text sample loading, interactive text areas, and dynamic data binding."""
    st.subheader("1. Text Input")
    col_load, col_reset, _ = st.columns([1, 1, 4])

    if col_load.button("Load Random Row", disabled=df.empty, use_container_width=True):
        sample = df.sample(1).iloc[0]
        st.session_state["input_text"] = str(sample.get("full_text", ""))
        st.session_state["meta_party"] = str(sample.get("party", ""))
        st.session_state["meta_speaker"] = str(sample.get("twitter_handle", sample.get("author", "")))
        
        raw_year = sample.get("year", sample.get("date", "2021"))
        try:
            st.session_state["meta_year"] = int(str(raw_year)[:4])
        except (ValueError, TypeError):
            st.session_state["meta_year"] = 2021
            
        for key in ("retrieved_chunks", "hyde_docs", "prediction"):
            st.session_state[key] = SESSION_STATE_DEFAULTS[key]
        st.rerun()

    if col_reset.button("Start New", use_container_width=True):
        for key in SESSION_STATE_DEFAULTS:
            st.session_state[key] = SESSION_STATE_DEFAULTS[key]
        st.rerun()

    st.text_area("Text to Analyse", height=160, key="input_text")

    st.subheader("1a. Metadata")
    st.caption("Pre-filled when loading from dataset or retrieved from the vector DB. Used for CHES ground truth lookup.")
    
    m1, m2, m3 = st.columns(3)
    m1.text_input("Party", key="meta_party")
    m2.text_input("Speaker", key="meta_speaker")
    m3.number_input("Year", min_value=1990, max_value=2030, step=1, key="meta_year")


def render_retrieval_section(retriever: Optional[PoliticalRAGRetriever], mode: str, k: int):
    """Executes collection queries, backfills context elements, and presents matching partitions."""
    st.subheader("2. Retrieval (Optional - will auto-trigger on prediction if not executed)")
    input_text = st.session_state["input_text"]

    if st.button("Retrieve Chunks", disabled=not input_text.strip() or retriever is None):
        chunks, docs = run_pipeline_retrieval(retriever, input_text, mode, k)
        st.session_state["retrieved_chunks"] = chunks
        st.session_state["hyde_docs"] = docs

        # Smart Metadata Backfilling from Vector DB Payload
        if chunks and not st.session_state["meta_party"]:
            top = chunks[0]
            st.session_state["meta_party"] = top.get("party", "")
            st.session_state["meta_speaker"] = top.get("speaker", "")
            date_str = top.get("date", "")
            if date_str:
                try:
                    st.session_state["meta_year"] = int(date_str[:4])
                except ValueError:
                    pass
        st.rerun()

    if mode == "HyDE" and st.session_state["hyde_docs"]:
        with st.expander("HyDE: Generated Hypothetical Documents", expanded=False):
            for i, doc in enumerate(st.session_state["hyde_docs"], 1):
                st.markdown(f"**Excerpt {i}:** {doc}")

    chunks = st.session_state["retrieved_chunks"]
    if chunks:
        st.subheader("3. Retrieved Chunks")
        for i, chunk in enumerate(chunks, 1):
            title = f"Chunk {i} — {chunk.get('party', 'Unknown Party')} | {chunk.get('speaker', 'Unknown Speaker')} | Score: {chunk.get('score', 0):.4f}"
            with st.expander(title, expanded=(i == 1)):
                st.write(chunk.get("text", ""))
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"Party: **{chunk.get('party', '—')}**")
                meta_cols[1].caption(f"Speaker: **{chunk.get('speaker', '—')}**")
                meta_cols[2].caption(f"Date: **{chunk.get('date', '—')}**")


def render_prediction_section(evaluator: BiasPredictor, retriever: Optional[PoliticalRAGRetriever], model_id: str, model_region: str, mode: str, k: int):
    """Dispatches processing frames directly out to OpenRouter endpoints and evaluates variance against CHES tables."""
    st.subheader("4. Bias Prediction")
    input_text = st.session_state["input_text"]
    current_party = st.session_state["meta_party"]
    current_year = int(st.session_state["meta_year"])

    if st.button("Run Prediction", type="primary", disabled=not input_text.strip()):
        if not st.session_state["retrieved_chunks"]:
            chunks, docs = run_pipeline_retrieval(retriever, input_text, mode, k)
            st.session_state["retrieved_chunks"] = chunks
            st.session_state["hyde_docs"] = docs

        with st.spinner(f"Predicting political bias score with LLM {model_id}..."):
            prediction = evaluator.predict_bias(
                text=input_text,
                model_id=model_id,
                context_chunks=st.session_state["retrieved_chunks"] or None,
            )
            
            st.session_state["prediction"] = prediction
            st.session_state["active_model_id"] = model_id
            st.session_state["active_model_region"] = model_region
            st.session_state["active_retrieval_mode"] = mode
            
            lrecon, galtan, lrgen = evaluator._get_closest_ches_score(current_party, current_year)

            log_evaluation_run(
                input_text=input_text,
                llm_choice=model_id,
                llm_region=model_region,
                retrieval_mode=mode,
                k_chunks=k,
                hyde_docs=st.session_state["hyde_docs"],
                retrieved_chunks=st.session_state["retrieved_chunks"],
                meta_party=current_party,
                meta_speaker=st.session_state["meta_speaker"],
                meta_year=current_year,
                output_score=prediction.get("bias_score"),
                output_justification=prediction.get("justification"),
                ches_lrgen=lrgen,
                ches_lrecon=lrecon,
                ches_galtan=galtan,
            )
            st.toast("Pipeline execution logged successfully!")
        st.rerun()

    prediction = st.session_state["prediction"]
    if not prediction:
        return

    predicted_score = prediction.get("bias_score")
    p_col, region_col = st.columns([3, 2])
    
    if isinstance(predicted_score, (int, float)):
        actual_model_id = st.session_state.get("active_model_id", model_id)
        actual_region = st.session_state.get("active_model_region", model_region)
        friendly_name = next((k for k, v in LLM_MODELS.items() if v["id"] == actual_model_id), actual_model_id)
        
        p_col.metric(
            label=f"{friendly_name} Predicted Bias",
            value=f"{predicted_score:.1f} / 10",
            delta=get_bias_label(predicted_score),
            delta_color="off",
        )

        region_col.metric(
            label="Model Sovereign Origin",
            value=actual_region
        )
    else:
        p_col.error("Prediction failed — see justification below.")

    with st.expander("Show Justification", expanded=False):
        st.info(prediction.get("justification", "No justification returned."))

    # -------------------------------------------------------------------
    # CHES Ground Truth Calculations (Display Rendering Only)
    # -------------------------------------------------------------------
    st.subheader("5. CHES Ground Truth Comparison")
    lrecon, galtan, lrgen = evaluator._get_closest_ches_score(current_party, current_year)

    gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
    gt_col1.metric("CHES lrgen (input text)", f"{lrgen:.2f}" if lrgen is not None else "N/A")
    gt_col2.metric("CHES lrecon", f"{lrecon:.2f}" if lrecon is not None else "N/A")
    gt_col3.metric("CHES galtan", f"{galtan:.2f}" if galtan is not None else "N/A")

    if isinstance(predicted_score, (int, float)) and lrgen is not None:
        gt_col4.metric("Absolute Error (lrgen)", f"{abs(predicted_score - lrgen):.2f}")

    chunks = st.session_state["retrieved_chunks"]
    if chunks:
        st.markdown("**CHES scores of retrieved chunk parties:**")
        chunk_ches_rows = []
        for i, chunk in enumerate(chunks, 1):
            c_party = chunk.get("party", "")
            date_str = chunk.get("date", "")
            c_year = int(date_str[:4]) if date_str and len(date_str) >= 4 else current_year
            c_lrecon, c_galtan, c_lrgen = evaluator._get_closest_ches_score(c_party, c_year)
            chunk_ches_rows.append({
                "Chunk": i, "Party": c_party, "Speaker": chunk.get("speaker", ""), "Year": c_year,
                "lrgen": round(c_lrgen, 2) if c_lrgen is not None else None,
                "lrecon": round(c_lrecon, 2) if c_lrecon is not None else None,
                "galtan": round(c_galtan, 2) if c_galtan is not None else None,
                "Sim Score": chunk.get("score", 0),
            })
        st.dataframe(pd.DataFrame(chunk_ches_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Political RAG Bias Detector", layout="wide")
    st.title("Political Bias Detection")
    st.caption("RAG-augmented LLM pipeline validated against CHES expert scores.")

    init_session_state()
    
    evaluator = get_evaluator()
    df = load_test_dataset()

    model_id, model_region, retrieval_mode, k_chunks = render_sidebar()
    retriever = get_retriever(retrieval_mode)

    render_input_section(df)
    render_retrieval_section(retriever, retrieval_mode, k_chunks)
    render_prediction_section(evaluator, retriever, model_id, model_region, retrieval_mode, k_chunks)


if __name__ == "__main__":
    run_streamlit_app()