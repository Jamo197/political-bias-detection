import datetime
import os
import sys
import uuid
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

RETRIEVAL_MODES = ["Simple", "TwoStage", "HyDE"]
EMBEDDING_MODELS = ["e5", "bge", "jina", "qwen3"]

LLM_MODELS = {
    "Grok 4.3": {"id": "x-ai/grok-4.3", "region": "Americas"},
    "DeepSeek V4 Flash": {"id": "deepseek/deepseek-v4-flash", "region": "China"},
}

DATA_PATH = "src/datasets/political_bias_articles_dataset.csv"
QDRANT_URL = "http://localhost:6333"

SESSION_STATE_DEFAULTS = {
    "input_text": "",
    "meta_party": "",
    "meta_speaker": "",
    "meta_source": "",
    "gt_ideology": None,
    "gt_economic": None,
    "gt_galtan": None,
    "retrieved_chunks": [],
    "hyde_docs": [],
    "prediction": None,
    "active_model_id": "",
    "active_model_region": "",
    "active_retrieval_mode": "",
    "run_id": None,
    "run_dir": None,
    "run_last_config": None,
}


def get_bias_label(score: Optional[float]) -> str:
    if score is None:
        return "Unknown"
    score = max(0.0, min(7.0, float(score)))
    if score < 1.5: return "Extreme Left"
    elif score < 3.0: return "Left"
    elif score < 3.8: return "Center-Left"
    elif score <= 4.2: return "Centrist / Moderate"
    elif score <= 5.0: return "Center-Right"
    elif score <= 6.0: return "Right"
    return "Extreme Right"


@st.cache_data
def load_test_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH)
        df["post_content"] = df["post_content"].astype(str)
        rename_dict = {
            "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
            "Bündnis 90 Die Grünen": "BÜNDNIS 90/DIE GRÜNEN",
            "Linke": "DIE LINKE",
            "Die Linke": "DIE LINKE",
        }
        df["party"] = df["party"].replace(rename_dict)
        return df
    except FileNotFoundError:
        st.error(f"Critical execution error: Test dataset not found at `{DATA_PATH}`")
        return pd.DataFrame()


@st.cache_resource
def get_evaluator() -> BiasPredictor:
    return BiasPredictor()


@st.cache_resource
def get_retriever(mode: str, embedding_model: str = "e5") -> Optional[PoliticalRAGRetriever]:
    _MODE_MAP = {"simple": "simple", "twostage": "two_stage", "hyde": "hyde"}
    internal_mode = _MODE_MAP.get(mode.lower().replace(" ", ""), mode.lower())
    try:
        return PoliticalRAGRetriever(
            qdrant_url=QDRANT_URL,
            model_key=embedding_model,
            retrieval_mode=internal_mode,
        )
    except Exception as e:
        st.warning(f"Could not connect to Qdrant ({e}). RAG disabled.")
        return None


def chunks_to_context_dicts(points: List[Any]) -> List[Dict[str, Any]]:
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
    if not retriever:
        return [], []
    chunks, docs = [], []
    try:
        if mode == "HyDE":
            strategy = retriever.retrieval_strategy
            if getattr(strategy, "hyde_llm", None) is None:
                st.warning("HyDE LLM not available. Reverting to basic vector search.")
            docs = strategy._generate_hypothetical_docs(text, num_docs=3)
        points = retriever.search(query=text, limit=k)
        chunks = chunks_to_context_dicts(points)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    return chunks, docs


def init_session_state():
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> Tuple[str, str, str, str, int]:
    with st.sidebar:
        st.header("Pipeline Configuration")
        llm_label = st.selectbox(
            "LLM Provider Target",
            list(LLM_MODELS.keys()),
            format_func=lambda x: f"{x} ({LLM_MODELS[x]['region']})",
        )
        selected_model_id = LLM_MODELS[llm_label]["id"]
        selected_model_region = LLM_MODELS[llm_label]["region"]

        st.divider()
        st.subheader("RAG Parameters")
        embedding_model = st.selectbox("Embedding Model", EMBEDDING_MODELS)
        retrieval_mode = st.selectbox("Retrieval Method", RETRIEVAL_MODES)
        k_chunks = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)

    return selected_model_id, selected_model_region, embedding_model, retrieval_mode, k_chunks


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return None


def render_input_section(df: pd.DataFrame):
    st.subheader("1. Text Input")
    col_load, col_reset, _ = st.columns([1, 1, 4])

    if col_load.button("Load Random Row", disabled=df.empty, use_container_width=True):
        sample = df.sample(1).iloc[0]
        st.session_state["input_text"] = str(sample.get("post_content", ""))
        st.session_state["meta_party"] = str(sample.get("party", ""))
        st.session_state["meta_speaker"] = str(sample.get("social_media_handle", ""))
        st.session_state["meta_source"] = str(sample.get("article_source", ""))
        st.session_state["gt_ideology"] = _safe_float(sample.get("final_label_ideology"))
        st.session_state["gt_economic"] = _safe_float(sample.get("final_label_economic"))
        st.session_state["gt_galtan"] = _safe_float(sample.get("final_label_galtan"))
        for key in ("retrieved_chunks", "hyde_docs", "prediction"):
            st.session_state[key] = SESSION_STATE_DEFAULTS[key]
        st.rerun()

    if col_reset.button("Start New", use_container_width=True):
        for key in SESSION_STATE_DEFAULTS:
            st.session_state[key] = SESSION_STATE_DEFAULTS[key]
        st.rerun()

    st.text_area("Text to Analyse (post_content)", height=160, key="input_text")

    st.subheader("1a. Metadata")
    st.caption("Pre-filled from dataset. Ground truth labels are used for evaluation.")

    m1, m2, m3 = st.columns(3)
    m1.text_input("Party", key="meta_party")
    m2.text_input("Speaker", key="meta_speaker")
    m3.text_input("Source", key="meta_source")


def render_retrieval_section(retriever: Optional[PoliticalRAGRetriever], mode: str, k: int):
    st.subheader("2. Retrieval")
    input_text = st.session_state["input_text"]

    if st.button("Retrieve Chunks", disabled=not input_text.strip() or retriever is None):
        chunks, docs = run_pipeline_retrieval(retriever, input_text, mode, k)
        st.session_state["retrieved_chunks"] = chunks
        st.session_state["hyde_docs"] = docs
        st.rerun()

    if mode == "HyDE" and st.session_state["hyde_docs"]:
        with st.expander("HyDE: Generated Hypothetical Documents", expanded=False):
            for i, doc in enumerate(st.session_state["hyde_docs"], 1):
                st.markdown(f"**Excerpt {i}:** {doc}")

    chunks = st.session_state["retrieved_chunks"]
    if chunks:
        st.subheader("3. Retrieved Chunks")
        for i, chunk in enumerate(chunks, 1):
            title = f"Chunk {i} — {chunk.get('party', '?')} | {chunk.get('speaker', '?')} | Score: {chunk.get('score', 0):.4f}"
            with st.expander(title, expanded=(i == 1)):
                st.write(chunk.get("text", ""))
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"Party: **{chunk.get('party', '—')}**")
                meta_cols[1].caption(f"Speaker: **{chunk.get('speaker', '—')}**")
                meta_cols[2].caption(f"Date: **{chunk.get('date', '—')}**")


def render_prediction_section(evaluator: BiasPredictor, retriever: Optional[PoliticalRAGRetriever], model_id: str, model_region: str, embedding_model: str, mode: str, k: int):
    st.subheader("4. Bias Prediction")
    text_index = st.session_state["index"] if "index" in st.session_state else ""
    input_text = st.session_state["input_text"]
    current_party = st.session_state["meta_party"]
    current_speaker = st.session_state["meta_speaker"]
    current_source = st.session_state["meta_source"]

    if st.button("Run Prediction", type="primary", disabled=not input_text.strip()):
        if not st.session_state["retrieved_chunks"]:
            chunks, docs = run_pipeline_retrieval(retriever, input_text, mode, k)
            st.session_state["retrieved_chunks"] = chunks
            st.session_state["hyde_docs"] = docs

        with st.spinner(f"Predicting with {model_id}..."):
            prediction = evaluator.predict_bias(
                text=input_text,
                model_id=model_id,
                context_chunks=st.session_state["retrieved_chunks"] or None,
            )

            st.session_state["prediction"] = prediction
            st.session_state["active_model_id"] = model_id
            st.session_state["active_model_region"] = model_region
            st.session_state["active_retrieval_mode"] = mode

            current_config = (model_id, mode, k)
            if (
                st.session_state["run_id"] is None
                or st.session_state["run_last_config"] != current_config
            ):
                new_run_id = uuid.uuid4().hex[:8]
                run_date = datetime.date.today().strftime("%Y-%m-%d")
                new_run_dir = f"logs/batch_runs/{run_date}_{new_run_id}"
                st.session_state["run_id"] = new_run_id
                st.session_state["run_dir"] = new_run_dir
                st.session_state["run_last_config"] = current_config

            active_run_id: str = st.session_state["run_id"]
            active_run_dir: str = st.session_state["run_dir"]

            os.makedirs(active_run_dir, exist_ok=True)

            log_evaluation_run(
                text_index=text_index,
                input_text=input_text,
                llm_choice=model_id,
                llm_region=model_region,
                retrieval_mode=mode,
                k_chunks=k,
                embedding_model=embedding_model,
                hybrid=False,
                is_rag=bool(st.session_state["retrieved_chunks"]),
                hyde_docs=st.session_state["hyde_docs"],
                retrieved_chunks=st.session_state["retrieved_chunks"],
                meta_party=current_party,
                meta_speaker=current_speaker,
                meta_source=current_source,
                output_score=prediction.get("bias_score"),
                output_justification=prediction.get("justification"),
                label_ideology=st.session_state["gt_ideology"],
                label_economic=st.session_state["gt_economic"],
                label_galtan=st.session_state["gt_galtan"],
                run_dir=active_run_dir,
                run_id=active_run_id,
                filename="streamlit_evaluation_logs.jsonl",
            )
            st.toast(f"Logged · run {active_run_id}")
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
            value=f"{predicted_score:.1f} / 7",
            delta=get_bias_label(predicted_score),
            delta_color="off",
        )
        region_col.metric(label="Model Origin", value=actual_region)
    else:
        p_col.error("Prediction failed — see justification below.")

    with st.expander("Show Justification", expanded=False):
        st.info(prediction.get("justification", "No justification returned."))

    st.subheader("5. Ground Truth Comparison")
    gt_ideology = st.session_state.get("gt_ideology")
    gt_economic = st.session_state.get("gt_economic")
    gt_galtan = st.session_state.get("gt_galtan")

    gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
    gt_col1.metric("Ideology", f"{gt_ideology:.2f}" if gt_ideology is not None else "N/A")
    gt_col2.metric("Economic", f"{gt_economic:.2f}" if gt_economic is not None else "N/A")
    gt_col3.metric("GAL-TAN", f"{gt_galtan:.2f}" if gt_galtan is not None else "N/A")

    if isinstance(predicted_score, (int, float)) and gt_ideology is not None:
        gt_col4.metric("Abs Error (ideology)", f"{abs(predicted_score - gt_ideology):.2f}")


def run_streamlit_app():
    st.set_page_config(page_title="Political RAG Bias Detector", layout="wide")
    st.title("Political Bias Detection")
    st.caption("RAG-augmented LLM pipeline validated against dataset ground truth labels.")

    init_session_state()
    evaluator = get_evaluator()
    df = load_test_dataset()

    model_id, model_region, embedding_model, retrieval_mode, k_chunks = render_sidebar()
    retriever = get_retriever(retrieval_mode, embedding_model)

    render_input_section(df)
    render_retrieval_section(retriever, retrieval_mode, k_chunks)
    render_prediction_section(evaluator, retriever, model_id, model_region, embedding_model, retrieval_mode, k_chunks)


if __name__ == "__main__":
    run_streamlit_app()
