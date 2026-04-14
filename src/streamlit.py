import time

import pandas as pd

import streamlit as st

from .logging.log_run import log_evaluation_run
from .prediction.bias_prediction import BaselineEvaluator

LLM_MODELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "mistral": "Mistral",
}


def get_bias_label(score: float) -> str:
    """
    Maps a continuous political bias score (0.0 - 10.0) to a categorical label.
    Handles None types gracefully if the API or CHES lookup fails.
    """
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
def load_test_dataset():
    """Caches the dataset so it doesn't reload on every UI interaction."""
    data_path = "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Test dataset not found at `{data_path}`")
        return pd.DataFrame()


def run_streamlit_app():
    st.set_page_config(page_title="LLM Bias Evaluator", layout="wide")
    st.title("Political Bias Detection")

    evaluator = BaselineEvaluator()
    df = load_test_dataset()

    # --- Sidebar Configuration ---
    st.sidebar.header("Pipeline Configuration")
    llm_choice = st.sidebar.selectbox("Select LLM", ["Mistral", "OpenAI", "Gemini"])

    use_rag = st.sidebar.toggle(
        "Enable RAG (Retrieval-Augmented Generation)", value=False
    )

    # Conditional UI: Only show RAG parameters if RAG is enabled
    db_choice = None
    k_chunks = 0
    if use_rag:
        st.sidebar.subheader("RAG Parameters")
        db_choice = st.sidebar.selectbox(
            "Select Vector Database / Embedding",
            [
                "ChromaDB + MPNet-Multilingual",
                "ChromaDB + OpenAI-Small",
                "Qdrant + GESIS-German",
            ],
        )
        k_chunks = st.sidebar.slider(
            "Top-K Chunks to Retrieve", min_value=1, max_value=10, value=3
        )

    # --- Dataset Integration ---
    st.subheader("1. Data Input")
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🎲 Load Random Row from Dataset", disabled=df.empty):
            if not df.empty:
                sample = df.sample(1).iloc[0]
                st.session_state["current_text"] = str(
                    sample.get("full_text", "")
                ).replace("/comma", ",")
                st.session_state["current_party"] = str(sample.get("party", ""))

    input_text = st.text_area(
        "Text to Analyze", value=st.session_state.get("current_text", ""), height=150
    )

    # --- Metadata & Ground Truth ---
    st.subheader("2. Metadata & CHES Ground Truth")
    st.markdown(
        "Ensure inputs match CHES formatting (e.g., proper country abbreviations) to successfully map the ground truth."
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    party = m_col1.text_input("Party", value=st.session_state.get("current_party", ""))
    country = m_col2.text_input("Country")
    year = m_col3.number_input("Year", step=1)

    # --- Execution ---
    if st.button("Run Classification", type="primary"):
        if not input_text.strip():
            st.warning("Please provide text to analyze.")
            return

        with st.spinner(f"Analyzing with {llm_choice}..."):
            # Step 0: Retrieval (if enabled)
            context = None
            if use_rag:
                context = "Not implemented"  # mock_retrieve_context(input_text, db_choice, k_chunks)
                st.info(f"**Retrieved Context:**\n{context}")

            # 1. Get LLM Prediction
            prediction = evaluator.predict_bias(input_text, llm_choice)

            # 2. Get Ground Truth
            ground_truth = (
                None  # evaluator.get_ches_ground_truth(party, country, int(year))
            )

            # 3. Display Metrics
            st.markdown("---")
            st.markdown("### Results")
            res_col1, res_col2 = st.columns(2)

            predicted_score = prediction.get("bias_score")
            res_col1.metric(
                label=f"{llm_choice} Predicted Bias",
                value=(
                    f"{predicted_score:.1f} ({get_bias_label(predicted_score)})"
                    if isinstance(predicted_score, (int, float))
                    else "Error"
                ),
            )

            if ground_truth is not None:
                res_col2.metric(
                    label="CHES Ground Truth (lrgen)", value=f"{ground_truth:.1f}"
                )

                # Calculate delta if both exist
                if isinstance(predicted_score, (int, float)):
                    delta = abs(predicted_score - ground_truth)
                    st.caption(f"**Absolute Error:** {delta:.2f}")
            else:
                res_col2.metric(label="CHES Ground Truth", value="No Match Found")

            st.info(f"**Justification:** {prediction.get('justification')}")

            log_evaluation_run(
                input_text=input_text,
                llm_choice=llm_choice,
                use_rag=use_rag,
                db_choice=db_choice,
                k_chunks=k_chunks,
                retrieved_context=context,
                output_score=prediction.get("bias_score"),
                output_justification=prediction.get("justification"),
                ches_ground_truth=ground_truth,
            )
