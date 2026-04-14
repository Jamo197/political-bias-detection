import time

import streamlit as st

from .logging.log_run import log_evaluation_run

LLM_MODELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "mistral": "Mistral",
}


# --- Mock Backend Classes (Replace with your actual logic) ---
def mock_retrieve_context(text, db_choice, top_k):
    # Simulate database retrieval delay
    time.sleep(0.5)
    return f"Retrieved context from {db_choice} for: {text[:20]}..."


def mock_llm_generation(text, context, llm_choice):
    # Simulate API call delay
    time.sleep(1)
    if context:
        return f"[{llm_choice} w/ RAG] Bias Score: Center-Right. Justification based on context..."
    return f"[{llm_choice} Base] Bias Score: Left-Leaning. Justification based on internal weights..."


# TODO: able to upload files like .csv
def run_streamlit():
    # --- Streamlit UI Configuration ---
    st.set_page_config(page_title="RAG Bias Evaluator", layout="wide")
    st.title("Political Bias Classification: Ablation Platform")

    # --- Sidebar: Parameter Selection ---
    st.sidebar.header("Pipeline Configuration")

    llm_choice = st.sidebar.selectbox(
        "Select Generator LLM",
        ["GPT-4o", "GPT-3.5-Turbo", "Llama-3-70b-Instruct", "Mixtral-8x7b"],
    )

    use_rag = st.sidebar.toggle(
        "Enable RAG (Retrieval-Augmented Generation)", value=True
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

    # --- Main Workspace: Input & Execution ---
    input_text = st.text_area(
        "Input Political Text (Speech or Tweet)",
        height=200,
        placeholder="Paste parliamentary speech or social media text here...",
    )

    if st.button("Classify Bias", type="primary"):
        if not input_text.strip():
            st.warning("Please enter text to classify.")
        else:
            with st.spinner("Executing pipeline..."):

                # Step 1: Retrieval (if enabled)
                context = None
                if use_rag:
                    context = mock_retrieve_context(input_text, db_choice, k_chunks)
                    st.info(f"**Retrieved Context:**\n{context}")

                # Step 2: Generation
                result = mock_llm_generation(input_text, context, llm_choice)

                # Step 3: Display Results
                st.success("Classification Complete")
                st.markdown("### Output")
                st.write(result)

                log_evaluation_run(
                    input_text=input_text,
                    llm_choice=llm_choice,
                    use_rag=use_rag,
                    db_choice=db_choice,
                    k_chunks=k_chunks,
                    retrieved_context=context,
                    output=result,
                )
