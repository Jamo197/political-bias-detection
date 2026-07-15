import datetime
import json
import os
from typing import Any, Dict, List, Optional


def log_evaluation_run(
    input_text: str,
    llm_choice: str,
    llm_region: str,
    retrieval_mode: str,
    k_chunks: int,
    hyde_docs: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    meta_party: str,
    meta_speaker: str,
    meta_source: str,
    output_score: Optional[float],
    output_justification: Optional[str],
    label_ideology: Optional[float],
    label_economic: Optional[float],
    label_galtan: Optional[float],
    run_dir: str,
    run_id: str,
    embedding_model: str = "none",
    hybrid: bool = False,
    is_rag: bool = True,
    filename: str = "evaluation_logs.jsonl",
):
    """Appends a structured JSONL log entry for a single prediction run.

    Parameters
    ----------
    input_text        : The raw text that was analysed.
    llm_choice        : LLM provider string slug path.
    llm_region        : Sovereign origin region of the model deployment target.
    retrieval_mode    : RAG strategy used (simple / twostage / hyde / no_rag).
    k_chunks          : Number of chunks retrieved (top-K).
    embedding_model   : Embedding model key (e5 / bge / jina / qwen3 / none).
    hybrid            : Whether bge hybrid (dense+sparse+RRF) was used.
    is_rag            : Whether RAG retrieval was performed at all.
    hyde_docs         : List of HyDE-generated hypothetical documents.
    retrieved_chunks  : List of retrieved chunk dicts with text + metadata.
    meta_party        : Party label of the input text.
    meta_speaker      : Social media handle of the input text author.
    meta_source       : Article source / media outlet.
    output_score      : Predicted bias score (0-7) or None on error.
    output_justification : LLM justification string or None on error.
    label_ideology    : Ground truth ideology label (1.2-6.0).
    label_economic    : Ground truth economic label (1.2-6.0).
    label_galtan      : Ground truth GAL-TAN label (1.2-5.9).
    run_dir           : Pre-computed base log directory for this run.
    run_id            : Short unique identifier for this run.
    filename          : JSONL filename within the condition subdirectory.
    """
    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": {
            "llm": llm_choice,
            "llm_region": llm_region,
            "embedding_model": embedding_model,
            "retrieval_mode": retrieval_mode,
            "hybrid": hybrid,
            "is_rag": is_rag,
            "k_chunks": k_chunks,
        },
        "input_metadata": {
            "party": meta_party,
            "speaker": meta_speaker,
            "source": meta_source,
        },
        "inputs": {
            "text": input_text,
            "hyde_docs": hyde_docs,
            "retrieved_chunks": retrieved_chunks,
        },
        "output": {
            "bias": output_score,
            "justification": output_justification,
        },
        "ground_truth": {
            "label_ideology": label_ideology,
            "label_economic": label_economic,
            "label_galtan": label_galtan,
        },
    }

    condition = "no_rag" if not is_rag else retrieval_mode
    logs_dir = os.path.join(run_dir, embedding_model, condition)
    os.makedirs(logs_dir, exist_ok=True)
    filepath_full = os.path.join(logs_dir, filename)

    if not os.path.exists(filepath_full):
        with open(filepath_full, "w", encoding="utf-8"):
            pass

    with open(filepath_full, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
