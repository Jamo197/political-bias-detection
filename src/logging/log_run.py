import datetime
import json
import os
from typing import Any, Dict, List, Optional


def log_evaluation_run(
    input_text: str,
    llm_choice: str,
    retrieval_mode: str,
    k_chunks: int,
    hyde_docs: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    meta_party: str,
    meta_speaker: str,
    meta_year: int,
    output_score: Optional[float],
    output_justification: Optional[str],
    ches_lrgen: Optional[float],
    ches_lrecon: Optional[float],
    ches_galtan: Optional[float],
    filepath: str = "evaluation_logs.jsonl",
):
    """
    Appends a structured JSONL log entry for a single prediction run.

    Parameters
    ----------
    input_text        : The raw text that was analysed.
    llm_choice        : LLM provider used (Mistral / OpenAI / Gemini).
    retrieval_mode    : RAG strategy used (Simple / TwoStage / HyDE).
    k_chunks          : Number of chunks retrieved (top-K).
    hyde_docs         : List of HyDE-generated hypothetical documents (empty if not HyDE).
    retrieved_chunks  : List of retrieved chunk dicts with text + metadata.
    meta_party        : Party label of the input text (from dataset or DB).
    meta_speaker      : Speaker name of the input text.
    meta_year         : Year of the input text.
    output_score      : Predicted bias score (0–10) or None on error.
    output_justification : LLM justification string or None on error.
    ches_lrgen        : CHES ground truth lrgen score for the input party/year.
    ches_lrecon       : CHES ground truth lrecon score for the input party/year.
    ches_galtan       : CHES ground truth galtan score for the input party/year.
    filepath          : Filename within the logs/ directory.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": {
            "llm": llm_choice,
            "retrieval_mode": retrieval_mode,
            "k_chunks": k_chunks,
        },
        "input_metadata": {
            "party": meta_party,
            "speaker": meta_speaker,
            "year": meta_year,
        },
        "inputs": {
            "text": input_text,
            "hyde_docs": hyde_docs,
            "retrieved_chunks": retrieved_chunks,
        },
        "output": {
            "score": output_score,
            "justification": output_justification,
        },
        "ches_ground_truth": {
            "lrgen": ches_lrgen,
            "lrecon": ches_lrecon,
            "galtan": ches_galtan,
        },
    }

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    filepath_full = os.path.join(logs_dir, filepath)

    if not os.path.exists(filepath_full):
        with open(filepath_full, "w", encoding="utf-8"):
            pass

    with open(filepath_full, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
