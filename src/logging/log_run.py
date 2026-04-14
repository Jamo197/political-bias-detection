import datetime
import json
import os


def log_evaluation_run(
    input_text,
    llm_choice,
    use_rag,
    db_choice,
    k_chunks,
    retrieved_context,
    output_score,
    output_justification,
    ches_ground_truth,
    filepath="evaluation_logs.jsonl",
):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": {
            "llm": llm_choice,
            "rag_enabled": use_rag,
            "vector_db": db_choice if use_rag else None,
            "k_chunks": k_chunks if use_rag else 0,
        },
        "inputs": {"text": input_text, "retrieved_context": retrieved_context},
        "output": {"score": output_score, "justification": output_justification},
        "ches_ground_truth": ches_ground_truth,
    }

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    filepath_full = os.path.join(logs_dir, filepath)

    with open(filepath_full, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
