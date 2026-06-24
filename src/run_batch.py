#!/usr/bin/env python3
import datetime
import logging
import os
import sys
import uuid
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# Adjust local workspace resolution paths to match root configuration
# Streamlit uses parents[1] assuming it runs inside a subdirectory (e.g., src/ or app/)
# Adjust this path depth if your batch script is placed elsewhere.
_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name in ["src", "scripts", "app"]
    else Path(__file__).resolve().parent
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.retrieval import PoliticalRAGRetriever
from rag.evaluator import BiasPredictor
from src.logging.log_run import log_evaluation_run

# ---------------------------------------------------------------------------
# Execution Tuning Configurations & Parameter Mappings
# ---------------------------------------------------------------------------
DATA_PATH = (
    _ROOT
    / "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bundestag_speeches_chunks"

# "mistralai/mistral-small-2603": "Europe",
# "openai/gpt-5.4-mini": "Americas",
# "anthropic/claude-sonnet-4.6": "Americas",
# "x-ai/grok-4.3": "Americas",
# "google/gemini-3.5-flash": "Americas",
# "meta-llama/llama-4-maverick": "Americas",
# "deepseek/deepseek-v4-flash": "China",
# "qwen/qwen3.7-plus": "China"
LLM_MODELS = {
    # "mistral": {"region": "Europe", "id": "mistralai/mistral-small-2603"},
    # "openai": {"region": "Americas", "id": "openai/gpt-5.4-mini"},
    "xai": {"region": "Americas", "id": "x-ai/grok-4.3"},
    "deepseek": {"region": "China", "id": "deepseek/deepseek-v4-flash"},
}


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("BatchRunner")


def write_batch_logs_info(
    run_dir: str, run_id: str, run_start: datetime.datetime
) -> None:
    """Writes a human-readable logs_info.md into the run directory."""
    os.makedirs(run_dir, exist_ok=True)
    model_rows = "\n".join(
        f"| {key} | {val['id']} | {val['region']} |" for key, val in LLM_MODELS.items()
    )
    content = f"""# Test Run Info

- **Run ID:** {run_id}
- **Start Time:** {run_start.strftime("%Y-%m-%d %H:%M:%S")}

## Parameters
- **Retrieval Mode:** {RETRIEVAL_MODE}
- **K Chunks:** {K_CHUNKS}
- **Random Seed:** {RANDOM_SEED}
- **Sample Size:** {SAMPLE_SIZE}

## Models
| Key | Model ID | Region |
|-----|----------|--------|
{model_rows}
"""
    with open(os.path.join(run_dir, "logs_info.md"), "w", encoding="utf-8") as f:
        f.write(content)


def run_batch_pipeline(
    model_key: str = "mistral",
    is_rag_mode: bool = True,
    run_dir: str = "",
    run_id: str = "",
):
    print(f"Initializing Unified Evaluator & {RETRIEVAL_MODE} Retrieval Subsystems...")

    # 1. Properly instantiate retriever with URL and Collection configurations
    _MODE_MAP = {"simple": "simple", "twostage": "two_stage", "hyde": "hyde"}
    internal_mode = _MODE_MAP.get(
        RETRIEVAL_MODE.lower().replace(" ", ""), RETRIEVAL_MODE.lower()
    )

    try:
        retriever = PoliticalRAGRetriever(
            qdrant_url=QDRANT_URL,
            chunk_collection=COLLECTION_NAME,
            retrieval_mode=internal_mode,  # pyright: ignore[reportArgumentType]
        )
    except Exception as e:
        print(f"Critical connection failure to Qdrant cluster: {e}")
        return

    evaluator = BiasPredictor()

    if not os.path.exists(DATA_PATH):
        print(
            f"Execution terminated: Target input dataset file missing at '{DATA_PATH}'"
        )
        return

    # 2. Load and sanitize primary evaluation dataset
    df = pd.read_csv(DATA_PATH)
    df["full_text"] = (
        df["full_text"].astype(str).str.replace("/comma", ",", regex=False)
    )

    # Party mapping synchronization for CHES matching parity
    rename_dict = {"B90Grune": "BÜNDNIS 90/DIE GRÜNEN", "Linke": "DIE LINKE"}
    df["party"] = df["party"].replace(rename_dict)

    # Extract random test batch slice
    evaluation_batch = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
    print(f"Loaded execution batch: processing {len(evaluation_batch)} items...")

    # 3. Processing Loop
    for idx, row in tqdm(
        evaluation_batch.iterrows(),
        total=len(evaluation_batch),
        desc="Evaluating Batches",
    ):
        text_content = str(row.get("full_text", "")).strip()
        if not text_content:
            continue

        hyde_docs = []
        context_chunks = []

        if is_rag_mode:
            # Execute Context Extraction Routes
            try:
                if RETRIEVAL_MODE == "HyDE":
                    strategy = retriever.retrieval_strategy
                    hyde_docs = strategy._generate_hypothetical_docs(  # type: ignore
                        text_content, num_docs=3
                    )

                points = retriever.search(query=text_content, limit=K_CHUNKS)

                # Map structural Qdrant PointStruct into dictionary payloads (matching app logic)
                for p in points:
                    payload = p.payload or {}
                    context_chunks.append(
                        {
                            "text": payload.get("text", ""),
                            "party": payload.get("party", ""),
                            "country": payload.get("country", ""),
                            "speaker": payload.get("speaker", ""),
                            "date": payload.get("date", ""),
                            "speech_id": payload.get("speech_id", ""),
                            "score": round(float(p.score), 4) if hasattr(p, "score") and p.score is not None else 0.0,  # type: ignore
                        }
                    )
            except Exception as e:
                logger.error(f"Retrieval pipeline exception on row index {idx}: {e}")

        # Resolve Ground Truth Metadata & CHES variables
        meta_party = str(row.get("party", ""))
        meta_speaker = str(row.get("twitter_handle", row.get("author", "Unknown")))
        try:
            meta_year = int(str(row.get("year", row.get("date", "2021")))[:4])
        except ValueError, TypeError:
            meta_year = 2021

        lrecon, galtan, lrgen = evaluator._get_closest_ches_score(meta_party, meta_year)

        try:
            prediction = evaluator.predict_bias(
                text=text_content,
                model_id=MODEL_TARGET,
                context_chunks=context_chunks if context_chunks else None,
                is_rag_mode=is_rag_mode,
            )
        except Exception as e:
            logger.error(
                f"OpenRouter prediction dispatch failed on row index {idx}: {e}"
            )
            continue

        log_evaluation_run(
            input_text=text_content,
            llm_choice=MODEL_TARGET,
            llm_region=LLM_REGION,
            retrieval_mode=RETRIEVAL_MODE,
            k_chunks=K_CHUNKS,
            hyde_docs=hyde_docs,
            retrieved_chunks=context_chunks,
            meta_party=meta_party,
            meta_speaker=meta_speaker,
            meta_year=meta_year,
            output_score=prediction.get("bias_score"),
            output_justification=prediction.get("justification"),
            ches_lrgen=lrgen,
            ches_lrecon=lrecon,
            ches_galtan=galtan,
            run_dir=run_dir,
            run_id=run_id,
            filename=f"{model_key}_{internal_mode}_evaluation_logs.jsonl",
        )

    print("Batch pipeline execution finalized cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Bias Classification Evaluator")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="TwoStage",
        choices=["Simple", "TwoStage", "HyDE"],
    )
    parser.add_argument("--k_chunks", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=33)

    args = parser.parse_args()

    RETRIEVAL_MODE = args.retrieval_mode
    K_CHUNKS = args.k_chunks
    SAMPLE_SIZE = args.sample_size
    RANDOM_SEED = args.random_seed

    _run_id = uuid.uuid4().hex[:8]
    _run_start = datetime.datetime.now()
    _run_date = _run_start.strftime("%Y-%m-%d")
    _run_dir = f"logs/batch_runs/{_run_date}_{_run_id}"

    write_batch_logs_info(_run_dir, _run_id, _run_start)
    print(f"Run ID: {_run_id}  |  Logs directory: {_run_dir}")

    for key, value in LLM_MODELS.items():
        print(
            f"Running batch evaluation with model from {key} (Region: {value['region']})"
        )
        MODEL_TARGET = value["id"]
        LLM_REGION = value["region"]
        run_batch_pipeline(key, False, run_dir=_run_dir, run_id=_run_id)
    # run_batch_pipeline(run_dir=_run_dir, run_id=_run_id)
