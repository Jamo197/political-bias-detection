#!/usr/bin/env python3
"""Batch evaluation runner for the multi-model RAG pipeline.

Supports testing multiple embedding models (e5, bge, jina, qwen3) × multiple
retrieval strategies (simple, simple_hybrid, hyde, twostage) + a no-RAG
baseline. Ground truth labels are read directly from the dataset.

Query-side embeddings:
  - e5, jina  : loaded locally (requires GPU)
  - bge, qwen3: use OpenRouter API via --query_backend openrouter (CPU only)

Bias prediction LLM:
  - OpenRouter (default, requires internet + OPENROUTER_API_KEY)
  - Local vLLM via --llm_base_url (HPC without internet)

Usage examples
--------------
  # No-RAG baseline only (CPU, no Qdrant needed)
  python -m src.run_batch --no_rag --sample_size 50

  # e5 with all strategies (GPU, local embeddings)
  python -m src.run_batch --embedding_model e5 \
      --strategies simple,hyde,twostage --device cuda

  # bge via OpenRouter (CPU, no GPU needed)
  python -m src.run_batch --embedding_model bge \
      --strategies simple,hyde,twostage --query_backend openrouter

  # qwen3 via OpenRouter (CPU, no vLLM server needed)
  python -m src.run_batch --embedding_model qwen3 \
      --strategies simple,hyde,twostage --query_backend openrouter
"""

import argparse
import datetime
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name in ["src", "scripts", "app"]
    else Path(__file__).resolve().parent
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.retrieval import PoliticalRAGRetriever, InProcessHyDELLM, CROSS_ENCODER_MODEL
from rag.evaluator import BiasPredictor
from rag.ingest.config import get_model_config
from rag.ingest.embedders import build_embedder
from src.logging.log_run import log_evaluation_run

DATA_PATH = _ROOT / "src/datasets/political_bias_articles_dataset.csv"

LLM_MODELS = {
    "ministral": {"region": "Europe", "id": "mistralai/Ministral-3-8B-Instruct-2512"},
}

STRATEGY_MAP = {
    "simple": {"mode": "simple", "hybrid": False},
    "simple_dense": {"mode": "simple", "hybrid": False},
    "simple_hybrid": {"mode": "simple", "hybrid": True},
    "hyde": {"mode": "hyde", "hybrid": False},
    "twostage": {"mode": "twostage", "hybrid": False},
}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("BatchRunner")


def write_batch_logs_info(
    run_dir: str, run_id: str, run_start: datetime.datetime, args: argparse.Namespace
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    model_rows = "\n".join(
        f"| {key} | {val['id']} | {val['region']} |" for key, val in LLM_MODELS.items()
    )
    strategies = args.strategies if not args.no_rag else "no_rag (baseline)"
    content = f"""# Test Run Info

- **Run ID:** {run_id}
- **Start Time:** {run_start.strftime("%Y-%m-%d %H:%M:%S")}

## Parameters
- **Embedding Model:** {args.embedding_model}
- **Query Backend:** {args.query_backend}
- **Strategies:** {strategies}
- **K Chunks:** {args.k_chunks}
- **Random Seed:** {args.random_seed}
- **Sample Size:** {args.sample_size}
- **RAG Mode:** {"False (no-RAG baseline)" if args.no_rag else "True"}
- **LLM Base URL:** {args.llm_base_url or "OpenRouter (default)"}

## Models
| Key | Model ID | Region |
|-----|----------|--------|
{model_rows}
"""
    with open(os.path.join(run_dir, "logs_info.md"), "w", encoding="utf-8") as f:
        f.write(content)


def load_and_prepare_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"Execution terminated: Target input dataset missing at '{DATA_PATH}'")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    # REMOVE data if media label is 0.0 or 1.0 (these are not valid labels for evaluation)
    before = len(df)
    df = df[~df["media_label"].isin([0.0, 1.0])]
    after = len(df)
    if before != after:
        print(
            f"Filtered out {before - after} rows with media_label 0.0 or 1.0 ({after} remaining)"
        )
    df["post_content"] = df["post_content"].astype(str)
    rename_dict = {
        "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
        "Bündnis 90 Die Grünen": "BÜNDNIS 90/DIE GRÜNEN",
        "Linke": "DIE LINKE",
        "Die Linke": "DIE LINKE",
    }
    df["party"] = df["party"].replace(rename_dict)
    return df


def resolve_metadata(row) -> Tuple[str, str, str]:
    meta_party = str(row.get("party", ""))
    meta_speaker = str(row.get("social_media_handle", "Unknown"))
    meta_source = str(row.get("article_source", "Unknown"))
    return meta_party, meta_speaker, meta_source


# fmt: off
def resolve_ground_truth(
    row,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def _safe_float(val) -> Optional[float]:
        try:
            f = float(val)
            return f if not pd.isna(f) else None
        except (ValueError, TypeError):
            return None

    return (
        _safe_float(row.get("final_label_ideology")),
        _safe_float(row.get("final_label_economic")),
        _safe_float(row.get("final_label_galtan")),
    )
# fmt: on


def chunks_to_context_dicts(points) -> List[Dict[str, Any]]:
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
                "score": (
                    round(float(p.score), 4)
                    if hasattr(p, "score") and p.score is not None
                    else 0.0
                ),
            }
        )
    return result


def run_condition_rag(
    evaluation_batch: pd.DataFrame,
    evaluator: BiasPredictor,
    llm_key: str,
    llm_val: Dict[str, str],
    strategy_label: str,
    strategy_mode: str,
    hybrid_flag: bool,
    embedding_model: str,
    embedder,
    cross_encoder,
    hyde_llm,
    qdrant_url: str,
    k_chunks: int,
    run_dir: str,
    run_id: str,
) -> None:
    try:
        retriever = PoliticalRAGRetriever(
            qdrant_url=qdrant_url,
            model_key=embedding_model,
            retrieval_mode=strategy_mode,
            hybrid=hybrid_flag,
            cross_encoder=cross_encoder,
            hyde_llm=hyde_llm,
            embedder=embedder,
        )
    except Exception as e:
        print(
            f"  [SKIP] Could not initialize retriever for {embedding_model}/{strategy_label}: {e}"
        )
        return

    desc = f"{embedding_model}/{strategy_label}/{llm_key}"
    for idx, row in tqdm(
        evaluation_batch.iterrows(),
        total=len(evaluation_batch),
        desc=desc,
    ):
        text_idx = str(row.get("index", "")).strip()
        text_content = str(row.get("post_content", "")).strip()
        if not text_content:
            continue

        hyde_docs: List[str] = []
        context_chunks: List[Dict[str, Any]] = []

        try:
            if strategy_mode == "hyde":
                strategy = retriever.retrieval_strategy
                hyde_docs = strategy._generate_hypothetical_docs(
                    text_content, num_docs=3
                )

            points = retriever.search(query=text_content, limit=k_chunks)
            context_chunks = chunks_to_context_dicts(points)
        except Exception as e:
            logger.error(f"Retrieval exception on row {idx}: {e}")

        meta_party, meta_speaker, meta_source = resolve_metadata(row)
        label_ideology, label_economic, label_galtan = resolve_ground_truth(row)

        try:
            prediction = evaluator.predict_bias(
                text=text_content,
                model_id=llm_val["id"],
                context_chunks=context_chunks if context_chunks else None,
                is_rag_mode=True,
            )
        except Exception as e:
            logger.error(f"Prediction failed on row {idx}: {e}")
            continue

        log_evaluation_run(
            text_index=text_idx,
            input_text=text_content,
            llm_choice=llm_val["id"],
            llm_region=llm_val["region"],
            retrieval_mode=strategy_label,
            k_chunks=k_chunks,
            embedding_model=embedding_model,
            hybrid=hybrid_flag,
            is_rag=True,
            hyde_docs=hyde_docs,
            retrieved_chunks=context_chunks,
            meta_party=meta_party,
            meta_speaker=meta_speaker,
            meta_source=meta_source,
            output_score=prediction.get("bias_score"),
            output_justification=prediction.get("justification"),
            label_ideology=label_ideology,
            label_economic=label_economic,
            label_galtan=label_galtan,
            run_dir=run_dir,
            run_id=run_id,
            filename=f"{llm_key}_{strategy_label}_evaluation_logs.jsonl",
        )


def run_condition_norag(
    evaluation_batch: pd.DataFrame,
    evaluator: BiasPredictor,
    llm_key: str,
    llm_val: Dict[str, str],
    run_dir: str,
    run_id: str,
) -> None:
    desc = f"no_rag/{llm_key}"
    for idx, row in tqdm(
        evaluation_batch.iterrows(),
        total=len(evaluation_batch),
        desc=desc,
    ):
        text_idx = str(row.get("index", "")).strip()
        text_content = str(row.get("post_content", "")).strip()
        if not text_content:
            continue

        meta_party, meta_speaker, meta_source = resolve_metadata(row)
        label_ideology, label_economic, label_galtan = resolve_ground_truth(row)

        try:
            prediction = evaluator.predict_bias(
                text=text_content,
                model_id=llm_val["id"],
                context_chunks=None,
                is_rag_mode=False,
            )
        except Exception as e:
            logger.error(f"Prediction failed on row {idx}: {e}")
            continue

        log_evaluation_run(
            text_index=text_idx,
            input_text=text_content,
            llm_choice=llm_val["id"],
            llm_region=llm_val["region"],
            retrieval_mode="no_rag",
            k_chunks=0,
            embedding_model="none",
            hybrid=False,
            is_rag=False,
            hyde_docs=[],
            retrieved_chunks=[],
            meta_party=meta_party,
            meta_speaker=meta_speaker,
            meta_source=meta_source,
            output_score=prediction.get("bias_score"),
            output_justification=prediction.get("justification"),
            label_ideology=label_ideology,
            label_economic=label_economic,
            label_galtan=label_galtan,
            run_dir=run_dir,
            run_id=run_id,
            filename=f"{llm_key}_no_rag_evaluation_logs.jsonl",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-model RAG bias classification evaluator"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="e5",
        choices=["e5", "bge", "jina", "qwen3"],
        help="Embedding model to use for retrieval (default: e5)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="simple",
        help="Comma-separated retrieval strategies: simple, simple_dense, "
        "simple_hybrid, hyde, twostage (default: simple)",
    )
    parser.add_argument(
        "--no_rag",
        action="store_true",
        help="Run no-RAG baseline only (skips all retrieval)",
    )
    parser.add_argument("--k_chunks", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default="http://localhost:6333",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help="vLLM server URL for qwen3 embeddings when --query_backend=local",
    )
    parser.add_argument(
        "--query_backend",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="Backend for query-side embeddings: local (GPU) or openrouter (CPU, bge/qwen3 only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for local model loading: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--hyde_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="In-process HyDE LLM model (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=None,
        help="Base URL for bias prediction LLM (default: OpenRouter). "
        "Set to a vLLM server URL for HPC without internet.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Shared run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Shared log directory (auto-generated if not provided)",
    )

    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]
    run_start = datetime.datetime.now()
    run_date = run_start.strftime("%Y-%m-%d")
    run_dir = args.run_dir or f"logs/batch_runs/{run_date}_{run_id}"

    write_batch_logs_info(run_dir, run_id, run_start, args)
    print(f"Run ID: {run_id}  |  Logs directory: {run_dir}")

    df = load_and_prepare_data()
    if args.sample_size is not None:
        evaluation_batch = df.sample(
            n=min(args.sample_size, len(df)), random_state=args.random_seed
        )
        print(f"Loaded sample: {len(evaluation_batch)} items (seed={args.random_seed})")
    else:
        evaluation_batch = df
        print(f"Loaded full dataset: {len(evaluation_batch)} items")

    evaluator = BiasPredictor(
        base_url=args.llm_base_url or "https://openrouter.ai/api/v1"
    )

    if args.no_rag:
        print("Running no-RAG baseline...")
        for llm_key, llm_val in LLM_MODELS.items():
            run_condition_norag(
                evaluation_batch, evaluator, llm_key, llm_val, run_dir, run_id
            )
        print("No-RAG baseline complete.")
        sys.exit(0)

    cfg = get_model_config(args.embedding_model)
    print(f"Loading embedding model: {args.embedding_model} ({cfg.hf_model_id})...")
    print(f"Query backend: {args.query_backend}")

    try:
        embedder = build_embedder(
            cfg,
            device=args.device,
            vllm_base_url=args.vllm_base_url,
            query_backend=args.query_backend,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        print(f"  Falling back to local backend for {args.embedding_model}.")
        embedder = build_embedder(
            cfg, device=args.device, vllm_base_url=args.vllm_base_url
        )

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # FIXME: use prediction LLM for HyDE generation
    hyde_llm = None
    if "hyde" in strategies:
        print(f"Loading HyDE LLM: {args.hyde_model}...")
        hyde_llm = InProcessHyDELLM(
            model_id=args.hyde_model, device=args.device or "auto"
        )

    cross_encoder = None
    if "twostage" in strategies:
        print(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}...")
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    for strategy_name in strategies:
        spec = STRATEGY_MAP.get(strategy_name)
        if spec is None:
            print(f"Unknown strategy '{strategy_name}', skipping.")
            continue

        strategy_mode = spec["mode"]
        hybrid_flag = spec["hybrid"]

        if hybrid_flag:
            if not cfg.hybrid_sparse:
                print(
                    f"Strategy '{strategy_name}' requires hybrid support, "
                    f"but {args.embedding_model} doesn't provide sparse vectors. Skipping."
                )
                continue
            if args.query_backend == "openrouter":
                print(
                    f"Strategy '{strategy_name}' requires local sparse vectors, "
                    f"but --query_backend=openrouter. Skipping."
                )
                continue

        for llm_key, llm_val in LLM_MODELS.items():
            print(f"Running {args.embedding_model}/{strategy_name}/{llm_key}...")
            run_condition_rag(
                evaluation_batch,
                evaluator,
                llm_key,
                llm_val,
                strategy_name,
                strategy_mode,
                hybrid_flag,
                args.embedding_model,
                embedder,
                cross_encoder,
                hyde_llm,
                args.qdrant_url,
                args.k_chunks,
                run_dir,
                run_id,
            )

    print("Batch pipeline execution finalized cleanly.")
