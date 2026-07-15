"""
Political RAG Bias Detector — Standalone Debug Script
======================================================
Runs the full RAG pipeline without Streamlit for easier debugging:
  1. Load a random sample from the evaluation dataset.
  2. Retrieve relevant chunks (Simple / TwoStage / HyDE).
  3. Run bias prediction via the chosen LLM.
  4. Compare against dataset ground truth labels.
  5. Log the run to disk.

Usage
-----
  python src/debug_rag_pipeline.py
  python src/debug_rag_pipeline.py --mode HyDE --embedding_model bge --k 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging.log_run import log_evaluation_run
from rag.evaluator import BiasPredictor
from rag.retrieval import PoliticalRAGRetriever

DATA_PATH = "src/datasets/political_bias_articles_dataset.csv"
QDRANT_URL = "http://localhost:6333"

RETRIEVAL_MODES = ["Simple", "TwoStage", "HyDE"]
EMBEDDING_MODELS = ["e5", "bge", "jina", "qwen3"]
LLM_MODELS = ["xai", "deepseek"]

_LLM_MAP = {
    "xai": "x-ai/grok-4.3",
    "deepseek": "deepseek/deepseek-v4-flash",
}


def get_bias_label(score: float) -> str:
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
        raise FileNotFoundError(f"Test dataset not found at `{DATA_PATH}`")


def get_evaluator() -> BiasPredictor:
    return BiasPredictor()


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
        print(f"[WARNING] Could not connect to Qdrant ({e}). RAG disabled.")
        return None


def chunks_to_context_dicts(points) -> List[Dict[str, Any]]:
    result = []
    for p in points:
        payload = p.payload or {}
        result.append({
            "text": payload.get("text", ""),
            "party": payload.get("party", ""),
            "country": payload.get("country", ""),
            "speaker": payload.get("speaker", ""),
            "date": payload.get("date", ""),
            "speech_id": payload.get("speech_id", ""),
            "score": round(float(p.score), 4),
        })
    return result


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return None


def run_pipeline(
    retrieval_mode: str = "simple",
    llm_choice: str = "xai",
    embedding_model: str = "e5",
    k_chunks: int = 3,
    sample_idx: Optional[int] = None,
) -> Dict[str, Any]:
    print("=" * 70)
    print("Political RAG Bias Detector — Debug Pipeline")
    print("=" * 70)

    print("\n[1/6] Loading dataset...")
    df = load_test_dataset()
    if df.empty:
        raise RuntimeError("Dataset is empty.")

    if sample_idx is not None:
        sample = df.iloc[sample_idx]
        print(f"   -> Using fixed row index: {sample_idx}")
    else:
        sample = df.sample(1).iloc[0]
        print(f"   -> Random row index: {sample.name}")

    input_text = str(sample.get("post_content", ""))
    meta_party = str(sample.get("party", ""))
    meta_speaker = str(sample.get("social_media_handle", ""))
    meta_source = str(sample.get("article_source", ""))
    gt_ideology = _safe_float(sample.get("final_label_ideology"))
    gt_economic = _safe_float(sample.get("final_label_economic"))
    gt_galtan = _safe_float(sample.get("final_label_galtan"))

    print(f"   Party  : {meta_party or '--'}")
    print(f"   Speaker: {meta_speaker or '--'}")
    print(f"   Source : {meta_source or '--'}")
    print(f"   Text   : {input_text[:200]}...")

    print("\n[2/6] Initialising evaluator...")
    evaluator = get_evaluator()

    print(f"[3/6] Initialising retriever (mode={retrieval_mode}, model={embedding_model})...")
    retriever = get_retriever(retrieval_mode, embedding_model)
    if retriever is None:
        raise RuntimeError("Retriever could not be initialised. Is Qdrant running?")

    print(f"\n[4/6] Running {retrieval_mode} retrieval (top_k={k_chunks})...")
    hyde_docs: List[str] = []

    if retrieval_mode.lower() == "hyde":
        strategy = retriever.retrieval_strategy
        if getattr(strategy, "hyde_llm", None) is None:
            print("   [WARNING] HyDE selected but LLM is unavailable.")
        else:
            hyde_docs = strategy._generate_hypothetical_docs(input_text, num_docs=3)
            print(f"   Generated {len(hyde_docs)} hypothetical document(s).")
            for i, doc in enumerate(hyde_docs, 1):
                print(f"   HyDE doc {i}: {doc[:200]}...")

    points = retriever.search(query=input_text, limit=k_chunks)
    retrieved_chunks = chunks_to_context_dicts(points)

    print(f"   Retrieved {len(retrieved_chunks)} chunk(s).")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n   --- Chunk {i} ---")
        print(f"   Party : {chunk.get('party', '--')}")
        print(f"   Speaker: {chunk.get('speaker', '--')}")
        print(f"   Score  : {chunk.get('score', 0):.4f}")
        print(f"   Text   : {chunk.get('text', '')[:300]}...")

    model_id = _LLM_MAP.get(llm_choice.lower(), llm_choice)
    print(f"\n[5/6] Running bias prediction with {model_id}...")
    prediction = evaluator.predict_bias(
        text=input_text,
        model_id=model_id,
        context_chunks=retrieved_chunks if retrieved_chunks else None,
    )

    predicted_score = prediction.get("bias_score")
    justification = prediction.get("justification", "No justification returned.")

    if isinstance(predicted_score, (int, float)):
        print(f"   Predicted Score: {predicted_score:.2f} / 7  ({get_bias_label(predicted_score)})")
    else:
        print(f"   Prediction failed — returned: {predicted_score}")
    print(f"   Justification: {justification[:300]}...")

    print("\n[6/6] Ground Truth Comparison")
    print(f"   Ideology: {gt_ideology:.2f}" if gt_ideology is not None else "   Ideology: N/A")
    print(f"   Economic: {gt_economic:.2f}" if gt_economic is not None else "   Economic: N/A")
    print(f"   GAL-TAN: {gt_galtan:.2f}" if gt_galtan is not None else "   GAL-TAN: N/A")

    if isinstance(predicted_score, (int, float)) and gt_ideology is not None:
        delta = abs(predicted_score - gt_ideology)
        print(f"   Absolute Error (ideology): {delta:.2f}")

    return {
        "input_text": input_text,
        "meta_party": meta_party,
        "meta_speaker": meta_speaker,
        "meta_source": meta_source,
        "retrieval_mode": retrieval_mode,
        "embedding_model": embedding_model,
        "llm_choice": model_id,
        "k_chunks": k_chunks,
        "hyde_docs": hyde_docs,
        "retrieved_chunks": retrieved_chunks,
        "prediction": prediction,
        "ground_truth": {
            "label_ideology": gt_ideology,
            "label_economic": gt_economic,
            "label_galtan": gt_galtan,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Debug the Political RAG Bias Detection pipeline end-to-end."
    )
    parser.add_argument(
        "--mode",
        choices=[m.lower() for m in RETRIEVAL_MODES],
        default="twostage",
        help="Retrieval mode (default: twostage)",
    )
    parser.add_argument(
        "--embedding_model",
        choices=[m.lower() for m in EMBEDDING_MODELS],
        default="e5",
        help="Embedding model for retrieval (default: e5)",
    )
    parser.add_argument(
        "--model",
        choices=[m.lower() for m in LLM_MODELS],
        default="xai",
        help="LLM provider (default: xai)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of chunks (default: 3)")
    parser.add_argument("--index", type=int, default=None, help="Specific row index.")
    args = parser.parse_args()

    mode = args.mode.capitalize() if args.mode != "twostage" else "TwoStage"
    result = run_pipeline(
        retrieval_mode=mode,
        llm_choice=args.model,
        embedding_model=args.embedding_model,
        k_chunks=args.k,
        sample_idx=args.index,
    )

    print("\n[DEBUG] Full prediction object:")
    print(json.dumps(result["prediction"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
