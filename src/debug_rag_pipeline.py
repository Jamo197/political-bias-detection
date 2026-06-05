"""
Political RAG Bias Detector — Standalone Debug Script
======================================================
Runs the full RAG pipeline without Streamlit for easier debugging:
  1. Load a random sample from the evaluation dataset.
  2. Retrieve relevant chunks (Simple / TwoStage / HyDE).
  3. Run bias prediction via the chosen LLM.
  4. Compare against CHES ground truth.
  5. Log the run to disk.

Usage
-----
  python src/debug_rag_pipeline.py
  python src/debug_rag_pipeline.py --mode HyDE --model Mistral --k 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Make project root importable when launched directly from repo root or src/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging.log_run import log_evaluation_run
from rag.evaluator import BiasPredictor
from rag.retrieval import PoliticalRAGRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = (
    "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset"
    "_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bundestag_speeches_chunks"

RETRIEVAL_MODES = ["Simple", "TwoStage", "HyDE"]
LLM_MODELS = ["Mistral", "OpenAI", "Gemini"]

# ---------------------------------------------------------------------------
# Helpers (mirroring streamlit.py)
# ---------------------------------------------------------------------------

def get_bias_label(score: float) -> str:
    """Maps a continuous 0-10 bias score to a human-readable category."""
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


def load_test_dataset() -> pd.DataFrame:
    """Loads the evaluation dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        df["full_text"] = df["full_text"].astype(str).str.replace("/comma", ",", regex=False)
        rename_dict = {
            "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
            "Linke": "DIE LINKE"
        }
        df["party"] = df["party"].replace(rename_dict)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Test dataset not found at `{DATA_PATH}`")


def get_evaluator() -> BiasPredictor:
    """Singleton evaluator (holds API clients + CHES DB)."""
    return BiasPredictor()


def get_retriever(mode: str) -> Optional[PoliticalRAGRetriever]:
    """
    Constructs a PoliticalRAGRetriever for the chosen mode.
    Returns None if Qdrant is unavailable.
    """
    _MODE_MAP = {
        "simple": "simple",
        "twostage": "two_stage",
        "hyde": "hyde",
    }
    internal_mode = _MODE_MAP.get(mode.lower().replace(" ", ""), mode.lower())

    try:
        return PoliticalRAGRetriever(
            qdrant_url=QDRANT_URL,
            chunk_collection=COLLECTION_NAME,
            retrieval_mode=internal_mode,
        )
    except Exception as e:
        print(f"[WARNING] Could not connect to Qdrant ({e}). RAG disabled.")
        return None


def chunks_to_context_dicts(points) -> List[Dict[str, Any]]:
    """Converts raw Qdrant PointStruct list to plain dicts for the LLM context."""
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
                "score": round(float(p.score), 4),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    retrieval_mode: str = "simple",
    llm_choice: str = "Mistral",
    k_chunks: int = 3,
    sample_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes one full prediction run:
      - loads data
      - retrieves chunks
      - predicts bias
      - fetches CHES ground truth
      - logs result

    Parameters
    ----------
    retrieval_mode : Retrieval strategy (Simple / TwoStage / HyDE).
    llm_choice     : LLM provider (Mistral / OpenAI / Gemini).
    k_chunks       : Number of chunks to retrieve.
    sample_idx     : If provided, use this DataFrame row index instead of random.

    Returns
    -------
    Dict with all intermediate and final results.
    """

    print("=" * 70)
    print("Political RAG Bias Detector — Debug Pipeline")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading dataset...")
    df = load_test_dataset()
    if df.empty:
        raise RuntimeError("Dataset is empty.")

    if sample_idx is not None:
        sample = df.iloc[sample_idx]
        print(f"   → Using fixed row index: {sample_idx}")
    else:
        sample = df.sample(1).iloc[0]
        print(f"   → Random row index: {sample.name}")

    input_text = str(sample.get("full_text", ""))
    meta_party = str(sample.get("party", ""))
    meta_speaker = str(sample.get("twitter_handle", sample.get("author", "")))
    raw_year = sample.get("year", sample.get("date", "2021"))
    try:
        meta_year = int(str(raw_year)[:4])
    except (ValueError, TypeError):
        meta_year = 2021

    print(f"   Party  : {meta_party or '—'}")
    print(f"   Speaker: {meta_speaker or '—'}")
    print(f"   Year   : {meta_year}")
    print(f"   Text   : {input_text[:200]}...")

    # -----------------------------------------------------------------------
    # 2. Initialise evaluator + retriever
    # -----------------------------------------------------------------------
    print("\n[2/6] Initialising evaluator...")
    evaluator = get_evaluator()

    print(f"[3/6] Initialising retriever (mode={retrieval_mode})...")
    retriever = get_retriever(retrieval_mode)
    if retriever is None:
        raise RuntimeError("Retriever could not be initialised. Is Qdrant running?")

    # -----------------------------------------------------------------------
    # 3. Retrieval
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Running {retrieval_mode} retrieval (top_k={k_chunks})...")
    hyde_docs: List[str] = []

    if retrieval_mode.lower() == "hyde":
        strategy = retriever.retrieval_strategy
        if getattr(strategy, "llm", None) is None:
            print(
                "   [WARNING] HyDE selected but Ollama appears unavailable. "
                "Falling back to plain vector search without hypothetical documents."
            )
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
        print(f"   Party : {chunk.get('party', '—')}")
        print(f"   Speaker: {chunk.get('speaker', '—')}")
        print(f"   Date   : {chunk.get('date', '—')}")
        print(f"   Score  : {chunk.get('score', 0):.4f}")
        print(f"   Text   : {chunk.get('text', '')[:300]}...")

    # -----------------------------------------------------------------------
    # 4. Prediction
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Running bias prediction with {llm_choice}...")
    prediction = evaluator.predict_bias(
        text=input_text,
        model_provider=llm_choice,
        context_chunks=retrieved_chunks if retrieved_chunks else None,
    )

    predicted_score = prediction.get("bias_score")
    justification = prediction.get("justification", "No justification returned.")

    if isinstance(predicted_score, (int, float)):
        print(f"   Predicted Score: {predicted_score:.2f} / 10  ({get_bias_label(predicted_score)})")
    else:
        print(f"   Prediction failed — returned: {predicted_score}")
    print(f"   Justification: {justification[:300]}...")

    # -----------------------------------------------------------------------
    # 5. CHES Ground Truth
    # -----------------------------------------------------------------------
    print("\n[6/6] CHES Ground Truth Comparison")
    lrecon, galtan, lrgen = evaluator._get_closest_ches_score(meta_party, meta_year)

    print(f"   CHES lrgen (input text) : {lrgen:.2f}" if lrgen is not None else "   CHES lrgen (input text) : N/A")
    print(f"   CHES lrecon             : {lrecon:.2f}" if lrecon is not None else "   CHES lrecon             : N/A")
    print(f"   CHES galtan             : {galtan:.2f}" if galtan is not None else "   CHES galtan             : N/A")

    if isinstance(predicted_score, (int, float)) and lrgen is not None:
        delta = abs(predicted_score - lrgen)
        print(f"   Absolute Error (lrgen)  : {delta:.2f}")

    # CHES scores for retrieved chunks
    if retrieved_chunks:
        print("\n   CHES scores of retrieved chunk parties:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            c_party = chunk.get("party", "")
            date_str = chunk.get("date", "")
            c_year = int(date_str[:4]) if date_str and len(date_str) >= 4 else meta_year
            c_lrecon, c_galtan, c_lrgen = evaluator._get_closest_ches_score(c_party, c_year)
            print(
                f"      Chunk {i} | Party: {c_party or '—':15} | "
                f"Year: {c_year} | lrgen: {c_lrgen if c_lrgen is not None else 'N/A':>6} | "
                f"lrecon: {c_lrecon if c_lrecon is not None else 'N/A':>6} | "
                f"galtan: {c_galtan if c_galtan is not None else 'N/A':>6} | "
                f"Sim: {chunk.get('score', 0):.4f}"
            )

    # -----------------------------------------------------------------------
    # 6. Logging
    # -----------------------------------------------------------------------
    # log_evaluation_run(
    #     input_text=input_text,
    #     llm_choice=llm_choice,
    #     retrieval_mode=retrieval_mode,
    #     k_chunks=k_chunks,
    #     hyde_docs=hyde_docs,
    #     retrieved_chunks=retrieved_chunks,
    #     meta_party=meta_party,
    #     meta_speaker=meta_speaker,
    #     meta_year=meta_year,
    #     output_score=predicted_score,
    #     output_justification=justification,
    #     ches_lrgen=lrgen,
    #     ches_lrecon=lrecon,
    #     ches_galtan=galtan,
    # )
    # print("\n   Run logged to `logs/evaluation_logs.jsonl`.")
    # print("=" * 70)

    return {
        "input_text": input_text,
        "meta_party": meta_party,
        "meta_speaker": meta_speaker,
        "meta_year": meta_year,
        "retrieval_mode": retrieval_mode,
        "llm_choice": llm_choice,
        "k_chunks": k_chunks,
        "hyde_docs": hyde_docs,
        "retrieved_chunks": retrieved_chunks,
        "prediction": prediction,
        "ches": {
            "lrgen": lrgen,
            "lrecon": lrecon,
            "galtan": galtan,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug the Political RAG Bias Detection pipeline end-to-end."
    )
    parser.add_argument(
        "--mode",
        choices=[m.lower() for m in RETRIEVAL_MODES],
        default="simple",
        help="Retrieval mode (default: simple)",
    )
    parser.add_argument(
        "--model",
        choices=[m.lower() for m in LLM_MODELS],
        default="mistral",
        help="LLM provider (default: mistral)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Use a specific DataFrame row index instead of random sampling.",
    )
    args = parser.parse_args()

    # Normalise back to title-case for downstream consistency
    mode = args.mode.capitalize() if args.mode != "twostage" else "TwoStage"
    model = args.model.capitalize()

    result = run_pipeline(
        retrieval_mode=mode,
        llm_choice=model,
        k_chunks=args.k,
        sample_idx=args.index,
    )

    # Also pretty-print the full prediction dict for inspection
    print("\n[DEBUG] Full prediction object:")
    print(json.dumps(result["prediction"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
    # evaluator = get_evaluator()
    # data = evaluator._get_closest_ches_score("Linke", 2021)
    # print(data)
    # print("dine")