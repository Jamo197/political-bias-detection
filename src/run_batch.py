#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Adjust local workspace resolution paths
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.retrieval import PoliticalRAGRetriever
from rag.evaluator import BiasPredictor
# Import your custom logging implementation
from src.logging.log_run import log_evaluation_run

# Execution Tuning Configurations
DATA_PATH = _ROOT / "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
MODEL_TARGET = "openai/gpt-4o-mini"  # OpenRouter Identifier
RETRIEVAL_MODE = "TwoStage"          # Simple / TwoStage / HyDE
K_CHUNKS = 3
SAMPLE_SIZE = 50

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("BatchRunner")

def run_batch_pipeline():
    print(f"Initializing Unified Evaluator & {RETRIEVAL_MODE} Retrieval Subsystems...")
    retriever = PoliticalRAGRetriever(retrieval_mode=RETRIEVAL_MODE)
    evaluator = BiasPredictor()

    if not os.path.exists(DATA_PATH):
        print(f"Execution terminated: Target input dataset file missing at '{DATA_PATH}'")
        return

    df = pd.read_csv(DATA_PATH)
    df["full_text"] = df["full_text"].astype(str).str.replace("/comma", ",", regex=False)
    
    # Extract random test batch slice
    evaluation_batch = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    print(f"Loaded execution batch: processing {len(evaluation_batch)} items...")

    for idx, row in tqdm(evaluation_batch.iterrows(), total=len(evaluation_batch), desc="Evaluating Batches"):
        text_content = str(row.get("full_text", "")).strip()
        if not text_content:
            continue
            
        hyde_docs = []
        context_chunks = []

        # Execute Context Extraction Routes
        try:
            # If using HyDE strategy, extract the generated documents explicitly for logs
            if RETRIEVAL_MODE.lower() == "hyde":
                strategy = retriever.retrieval_strategy
                hyde_docs = strategy._generate_hypothetical_docs(text_content, num_docs=3)

            points = retriever.search(query=text_content, limit=K_CHUNKS)
            for p in points:
                payload = p.payload or {}
                context_chunks.append({
                    "text": payload.get("text", ""),
                    "party": payload.get("party", ""),
                    "country": payload.get("country", ""),
                    "speaker": payload.get("speaker", ""),
                    "date": payload.get("date", ""),
                    "score": round(float(p.score), 4) if hasattr(p, 'score') and p.score else 0.0,
                })
        except Exception as e:
            logger.error(f"Retrieval fallback triggered: {e}")

        # Resolve Ground Truth Metadata & CHES variables
        meta_party = str(row.get("party", ""))
        meta_speaker = str(row.get("twitter_handle", row.get("author", "Unknown")))
        try:
            meta_year = int(str(row.get("year", row.get("date", "2021")))[:4])
        except:
            meta_year = 2021
            
        lrecon, galtan, lrgen = evaluator._get_closest_ches_score(meta_party, meta_year)

        # Route Payload out to OpenRouter
        prediction = evaluator.predict_bias(
            text=text_content,
            model_id=MODEL_TARGET,
            context_chunks=context_chunks if context_chunks else None
        )

        # Execute standard logging pass via your log_evaluation_run hook
        log_evaluation_run(
            input_text=text_content,
            llm_choice=MODEL_TARGET,
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
            filepath="batch_evaluation_logs.jsonl" # Segregated file to avoid blending with UI runs
        )

    print("Batch pipeline execution finalized cleanly.")

if __name__ == "__main__":
    run_batch_pipeline()