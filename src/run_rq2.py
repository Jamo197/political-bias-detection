#!/usr/bin/env python3
"""RQ2 cross-cultural evaluation runner (conditions C0-C5).

Every tweet of a country is scored under the conditions of the RQ2 test concept;
only the retrieval source and the country information change:

    C0  no RAG                         no country in prompt (= RQ1 no-RAG prompt)
    C1  no RAG                         country + CHES scale definition
    C2  in-country retrieval           country_code = target
    C3  wrong-country retrieval        country_code = donor (de for all, at for de)
    C4  pooled retrieval               no filter
    C5  in-country, English            target, translated corpus + translated tweet

C0/C1 run once per country and generator; C2-C5 once per retrieval strategy.
All retrieval conditions use ONE collection per embedding model holding every
country (``chunks_<model><collection_suffix>``, default ``_parlamint``).

Input: ``src/datasets/EU_tweets_rq2.csv`` (built by
``src/Cross Cultural Analysis/label_tweets.py``). Labels are CHES (1-7):
lrgen -> label_ideology, lrecon -> label_economic, galtan -> label_galtan.

Logs: ``<run_dir>/<emb>/<strategy|no_rag>/<llm>_<country>_<cond>_<strategy>.jsonl``
with an extra ``rq2`` block (condition, countries, CHES id, family, party_cue).
Re-running skips tweets already logged in that file (resumable).

Example::

    python -m src.run_rq2 --country at --conditions C0,C1,C2,C3,C4 \\
        --embedding_model bge --strategies simple_hybrid,twostage \\
        --llm qwen-32B --llm_base_url http://<vllm-host>/v1 --qdrant_url $QDRANT_URL
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.evaluator import BiasPredictor
from rag.ingest.config import COLLECTION_SUFFIX, get_model_config
from rag.ingest.embedders import build_embedder
from rag.retrieval import CROSS_ENCODER_MODEL, OpenAIHyDELLM, PoliticalRAGRetriever
from src.logging.log_run import log_evaluation_run
from src.run_batch import STRATEGY_MAP, chunks_to_context_dicts

DATA_PATH = _ROOT / "src/datasets/EU_tweets_rq2.csv"

COUNTRY_NAMES = {
    "at": "Austria", "be": "Belgium", "de": "Germany", "dk": "Denmark",
    "fr": "France", "gb": "United Kingdom", "gr": "Greece", "it": "Italy",
    "lv": "Latvia", "nl": "Netherlands", "pl": "Poland", "pt": "Portugal",
    "se": "Sweden", "si": "Slovenia",
}  # fmt: skip

# Fixed in advance (test concept): Germany is the donor for every country,
# Austria the donor for Germany.
DONOR = {"de": "at"}
DEFAULT_DONOR = "de"

# source: which country the retrieval filter selects ("target" / "donor" /
# "pooled"); None = no retrieval.
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "C0": {"source": None, "state_country": False, "english": False},
    "C1": {"source": None, "state_country": True, "english": False},
    "C2": {"source": "target", "state_country": True, "english": False},
    "C3": {"source": "donor", "state_country": True, "english": False},
    "C4": {"source": "pooled", "state_country": True, "english": False},
    "C5": {"source": "target", "state_country": True, "english": True},
}

# Generators from RQ1 (vLLM ids). RQ2 uses one per model family.
GENERATORS = {
    "qwen-3B": {"id": "Qwen/Qwen2.5-3B-Instruct", "region": "China"},
    "qwen-7B": {"id": "Qwen/Qwen2.5-7B-Instruct", "region": "China"},
    "qwen-32B": {"id": "Qwen/Qwen2.5-32B-Instruct", "region": "China"},
    "qwen-72B": {"id": "RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic", "region": "China"},
    "llama-3B": {"id": "meta-llama/Llama-3.2-3B-Instruct", "region": "Americas"},
    "llama-8B": {"id": "meta-llama/Llama-3.1-8B-Instruct", "region": "Americas"},
    "llama-70B": {"id": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8", "region": "Americas"},
    "ministral-3B": {"id": "mistralai/Ministral-3-3B-Instruct-2512", "region": "Europe"},
    "ministral-8B": {"id": "mistralai/Ministral-3-8B-Instruct-2512", "region": "Europe"},
    "ministral-14B": {"id": "mistralai/Ministral-3-14B-Instruct-2512", "region": "Europe"},
}


def retrieval_country(target: str, source: Optional[str]) -> Optional[str]:
    """Country code the Qdrant filter selects; None = pooled / no retrieval."""
    if source == "target":
        return target
    if source == "donor":
        return DONOR.get(target, DEFAULT_DONOR)
    return None


def base_collection(model_key: str) -> str:
    """Collection name without any COLLECTION_SUFFIX taken from the environment."""
    name = get_model_config(model_key).collection
    return name[: -len(COLLECTION_SUFFIX)] if COLLECTION_SUFFIX else name


def load_tweets(path: Path, country: str, sample_size: Optional[int], seed: Optional[int]) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"RQ2 dataset missing at '{path}'. Run label_tweets.py first.")
    df = pd.read_csv(path, dtype={"text_id": str, "tweet_id": str})
    df = df[df["country_code"] == country]
    if df.empty:
        sys.exit(f"No tweets for country '{country}' in {path}.")
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed)
    return df


def logged_ids(path: Path) -> set:
    """text_index values already in a log file (for resuming)."""
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                ids.add(json.loads(line)["input_metadata"]["text_index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _float(value) -> Optional[float]:
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def run_condition(
    tweets: pd.DataFrame,
    country: str,
    cond: str,
    evaluator: BiasPredictor,
    llm_key: str,
    run_dir: str,
    run_id: str,
    k_chunks: int,
    retriever: Optional[PoliticalRAGRetriever] = None,
    strategy_label: str = "no_rag",
    embedding_model: str = "none",
    hybrid: bool = False,
    collection: Optional[str] = None,
) -> None:
    spec = CONDITIONS[cond]
    is_rag = spec["source"] is not None
    text_col = "tweet_text_en" if spec["english"] else "tweet_text"
    if text_col not in tweets.columns:
        print(f"  [SKIP] {cond}: column '{text_col}' missing (translate tweets first).")
        return
    llm = GENERATORS[llm_key]
    country_name = COUNTRY_NAMES.get(country, country.upper())
    source_code = retrieval_country(country, spec["source"])
    filename = f"{llm_key}_{country}_{cond}_{strategy_label}.jsonl"
    log_path = Path(run_dir) / (embedding_model if is_rag else "none") / (
        strategy_label if is_rag else "no_rag"
    ) / filename
    done = logged_ids(log_path)

    desc = f"{country}/{cond}/{embedding_model}/{strategy_label}/{llm_key}"
    for _, row in tqdm(tweets.iterrows(), total=len(tweets), desc=desc):
        text_id = str(row["text_id"])
        raw = row.get(text_col)
        text = "" if pd.isna(raw) else str(raw).strip()  # untranslated (NaN) -> skip
        if not text or text_id in done:
            continue

        hyde_docs: List[str] = []
        context: List[Dict[str, Any]] = []
        if is_rag and retriever is not None:
            try:
                if strategy_label.startswith("hyde"):
                    hyde_docs = retriever.retrieval_strategy._generate_hypothetical_docs(
                        text, num_docs=3
                    )
                context = chunks_to_context_dicts(retriever.search(query=text, limit=k_chunks))
            except Exception as e:  # noqa: BLE001 - log and score without context
                print(f"  Retrieval failed for {text_id}: {e}")

        prediction = evaluator.predict_bias(
            text=text,
            model_id=llm["id"],
            context_chunks=context or None,
            is_rag_mode=is_rag,
            country=country_name if spec["state_country"] else None,
        )

        log_evaluation_run(
            text_index=text_id,
            input_text=text,
            llm_choice=llm["id"],
            llm_region=llm["region"],
            retrieval_mode=strategy_label if is_rag else "no_rag",
            k_chunks=k_chunks if is_rag else 0,
            embedding_model=embedding_model if is_rag else "none",
            hybrid=hybrid if is_rag else False,
            is_rag=is_rag,
            hyde_docs=hyde_docs,
            retrieved_chunks=context,
            meta_party=str(row.get("canonical_party", "")),
            meta_speaker=str(row.get("name", "")),
            meta_source="twitter",
            output_score=prediction.get("bias_score"),
            output_justification=prediction.get("justification"),
            label_ideology=_float(row.get("label_lrgen")),
            label_economic=_float(row.get("label_lrecon")),
            label_galtan=_float(row.get("label_galtan")),
            run_dir=run_dir,
            run_id=run_id,
            filename=filename,
            extra_metadata={
                "condition": cond,
                "target_country": country,
                "retrieval_country": source_code if is_rag else None,
                "retrieval_source": spec["source"],
                "country_in_prompt": spec["state_country"],
                "english": spec["english"],
                "collection": collection if is_rag else None,
                "ches_party_id": int(row["ches_party_id"]),
                "canonical_party": str(row.get("canonical_party", "")),
                "party_family": str(row.get("ches_family", "")),
                "ches_wave": int(row["ches_wave"]) if pd.notna(row.get("ches_wave")) else None,
                "party_cue": bool(row.get("party_cue", False)),
            },
        )


def write_info(run_dir: str, run_id: str, args: argparse.Namespace) -> None:
    os.makedirs(run_dir, exist_ok=True)
    lines = [
        "# RQ2 Run Info",
        "",
        f"- **Run ID:** {run_id}",
        f"- **Start:** {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- **Countries:** {args.country}",
        f"- **Conditions:** {args.conditions}",
        f"- **Embedding model:** {args.embedding_model}",
        f"- **Strategies:** {args.strategies}",
        f"- **Generator:** {args.llm} ({GENERATORS[args.llm]['id']})",
        f"- **K chunks:** {args.k_chunks}",
        f"- **Collections:** {base_collection(args.embedding_model)}{args.collection_suffix}"
        f" (C5: {args.en_collection_suffix})",
        f"- **Sample size / seed:** {args.sample_size} / {args.random_seed}",
        f"- **LLM base URL:** {args.llm_base_url}",
    ]
    with open(os.path.join(run_dir, f"rq2_info_{args.llm}_{args.embedding_model}.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--country", required=True, help="Comma-separated codes, e.g. at or at,de")
    parser.add_argument("--conditions", default="C0,C1,C2,C3,C4")
    parser.add_argument("--embedding_model", default="bge", choices=["e5", "bge", "jina", "qwen3"])
    parser.add_argument("--strategies", default="simple", help=f"Comma list of {sorted(STRATEGY_MAP)}")
    parser.add_argument("--llm", default="qwen-32B", choices=sorted(GENERATORS))
    parser.add_argument("--llm_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--k_chunks", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=None, help="Per country (smoke tests).")
    parser.add_argument("--random_seed", type=int, default=33)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--collection_suffix", default="_parlamint")
    parser.add_argument("--en_collection_suffix", default="_parlamint_en")
    parser.add_argument("--vllm_embed_url", default=None)
    parser.add_argument("--query_backend", default="local", choices=["local", "openrouter"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args(argv)

    countries = [c.strip() for c in args.country.split(",") if c.strip()]
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        sys.exit(f"Unknown condition(s) {unknown}; choose from {sorted(CONDITIONS)}")
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGY_MAP:
            sys.exit(f"Unknown strategy '{s}'; choose from {sorted(STRATEGY_MAP)}")

    run_id = args.run_id or f"rq2_{uuid.uuid4().hex[:8]}"
    run_dir = args.run_dir or f"logs/rq2_runs/{datetime.date.today()}_{run_id}"
    write_info(run_dir, run_id, args)
    print(f"Run ID: {run_id} | Logs: {run_dir}")

    evaluator = BiasPredictor(base_url=args.llm_base_url)
    llm_id = GENERATORS[args.llm]["id"]

    rag_conditions = [c for c in conditions if CONDITIONS[c]["source"] is not None]
    cfg = get_model_config(args.embedding_model)
    embedder = cross_encoder = None
    if rag_conditions:
        embedder = build_embedder(
            cfg,
            device=args.device,
            vllm_embed_url=args.vllm_embed_url,
            query_backend=args.query_backend,
        )
        if any(STRATEGY_MAP[s]["mode"] == "twostage" for s in strategies):
            from sentence_transformers import CrossEncoder

            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    for country in countries:
        tweets = load_tweets(args.data, country, args.sample_size, args.random_seed)
        print(f"\n=== {country}: {len(tweets)} tweets ===")

        for cond in conditions:
            spec = CONDITIONS[cond]
            if spec["source"] is None:
                run_condition(tweets, country, cond, evaluator, args.llm, run_dir, run_id, args.k_chunks)
                continue

            suffix = args.en_collection_suffix if spec["english"] else args.collection_suffix
            collection = base_collection(args.embedding_model) + suffix
            source_code = retrieval_country(country, spec["source"])
            # HyDE writes speeches "from" the country it retrieves from.
            hyde_country = COUNTRY_NAMES.get(source_code or country, country)
            for strategy in strategies:
                mode, hybrid = STRATEGY_MAP[strategy]["mode"], STRATEGY_MAP[strategy]["hybrid"]
                if hybrid and not cfg.hybrid_sparse:
                    print(f"  [SKIP] {strategy}: {args.embedding_model} has no sparse vectors.")
                    continue
                hyde_llm = (
                    OpenAIHyDELLM(base_url=args.llm_base_url, model_id=llm_id)
                    if mode == "hyde"
                    else None
                )
                retriever = PoliticalRAGRetriever(
                    qdrant_url=args.qdrant_url,
                    model_key=args.embedding_model,
                    retrieval_mode=mode,
                    country_context=hyde_country,
                    hybrid=hybrid,
                    cross_encoder=cross_encoder,
                    hyde_llm=hyde_llm,
                    embedder=embedder,
                    country_code=source_code,
                    collection=collection,
                )
                run_condition(
                    tweets, country, cond, evaluator, args.llm, run_dir, run_id,
                    args.k_chunks, retriever, strategy, args.embedding_model, hybrid, collection,
                )  # fmt: skip

    print("RQ2 run finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
