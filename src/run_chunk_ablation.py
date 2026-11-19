#!/usr/bin/env python3
"""Chunk-ablation runs that reuse already-logged retrieval (no Qdrant needed).

Three modes:

``rank``
    Rank the judged model families by how many of their retrieved chunks the
    LLM judge rated R_top >= ``--min_rtop`` (default 4). Prints a table and
    writes ``chunk_ablation_rtop_ranking.csv`` into ``--out_root``. Needs no
    LLM server.

``random``
    Experiment 1. For every post, replace the retrieved context by ``--k_chunks``
    chunks drawn at random from a pool of chunks that were retrieved in an
    existing e5/simple log (the post's own retrieved chunks are excluded). The
    draw is seeded per post (``seed`` + ``text_index``), so every model sees
    exactly the same random chunks (paired comparison across models).

``rtop``
    Experiment 2. For the 100 judged posts of one model family
    (``RAG Analysis/LLM judge results/families/<family>``), re-predict using only
    the logged e5/simple chunks the judge rated R_top >= ``--min_rtop``
    (variable k, 1..5). ``--select low`` instead keeps only the chunks BELOW the
    threshold: a control with the same posts and a similar chunk count but
    topically irrelevant context. Posts without any chunk in the selected group
    are skipped (``--no_chunk_policy skip``) or predicted without context
    (``--no_chunk_policy norag``); either way they are counted in the summary.

Predictions use the same ``BiasPredictor`` (prompt, temperature 0, JSON schema)
as ``src/run_batch.py``; only the context differs. Ground truth, text and
metadata are copied from the existing party_label logs (not from the CSV), so
the labels match the main experiments. Output goes to
``<out_root>/<run_id>/<embedding>/<condition>/party_label_*.jsonl`` in the
normal log format, deliberately OUTSIDE ``logs/batch_runs`` so the main
evaluation matrix does not pick it up by accident.

Examples:
    python -m src.run_chunk_ablation --mode rank
    python -m src.run_chunk_ablation --mode rtop --family llama_8b --llm_base_url http://HOST:8000/v1
    python -m src.run_chunk_ablation --mode random --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --model_key llama-8B --llm_base_url http://HOST:8000/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.evaluator import BiasPredictor  # noqa: E402
from src.logging.log_run import log_evaluation_run  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ChunkAblation")

JUDGE_ROOT = _ROOT / "RAG Analysis" / "LLM judge results" / "families"
# e5/simple log of Llama-70B: all 775 posts, k=5 retrieved chunks, CHES labels.
DEFAULT_SOURCE_LOG = (
    "logs/batch_runs/2026-08-17_eval_matrix_20260817_123007/e5/simple/"
    "party_label_llama-70B_simple_evaluation_logs.jsonl"
)
DEFAULT_OUT_ROOT = "logs/chunk_ablation"

# family key -> short model key used in log file names (same style as the matrix runs)
FAMILY_MODEL_KEY = {
    "llama_3b": "llama-3B",
    "llama_8b": "llama-8B",
    "llama_70b": "llama-70B",
    "mistral_3b": "mistral-3B",
    "mistral_8b": "mistral-8B",
    "mistral_14b": "mistral-14B",
    "qwen_3b": "qwen-3B",
    "qwen_7b": "qwen-7B",
    "qwen_32b": "qwen-32B",
    "qwen_72b": "qwen-72B",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_judge_rtop(family: str) -> Dict[str, Dict[int, int]]:
    """{text_index: {chunk_index: R_top}} from the LLM judge output of a family."""
    path = JUDGE_ROOT / family / "llm_judge_annotations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Judge annotations not found: {path}")
    out: Dict[str, Dict[int, int]] = defaultdict(dict)
    for r in read_jsonl(path):
        rtop = (r.get("derived_labels") or {}).get("R_top")
        if rtop is None:
            continue
        out[str(r["text_index"])][int(r["chunk_index"])] = int(rtop)
    return out


def load_family_qualitative(family: str) -> Dict[str, Dict[str, Any]]:
    """The 100 judged posts of a family, keyed by text_index (has the logged chunks)."""
    summary = json.loads((JUDGE_ROOT / "summary.json").read_text(encoding="utf-8"))
    src = _ROOT / summary[family]["source_file"]
    return {str(r["text_index"]): r for r in read_jsonl(src)}


def clean_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": c.get("text", ""),
        "party": c.get("party", ""),
        "country": c.get("country", ""),
        "speaker": c.get("speaker", ""),
        "date": c.get("date", ""),
        "speech_id": c.get("speech_id", ""),
        "score": c.get("score", 0.0),
    }


def chunk_key(c: Dict[str, Any]) -> str:
    return f"{c.get('speech_id', '')}|{(c.get('text') or '').strip()[:120]}"


# --------------------------------------------------------------------------- #
# mode: rank
# --------------------------------------------------------------------------- #
def run_rank(args: argparse.Namespace) -> None:
    rows = []
    for family in FAMILY_MODEL_KEY:
        path = JUDGE_ROOT / family / "llm_judge_annotations.jsonl"
        if not path.exists():
            continue
        per_text: Counter = Counter()
        total = 0
        for r in read_jsonl(path):
            rtop = (r.get("derived_labels") or {}).get("R_top")
            if rtop is None:
                continue
            total += 1
            if rtop >= args.min_rtop:
                per_text[str(r["text_index"])] += 1
        n_hi = sum(per_text.values())
        rows.append(
            {
                "family": family,
                "chunks_judged": total,
                f"chunks_rtop>={args.min_rtop}": n_hi,
                "share": round(n_hi / total, 3) if total else 0.0,
                "texts_with_>=1": len(per_text),
                "texts_with_>=3": sum(v >= 3 for v in per_text.values()),
            }
        )
    key = f"chunks_rtop>={args.min_rtop}"
    rows.sort(key=lambda r: r[key], reverse=True)
    print(f"{'family':<13}{'judged':>8}{key:>16}{'share':>8}{'texts>=1':>10}{'texts>=3':>10}")
    for r in rows:
        print(
            f"{r['family']:<13}{r['chunks_judged']:>8}{r[key]:>16}{r['share']:>8}"
            f"{r['texts_with_>=1']:>10}{r['texts_with_>=3']:>10}"
        )
    out_dir = _ROOT / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "chunk_ablation_rtop_ranking.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_csv}")


# --------------------------------------------------------------------------- #
# job construction
# --------------------------------------------------------------------------- #
def build_jobs_random(args: argparse.Namespace) -> List[Dict[str, Any]]:
    source = read_jsonl(_ROOT / args.source_log)
    pool: Dict[str, Dict[str, Any]] = {}
    for rec in source:
        for c in rec["inputs"]["retrieved_chunks"]:
            if (c.get("text") or "").strip():
                pool.setdefault(chunk_key(c), clean_chunk(c))
    pool_keys = sorted(pool)
    print(f"Random pool: {len(pool_keys)} unique chunks from {len(source)} posts ({args.source_log})")

    jobs = []
    for rec in source:
        text_index = str(rec["input_metadata"]["text_index"])
        own = {chunk_key(c) for c in rec["inputs"]["retrieved_chunks"]}
        candidates = [k for k in pool_keys if k not in own]
        rng = random.Random(f"{args.seed}-{text_index}")
        picked = rng.sample(candidates, args.k_chunks)
        jobs.append(
            {
                "rec": rec,
                "chunks": [pool[k] for k in picked],
                "note": "random",
            }
        )
    return jobs


def build_jobs_rtop(args: argparse.Namespace) -> List[Dict[str, Any]]:
    judged = load_judge_rtop(args.family)
    posts = load_family_qualitative(args.family)
    jobs = []
    n_with = n_without = 0
    for text_index, q in posts.items():
        logged = q["rag"]["retrieved_chunks"]
        labels = judged.get(text_index, {})
        keep = []
        for idx, c in enumerate(logged):
            rtop = labels.get(idx)
            if rtop is None:
                continue
            is_high = rtop >= args.min_rtop
            if (args.select == "high") == is_high:
                keep.append((idx, c, rtop))
        n_with += bool(keep)
        n_without += not keep
        # Adapt the qualitative record to the party_label log record layout.
        rec = {
            "input_metadata": q["input_metadata"],
            "inputs": {"text": q["input_text"], "retrieved_chunks": logged},
            "ground_truth": {
                "label_ideology": q["ground_truth"]["label_ideology"],
                "label_economic": q["ground_truth"].get("label_economic"),
                "label_galtan": q["ground_truth"].get("label_galtan"),
            },
            "parameters": q["rag"]["parameters"],
        }
        jobs.append(
            {
                "rec": rec,
                "chunks": [clean_chunk(c) for _, c, _ in keep],
                "kept_idx": [i for i, _, _ in keep],
                "kept_rtop": [r for _, _, r in keep],
                "note": f"rtop_{args.select}",
            }
        )
    print(
        f"{args.family}: {len(posts)} judged posts | {n_with} with >=1 chunk "
        f"({args.select}, threshold {args.min_rtop}) | {n_without} without"
    )
    return jobs


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def run_jobs(
    jobs: List[Dict[str, Any]],
    args: argparse.Namespace,
    model_id: str,
    model_key: str,
    condition: str,
    embedding_label: str,
) -> None:
    out_dir = _ROOT / args.out_root / args.run_id
    out_file = out_dir / embedding_label / condition / (
        f"party_label_{model_key}_{condition}_evaluation_logs.jsonl"
    )

    done = set()
    if out_file.exists():
        for r in read_jsonl(out_file):
            if r["output"]["bias"] is not None:
                done.add(str(r["input_metadata"]["text_index"]))
        print(f"Resume: {len(done)} posts already done in {out_file}")

    skipped_no_chunks = 0
    todo = []
    for job in jobs:
        tid = str(job["rec"]["input_metadata"]["text_index"])
        if tid in done:
            continue
        if not job["chunks"] and args.no_chunk_policy == "skip":
            skipped_no_chunks += 1
            continue
        todo.append(job)
    print(
        f"{condition}: {len(todo)} posts to predict, {skipped_no_chunks} skipped "
        f"(no chunk in group), model={model_id}"
    )

    predictor = BiasPredictor(base_url=args.llm_base_url, api_key="EMPTY")
    lock = threading.Lock()

    def predict(job: Dict[str, Any]) -> Dict[str, Any]:
        rec = job["rec"]
        chunks = job["chunks"]
        pred = predictor.predict_bias(
            text=rec["inputs"]["text"].strip(),
            model_id=model_id,
            context_chunks=chunks if chunks else None,
            is_rag_mode=bool(chunks),
        )
        return {"job": job, "pred": pred}

    def write(result: Dict[str, Any]) -> None:
        job, pred = result["job"], result["pred"]
        rec = job["rec"]
        meta = rec["input_metadata"]
        gt = rec["ground_truth"]
        with lock:
            log_evaluation_run(
                text_index=str(meta["text_index"]),
                input_text=rec["inputs"]["text"],
                llm_choice=model_id,
                llm_region=args.llm_region,
                retrieval_mode=condition,
                k_chunks=len(job["chunks"]),
                embedding_model=embedding_label,
                hybrid=False,
                is_rag=bool(job["chunks"]),
                hyde_docs=[],
                retrieved_chunks=job["chunks"],
                meta_party=meta.get("party", ""),
                meta_speaker=meta.get("speaker", ""),
                meta_source=meta.get("source", ""),
                output_score=pred.get("bias_score"),
                output_justification=pred.get("justification"),
                label_ideology=gt.get("label_ideology"),
                label_economic=gt.get("label_economic"),
                label_galtan=gt.get("label_galtan"),
                run_dir=str(out_dir),
                run_id=args.run_id,
                filename=out_file.name,
            )

    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(predict, j) for j in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=condition):
            res = fut.result()
            if res["pred"].get("bias_score") is None:
                n_fail += 1
            write(res)

    summary = {
        "run_id": args.run_id,
        "condition": condition,
        "model_id": model_id,
        "model_key": model_key,
        "posts_total": len(jobs),
        "posts_predicted_now": len(todo),
        "posts_resumed": len(done),
        "posts_skipped_no_chunks": skipped_no_chunks,
        "predictions_failed": n_fail,
        "chunks_per_post": dict(Counter(len(j["chunks"]) for j in jobs)),
        "args": {k: v for k, v in vars(args).items()},
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Done. Failed predictions: {n_fail}. Logs: {out_file}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["rank", "random", "rtop"])
    p.add_argument("--family", help="rtop mode: judged family, e.g. llama_8b (see --mode rank)")
    p.add_argument("--model_id", help="Served model id. rtop: defaults to the id used in the original log")
    p.add_argument("--model_key", help="Short key for file names, e.g. llama-8B (rtop: derived from --family)")
    p.add_argument("--llm_base_url", help="vLLM OpenAI-compatible URL, e.g. http://node01:8000/v1")
    p.add_argument("--llm_region", default="unknown")
    p.add_argument("--k_chunks", type=int, default=5, help="random mode: chunks per post")
    p.add_argument("--seed", type=int, default=33, help="random mode: base seed")
    p.add_argument("--source_log", default=DEFAULT_SOURCE_LOG, help="random mode: e5/simple log (pool + posts)")
    p.add_argument("--min_rtop", type=int, default=4)
    p.add_argument("--select", choices=["high", "low"], default="high",
                   help="rtop mode: keep chunks with R_top >= min_rtop (high) or < min_rtop (low, control)")
    p.add_argument("--no_chunk_policy", choices=["skip", "norag"], default="skip",
                   help="rtop mode: posts without a chunk in the group are skipped or predicted without context")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--run_id", default=None)
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--print_model_id", action="store_true",
                   help="rtop mode: print the model id of --family and exit (used by the sbatch script)")
    args = p.parse_args()

    if args.mode == "rank":
        run_rank(args)
        return

    if args.mode == "rtop":
        if not args.family or args.family not in FAMILY_MODEL_KEY:
            p.error(f"--family required, one of: {', '.join(FAMILY_MODEL_KEY)}")
        q = load_family_qualitative(args.family)
        logged_model = next(iter(q.values()))["rag"]["parameters"]["llm"]
        args.model_id = args.model_id or logged_model
        args.model_key = args.model_key or FAMILY_MODEL_KEY[args.family]
        if args.print_model_id:
            print(args.model_id)
            return

    if not args.model_id or not args.model_key:
        p.error("--model_id and --model_key are required in random mode")
    if not args.llm_base_url:
        p.error("--llm_base_url is required")

    if args.mode == "random":
        condition = f"random{args.k_chunks}"
        run_id = args.run_id or f"chunk_ablation_random_{args.model_key}"
        jobs = build_jobs_random(args)
        embedding_label = "e5pool"
    else:
        condition = f"rtop{args.min_rtop}plus" if args.select == "high" else f"rtoplt{args.min_rtop}"
        run_id = args.run_id or f"chunk_ablation_rtop_{args.family}"
        jobs = build_jobs_rtop(args)
        embedding_label = "e5"
    args.run_id = run_id

    run_jobs(jobs, args, args.model_id, args.model_key, condition, embedding_label)


if __name__ == "__main__":
    main()
