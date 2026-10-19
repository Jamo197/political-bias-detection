"""Dump the exact prompts BiasPredictor sends to the LLM (No-RAG vs. RAG).

No LLM is called: BiasPredictor.client.chat.completions.create is replaced by a
recorder that stores the request and returns a stub answer. The RAG variant runs
the real retrieval (Qdrant + embedder), so the chunks are the ones a prediction
run would actually see.

Usage:
    python -m scripts.dump_example_prompts --row_index 0
    python -m scripts.dump_example_prompts --embedding_model e5 --strategy simple --k_chunks 5 --random_seed 33

Writes to <out_dir>/ (default results/example_prompts/<timestamp>/):
    norag_prompt.txt, rag_prompt.txt        human-readable system + user message
    norag_request.json, rag_request.json    full request payload (messages, params)
    retrieved_chunks.json                   chunks with scores
"""

import argparse
import datetime
import faulthandler
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# If the job hangs, dump every thread's stack to stderr every 2 minutes.
faulthandler.dump_traceback_later(120, repeat=True)
_T0 = time.time()


def step(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluator import BiasPredictor  # noqa: E402

STRATEGY_MODES = {"simple": "simple", "twostage": "two_stage"}
STUB_ANSWER = '{"bias_score": 4.0, "justification": "stub"}'


def make_recorder(predictor: BiasPredictor):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=STUB_ANSWER)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    predictor.client.chat.completions.create = fake_create
    return captured


def capture(predictor, captured, **predict_kwargs) -> dict:
    captured.clear()
    predictor.predict_bias(**predict_kwargs)
    return dict(captured)


def render(request: dict, header: str) -> str:
    sys_msg, user_msg = (m["content"] for m in request["messages"])
    params = {k: v for k, v in request.items() if k != "messages"}
    return (
        f"{header}\n"
        f"{'=' * 80}\n"
        f"[request parameters]\n{json.dumps(params, indent=2, ensure_ascii=False)}\n\n"
        f"{'=' * 80}\n[system message]\n{sys_msg}\n\n"
        f"{'=' * 80}\n[user message]\n{user_msg}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", default="e5", choices=["e5", "bge", "jina", "qwen3"])
    parser.add_argument("--strategy", default="simple", choices=list(STRATEGY_MODES))
    parser.add_argument("--k_chunks", type=int, default=5)
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--row_index", type=int, default=None, help="Fixed row of the prepared dataset")
    parser.add_argument("--random_seed", type=int, default=33, help="Used when --row_index is not set")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--from_log",
        default=None,
        help="Batch-run .jsonl log of a RAG condition. Takes the input text AND the retrieved "
        "chunks from one logged row, so no dataset, embedder or Qdrant is needed.",
    )
    parser.add_argument("--log_row", type=int, default=0, help="Row of --from_log to use")
    parser.add_argument("--skip_rag", action="store_true", help="Only dump the No-RAG prompt")
    args = parser.parse_args()

    out_dir = Path(
        args.out_dir
        or PROJECT_ROOT / "results/example_prompts" / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logged_chunks = None
    if args.from_log:
        with open(args.from_log, encoding="utf-8") as f:
            entry = json.loads(f.readlines()[args.log_row])
        text = entry["inputs"]["text"].strip()
        logged_chunks = entry["inputs"]["retrieved_chunks"]
        print(f"Input row (from log): text_index={entry['input_metadata'].get('text_index')}\n{text}\n")
    else:
        from src.run_batch import chunks_to_context_dicts, load_and_prepare_data

        step("loading dataset")
        df = load_and_prepare_data()
        row = df.iloc[args.row_index] if args.row_index is not None else df.sample(1, random_state=args.random_seed).iloc[0]
        text = str(row["post_content"]).strip()
        print(f"Input row: index={row.get('index')} party={row.get('party')}\n{text}\n")

    predictor = BiasPredictor(base_url="http://localhost:8000/v1", api_key="EMPTY")
    captured = make_recorder(predictor)

    norag = capture(predictor, captured, text=text, model_id=args.model_id, context_chunks=None, is_rag_mode=False)
    (out_dir / "norag_request.json").write_text(json.dumps(norag, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "norag_prompt.txt").write_text(render(norag, "No-RAG baseline prompt"), encoding="utf-8")
    print(f"Wrote No-RAG prompt to {out_dir}/norag_prompt.txt")

    if args.skip_rag:
        return

    if logged_chunks is not None:
        chunks = logged_chunks[: args.k_chunks]
        header = f"RAG prompt (chunks taken from log {args.from_log}, row {args.log_row})"
    else:
        chunks = retrieve_chunks(args, text, chunks_to_context_dicts)
        header = f"RAG prompt ({args.embedding_model} / {args.strategy} / k={args.k_chunks})"
    (out_dir / "retrieved_chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    rag = capture(predictor, captured, text=text, model_id=args.model_id, context_chunks=chunks, is_rag_mode=True)
    (out_dir / "rag_request.json").write_text(json.dumps(rag, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rag_prompt.txt").write_text(render(rag, header), encoding="utf-8")
    print(f"Wrote RAG prompt ({len(chunks)} chunks) to {out_dir}/rag_prompt.txt")


def retrieve_chunks(args, text, chunks_to_context_dicts):
    from rag.ingest.config import get_model_config
    from rag.ingest.embedders import build_embedder
    from rag.retrieval import CROSS_ENCODER_MODEL, PoliticalRAGRetriever

    step("loading embedder / retriever")
    cross_encoder = None
    if args.strategy == "twostage":
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    embedder = build_embedder(get_model_config(args.embedding_model), device=args.device)
    retriever = PoliticalRAGRetriever(
        qdrant_url=args.qdrant_url,
        model_key=args.embedding_model,
        retrieval_mode=STRATEGY_MODES[args.strategy],
        cross_encoder=cross_encoder,
        embedder=embedder,
    )
    chunks = chunks_to_context_dicts(retriever.search(query=text, limit=args.k_chunks))
    step(f"retrieved {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    main()
