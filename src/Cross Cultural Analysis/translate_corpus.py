# FIXME: Use ParlaMint 4.0 for translations, just test some
"""Machine-translate the RQ2 corpus and tweets to English (condition C5).

Uses any OpenAI-compatible endpoint (vLLM on the HPC, or OpenRouter). The SAME
model and prompt translate speeches and tweets, so C5 differs from C2 only by
language.

* ``chunks``: reads ``chunks_balanced.jsonl`` and writes ``chunks_en.jsonl`` with
  identical ``chunk_id``/metadata; ``text`` is English, ``text_original`` keeps
  the source. Ingest it with ``COLLECTION_SUFFIX=_parlamint_en``.
* ``tweets``: adds ``tweet_text_en`` to ``EU_tweets_rq2.csv`` (in place).

Both are resumable: already translated ids are skipped on a re-run.

Usage (repo root)::

    python "src/Cross Cultural Analysis/translate_corpus.py" chunks \\
        --input rag/ingest/artifacts/parlamint/chunks_balanced.jsonl \\
        --base-url http://<vllm-host>/v1 --model Qwen/Qwen2.5-32B-Instruct
    python "src/Cross Cultural Analysis/translate_corpus.py" tweets \\
        --base-url http://<vllm-host>/v1 --model Qwen/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cca_common import COUNTRY_NAMES, HERE

DEFAULT_TWEETS = HERE.parent / "datasets" / "EU_tweets_rq2.csv"

SYSTEM_PROMPT = (
    "You are a professional translator of political texts. Translate the user's "
    "text from {country} into English. Keep party names, abbreviations, @handles, "
    "#hashtags and URLs unchanged. Do not explain, summarise or add anything: "
    "return only the translation."
)


class Translator:
    def __init__(self, base_url: str, model: str, max_tokens: int = 1200):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url, api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY")
        )
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, text: str, country_code: str) -> str | None:
        country = COUNTRY_NAMES.get(country_code, country_code)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(country=country),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception as e:  # noqa: BLE001 - keep going, retried on the next run
            print(f"Translation failed: {e}")
            return None


def translate_chunks(args: argparse.Namespace, translate: Translator) -> None:
    out = args.output or args.input.with_name("chunks_en.jsonl")
    done: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as handle:
            done = {json.loads(line)["chunk_id"] for line in handle if line.strip()}
    with open(args.input, encoding="utf-8") as handle:
        todo = [
            r
            for r in map(json.loads, filter(str.strip, handle))
            if r["chunk_id"] not in done
        ]
    print(f"{len(done)} chunks already translated, {len(todo)} to go -> {out}")

    def work(rec: dict) -> dict | None:
        english = translate(rec["text"], rec.get("country_code", ""))
        return (
            None
            if english is None
            else {**rec, "text": english, "text_original": rec["text"]}
        )

    failed = 0
    with (
        open(out, "a", encoding="utf-8") as sink,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        for result in tqdm(pool.map(work, todo), total=len(todo), desc="Chunks"):
            if result is None:
                failed += 1
                continue
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Done ({failed} failed; re-run to retry).")


def translate_tweets(args: argparse.Namespace, translate: Translator) -> None:
    path = args.input or DEFAULT_TWEETS
    df = pd.read_csv(path, dtype={"text_id": str, "tweet_id": str})
    if "tweet_text_en" not in df.columns:
        df["tweet_text_en"] = pd.NA
    todo = df.index[df["tweet_text_en"].isna()].tolist()
    print(f"{len(df) - len(todo)} tweets already translated, {len(todo)} to go")
    rows = [(df.at[i, "tweet_text"], df.at[i, "country_code"]) for i in todo]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda r: translate(*r), rows), total=len(rows), desc="Tweets"
            )
        )
    for i, english in zip(todo, results):
        if english is not None:
            df.at[i, "tweet_text_en"] = english
    df.to_csv(path, index=False)
    print(f"Wrote {df['tweet_text_en'].notna().sum()} translations to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("what", choices=["chunks", "tweets"])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="chunks only")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.what == "chunks" and args.input is None:
        parser.error("chunks needs --input")

    translate = Translator(args.base_url, args.model)
    (translate_chunks if args.what == "chunks" else translate_tweets)(args, translate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
