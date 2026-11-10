"""Jev (TypeSafe System One) judge for retrieved RAG chunks.

Runs the calibrated Jev decision model through the TypeSafe Python SDK over
annotated chunks (the format written by ``run_streamlit.py``) and, when the
chunks carry human labels, reports agreement with them (quadratic-weighted κ
for the ordinal R_top / R_ideo, Cohen's κ for the binary A_caus).

The questions, and the rules that turn their answers into labels, live in
``judge_question_sets.py`` as named *question sets* (``v1``, ``v2``, ...).
Several sets can be judged in one run (``--question-set v1,v2``): their
questions go into the SAME request per chunk, so every set answers about
identical state and the comparison is paired. Questions with identical wording
in two sets are sent once (their wire id is ``<name>__<hash>``). Each set gets
its own output folder, plus a ``comparison.json`` with per-dimension deltas
against the first set and text-level bootstrap CIs.

Labels (v1 rules, kept for v2 so differences come from the wording):

* ``R_top``  — 1 + the highest gate answered yes: T_domain (2), T_subarea (3),
  T_item (4), T_central (5)
* ``R_ideo`` — 5 = party AND stance, 4 = stance AND evaluative, 3 = evaluative,
  else 1 / 2 (I_procedural); v2 uses I_lr_position as the stance gate
* ``A_caus`` — deterministic from citations by number / speaker name; otherwise
  the model (v1: one noul; v2: min of two nouls)

v2 additionally reports ``R_top@holistic`` / ``R_ideo@holistic`` (Score
questions with the annotators' rubric) as alternative labels. Every row also
stores continuous ``scores`` (for Spearman / risk-coverage) and ``details``
(A_caus path, Score confidence, whether the input names a specific item).

R_top / R_ideo are asked with the chunk's speaker metadata withheld
(``--state-mode nometa``, the annotators' "ignore metadata" rule). Two further
layers sit on top:

* ``--posthoc-model`` (default: the R_ideo cut points of round 3b, see
  ``DEFAULT_POSTHOC_MODEL``; ``none`` to switch off): a fitted mapping from the
  gate probabilities to the level (``calibrate.py --export-quantile``). It is
  only applied to a question set whose hash matches the one it was fitted on;
* ``needs_review`` (always computed): per-dimension flag for chunks whose gate
  answers are near the threshold, contradict each other, or (v2) disagree with
  the holistic Score question by two or more levels. A dimension read through
  cut points is flagged by its distance to the nearest cut point instead;
* ``spot_check``: a deterministic random ~5 % of chunks that a human rates
  blind, to estimate the error of the labels nobody reviews.

Judge-assisted annotation of the generator results (all model families):

    python src/run_llm_judge.py --samples all --run-id families --concurrency 8
    python src/run_llm_judge.py --samples all --run-id families --resume   # after a crash / failed chunks
    streamlit run src/run_streamlit.py -- --review "RAG Analysis/LLM judge results/families/llama_70b" --annotator Jannes
    python src/merge_judge_review.py "RAG Analysis/LLM judge results/families"

The SDK talks to the native TypeSafe API with ``TYPESAFE_API_KEY`` from
``.env.local`` and falls back to OpenRouter with ``OPENROUTER_API_KEY_ME``.

Usage (from the project root):

    python src/run_llm_judge.py --list-question-sets
    python src/run_llm_judge.py --limit 5
    python src/run_llm_judge.py --question-set v1,v2 --run-id v1_vs_v2
    python src/run_llm_judge.py --question-set v2 --questions-file <v2_candidate.json> --run-id v2b
    python src/run_llm_judge.py --rederive-from "RAG Analysis/LLM judge results/<run>/llm_judge_annotations.jsonl"
    python src/run_llm_judge.py --rederive-from <flat.jsonl> --posthoc-model <posthoc.json>

    # calibration run on the consensus chunks without the cut points
    python src/run_llm_judge.py --posthoc-model none

The extended agreement report (Krippendorff's alpha with bootstrap CIs, judge
as a 4th rater, strata, risk-coverage, post-hoc calibration CV) is produced
offline by "RAG Analysis/LLM calib/judge_calibration/calibrate.py".

Each run writes into ``RAG Analysis/LLM judge results/<run_id>/`` (one set) or
``.../<run_id>/<set>/`` (several sets):

* ``llm_judge_annotations.jsonl`` — flat, one line per (record, chunk) pair,
  with the raw Jev answers under the set's local question names.
* ``llm_judge_qualitative_annotations.jsonl`` — the same judgments grouped per
  text in the human annotation format.
* ``llm_judge_report.json`` — the agreement report.
* ``comparison.json`` (several sets only, in ``<run_id>/``).

Usage per row is the whole request, which is shared between the sets of a
paired run; do not add the per-set totals.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from typesafe_sdk import RetryPolicy, TypeSafeClient

from judge_question_sets import (
    A_CAUS_THRESHOLD,
    QUESTION_SETS,
    QuestionSet,
    hash_questions,
)
from run_streamlit import build_annotation_record, load_existing_annotations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

# Native TypeSafe API (TYPESAFE_API_KEY). Pinned to a version id so the
# calibration stays tied to one model version; the response's `model` field
# records the version that actually answered.
TYPESAFE_JEV_MODEL = os.getenv("TYPESAFE_JEV_MODEL", "jev-1.14.0")
# OpenRouter-compatible endpoint (OPENROUTER_API_KEY_ME), used only when no
# TYPESAFE_API_KEY is present.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_JEV_MODEL = os.getenv("OPENROUTER_JEV_MODEL", "~typesafe/jev-latest")

# Calibration reference: median / majority of the three annotators after the
# discussion round (see human_agreement_report_revised.json).
DEFAULT_ANNOTATIONS_PATH = (
    "RAG Analysis/human_annotations/revised/consensus_reference_annotations.jsonl"
)
DEFAULT_OUTPUT_ROOT = "RAG Analysis/LLM judge results"
# Best post-hoc mapping (round 3b): distribution-matched R_ideo cut points for
# question set v1. Applied by default; R_top stays on the gate ladder.
DEFAULT_POSTHOC_MODEL = (
    "RAG Analysis/LLM calib/judge_calibration/"
    "posthoc_quantile_2026-10-23_142551_R_ideo.json"
)
# --samples all: the generator results of every model family and size.
QUALITATIVE_SAMPLES_GLOB = "results/qualitative/*/*/qualitative_e5_simple.jsonl"
FLAT_SNAPSHOT_NAME = "llm_judge_annotations.jsonl"
GROUPED_SNAPSHOT_NAME = "llm_judge_qualitative_annotations.jsonl"
REPORT_NAME = "llm_judge_report.json"
COMPARISON_NAME = "comparison.json"

DIMENSIONS = ("R_top", "R_ideo", "A_caus")
ORDINAL_DIMENSIONS = ("R_top", "R_ideo")
KAPPA_TARGET = 0.70  # below this the judge lacks construct validity
BOOTSTRAP_SAMPLES = 2000

# Review routing for labels read through cut points: a chunk is flagged when its
# summed gate logit lies within this distance of a cut point. On the 125
# consensus chunks (R_ideo) this flags 31 % and catches 20 of 43 errors, against
# 34 % and 17 of 43 for the gate-near-0.5 rule (2.0: 54 % and 32 of 43).
REVIEW_CUT_BAND = 1.0
# Deterministic random share of chunks rated blind by a human, whatever their
# flags: estimates the error rate of the labels nobody reviewed.
SPOT_CHECK_SHARE = 0.05

REQUEST_TIMEOUT_SECONDS = 180  # Jev calls can take ~60s; keep headroom
# The SDK owns backoff, jitter, and Retry-After handling. `timeout=None` keeps
# the retry budget unbounded so a slow first attempt does not suppress retries;
# each individual attempt is still capped by the client timeout above.
RETRY_POLICY = RetryPolicy(
    max_retries=4, backoff_initial=1.0, backoff_max=30.0, timeout=None
)

EVALUATION_TASK = (
    "Evaluating evidence quality in a RAG pipeline for political stance and "
    "bias classification. `input_text` is a German political text (social "
    "media post, speech snippet, or quote) that was classified for political "
    "bias by a generator. `retrieved_chunk` is the passage the generator "
    "retrieved from a corpus of German parliamentary speeches as evidence, "
    "and `generator_justification` is the rationale it produced for its "
    "classification. Judge whether the retrieved chunk is valid, actionable, "
    "and causally utilized evidence for classifying the input text. The "
    "question rubrics are written in English; the texts themselves are German."
)

# State for the content questions in --state-mode minimal: only what the human
# annotators judged. Changed from "sentence markers" to "interjection markers"
# (what [I-n] is); this only affects --state-mode minimal, not the nometa default.
# CHECK: if chunk_text contains the re-filled heckle text rather than bare
# markers, give the judge what the annotators saw.
CONTENT_TASK = (
    "Comparing a passage from German parliamentary speeches with a German "
    "political text (social media post, speech snippet, or quote). `input_text` "
    "is the political text. `retrieved_chunk.text` is the passage; it may begin "
    "or end mid-sentence. Markers such as [I-3] show where an interjection from "
    "the floor (heckle, applause) occurred; they are not the speaker's words. "
    "Judge only what is written in the two texts. The question rubrics are "
    "written in English; the texts themselves are German."
)


# ---------------------------------------------------------------------------
# Citation evidence and state building
# ---------------------------------------------------------------------------


# Numeric references to retrieved chunks: bracket groups such as "[2]", "[2, 3]",
# "[1-3]", "[1 and 4]" (but not sentence markers like "[I-4]"), and keyword forms
# such as "chunk 2", "anchors 1 and 3". Superset of the single "[n]" pattern in
# human_annotations/update_acaus_from_citations.py, which misses list citations.
_BRACKET_CITATION = re.compile(r"\[\s*(\d+(?:\s*(?:,|;|&|and|-|–|to)\s*\d+)*)\s*\]")
_KEYWORD_CITATION = re.compile(
    r"\b(?:chunks?|anchors?|references?|passages?|excerpts?)\s*(?:no\.?\s*|#\s*)?"
    r"(\d+(?:\s*(?:,|;|&|and|-|–|to)\s*\d+)*)",
    re.IGNORECASE,
)
_RANGE_SEPARATORS = re.compile(r"\s*(?:-|–|to)\s*")


def _expand_reference_group(group: str) -> set[int]:
    numbers: set[int] = set()
    for part in re.split(r"\s*(?:,|;|&|and)\s*", group.strip()):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if (
            len(bounds) == 2
            and _RANGE_SEPARATORS.search(part)
            and bounds[0] <= bounds[1]
        ):
            numbers.update(range(bounds[0], bounds[1] + 1))
        else:
            numbers.update(bounds)
    return numbers


def cited_chunk_numbers(justification: str) -> set[int]:
    """1-based chunk numbers the justification cites by number, in any form."""
    cited: set[int] = set()
    for pattern in (_BRACKET_CITATION, _KEYWORD_CITATION):
        for match in pattern.finditer(justification or ""):
            cited |= _expand_reference_group(match.group(1))
    return cited


def citation_info(justification: str, chunk_number: int, speaker: str) -> dict:
    """Deterministic evidence that the justification refers to one chunk.

    Reproduces how the human A_caus labels behave (checked on the consensus
    set): a number-cited chunk is 1; once the justification cites chunks by
    number, uncited chunks are 0 unless their speaker is named; without any
    numeric citation the call is left to the model (speaker / party reference).
    """
    cited = cited_chunk_numbers(justification)
    last_name = (speaker or "").strip().split(" ")[-1] if speaker else ""
    named = len(last_name) >= 3 and bool(
        re.search(rf"\b{re.escape(last_name)}\b", justification or "")
    )
    return {
        "cited_by_number": chunk_number in cited,
        "any_numeric_citation": bool(cited),
        "speaker_named": named,
    }


def build_state(record: dict, chunk: dict) -> dict:
    """Build the Jev (System One) state for one (record, chunk) pair.

    State is a JSON object; each rubric references these fields by backticked
    path (e.g. ``retrieved_chunk.text``). The rubric lives in the questions;
    the state carries the material to judge. No similarity/retrieval scores
    are included.
    """
    meta = chunk.get("chunk_metadata") or {}
    number = chunk.get("chunk_index", 0) + 1
    rag = (record.get("source_context") or {}).get("rag") or {}
    justification = (
        rag.get("justification") or ""
    ).strip() or "No justification provided."

    retrieved_chunk = {
        "chunk_identifier": f"[{number}]",
        "citation_forms": (
            f'"[{number}]", a list containing it such as "[{number}, ...]", '
            f'"reference {number}", "anchor {number}" or "Chunk {number}"'
        ),
        "party": meta.get("party", "Unknown"),
        "speaker": meta.get("speaker", "Unknown"),
        "source": meta.get("source", "Unknown"),
    }
    if meta.get("date"):
        retrieved_chunk["date"] = str(meta["date"])
    retrieved_chunk["text"] = (chunk.get("chunk_text") or "").strip()

    return {
        "evaluation_task": EVALUATION_TASK,
        "input_text": (record.get("input_text") or "").strip(),
        "retrieved_chunk": retrieved_chunk,
        "generator_justification": justification,
    }


def build_nometa_state(record: dict, chunk: dict) -> dict:
    """Full state minus the chunk's metadata (party, speaker, source, identifier).

    Keeps the original task text and the generator justification, so the only
    difference to the full state is the annotators' rule 'ignore speaker metadata'.
    """
    state = build_state(record, chunk)
    state["retrieved_chunk"] = {"text": state["retrieved_chunk"]["text"]}
    return state


def build_content_state(record: dict, chunk: dict) -> dict:
    """Minimal state for the content questions: the two texts and nothing else."""
    return {
        "evaluation_task": CONTENT_TASK,
        "input_text": (record.get("input_text") or "").strip(),
        "retrieved_chunk": {"text": (chunk.get("chunk_text") or "").strip()},
    }


# ---------------------------------------------------------------------------
# Unannotated generator results (--samples)
# ---------------------------------------------------------------------------


def resolve_sample_paths(specs: list[str]) -> list[Path]:
    """Expand ``all`` and glob patterns into existing sample files."""
    paths: list[Path] = []
    for spec in specs:
        pattern = QUALITATIVE_SAMPLES_GLOB if spec == "all" else spec
        matches = sorted(glob.glob(pattern)) or ([spec] if Path(spec).exists() else [])
        if not matches:
            raise SystemExit(f"No sample file matches {spec!r}")
        paths.extend(Path(m) for m in matches)
    return list(dict.fromkeys(paths))


def sample_label(path: Path) -> str:
    """Output folder name: 'llama_8b' for results/qualitative/llama/8b/<file>."""
    if path.name.startswith("qualitative_") and path.parent.parent.name:
        return f"{path.parent.parent.name}_{path.parent.name}"
    return path.stem


def load_sample_records(path: Path) -> dict:
    """Generator results (``qualitative_*.jsonl``) as unlabelled annotation records.

    Same shape as ``run_streamlit.build_annotation_record`` writes, so the judge
    sees exactly the fields it saw on the calibration chunks (chunk metadata
    without the date, as in the Streamlit records).
    """
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        chunks = (sample.get("rag") or {}).get("retrieved_chunks") or []
        chunk_annotations = [
            {
                "chunk_index": i,
                "chunk_text": chunk.get("text", ""),
                "chunk_metadata": {
                    "party": chunk.get("party", ""),
                    "speaker": chunk.get("speaker", ""),
                    "source": chunk.get("source", ""),
                    "score": chunk.get("score", 0.0),
                },
            }
            for i, chunk in enumerate(chunks)
        ]
        record = build_annotation_record(sample, chunk_annotations, annotator="")
        record["sample_manifest"] = {"source_file": str(path)}
        records[sample.get("text_index")] = record
    return records


def is_spot_check(source: str, text_index, chunk_index) -> bool:
    """Deterministic ~SPOT_CHECK_SHARE sample: stable across re-runs and rederives."""
    key = f"{source}|{text_index}|{chunk_index}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF < SPOT_CHECK_SHARE


# ---------------------------------------------------------------------------
# Jev client (TypeSafe Python SDK)
# ---------------------------------------------------------------------------


def build_client(model: str | None) -> tuple[TypeSafeClient, str]:
    """Create the Jev client and resolve the model id to send.

    Prefers the native TypeSafe API (``TYPESAFE_API_KEY``); falls back to the
    OpenRouter-compatible endpoint (``OPENROUTER_API_KEY_ME``).
    """
    typesafe_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY_ME", "").strip()
    if typesafe_key:
        resolved = model or TYPESAFE_JEV_MODEL
        options = {"api_key": typesafe_key}
    elif openrouter_key:
        resolved = model or OPENROUTER_JEV_MODEL
        options = {"api_key": openrouter_key, "base_url": OPENROUTER_BASE_URL}
    else:
        raise SystemExit(
            "Error: set TYPESAFE_API_KEY or OPENROUTER_API_KEY_ME in .env.local"
        )
    client = TypeSafeClient(
        model=resolved,
        timeout=REQUEST_TIMEOUT_SECONDS,
        retry=RETRY_POLICY,
        **options,
    )
    return client, resolved


def judge_chunk(
    client: TypeSafeClient, state: dict, questions: dict, model: str
) -> dict:
    """Judge one chunk via the SDK; returns {'answers', 'usage', 'model'}.

    The SDK validates the response against its answer schema; answers are
    converted to plain dicts so the raw values can be stored in the JSONL. A
    Score answer's ``legend`` (the level texts) is dropped: it repeats the
    question and would bloat every row.
    """
    response = client.system_one(state=state, questions=questions, model=model)

    answers: dict[str, dict] = {}
    for name, question in questions.items():
        answer = response.answers.get(name)
        if answer is None or answer.type != question["type"]:
            raise RuntimeError(f"Missing or malformed answer for '{name}': {answer!r}")
        dump = answer.model_dump()
        dump.pop("legend", None)
        answers[name] = dump

    usage = response.usage.model_dump()
    # The native TypeSafe API reports tokens only; OpenRouter also reports cost.
    raw_usage = (response.raw_http_response.json() or {}).get("usage") or {}
    if raw_usage.get("cost") is not None:
        usage["cost"] = raw_usage["cost"]

    return {"answers": answers, "usage": usage, "model": response.model}


def _add_usage(total: dict, extra: dict) -> dict:
    keys = set(total) | set(extra)
    return {k: (total.get(k) or 0) + (extra.get(k) or 0) for k in keys}


# ---------------------------------------------------------------------------
# Request plan: several question sets in one request
# ---------------------------------------------------------------------------


@dataclass
class RequestPlan:
    """Wire ids for the union of several question sets.

    ``wire[set][local_name]`` is the id sent to the API. Identical questions
    share a wire id and are therefore asked once.
    """

    content: dict  # wire id -> question (reduced state)
    full: dict  # wire id -> question (full state)
    wire: dict  # set name -> {local name -> wire id}


def build_request_plan(sets: list[QuestionSet], state_mode: str) -> RequestPlan:
    content, full, wire = {}, {}, {}
    for qs in sets:
        wire[qs.name] = {}
        for name, question in qs.questions.items():
            wid = f"{name}__{hash_questions(question)[:8]}"
            wire[qs.name][name] = wid
            needs_full = state_mode == "full" or name in qs.full_state_questions
            (full if needs_full else content)[wid] = question
    return RequestPlan(content=content, full=full, wire=wire)


# ---------------------------------------------------------------------------
# Post-hoc mapping (optional)
# ---------------------------------------------------------------------------


def apply_posthoc(raw_answers: dict, spec: dict) -> int:
    """Level from a fitted ordinal mapping (see ``calibrate.py --export-posthoc``).

    Each threshold is a logistic model on the clipped gate logits; the level is
    ``base`` plus the number of thresholds the chunk reaches.
    """
    probs = np.clip([raw_answers[g]["noul"] for g in spec["gates"]], 0.02, 0.98)
    logits = np.log(probs / (1 - probs))
    level = spec["base"]
    for threshold in spec["thresholds"]:
        if "constant" in threshold:
            level += int(threshold["constant"])
        else:
            z = float(np.dot(threshold["coef"], logits)) + threshold["intercept"]
            level += int(z >= 0)
    return level


def posthoc_uncertain(raw_answers: dict, spec: dict, band: float = REVIEW_CUT_BAND) -> bool:
    """True when the chunk lies within ``band`` of a cut point (logit units for
    the quantile cut points, whose coefficients are all 1)."""
    probs = np.clip([raw_answers[g]["noul"] for g in spec["gates"]], 0.02, 0.98)
    logits = np.log(probs / (1 - probs))
    for threshold in spec["thresholds"]:
        if "constant" in threshold:
            continue
        z = float(np.dot(threshold["coef"], logits)) + threshold["intercept"]
        if abs(z) < band:
            return True
    return False


def load_posthoc(path: str | None) -> dict | None:
    """Fitted mapping(s) written by ``calibrate.py --export-quantile`` / ``--export-posthoc``."""
    if not path or path.lower() == "none":
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def posthoc_for_set(posthoc: dict | None, qs: QuestionSet) -> dict:
    """The mappings that may be applied to this set's answers.

    Cut points live on the scale of the gate probabilities, so they only fit the
    wording they were fitted on (the v5 cut points applied to a stricter
    T_subarea shifted R_top down by a whole level). A mapping fitted on another
    hash, or using gates the set does not ask, is skipped with a warning.
    """
    usable = {}
    for dim, spec in (posthoc or {}).items():
        fitted = (spec.get("fitted_on") or {}).get("questions_hash")
        missing = set(spec.get("gates", [])) - set(qs.questions)
        if missing or (fitted and fitted != qs.hash):
            why = (
                f"gates {sorted(missing)} not in the set"
                if missing
                else f"fitted on {fitted}, set is {qs.hash}"
            )
            print(
                f"Warning: {dim} post-hoc mapping not applied to set '{qs.name}' ({why}).",
                file=sys.stderr,
            )
            continue
        usable[dim] = spec
    return usable


# ---------------------------------------------------------------------------
# Deriving one result row
# ---------------------------------------------------------------------------


def make_row(
    qs: QuestionSet, base: dict, raw: dict, citation: dict, posthoc: dict
) -> dict:
    derived = qs.derive(raw, citation)
    labels = dict(derived["labels"])
    review = qs.needs_review(raw, citation, derived)
    for dim, spec in posthoc.items():
        labels[dim] = apply_posthoc(raw, spec)
        # the gate-near-0.5 rule does not describe a label read through cut points
        review[dim] = posthoc_uncertain(raw, spec)
    review["any"] = any(review[d] for d in DIMENSIONS if d in review)
    return {
        **base,
        "question_set": qs.name,
        "questions_hash": qs.hash,
        "raw_answers": raw,
        "citation": citation,
        "derived_labels": labels,
        "uncalibrated_labels": derived["labels"] if posthoc else None,
        "alt_labels": derived["alt_labels"],
        "scores": derived["scores"],
        "details": derived["details"],
        "needs_review": review,
        "spot_check": is_spot_check(
            base.get("source", ""), base.get("text_index"), base.get("chunk_index")
        ),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_previous_rows(path: Path) -> list[dict]:
    """Rows of an earlier (partial) run that were judged without error."""
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if "error" not in r]


def run_judging(
    args: argparse.Namespace,
    sets: list[QuestionSet],
    records: dict,
    source: str,
    previous: list[dict] | None = None,
) -> dict[str, list[dict]]:
    """Judge every chunk once for all sets; return the results per set.

    ``previous`` (one set only, ``--resume``): rows already judged, kept as they
    are; only the remaining chunks are sent.
    """
    done = {(r["text_index"], r["chunk_index"]) for r in previous or []}
    tasks = [
        (record, chunk)
        for record in records.values()
        for chunk in record.get("chunk_annotations", [])
        if (record.get("text_index"), chunk.get("chunk_index")) not in done
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    posthoc = load_posthoc(args.posthoc_model)
    posthoc_by_set = {qs.name: posthoc_for_set(posthoc, qs) for qs in sets}
    plan = build_request_plan(sets, args.state_mode)
    client, args.model = build_client(args.model)

    n_requests = int(bool(plan.content)) + int(bool(plan.full))
    print(f"Judge model: {args.model}")
    for qs in sets:
        print(f"Question set {qs.name}: hash {qs.hash}, {len(qs.questions)} questions")
    print(
        f"Distinct questions sent: {len(plan.content) + len(plan.full)} "
        f"in {n_requests} request(s) per chunk"
    )
    print(f"State mode: {args.state_mode}")
    if done:
        print(f"Resuming: {len(done)} chunk(s) already judged")
    print(f"Judging {len(tasks)} chunk(s) of {source} at concurrency {args.concurrency}")

    def _process(record: dict, chunk: dict) -> dict[str, dict]:
        base = {
            "source": source,
            "text_index": record.get("text_index"),
            "chunk_index": chunk.get("chunk_index"),
            "model": args.model,
            "state_mode": args.state_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "human": {dim: chunk.get(dim) for dim in DIMENSIONS},
        }
        try:
            state = build_state(record, chunk)
            content_state = {
                "full": state,
                "nometa": build_nometa_state(record, chunk),
                "minimal": build_content_state(record, chunk),
            }[args.state_mode]
            answers, usage, response_model = {}, {}, None
            for st, questions in ((content_state, plan.content), (state, plan.full)):
                if not questions:
                    continue
                judged = judge_chunk(client, st, questions, args.model)
                answers.update(judged["answers"])
                usage = _add_usage(usage, judged["usage"])
                response_model = judged["model"]
            citation = citation_info(
                state["generator_justification"],
                chunk.get("chunk_index", 0) + 1,
                state["retrieved_chunk"]["speaker"],
            )
        except Exception as e:  # noqa: BLE001
            return {
                qs.name: {
                    **base,
                    "question_set": qs.name,
                    "questions_hash": qs.hash,
                    "error": str(e),
                }
                for qs in sets
            }

        rows = {}
        for qs in sets:
            raw = {local: answers[wid] for local, wid in plan.wire[qs.name].items()}
            try:
                row = make_row(qs, base, raw, citation, posthoc_by_set[qs.name])
                row.update(response_model=response_model, usage=usage)
            except Exception as e:  # noqa: BLE001  (a bug in one set's derivation)
                row = {
                    **base,
                    "question_set": qs.name,
                    "questions_hash": qs.hash,
                    "raw_answers": raw,
                    "error": f"derive: {e}",
                }
            rows[qs.name] = row
        return rows

    results: dict[str, list[dict]] = {qs.name: list(previous or []) for qs in sets}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_process, r, c) for r, c in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            for name, row in f.result().items():
                results[name].append(row)
    client.close()
    for rows in results.values():
        rows.sort(key=lambda r: (str(r.get("text_index")), r.get("chunk_index") or 0))
    return results


def rederive_results(
    path: str, records: dict, set_override: QuestionSet | None, posthoc: dict | None
) -> tuple[list[dict], QuestionSet]:
    """Re-apply the current label rules to a stored run, without API calls.

    The set is taken from the rows' ``question_set`` field (rows written before
    question sets existed are v1), unless ``--question-set`` names one.
    """
    human = {
        (text_index, chunk.get("chunk_index")): {d: chunk.get(d) for d in DIMENSIONS}
        for text_index, record in records.items()
        for chunk in record.get("chunk_annotations", [])
    }
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = {r.get("question_set") or "v1" for r in rows}
    if set_override is None:
        if len(names) != 1:
            raise SystemExit(
                f"Rows from several question sets {sorted(names)}; pass --question-set"
            )
        qs = QUESTION_SETS[names.pop()]
    else:
        qs = set_override
    hashes = {r.get("questions_hash") for r in rows if r.get("questions_hash")}
    if hashes and hashes != {qs.hash}:
        print(
            f"Note: stored rows have questions hash {sorted(hashes)}, set '{qs.name}' "
            f"is {qs.hash}; labels are re-derived with the '{qs.name}' rules.",
            file=sys.stderr,
        )
    ph = posthoc_for_set(posthoc, qs)

    results = []
    for row in rows:
        if "error" in row:
            results.append(row)
            continue
        keep = {
            k: row[k]
            for k in (
                "source",
                "text_index",
                "chunk_index",
                "model",
                "state_mode",
                "timestamp",
                "response_model",
                "usage",
            )
            if k in row
        }
        keep["human"] = human.get(
            (row["text_index"], row["chunk_index"]), row.get("human")
        )
        new = make_row(qs, keep, row["raw_answers"], row["citation"], ph)
        new["questions_hash"] = row.get("questions_hash", new["questions_hash"])
        results.append(new)
    return results, qs


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def write_snapshots(
    results: list[dict], records: dict, model: str, run_dir: Path
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    flat_path = run_dir / FLAT_SNAPSHOT_NAME
    grouped_path = run_dir / GROUPED_SNAPSHOT_NAME
    _write_jsonl(flat_path, results)
    _write_jsonl(grouped_path, build_judge_annotation_records(results, records, model))
    print(f"Flat annotations saved to: {flat_path}")
    print(f"Grouped annotations saved to: {grouped_path}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_judge_annotation_records(
    results: list[dict], records: dict, model: str
) -> list[dict]:
    """Group flat per-chunk judge results into human-style per-text records.

    Mirrors the shape written by ``run_streamlit.build_annotation_record`` so a
    judge run and a human annotator's file can be compared key-for-key. The
    judge's labels live in the same ``R_top`` / ``R_ideo`` / ``A_caus`` fields;
    the reference labels are kept under ``human_labels`` and the raw answers
    under ``raw_answers``.
    """
    chunk_lookup = {
        (text_index, chunk.get("chunk_index")): chunk
        for text_index, record in records.items()
        for chunk in record.get("chunk_annotations", [])
    }

    grouped: dict = {}
    for res in results:
        grouped.setdefault(res.get("text_index"), []).append(res)

    annotation_records = []
    for text_index, entries in grouped.items():
        sample = records.get(text_index, {})
        chunk_annotations = []
        for res in sorted(entries, key=lambda r: r.get("chunk_index") or 0):
            chunk = chunk_lookup.get((text_index, res.get("chunk_index")), {})
            meta = chunk.get("chunk_metadata") or {}
            labels = res.get("derived_labels") or {}
            annotation = {
                "chunk_index": res.get("chunk_index"),
                "chunk_text": (chunk.get("chunk_text") or "").strip(),
                "chunk_metadata": {
                    "party": meta.get("party", ""),
                    "speaker": meta.get("speaker", ""),
                    "source": meta.get("source", ""),
                    "score": meta.get("score", 0.0),
                },
                "R_top": labels.get("R_top"),
                "R_ideo": labels.get("R_ideo"),
                "A_caus": labels.get("A_caus"),
                "notes": "",
                "timestamp": res.get("timestamp"),
                # Judge-only extras, not part of the human schema.
                "question_set": res.get("question_set"),
                "human_labels": res.get("human"),
                "alt_labels": res.get("alt_labels"),
                "response_model": res.get("response_model"),
                "raw_answers": res.get("raw_answers"),
                "citation": res.get("citation"),
                "needs_review": res.get("needs_review"),
                "spot_check": res.get("spot_check"),
                "usage": res.get("usage"),
            }
            if "error" in res:
                annotation["error"] = res["error"]
            chunk_annotations.append(annotation)

        annotation_records.append(
            {
                "text_index": text_index,
                "input_text": sample.get("input_text", ""),
                "source_context": sample.get("source_context") or {},
                "sample_manifest": sample.get("sample_manifest") or {},
                "chunk_annotations": chunk_annotations,
                "annotator": model,
                "sample_notes": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return annotation_records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _human_dim(key: str) -> str:
    """'R_top@holistic' -> 'R_top'."""
    return key.split("@", 1)[0]


def _judge_label(row: dict, key: str):
    if key in DIMENSIONS:
        return (row.get("derived_labels") or {}).get(key)
    return (row.get("alt_labels") or {}).get(key)


def _metric_keys(results: list[dict]) -> list[str]:
    alt = sorted(
        {k for r in results if "error" not in r for k in (r.get("alt_labels") or {})}
    )
    return [*DIMENSIONS, *alt]


def _safe_kappa(y1, y2, labels, weights=None):
    """Compute Cohen's kappa, handling the all-agree edge case."""
    k = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    if np.isnan(k):
        return 1.0 if len(y1) > 0 and np.all(y1 == y2) else None
    return float(k)


def _pairs(results: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, list]:
    dim = _human_dim(key)
    h_vals, j_vals, texts = [], [], []
    for r in results:
        if "error" in r:
            continue
        h, j = (r.get("human") or {}).get(dim), _judge_label(r, key)
        if h is None or j is None:
            continue
        h_vals.append(int(h))
        j_vals.append(int(j))
        texts.append(r.get("text_index"))
    return np.array(h_vals), np.array(j_vals), texts


def compute_metrics(results: list[dict]) -> dict:
    """Agreement per dimension (and per alternative label, e.g. R_top@holistic).

    A dimension without reference labels (unannotated chunks) maps to None.
    ``spearman`` correlates the continuous score with the human label.
    """
    per_dim = {}
    for key in _metric_keys(results):
        dim = _human_dim(key)
        ordinal = dim in ORDINAL_DIMENSIONS
        labels = [1, 2, 3, 4, 5] if ordinal else [0, 1]
        h_arr, j_arr, _ = _pairs(results, key)
        if not len(h_arr):
            per_dim[key] = None
            continue
        kappa = _safe_kappa(
            h_arr, j_arr, labels, weights="quadratic" if ordinal else None
        )

        sh, ss = [], []
        for r in results:
            score = (r.get("scores") or {}).get(key)
            h = (r.get("human") or {}).get(dim)
            if "error" not in r and score is not None and h is not None:
                sh.append(int(h))
                ss.append(float(score))
        rho = (
            spearmanr(sh, ss).statistic
            if len(set(ss)) > 1 and len(set(sh)) > 1
            else None
        )

        per_dim[key] = {
            "n": len(h_arr),
            "exact_agreement": float(np.mean(h_arr == j_arr)),
            "qwk": kappa if ordinal else None,
            "kappa": None if ordinal else kappa,
            "spearman": None if rho is None or np.isnan(rho) else float(rho),
            "mean_diff": float(np.mean(j_arr - h_arr)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(h_arr, j_arr, labels=labels).tolist(),
            "below_threshold": kappa is not None and kappa < KAPPA_TARGET,
        }
    return per_dim


def a_caus_by_path(results: list[dict]) -> dict:
    """Where A_caus errors come from: deterministic citation rules or the model."""
    out: dict = defaultdict(
        lambda: {"n": 0, "agree": 0, "false_pos": 0, "false_neg": 0}
    )
    for r in results:
        if "error" in r:
            continue
        h = (r.get("human") or {}).get("A_caus")
        if h is None:
            continue
        path = (r.get("details") or {}).get("A_caus_path", "unknown")
        j, h = r["derived_labels"]["A_caus"], int(h)
        cell = out[path]
        cell["n"] += 1
        cell["agree"] += int(j == h)
        cell["false_pos"] += int(j == 1 and h == 0)
        cell["false_neg"] += int(j == 0 and h == 1)
    return dict(out)


def metrics_by_input_specificity(results: list[dict]) -> dict:
    """R_top agreement split by whether the input names a specific item (v2)."""
    groups: dict = defaultdict(list)
    for r in results:
        flag = (r.get("details") or {}).get("input_has_item")
        if flag is not None:
            groups["item" if flag else "no_item"].append(r)
    out = {}
    for name, rows in groups.items():
        h, j, _ = _pairs(rows, "R_top")
        if len(h):
            out[name] = {
                "n": len(h),
                "qwk": _safe_kappa(h, j, [1, 2, 3, 4, 5], weights="quadratic"),
                "mean_diff": float(np.mean(j - h)),
            }
    return out


def get_worst_disagreements(results: list[dict], n: int = 5) -> list[dict]:
    """Return the top-N largest |diff| disagreements for the ordinal dims."""
    disagreements = []
    for r in results:
        if "error" in r:
            continue
        for dim in ORDINAL_DIMENSIONS:
            h = (r.get("human") or {}).get(dim)
            if h is None:
                continue
            j = r["derived_labels"][dim]
            if j != int(h):
                disagreements.append(
                    {
                        "text_index": r["text_index"],
                        "chunk_index": r["chunk_index"],
                        "dimension": dim,
                        "human": int(h),
                        "judge": j,
                        "diff": abs(j - int(h)),
                    }
                )
    disagreements.sort(
        key=lambda x: (-x["diff"], str(x["text_index"]), x["chunk_index"])
    )
    return disagreements[:n]


def compute_usage_totals(results: list[dict]) -> dict:
    """Sum usage across judged records (OpenRouter also reports cost)."""
    usages = [r["usage"] for r in results if isinstance(r.get("usage"), dict)]
    return {
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usages),
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usages),
        "cost": float(sum(u.get("cost") or 0.0 for u in usages)),
    }


# ---------------------------------------------------------------------------
# Paired comparison between question sets
# ---------------------------------------------------------------------------


def _agreement(h: np.ndarray, j: np.ndarray, dim: str) -> float | None:
    ordinal = dim in ORDINAL_DIMENSIONS
    labels = [1, 2, 3, 4, 5] if ordinal else [0, 1]
    return _safe_kappa(h, j, labels, weights="quadratic" if ordinal else None)


def compare_sets(results_by_set: dict[str, list[dict]], seed: int = 0) -> dict:
    """Paired comparison of every set against the first one.

    Uses chunks judged without error by all sets. The CI of the κ difference is
    a cluster bootstrap over texts (chunks of one text are not independent).
    For every label key of a set (incl. @holistic) the baseline is the first
    set's main label of the same dimension.
    """
    names = list(results_by_set)
    baseline = names[0]
    index = {
        name: {(r["text_index"], r["chunk_index"]): r for r in rows if "error" not in r}
        for name, rows in results_by_set.items()
    }
    common = sorted(set.intersection(*(set(i) for i in index.values())), key=str)
    rng = np.random.default_rng(seed)

    out = {"baseline": baseline, "n_common_chunks": len(common), "sets": {}}
    for name in names[1:]:
        cmp_rows = [index[name][k] for k in common]
        out["sets"][name] = {}
        for key in _metric_keys(cmp_rows):
            dim = _human_dim(key)
            triples = []  # (text, human, baseline judge, candidate judge)
            for k in common:
                h = (index[baseline][k].get("human") or {}).get(dim)
                jb, jc = _judge_label(index[baseline][k], dim), _judge_label(
                    index[name][k], key
                )
                if None not in (h, jb, jc):
                    triples.append((k[0], int(h), int(jb), int(jc)))
            if not triples:
                continue
            texts = np.array([t[0] for t in triples], dtype=object)
            h, jb, jc = (np.array([t[i] for t in triples]) for i in (1, 2, 3))
            kb, kc = _agreement(h, jb, dim), _agreement(h, jc, dim)

            by_text: dict = defaultdict(list)
            for i, t in enumerate(texts):
                by_text[t].append(i)
            ids = list(by_text)
            deltas = []
            for _ in range(BOOTSTRAP_SAMPLES):
                idx = [
                    i for t in rng.choice(len(ids), len(ids)) for i in by_text[ids[t]]
                ]
                a, b = _agreement(h[idx], jb[idx], dim), _agreement(
                    h[idx], jc[idx], dim
                )
                if a is not None and b is not None:
                    deltas.append(b - a)
            err_b, err_c = np.abs(jb - h), np.abs(jc - h)
            out["sets"][name][key] = {
                "n": len(h),
                "baseline_kappa": kb,
                "kappa": kc,
                "delta": None if kb is None or kc is None else kc - kb,
                "delta_ci95": (
                    [
                        float(np.percentile(deltas, 2.5)),
                        float(np.percentile(deltas, 97.5)),
                    ]
                    if deltas
                    else None
                ),
                "baseline_exact": float(np.mean(jb == h)),
                "exact": float(np.mean(jc == h)),
                "chunks_closer": int(np.sum(err_c < err_b)),
                "chunks_further": int(np.sum(err_c > err_b)),
                "label_changed": int(np.sum(jc != jb)),
            }
    return out


def print_comparison(comparison: dict, path: Path) -> None:
    print(
        f"\n=== Paired comparison vs. {comparison['baseline']} "
        f"({comparison['n_common_chunks']} common chunks) ==="
    )
    header = (
        f"{'Set':<5} {'Label':<16} {'κ base':>7} {'κ new':>7} {'Δκ':>7} "
        f"{'95% CI (texts)':>17} {'closer':>7} {'further':>8}"
    )
    print(header)
    print("-" * len(header))
    fmt = lambda x: f"{x:>7.3f}" if x is not None else f"{'N/A':>7}"  # noqa: E731
    for name, keys in comparison["sets"].items():
        for key, c in keys.items():
            ci = c["delta_ci95"]
            ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "N/A"
            delta = f"{c['delta']:>+7.3f}" if c["delta"] is not None else f"{'N/A':>7}"
            print(
                f"{name:<5} {key:<16} {fmt(c['baseline_kappa'])} {fmt(c['kappa'])} "
                f"{delta} {ci_s:>17} {c['chunks_closer']:>7} {c['chunks_further']:>8}"
            )
    print(
        "κ = QWK for R_top / R_ideo, Cohen's κ for A_caus. closer / further: chunks "
        "whose |judge - human| shrank / grew against the baseline."
    )
    path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Comparison saved to: {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_and_save_report(
    results: list[dict],
    per_dim: dict,
    worst: list[dict],
    model: str,
    qs: QuestionSet,
    report_path: Path,
) -> None:
    """Print a concise report to stdout and write a JSON report to disk."""
    judged = sum("derived_labels" in r for r in results)
    usage_totals = compute_usage_totals(results)
    by_path = a_caus_by_path(results)
    by_specificity = metrics_by_input_specificity(results)
    hashes = sorted(
        {r.get("questions_hash") for r in results if r.get("questions_hash")}
    )

    print(f"\n=== Jev Judge Agreement Report: question set {qs.name} ===")
    print(f"Model: {model}")
    print(f"Questions hash: {'+'.join(hashes) or qs.hash}")
    print(f"Judged chunks: {judged} / {len(results)}")

    print("\nPer-dimension metrics:")
    header = f"{'Label':<16} {'n':>5} {'Agree%':>7} {'κ/κ_w':>7} {'ρ':>6} {'MeanΔ':>7}"
    print(header)
    print("-" * len(header))
    for key, d in per_dim.items():
        if d is None:
            print(f"{key:<16} {'—':>5} {'—':>7} {'—':>7} {'—':>6} {'—':>7}")
            continue
        metric = d["qwk"] if _human_dim(key) in ORDINAL_DIMENSIONS else d["kappa"]
        metric_str = f"{metric:>7.3f}" if metric is not None else f"{'N/A':>7}"
        rho = f"{d['spearman']:>6.2f}" if d["spearman"] is not None else f"{'N/A':>6}"
        flag = f" ⚠️ below {KAPPA_TARGET:.2f}" if d["below_threshold"] else ""
        print(
            f"{key:<16} {d['n']:>5} {d['exact_agreement']*100:>6.1f}% "
            f"{metric_str} {rho} {d['mean_diff']:>+7.2f}{flag}"
        )
    print("ρ: Spearman of the continuous score with the human label.")

    print("\nConfusion matrices (rows: reference, columns: judge):")
    for key, d in per_dim.items():
        if d is None:
            continue
        print(f"\n{key} (labels: {d['labels']})")
        for row, label in zip(d["confusion_matrix"], d["labels"]):
            print(f"  {label}: {row}")

    if by_path:
        print("\nA_caus by decision path:")
        for path, c in sorted(by_path.items()):
            print(
                f"  {path:<20} n={c['n']:>3}  agree={c['agree']:>3}  "
                f"FP={c['false_pos']:>2}  FN={c['false_neg']:>2}"
            )
    if by_specificity:
        print("\nR_top by input specificity:")
        for name, c in by_specificity.items():
            qwk = f"{c['qwk']:.3f}" if c["qwk"] is not None else "N/A"
            print(f"  {name:<8} n={c['n']:>3}  QWK={qwk}  meanΔ={c['mean_diff']:+.2f}")

    if worst:
        print(f"\nTop {len(worst)} disagreements:")
        for wd in worst:
            print(
                f"  sample {wd['text_index']} chunk {wd['chunk_index']} "
                f"{wd['dimension']}: human={wd['human']} judge={wd['judge']} "
                f"diff={wd['diff']}"
            )

    review_share = {
        key: float(
            np.mean([r["needs_review"][key] for r in results if "needs_review" in r])
        )
        for key in (*DIMENSIONS, "any")
        if any("needs_review" in r for r in results)
    }
    judged_rows = [r for r in results if "needs_review" in r]
    if judged_rows:
        review_share["spot_check"] = float(np.mean([bool(r.get("spot_check")) for r in judged_rows]))
        review_share["to_review"] = float(
            np.mean([r["needs_review"]["any"] or bool(r.get("spot_check")) for r in judged_rows])
        )
    if review_share:
        shares = ", ".join(f"{k} {v*100:.0f}%" for k, v in review_share.items())
        print(f"\nRouted to human review: {shares}")

    print(
        f"\nUsage: {usage_totals['input_tokens']:,} input tokens, "
        f"{usage_totals['output_tokens']:,} output tokens, "
        f"total cost ${usage_totals['cost']:.6f}"
    )

    report = {
        "model": model,
        "question_set": qs.name,
        "questions_hash": "+".join(hashes) or qs.hash,
        "a_caus_threshold": A_CAUS_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(results),
        "judged_chunks": judged,
        "usage_totals": usage_totals,
        "per_dimension": per_dim,
        "a_caus_by_path": by_path,
        "r_top_by_input_specificity": by_specificity,
        "needs_review_share": review_share,
        "worst_disagreements": worst,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge retrieved chunks with the calibrated Jev model and "
        "report agreement with the reference annotations."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to send (default: native TypeSafe "
        f"{TYPESAFE_JEV_MODEL} when TYPESAFE_API_KEY is set, else OpenRouter "
        f"{OPENROUTER_JEV_MODEL}). The response records the version that answered.",
    )
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Annotation JSONL with the chunks to judge",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Unannotated generator results to judge instead of --annotations: "
        "'all' (every results/qualitative/<family>/<size>/qualitative_e5_simple.jsonl), "
        "file paths or glob patterns. Each file gets its own folder "
        "<run_id>/<family>_<size>/ (question set v1 only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --samples or --annotations and --run-id of an earlier run: keep "
        "the chunks already judged without error and judge only the rest",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for per-run results (a run subfolder is created inside).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Name of the output subfolder (default: timestamp).",
    )
    parser.add_argument(
        "--question-set",
        default=None,
        help=f"Comma-separated question sets ({', '.join(QUESTION_SETS)}). Several "
        "sets are judged in the same requests; the first is the baseline of the "
        "paired comparison. Default: v1 (with --rederive-from: the set stored in "
        "the rows).",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="JSON file with candidate questions that replace the questions of the "
        "(single) --question-set; the set's label rules are kept",
    )
    parser.add_argument(
        "--list-question-sets",
        action="store_true",
        help="Print the available question sets and their hashes, then exit",
    )
    parser.add_argument(
        "--state-mode",
        choices=("full", "minimal", "nometa"),
        default="nometa",
        help="nometa (default): the content questions see the full state without "
        "the chunk's party / speaker / source, as the annotators were told to "
        "ignore speaker metadata. full: every question sees the whole state. "
        "minimal: the content questions see only input_text and "
        "retrieved_chunk.text. Questions a set marks as full-state (A_caus and "
        "its v2 replacements) always get the full state (two requests per chunk "
        "in both reduced modes)",
    )
    parser.add_argument(
        "--posthoc-model",
        default=DEFAULT_POSTHOC_MODEL,
        help="JSON from calibrate.py --export-quantile or --export-posthoc: map the "
        "gate probabilities to R_top / R_ideo with a mapping fitted on the "
        "calibration set; those dimensions are then flagged for review by their "
        "distance to the cut points. Applied only to a set whose questions hash "
        "matches the fitted one. Default: the R_ideo cut points for v1; 'none' "
        "reads every label off the gate ladder",
    )
    parser.add_argument(
        "--rederive-from",
        default=None,
        help="Flat llm_judge_annotations.jsonl of an earlier run: re-apply the "
        "current label rules to its stored answers (no API calls)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N chunks (smoke test)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="ThreadPoolExecutor workers"
    )
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=5,
        help="Print top N worst disagreements for R_top / R_ideo",
    )
    return parser


def resolve_sets(args: argparse.Namespace) -> list[QuestionSet] | None:
    if not args.question_set:
        sets = None
    else:
        names = [n.strip() for n in args.question_set.split(",") if n.strip()]
        unknown = [n for n in names if n not in QUESTION_SETS]
        if unknown:
            raise SystemExit(
                f"Unknown question set(s) {unknown}; available: {list(QUESTION_SETS)}"
            )
        if len(set(names)) != len(names):
            raise SystemExit("A question set is listed twice")
        sets = [QUESTION_SETS[n] for n in names]
    if args.questions_file:
        if not sets or len(sets) != 1:
            raise SystemExit("--questions-file needs exactly one --question-set")
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        sets = [sets[0].with_questions(questions)]
    return sets


def judge_sample_files(
    args: argparse.Namespace, sets: list[QuestionSet] | None, run_dir: Path
) -> None:
    """Judge every chunk of each generator result file into <run_dir>/<label>/.

    Files are written one at a time, so an interrupted run loses at most the file
    in progress; ``--resume`` then judges only what is missing.
    """
    sets = sets or [QUESTION_SETS["v1"]]
    if len(sets) > 1:
        raise SystemExit("--samples takes one question set")
    qs = sets[0]
    summary = {}
    for path in resolve_sample_paths(args.samples):
        label = sample_label(path)
        set_dir = run_dir / label
        records = load_sample_records(path)
        previous = load_previous_rows(set_dir / FLAT_SNAPSHOT_NAME) if args.resume else None
        print(f"\n##### {label}: {path} ({len(records)} texts)")
        results = run_judging(args, sets, records, label, previous)[qs.name]
        write_snapshots(results, records, args.model, set_dir)
        print_and_save_report(
            results, compute_metrics(results), [], args.model, qs, set_dir / REPORT_NAME
        )
        ok = [r for r in results if "error" not in r]
        summary[label] = {
            "source_file": str(path),
            "chunks": len(results),
            "failed": len(results) - len(ok),
            **{
                f"review_{k}": int(sum(r["needs_review"][k] for r in ok))
                for k in (*DIMENSIONS, "any")
            },
            "spot_check": int(sum(bool(r.get("spot_check")) for r in ok)),
        }

    print("\n=== Summary (chunks flagged per dimension) ===")
    print(f"{'file':<14}{'chunks':>7}{'failed':>7}{'R_top':>7}{'R_ideo':>7}{'A_caus':>7}{'any':>6}{'spot':>6}")
    for label, s in summary.items():
        print(
            f"{label:<14}{s['chunks']:>7}{s['failed']:>7}{s['review_R_top']:>7}"
            f"{s['review_R_ideo']:>7}{s['review_A_caus']:>7}{s['review_any']:>6}{s['spot_check']:>6}"
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if any(s["failed"] for s in summary.values()):
        print("\nWarning: some chunks failed; re-run with --resume and the same --run-id.")


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.list_question_sets:
        for qs in QUESTION_SETS.values():
            print(
                f"{qs.name}: hash {qs.hash}, {len(qs.questions)} questions "
                f"({len(qs.full_state_questions)} full-state) - {qs.description}"
            )
        return

    sets = resolve_sets(args)
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    print(f"Output directory: {run_dir.resolve()}")

    if args.samples:
        judge_sample_files(args, sets, run_dir)
        return

    if args.rederive_from:
        if sets and len(sets) > 1:
            raise SystemExit("--rederive-from takes one question set")
        records = load_existing_annotations(args.annotations)
        results, qs = rederive_results(
            args.rederive_from,
            records,
            sets[0] if sets else None,
            load_posthoc(args.posthoc_model),
        )
        args.model = next((r["model"] for r in results if r.get("model")), "unknown")
        results_by_set, sets = {qs.name: results}, [qs]
    else:
        sets = sets or [QUESTION_SETS["v1"]]
        records = load_existing_annotations(args.annotations)
        if not records:
            raise SystemExit(f"No annotations found in {args.annotations}")
        if args.resume and len(sets) > 1:
            raise SystemExit("--resume takes one question set")
        previous = load_previous_rows(run_dir / FLAT_SNAPSHOT_NAME) if args.resume else None
        results_by_set = run_judging(
            args, sets, records, Path(args.annotations).stem, previous
        )

    paired = len(sets) > 1
    failed = 0
    for qs in sets:
        results = results_by_set[qs.name]
        set_dir = run_dir / qs.name if paired else run_dir
        write_snapshots(results, records, args.model, set_dir)
        print_and_save_report(
            results,
            compute_metrics(results),
            get_worst_disagreements(results, n=args.show_disagreements),
            args.model,
            qs,
            set_dir / REPORT_NAME,
        )
        failed = max(failed, sum("error" in r for r in results))

    if paired:
        print_comparison(compare_sets(results_by_set), run_dir / COMPARISON_NAME)
    if failed:
        print(f"\nWarning: {failed} chunk(s) failed judging.")


if __name__ == "__main__":
    main()
