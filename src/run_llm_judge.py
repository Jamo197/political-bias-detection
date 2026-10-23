"""Jev (TypeSafe System One) judge for retrieved RAG chunks.

Runs the calibrated Jev decision model through the TypeSafe Python SDK over
annotated chunks (the format written by ``run_streamlit.py``) and, when the
chunks carry human labels, reports agreement with them (quadratic-weighted κ
for the ordinal R_top / R_ideo, Cohen's κ for the binary A_caus).

Each (record, chunk) pair becomes one System One request whose typed yes/no
(Noul) questions are answered independently. R_top and R_ideo are *ladders*:
each level is gated by one question with a plain test, and the label is the
highest level whose gate the model answers yes to (probability >= 0.5):

* ``R_top``  — T_domain (>= 2), T_subarea (>= 3), T_item (>= 4), T_central (5)
* ``R_ideo`` — I_evaluative (3), I_stance (4), I_party (5); I_procedural (1 vs 2)
* ``A_caus`` — probability that the justification relied on this chunk

The ladder questions (``JEV_QUESTIONS``) and the A_caus citation gate were
calibrated against the revised consensus annotations; the iteration history
(``CALIBRATION_LOG.md``) and the tooling live in
"RAG Analysis/judge_calibration/". By default nothing is rescaled or fitted
afterwards: every label can be read off the individual answers. R_top / R_ideo
are asked with the chunk's speaker metadata withheld (``--state-mode nometa``,
the annotators' "ignore metadata" rule; it raised R_ideo alpha from 0.49 to 0.56
without fitting anything). Two further layers sit on top:

* ``--posthoc-model`` (off; no longer recommended, it only helped before the
  metadata was withheld): an ordinal-logistic mapping from the gate
  probabilities, fitted by ``calibrate.py --export-posthoc``;
* ``needs_review`` (always computed): per-dimension flag for chunks whose gate
  answers are near the threshold or contradict each other, to be escalated to a
  human instead of trusting the label.

The SDK talks to the native TypeSafe API with ``TYPESAFE_API_KEY`` from
``.env.local`` and falls back to OpenRouter with ``OPENROUTER_API_KEY_ME``.

Usage (from the project root):

    python src/run_llm_judge.py --limit 5
    python src/run_llm_judge.py --annotations <file.jsonl> --concurrency 8
    python src/run_llm_judge.py --questions-file <questions_vN.json> --run-id vN
    python src/run_llm_judge.py --rederive-from "RAG Analysis/<run>/llm_judge_annotations.jsonl"
    python src/run_llm_judge.py --rederive-from <flat.jsonl> --posthoc-model <posthoc.json>

The extended agreement report (Krippendorff's alpha with bootstrap CIs, judge
as a 4th rater, strata, risk-coverage, post-hoc calibration CV) is produced
offline by "RAG Analysis/judge_calibration/calibrate.py".

Each run writes into ``RAG Analysis/<run_id>/``:

* ``llm_judge_annotations.jsonl`` — flat, one line per (record, chunk) pair,
  with the raw Jev answers (gate probabilities, usage).
* ``llm_judge_qualitative_annotations.jsonl`` — the same judgments grouped per
  text in the human annotation format, so the judge's labels can be compared
  directly against an annotator's file.
* ``llm_judge_report.json`` — the agreement report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from typesafe_sdk import RetryPolicy, TypeSafeClient

from run_streamlit import load_existing_annotations

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
DEFAULT_OUTPUT_ROOT = "RAG Analysis"
FLAT_SNAPSHOT_NAME = "llm_judge_annotations.jsonl"
GROUPED_SNAPSHOT_NAME = "llm_judge_qualitative_annotations.jsonl"
REPORT_NAME = "llm_judge_report.json"

DIMENSIONS = ("R_top", "R_ideo", "A_caus")
ORDINAL_DIMENSIONS = ("R_top", "R_ideo")
KAPPA_TARGET = 0.70  # below this the judge lacks construct validity

# Ladder gates, lowest level first: R_top level = 1 + the highest gate answered
# yes (none: 1). R_ideo is derived in derive_labels().
R_TOP_GATES = ("T_domain", "T_subarea", "T_item", "T_central")
R_IDEO_GATES = ("I_procedural", "I_evaluative", "I_stance", "I_party")
R_IDEO_LADDER = ("I_evaluative", "I_stance", "I_party")  # levels 3, 4, 5
GATE_THRESHOLD = 0.5  # Noul probability >= threshold -> gate is yes

A_CAUS_THRESHOLD = 0.5  # Noul probability >= threshold -> A_caus = 1

# Trust-or-escalate: a dimension is flagged for human review when any of its
# gates sits within this distance of the threshold, or the ladder contradicts
# itself (a higher level answered yes while a lower one is no). R_top has four
# gates that are rarely all decisive, so its band is narrower; on the 125
# consensus chunks this routes R_top 39 %, R_ideo 32 %, A_caus 13 % to review.
REVIEW_BANDS = {"R_top": 0.10, "R_ideo": 0.20, "A_caus": 0.20}

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

# State for the content questions (R_top / R_ideo) in --state-mode minimal: only
# what the human annotators judged. They were told to rate the chunk text and to
# ignore speaker metadata, and the rubrics never use the generator justification.
CONTENT_TASK = (
    "Comparing a passage from German parliamentary speeches with a German "
    "political text (social media post, speech snippet, or quote). `input_text` "
    "is the political text. `retrieved_chunk.text` is the passage; it may begin "
    "or end mid-sentence and contains sentence markers such as [I-3]. Judge only "
    "what is written in the two texts. The question rubrics are written in "
    "English; the texts themselves are German."
)

# Calibrated ladder questions. Structure follows the TypeSafe docs: instructions
# are objects ("question" + "compare"/"inspect" + "focus") that reference state
# fields by backticked path, and every question has an explicit true/false test.
# The R_ideo gates carry a burden-of-proof focus ("false unless the text itself
# clearly shows it"): without it the model said yes far more often than the
# annotators (I_party 70 % vs 19 %). To iterate, edit this constant and re-run;
# the questions hash stored with every result identifies the version.
JEV_QUESTIONS = {
    "T_domain": {
        "type": "noul",
        "instructions": {
            "question": "Do the input text and the chunk belong to the same policy field, so that the chunk is part of the same area of debate?",
            "compare": [
                "`input_text`",
                "`retrieved_chunk.text`"
            ],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme."
        },
        "criteria": {
            "true": "Both belong to the same concrete policy field (for example both are about criminal law, the pandemic, climate and energy, migration, the armed forces, pensions, a named conflict), even if they discuss different measures within it.",
            "false": "They belong to different fields, or the only overlap is something generic: both are 'politics', both involve the same party or person, or they share a word. Money, taxes, costs or government spending count as a shared field only when the input text is itself about that subject (for example public debt, or who pays for state spending); a passing mention of costs in an input text about another issue does not. If the input text is about one concrete issue and the chunk about another (for example consulting contracts for a ministry vs. an excess-profit tax; protest forms vs. extremist organisations), answer false."
        }
    },
    "T_subarea": {
        "type": "noul",
        "instructions": {
            "question": "Do the input text and the chunk argue about the same narrow policy question?",
            "compare": [
                "`input_text`",
                "`retrieved_chunk.text`"
            ],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme."
        },
        "criteria": {
            "true": "The chunk's main subject is the same narrow policy question as the input text's: the same kind of decision or problem (for example how a specific cost is shared, whether a specific obligation should exist, how a specific programme is financed), even if the chunk concerns another detail, institution or year than the input text's own item.",
            "false": "The chunk's main subject is a different question that merely sits under the same umbrella theme (for example the input text is about paying for tests and the chunk is about a vaccination mandate; or both are about 'the military' but one is about equipment and the other about conscription), or it touches the shared theme only in passing."
        }
    },
    "T_item": {
        "type": "noul",
        "instructions": {
            "question": "Does the chunk discuss the specific item that the input text takes a position on?",
            "compare": [
                "`input_text`",
                "`retrieved_chunk.text`"
            ],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme."
        },
        "criteria": {
            "true": "The chunk addresses that same bill, measure, event, figure or named action - at least as one part of its content, from any angle and with any position.",
            "false": "The chunk does not discuss that item: it covers other measures, a more general question, or a different instance of the same kind of thing."
        }
    },
    "T_central": {
        "type": "noul",
        "instructions": {
            "question": "Is the chunk mainly about the same bill, event, claim or decision that the input text refers to, and does it speak to the point the input text makes?",
            "compare": [
                "`input_text`",
                "`retrieved_chunk.text`"
            ],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme."
        },
        "criteria": {
            "true": "The chunk's main subject is that same item - named or unmistakably identified - and it addresses the specific point or claim the input text makes about it (support or opposition does not matter). A summary of the chunk would name this item as its main subject.",
            "false": "The chunk covers the same wider issue but a different point (for example a vaccination mandate when the input text is about a statement on vaccination data), mentions the item only briefly, treats it as one of several topics in a broad speech, or is only the occasion of the debate."
        }
    },
    "I_procedural": {
        "type": "noul",
        "instructions": {
            "question": "Is the chunk only floor management or formalities?",
            "inspect": "`retrieved_chunk.text`"
        },
        "criteria": {
            "true": "The chunk consists of procedure only: addressing the chair, thanks, greetings, announcing agenda items or votes, with no substantive statement about policy.",
            "false": "The chunk contains substantive statements about policy, facts or political actors."
        }
    },
    "I_evaluative": {
        "type": "noul",
        "instructions": {
            "question": "Does the speaker argue, evaluate or demand something about policy?",
            "inspect": "`retrieved_chunk.text`"
        },
        "criteria": {
            "true": "The speaker argues for or against a policy, judges the actions of political actors, or makes demands.",
            "false": "The speaker only reports facts, procedures or the positions of others without evaluating or demanding anything."
        }
    },
    "I_stance": {
        "type": "noul",
        "instructions": {
            "question": "Does the speaker take a strong, partisan side on a contested political question?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Burden of proof: the answer is false unless the text itself clearly shows it. Answer true only if the passage leaves no reasonable doubt on this. If the passage is ambiguous, mixed or only partly fits, answer false."
        },
        "criteria": {
            "true": "The speaker takes a clear side with charged or polarising language, sweeping judgements of political opponents or of the government's record, or a programmatic demand typical of one political camp (for example market-liberal, welfare-state, nationalist, ecological, law-and-order). The partisanship is the point of the passage.",
            "false": "The speaker disagrees or criticises in a measured, technical or procedural way (rebutting a motion point by point, legal or fact-based reasoning, explaining or defending what the government did), expresses empathy or personal views, weighs pros and cons, or appeals for cross-party unity. Disagreement alone is not enough."
        }
    },
    "I_party": {
        "type": "noul",
        "instructions": {
            "question": "Does the speaker declare what their own party or faction stands for, demands or rejects?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Burden of proof: the answer is false unless the text itself clearly shows it. Answer true only if the passage leaves no reasonable doubt on this. If the passage is ambiguous, mixed or only partly fits, answer false."
        },
        "criteria": {
            "true": "Both hold: (1) the chunk text explicitly names the speaker's own party or faction as the holder of the position ('we as the Left', 'for us in the CDU/CSU', 'we, the AfD', 'my faction', 'our motion') and (2) it states a programmatic position of that party - what it demands, rejects, proposes or will do ('we reject this', 'we demand', 'we will vote against', 'our proposal is').",
            "false": "At least one fails. In particular these do NOT count: a generic 'we' (Germany, the Bundestag, everyone: 'we need a new mobility policy'); speaking as a minister or for the government; naming other parties only to attack or address them; a passing mention of one's own party inside an attack on others; a position that is merely typical of the party but never tied to it in the text."
        }
    },
    "A_caus": {
        "type": "noul",
        "instructions": {
            "question": "Did the generator justification demonstrably draw on this specific retrieved chunk?",
            "compare": [
                "`generator_justification`",
                "`retrieved_chunk`"
            ],
            "focus": "The justification refers to retrieved reference chunks by number, by speaker name, or by the party whose reference statements it invokes. Decide whether it refers to THIS chunk (see `retrieved_chunk.citation_forms`, `retrieved_chunk.speaker`, `retrieved_chunk.party`). If the justification cites other chunks by number but not this chunk's number or speaker, answer false even when this chunk's party is also named - a shared party name is then only context. A party named only as the camp the input text itself belongs to is not a reference to a chunk."
        },
        "criteria": {
            "true": "The justification refers to this chunk: it cites this chunk's number in any form (a single reference, a list such as '[2, 3]', 'reference [2]', 'anchor 2', 'Chunk 2'), OR it names this chunk's speaker, OR it invokes the statements or stance of this chunk's party as comparison evidence (e.g. 'echoing the SPD and FDP speakers in the reference anchors', 'similar to the AfD speakers') in a way that matches the content of this chunk.",
            "false": "The justification does not refer to this chunk: its number is not cited, its speaker is not named, and its party is not invoked as reference evidence (a party mentioned only as the label for the input text's own orientation does not count; neither does a party name that merely overlaps with chunks cited by other numbers), or it relies on other chunks or on generic reasoning that could have been produced from the input text and pre-training alone."
        }
    }
}


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
    """Minimal state for the content questions: the two texts and nothing else.

    No party, speaker, source, chunk identifier or generator justification, so
    the topical and ideological answers cannot lean on metadata.
    """
    return {
        "evaluation_task": CONTENT_TASK,
        "input_text": (record.get("input_text") or "").strip(),
        "retrieved_chunk": {"text": (chunk.get("chunk_text") or "").strip()},
    }


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
    converted to plain dicts so the raw values can be stored in the JSONL.
    """
    response = client.system_one(state=state, questions=questions, model=model)

    answers: dict[str, dict] = {}
    for name, question in questions.items():
        answer = response.answers.get(name)
        if answer is None or answer.type != question["type"]:
            raise RuntimeError(f"Missing or malformed answer for '{name}': {answer!r}")
        answers[name] = answer.model_dump()

    usage = response.usage.model_dump()
    # The native TypeSafe API reports tokens only; OpenRouter also reports cost.
    raw_usage = (response.raw_http_response.json() or {}).get("usage") or {}
    if raw_usage.get("cost") is not None:
        usage["cost"] = raw_usage["cost"]

    return {"answers": answers, "usage": usage, "model": response.model}


# ---------------------------------------------------------------------------
# Label derivation (raw answers -> integer labels on the human scales)
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


def derive_labels(raw_answers: dict, citation: dict, posthoc: dict | None = None) -> dict:
    """Turn the raw gate answers into integer labels on the human scales.

    R_top: 1 + the highest gate answered yes (domain, sub-area, item, central).

    R_ideo: 5 = own-party position and a strong stance; 4 = strong stance;
    3 = argues or evaluates; otherwise 1 (procedure only) or 2 (descriptive).

    ``posthoc`` (optional, off by default) replaces the R_top / R_ideo labels of
    the dimensions it contains with the fitted ordinal mapping.

    A_caus: citations are checked deterministically (see ``citation_info``), as
    the human labels behave. A chunk cited by number or whose speaker is named
    is 1; once other chunks are cited by number, an uncited chunk is 0;
    otherwise (no numeric citations) the model's probability decides.
    """
    yes = {
        gate: raw_answers[gate]["noul"] >= GATE_THRESHOLD
        for gate in (*R_TOP_GATES, *R_IDEO_GATES)
        if gate in raw_answers  # I_procedural is optional (never fires in the data)
    }
    yes.setdefault("I_procedural", False)
    labels = {
        "R_top": 1
        + max((i for i, gate in enumerate(R_TOP_GATES, start=1) if yes[gate]), default=0)
    }
    if yes["I_party"] and yes["I_stance"]:
        labels["R_ideo"] = 5
    elif yes["I_stance"] and yes["I_evaluative"]:
        labels["R_ideo"] = 4
    elif yes["I_evaluative"]:
        labels["R_ideo"] = 3
    else:
        labels["R_ideo"] = 1 if yes["I_procedural"] else 2

    if citation["cited_by_number"] or citation["speaker_named"]:
        labels["A_caus"] = 1
    elif citation["any_numeric_citation"]:
        labels["A_caus"] = 0
    else:
        labels["A_caus"] = int(raw_answers["A_caus"]["noul"] >= A_CAUS_THRESHOLD)
    for dim, spec in (posthoc or {}).items():
        labels[dim] = apply_posthoc(raw_answers, spec)
    return labels


def needs_review(raw_answers: dict, citation: dict) -> dict:
    """Per-dimension flag: should a human check this chunk's label?

    Selective escalation: the judge's label is trusted only where its gate
    answers are decisive and consistent. Gates absent from ``raw_answers`` (an
    older question set) are ignored.
    """

    def uncertain(gates: tuple[str, ...], band: float) -> bool:
        probs = [raw_answers[g]["noul"] for g in gates if g in raw_answers]
        if any(abs(p - GATE_THRESHOLD) < band for p in probs):
            return True
        answered = [p >= GATE_THRESHOLD for p in probs]
        # inconsistent ladder: some higher gate yes after a lower gate no
        return any(not lo and any(answered[i + 1 :]) for i, lo in enumerate(answered))

    if (
        citation["cited_by_number"]
        or citation["speaker_named"]
        or citation["any_numeric_citation"]
    ):
        a_caus = False  # decided deterministically from the citations
    else:
        a_caus = (
            abs(raw_answers["A_caus"]["noul"] - A_CAUS_THRESHOLD)
            < REVIEW_BANDS["A_caus"]
        )
    flags = {
        "R_top": uncertain(R_TOP_GATES, REVIEW_BANDS["R_top"]),
        "R_ideo": uncertain(R_IDEO_LADDER, REVIEW_BANDS["R_ideo"]),
        "A_caus": a_caus,
    }
    flags["any"] = any(flags.values())
    return flags


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_judging(
    args: argparse.Namespace, run_dir: Path
) -> tuple[list[dict], int, str]:
    """Load annotations, judge chunks with Jev, write snapshots, return results."""
    records = load_existing_annotations(args.annotations)
    if not records:
        print("No annotations found. Exiting.", file=sys.stderr)
        sys.exit(1)

    tasks = [
        (record, chunk)
        for record in records.values()
        for chunk in record.get("chunk_annotations", [])
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    questions = load_questions(args.questions_file)
    questions_hash = hash_questions(questions)
    posthoc = load_posthoc(args.posthoc_model)
    client, args.model = build_client(args.model)

    print(f"Judge model: {args.model}")
    print(f"Questions hash: {questions_hash}")
    print(f"State mode: {args.state_mode}")
    print(f"Judging {len(tasks)} chunk(s) at concurrency {args.concurrency}")

    def _process(record: dict, chunk: dict) -> dict:
        result = {
            "text_index": record.get("text_index"),
            "chunk_index": chunk.get("chunk_index"),
            "model": args.model,
            "questions_hash": questions_hash,
            "state_mode": args.state_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "human": {dim: chunk.get(dim) for dim in DIMENSIONS},
        }
        try:
            state = build_state(record, chunk)
            if args.state_mode in ("minimal", "nometa"):
                # content questions get a reduced state; A_caus keeps the full
                # state (it needs the justification and chunk metadata)
                content = {k: q for k, q in questions.items() if k != "A_caus"}
                content_state = (
                    build_content_state(record, chunk)
                    if args.state_mode == "minimal"
                    else build_nometa_state(record, chunk)
                )
                judged = judge_chunk(client, content_state, content, args.model)
                if "A_caus" in questions:
                    extra = judge_chunk(
                        client, state, {"A_caus": questions["A_caus"]}, args.model
                    )
                    judged["answers"].update(extra["answers"])
                    judged["usage"] = {
                        k: (judged["usage"].get(k) or 0) + (extra["usage"].get(k) or 0)
                        for k in judged["usage"]
                    }
            else:
                judged = judge_chunk(client, state, questions, args.model)
            citation = citation_info(
                state["generator_justification"],
                chunk.get("chunk_index", 0) + 1,
                state["retrieved_chunk"]["speaker"],
            )
            result.update(
                response_model=judged["model"],
                raw_answers=judged["answers"],
                citation=citation,
                derived_labels=derive_labels(judged["answers"], citation, posthoc),
                uncalibrated_labels=derive_labels(judged["answers"], citation) if posthoc else None,
                needs_review=needs_review(judged["answers"], citation),
                usage=judged["usage"],
            )
        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_process, r, c) for r, c in tasks]
        results = [
            f.result()
            for f in tqdm(as_completed(futures), total=len(futures), desc="Judging")
        ]
    failed = sum("error" in r for r in results)

    write_snapshots(results, records, args.model, run_dir)
    client.close()
    return results, failed, questions_hash


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


def load_posthoc(path: str | None) -> dict | None:
    """Fitted ordinal mapping(s) written by ``calibrate.py --export-posthoc``."""
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_questions(questions: dict) -> str:
    return hashlib.sha256(
        json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def load_questions(path: str | None) -> dict:
    """The built-in ``JEV_QUESTIONS`` or a candidate set from a JSON file."""
    if not path:
        return JEV_QUESTIONS
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {*R_TOP_GATES, *R_IDEO_LADDER, "A_caus"}
    missing = required - set(questions)
    if missing:
        raise SystemExit(f"--questions-file is missing question(s): {sorted(missing)}")
    return questions


def rederive_results(
    path: str, records: dict, posthoc: dict | None = None
) -> tuple[list[dict], str]:
    """Re-apply the current label rules to a stored run, without API calls.

    Takes the flat JSONL of an earlier run (raw gate probabilities and citation
    evidence are stored per chunk) and recomputes ``derived_labels`` and
    ``needs_review`` with the present ``derive_labels`` / ``needs_review``.
    """
    human = {
        (text_index, chunk.get("chunk_index")): {d: chunk.get(d) for d in DIMENSIONS}
        for text_index, record in records.items()
        for chunk in record.get("chunk_annotations", [])
    }
    results, hashes = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hashes.add(row.get("questions_hash"))
        if "error" not in row:
            row["derived_labels"] = derive_labels(row["raw_answers"], row["citation"], posthoc)
            if posthoc:
                row["uncalibrated_labels"] = derive_labels(row["raw_answers"], row["citation"])
            row["needs_review"] = needs_review(row["raw_answers"], row["citation"])
            row["human"] = human.get((row["text_index"], row["chunk_index"]), row["human"])
        results.append(row)
    return results, "+".join(sorted(h for h in hashes if h)) or "unknown"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
                "human_labels": res.get("human"),
                "response_model": res.get("response_model"),
                "raw_answers": res.get("raw_answers"),
                "citation": res.get("citation"),
                "needs_review": res.get("needs_review"),
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


def _safe_kappa(y1, y2, labels, weights=None):
    """Compute Cohen's kappa, handling the all-agree edge case."""
    k = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    if np.isnan(k):
        return 1.0 if len(y1) > 0 and np.all(y1 == y2) else None
    return float(k)


def compute_metrics(results: list[dict]) -> dict:
    """Compute agreement metrics per dimension on the derived labels.

    A dimension without reference labels (unannotated chunks) maps to None.
    """
    per_dim = {}
    for dim in DIMENSIONS:
        ordinal = dim in ORDINAL_DIMENSIONS
        labels = [1, 2, 3, 4, 5] if ordinal else [0, 1]
        h_vals, j_vals = [], []
        for r in results:
            if "error" in r:
                continue
            h = (r.get("human") or {}).get(dim)
            j = r["derived_labels"].get(dim)
            if j is None or h is None:
                continue
            h_vals.append(int(h))
            j_vals.append(int(j))

        if not h_vals:
            per_dim[dim] = None
            continue

        h_arr, j_arr = np.array(h_vals), np.array(j_vals)
        kappa = _safe_kappa(
            h_arr, j_arr, labels, weights="quadratic" if ordinal else None
        )
        per_dim[dim] = {
            "n": len(h_arr),
            "exact_agreement": float(np.mean(h_arr == j_arr)),
            "qwk": kappa if ordinal else None,
            "kappa": None if ordinal else kappa,
            "mean_diff": float(np.mean(j_arr - h_arr)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(h_arr, j_arr, labels=labels).tolist(),
            "below_threshold": kappa is not None and kappa < KAPPA_TARGET,
        }
    return per_dim


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
    disagreements.sort(key=lambda x: (-x["diff"], x["text_index"], x["chunk_index"]))
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
# Report
# ---------------------------------------------------------------------------


def print_and_save_report(
    results: list[dict],
    per_dim: dict,
    worst: list[dict],
    model: str,
    questions_hash: str,
    report_path: Path,
) -> None:
    """Print a concise report to stdout and write a JSON report to disk."""
    judged = sum("derived_labels" in r for r in results)
    usage_totals = compute_usage_totals(results)

    print("\n=== Jev Judge Agreement Report ===")
    print(f"Model: {model}")
    print(f"Questions hash: {questions_hash}")
    print(f"Judged chunks: {judged} / {len(results)}")

    print("\nPer-dimension metrics:")
    header = f"{'Dim':<12} {'n':>6} {'Agree%':>7} {'κ/κ_w':>7} {'MeanΔ':>7}"
    print(header)
    print("-" * len(header))
    for dim in DIMENSIONS:
        d = per_dim[dim]
        if d is None:
            print(f"{dim:<12} {'—':>6} {'—':>7} {'—':>7} {'—':>7}")
            continue
        metric = d["qwk"] if dim in ORDINAL_DIMENSIONS else d["kappa"]
        metric_str = f"{metric:>6.3f}" if metric is not None else "   N/A"
        flag = f" ⚠️ below {KAPPA_TARGET:.2f}" if d["below_threshold"] else ""
        print(
            f"{dim:<12} {d['n']:>6} {d['exact_agreement']*100:>6.1f}% "
            f"{metric_str} {d['mean_diff']:>+6.2f}{flag}"
        )

    print("\nConfusion matrices (rows: reference, columns: judge):")
    for dim in DIMENSIONS:
        d = per_dim[dim]
        if d is None:
            continue
        print(f"\n{dim} (labels: {d['labels']})")
        for row, label in zip(d["confusion_matrix"], d["labels"]):
            print(f"  {label}: {row}")

    if worst:
        print(f"\nTop {len(worst)} disagreements:")
        for wd in worst:
            print(
                f"  sample {wd['text_index']} chunk {wd['chunk_index']} "
                f"{wd['dimension']}: human={wd['human']} judge={wd['judge']} "
                f"diff={wd['diff']}"
            )

    review_share = {
        key: float(np.mean([r["needs_review"][key] for r in results if "needs_review" in r]))
        for key in (*DIMENSIONS, "any")
        if any("needs_review" in r for r in results)
    }
    if review_share:
        shares = ", ".join(f"{k} {v*100:.0f}%" for k, v in review_share.items())
        print(f"\nRouted to human review (uncertain or inconsistent gates): {shares}")

    print(
        f"\nUsage: {usage_totals['input_tokens']:,} input tokens, "
        f"{usage_totals['output_tokens']:,} output tokens (free), "
        f"total cost ${usage_totals['cost']:.6f}"
    )

    report = {
        "model": model,
        "questions_hash": questions_hash,
        "a_caus_threshold": A_CAUS_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(results),
        "judged_chunks": judged,
        "usage_totals": usage_totals,
        "per_dimension": per_dim,
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
        "--questions-file",
        default=None,
        help="JSON file with a candidate question set (same shape as "
        "JEV_QUESTIONS) instead of the built-in one",
    )
    parser.add_argument(
        "--state-mode",
        choices=("full", "minimal", "nometa"),
        default="nometa",
        help="nometa (default): the content questions (R_top / R_ideo) see the "
        "full state without the chunk's party / speaker / source, as the human "
        "annotators were told to ignore speaker metadata. full: every question "
        "sees the whole state. minimal: the content questions see only "
        "input_text and retrieved_chunk.text. A_caus always gets the full state "
        "(two requests per chunk in both reduced modes)",
    )
    parser.add_argument(
        "--posthoc-model",
        default=None,
        help="JSON from calibrate.py --export-posthoc: map the gate probabilities "
        "to R_top / R_ideo with a mapping fitted on the calibration set "
        "(default: off, labels are read straight off the gate answers)",
    )
    parser.add_argument(
        "--rederive-from",
        default=None,
        help="Flat llm_judge_annotations.jsonl of an earlier run: re-apply the "
        "current label rules to its stored gate probabilities (no API calls)",
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


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    print(f"Output directory: {run_dir.resolve()}")

    if args.rederive_from:
        records = load_existing_annotations(args.annotations)
        results, questions_hash = rederive_results(
            args.rederive_from, records, load_posthoc(args.posthoc_model)
        )
        args.model = next((r["model"] for r in results if r.get("model")), "unknown")
        failed = sum("error" in r for r in results)
        write_snapshots(results, records, args.model, run_dir)
    else:
        results, failed, questions_hash = run_judging(args, run_dir)
    print_and_save_report(
        results,
        compute_metrics(results),
        get_worst_disagreements(results, n=args.show_disagreements),
        args.model,
        questions_hash,
        run_dir / REPORT_NAME,
    )
    if failed:
        print(f"\nWarning: {failed} chunk(s) failed judging.")


if __name__ == "__main__":
    main()
