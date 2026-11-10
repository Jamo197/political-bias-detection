"""Annotation app for retrieved RAG chunks (R_top, R_ideo, A_caus).

Two modes:

* annotation (default): a seeded random sample of generator results, rated
  from scratch;
* review: the output of the LLM judge (``run_llm_judge.py --samples``) is
  loaded and a fixed selection of texts is shown: every Q1 (RAG win) and Q2
  (RAG distraction) text and 10 random texts each of Q3 and Q4
  (``REVIEW_QUOTAS``, seeded per folder, stored in review_selection.json). Flagged
  dimensions are marked and pre-filled with the judge's label; spot-check
  chunks (a random 5 %) are rated blind, so their agreement with the judge
  estimates the error of the labels nobody reviewed. The reviewed records go
  to their own file and are merged with the judge labels by
  ``merge_judge_review.py``.

    streamlit run src/run_streamlit.py
    streamlit run src/run_streamlit.py -- --review "RAG Analysis/LLM judge results/<run>/llama_70b"
    streamlit run src/run_streamlit.py -- --review <folder> --annotator Jannes --all
    streamlit run src/run_streamlit.py -- --review <folder> --quotas Q1=all,Q2=all,Q3=5,Q4=5
"""

import argparse
import datetime
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

DEFAULT_ANNOTATOR = "Human Annotator"
APP_ROOT = Path(__file__).resolve().parent.parent
QUALITATIVE_SAMPLE_PATH = APP_ROOT / "results/qualitative/llama/8b/qualitative_e5_simple.jsonl"
ANNOTATION_LOG_PATH = APP_ROOT / "annotations/qualitative_annotations.jsonl"
ANNOTATION_SAMPLE_LIMIT = 25
ANNOTATION_SEED = 42
JUDGE_GROUPED_NAME = "llm_judge_qualitative_annotations.jsonl"
REVIEWED_NAME = "reviewed_annotations.jsonl"
SELECTION_NAME = "review_selection.json"
# Texts reviewed per quadrant (None = all): the RAG wins and distractions in
# full, a random sample of the rest. The texts not selected keep the judge labels.
REVIEW_QUOTAS = {"Q1": None, "Q2": None, "Q3": 10, "Q4": 10}
REVIEW_SEED = 42
DIMENSIONS = ("R_top", "R_ideo", "A_caus")

SESSION_STATE_DEFAULTS = {
    "annotation_samples": [],
    "annotation_index": 0,
    "annotation_seed": ANNOTATION_SEED,
    "annotation_annotator": DEFAULT_ANNOTATOR,
    "annotation_sample_notes": "",
    "annotation_log_path": ANNOTATION_LOG_PATH,
    "chunk_annotations": {},  # Maps chunk_index -> {R_top, R_ideo, A_caus, notes}
    "existing_annotations": {},  # Maps text_index -> latest saved annotation record
    "chunk_annotations_loaded_for": None,  # text_index of the sample currently in chunk state
    "acaus_auto_detected": set(),  # Chunk indices where A_caus was auto-set from the RAG justification
    "review_mode": None,  # {"judge_path", "show_all"} when reviewing judge output
}

# Rubric descriptions from RAG Analysis/RAG Analysis.md
R_TOP_OPTIONS = {
    1: "IRRELEVANT: Different domain and issue entirely.",
    2: "BROAD DOMAIN ONLY: Shares high-level domain, but addresses a different policy.",
    3: "RELATED SUB-ISSUE: Same policy area and mechanism, but different target/context.",
    4: "DIRECT POLICY OVERLAP: Same exact policy debate, differing only in minor scope.",
    5: "IDENTICAL TARGET: Exact entity, legislation, or policy mechanism.",
}

R_IDEO_OPTIONS = {
    1: "DESCRIPTIVE / PROCEDURAL: Purely administrative, factual, or neutral metrics.",
    2: "BALANCED OVERVIEW: Mentions political controversy but gives equal weight/neutral tone.",
    3: "IMPLICIT VALUE FRAMING: Uses biased terminology or selective facts without naming actors.",
    4: "CLEAR IDEOLOGICAL STANCE: Unambiguous ideological orientation (e.g., social democratic, libertarian).",
    5: "EXPLICIT PARTY / MANIFESTO GROUNDING: Cites specific party doctrine, voting positions, or platforms.",
}


@st.cache_data
def load_qualitative_samples(
    path: str | Path = QUALITATIVE_SAMPLE_PATH,
) -> List[Dict[str, Any]]:
    """Load qualitative samples from JSONL file."""
    sample_path = Path(path).expanduser().resolve()
    if not sample_path.exists():
        st.error(f"Qualitative sample file not found: {sample_path}")
        return []

    rows: List[Dict[str, Any]] = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
    return rows


def load_existing_annotations(
    path: str | Path = ANNOTATION_LOG_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Load existing annotation records; latest record per text_index wins."""
    annotations_path = Path(path).expanduser().resolve()
    if not annotations_path.exists():
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text_index = record.get("text_index")
            if text_index is not None:
                records[text_index] = record
    return records


def select_random_samples(
    rows: List[Dict[str, Any]], limit: int = 25, seed: int = 42
) -> List[Dict[str, Any]]:
    """Select a random subset of samples with deterministic seeding."""
    if not rows:
        return []
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(rows) <= limit:
        sample_rows = list(rows)
    else:
        sample_rows = random.Random(seed).sample(rows, limit)
    return sample_rows


def build_annotation_record(
    sample: Dict[str, Any],
    chunk_annotations: List[Dict[str, Any]],
    annotator: str,
    sample_notes: str = "",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a structured annotation record with per-chunk rubric responses.

    Parameters
    ----------
    sample : dict
        The original sample with input_text, metadata, and rag.retrieved_chunks.
    chunk_annotations : list
        List of dicts, one per chunk, each containing {chunk_index, chunk_text, chunk_metadata, R_top, R_ideo, A_caus, notes, timestamp}.
    annotator : str
        Name or ID of the annotator.
    sample_notes : str
        Optional global notes for the entire sample.
    timestamp : str, optional
        ISO timestamp; defaults to now.

    Returns
    -------
    dict
        Record with text_index, input_text, source_context, chunk_annotations list, annotator, and sample_notes.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    source_context = {
        "text_index": sample.get("text_index"),
        "quadrant": sample.get("quadrant"),
        "input_metadata": sample.get("input_metadata", {}),
        "ground_truth": sample.get("ground_truth", {}),
        "metrics": sample.get("metrics", {}),
        "baseline": sample.get("baseline", {}),
        "rag": sample.get("rag", {}),
    }
    sample_manifest = {
        "seed": ANNOTATION_SEED,
        "sample_limit": ANNOTATION_SAMPLE_LIMIT,
        "text_index": sample.get("text_index"),
    }

    return {
        "text_index": sample.get("text_index"),
        "input_text": sample.get("input_text", ""),
        "source_context": source_context,
        "sample_manifest": sample_manifest,
        "chunk_annotations": chunk_annotations,
        "annotator": annotator or "",
        "sample_notes": sample_notes or "",
        "timestamp": timestamp,
    }


def save_annotation_record(
    record: Dict[str, Any],
    output_path: str | Path = ANNOTATION_LOG_PATH,
) -> Path:
    """Save one latest annotation record per text index.

    Existing records are deduplicated by text index and the supplied record
    replaces the record for its sample. The replacement is atomic so a
    browser refresh or process interruption does not leave a partial file.
    """
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = load_existing_annotations(output_file)
    text_index = record.get("text_index")
    if text_index is None:
        raise ValueError("Annotation record must contain a text_index")
    records[text_index] = record

    temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
    with temporary_file.open("w", encoding="utf-8") as handle:
        for saved_record in records.values():
            handle.write(json.dumps(saved_record, ensure_ascii=False) + "\n")
    temporary_file.replace(output_file)
    return output_file


def detect_cited_chunks(justification: str, n_chunks: int) -> set:
    """Detect 1-based [N] chunk references in the RAG justification.

    Returns a set of 0-based chunk indices that were cited, e.g. the phrase
    "as evident from the reference anchor [2]" cites chunk index 1.
    """
    if not justification or n_chunks <= 0:
        return set()
    cited = set()
    for match in re.finditer(r"\[(\d+)\]", justification):
        ref = int(match.group(1))
        if 1 <= ref <= n_chunks:
            cited.add(ref - 1)
    return cited


def parse_app_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Arguments after ``--`` on the streamlit command line."""
    parser = argparse.ArgumentParser(prog="run_streamlit.py", add_help=False)
    parser.add_argument("--review", default=None,
                        help="judge run folder (or its llm_judge_qualitative_annotations.jsonl)")
    parser.add_argument("--out", default=None,
                        help="reviewed records (default: reviewed_annotations.jsonl next to the judge file)")
    parser.add_argument("--annotator", default=None)
    parser.add_argument("--all", action="store_true",
                        help="review mode: show every text instead of the quadrant selection")
    parser.add_argument("--quotas", default=None,
                        help="review mode: texts per quadrant, e.g. Q1=all,Q2=all,Q3=10,Q4=10")
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args


def resolve_judge_path(path: str | Path) -> Path:
    judge_path = Path(path).expanduser().resolve()
    if judge_path.is_dir():
        judge_path = judge_path / JUDGE_GROUPED_NAME
    if not judge_path.exists():
        raise FileNotFoundError(f"Judge output not found: {judge_path}")
    return judge_path


def chunk_needs_human(chunk: Dict[str, Any]) -> bool:
    return bool((chunk.get("needs_review") or {}).get("any") or chunk.get("spot_check"))


def parse_quotas(spec: Optional[str]) -> Dict[str, Optional[int]]:
    """'Q1=all,Q3=10' -> {'Q1': None, 'Q3': 10}; None -> REVIEW_QUOTAS."""
    if not spec:
        return dict(REVIEW_QUOTAS)
    quotas: Dict[str, Optional[int]] = {}
    for part in spec.split(","):
        quadrant, _, value = part.strip().partition("=")
        quotas[quadrant.strip()] = None if value.strip() in ("", "all") else int(value)
    return quotas


def quadrant_of(record: Dict[str, Any]) -> str:
    """'Q1_RAG_Win' -> 'Q1'."""
    quadrant = (record.get("source_context") or {}).get("quadrant") or record.get("quadrant") or ""
    return quadrant.split("_")[0] or "unknown"


def select_review_texts(
    records: Dict[str, Dict[str, Any]],
    quotas: Dict[str, Optional[int]],
    salt: str,
    seed: int = REVIEW_SEED,
) -> List[str]:
    """Text ids to review: per quadrant all (quota None) or a seeded random sample.

    ``salt`` (the folder name) gives every model family and size its own draw.
    Quadrants without a quota are skipped. Ordered by quadrant, then file order.
    """
    by_quadrant: Dict[str, List[str]] = {}
    for text_index, record in records.items():
        by_quadrant.setdefault(quadrant_of(record), []).append(text_index)
    selected = []
    for quadrant in sorted(by_quadrant):
        if quadrant not in quotas:
            continue
        ids, quota = by_quadrant[quadrant], quotas[quadrant]
        if quota is not None and quota < len(ids):
            chosen = set(random.Random(f"{seed}|{salt}|{quadrant}").sample(sorted(ids), quota))
            ids = [t for t in ids if t in chosen]
        selected.extend(ids)
    return selected


def write_review_selection(judge_path: Path, quotas: Dict[str, Optional[int]], show_all: bool) -> List[str]:
    """Select the texts to review and store the choice next to the judge output.

    An existing selection with the same quotas is reused, so the set stays
    fixed even if the selection rule changes later.
    """
    records = load_existing_annotations(judge_path)
    path = judge_path.parent / SELECTION_NAME
    if show_all:
        return list(records)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("quotas") == quotas:
            return saved["text_indices"]
    selected = select_review_texts(records, quotas, judge_path.parent.name)
    counts: Dict[str, int] = {}
    for t in selected:
        counts[quadrant_of(records[t])] = counts.get(quadrant_of(records[t]), 0) + 1
    path.write_text(
        json.dumps(
            {"quotas": quotas, "seed": REVIEW_SEED, "salt": judge_path.parent.name,
             "per_quadrant": counts, "text_indices": selected},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def load_review_samples(judge_path: Path, text_indices: List[str]) -> List[Dict[str, Any]]:
    """Judge records of the selected texts as app samples, in selection order.

    Each sample is the original generator result (``source_context``) plus the
    judge's chunk annotations under ``judge_chunks`` (chunk_index -> entry).
    """
    records = load_existing_annotations(judge_path)
    samples = []
    for text_index in text_indices:
        record = records[text_index]
        judge_chunks = {
            c["chunk_index"]: c for c in record.get("chunk_annotations", [])
        }
        sample = dict(record.get("source_context") or {})
        sample["text_index"] = record.get("text_index")
        sample["input_text"] = record.get("input_text", "")
        sample["judge_chunks"] = judge_chunks
        samples.append(sample)
    return samples


def judge_flags(judge_chunk: Dict[str, Any]) -> List[str]:
    review = judge_chunk.get("needs_review") or {}
    return [dim for dim in DIMENSIONS if review.get(dim)]


def load_chunk_state_for_sample(sample: Dict[str, Any]):
    """Populate chunk annotation state for the given sample.

    Precedence per chunk: existing annotation record > judge labels (review
    mode; empty for spot-check chunks, which are rated blind) > auto-detected
    A_caus (chunk cited as [N] in the RAG justification) > static defaults.
    Also pre-fills sample notes and syncs widget keys so pre-filled values
    render correctly.
    """
    text_index = sample.get("text_index")
    retrieved_chunks = sample.get("rag", {}).get("retrieved_chunks", [])
    existing = st.session_state.get("existing_annotations", {}).get(text_index, {})
    existing_by_idx = {
        entry.get("chunk_index"): entry
        for entry in existing.get("chunk_annotations", [])
        if entry.get("chunk_index") is not None
    }

    cited = detect_cited_chunks(
        sample.get("rag", {}).get("justification", "") or "",
        len(retrieved_chunks),
    )

    judge_chunks = sample.get("judge_chunks") or {}

    chunk_annotations: Dict[int, Dict[str, Any]] = {}
    auto_detected = set()
    for chunk_idx in range(len(retrieved_chunks)):
        prev = existing_by_idx.get(chunk_idx)
        judged = judge_chunks.get(chunk_idx)
        if prev is not None:
            state = {
                "R_top": prev.get("R_top", 3),
                "R_ideo": prev.get("R_ideo", 3),
                "A_caus": prev.get("A_caus", 0),
                "notes": prev.get("notes", ""),
            }
        elif judged is not None:
            blind = bool(judged.get("spot_check"))
            state = {
                dim: None if blind else judged.get(dim) for dim in DIMENSIONS
            }
            state["notes"] = ""
        else:
            state = {
                "R_top": 3,
                "R_ideo": 3,
                "A_caus": 1 if chunk_idx in cited else 0,
                "notes": "",
            }
            if chunk_idx in cited:
                auto_detected.add(chunk_idx)
        chunk_annotations[chunk_idx] = state

        # Sync widget keys so the pre-filled values render correctly
        st.session_state[f"chunk_{chunk_idx}_rtop"] = state["R_top"]
        st.session_state[f"chunk_{chunk_idx}_rideo"] = state["R_ideo"]
        st.session_state[f"chunk_{chunk_idx}_acaus"] = state["A_caus"]
        st.session_state[f"chunk_{chunk_idx}_notes"] = state["notes"]

    st.session_state["chunk_annotations"] = chunk_annotations
    st.session_state["acaus_auto_detected"] = auto_detected
    st.session_state["annotation_sample_notes"] = existing.get("sample_notes", "")
    st.session_state["chunk_annotations_loaded_for"] = text_index


def init_session_state():
    """Initialize Streamlit session state with default values."""
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Always reload annotations from disk so the view stays in sync with the file
    st.session_state["existing_annotations"] = load_existing_annotations(
        st.session_state.get("annotation_log_path", ANNOTATION_LOG_PATH)
    )

    review = st.session_state.get("review_mode")
    if review and not st.session_state.get("annotation_samples"):
        judge_path = Path(review["judge_path"])
        st.session_state["annotation_samples"] = load_review_samples(
            judge_path,
            write_review_selection(judge_path, review["quotas"], review["show_all"]),
        )
    elif not st.session_state.get("annotation_samples"):
        rows = load_qualitative_samples()
        st.session_state["annotation_samples"] = select_random_samples(
            rows,
            limit=ANNOTATION_SAMPLE_LIMIT,
            seed=st.session_state.get("annotation_seed", ANNOTATION_SEED),
        )


def render_chunk_form(chunk_index: int, chunk: Dict[str, Any]):
    """Render a per-chunk annotation form with 4 rubric dimensions.

    Updates session state chunk_annotations[chunk_index] directly.
    """
    chunk_text = chunk.get("text", "")
    chunk_metadata = {
        "party": chunk.get("party", "—"),
        "speaker": chunk.get("speaker", "—"),
        "source": chunk.get("source", "—"),
        "score": chunk.get("score", 0.0),
    }

    # Initialize chunk annotation state if not present
    if chunk_index not in st.session_state["chunk_annotations"]:
        st.session_state["chunk_annotations"][chunk_index] = {
            "R_top": 3,
            "R_ideo": 3,
            "A_caus": 0,
            "notes": "",
        }

    chunk_state = st.session_state["chunk_annotations"][chunk_index]
    judged = (sample_judge_chunks() or {}).get(chunk_index)
    flags = judge_flags(judged) if judged else []
    if judged and judged.get("spot_check"):
        st.info(
            "Spot check: rate all three dimensions from scratch (the judge's labels are "
            "hidden; this sample measures the judge's error on unreviewed chunks)."
        )
    elif flags:
        st.warning(
            "Judge uncertain on **" + ", ".join(flags) + "**; check these. "
            "The other dimensions are the judge's label and are kept unless you change them."
        )
    elif judged:
        st.caption("Not flagged: the judge's labels are kept unless you change them.")
    mark = lambda dim: " ⚠ review" if dim in flags else ""  # noqa: E731

    # Display chunk metadata
    meta_cols = st.columns([1, 1, 1, 1])
    meta_cols[0].caption(f"Party: **{chunk_metadata['party']}**")
    meta_cols[1].caption(f"Speaker: **{chunk_metadata['speaker']}**")
    meta_cols[2].caption(f"Source: **{chunk_metadata['source']}**")
    meta_cols[3].caption(f"Score: **{chunk_metadata['score']:.4f}**")

    # Display chunk text
    st.write(chunk_text)

    st.divider()

    # Rubric form — Row 1: Likert scales with full rubric descriptions
    # Widget values are managed via session state keys (pre-filled by
    # load_chunk_state_for_sample), so no index/value params are passed here.
    likert_cols = st.columns([1, 1])

    r_top = likert_cols[0].selectbox(
        "R_top (Topical Relevance)" + mark("R_top"),
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"[{x}] {R_TOP_OPTIONS[x]}",
        help="Does the chunk discuss the exact policy issue, entity, or debate present in the input text?",
        key=f"chunk_{chunk_index}_rtop",
    )
    chunk_state["R_top"] = r_top

    r_ideo = likert_cols[1].selectbox(
        "R_ideo (Ideological Specificity)" + mark("R_ideo"),
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"[{x}] {R_IDEO_OPTIONS[x]}",
        help="Does the chunk provide unambiguous grounding for how a specific political party, faction, or ideology views this topic?",
        key=f"chunk_{chunk_index}_rideo",
    )
    chunk_state["R_ideo"] = r_ideo

    # Rubric form — binary attribution scale
    a_caus = st.radio(
        "A_caus (Attribution)" + mark("A_caus"),
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Did the model rely on this chunk in its reasoning?",
        key=f"chunk_{chunk_index}_acaus",
    )
    chunk_state["A_caus"] = a_caus

    if chunk_index in st.session_state.get("acaus_auto_detected", set()):
        st.caption(
            f"Auto-set: chunk cited as [{chunk_index + 1}] in the RAG justification."
        )

    # Chunk-specific notes
    chunk_notes = st.text_area(
        "Chunk notes",
        key=f"chunk_{chunk_index}_notes",
        height=60,
    )
    chunk_state["notes"] = chunk_notes


def sample_judge_chunks() -> Dict[int, Dict[str, Any]]:
    """Judge chunk entries of the sample on screen (review mode), else {}."""
    rows = st.session_state.get("annotation_samples", [])
    index = int(st.session_state.get("annotation_index", 0))
    return rows[index].get("judge_chunks", {}) if rows else {}


def render_annotation_section():
    """Render the qualitative annotation interface with per-chunk rubric forms."""
    st.title("Qualitative Annotation Workspace")
    st.caption(
        "Annotate retrieved chunks using the 3-dimensional rubric (R_top, R_ideo, A_caus)."
    )

    rows = st.session_state.get("annotation_samples", [])
    index = int(st.session_state.get("annotation_index", 0))
    if not rows:
        st.warning("No qualitative data available yet.")
        return

    sample = rows[index]
    metadata = sample.get("input_metadata", {})
    quadrant = sample.get("quadrant", "unknown")

    # Pre-fill annotation state when the displayed sample changes
    sample_id = sample.get("text_index")
    if st.session_state.get("chunk_annotations_loaded_for") != sample_id:
        load_chunk_state_for_sample(sample)

    st.subheader(f"Sample {index + 1} / {len(rows)} — {quadrant}")
    review = st.session_state.get("review_mode")
    if review:
        done = sum(
            1 for r in rows if r.get("text_index") in st.session_state.get("existing_annotations", {})
        )
        st.caption(
            f"Review mode: {Path(review['judge_path']).parent.name} ({quadrant_of(sample)}) | "
            f"{done} / {len(rows)} texts reviewed | saving to {st.session_state['annotation_log_path']}"
        )

    if sample_id in st.session_state.get("existing_annotations", {}):
        st.info("Previously annotated — values loaded from file.")

    # ===== SOURCE TEXT SECTION =====
    st.write("## Source Text")
    source_meta_cols = st.columns(4)
    source_meta_cols[0].caption(f"Party: **{metadata.get('party', '—')}**")
    source_meta_cols[1].caption(f"Speaker: **{metadata.get('speaker', '—')}**")
    source_meta_cols[2].caption(f"Source: **{metadata.get('source', '—')}**")
    source_meta_cols[3].caption(f"Text ID: **{sample.get('text_index', '—')}**")

    st.write(sample.get("input_text", ""))

    # ===== PREDICTIONS & JUSTIFICATIONS SECTION =====
    st.write("## Predictions & Justifications")
    metrics = sample.get("metrics", {})
    ground_truth = sample.get("ground_truth", {}).get("label_ideology")

    pred_cols = st.columns(6)
    pred_cols[0].metric(
        "Ground Truth",
        f"{ground_truth:.1f}" if ground_truth is not None else "—",
        help="The actual value of the party ideological stance.",
    )
    pred_cols[1].metric(
        "Baseline (No RAG)",
        f"{metrics.get('base_prediction', 0):.1f}",
        help="What the LLM predicted without retrieved context.",
    )
    pred_cols[2].metric(
        "RAG Prediction",
        f"{metrics.get('rag_prediction', 0):.1f}",
        help="What the LLM predicted with retrieved context.",
    )
    pred_cols[3].metric(
        "Base Error",
        f"{metrics.get('base_error', 0):.1f}",
        help="The absolute error of the prediction without RAG.",
    )
    pred_cols[4].metric(
        "RAG Error",
        f"{metrics.get('rag_error', 0):.1f}",
        help="The absolute error of the prediction with RAG.",
    )
    pred_cols[5].metric(
        "Directional Shift",
        f"{metrics.get('directional_shift', 0):+.1f}",
        help="How the prediction moved: minus means more left; plus means more right.",
    )
    st.info(
        "**Metric guide:** Ground Truth is the actual party ideological stance value. "
        "Baseline (No RAG) is what the LLM predicted without retrieved context. "
        "RAG Prediction is what it predicted with retrieved context. Base Error is "
        "the absolute error without RAG; RAG Error is the absolute error with RAG."
    )

    just_cols = st.columns(2)
    with just_cols[0].expander("Baseline Justification (No RAG)", expanded=False):
        st.caption("The justification the LLM returned for its prediction without RAG.")
        st.write(sample.get("baseline", {}).get("justification", "—"))
    with just_cols[1].expander("RAG Justification", expanded=False):
        st.caption(
            "The justification returned with RAG, including anchor points showing "
            "which retrieved chunks the LLM used or found helpful."
        )
        st.write(sample.get("rag", {}).get("justification", "—"))

    st.divider()

    # ===== RETRIEVED CHUNKS SECTION =====
    st.write("## Retrieved Chunks")
    retrieved_chunks = sample.get("rag", {}).get("retrieved_chunks", [])

    if not retrieved_chunks:
        st.info("No retrieved chunks available for this sample.")
    else:
        judge_chunks = sample.get("judge_chunks") or {}
        for chunk_idx, chunk in enumerate(retrieved_chunks):
            judged = judge_chunks.get(chunk_idx)
            tag, expanded = "", chunk_idx == 0
            if judged:
                expanded = chunk_needs_human(judged)
                if judged.get("spot_check"):
                    tag = "🎯 spot check — "
                elif judge_flags(judged):
                    tag = "⚠ review " + ", ".join(judge_flags(judged)) + " — "
            with st.expander(
                f"{tag}Chunk {chunk_idx + 1} — {chunk.get('party', '?')} | Score: {chunk.get('score', 0):.4f}",
                expanded=expanded,
            ):
                render_chunk_form(chunk_idx, chunk)

    st.divider()

    # ===== SAMPLE-LEVEL METADATA =====
    st.write("## Annotation Metadata")
    ann_cols = st.columns([2, 1])
    ann_cols[0].text_input(
        "Annotator",
        key="annotation_annotator",
    )
    ann_cols[1].text_area(
        "Sample notes",
        key="annotation_sample_notes",
        height=50,
    )

    st.divider()

    # ===== NAVIGATION AND SAVE =====
    # Note: chunk state is reloaded automatically by the
    # "chunk_annotations_loaded_for" tracker whenever the displayed sample changes.
    nav_cols = st.columns([1, 1, 2])
    if nav_cols[0].button("Previous") and index > 0:
        st.session_state["annotation_index"] = index - 1
        st.rerun()
    if not review and nav_cols[1].button("Reset sample set"):
        st.session_state["annotation_samples"] = select_random_samples(
            load_qualitative_samples(),
            limit=ANNOTATION_SAMPLE_LIMIT,
            seed=st.session_state.get("annotation_seed", ANNOTATION_SEED),
        )
        st.session_state["annotation_index"] = 0
        st.rerun()

    can_go_next = index < len(rows) - 1
    save_button_label = (
        "Save annotations and go next" if can_go_next else "Save annotations"
    )
    if nav_cols[2].button(save_button_label, type="primary"):
        missing = [
            chunk_idx + 1
            for chunk_idx, chunk_data in st.session_state["chunk_annotations"].items()
            if any(chunk_data.get(dim) is None for dim in DIMENSIONS)
        ]
        if missing:
            st.error(f"Rate every dimension first (chunk {', '.join(map(str, sorted(missing)))}).")
            st.stop()
        judge_chunks = sample.get("judge_chunks") or {}
        # Convert chunk_annotations dict to list
        chunk_annotations_list = []
        for chunk_idx, chunk_data in sorted(
            st.session_state["chunk_annotations"].items()
        ):
            retrieved_chunk = (
                retrieved_chunks[chunk_idx] if chunk_idx < len(retrieved_chunks) else {}
            )
            annotation_entry = {
                "chunk_index": chunk_idx,
                "chunk_text": retrieved_chunk.get("text", ""),
                "chunk_metadata": {
                    "party": retrieved_chunk.get("party", ""),
                    "speaker": retrieved_chunk.get("speaker", ""),
                    "source": retrieved_chunk.get("source", ""),
                    "score": retrieved_chunk.get("score", 0.0),
                },
                "R_top": chunk_data.get("R_top", 3),
                "R_ideo": chunk_data.get("R_ideo", 3),
                "A_caus": chunk_data.get("A_caus", 0),
                "notes": chunk_data.get("notes", ""),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            judged = judge_chunks.get(chunk_idx)
            if judged is not None:
                annotation_entry["judge_labels"] = {dim: judged.get(dim) for dim in DIMENSIONS}
                annotation_entry["needs_review"] = judged.get("needs_review")
                annotation_entry["spot_check"] = bool(judged.get("spot_check"))
            chunk_annotations_list.append(annotation_entry)

        record = build_annotation_record(
            sample=sample,
            chunk_annotations=chunk_annotations_list,
            annotator=st.session_state.get("annotation_annotator", DEFAULT_ANNOTATOR),
            sample_notes=st.session_state.get("annotation_sample_notes", ""),
        )
        if review:
            record["sample_manifest"] = {"review_of": str(review["judge_path"])}
        save_annotation_record(
            record,
            st.session_state.get("annotation_log_path", ANNOTATION_LOG_PATH),
        )
        # Register the saved record so navigating back shows the saved values
        st.session_state["existing_annotations"][sample.get("text_index")] = record
        st.success(
            f"Saved {len(chunk_annotations_list)} chunk annotations for sample {sample.get('text_index')}"
        )

        if can_go_next:
            st.session_state["annotation_index"] = index + 1
            st.rerun()

    annotation_path = (
        Path(st.session_state.get("annotation_log_path", ANNOTATION_LOG_PATH))
        .expanduser()
        .resolve()
    )
    if annotation_path.exists():
        st.download_button(
            "Download annotations",
            data=annotation_path.read_bytes(),
            file_name=annotation_path.name,
            mime="application/jsonl",
            help="Download the annotation file and send it back after completing the samples.",
        )


def run_streamlit_app():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Qualitative Annotation Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if "review_mode" not in st.session_state:
        args = parse_app_args()
        if args.review:
            judge_path = resolve_judge_path(args.review)
            st.session_state["review_mode"] = {
                "judge_path": str(judge_path),
                "show_all": args.all,
                "quotas": parse_quotas(args.quotas),
            }
            st.session_state["annotation_log_path"] = Path(args.out) if args.out else judge_path.parent / REVIEWED_NAME
        else:
            st.session_state["review_mode"] = None
        if args.annotator:
            st.session_state["annotation_annotator"] = args.annotator
    init_session_state()
    render_annotation_section()


if __name__ == "__main__":
    run_streamlit_app()
