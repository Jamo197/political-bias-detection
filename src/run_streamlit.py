import datetime
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

DEFAULT_ANNOTATOR = "Jannes Lampe"

SESSION_STATE_DEFAULTS = {
    "annotation_samples": [],
    "annotation_index": 0,
    "annotation_seed": 42,
    "annotation_annotator": DEFAULT_ANNOTATOR,
    "annotation_sample_notes": "",
    "annotation_log_path": "RAG Analysis/qualitative_annotations.jsonl",
    "chunk_annotations": {},  # Maps chunk_index -> {R_top, R_ideo, N_info, A_caus, notes}
    "existing_annotations": {},  # Maps text_index -> latest saved annotation record
    "chunk_annotations_loaded_for": None,  # text_index of the sample currently in chunk state
    "acaus_auto_detected": set(),  # Chunk indices where A_caus was auto-set from the RAG justification
}

QUALITATIVE_SAMPLE_PATH = Path("results/qualitative/qualitative_capacity_8b.jsonl")

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
    sample_path = Path(path)
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
    path: str | Path = "RAG Analysis/qualitative_annotations.jsonl",
) -> Dict[str, Dict[str, Any]]:
    """Load existing annotation records; latest record per text_index wins."""
    annotations_path = Path(path)
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
        List of dicts, one per chunk, each containing {chunk_index, chunk_text, chunk_metadata, R_top, R_ideo, N_info, A_caus, notes, timestamp}.
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

    return {
        "text_index": sample.get("text_index"),
        "input_text": sample.get("input_text", ""),
        "source_context": source_context,
        "chunk_annotations": chunk_annotations,
        "annotator": annotator or "",
        "sample_notes": sample_notes or "",
        "timestamp": timestamp,
    }


def append_annotation_record(
    record: Dict[str, Any],
    output_path: str | Path = "logs/qualitative_annotations.jsonl",
) -> Path:
    """Append an annotation record to the output JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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


def load_chunk_state_for_sample(sample: Dict[str, Any]):
    """Populate chunk annotation state for the given sample.

    Precedence per chunk: existing annotation record > auto-detected A_caus
    (chunk cited as [N] in the RAG justification) > static defaults.
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

    chunk_annotations: Dict[int, Dict[str, Any]] = {}
    auto_detected = set()
    for chunk_idx in range(len(retrieved_chunks)):
        prev = existing_by_idx.get(chunk_idx)
        if prev is not None:
            state = {
                "R_top": prev.get("R_top", 3),
                "R_ideo": prev.get("R_ideo", 3),
                "N_info": prev.get("N_info", 2),
                "A_caus": prev.get("A_caus", 0),
                "notes": prev.get("notes", ""),
            }
        else:
            state = {
                "R_top": 3,
                "R_ideo": 3,
                "N_info": 2,
                "A_caus": 1 if chunk_idx in cited else 0,
                "notes": "",
            }
            if chunk_idx in cited:
                auto_detected.add(chunk_idx)
        chunk_annotations[chunk_idx] = state

        # Sync widget keys so the pre-filled values render correctly
        st.session_state[f"chunk_{chunk_idx}_rtop"] = state["R_top"]
        st.session_state[f"chunk_{chunk_idx}_rideo"] = state["R_ideo"]
        st.session_state[f"chunk_{chunk_idx}_ninfo"] = state["N_info"]
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
        st.session_state.get(
            "annotation_log_path", "RAG Analysis/qualitative_annotations.jsonl"
        )
    )

    if not st.session_state.get("annotation_samples"):
        rows = load_qualitative_samples()
        st.session_state["annotation_samples"] = select_random_samples(
            rows, limit=25, seed=st.session_state.get("annotation_seed", 42)
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
            "N_info": 2,
            "A_caus": 0,
            "notes": "",
        }

    chunk_state = st.session_state["chunk_annotations"][chunk_index]

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
        "R_top (Topical Relevance)",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"[{x}] {R_TOP_OPTIONS[x]}",
        help="Does the chunk discuss the exact policy issue, entity, or debate present in the input text?",
        key=f"chunk_{chunk_index}_rtop",
    )
    chunk_state["R_top"] = r_top

    r_ideo = likert_cols[1].selectbox(
        "R_ideo (Ideological Specificity)",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"[{x}] {R_IDEO_OPTIONS[x]}",
        help="Does the chunk provide unambiguous grounding for how a specific political party, faction, or ideology views this topic?",
        key=f"chunk_{chunk_index}_rideo",
    )
    chunk_state["R_ideo"] = r_ideo

    # Rubric form — Row 2: binary/3-point scales
    radio_cols = st.columns([1, 1])

    n_info = radio_cols[0].radio(
        "N_info (Information)",
        options=[1, 2, 3],
        horizontal=True,
        help="1=No new info, 3=High informational delta",
        key=f"chunk_{chunk_index}_ninfo",
    )
    chunk_state["N_info"] = n_info

    a_caus = radio_cols[1].radio(
        "A_caus (Attribution)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Did the model rely on this chunk in its reasoning?",
        key=f"chunk_{chunk_index}_acaus",
    )
    chunk_state["A_caus"] = a_caus

    if chunk_index in st.session_state.get("acaus_auto_detected", set()):
        radio_cols[1].caption(
            f"Auto-set: chunk cited as [{chunk_index + 1}] in the RAG justification."
        )

    # Chunk-specific notes
    chunk_notes = st.text_area(
        "Chunk notes",
        key=f"chunk_{chunk_index}_notes",
        height=60,
    )
    chunk_state["notes"] = chunk_notes


def render_annotation_section():
    """Render the qualitative annotation interface with per-chunk rubric forms."""
    st.title("Qualitative Annotation Workspace")
    st.caption(
        "Annotate retrieved chunks using the 4-dimensional rubric (R_top, R_ideo, N_info, A_caus)."
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
        "Ground Truth", f"{ground_truth:.1f}" if ground_truth is not None else "—"
    )
    pred_cols[1].metric("Baseline (No RAG)", f"{metrics.get('base_prediction', 0):.1f}")
    pred_cols[2].metric("RAG Prediction", f"{metrics.get('rag_prediction', 0):.1f}")
    pred_cols[3].metric("Base Error", f"{metrics.get('base_error', 0):.1f}")
    pred_cols[4].metric("RAG Error", f"{metrics.get('rag_error', 0):.1f}")
    pred_cols[5].metric(
        "Directional Shift", f"{metrics.get('directional_shift', 0):+.1f}"
    )

    just_cols = st.columns(2)
    with just_cols[0].expander("Baseline Justification (No RAG)", expanded=False):
        st.write(sample.get("baseline", {}).get("justification", "—"))
    with just_cols[1].expander("RAG Justification", expanded=False):
        st.write(sample.get("rag", {}).get("justification", "—"))

    st.divider()

    # ===== RETRIEVED CHUNKS SECTION =====
    st.write("## Retrieved Chunks")
    retrieved_chunks = sample.get("rag", {}).get("retrieved_chunks", [])

    if not retrieved_chunks:
        st.info("No retrieved chunks available for this sample.")
    else:
        for chunk_idx, chunk in enumerate(retrieved_chunks):
            with st.expander(
                f"Chunk {chunk_idx + 1} — {chunk.get('party', '?')} | Score: {chunk.get('score', 0):.4f}",
                expanded=(chunk_idx == 0),
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
    nav_cols = st.columns([1, 1, 1, 2])
    if nav_cols[0].button("Previous") and index > 0:
        st.session_state["annotation_index"] = index - 1
        st.rerun()
    if nav_cols[1].button("Next") and index < len(rows) - 1:
        st.session_state["annotation_index"] = index + 1
        st.rerun()
    if nav_cols[2].button("Reset sample set"):
        st.session_state["annotation_samples"] = select_random_samples(
            load_qualitative_samples(),
            limit=25,
            seed=st.session_state.get("annotation_seed", 42),
        )
        st.session_state["annotation_index"] = 0
        st.rerun()

    if nav_cols[3].button("Save all annotations", type="primary"):
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
                "N_info": chunk_data.get("N_info", 2),
                "A_caus": chunk_data.get("A_caus", 0),
                "notes": chunk_data.get("notes", ""),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            chunk_annotations_list.append(annotation_entry)

        record = build_annotation_record(
            sample=sample,
            chunk_annotations=chunk_annotations_list,
            annotator=st.session_state.get("annotation_annotator", DEFAULT_ANNOTATOR),
            sample_notes=st.session_state.get("annotation_sample_notes", ""),
        )
        append_annotation_record(
            record,
            st.session_state.get(
                "annotation_log_path", "RAG Analysis/qualitative_annotations.jsonl"
            ),
        )
        # Register the saved record so navigating back shows the saved values
        st.session_state["existing_annotations"][sample.get("text_index")] = record
        st.success(
            f"Saved {len(chunk_annotations_list)} chunk annotations for sample {sample.get('text_index')}"
        )

        if index < len(rows) - 1:
            st.session_state["annotation_index"] = index + 1
            st.rerun()


def run_streamlit_app():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Qualitative Annotation Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    render_annotation_section()


if __name__ == "__main__":
    run_streamlit_app()
