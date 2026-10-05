"""Streamlit app to compare human qualitative annotations against each other.

Run from anywhere:

    streamlit run "RAG Analysis/human_annotations/run_annotation_comparison.py"

Agreement numbers come from ``compare_human_annotations.py`` so they match its
CLI report. The Disagreements tab steps through chunks the annotators rated
differently and shows the source text, RAG justification, chunk text and each
annotator's ratings and notes side by side. Optionally, a consensus rating can
be stored per chunk in ``annotations/consensus_annotations.jsonl``.
"""

import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent.parent
for path in (SCRIPT_DIR, APP_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_human_annotations import (  # noqa: E402
    ALL_DIMENSIONS,
    DEFAULT_PATTERN,
    DIMENSIONS,
    annotator_name,
    build_table,
    compute_annotator_bias,
    compute_multirater_metrics,
    discover_annotation_files,
)
from run_streamlit import R_IDEO_OPTIONS, R_TOP_OPTIONS  # noqa: E402

CONSENSUS_PATH = APP_ROOT / "annotations/consensus_annotations.jsonl"
DIM_LABELS = {
    "R_top": "R_top (topical relevance)",
    "R_ideo": "R_ideo (ideological specificity)",
    "A_caus": "A_caus (attribution)",
    "N_info": "N_info",
}

ItemKey = Tuple[str, int]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_raw_records(files: Tuple[str, ...]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load ``annotator -> text_index -> record`` for the given files."""
    records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for file in files:
        path = Path(file)
        per_text: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("text_index") is not None:
                    per_text[str(record["text_index"]).strip()] = record
        records[annotator_name(path)] = per_text
    return records


@st.cache_data
def load_table(files: Tuple[str, ...], dims: Tuple[str, ...], alignment: str):
    return build_table([Path(f) for f in files], list(dims), alignment)


@st.cache_data
def load_metrics(files: Tuple[str, ...], dims: Tuple[str, ...], alignment: str):
    table, _ = load_table(files, dims, alignment)
    metrics = {d: compute_multirater_metrics(table, d, ALL_DIMENSIONS[d]) for d in dims}
    bias = {d: compute_annotator_bias(table, d) for d in dims}
    return metrics, bias


def load_consensus(path: Path = CONSENSUS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load consensus rows keyed ``"text_index|chunk_index"``."""
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[f"{row['text_index']}|{row['chunk_index']}"] = row
    return rows


def save_consensus_row(row: Dict[str, Any], path: Path = CONSENSUS_PATH) -> None:
    """Replace the consensus row for one chunk; atomic like the annotation app."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_consensus(path)
    rows[f"{row['text_index']}|{row['chunk_index']}"] = row
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for saved in rows.values():
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt(value: Optional[float], spec: str = ".3f") -> str:
    return f"{value:{spec}}" if value is not None else "N/A"


def find_chunk(
    raw: Dict[str, Dict[str, Dict[str, Any]]], key: ItemKey
) -> Dict[str, Dict[str, Any]]:
    """Per annotator: the chunk annotation entry for ``key`` (if rated)."""
    text_index, chunk_index = key
    found = {}
    for name, per_text in raw.items():
        record = per_text.get(text_index)
        if record is None:
            continue
        for chunk in record.get("chunk_annotations") or []:
            if chunk.get("chunk_index") == chunk_index:
                found[name] = chunk
                break
    return found


def first_record(raw: Dict[str, Dict[str, Dict[str, Any]]], text_index: str):
    for per_text in raw.values():
        if text_index in per_text:
            return per_text[text_index]
    return {}


def build_disagreement_items(table, dims: List[str], min_spread: int, only_disagree: bool):
    """One entry per chunk: per-dim ratings and spread, filtered and sorted."""
    items = []
    for i, key in enumerate(table.keys):
        ratings: Dict[str, Dict[str, int]] = {}
        spreads: Dict[str, int] = {}
        for dim in dims:
            row = table.matrices[dim][i]
            present = {
                name: int(row[j])
                for j, name in enumerate(table.annotators)
                if not np.isnan(row[j])
            }
            ratings[dim] = present
            spreads[dim] = (
                max(present.values()) - min(present.values())
                if len(present) >= 2
                else 0
            )
        max_spread = max(spreads.values(), default=0)
        total_spread = sum(spreads.values())
        if only_disagree and max_spread < max(min_spread, 1):
            continue
        items.append(
            {
                "key": key,
                "ratings": ratings,
                "spreads": spreads,
                "max_spread": max_spread,
                "total_spread": total_spread,
            }
        )
    return items


def highlight_citation(text: str, chunk_index: int) -> str:
    """Bold the ``[N]`` reference that cites this chunk in the justification."""
    target = chunk_index + 1
    return re.sub(
        r"\[(\d+)\]",
        lambda m: f"**:red[[{m.group(1)}]]**" if int(m.group(1)) == target else m.group(0),
        text or "",
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def render_overview(table, diagnostics, metrics, bias, dims: List[str]):
    st.subheader("Agreement overview")
    st.caption(
        f"Annotators: {', '.join(table.annotators)} · items analysed: "
        f"{diagnostics['analysed_keys']} (shared by all {diagnostics['shared_keys']}, "
        f"union {diagnostics['union_keys']})"
    )
    for name, missing in diagnostics["missing_from"].items():
        if missing:
            st.warning(f"{name} did not rate {len(missing)} chunk(s) that others rated.")

    cols = st.columns(len(dims))
    for col, dim in zip(cols, dims):
        multi = metrics[dim]
        col.metric(
            f"{dim} — Krippendorff α",
            fmt(multi["alpha"]),
            help=f"{multi['alpha_level']} alpha across all raters; "
            f"mean pairwise kappa {fmt(multi['mean_pairwise_kappa'])}",
        )
        col.caption(
            f"mean pairwise κ {fmt(multi['mean_pairwise_kappa'])} "
            f"({fmt(multi['min_pairwise_kappa'])} … {fmt(multi['max_pairwise_kappa'])})"
        )

    for dim in dims:
        multi = metrics[dim]
        st.markdown(f"### {DIM_LABELS.get(dim, dim)}")
        rows = []
        for pair, entry in multi["pairwise"].items():
            if entry is None:
                continue
            rows.append(
                {
                    "pair": pair.replace("|", " vs "),
                    "n": entry["n"],
                    "agree %": round(entry["exact_agreement"] * 100, 1),
                    "kappa": entry["kappa"],
                    "alpha": entry["alpha"],
                    "mean diff (a−b)": round(entry["mean_diff_a_minus_b"], 2),
                    "mean |diff|": round(entry["mean_abs_diff"], 2),
                    "within 1 %": (
                        round(entry["pct_within_1"] * 100, 1)
                        if entry["pct_within_1"] is not None
                        else None
                    ),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        bias_rows = [
            {
                "annotator": name,
                "n": stats["n"],
                "mean level": stats["mean_level"],
                "deviation from item mean": stats["mean_deviation_from_item_mean"],
            }
            for name, stats in bias[dim].items()
        ]
        st.caption("Annotator bias (positive deviation = rates higher than the others)")
        st.dataframe(pd.DataFrame(bias_rows), hide_index=True, width="stretch")

        with st.expander("Confusion matrices (rows = first annotator, columns = second)"):
            pair_cols = st.columns(max(len(multi["pairwise"]), 1))
            for col, (pair, entry) in zip(pair_cols, multi["pairwise"].items()):
                if entry is None:
                    continue
                labels = entry["labels"]
                frame = pd.DataFrame(
                    entry["confusion_matrix"],
                    index=[f"{entry['rater_a']}={v}" for v in labels],
                    columns=[f"{entry['rater_b']}={v}" for v in labels],
                )
                col.caption(pair.replace("|", " vs "))
                col.dataframe(frame)


def build_export_csv(items, raw, table, dims: List[str]) -> bytes:
    """One row per chunk: context, each annotator's ratings/notes, spreads.

    Empty ``revised_<dim>_<annotator>`` and ``change_note_<annotator>``
    columns let each annotator enter a changed rating and say what they
    changed and why. UTF-8 with BOM so Sheets/Excel keep umlauts.
    """
    rows = []
    for item in items:
        text_index, chunk_index = item["key"]
        record = first_record(raw, text_index)
        chunks = find_chunk(raw, item["key"])
        any_chunk = next(iter(chunks.values()), {})
        chunk_meta = any_chunk.get("chunk_metadata", {}) or {}
        row: Dict[str, Any] = {
            "text_index": text_index,
            "chunk_number": chunk_index + 1,
            "chunk_party": chunk_meta.get("party", ""),
            "chunk_speaker": chunk_meta.get("speaker", ""),
            "input_text": record.get("input_text", ""),
            "chunk_text": any_chunk.get("chunk_text", ""),
        }
        for dim in dims:
            for name in table.annotators:
                row[f"{dim}_{name}"] = item["ratings"][dim].get(name, "")
            row[f"{dim}_spread"] = item["spreads"][dim]
        for name in table.annotators:
            row[f"notes_{name}"] = ((chunks.get(name) or {}).get("notes") or "").strip()
        for name in table.annotators:
            for dim in dims:
                row[f"revised_{dim}_{name}"] = ""
            row[f"change_note_{name}"] = ""
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def render_disagreements(table, raw, dims: List[str], min_spread: int, only_disagree: bool, sort_by: str, enable_consensus: bool):
    items = build_disagreement_items(table, dims, min_spread, only_disagree)
    if sort_by == "Largest disagreement":
        items.sort(key=lambda it: (-it["max_spread"], -it["total_spread"]))
    if not items:
        st.info("No chunks match the current filters.")
        return

    st.download_button(
        f"Export {len(items)} chunks as CSV",
        data=build_export_csv(items, raw, table, dims),
        file_name="annotation_disagreements.csv",
        mime="text/csv",
        help="Exports the chunks matching the current filters, in the current "
        "order. Import into Google Sheets via File > Import > Upload.",
    )

    labels = {
        idx: f"text {it['key'][0]} · chunk {it['key'][1] + 1} · "
        + (
            ", ".join(f"{d}:{it['spreads'][d]}" for d in dims if it["spreads"][d])
            or "no disagreement"
        )
        for idx, it in enumerate(items)
    }

    if st.session_state.get("cmp_select") not in labels:
        st.session_state["cmp_select"] = 0

    def go(delta: int):
        st.session_state["cmp_select"] = max(
            0, min(len(items) - 1, st.session_state["cmp_select"] + delta)
        )

    nav = st.columns([1, 1, 4])
    nav[0].button("Previous", on_click=go, args=(-1,), width="stretch")
    nav[1].button("Next", on_click=go, args=(1,), width="stretch")
    nav[2].selectbox(
        f"Item ({len(items)} chunks)",
        options=list(labels),
        format_func=labels.get,
        key="cmp_select",
        label_visibility="collapsed",
    )

    item = items[st.session_state["cmp_select"]]
    text_index, chunk_index = item["key"]
    record = first_record(raw, text_index)
    context = record.get("source_context", {}) or {}
    chunks = find_chunk(raw, item["key"])
    any_chunk = next(iter(chunks.values()), {})
    chunk_meta = any_chunk.get("chunk_metadata", {}) or {}
    input_meta = context.get("input_metadata", {}) or {}

    st.subheader(
        f"Item {st.session_state['cmp_select'] + 1} / {len(items)} — "
        f"text {text_index}, chunk {chunk_index + 1}"
    )

    st.write("## Source Text")
    meta_cols = st.columns(4)
    meta_cols[0].caption(f"Party: **{input_meta.get('party', '—')}**")
    meta_cols[1].caption(f"Speaker: **{input_meta.get('speaker', '—')}**")
    meta_cols[2].caption(f"Source: **{input_meta.get('source', '—')}**")
    meta_cols[3].caption(f"Quadrant: **{context.get('quadrant', '—')}**")
    st.write(record.get("input_text", ""))

    justification = (context.get("rag", {}) or {}).get("justification", "")
    with st.expander("RAG justification (this chunk's citation highlighted)"):
        st.markdown(highlight_citation(justification or "—", chunk_index))

    st.write("## Chunk")
    chunk_cols = st.columns(3)
    chunk_cols[0].caption(f"Party: **{chunk_meta.get('party', '—')}**")
    chunk_cols[1].caption(f"Speaker: **{chunk_meta.get('speaker', '—')}**")
    chunk_cols[2].caption(f"Source: **{chunk_meta.get('source', '—')}**")
    st.write(any_chunk.get("chunk_text", ""))

    st.write("## Ratings")
    with st.expander("Rubric"):
        rubric_cols = st.columns(2)
        rubric_cols[0].markdown(
            "**R_top**\n\n" + "\n".join(f"- [{k}] {v}" for k, v in R_TOP_OPTIONS.items())
        )
        rubric_cols[1].markdown(
            "**R_ideo**\n\n" + "\n".join(f"- [{k}] {v}" for k, v in R_IDEO_OPTIONS.items())
        )

    annotator_cols = st.columns(len(table.annotators))
    for col, name in zip(annotator_cols, table.annotators):
        col.markdown(f"#### {name}")
        entry = chunks.get(name)
        if entry is None:
            col.caption("Did not rate this chunk.")
            continue
        for dim in dims:
            value = item["ratings"][dim].get(name)
            differs = item["spreads"][dim] > 0
            text = "—" if value is None else str(value)
            if dim == "A_caus" and value is not None:
                text = "Yes" if value == 1 else "No"
            marker = ":red[" + text + "]" if differs else text
            col.markdown(f"**{dim}:** {marker}")
        note = (entry.get("notes") or "").strip()
        col.caption("Chunk notes")
        col.write(note or "—")
        sample_note = (raw.get(name, {}).get(text_index, {}).get("sample_notes") or "").strip()
        if sample_note:
            col.caption("Sample notes")
            col.write(sample_note)

    if enable_consensus:
        render_consensus_form(item, dims, table.annotators)


def render_consensus_form(item, dims: List[str], annotators: List[str]):
    text_index, chunk_index = item["key"]
    storage_key = f"{text_index}|{chunk_index}"
    saved = load_consensus().get(storage_key)

    st.write("## Consensus")
    if saved:
        st.info("Consensus already saved for this chunk; values loaded.")
    values: Dict[str, int] = {}
    form_cols = st.columns(len(dims))
    for col, dim in zip(form_cols, dims):
        levels = ALL_DIMENSIONS[dim]["levels"]
        proposed = item["ratings"][dim]
        default = (saved or {}).get("consensus", {}).get(dim)
        if default is None and proposed:
            # Median of the raters' values is a sensible starting point.
            default = int(round(float(np.median(list(proposed.values())))))
        if default not in levels:
            default = levels[0]
        values[dim] = col.radio(
            dim,
            options=levels,
            index=levels.index(default),
            horizontal=True,
            key=f"consensus_{storage_key}_{dim}",
        )
    note = st.text_area(
        "Consensus note",
        value=(saved or {}).get("note", ""),
        key=f"consensus_{storage_key}_note",
        height=60,
    )
    if st.button("Save consensus", type="primary", key=f"consensus_{storage_key}_save"):
        save_consensus_row(
            {
                "text_index": text_index,
                "chunk_index": chunk_index,
                "consensus": values,
                "annotator_ratings": item["ratings"],
                "note": note,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        st.success(f"Saved consensus to {CONSENSUS_PATH}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def run_app():
    st.set_page_config(
        page_title="Annotation Comparison",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Human Annotation Comparison")
    st.caption("Compare the annotators' R_top, R_ideo and A_caus ratings chunk by chunk.")

    try:
        all_files = discover_annotation_files(SCRIPT_DIR, DEFAULT_PATTERN)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with st.expander("Filters", expanded=True):
        row1 = st.columns([2, 2, 2])
        chosen = row1[0].multiselect(
            "Annotators",
            options=[annotator_name(f) for f in all_files],
            default=[annotator_name(f) for f in all_files],
        )
        dims = row1[1].multiselect(
            "Dimensions",
            options=list(ALL_DIMENSIONS),
            default=list(DIMENSIONS),
            format_func=lambda d: DIM_LABELS.get(d, d),
        )
        alignment = row1[2].radio(
            "Alignment",
            options=["shared", "all"],
            horizontal=True,
            help="shared: chunks rated by every selected annotator; all: union.",
        )
        row2 = st.columns([2, 2, 2])
        only_disagree = row2[0].checkbox("Only chunks with disagreement", value=True)
        min_spread = row2[1].slider(
            "Minimum spread (max − min rating)", 1, 4, 1, disabled=not only_disagree
        )
        sort_by = row2[2].radio(
            "Sort", ["Largest disagreement", "Text order"], horizontal=True
        )
        enable_consensus = st.checkbox(
            "Enable consensus editing (writes annotations/consensus_annotations.jsonl)",
            value=False,
        )
        if st.button("Reload files"):
            st.cache_data.clear()
            st.rerun()

    files = tuple(str(f) for f in all_files if annotator_name(f) in chosen)
    if len(files) < 2:
        st.warning("Select at least two annotators.")
        return
    if not dims:
        st.warning("Select at least one dimension.")
        return

    try:
        table, diagnostics = load_table(files, tuple(dims), alignment)
        metrics, bias = load_metrics(files, tuple(dims), alignment)
    except ValueError as exc:
        st.error(f"Could not load annotations: {exc}")
        return
    if not table.keys:
        st.warning("No chunks to compare after alignment.")
        return
    raw = load_raw_records(files)

    overview_tab, disagreement_tab = st.tabs(["Overview", "Disagreements"])
    with overview_tab:
        render_overview(table, diagnostics, metrics, bias, dims)
    with disagreement_tab:
        render_disagreements(
            table, raw, dims, min_spread, only_disagree, sort_by, enable_consensus
        )


if __name__ == "__main__":
    run_app()
