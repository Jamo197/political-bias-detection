"""Streamlit app to compare the human annotations and the LLM judge.

Run from anywhere:

    streamlit run "RAG Analysis/human_annotations/run_annotation_comparison.py"

Two views (tabs):

* **Human vs LLM** — per chunk, what each annotator, the consensus and the Jev
  judge (``src/run_llm_judge.py``) scored. Annotators come from the revised
  ``revised/consensus_reference_annotations.jsonl`` (ratings after the
  discussion round); the judge from any ``RAG Analysis/<run_id>/`` folder.
* **Annotator agreement** — the annotators against each other (below).

Agreement numbers come from ``compare_human_annotations.py`` so they match its
CLI report. The Disagreements tab steps through chunks the annotators rated
differently and shows the source text, RAG justification, chunk text and each
annotator's ratings and notes side by side. Optionally, a consensus rating can
be stored per chunk in ``annotations/consensus_annotations.jsonl``; all saved
consensus chunks can be exported as a shareable markdown file.

Annotators' ratings can also be revised after a discussion. Revisions go to
``annotations/revised_annotations.jsonl`` (the annotation files stay untouched)
and feed the agreement numbers, which are shown next to the original ones.
"""

import dataclasses
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import cohen_kappa_score, confusion_matrix

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
REVISIONS_PATH = APP_ROOT / "annotations/revised_annotations.jsonl"
# Human vs LLM view: revised annotator ratings and the judge runs to compare.
CONSENSUS_REFERENCE_PATH = SCRIPT_DIR / "revised/consensus_reference_annotations.jsonl"
REVISED_NOTES_PATH = SCRIPT_DIR / "revised/Annotations - annotation_disagreements.csv"
JUDGE_RUNS_ROOT = SCRIPT_DIR.parent
JUDGE_FLAT_NAME = "llm_judge_annotations.jsonl"
JUDGE_DIMS = ["R_top", "R_ideo", "A_caus"]
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
def load_metrics(
    files: Tuple[str, ...],
    dims: Tuple[str, ...],
    alignment: str,
    revisions_json: str = "{}",
):
    """Agreement metrics; ``revisions_json`` is part of the cache key."""
    table, _ = load_table(files, dims, alignment)
    table = apply_revisions(table, json.loads(revisions_json))
    metrics = {d: compute_multirater_metrics(table, d, ALL_DIMENSIONS[d]) for d in dims}
    bias = {d: compute_annotator_bias(table, d) for d in dims}
    return metrics, bias


def _load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load jsonl rows keyed ``"text_index|chunk_index"``."""
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


def _save_rows(rows: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Atomic write like the annotation app."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for saved in rows.values():
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_consensus(path: Path = CONSENSUS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load consensus rows keyed ``"text_index|chunk_index"``."""
    return _load_rows(path)


def save_consensus_row(row: Dict[str, Any], path: Path = CONSENSUS_PATH) -> None:
    """Replace the consensus row for one chunk."""
    rows = _load_rows(path)
    rows[f"{row['text_index']}|{row['chunk_index']}"] = row
    _save_rows(rows, path)


def load_revisions(path: Path = REVISIONS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load revision rows keyed ``"text_index|chunk_index"``.

    A row holds ``revised`` and ``original`` as ``annotator -> dim -> value``
    (only the values that were changed), plus a free-text ``note``.
    """
    return _load_rows(path)


def save_revision_row(
    text_index: str,
    chunk_index: int,
    revised: Dict[str, Dict[str, int]],
    original: Dict[str, Dict[str, int]],
    note: str,
    path: Path = REVISIONS_PATH,
) -> None:
    """Replace the revision row for one chunk; no changes removes the row."""
    rows = _load_rows(path)
    key = f"{text_index}|{chunk_index}"
    if revised:
        rows[key] = {
            "text_index": text_index,
            "chunk_index": chunk_index,
            "revised": revised,
            "original": original,
            "note": note,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    else:
        rows.pop(key, None)
    _save_rows(rows, path)


def revised_values(revisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """``"text|chunk" -> annotator -> dim -> value``; input for ``apply_revisions``."""
    return {key: row["revised"] for key, row in revisions.items()}


def apply_revisions(table, revised: Dict[str, Dict[str, Dict[str, int]]]):
    """Copy of ``table`` with revised values; missing ratings stay missing."""
    if not revised:
        return table
    position = {key: i for i, key in enumerate(table.keys)}
    matrices = {dim: matrix.copy() for dim, matrix in table.matrices.items()}
    for storage_key, per_annotator in revised.items():
        text_index, chunk_index = storage_key.rsplit("|", 1)
        i = position.get((text_index, int(chunk_index)))
        if i is None:
            continue
        for name, per_dim in per_annotator.items():
            if name not in table.annotators:
                continue
            j = table.annotators.index(name)
            for dim, value in per_dim.items():
                if dim in matrices and not np.isnan(matrices[dim][i, j]):
                    matrices[dim][i, j] = value
    return dataclasses.replace(table, matrices=matrices)


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


def build_disagreement_items(
    table, dims: List[str], min_spread: int, only_disagree: bool
):
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


def highlight_citation(
    text: str, chunk_index: int, plain_markdown: bool = False
) -> str:
    """Bold the ``[N]`` / ``[N, M]`` reference that cites this chunk.

    ``plain_markdown`` skips Streamlit's ``:red[]`` colour directive so the
    result renders in any markdown viewer.
    """
    target = chunk_index + 1

    def mark(match: "re.Match[str]") -> str:
        if target not in {int(n) for n in re.findall(r"\d+", match.group(1))}:
            return match.group(0)
        if plain_markdown:
            return f"**\\[{match.group(1)}\\]**"
        return f"**:red[[{match.group(1)}]]**"

    # Single ("[2]") and list ("[2, 3]") citations.
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", mark, text or "")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def render_revision_summary(metrics, original_metrics, revisions, dims: List[str]):
    """Original vs revised agreement plus a log of every changed rating."""
    changed = sum(
        len(per_dim)
        for row in revisions.values()
        for per_dim in row["revised"].values()
    )
    st.info(
        f"Showing **revised** ratings: {changed} rating(s) changed in "
        f"{len(revisions)} chunk(s). Original files are untouched."
    )
    rows = []
    for dim in dims:
        before, after = original_metrics[dim], metrics[dim]
        rows.append(
            {
                "dimension": dim,
                "alpha original": before["alpha"],
                "alpha revised": after["alpha"],
                "Δ alpha": (
                    after["alpha"] - before["alpha"]
                    if None not in (before["alpha"], after["alpha"])
                    else None
                ),
                "mean κ original": before["mean_pairwise_kappa"],
                "mean κ revised": after["mean_pairwise_kappa"],
                "Δ mean κ": (
                    after["mean_pairwise_kappa"] - before["mean_pairwise_kappa"]
                    if None
                    not in (after["mean_pairwise_kappa"], before["mean_pairwise_kappa"])
                    else None
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).round(3), hide_index=True, width="stretch"
    )
    with st.expander(f"Revision log ({changed} changes)"):
        log = [
            {
                "text": row["text_index"],
                "chunk": row["chunk_index"] + 1,
                "annotator": name,
                "dimension": dim,
                "original": row["original"].get(name, {}).get(dim),
                "revised": value,
                "note": row.get("note", ""),
            }
            for row in revisions.values()
            for name, per_dim in row["revised"].items()
            for dim, value in per_dim.items()
        ]
        st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")


def render_overview(
    table,
    diagnostics,
    metrics,
    bias,
    dims: List[str],
    original_metrics=None,
    revisions=None,
):
    st.subheader("Agreement overview")
    st.caption(
        f"Annotators: {', '.join(table.annotators)} · items analysed: "
        f"{diagnostics['analysed_keys']} (shared by all {diagnostics['shared_keys']}, "
        f"union {diagnostics['union_keys']})"
    )
    for name, missing in diagnostics["missing_from"].items():
        if missing:
            st.warning(
                f"{name} did not rate {len(missing)} chunk(s) that others rated."
            )

    if revisions:
        render_revision_summary(metrics, original_metrics, revisions, dims)

    cols = st.columns(len(dims))
    for col, dim in zip(cols, dims):
        multi = metrics[dim]
        delta = None
        if revisions and None not in (multi["alpha"], original_metrics[dim]["alpha"]):
            delta = f"{multi['alpha'] - original_metrics[dim]['alpha']:+.3f}"
        col.metric(
            f"{dim} — Krippendorff α",
            fmt(multi["alpha"]),
            delta=delta,
            help=f"{multi['alpha_level']} alpha across all raters; "
            f"mean pairwise kappa {fmt(multi['mean_pairwise_kappa'])}",
        )
        col.caption(
            f"mean pairwise κ {fmt(multi['mean_pairwise_kappa'])} "
            f"({fmt(multi['min_pairwise_kappa'])} … {fmt(multi['max_pairwise_kappa'])})"
        )

    if st.checkbox("Show detailed metrics per dimension", value=False):
        render_dimension_details(metrics, bias, dims)


def render_dimension_details(metrics, bias, dims: List[str]):
    """Pairwise agreement, annotator bias and confusion matrices per dimension."""
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

        with st.expander(
            "Confusion matrices (rows = first annotator, columns = second)"
        ):
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


def _quote(text: str) -> str:
    """Render multi-line text as a markdown blockquote."""
    lines = (text or "").strip().splitlines() or ["—"]
    # Two trailing spaces keep the original line breaks in rendered markdown.
    return "\n".join(f"> {line}  " if line.strip() else ">" for line in lines)


def _rating_text(dim: str, value: Optional[int]) -> str:
    if value is None:
        return "—"
    if dim == "A_caus":
        return "Yes" if value == 1 else "No"
    return str(value)


def _sort_key(item) -> Tuple[int, str, int]:
    text_index, chunk_index = item["key"]
    numeric = int(text_index) if text_index.isdigit() else 10**9
    return (numeric, text_index, chunk_index)


def build_consensus_markdown(
    items,
    raw,
    table,
    dims: List[str],
    consensus: Dict[str, Dict[str, Any]],
    revisions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Shareable markdown of every chunk with a saved consensus.

    Each chunk mirrors the app: source text, RAG justification, chunk, the
    annotators' ratings and notes, and the consensus with its note. Revised
    ratings (``items`` carry the revised values) are marked ``(was X)``.
    """
    revisions = revisions or {}
    decided = sorted(
        (it for it in items if f"{it['key'][0]}|{it['key'][1]}" in consensus),
        key=_sort_key,
    )
    annotators = table.annotators
    lines: List[str] = [
        "# Annotation consensus",
        "",
        f"- Generated: {datetime.date.today().isoformat()}",
        f"- Annotators: {', '.join(annotators)}",
        f"- Dimensions: {', '.join(dims)}",
        f"- Chunks with consensus: {len(decided)}",
        "",
    ]

    if decided:
        lines += [
            "## Summary",
            "",
            "Share of chunks where an annotator's current rating equals the updated value.",
            "",
            "| annotator | " + " | ".join(dims) + " |",
            "|---|" + "---|" * len(dims),
        ]
        for name in annotators:
            cells = []
            for dim in dims:
                pairs = [
                    (
                        it["ratings"][dim][name],
                        consensus[f"{it['key'][0]}|{it['key'][1]}"]["consensus"].get(
                            dim
                        ),
                    )
                    for it in decided
                    if name in it["ratings"][dim]
                ]
                pairs = [(own, final) for own, final in pairs if final is not None]
                cells.append(
                    f"{100 * sum(own == final for own, final in pairs) / len(pairs):.0f} % ({len(pairs)})"
                    if pairs
                    else "—"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "---", ""]

    for item in decided:
        text_index, chunk_index = item["key"]
        saved = consensus[f"{text_index}|{chunk_index}"]
        record = first_record(raw, text_index)
        context = record.get("source_context", {}) or {}
        input_meta = context.get("input_metadata", {}) or {}
        chunks = find_chunk(raw, item["key"])
        any_chunk = next(iter(chunks.values()), {})
        chunk_meta = any_chunk.get("chunk_metadata", {}) or {}
        justification = (context.get("rag", {}) or {}).get("justification", "")

        lines += [
            f"## Text {text_index} · chunk {chunk_index + 1}",
            "",
            "### Source text",
            "",
            f"**Party:** {input_meta.get('party', '—')} · "
            f"**Speaker:** {input_meta.get('speaker', '—')} · "
            f"**Source:** {input_meta.get('source', '—')} · "
            f"**Quadrant:** {context.get('quadrant', '—')}",
            "",
            _quote(record.get("input_text", "")),
            "",
            "### RAG justification",
            "",
            f"_Citation of this chunk: **\\[{chunk_index + 1}\\]**_",
            "",
            _quote(highlight_citation(justification, chunk_index, plain_markdown=True)),
            "",
            "### Chunk",
            "",
            f"**Party:** {chunk_meta.get('party', '—')} · "
            f"**Speaker:** {chunk_meta.get('speaker', '—')} · "
            f"**Source:** {chunk_meta.get('source') or '—'}",
            "",
            _quote(any_chunk.get("chunk_text", "")),
            "",
            "### Ratings",
            "",
            "| dimension | " + " | ".join(annotators) + " | **Updated Values** |",
            "|---|" + "---|" * (len(annotators) + 1),
        ]
        for dim in dims:
            cells = []
            was = revisions.get(f"{text_index}|{chunk_index}", {}).get("original", {})
            for name in annotators:
                text = _rating_text(dim, item["ratings"][dim].get(name))
                if name in was and dim in was[name]:
                    text += f" (was {_rating_text(dim, was[name][dim])})"
                cells.append(text)
            final = _rating_text(dim, saved.get("consensus", {}).get(dim))
            lines.append(f"| {dim} | " + " | ".join(cells) + f" | **{final}** |")
        lines.append("")

        note_lines = []
        for name in annotators:
            parts = []
            chunk_note = ((chunks.get(name) or {}).get("notes") or "").strip()
            sample_note = (
                raw.get(name, {}).get(text_index, {}).get("sample_notes") or ""
            ).strip()
            if chunk_note:
                parts.append(chunk_note)
            if sample_note:
                parts.append(f"(sample note) {sample_note}")
            if parts:
                body = "  \n  ".join(" ".join(parts).splitlines())
                note_lines.append(f"- **{name}:** {body}")
        if note_lines:
            lines += ["**Annotator notes**", ""] + note_lines + [""]

        revision_note = (
            revisions.get(f"{text_index}|{chunk_index}", {}).get("note") or ""
        ).strip()
        if revision_note:
            lines += ["**Revision note**", "", _quote(revision_note), ""]

        consensus_note = (saved.get("note") or "").strip()
        lines += [
            "**Discussion note**",
            "",
            _quote(consensus_note) if consensus_note else "—",
            "",
            "---",
            "",
        ]

    lines += [
        "## Rubric",
        "",
        "**R_top**",
        "",
        *(f"- [{k}] {v}" for k, v in R_TOP_OPTIONS.items()),
        "",
        "**R_ideo**",
        "",
        *(f"- [{k}] {v}" for k, v in R_IDEO_OPTIONS.items()),
        "",
    ]
    return "\n".join(lines)


def render_disagreements(
    table,
    raw,
    dims: List[str],
    min_spread: int,
    only_disagree: bool,
    sort_by: str,
    enable_consensus: bool,
    enable_revision: bool = False,
    revisions: Optional[Dict[str, Dict[str, Any]]] = None,
):
    revisions = revisions or {}
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
            "**R_top**\n\n"
            + "\n".join(f"- [{k}] {v}" for k, v in R_TOP_OPTIONS.items())
        )
        rubric_cols[1].markdown(
            "**R_ideo**\n\n"
            + "\n".join(f"- [{k}] {v}" for k, v in R_IDEO_OPTIONS.items())
        )

    if enable_consensus:
        render_consensus_form(item, dims, table.annotators)

    st.write("### Annotator ratings")
    storage_key = f"{text_index}|{chunk_index}"
    revision = revisions.get(storage_key) or {}
    originals = revision.get("original", {})
    chosen: Dict[str, Dict[str, int]] = {}

    annotator_cols = st.columns(len(table.annotators))
    for col, name in zip(annotator_cols, table.annotators):
        col.markdown(f"#### {name}")
        entry = chunks.get(name)
        if entry is None:
            col.caption("Did not rate this chunk.")
            continue
        for dim in dims:
            value = item["ratings"][dim].get(name)
            was = originals.get(name, {}).get(dim)
            if enable_revision and value is not None:
                levels = ALL_DIMENSIONS[dim]["levels"]
                chosen.setdefault(name, {})[dim] = col.radio(
                    f"{dim}" + (f" (was {_rating_text(dim, was)})" if was is not None else ""),
                    options=levels,
                    index=levels.index(value),
                    format_func=lambda v, d=dim: _rating_text(d, v),
                    horizontal=True,
                    key=f"rev_{storage_key}_{name}_{dim}",
                )
                continue
            differs = item["spreads"][dim] > 0
            text = _rating_text(dim, value)
            if was is not None:
                text += f" (was {_rating_text(dim, was)})"
            marker = ":red[" + text + "]" if differs else text
            col.markdown(f"**{dim}:** {marker}")
        note = (entry.get("notes") or "").strip()
        col.caption("Chunk notes")
        col.write(note or "—")
        sample_note = (
            raw.get(name, {}).get(text_index, {}).get("sample_notes") or ""
        ).strip()
        if sample_note:
            col.caption("Sample notes")
            col.write(sample_note)

    if enable_revision:
        render_revision_form(item, dims, chosen, revision)


def render_revision_form(
    item,
    dims: List[str],
    chosen: Dict[str, Dict[str, int]],
    revision: Dict[str, Any],
):
    """Save the radios above as revisions of the annotators' original ratings."""
    text_index, chunk_index = item["key"]
    storage_key = f"{text_index}|{chunk_index}"
    note = st.text_area(
        "Revision note (what changed and why)",
        value=revision.get("note", ""),
        key=f"rev_{storage_key}_note",
        height=60,
    )

    def reset():
        save_revision_row(text_index, chunk_index, {}, {}, "")
        for key in [k for k in st.session_state if k.startswith(f"rev_{storage_key}_")]:
            del st.session_state[key]

    buttons = st.columns([1, 1, 4])
    if buttons[0].button("Save revised ratings", type="primary", key=f"rev_{storage_key}_save"):
        revised = {
            name: dict(per_dim) for name, per_dim in revision.get("revised", {}).items()
        }
        original = {
            name: dict(per_dim) for name, per_dim in revision.get("original", {}).items()
        }
        for name, per_dim in chosen.items():
            for dim in dims:
                if dim not in per_dim:
                    continue
                # The value before any revision: stored original, else what is shown.
                before = original.get(name, {}).pop(dim, item["ratings"][dim].get(name))
                revised.get(name, {}).pop(dim, None)
                if per_dim[dim] != before:
                    revised.setdefault(name, {})[dim] = per_dim[dim]
                    original.setdefault(name, {})[dim] = before
        revised = {name: v for name, v in revised.items() if v}
        original = {name: v for name, v in original.items() if v}
        save_revision_row(text_index, chunk_index, revised, original, note)
        st.toast(
            f"Saved {sum(len(v) for v in revised.values())} revised rating(s)"
            if revised
            else "No changes to the original ratings; revision removed."
        )
        st.rerun()  # agreement numbers in the Overview tab need recomputing
    buttons[1].button(
        "Reset to original",
        on_click=reset,
        disabled=not revision,
        key=f"rev_{storage_key}_reset",
    )


def render_consensus_form(item, dims: List[str], annotators: List[str]):
    text_index, chunk_index = item["key"]
    storage_key = f"{text_index}|{chunk_index}"
    saved = load_consensus().get(storage_key)

    st.write("## Revision")
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
# Human vs LLM
# ---------------------------------------------------------------------------


@st.cache_data
def load_reference(path: str, mtime: float) -> Dict[str, Dict[str, Any]]:
    """Revised consensus records keyed by ``text_index`` (``mtime`` busts the cache)."""
    records: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["text_index"]).strip()] = record
    return records


@st.cache_data
def load_judge_run(path: str, mtime: float) -> Dict[ItemKey, Dict[str, Any]]:
    """Flat judge results keyed ``(text_index, chunk_index)``."""
    results: Dict[ItemKey, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                results[(str(row["text_index"]).strip(), row["chunk_index"])] = row
    return results


@st.cache_data
def load_revised_notes(path: str, mtime: float) -> Dict[ItemKey, Dict[str, str]]:
    """Discussion-round notes per ``(text_index, chunk_index)`` and annotator."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    note_columns = [c for c in frame.columns if c.startswith("notes_")]
    notes: Dict[ItemKey, Dict[str, str]] = {}
    for _, row in frame.iterrows():
        key = (str(row["text_index"]).strip(), int(row["chunk_number"]) - 1)
        notes[key] = {
            c[len("notes_") :]: row[c].strip() for c in note_columns if row[c].strip()
        }
    return notes


def discover_judge_runs() -> List[Path]:
    """Judge result files under ``RAG Analysis/<run_id>/`` (and the calibration
    runs in ``judge_calibration/runs/``), newest first."""
    found = [
        *JUDGE_RUNS_ROOT.glob(f"*/{JUDGE_FLAT_NAME}"),
        *(JUDGE_RUNS_ROOT / "judge_calibration/runs").glob(f"*/{JUDGE_FLAT_NAME}"),
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _run_label(path: Path) -> str:
    stamp = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return f"{path.parent.name}  ({stamp:%Y-%m-%d %H:%M})"


def build_llm_items(reference, judge) -> List[Dict[str, Any]]:
    """One entry per chunk judged in the run: consensus, annotators and LLM labels."""
    items = []
    for text_index, record in reference.items():
        for chunk in record.get("chunk_annotations") or []:
            key = (text_index, chunk["chunk_index"])
            result = judge.get(key)
            if result is None or "error" in result:
                continue
            llm = result["derived_labels"]
            ratings = chunk.get("human_ratings") or {}
            consensus = {dim: chunk.get(dim) for dim in JUDGE_DIMS}
            items.append(
                {
                    "key": key,
                    "record": record,
                    "chunk": chunk,
                    "result": result,
                    "consensus": consensus,
                    "annotators": {dim: ratings.get(dim, {}) for dim in JUDGE_DIMS},
                    "llm": {dim: llm.get(dim) for dim in JUDGE_DIMS},
                    "diff": {
                        dim: llm[dim] - consensus[dim]
                        for dim in JUDGE_DIMS
                        if llm.get(dim) is not None and consensus[dim] is not None
                    },
                }
            )
    return items


def _kappa(a: List[int], b: List[int], dim: str) -> Optional[float]:
    """Quadratic-weighted κ for the ordinal dims, plain κ for A_caus."""
    if not a:
        return None
    ordinal = dim != "A_caus"
    value = cohen_kappa_score(
        a,
        b,
        labels=[1, 2, 3, 4, 5] if ordinal else [0, 1],
        weights="quadratic" if ordinal else None,
    )
    return None if np.isnan(value) else float(value)


def llm_agreement(items, dim: str) -> Dict[str, Any]:
    """LLM vs consensus and vs each annotator, with human–human κ for scale."""
    rated = [it for it in items if dim in it["diff"]]
    llm = [it["llm"][dim] for it in rated]
    consensus = [it["consensus"][dim] for it in rated]
    names = sorted({n for it in rated for n in it["annotators"][dim]})

    def paired(name: str):
        pairs = [
            (it["annotators"][dim][name], it["llm"][dim])
            for it in rated
            if name in it["annotators"][dim]
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    per_annotator = {n: _kappa(*paired(n), dim) for n in names}
    human_pairs = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            both = [
                (it["annotators"][dim][first], it["annotators"][dim][second])
                for it in rated
                if first in it["annotators"][dim] and second in it["annotators"][dim]
            ]
            value = _kappa([p[0] for p in both], [p[1] for p in both], dim)
            if value is not None:
                human_pairs.append(value)
    return {
        "n": len(rated),
        "agree": float(np.mean(np.array(llm) == np.array(consensus))) if rated else None,
        "kappa": _kappa(consensus, llm, dim),
        "bias": float(np.mean(np.array(llm) - np.array(consensus))) if rated else None,
        "per_annotator": per_annotator,
        "human_human": float(np.mean(human_pairs)) if human_pairs else None,
        "labels": [0, 1] if dim == "A_caus" else [1, 2, 3, 4, 5],
        "confusion": (
            confusion_matrix(
                consensus, llm, labels=[0, 1] if dim == "A_caus" else [1, 2, 3, 4, 5]
            )
            if rated
            else None
        ),
    }


def a_caus_path(result: Dict[str, Any]) -> str:
    """How the judge's A_caus label came about (mirrors ``derive_labels``)."""
    citation = result.get("citation")
    if not citation:
        return "model probability only (this run stored no citation evidence)"
    if citation.get("cited_by_number"):
        return "chunk cited by number → 1"
    if citation.get("speaker_named"):
        return "speaker named in the justification → 1"
    if citation.get("any_numeric_citation"):
        return "other chunks cited by number, this one not → 0"
    return "no numeric citations → the model's probability decides"


# Ladder gates behind R_top / R_ideo (see src/run_llm_judge.py). The label prefix
# keeps the bars in ladder order; levels in brackets are what a "yes" unlocks.
GATE_LABELS = {
    "R_top": {
        "T_domain": "T1 same broad field (≥2)",
        "T_subarea": "T2 same narrow question (≥3)",
        "T_item": "T3 input's item discussed (≥4)",
        "T_central": "T4 item is the main subject (5)",
    },
    "R_ideo": {
        "I_procedural": "I1 only procedure (1, else 2)",
        "I_evaluative": "I2 argues or evaluates (≥3)",
        "I_stance": "I3 strong partisan side (≥4)",
        "I_party": "I4 own party's position (5)",
    },
}
GATE_RULES = {
    "R_top": "Label = 1 + the highest gate that is yes.",
    "R_ideo": "5 = I3 and I4 · 4 = I3 and I2 · 3 = I2 · otherwise 1 (I1 yes) or 2.",
}


def explain_score(answer: Dict[str, Any]) -> str:
    """Markdown: the label is the most probable rubric level."""
    probabilities = answer.get("probabilities") or {}
    if not probabilities:
        return "No probabilities stored for this answer."
    best = max(probabilities, key=probabilities.get)
    return (
        f"Most probable rubric level: **{int(best) + 1}** (p = {probabilities[best]:.2f}) "
        f"· weighted score {answer.get('score', 0.0) + 1:.2f}"
    )


def render_llm_summary(items, dims: List[str]):
    stats = {dim: llm_agreement(items, dim) for dim in dims}
    rows = []
    for dim in dims:
        s = stats[dim]
        row = {
            "dimension": dim,
            "n": s["n"],
            "exact agree %": round(100 * s["agree"], 1) if s["agree"] is not None else None,
            "LLM vs consensus": s["kappa"],
            "bias (LLM − consensus)": s["bias"],
        }
        for name, value in s["per_annotator"].items():
            row[f"LLM vs {name}"] = value
        row["annotators among themselves"] = s["human_human"]
        rows.append(row)
    st.caption(
        "κ is quadratic-weighted for R_top / R_ideo and plain Cohen's κ for A_caus. "
        "“Annotators among themselves” is the mean pairwise κ of the (revised) human "
        "ratings: the level the LLM is measured against."
    )
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")

    with st.expander("Confusion matrices (rows = consensus, columns = LLM)"):
        cols = st.columns(len(dims))
        for col, dim in zip(cols, dims):
            matrix = stats[dim]["confusion"]
            if matrix is None:
                continue
            labels = stats[dim]["labels"]
            col.caption(dim)
            col.dataframe(
                pd.DataFrame(
                    matrix,
                    index=[f"consensus={v}" for v in labels],
                    columns=[f"LLM={v}" for v in labels],
                )
            )


def render_llm_item(item, notes: Dict[str, str], dims: List[str]):
    text_index, chunk_index = item["key"]
    record, chunk, result = item["record"], item["chunk"], item["result"]
    context = record.get("source_context", {}) or {}
    input_meta = context.get("input_metadata", {}) or {}
    chunk_meta = chunk.get("chunk_metadata", {}) or {}

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
    chunk_cols[2].caption(f"Retrieval score: **{chunk_meta.get('score', '—')}**")
    st.write(chunk.get("chunk_text", ""))

    st.write("## Ratings")
    names = sorted({n for dim in dims for n in item["annotators"][dim]})
    header = ["dimension", *names, "**consensus**", "**LLM**", "LLM − consensus"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dim in dims:
        cells = [dim]
        for name in names:
            value = item["annotators"][dim].get(name)
            cells.append(_rating_text(dim, value))
        consensus, llm = item["consensus"][dim], item["llm"][dim]
        differs = dim in item["diff"] and item["diff"][dim] != 0
        llm_text = _rating_text(dim, llm)
        cells += [
            f"**{_rating_text(dim, consensus)}**",
            f"**:red[{llm_text}]**" if differs else f"**{llm_text}**",
            f"{item['diff'][dim]:+d}" if dim in item["diff"] else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))

    st.write("### How the LLM got there")
    detail_cols = st.columns(3)
    for col, dim in zip(detail_cols[:2], ("R_top", "R_ideo")):
        if dim not in dims:
            continue
        answers = result.get("raw_answers") or {}
        col.markdown(f"**{dim}**")
        gates = GATE_LABELS[dim]
        if all(g in answers for g in gates):  # ladder run
            yes = [g for g in gates if answers[g]["noul"] >= 0.5]
            col.markdown(
                f"Label **{result['derived_labels'][dim]}** · "
                f"gates yes: {', '.join(yes) or 'none'}"
            )
            col.bar_chart(
                pd.DataFrame({"p(yes)": {gates[g]: answers[g]["noul"] for g in gates}}),
                height=170,
            )
            col.caption(f"A gate is yes at p ≥ 0.5. {GATE_RULES[dim]}")
            continue
        answer = answers.get(dim) or {}  # older five-level score runs
        col.markdown(explain_score(answer))
        probabilities = answer.get("probabilities") or {}
        if probabilities:
            chart = pd.DataFrame(
                {"probability": {str(int(k) + 1): v for k, v in probabilities.items()}}
            )
            col.bar_chart(chart, height=140)
            col.caption("Chart: probability of each rubric level; the label is the tallest bar.")
    if "A_caus" in dims:
        answer = (result.get("raw_answers") or {}).get("A_caus") or {}
        detail_cols[2].markdown("**A_caus**")
        detail_cols[2].caption(f"p(relied on this chunk) = {answer.get('noul', float('nan')):.2f}")
        detail_cols[2].write(a_caus_path(result))

    st.write("### Annotator notes (discussion round)")
    note_cols = st.columns(max(len(names), 1))
    for col, name in zip(note_cols, names):
        col.markdown(f"**{name}**")
        col.write(notes.get(name) or "—")


def render_llm_comparison():
    if not CONSENSUS_REFERENCE_PATH.exists():
        st.error(f"Missing reference annotations: {CONSENSUS_REFERENCE_PATH}")
        return
    runs = discover_judge_runs()
    if not runs:
        st.info(
            f"No judge runs found under {JUDGE_RUNS_ROOT}. Run "
            "`python src/run_llm_judge.py` first."
        )
        return

    reference = load_reference(
        str(CONSENSUS_REFERENCE_PATH), CONSENSUS_REFERENCE_PATH.stat().st_mtime
    )
    notes = load_revised_notes(
        str(REVISED_NOTES_PATH),
        REVISED_NOTES_PATH.stat().st_mtime if REVISED_NOTES_PATH.exists() else 0.0,
    )

    top = st.columns([3, 2, 1])
    run_path = top[0].selectbox("LLM judge run", runs, format_func=_run_label, key="llm_run")
    dims = top[1].multiselect(
        "Dimensions",
        options=JUDGE_DIMS,
        default=JUDGE_DIMS,
        format_func=lambda d: DIM_LABELS.get(d, d),
        key="llm_dims",
    )
    if top[2].button("Reload files", key="llm_reload"):
        st.cache_data.clear()
        st.rerun()
    if not dims:
        st.warning("Select at least one dimension.")
        return

    judge = load_judge_run(str(run_path), run_path.stat().st_mtime)
    items = build_llm_items(reference, judge)
    total = sum(len(r.get("chunk_annotations") or []) for r in reference.values())
    if not items:
        st.warning("None of the judged chunks match the reference annotations.")
        return
    sample = next(iter(judge.values()))
    st.caption(
        f"Judge model **{sample.get('response_model') or sample.get('model', '—')}** · "
        f"questions hash `{sample.get('questions_hash', '—')}` · "
        f"{len(items)} of {total} reference chunks judged"
        f"{' · failed: ' + str(sum('error' in r for r in judge.values())) if any('error' in r for r in judge.values()) else ''}. "
        "Annotator ratings are the revised ones (after the discussion round)."
    )

    render_llm_summary(items, dims)

    st.subheader("Chunk by chunk")
    row = st.columns([2, 2, 2, 2])
    only_diff = row[0].checkbox(
        "Only chunks where the LLM differs from the consensus", value=True, key="llm_only_diff"
    )
    min_diff = row[1].slider(
        "Minimum |LLM − consensus|", 1, 4, 1, disabled=not only_diff, key="llm_min_diff"
    )
    direction = row[2].radio(
        "Direction",
        ["Any", "LLM higher", "LLM lower"],
        horizontal=True,
        disabled=not only_diff,
        key="llm_direction",
    )
    sort_by = row[3].radio(
        "Sort", ["Largest difference", "Text order"], horizontal=True, key="llm_sort"
    )

    def wanted(item) -> bool:
        diffs = [item["diff"][d] for d in dims if d in item["diff"]]
        if not only_diff:
            return True
        if direction == "LLM higher":
            diffs = [d for d in diffs if d > 0]
        elif direction == "LLM lower":
            diffs = [d for d in diffs if d < 0]
        return any(abs(d) >= min_diff for d in diffs)

    shown = [it for it in items if wanted(it)]
    if sort_by == "Largest difference":
        shown.sort(
            key=lambda it: (
                -max((abs(it["diff"][d]) for d in dims if d in it["diff"]), default=0),
                -sum(abs(it["diff"][d]) for d in dims if d in it["diff"]),
            )
        )
    else:
        shown.sort(key=lambda it: (int(it["key"][0]) if it["key"][0].isdigit() else 10**9, it["key"][1]))
    if not shown:
        st.info("No chunks match the current filters.")
        return

    labels = {
        idx: f"text {it['key'][0]} · chunk {it['key'][1] + 1} · "
        + (
            ", ".join(
                f"{d} LLM {_rating_text(d, it['llm'][d])} vs {_rating_text(d, it['consensus'][d])}"
                for d in dims
                if it["diff"].get(d)
            )
            or "LLM matches the consensus"
        )
        for idx, it in enumerate(shown)
    }
    if st.session_state.get("llm_select") not in labels:
        st.session_state["llm_select"] = 0

    def go(delta: int):
        st.session_state["llm_select"] = max(
            0, min(len(shown) - 1, st.session_state["llm_select"] + delta)
        )

    nav = st.columns([1, 1, 4])
    nav[0].button("Previous", on_click=go, args=(-1,), width="stretch", key="llm_prev")
    nav[1].button("Next", on_click=go, args=(1,), width="stretch", key="llm_next")
    nav[2].selectbox(
        f"Item ({len(shown)} chunks)",
        options=list(labels),
        format_func=labels.get,
        key="llm_select",
        label_visibility="collapsed",
    )
    item = shown[st.session_state["llm_select"]]
    st.subheader(
        f"Item {st.session_state['llm_select'] + 1} / {len(shown)} — "
        f"text {item['key'][0]}, chunk {item['key'][1] + 1}"
    )
    render_llm_item(item, notes.get(item["key"], {}), dims)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def render_annotator_agreement():
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
        row3 = st.columns([2, 2, 2])
        use_revised = row3[0].checkbox(
            "Use revised ratings",
            value=True,
            help="Apply saved revisions to the ratings and agreement numbers. "
            "Untick to see the original annotations only.",
        )
        enable_revision = row3[1].checkbox(
            "Enable editing annotator ratings (writes annotations/revised_annotations.jsonl)",
            value=False,
            disabled=not use_revised,
        )
        enable_consensus = row3[2].checkbox(
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
        revisions = load_revisions() if use_revised else {}
        revised = revised_values(revisions)
        metrics, bias = load_metrics(
            files, tuple(dims), alignment, json.dumps(revised, sort_keys=True)
        )
        original_metrics, _ = load_metrics(files, tuple(dims), alignment)
        table = apply_revisions(table, revised)
    except ValueError as exc:
        st.error(f"Could not load annotations: {exc}")
        return
    if not table.keys:
        st.warning("No chunks to compare after alignment.")
        return
    raw = load_raw_records(files)

    consensus = load_consensus()
    all_items = build_disagreement_items(table, dims, 1, False)
    consensus_items = [
        it for it in all_items if f"{it['key'][0]}|{it['key'][1]}" in consensus
    ]
    st.download_button(
        f"Export consensus as Markdown ({len(consensus_items)} chunks)",
        data=build_consensus_markdown(
            all_items, raw, table, dims, consensus, revisions
        ),
        file_name="annotation_consensus.md",
        mime="text/markdown",
        disabled=not consensus_items,
        help="All chunks with a saved consensus (independent of the filters "
        "below), with source text, ratings, notes and the consensus note.",
    )

    overview_tab, disagreement_tab = st.tabs(["Overview", "Disagreements"])
    with overview_tab:
        render_overview(
            table, diagnostics, metrics, bias, dims, original_metrics, revisions
        )
    with disagreement_tab:
        render_disagreements(
            table,
            raw,
            dims,
            min_spread,
            only_disagree,
            sort_by,
            enable_consensus,
            enable_revision and use_revised,
            revisions,
        )


def run_app():
    st.set_page_config(
        page_title="Annotation Comparison",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Annotation Comparison")
    st.caption(
        "Compare the annotators' R_top, R_ideo and A_caus ratings with each other "
        "and with the LLM judge, chunk by chunk."
    )
    llm_tab, agreement_tab = st.tabs(["Human vs LLM", "Annotator agreement"])
    with llm_tab:
        render_llm_comparison()
    with agreement_tab:
        render_annotator_agreement()


if __name__ == "__main__":
    run_app()
