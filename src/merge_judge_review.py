"""Merge the LLM judge's labels with the human review into final annotations.

For every judge output folder (``run_llm_judge.py --samples``, one folder per
model family and size) this combines

* ``llm_judge_qualitative_annotations.jsonl`` - the judge's labels and flags,
* ``reviewed_annotations.jsonl`` - the texts reviewed in ``run_streamlit.py
  -- --review <folder>`` (optional; missing means nothing reviewed yet),
* ``review_selection.json`` - which texts were selected for review (written by
  the app; without it the default quadrant selection is recomputed). Texts
  outside the selection keep the judge labels and are not counted as pending,
* ``--human`` - fully human-annotated records (e.g. the consensus file), which
  win for identical (input, chunk) pairs in every folder; A_caus only where the
  generator justification is identical too (it is about that justification),
* ``--transfer-from`` - reviewed folders of OTHER models. Retrieval does not
  depend on the generator, so the same input text retrieves the same chunks
  for every model; R_top and R_ideo depend only on (input text, chunk text), so
  a reviewed label is copied to every identical pair (median if the pair was
  reviewed in several models). A_caus depends on the generator's justification
  and is never transferred,

into ``final_annotations.jsonl`` (human annotation format). Every chunk records
where each label came from (``label_source``):

* ``human_consensus`` - from ``--human``;
* ``spot_check`` - rated blind in the random spot-check sample;
* ``human_review`` - a flagged dimension a human checked;
* ``human_changed`` - an unflagged dimension a human changed anyway;
* ``judge_seen`` - an unflagged dimension of a reviewed text, kept as judged;
* ``transferred`` - R_top / R_ideo of the same (input, chunk) pair reviewed for
  another model (``--transfer-from``);
* ``judge`` - not seen by a human.

It also reports what the review tells about the judge: how often the human
changed a flagged label, and - from the spot checks - the agreement of the
judge with the human on dimensions the judge did NOT flag, which is the error
estimate for the ``judge`` labels in the final data.

    python src/merge_judge_review.py "RAG Analysis/LLM judge results/<run>"
    python src/merge_judge_review.py "RAG Analysis/LLM judge results/<run>/llama_8b" \\
        --human "RAG Analysis/human_annotations/revised/consensus_reference_annotations.jsonl"
    # all folders, copying reviewed R_top / R_ideo across models
    python src/merge_judge_review.py "RAG Analysis/LLM judge results/<run>" \
        --transfer-from "RAG Analysis/LLM judge results/<run>/llama_70b" <...>/mistral_14b <...>/qwen_3b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_streamlit import (
    DIMENSIONS,
    JUDGE_GROUPED_NAME,
    REVIEW_QUOTAS,
    REVIEWED_NAME,
    SELECTION_NAME,
    load_existing_annotations,
    quadrant_of,
    select_review_texts,
)

FINAL_NAME = "final_annotations.jsonl"
TRANSFER_DIMENSIONS = ("R_top", "R_ideo")  # depend on (input, chunk) only
SUMMARY_NAME = "review_summary.json"


def judge_folders(paths: list[str]) -> list[Path]:
    """The given folders that hold judge output, or their direct subfolders that do."""
    folders = []
    for p in map(Path, paths):
        if (p / JUDGE_GROUPED_NAME).exists():
            folders.append(p)
        else:
            folders.extend(sorted(d for d in p.iterdir() if (d / JUDGE_GROUPED_NAME).exists()))
    if not folders:
        raise SystemExit(f"No {JUDGE_GROUPED_NAME} found under {paths}")
    return folders


def _pair_key(input_text: str, chunk_text: str) -> tuple[str, str]:
    return ((input_text or "").strip(), (chunk_text or "").strip())


def _justification(record: dict) -> str:
    return (((record.get("source_context") or {}).get("rag") or {}).get("justification") or "").strip()


def human_chunks(path: str | None) -> dict:
    """(input, chunk) -> labelled chunk plus the justification it was rated against."""
    if not path:
        return {}
    return {
        _pair_key(record.get("input_text", ""), c.get("chunk_text", "")): {
            **c,
            "justification": _justification(record),
        }
        for record in load_existing_annotations(path).values()
        for c in record.get("chunk_annotations", [])
        if all(c.get(d) is not None for d in DIMENSIONS)
    }


def transfer_labels(folders: list[Path]) -> tuple[dict, dict]:
    """(input, chunk) -> {R_top, R_ideo} from the human-reviewed chunks of ``folders``.

    A pair reviewed in several models gets the lower median of its labels; the
    returned stats count how often those repeated ratings agreed (the
    reviewer's own consistency, the ceiling for any transfer).
    """
    ratings: dict = defaultdict(lambda: {d: [] for d in TRANSFER_DIMENSIONS})
    for folder in folders:
        path = folder / REVIEWED_NAME
        if not path.exists():
            raise SystemExit(f"--transfer-from {folder}: no {REVIEWED_NAME}")
        for record in load_existing_annotations(path).values():
            for c in record.get("chunk_annotations", []):
                key = _pair_key(record.get("input_text", ""), c.get("chunk_text", ""))
                for d in TRANSFER_DIMENSIONS:
                    if c.get(d) is not None:
                        ratings[key][d].append(int(c[d]))
    labels, repeated = {}, {d: [0, 0] for d in TRANSFER_DIMENSIONS}
    for key, by_dim in ratings.items():
        labels[key] = {}
        for d, values in by_dim.items():
            if not values:
                continue
            labels[key][d] = sorted(values)[(len(values) - 1) // 2]
            if len(values) > 1:
                repeated[d][0] += 1
                repeated[d][1] += int(len(set(values)) == 1)
    stats = {
        "pairs": len(labels),
        "repeated_pairs_consistent": {d: {"n": n, "consistent": c} for d, (n, c) in repeated.items()},
    }
    return labels, stats


def merge_chunk(
    judged: dict,
    reviewed: dict | None,
    human: dict | None,
    transferred: dict | None = None,
    justification: str = "",
) -> tuple[dict, dict]:
    """Final labels and their sources for one chunk."""
    flags = judged.get("needs_review") or {}
    labels, sources = {}, {}
    for dim in DIMENSIONS:
        j = judged.get(dim)
        if human is not None and (dim != "A_caus" or human["justification"] == justification):
            labels[dim], sources[dim] = human[dim], "human_consensus"
        elif reviewed is not None:
            labels[dim] = reviewed[dim]
            if reviewed.get("spot_check"):
                sources[dim] = "spot_check"
            elif flags.get(dim):
                sources[dim] = "human_review"
            else:
                sources[dim] = "human_changed" if reviewed[dim] != j else "judge_seen"
        elif transferred and dim in transferred:
            labels[dim], sources[dim] = transferred[dim], "transferred"
        else:
            labels[dim], sources[dim] = j, "judge"
    return labels, sources


def merge_folder(folder: Path, human: dict, transfer: dict | None = None) -> dict:
    judge = load_existing_annotations(folder / JUDGE_GROUPED_NAME)
    reviewed_path = folder / REVIEWED_NAME
    reviewed = load_existing_annotations(reviewed_path) if reviewed_path.exists() else {}
    selection_path = folder / SELECTION_NAME
    selected = set(
        json.loads(selection_path.read_text(encoding="utf-8"))["text_indices"]
        if selection_path.exists()
        else select_review_texts(judge, REVIEW_QUOTAS, folder.name)
    )

    stats = {
        "texts": len(judge),
        "texts_selected": len(selected),
        "texts_reviewed": sum(t in reviewed for t in judge),
        "transferred_chunks": 0,
        "reviewed_per_quadrant": defaultdict(int),
        "chunks": 0,
        "failed": 0,
        "pending": 0,  # flagged / spot-check chunks not reviewed yet
        "sources": defaultdict(int),
        "flagged": {d: {"n": 0, "changed": 0} for d in DIMENSIONS},
        # spot checks, split by whether the judge had flagged the dimension
        "spot": {d: {"unflagged": [], "flagged": []} for d in DIMENSIONS},
    }
    out = []
    for text_index, record in judge.items():
        if text_index in reviewed:
            stats["reviewed_per_quadrant"][quadrant_of(record)] += 1
        rev_chunks = {
            c["chunk_index"]: c for c in (reviewed.get(text_index) or {}).get("chunk_annotations", [])
        }
        chunks = []
        for judged in record.get("chunk_annotations", []):
            stats["chunks"] += 1
            if judged.get("error"):
                stats["failed"] += 1
            idx = judged["chunk_index"]
            rev = rev_chunks.get(idx)
            hum = human.get(_pair_key(record.get("input_text", ""), judged.get("chunk_text", "")))
            flags = judged.get("needs_review") or {}
            if reviewed and rev is None and hum is None and text_index in selected:
                stats["pending"] += 1  # selected for review, not reviewed yet
            moved = (transfer or {}).get(_pair_key(record.get("input_text", ""), judged.get("chunk_text", "")))
            labels, sources = merge_chunk(judged, rev, hum, moved, _justification(record))
            stats["transferred_chunks"] += int("transferred" in sources.values())
            for dim in DIMENSIONS:
                stats["sources"][sources[dim]] += 1
                if rev is None or judged.get(dim) is None:
                    continue
                if judged.get("spot_check"):
                    side = "flagged" if flags.get(dim) else "unflagged"
                    stats["spot"][dim][side].append((judged[dim], rev[dim]))
                elif flags.get(dim):
                    stats["flagged"][dim]["n"] += 1
                    stats["flagged"][dim]["changed"] += int(rev[dim] != judged[dim])
            chunks.append(
                {
                    "chunk_index": idx,
                    "chunk_text": judged.get("chunk_text", ""),
                    "chunk_metadata": judged.get("chunk_metadata", {}),
                    **labels,
                    "notes": (rev or {}).get("notes", ""),
                    "label_source": sources,
                    "judge_labels": {d: judged.get(d) for d in DIMENSIONS},
                    "needs_review": flags,
                    "spot_check": bool(judged.get("spot_check")),
                }
            )
        out.append(
            {
                "text_index": text_index,
                "input_text": record.get("input_text", ""),
                "source_context": record.get("source_context") or {},
                "sample_manifest": record.get("sample_manifest") or {},
                "chunk_annotations": chunks,
                "annotator": "judge+review",
                "reviewer": (reviewed.get(text_index) or {}).get("annotator", ""),
                "sample_notes": (reviewed.get(text_index) or {}).get("sample_notes", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    with (folder / FINAL_NAME).open("w", encoding="utf-8") as handle:
        for rec in out:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["sources"] = dict(stats["sources"])
    stats["reviewed_per_quadrant"] = dict(sorted(stats["reviewed_per_quadrant"].items()))
    return stats


def agreement(pairs: list) -> dict | None:
    if not pairs:
        return None
    return {
        "n": len(pairs),
        "exact": sum(j == h for j, h in pairs) / len(pairs),
        "within1": sum(abs(j - h) <= 1 for j, h in pairs) / len(pairs),
    }


def _pct(x: float | None) -> str:
    return f"{x * 100:5.1f}%" if x is not None else "    —"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="judge run folder(s) or per-file folder(s)")
    parser.add_argument("--transfer-from", nargs="+", default=None,
                        help="reviewed folders whose R_top / R_ideo labels are copied to identical "
                             "(input, chunk) pairs of the other folders")
    parser.add_argument("--human", default=None,
                        help="fully human-annotated JSONL whose labels win for matching chunks")
    args = parser.parse_args()

    human = human_chunks(args.human)
    folders = judge_folders(args.paths)
    transfer, transfer_stats = (
        transfer_labels(judge_folders(args.transfer_from)) if args.transfer_from else ({}, None)
    )
    if transfer_stats:
        rep = transfer_stats["repeated_pairs_consistent"]
        print(
            f"Transfer: {transfer_stats['pairs']} reviewed (input, chunk) pairs; pairs reviewed "
            "in several models got the same label in "
            + ", ".join(f"{d} {v['consistent']}/{v['n']}" for d, v in rep.items())
        )
    per_folder = {f.name: merge_folder(f, human, transfer) for f in folders}

    print(f"{'folder':<14}{'texts':>6}{'selected':>9}{'reviewed':>9}{'chunks':>7}{'transf.':>8}{'pending':>8}{'failed':>7}  labels by source")
    pooled_spot = {d: {"unflagged": [], "flagged": []} for d in DIMENSIONS}
    pooled_flag = {d: {"n": 0, "changed": 0} for d in DIMENSIONS}
    for name, s in per_folder.items():
        srcs = ", ".join(f"{k} {v}" for k, v in sorted(s["sources"].items()))
        print(f"{name:<14}{s['texts']:>6}{s['texts_selected']:>9}{s['texts_reviewed']:>9}{s['chunks']:>7}{s['transferred_chunks']:>8}{s['pending']:>8}{s['failed']:>7}  {srcs}")
        for d in DIMENSIONS:
            for side in ("unflagged", "flagged"):
                pooled_spot[d][side] += s["spot"][d][side]
            for k in ("n", "changed"):
                pooled_flag[d][k] += s["flagged"][d][k]

    print("\nAll folders pooled:")
    print(f"{'dim':<8}{'flagged reviewed':>17}{'changed':>9}  | spot check, unflagged: {'n':>4}{'exact':>8}{'±1':>8}  | flagged: {'n':>4}{'exact':>8}")
    summary_dims = {}
    for d in DIMENSIONS:
        f = pooled_flag[d]
        unf, fl = agreement(pooled_spot[d]["unflagged"]), agreement(pooled_spot[d]["flagged"])
        summary_dims[d] = {
            "flagged_reviewed": f["n"],
            "flagged_changed_share": f["changed"] / f["n"] if f["n"] else None,
            "spot_unflagged": unf,
            "spot_flagged": fl,
        }
        print(
            f"{d:<8}{f['n']:>17}{_pct(summary_dims[d]['flagged_changed_share']):>9}  | "
            f"{'':>23}{(unf or {}).get('n', 0):>4}{_pct((unf or {}).get('exact')):>8}"
            f"{_pct((unf or {}).get('within1')):>8}  | {'':>8}{(fl or {}).get('n', 0):>4}{_pct((fl or {}).get('exact')):>8}"
        )
    print(
        "\nspot check, unflagged = judge vs. blind human on dimensions the judge did not flag: "
        "the error estimate for labels with source 'judge'. Spot checks only come from the "
        "selected texts, so Q3 / Q4 contribute fewer of them."
    )
    pending = sum(s["pending"] for s in per_folder.values())
    if pending:
        print(f"Note: {pending} chunk(s) of selected texts not reviewed yet (they keep the judge label).")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "human_file": args.human,
        "transfer_from": args.transfer_from,
        "transfer": transfer_stats,
        "folders": {
            name: {k: v for k, v in s.items() if k != "spot"}
            | {"spot": {d: {side: agreement(p) for side, p in sides.items()} for d, sides in s["spot"].items()}}
            for name, s in per_folder.items()
        },
        "pooled": summary_dims,
    }
    target = folders[0].parent if len(folders) > 1 else folders[0]
    (target / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFinal annotations: <folder>/{FINAL_NAME}; summary: {target / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
