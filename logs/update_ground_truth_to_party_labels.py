#!/usr/bin/env python3
"""update_ground_truth_to_party_labels.py

Loads batch-run JSONL logs, looks up each entry's text_index in the article dataset
CSV, and replaces ground_truth labels (final_label_*) with party_label_* variants.

New JSONL files are saved alongside the originals with names prefixed by
"party_label_".

Usage:
    python src/update_ground_truth_to_party_labels.py
"""

import csv
import glob
import json
import os
from pathlib import Path

CSV_PATH = Path("src/datasets/political_bias_articles_dataset.csv")
BASE_DIR = Path("logs/batch_runs")


def load_csv_lookup(csv_path: Path) -> dict:
    """Build a lookup dict keyed by the CSV 'index' column."""
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            idx = row.get("index", "").strip()
            if not idx:
                continue
            lookup[idx] = {
                "party_label_ideology": row.get("party_label_ideology"),
                "party_label_economic": row.get("party_label_economic"),
                "party_label_galtan": row.get("party_label_galtan"),
            }
    return lookup


def process_jsonl(file_path: Path, lookup: dict) -> Path:
    """Rewrite a single JSONL file with party_label ground truths."""
    new_name = f"party_label_{file_path.name}"
    out_path = file_path.with_name(new_name)

    updated = 0
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as in_fh, \
         open(out_path, "w", encoding="utf-8") as out_fh:
        for line_idx, line in enumerate(in_fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                out_fh.write(line + "\n")
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed JSON in {file_path}:{line_idx} — {e}")
                out_fh.write(line + "\n")
                continue

            text_index = str(entry.get("input_metadata", {}).get("text_index", "")).strip()
            if text_index not in lookup:
                print(f"  Warning: text_index '{text_index}' not found in CSV (line {line_idx} of {file_path}). Leaving ground_truth unchanged.")
                skipped += 1
                out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            party_vals = lookup[text_index]
            gt = entry.get("ground_truth", {})
            gt["label_ideology"] = party_vals["party_label_ideology"]
            gt["label_economic"] = party_vals["party_label_economic"]
            gt["label_galtan"] = party_vals["party_label_galtan"]
            entry["ground_truth"] = gt
            updated += 1

            out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Written: {out_path}  (updated {updated} rows, skipped {skipped})")
    return out_path


def main():
    print(f"Loading dataset lookup from {CSV_PATH} ...")
    lookup = load_csv_lookup(CSV_PATH)
    print(f"  Loaded {len(lookup)} entries.\n")

    jsonl_files = sorted(BASE_DIR.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found under {BASE_DIR}")
        return

    print(f"Processing {len(jsonl_files)} JSONL files …\n")
    for fp in jsonl_files:
        # Skip files that already seem to be party_label outputs
        if fp.name.startswith("party_label_"):
            continue
        print(f"→ {fp}")
        process_jsonl(fp, lookup)

    print("\nDone.")


if __name__ == "__main__":
    main()
