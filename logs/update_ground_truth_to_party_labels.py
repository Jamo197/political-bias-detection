#!/usr/bin/env python3
"""update_ground_truth_to_party_labels.py

Loads batch-run JSONL logs and replaces each entry's ground_truth labels with
party-level CHES 2024 scores, keyed by input_metadata.party. CHES scores run
0-10 and are rescaled to the dataset's 1-7 scale with:

    new = 1 + (old / 10) * 6

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

CHES_CSV_PATH = Path("src/datasets/ground_truth/ches_germany_1999-2024.csv")
BASE_DIR = Path("logs/batch_runs")
CHES_YEAR = "2024"

PARTY_ALIASES = {
    "AfD": "AfD",
    "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
    "Bündnis 90 Die Grünen": "BÜNDNIS 90/DIE GRÜNEN",
    "CDU": "CDU",
    "CDU/CSU": "CDU",
    "CSU": "CSU",
    "DIE LINKE": "DIE LINKE",
    "Die Linke": "DIE LINKE",
    "Linke": "DIE LINKE",
    "FDP": "FDP",
    "SPD": "SPD",
}


def rescale(old_0_to_10: float) -> float:
    """Rescale a 0-10 CHES score to the dataset's 1-7 scale."""
    return round(1 + (old_0_to_10 / 10) * 6, 1)


def load_ches_lookup(csv_path: Path) -> dict:
    """Build a lookup dict of rescaled CHES 2024 scores, keyed by std_party."""
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("year", "").strip() != CHES_YEAR:
                continue
            party = row.get("std_party", "").strip()
            if not party:
                continue
            lookup[party] = {
                "label_ideology": rescale(float(row["lrgen"])),
                "label_economic": rescale(float(row["lrecon"])),
                "label_galtan": rescale(float(row["galtan"])),
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

            party = str(entry.get("input_metadata", {}).get("party", "")).strip()
            std_party = PARTY_ALIASES.get(party)
            if std_party is None or std_party not in lookup:
                print(f"  Warning: party '{party}' not found in CHES lookup (line {line_idx} of {file_path}). Leaving ground_truth unchanged.")
                skipped += 1
                out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            party_vals = lookup[std_party]
            gt = entry.get("ground_truth", {})
            gt["label_ideology"] = party_vals["label_ideology"]
            gt["label_economic"] = party_vals["label_economic"]
            gt["label_galtan"] = party_vals["label_galtan"]
            entry["ground_truth"] = gt
            updated += 1

            out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Written: {out_path}  (updated {updated} rows, skipped {skipped})")
    return out_path


def main():
    print(f"Loading CHES {CHES_YEAR} lookup from {CHES_CSV_PATH} ...")
    lookup = load_ches_lookup(CHES_CSV_PATH)
    print(f"  Loaded {len(lookup)} parties.\n")

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
