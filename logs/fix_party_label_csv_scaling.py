#!/usr/bin/env python3
"""fix_party_label_csv_scaling.py

One-off fix for political_bias_articles_dataset.csv: recomputes
party_label_ideology/economic/galtan from CHES 2024 scores (0-10 scale),
rescaled to the dataset's 1-7 scale with:

    new = 1 + (old / 10) * 6

These columns previously used a wrong formula that assumed CHES ran 1-10.
Only the party_label_* columns are touched; every other column (including
final_label_*) is left untouched.

Usage:
    python logs/fix_party_label_csv_scaling.py
"""

import csv
from pathlib import Path

CSV_PATH = Path("src/datasets/political_bias_articles_dataset.csv")
CHES_CSV_PATH = Path("src/datasets/ground_truth/ches_germany_1999-2024.csv")
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
                "party_label_ideology": rescale(float(row["lrgen"])),
                "party_label_economic": rescale(float(row["lrecon"])),
                "party_label_galtan": rescale(float(row["galtan"])),
            }
    return lookup


def main():
    print(f"Loading CHES {CHES_YEAR} lookup from {CHES_CSV_PATH} ...")
    lookup = load_ches_lookup(CHES_CSV_PATH)
    print(f"  Loaded {len(lookup)} parties.\n")

    print(f"Reading {CSV_PATH} ...")
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    skipped = 0
    for row in rows:
        party = row.get("party", "").strip()
        std_party = PARTY_ALIASES.get(party)
        if std_party is None or std_party not in lookup:
            print(f"  Warning: party '{party}' (index {row.get('index')}) not found in CHES lookup. Leaving row unchanged.")
            skipped += 1
            continue

        party_vals = lookup[std_party]
        row["party_label_ideology"] = party_vals["party_label_ideology"]
        row["party_label_economic"] = party_vals["party_label_economic"]
        row["party_label_galtan"] = party_vals["party_label_galtan"]
        updated += 1

    print(f"Writing {CSV_PATH} ...")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Updated {updated} rows, skipped {skipped} rows.")


if __name__ == "__main__":
    main()
