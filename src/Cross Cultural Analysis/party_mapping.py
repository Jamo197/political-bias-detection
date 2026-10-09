"""Build and check the per-country party mapping (raw label -> canonical party + CHES id).

``party_mapping.csv`` is meant to be reviewed by hand. Columns::

    country_code, source (tweet|parlamint), raw_label, canonical_party, ches_party_id, status

``status``: ``auto`` (tweet label, CHES id from the dataset), ``seed`` (hard-coded guess
for a ParlaMint party code, verify!), ``suggested`` (fuzzy match, verify!),
``unmapped`` (fill in ches_party_id), ``manual`` (your edit; never overwritten on rebuild).

Usage::

    python party_mapping.py build      # tweets (+ any exported speeches) -> party_mapping.csv
    python party_mapping.py coverage   # share of tweets / speeches with a CHES id
    python party_mapping.py validate   # check every CHES id against 1999-2024_CHES.csv
    python party_mapping.py apply      # write ches_party_id/party_canonical into exported speech JSONs
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from cca_common import (
    DEFAULT_CHES,
    DEFAULT_MAPPING,
    DEFAULT_SPEECHES_DIR,
    DEFAULT_TWEETS_OUT,
    NAME_TO_CODE,
    PartyMapping,
    ches_country_code,
    clean_label,
    label_key,
    to_ches_id,
)

FIELDS = [
    "country_code",
    "source",
    "raw_label",
    "canonical_party",
    "ches_party_id",
    "status",
]

# Guesses for ParlaMint party codes -> CHES id. Always flagged for review.
SEED: dict[str, dict[str, int]] = {
    "pl": {"PiS": 2605, "PO": 2603, "KP-PSL": 2606, "PSL": 2606},
    "gb": {"CON": 1101, "LAB": 1102, "SNP": 1105, "GREEN": 1107},
    "nl": {"VVD": 1003, "D66": 1004, "PVV": 1017, "CDA": 1001, "SP": 1014,
           "GL": 1005, "PvdA": 1002, "CU": 1016, "SGP": 1006, "PvdD": 1018,
           "FvD": 1051},
    "fr": {"LREM": 626, "LR": 609, "RN": 610, "LFI": 627, "MODEM": 613, "PS": 602},
    "gr": {"ND": 402, "SYRIZA": 403, "KKE": 404},
    "es": {"PSOE": 501, "PP": 502, "VOX": 527, "CS": 526},
}  # fmt: skip


def iter_speech_parties(speeches_dir: Path):
    """Yield ``(country_code, party, ches_party_id)`` for every exported speech."""
    for path in speeches_dir.rglob("speeches/*.json"):
        for speech in json.loads(path.read_text(encoding="utf-8")):
            meta = speech.get("metadata", {})
            code = meta.get("country_code") or NAME_TO_CODE.get(meta.get("country", ""))
            if code:
                yield code, clean_label(meta.get("party")), meta.get("ches_party_id")


def party_labels(party: str) -> tuple[str, str]:
    """Full ParlaMint code plus its first part (party switchers carry ``A;B``)."""
    party = clean_label(party)
    return party, party.split(";")[0].strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(tweets_path: Path, speeches_dir: Path, out_path: Path) -> None:
    rows: dict[tuple[str, str], dict[str, str]] = {}

    def put(row: dict[str, str]) -> None:
        rows.setdefault((row["country_code"], label_key(row["raw_label"])), row)

    # 1. Manual edits survive a rebuild.
    for row in read_rows(out_path):
        if row.get("status") == "manual":
            put({field: row.get(field, "") for field in FIELDS})

    # 2. Tweet labels: the dataset's own CHES ids.
    tweets = pd.read_csv(tweets_path)
    canonical: dict[tuple[str, int], str] = {}
    for rec in tweets.to_dict("records"):
        code, ches = rec["country_code"], to_ches_id(rec["ches_party_id"])
        canonical[(code, ches)] = clean_label(rec["canonical_party"])
        for label in (rec["party"], rec["party_official"]):
            if clean_label(label):
                put(
                    {
                        "country_code": code,
                        "source": "tweet",
                        "raw_label": clean_label(label),
                        "canonical_party": canonical[(code, ches)],
                        "ches_party_id": str(ches),
                        "status": "auto",
                    }
                )

    tweet_labels: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    for (code, key), row in list(rows.items()):
        tweet_labels[code][key] = (row["canonical_party"], int(row["ches_party_id"]))

    # 3. ParlaMint party codes (only available once speeches have been exported).
    seen: Counter[tuple[str, str]] = Counter()
    if speeches_dir.exists():
        seen.update((code, party) for code, party, _ in iter_speech_parties(speeches_dir))
    for (code, party), _count in sorted(seen.items()):
        if not party or party == "UNKNOWN" or (code, label_key(party)) in rows:
            continue
        seed = {k.casefold(): v for k, v in SEED.get(code, {}).items()}.get(label_key(party))
        match = difflib.get_close_matches(
            label_key(party), list(tweet_labels[code]), n=1, cutoff=0.6
        )
        if seed:
            name, ches, status = canonical.get((code, seed), party), seed, "seed"
        elif match:
            (name, ches), status = tweet_labels[code][match[0]], "suggested"
        else:
            name, ches, status = "", "", "unmapped"
        put(
            {
                "country_code": code,
                "source": "parlamint",
                "raw_label": party,
                "canonical_party": name,
                "ches_party_id": str(ches),
                "status": status,
            }
        )

    ordered = sorted(rows.values(), key=lambda r: (r["country_code"], r["source"], r["raw_label"]))
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    counts = Counter(r["status"] for r in ordered)
    print(f"Wrote {len(ordered)} mapping rows to {out_path}: {dict(counts)}")
    if not seen:
        print(f"No exported speeches found in {speeches_dir}; ParlaMint rows not generated.")


def coverage(tweets_path: Path, speeches_dir: Path, mapping_path: Path) -> None:
    mapping = PartyMapping.from_csv(mapping_path)

    tweets = pd.read_csv(tweets_path)
    tweets["has_ches"] = tweets["ches_party_id"].notna()
    print("Tweets with a CHES id per country:")
    print(tweets.groupby("country")["has_ches"].agg(["mean", "size"]).round(3).to_string(), "\n")

    if not speeches_dir.exists():
        print(f"No exported speeches in {speeches_dir}; skipping speech coverage.")
        return
    total: Counter[str] = Counter()
    mapped: Counter[str] = Counter()
    unmapped: Counter[tuple[str, str]] = Counter()
    for code, party, _ in iter_speech_parties(speeches_dir):
        total[code] += 1
        if mapping.lookup(code, *party_labels(party))[1] is not None:
            mapped[code] += 1
        else:
            unmapped[(code, party)] += 1
    print("Speeches with a CHES id per country:")
    for code in sorted(total):
        print(f"  {code}: {mapped[code] / total[code]:.1%} of {total[code]}")
    print("\nTop unmapped ParlaMint parties (fill these in party_mapping.csv):")
    for (code, party), count in unmapped.most_common(40):
        print(f"  {code}  {party or '<empty>'}: {count}")


def apply(speeches_dir: Path, mapping_path: Path) -> None:
    """Add ``party_canonical`` + ``ches_party_id`` to already exported speech files.

    Lets you edit party_mapping.csv after the (slow) export without re-downloading.
    Files are rewritten in place via a temp file; running it twice is harmless.
    """
    mapping = PartyMapping.from_csv(mapping_path)
    files = sorted(speeches_dir.rglob("speeches/*.json"))
    total = mapped = 0
    for path in files:
        speeches = json.loads(path.read_text(encoding="utf-8"))
        for speech in speeches:
            meta = speech["metadata"]
            code = meta.get("country_code") or NAME_TO_CODE.get(meta.get("country", ""))
            canonical, ches_id = mapping.lookup(code or "", *party_labels(meta.get("party")))
            meta["party_canonical"], meta["ches_party_id"] = canonical, ches_id
            total += 1
            mapped += ches_id is not None
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(speeches, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    share = mapped / total if total else 0
    print(f"{len(files)} files, {total} speeches, {mapped} with a CHES id ({share:.1%})")


def validate(tweets_path: Path, mapping_path: Path, ches_path: Path) -> None:
    """Check CHES ids used in tweets/mapping against the CHES file."""
    ches = pd.read_csv(ches_path)
    known = ches.groupby("party_id").agg(
        ches_party=("party", "last"), first=("year", "min"), last=("year", "max")
    )

    tweets = pd.read_csv(tweets_path)
    used: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for rec in tweets.to_dict("records"):
        ches_id = to_ches_id(rec["ches_party_id"])
        used[(rec["country_code"], ches_id)][clean_label(rec["party_official"])] += 1
    for row in read_rows(mapping_path):
        ches_id = to_ches_id(row.get("ches_party_id"))
        if ches_id is not None and row["source"] == "parlamint":
            used[(row["country_code"], ches_id)][f"[mapping] {row['raw_label']}"] += 0

    problems = 0
    for (code, ches_id), labels in sorted(used.items()):
        issues = []
        if ches_id not in known.index:
            issues.append("id not in CHES file")
        else:
            if ches_country_code(ches_id) != code:
                issues.append(f"id belongs to country {ches_country_code(ches_id)}")
            if known.loc[ches_id, "last"] < 2019:
                issues.append(f"party only in CHES until {known.loc[ches_id, 'last']}")
        real = {k.casefold() for k in labels if not k.startswith("[mapping]")}
        if len(real) > 1:
            issues.append("several labels share this id")
        if issues:
            problems += 1
            abbr = known.loc[ches_id, "ches_party"] if ches_id in known.index else "?"
            shown = ", ".join(f"{k} ({v})" if v else k for k, v in labels.items())
            print(f"{code} {ches_id} [CHES: {abbr}] {'; '.join(issues)}\n    {shown}")
    print(f"\n{problems} of {len(used)} (country, CHES id) groups need a look.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build", "coverage", "validate", "apply"])
    parser.add_argument("--tweets", type=Path, default=DEFAULT_TWEETS_OUT)
    parser.add_argument("--speeches-dir", type=Path, default=DEFAULT_SPEECHES_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--ches", type=Path, default=DEFAULT_CHES)
    args = parser.parse_args(argv)

    if args.command == "build":
        build(args.tweets, args.speeches_dir, args.mapping)
    elif args.command == "apply":
        apply(args.speeches_dir, args.mapping)
    elif args.command == "validate":
        validate(args.tweets, args.mapping, args.ches)
    else:
        coverage(args.tweets, args.speeches_dir, args.mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
