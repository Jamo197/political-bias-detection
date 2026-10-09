"""Draw more tweets for the EU tweet dataset from the huge CHES-integrated CSV.

The source (``ches_integrated_tweets_members_parties.csv``, ~2 GB, ~9.5M rows) is
never loaded whole: it is streamed in chunks, so memory stays at a few hundred MB.

Two phases, so nothing hits the Twitter oEmbed endpoint before the party/CHES
assignment has been reviewed::

    # 1. Stream the source twice (counts, then sampling). Writes the sample plan and
    #    a party review table (country, party labels, CHES id + CHES abbreviation).
    python extract_more_tweets.py plan --per-party 75

    # 2. Check party_review.csv, put corrections into ches_overrides.csv, re-run plan.
    #    (`learn-overrides` derives ches_overrides.csv from CHES ids you fixed by hand
    #    in EU_tweets_clean.csv.)
    # 3. Fetch the tweet texts (resumable; Ctrl-C and re-run is safe).
    python extract_more_tweets.py fetch

``plan`` tops each (country, CHES party) up to ``--per-party`` tweets, counting what
``EU_tweets_clean.csv`` already holds, and never re-draws known tweet ids. The result
``EU_tweets_extra.csv`` has the same columns as ``EU_tweets_with_parties_dataset.csv``
(country already resolved), so it can be appended to it, followed by ``clean_tweets.py``.

``ches_overrides.csv`` (optional) columns: ``country_code, party, party_official,
ches_party_id``. It overrides the CHES id of matching source rows, because the
notebook's substring matching can assign wrong ids (e.g. Slovenian "SD" -> SDS 2902).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cca_common import (
    COUNTRY_NAMES,
    DEFAULT_CHES,
    DEFAULT_TWEETS_IN,
    DEFAULT_TWEETS_OUT,
    HERE,
    ches_country_code,
    clean_label,
    label_key,
    to_ches_id,
)
from clean_tweets import resolve_country

DATASETS = HERE.parent / "datasets"
DEFAULT_SOURCE = Path(
    "/Users/janneslampe/Desktop/Coding/Master Thesis/Twitter Parliamentarian Database"
    "/ches_integrated_tweets_members_parties.csv"
)
DEFAULT_PLAN = DATASETS / "EU_tweets_extra_plan.csv"
DEFAULT_EXTRA = DATASETS / "EU_tweets_extra.csv"
DEFAULT_REVIEW = HERE / "party_review.csv"
DEFAULT_OVERRIDES = HERE / "ches_overrides.csv"

# Countries that exist in both the tweet database and ParlaMint3 (no es, fi, ie),
# plus Germany (speeches from the Bundestag corpus, see bundestag_to_parlamint_dir.py).
DEFAULT_COUNTRIES = "at be de dk fr gb gr it lv nl pl pt se si".split()

SOURCE_COLS = [
    "country", "party", "name", "uid", "date", "tweet_id", "party_id",
    "party_official", "party_abbr", "ches_party_id",
]  # fmt: skip
TARGET_COLS = [
    "country", "party", "name", "uid", "date", "tweet_id", "party_id",
    "party_official", "ches_party_id",
]  # fmt: skip
CHUNK_ROWS = 500_000
STRATUM = ["country_code", "ches_party_id"]

URL_ONLY_PATTERN = re.compile(
    r"^(?:https?://\S+|www\.\S+)(\s+(?:https?://\S+|www\.\S+))*$"
)


# --------------------------------------------------------------------------- helpers
def load_overrides(path: Path) -> dict[tuple[str, str, str], int]:
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ches = to_ches_id(row.get("ches_party_id"))
            if ches is not None:
                key = (
                    clean_label(row["country_code"]).lower(),
                    label_key(row["party"]),
                    label_key(row["party_official"]),
                )
                out[key] = ches
    return out


def known_tweet_ids(*paths: Path) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if path.exists():
            ids.update(pd.read_csv(path, usecols=["tweet_id"], dtype=str)["tweet_id"])
    return ids


def existing_counts(path: Path) -> Counter[tuple[str, int]]:
    """Tweets already collected per (country_code, CHES id)."""
    if not path.exists():
        return Counter()
    df = pd.read_csv(path, usecols=["country_code", "ches_party_id"])
    return Counter(zip(df["country_code"], df["ches_party_id"].astype(int)))


def iter_source(source: Path, countries: set[str], overrides, exclude: set[str]):
    """Yield filtered, normalized chunks: only target countries with a CHES id."""
    reader = pd.read_csv(
        source,
        usecols=SOURCE_COLS + ["country_member"],
        dtype=str,
        chunksize=CHUNK_ROWS,
    )
    resolved: dict[tuple, str | None] = {}
    for chunk in reader:
        chunk = chunk[chunk["ches_party_id"].notna() & chunk["tweet_id"].notna()]
        chunk = chunk[~chunk["tweet_id"].isin(exclude)]
        if chunk.empty:
            continue

        codes = []
        for country, member, ches in zip(
            chunk["country"], chunk["country_member"], chunk["ches_party_id"]
        ):
            key = (country, member, ches)
            if key not in resolved:
                code, _ = resolve_country(country, ches)
                # Same fallback as the notebook: member country, then CHES prefix.
                resolved[key] = code or resolve_country(member, ches)[0]
            codes.append(resolved[key])
        chunk = chunk.assign(country_code=codes)
        chunk = chunk[chunk["country_code"].isin(countries)]
        if chunk.empty:
            continue

        ches_ids = [
            overrides.get((cc, label_key(p), label_key(o)), to_ches_id(c))
            for cc, p, o, c in zip(
                chunk["country_code"],
                chunk["party"],
                chunk["party_official"],
                chunk["ches_party_id"],
            )
        ]
        chunk = chunk.assign(ches_party_id=ches_ids).dropna(subset=["ches_party_id"])
        chunk["ches_party_id"] = chunk["ches_party_id"].astype(int)
        chunk["country"] = chunk["country_code"].map(COUNTRY_NAMES)
        yield chunk


def get_tweet_text_oembed(session, tweet_id: str, retries: int = 3) -> str | None:
    """Tweet text, '[Deleted or Private Tweet]' on 404, None after repeated failures."""
    from bs4 import BeautifulSoup

    url = (
        "https://publish.twitter.com/oembed"
        f"?url=https://x.com/i/status/{tweet_id}&omit_script=true"
    )
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=15)
        except Exception:
            time.sleep(2**attempt)
            continue
        if response.status_code == 200:
            html = response.json().get("html", "")
            paragraph = BeautifulSoup(html, "html.parser").find("p")
            return paragraph.get_text() if paragraph else ""
        if response.status_code == 404:
            return "[Deleted or Private Tweet]"
        time.sleep(5 * 2**attempt)  # 429 / 5xx: back off
    return None


def is_valid_tweet_text(text: str | None) -> bool:
    """Present, not deleted, not URL-only and at least 25 characters."""
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    if not cleaned or cleaned == "[Deleted or Private Tweet]":
        return False
    return not URL_ONLY_PATTERN.fullmatch(cleaned) and len(cleaned) >= 25


# --------------------------------------------------------------------------- plan
def plan(args: argparse.Namespace) -> None:
    countries = {c.lower() for c in args.countries}
    overrides = load_overrides(args.overrides)
    exclude = known_tweet_ids(DEFAULT_TWEETS_IN, args.extra)
    have = existing_counts(args.existing)
    print(f"{len(exclude)} known tweet ids, {len(overrides)} CHES overrides")

    # Pass 1: tweets available per stratum + the party review table.
    available: Counter[tuple[str, int]] = Counter()
    labels: Counter[tuple] = Counter()
    for chunk in iter_source(args.source, countries, overrides, exclude):
        sizes = chunk.groupby(STRATUM).size()
        available.update({k: int(v) for k, v in sizes.items()})
        combo = chunk.groupby(
            ["country_code", "party", "party_abbr", "party_official", "ches_party_id"],
            dropna=False,
        ).size()
        labels.update({k: int(v) for k, v in combo.items()})
    write_review(labels, args.ches, args.review)

    # Parties with fewer source tweets than --min-available are skipped entirely
    # (too small for party-level metrics; label_tweets.py drops them anyway).
    need = {
        key: max(0, args.per_party - have[key])
        for key in available
        if available[key] >= args.min_available
    }
    fraction = {
        key: min(1.0, math.ceil(n * args.oversample) / available[key])
        for key, n in need.items()
    }
    print(
        f"{sum(available.values())} candidate tweets in {len(available)} strata; "
        f"{sum(1 for n in need.values() if n)} strata still need tweets"
    )

    # Pass 2: Bernoulli-sample each stratum to about need * oversample tweets.
    rng = np.random.default_rng(args.seed)
    picked = []
    for chunk in iter_source(args.source, countries, overrides, exclude):
        keys = list(zip(chunk["country_code"], chunk["ches_party_id"]))
        p = np.fromiter((fraction.get(k, 0.0) for k in keys), float, len(keys))
        picked.append(chunk[rng.random(len(chunk)) < p])
    sample = pd.concat(picked).drop_duplicates("tweet_id")
    sample["need"] = [need[k] for k in zip(sample["country_code"], sample["ches_party_id"])]
    sample = sample.sample(frac=1, random_state=args.seed)  # shuffled fetch order
    sample[TARGET_COLS + ["country_code", "need"]].to_csv(args.out, index=False)
    print(f"Wrote {len(sample)} planned tweets to {args.out}")
    print(sample.groupby("country_code").size().to_string())


def write_review(labels: Counter, ches_path: Path, out_path: Path) -> None:
    """One row per (country, party labels, CHES id) so wrong ids are easy to spot."""
    ches = pd.read_csv(ches_path)
    known = ches.groupby("party_id").agg(
        ches_abbr=("party", "last"), ches_last_year=("year", "max")
    )
    rows = []
    for (cc, party, abbr, official, ches_id), n in labels.items():
        in_ches = ches_id in known.index
        rows.append(
            {
                "country_code": cc,
                "party": party,
                "party_abbr": abbr,
                "party_official": official,
                "ches_party_id": ches_id,
                "ches_abbr": known.loc[ches_id, "ches_abbr"] if in_ches else "",
                "ches_last_year": known.loc[ches_id, "ches_last_year"] if in_ches else "",
                "id_matches_country": ches_country_code(ches_id) == cc,
                "tweets": n,
            }
        )
    review = pd.DataFrame(rows).sort_values(["country_code", "ches_party_id", "tweets"])
    review.to_csv(out_path, index=False)
    stale = review[
        (review["ches_last_year"] != "") & (review["ches_last_year"].astype(float) < 2019)
    ]
    print(
        f"Wrote {len(review)} party rows to {out_path} "
        f"({len(stale)} point to CHES parties that ended before 2019: check them)"
    )


# --------------------------------------------------------------------------- overrides
def learn_overrides(args: argparse.Namespace) -> None:
    """Turn your manual CHES fixes in the dataset into ``ches_overrides.csv``.

    Compares the CHES id of every dataset tweet with the id the source file assigns to
    the same tweet; label combinations that were consistently changed become overrides.
    """
    fixed = pd.read_csv(args.dataset, usecols=["tweet_id", "country_code", "ches_party_id"], dtype={"tweet_id": str})
    fixed = fixed.set_index("tweet_id")
    rows = []
    for chunk in pd.read_csv(
        args.source,
        usecols=["tweet_id", "party", "party_official", "ches_party_id"],
        dtype=str,
        chunksize=CHUNK_ROWS,
    ):
        chunk = chunk[chunk["tweet_id"].isin(fixed.index)].drop_duplicates("tweet_id")
        chunk = chunk.join(fixed, on="tweet_id", rsuffix="_fixed")
        rows.append(chunk)
    merged = pd.concat(rows)
    merged["source_id"] = merged["ches_party_id"].map(to_ches_id)
    merged["fixed_id"] = merged["ches_party_id_fixed"].astype(int)

    out, conflicts = [], 0
    labels = ["country_code", "party", "party_official"]
    for key, group in merged.groupby(labels, dropna=False):
        changed = group[group["source_id"] != group["fixed_id"]]
        if changed.empty:
            continue
        if changed["fixed_id"].nunique() > 1 or len(changed) < len(group):
            conflicts += 1  # same label, different fixes: needs a per-tweet decision
            print(f"Not a clean label rule, skipped: {key}")
            continue
        out.append({**dict(zip(labels, key)), "ches_party_id": int(changed["fixed_id"].iloc[0]), "was": int(changed["source_id"].iloc[0]), "tweets": len(group)})
    result = pd.DataFrame(out)
    result.to_csv(args.overrides, index=False)
    print(result.to_string(index=False) if len(result) else "No differences found.")
    print(f"\nWrote {len(result)} overrides to {args.overrides} ({conflicts} skipped)")


# --------------------------------------------------------------------------- fetch
def fetch(args: argparse.Namespace) -> None:
    import requests
    from tqdm.auto import tqdm

    planned = pd.read_csv(args.plan, dtype={"tweet_id": str, "uid": str})
    skipped_path = args.out.with_suffix(".skipped.txt")
    done: set[str] = set()
    collected: Counter[tuple[str, int]] = Counter()
    if args.out.exists():
        previous = pd.read_csv(args.out, dtype={"tweet_id": str})
        done.update(previous["tweet_id"])
        for country, ches in zip(previous["country"], previous["ches_party_id"]):
            collected[(country, int(ches))] += 1
    if skipped_path.exists():
        done.update(skipped_path.read_text().split())

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (thesis research; oembed text lookup)"
    written = attempts = 0
    write_header = not args.out.exists()
    try:
        with open(args.out, "a", newline="", encoding="utf-8") as out, open(
            skipped_path, "a", encoding="utf-8"
        ) as skipped:
            writer = csv.DictWriter(out, fieldnames=TARGET_COLS + ["tweet_text"])
            if write_header:
                writer.writeheader()
            for row in tqdm(planned.to_dict("records"), desc="Fetching tweets", unit="tweet"):
                if args.limit and attempts >= args.limit:
                    break
                stratum = (row["country"], int(row["ches_party_id"]))
                if row["tweet_id"] in done or collected[stratum] >= row["need"]:
                    continue
                attempts += 1
                text = get_tweet_text_oembed(session, row["tweet_id"])
                if text is not None:  # None = transient failure, retried next run
                    if is_valid_tweet_text(text):
                        writer.writerow(
                            {**{c: row[c] for c in TARGET_COLS}, "tweet_text": text.strip()}
                        )
                        out.flush()
                        collected[stratum] += 1
                        written += 1
                    else:
                        skipped.write(row["tweet_id"] + "\n")
                        skipped.flush()
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("Interrupted; re-run to resume.")
    print(f"Fetched {attempts} tweets, kept {written}, output: {args.out}")


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="Stream the source and sample tweet ids.")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--existing", type=Path, default=DEFAULT_TWEETS_OUT)
    p.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    p.add_argument("--out", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    p.add_argument("--ches", type=Path, default=DEFAULT_CHES)
    p.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    p.add_argument("--per-party", type=int, default=75, help="Target tweets per (country, party).")
    p.add_argument("--min-available", type=int, default=50, help="Skip parties with fewer source tweets.")
    p.add_argument("--oversample", type=float, default=1.6, help="Plan extra ids: deleted/URL-only tweets are dropped.")
    p.add_argument("--seed", type=int, default=42)

    o = sub.add_parser("learn-overrides", help="Derive ches_overrides.csv from your manual fixes.")
    o.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    o.add_argument("--dataset", type=Path, default=DEFAULT_TWEETS_OUT)
    o.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)

    f = sub.add_parser("fetch", help="Fetch tweet texts for the plan (resumable).")
    f.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    f.add_argument("--out", type=Path, default=DEFAULT_EXTRA)
    f.add_argument("--sleep", type=float, default=0.5, help="Seconds between requests.")
    f.add_argument("--limit", type=int, default=0, help="Max requests this run (0 = all).")

    args = parser.parse_args(argv)
    {"plan": plan, "fetch": fetch, "learn-overrides": learn_overrides}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
