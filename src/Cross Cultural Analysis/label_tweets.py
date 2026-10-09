"""Build the RQ2 evaluation set: CHES labels, party family and ``party_cue`` per tweet.

Input is the cleaned tweet file (``EU_tweets_clean.csv``, CHES id per tweet). For
each tweet this adds

* ``label_lrgen`` / ``label_lrecon`` / ``label_galtan``: CHES scores of the party
  from the wave nearest to the tweet year (2019 for 2020-21 tweets), rescaled
  from 0-10 to the project scale with ``1 + 0.6 * x`` (0 -> 1, 5 -> 4, 10 -> 7).
  ``ches_wave`` records which wave was used.
* ``ches_family`` (CHES family name) for the within-family analysis.
* ``party_cue``: True if the tweet names a party of its country (party names,
  abbreviations, party handles or hashtags), so results can be split by it.
* ``text_id`` (= ``tweet_id``), the stable id used in the run logs.

Parties with fewer than ``--min-party-tweets`` tweets are dropped, larger ones are
capped at ``--max-per-party`` (seeded sample).

Usage::

    python label_tweets.py                                # all countries
    python label_tweets.py --countries at de --max-per-party 75
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from cca_common import (
    DEFAULT_CHES,
    DEFAULT_MAPPING,
    DEFAULT_TWEETS_OUT,
    HERE,
    ches_country_code,
    clean_label,
)

DEFAULT_OUT = HERE.parent / "datasets" / "EU_tweets_rq2.csv"

CHES_FAMILIES = {
    1: "radical right",
    2: "conservative",
    3: "liberal",
    4: "christian democrat",
    5: "socialist",
    6: "radical left",
    7: "green",
    8: "regionalist",
    9: "no family",
    10: "confessional",
    11: "agrarian/centre",
}
DIMENSIONS = {"lrgen": "label_lrgen", "lrecon": "label_lrecon", "galtan": "label_galtan"}

# Party accounts and spellings the mapping does not contain. Matched like any
# other cue (handles / hashtags as substrings of @/# tokens).
EXTRA_CUES: dict[str, list[str]] = {
    "at": ["SPÖ", "ÖVP", "FPÖ", "Grüne", "Grünen", "NEOS", "spoe", "oevp", "fpoe",
           "volkspartei", "neos_lab", "gruene_austria"],
    "de": ["SPD", "CDU", "CSU", "FDP", "AfD", "Grüne", "Grünen", "Linke", "LINKE",
           "spdde", "cducsubt", "fdpbt", "afdimbundestag", "gruenebundestag",
           "linksfraktion", "dielinke"],
}  # fmt: skip

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})
_TOKEN = re.compile(r"[@#]([\w\-]+)")


def ascii_variants(term: str) -> set[str]:
    """``FPÖ`` -> {``FPÖ``, ``FPOE``, ``FPO``}: CHES and handles drop umlauts."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", term) if not unicodedata.combining(c)
    )
    return {term, term.translate(_FOLD), stripped}


def nearest_wave_labels(ches: pd.DataFrame, years: list[int]) -> dict[tuple[int, int], dict]:
    """``(party_id, year) -> {wave, label_*, family}`` from the nearest CHES wave."""
    by_party = {pid: grp.sort_values("year") for pid, grp in ches.groupby("party_id")}
    out = {}
    for pid, grp in by_party.items():
        grp = grp.dropna(subset=list(DIMENSIONS), how="all")
        if grp.empty:
            continue
        for year in years:
            # Nearest wave; ties go to the earlier (already published) wave.
            row = grp.loc[(grp["year"] - year).abs().add(grp["year"] > year).idxmin()]
            out[(pid, year)] = {
                "ches_wave": int(row["year"]),
                **{col: round(1 + 0.6 * row[dim], 3) for dim, col in DIMENSIONS.items()},
                "ches_family": CHES_FAMILIES.get(int(row["family"]), "unknown")
                if pd.notna(row["family"])
                else "unknown",
            }
    return out


def cue_terms(
    mapping_path: Path, ches: pd.DataFrame, since_wave: int = 2019
) -> dict[str, set[str]]:
    """Party cue terms per country from the mapping, CHES abbreviations and EXTRA_CUES.

    Only CHES parties present since ``since_wave`` are used: old abbreviations
    (e.g. Polish ``RP``) collide with ordinary words.
    """
    terms: dict[str, set[str]] = defaultdict(set)
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    for rec in mapping.to_dict("records"):
        for label in (rec["raw_label"], rec["canonical_party"]):
            if clean_label(label):
                terms[rec["country_code"]].add(clean_label(label))
    recent = ches[ches["year"] >= since_wave]
    for pid, abbr in recent[["party_id", "party"]].drop_duplicates().itertuples(index=False):
        code = ches_country_code(pid)
        if code and clean_label(abbr):
            terms[code].add(clean_label(abbr))
    for code, extra in EXTRA_CUES.items():
        terms[code].update(extra)
    return terms


class PartyCueMatcher:
    """Flags tweets that name a party of their country.

    * Multi-word names (``Social Democratic Party``) match case-insensitively.
    * Single words / abbreviations match case-sensitively as whole words, so the
      adjective ``grünen`` or the word ``union`` in lower case do not count.
    * In ``@handles`` and ``#hashtags`` any term of 3+ letters counts as a
      substring (``@fpoe_tv`` contains ``fpoe``).
    """

    def __init__(self, terms: set[str]):
        variants = {v for t in terms for v in ascii_variants(t) if len(v) >= 2}
        multi = sorted({v for v in variants if " " in v}, key=len, reverse=True)
        single = sorted({v for v in variants if " " not in v}, key=len, reverse=True)
        self.multi = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, multi)) + r")(?!\w)", re.I)
            if multi
            else None
        )
        self.single = (
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, single)) + r")(?!\w)")
            if single
            else None
        )
        self.handle_terms = {
            v.casefold().replace(" ", "") for v in variants if len(v) >= 3
        }

    def __call__(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        if self.multi and self.multi.search(text):
            return True
        if self.single and self.single.search(text):
            return True
        for token in _TOKEN.findall(text):
            folded = {v.casefold() for v in ascii_variants(token)}
            if any(term in tok for tok in folded for term in self.handle_terms):
                return True
        return False


def build(
    tweets: pd.DataFrame,
    ches: pd.DataFrame,
    mapping_path: Path,
    countries: list[str] | None,
    min_party_tweets: int,
    max_per_party: int | None,
    seed: int,
) -> pd.DataFrame:
    df = tweets.dropna(subset=["ches_party_id", "tweet_text", "country_code"]).copy()
    if countries:
        df = df[df["country_code"].isin(countries)]
    df["ches_party_id"] = df["ches_party_id"].astype(int)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.fillna(2020).astype(int)

    labels = nearest_wave_labels(ches, sorted(df["year"].unique()))
    extra = [labels.get((pid, yr), {}) for pid, yr in zip(df["ches_party_id"], df["year"])]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    missing = df["label_lrgen"].isna()
    if missing.any():
        print(f"Dropping {missing.sum()} tweets whose CHES id has no scores:")
        print(df[missing].groupby(["country_code", "ches_party_id"]).size().to_string())
    df = df[~missing]

    sizes = df.groupby(["country_code", "ches_party_id"])["tweet_id"].transform("size")
    small = df[sizes < min_party_tweets]
    if len(small):
        print(f"Dropping {len(small)} tweets of parties with < {min_party_tweets} tweets:")
        print(small.groupby(["country_code", "canonical_party"]).size().to_string())
    df = df[sizes >= min_party_tweets]
    if max_per_party:
        df = (
            df.sample(frac=1, random_state=seed)
            .groupby(["country_code", "ches_party_id"])
            .head(max_per_party)
        )

    matchers = {code: PartyCueMatcher(t) for code, t in cue_terms(mapping_path, ches).items()}
    df["party_cue"] = [
        matchers[code](text) if code in matchers else False
        for code, text in zip(df["country_code"], df["tweet_text"])
    ]
    df["text_id"] = df["tweet_id"].astype(str)
    return df.sort_values(["country_code", "ches_party_id", "text_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TWEETS_OUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ches", type=Path, default=DEFAULT_CHES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--min-party-tweets", type=int, default=20)
    parser.add_argument("--max-per-party", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = build(
        pd.read_csv(args.input, dtype={"tweet_id": str}),
        pd.read_csv(args.ches),
        args.mapping,
        args.countries,
        args.min_party_tweets,
        args.max_per_party,
        args.seed,
    )
    df.to_csv(args.output, index=False)
    summary = df.groupby(["country_code", "canonical_party"]).agg(
        tweets=("text_id", "size"),
        lrgen=("label_lrgen", "first"),
        family=("ches_family", "first"),
        party_cue=("party_cue", "mean"),
    )
    print(summary.round(2).to_string())
    print(f"\nWrote {len(df)} tweets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
