"""Fix the country of every tweet and add ``country_code`` / ``canonical_party``.

The raw tweet CSV has regions (``Hauts-de-Seine``) and ``European Parliament`` in the
``country`` column. The leading digits of ``ches_party_id`` encode the country, so
those rows are resolved from the CHES id. Rows whose listed country is a real country
but disagrees with their CHES id are reported (and the listed country is kept).

Usage::

    python "src/Cross Cultural Analysis/clean_tweets.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cca_common import (
    COUNTRY_NAMES,
    DEFAULT_CHES,
    DEFAULT_TWEETS_IN,
    DEFAULT_TWEETS_OUT,
    NAME_ALIASES,
    NAME_TO_CODE,
    REGION_TO_COUNTRY,
    ches_country_code,
    clean_label,
)


def resolve_country(country: str, ches_id: object) -> tuple[str | None, str]:
    """Return ``(country_code, how)`` where ``how`` says how it was resolved."""
    country = clean_label(country)
    if country in NAME_TO_CODE:
        return NAME_TO_CODE[country], "listed"
    if country in REGION_TO_COUNTRY:
        return NAME_TO_CODE[REGION_TO_COUNTRY[country]], "region"
    code = ches_country_code(ches_id)
    return code, "ches" if code else "unresolved"


def ches_abbreviations(ches_path: Path) -> dict[int, str]:
    """CHES party id -> abbreviation (latest wave), e.g. 2605 -> 'PiS'."""
    ches = pd.read_csv(ches_path).sort_values("year")
    return ches.groupby("party_id")["party"].last().map(clean_label).to_dict()


def clean_tweets(tweets: pd.DataFrame, ches_path: Path = DEFAULT_CHES) -> pd.DataFrame:
    out = tweets.copy()
    out["country_original"] = out["country"]

    resolved = [
        resolve_country(country, ches)
        for country, ches in zip(out["country"], out["ches_party_id"])
    ]
    out["country_code"] = [code for code, _ in resolved]
    out["country_resolution"] = [how for _, how in resolved]
    out["country"] = out["country_code"].map(COUNTRY_NAMES)

    # Canonical party name: the CHES abbreviation of the id (fallback: most frequent
    # official name for ids missing from the CHES file).
    abbreviations = ches_abbreviations(ches_path)
    fallback = out.groupby("ches_party_id")["party_official"].agg(
        lambda s: s.map(clean_label).value_counts().index[0]
    )
    out["canonical_party"] = [
        abbreviations.get(int(i)) or fallback[i] for i in out["ches_party_id"]
    ]
    return out


def report(cleaned: pd.DataFrame) -> None:
    print("Tweets per country:")
    print(cleaned["country"].fillna("UNRESOLVED").value_counts().to_string(), "\n")

    changed = cleaned[cleaned["country_resolution"].isin(["region", "ches"])]
    if len(changed):
        print(f"{len(changed)} rows had a non-country value; resolved as:")
        print(
            changed[["country_original", "country", "country_resolution", "party"]]
            .drop_duplicates()
            .to_string(index=False),
            "\n",
        )

    # Listed country vs. country implied by the CHES id.
    implied = cleaned["ches_party_id"].map(ches_country_code)
    conflicts = cleaned[implied.notna() & (implied != cleaned["country_code"])]
    if len(conflicts):
        print(f"WARNING: {len(conflicts)} rows where the CHES id disagrees with the country:")
        print(
            conflicts[["country", "ches_party_id", "party_official"]]
            .drop_duplicates()
            .to_string(index=False),
            "\n",
        )

    missing = cleaned[cleaned["country_code"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} rows without a country:")
        print(missing[["country_original", "party", "ches_party_id"]].to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TWEETS_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_TWEETS_OUT)
    parser.add_argument("--ches", type=Path, default=DEFAULT_CHES)
    args = parser.parse_args(argv)

    cleaned = clean_tweets(pd.read_csv(args.input), args.ches)
    report(cleaned)
    cleaned.to_csv(args.output, index=False)
    print(f"\nWrote {len(cleaned)} tweets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
