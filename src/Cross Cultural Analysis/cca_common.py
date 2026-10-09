"""Shared helpers for the cross-cultural dataset: country codes, CHES ids, party mapping.

Kept free of heavy dependencies (no ``datasets``/``pandas``) so it can be imported
from anywhere in the cross-cultural scripts.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_MAPPING = HERE / "party_mapping.csv"
DEFAULT_SPEECHES_DIR = HERE.parents[1] / "extraction" / "datasets" / "parlamint_data"
DEFAULT_TWEETS_IN = HERE.parent / "datasets" / "EU_tweets_with_parties_dataset.csv"
DEFAULT_TWEETS_OUT = HERE.parent / "datasets" / "EU_tweets_clean.csv"
DEFAULT_CHES = HERE.parent / "datasets" / "ground_truth" / "1999-2024_CHES.csv"

UNKNOWN = "UNKNOWN"

# ParlaMint-style lowercase codes. Includes tweet-only countries (ie, cy, lu, mt, sk).
COUNTRY_NAMES = {
    "at": "Austria",
    "ba": "Bosnia and Herzegovina",
    "be": "Belgium",
    "bg": "Bulgaria",
    "cy": "Cyprus",
    "cz": "Czechia",
    "dk": "Denmark",
    "ee": "Estonia",
    "es": "Spain",
    "es-ct": "Spain (Catalonia)",
    "es-ga": "Spain (Galicia)",
    "es-pv": "Spain (Basque Country)",
    "fi": "Finland",
    "fr": "France",
    "gb": "United Kingdom",
    "gr": "Greece",
    "hr": "Croatia",
    "hu": "Hungary",
    "ie": "Ireland",
    "is": "Iceland",
    "it": "Italy",
    "lu": "Luxembourg",
    "lv": "Latvia",
    "mt": "Malta",
    "nl": "Netherlands",
    "no": "Norway",
    "pl": "Poland",
    "pt": "Portugal",
    "rs": "Serbia",
    "se": "Sweden",
    "si": "Slovenia",
    "sk": "Slovakia",
    "tr": "Turkey",
    "ua": "Ukraine",
    "de": "Germany",
}
NAME_ALIASES = {"Czech Republic": "Czechia"}
NAME_TO_CODE = {name: code for code, name in COUNTRY_NAMES.items()}
NAME_TO_CODE.update({alias: NAME_TO_CODE[name] for alias, name in NAME_ALIASES.items()})

# The leading digits of a CHES party id encode the country (id // 100).
CHES_COUNTRY = {
    1: "be", 2: "dk", 3: "de", 4: "gr", 5: "es", 6: "fr", 7: "ie", 8: "it",
    10: "nl", 11: "gb", 12: "pt", 13: "at", 14: "fi", 16: "se", 20: "bg",
    21: "cz", 22: "ee", 23: "hu", 24: "lv", 25: "lt", 26: "pl", 27: "ro",
    28: "sk", 29: "si", 31: "hr", 37: "mt", 38: "lu", 40: "cy",
}  # fmt: skip

# Tweet `country` cells that hold a region / constituency instead of a country.
REGION_TO_COUNTRY = {
    "Pyrénées-Orientales": "France",
    "Hauts-de-Seine": "France",
    "Varsinais-Suomi": "Finland",
    "Cavan-Monaghan": "Ireland",
    "West-Vlaanderen": "Belgium",
}

WHITESPACE_REGEX = re.compile(r"\s+")


def clean_label(value: Any) -> str:
    """Whitespace-normalized string; None/NaN -> ''."""
    if value is None or value != value:
        return ""
    return WHITESPACE_REGEX.sub(" ", str(value)).strip()


def label_key(value: Any) -> str:
    """Case-insensitive lookup key for party labels."""
    return clean_label(value).casefold()


def ches_country_code(ches_id: Any) -> str | None:
    """Country code implied by a CHES party id (e.g. 2605 -> 'pl')."""
    try:
        return CHES_COUNTRY.get(int(ches_id) // 100)
    except (TypeError, ValueError):
        return None


def to_ches_id(value: Any) -> int | None:
    text = clean_label(value)
    try:
        return int(float(text)) if text else None
    except ValueError:
        return None


class PartyMapping:
    """Lookup ``(country_code, raw party label) -> (canonical_party, ches_party_id)``."""

    def __init__(self, rows: list[dict[str, str]] | None = None):
        self._lookup: dict[tuple[str, str], tuple[str, int]] = {}
        for row in rows or []:
            ches = to_ches_id(row.get("ches_party_id"))
            raw = clean_label(row.get("raw_label"))
            if ches is None or not raw:
                continue
            key = (clean_label(row.get("country_code")).lower(), label_key(raw))
            self._lookup[key] = (clean_label(row.get("canonical_party")) or raw, ches)

    @classmethod
    def from_csv(cls, path: Path | str) -> "PartyMapping":
        with open(path, newline="", encoding="utf-8") as handle:
            return cls(list(csv.DictReader(handle)))

    def lookup(self, country_code: str, *labels: Any) -> tuple[str, int | None]:
        """First matching label wins; unmapped -> ('UNKNOWN', None)."""
        for label in labels:
            hit = self._lookup.get((country_code.lower(), label_key(label)))
            if hit:
                return hit
        return UNKNOWN, None


def apply_party_mapping(
    mapping: PartyMapping, country_code: str, *labels: Any
) -> tuple[str, int | None]:
    """Convenience wrapper around :meth:`PartyMapping.lookup`."""
    return mapping.lookup(country_code, *labels)
