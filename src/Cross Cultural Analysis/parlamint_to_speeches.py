"""Convert ParlaMint3 speeches into the Bundestag ``*_speeches_cleaned.json`` schema.

Each output file holds all speeches of one sitting day as a JSON list of::

    {"text": str, "interjections": [], "metadata": {speech_id, protocol_id,
     topic_id, speaker, party, role, date, year, country, legislative_period,
     source, party_status, speaker_gender, body}}

Layout (mirrors ``extraction/datasets/bundestag_data/WP_xx/YYYY-MM/speeches``)::

    <out>/<CC>/TERM_<term>/<YYYY-MM>/speeches/<CC>_<YYYY-MM-DD>_speeches_cleaned.json

Usage::

    python parlamint_to_speeches.py --list-configs
    python parlamint_to_speeches.py --config at lv --since 2017
    python parlamint_to_speeches.py --config at --since 2017 --limit 200
    python parlamint_to_speeches.py --config pl --mapping party_mapping.csv
    python parlamint_to_speeches.py --config at --since 2020 --until 2021-12-31 \
        --max-speeches 20000

With ``--mapping`` (built by ``party_mapping.py``) every speech also gets
``ches_party_id`` and ``party_canonical`` so it can be compared with CHES.

Speeches by the chair (``Speaker_role == "Chairperson"``) are dropped: they are
procedural ("Das Wort hat ...") and carry no party position.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cca_common import (
    COUNTRY_NAMES,
    DEFAULT_MAPPING,
    DEFAULT_SPEECHES_DIR,
    PartyMapping,
)
from parlmint_dataset import ParlaMintConfig, list_configs, load_full, load_hf_token

logger = logging.getLogger(__name__)

DEFAULT_OUT = DEFAULT_SPEECHES_DIR
DEFAULT_SINCE = "2020"
# RQ2 shared window: the Twitter Parliamentarian Database tweets end in 2021.
DEFAULT_UNTIL = "2021-12-31"
SKIP_ROLES = {"chairperson"}
SOURCE = "ParlaMint 3.0"

WHITESPACE_REGEX = re.compile(r"\s+")


def _clean(value: Any) -> str:
    """Normalize a possibly-missing cell to a stripped string."""
    if value is None or value != value:  # None or NaN
        return ""
    return WHITESPACE_REGEX.sub(" ", str(value)).strip()


def _flip_name(name: str) -> str:
    """ParlaMint "Last, First" -> Bundestag-style "First Last"."""
    if "," in name:
        last, first = (part.strip() for part in name.split(",", 1))
        return f"{first} {last}".strip()
    return name


def _protocol_id(speech_id: str) -> str:
    """Sitting id: the utterance ID without its trailing utterance part.

    Most corpora use ``<sitting>_<utterance>`` (AT: ``..._d7e826``); others
    (LV) use ``<sitting>-U<n>`` with only the country prefix before ``_``.
    """
    head = speech_id.rsplit("_", 1)[0]
    if re.fullmatch(r"ParlaMint-[A-Za-z-]+", head):
        return re.sub(r"[-.][A-Za-z]+\d+$", "", speech_id)
    return head


def _term(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def row_to_speech(
    row: dict[str, Any],
    country_code: str,
    mapping: PartyMapping | None = None,
) -> dict[str, Any]:
    """Map one ParlaMint row to the Bundestag speech record."""
    speech_id = _clean(row.get("ID"))
    date = _clean(row.get("Date"))
    party_code = _clean(row.get("Speaker_party"))
    party_name = _clean(row.get("Speaker_party_name"))
    party = party_code or party_name or "UNKNOWN"
    speech = {
        "text": _clean(row.get("text")),
        "interjections": [],
        "metadata": {
            "speech_id": speech_id,
            "protocol_id": _protocol_id(speech_id),
            "topic_id": _clean(row.get("Agenda")),
            "speaker": _flip_name(_clean(row.get("Speaker_name"))) or "UNKNOWN",
            "party": party,
            "role": _clean(row.get("Speaker_role")),
            "date": date,
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "country": COUNTRY_NAMES.get(country_code, country_code.upper()),
            "country_code": country_code,
            "legislative_period": _term(_clean(row.get("Term"))),
            "source": SOURCE,
            "party_status": _clean(row.get("Party_status")),
            "speaker_gender": _clean(row.get("Speaker_gender")),
            "body": _clean(row.get("Body")),
        },
    }
    if mapping is not None:
        # Party switchers can carry "A;B": fall back to the first listed code.
        first_code = party_code.split(";")[0].strip()
        canonical, ches_id = mapping.lookup(
            country_code, party_code, party_name, first_code
        )
        speech["metadata"]["party_canonical"] = canonical
        speech["metadata"]["ches_party_id"] = ches_id
    return speech


def export_country(
    config_name: str,
    out_root: Path = DEFAULT_OUT,
    since: str | None = DEFAULT_SINCE,
    limit: int | None = None,
    mapping: PartyMapping | None = None,
    until: str | None = DEFAULT_UNTIL,
    max_speeches: int | None = None,
    seed: int = 42,
) -> int:
    """Download one country, convert, and write per-sitting JSON files.

    ``max_speeches`` keeps a seeded random sample of that many speeches, a
    coarse pre-cap so large parliaments do not dominate chunking time (the
    exact per-country balance is done later by ``balance_chunks.py``).

    Returns the number of speeches written.
    """
    data = load_full(
        ParlaMintConfig(config_name=config_name), limit=limit, since=since, until=until
    )
    speeches = []
    for row in data.to_dict("records"):
        speech = row_to_speech(row, config_name, mapping)
        meta = speech["metadata"]
        if not speech["text"] or not meta["date"]:
            continue
        if meta["role"].casefold() in SKIP_ROLES:
            continue
        speeches.append(speech)
    if max_speeches is not None and len(speeches) > max_speeches:
        speeches = random.Random(seed).sample(speeches, max_speeches)

    sittings: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for speech in speeches:
        meta = speech["metadata"]
        key = (str(meta["legislative_period"]), meta["date"])
        sittings[key].append(speech)

    written = 0
    for (term, date), speeches in sittings.items():
        year_month = date[:7]
        directory = (
            out_root
            / config_name.upper()
            / f"TERM_{term or 'unknown'}"
            / year_month
            / "speeches"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{config_name.upper()}_{date}_speeches_cleaned.json"
        path.write_text(
            json.dumps(speeches, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += len(speeches)
    logger.info(
        "%s: wrote %d speeches in %d sittings to %s",
        config_name,
        written,
        len(sittings),
        out_root / config_name.upper(),
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", nargs="+", help="ParlaMint3 config(s), e.g. at lv.")
    parser.add_argument(
        "--since", default=DEFAULT_SINCE, help="Keep Date >= this (default 2020)."
    )
    parser.add_argument(
        "--until",
        default=DEFAULT_UNTIL,
        help=f"Keep Date <= this (default {DEFAULT_UNTIL}); 'none' disables it.",
    )
    parser.add_argument(
        "--max-speeches",
        type=int,
        default=None,
        help="Random sample of at most N speeches per country (seeded).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only first N rows per country."
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output root directory."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        nargs="?",
        const=DEFAULT_MAPPING,
        default=None,
        help="party_mapping.csv; adds ches_party_id + party_canonical (default file if flag has no value).",
    )
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--env-file", default=".env.local")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_hf_token(args.env_file)
    configs = list_configs()

    if args.list_configs or not args.config:
        print(", ".join(configs))
        return 0

    unknown = [c for c in args.config if c not in configs]
    if unknown:
        print(f"Unknown configuration(s): {unknown}. Choose from: {', '.join(configs)}")
        return 2

    mapping = PartyMapping.from_csv(args.mapping) if args.mapping else None
    for name in args.config:
        export_country(
            name,
            args.out,
            since=args.since,
            limit=args.limit,
            mapping=mapping,
            until=None if args.until.lower() == "none" else args.until,
            max_speeches=args.max_speeches,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
