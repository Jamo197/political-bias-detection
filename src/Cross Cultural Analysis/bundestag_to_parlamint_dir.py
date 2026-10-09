"""Copy Bundestag speeches into the ParlaMint export tree as country ``DE``.

RQ2 keeps every country in ONE collection per embedding model, so Germany has to
be chunked and ingested together with the ParlaMint countries. The Bundestag JSONs
already use the same schema; this script only

* keeps the shared RQ2 window (``--since`` / ``--until``, default 2020-2021),
* adds ``country_code="de"``, ``party_canonical`` and ``ches_party_id`` (via
  ``party_mapping.csv``), and
* writes them to ``<out>/DE/WP_<period>/<YYYY-MM>/speeches/DE_<file>``.

Usage::

    python bundestag_to_parlamint_dir.py
    python bundestag_to_parlamint_dir.py --since 2020-01-01 --until 2021-12-31 \\
        --out extraction/datasets/parlamint_data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cca_common import DEFAULT_MAPPING, DEFAULT_SPEECHES_DIR, HERE, PartyMapping

DEFAULT_SOURCE = HERE.parents[1] / "extraction" / "datasets" / "bundestag_data"
DEFAULT_SINCE = "2020-01-01"
DEFAULT_UNTIL = "2021-12-31"
COUNTRY_CODE = "de"


def convert(
    source: Path,
    out_root: Path,
    mapping: PartyMapping,
    since: str = DEFAULT_SINCE,
    until: str = DEFAULT_UNTIL,
) -> Counter[str]:
    """Copy all speeches with ``since <= date <= until``; returns speeches per party."""
    parties: Counter[str] = Counter()
    for path in sorted(source.rglob("speeches/*_cleaned.json")):
        speeches = json.loads(path.read_text(encoding="utf-8"))
        kept = []
        for speech in speeches:
            meta = speech.get("metadata", {})
            date = str(meta.get("date") or "")[:10]
            if not date or not since <= date <= until:
                continue
            canonical, ches_id = mapping.lookup(COUNTRY_CODE, meta.get("party"))
            meta.update(
                country="Germany",
                country_code=COUNTRY_CODE,
                party_canonical=canonical,
                ches_party_id=ches_id,
            )
            kept.append(speech)
            parties[f"{meta.get('party')} -> {canonical} ({ches_id})"] += 1
        if not kept:
            continue
        # .../WP_19/2020-01/speeches/<file> -> <out>/DE/WP_19/2020-01/speeches/DE_<file>
        rel = path.relative_to(source).parent
        target = out_root / "DE" / rel / f"DE_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return parties


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPEECHES_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument("--until", default=DEFAULT_UNTIL)
    args = parser.parse_args(argv)

    parties = convert(
        args.source, args.out, PartyMapping.from_csv(args.mapping), args.since, args.until
    )
    total = sum(parties.values())
    print(f"Wrote {total} German speeches ({args.since}..{args.until}) to {args.out / 'DE'}")
    for label, count in parties.most_common():
        print(f"  {count:6d}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
