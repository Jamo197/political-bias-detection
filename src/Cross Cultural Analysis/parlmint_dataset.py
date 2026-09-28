"""Utilities for loading, validating and exploring the cjvt/ParlaMint3 dataset.

Refactored from the ``parlamint_dataset.ipynb`` exploration notebook so the
loading and quality-check logic can be reused from regular Python code.

Typical usage::

    from parlmint_dataset import ParlaMintConfig, export_to_csv, load_full, quality_report

    config = ParlaMintConfig(config_name="lv")
    df = load_full(config, limit=100)            # downloads the archive, loads 100 rows
    df = load_full(config, since="2016")         # only rows from 2016 onward
    print(quality_report(df))

    export_to_csv(config, "lv.csv", since="2016")  # download, filter, save CSV

Note: ParlaMint3 ships TAR archives, which the ``datasets`` library cannot
stream, so ``load_sample()`` (streaming) always comes back empty for this
dataset. Use ``load_full()`` with ``limit`` to sample without the RAM cost
of the full split.

The module can also be run as a script::

    python parlmint_dataset.py --list-configs
    python parlmint_dataset.py --config lv --limit 100
    python parlmint_dataset.py --config lv --since 2016 --output   # recent rows -> parlmint_lv.csv
    python parlmint_dataset.py --config lv --since 2016 --limit 1000 -o lv_recent.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from datasets import get_dataset_config_names, load_dataset_builder
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATASET_ID = "cjvt/ParlaMint3"
DEFAULT_SPLIT = "train"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_ENV_FILE = Path(".env.local")

QUALITY_COLUMNS = ("ID", "text", "Date")
SPEAKER_METADATA_COLUMNS = (
    "Speaker_name",
    "Speaker_role",
    "Speaker_party",
    "Speaker_party_name",
    "Speaker_gender",
    "Speaker_MP",
    "Speaker_Minister",
)

__all__ = [
    "DATASET_ID",
    "QUALITY_COLUMNS",
    "SPEAKER_METADATA_COLUMNS",
    "ParlaMintConfig",
    "QualityReport",
    "environment_info",
    "list_configs",
    "load_dataframe",
    "load_full",
    "load_hf_token",
    "load_parlamint",
    "load_sample",
    "get_schema_fields",
    "parse_dates",
    "quality_report",
    "rows_from_final_year",
    "save_csv",
    "export_to_csv",
    "filter_since",
    "stream_records",
    "text_length_stats",
    "top_parties",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParlaMintConfig:
    """Settings identifying one ParlaMint3 configuration/split."""

    config_name: str | None = None  # None -> first available configuration
    revision: str | None = None  # commit hash for a reproducible snapshot
    split: str = DEFAULT_SPLIT
    dataset_id: str = DATASET_ID

    def resolved_config_name(self) -> str:
        """Return ``config_name``, defaulting to the first available configuration."""
        if self.config_name is not None:
            return self.config_name
        return list_configs(self.dataset_id, self.revision)[0]


def load_hf_token(env_file: str | Path = DEFAULT_ENV_FILE) -> None:
    """Load ``HF_TOKEN`` from an .env file into the environment, if present.

    The ``datasets`` library picks the token up from the environment
    automatically; nothing is changed when the file or variable is missing.
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    if not os.environ.get("HF_TOKEN"):
        logger.debug("HF_TOKEN is not set; gated datasets will be unavailable.")


def environment_info() -> dict[str, str]:
    """Versions of the relevant runtime components."""
    return {
        "python": sys.version.split()[0],
        "datasets": datasets.__version__,
        "pandas": pd.__version__,
    }


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def list_configs(
    dataset_id: str = DATASET_ID, revision: str | None = None
) -> list[str]:
    """Return all configuration names available for the dataset."""
    return get_dataset_config_names(dataset_id, revision=revision)


def load_parlamint(
    config: ParlaMintConfig, *, streaming: bool = True, split: str | None = None
) -> datasets.Dataset | datasets.IterableDataset:
    """Load the configured dataset split (streaming by default).

    ``split`` overrides ``config.split`` and accepts slicing syntax
    (e.g. ``"train[:100]"``).
    """
    return datasets.load_dataset(
        config.dataset_id,
        config.resolved_config_name(),
        split=split or config.split,
        revision=config.revision,
        streaming=streaming,
        trust_remote_code=True,
    )  # type: ignore


def stream_records(dataset: Iterable[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Materialize the first ``n`` records from an iterable dataset."""
    return list(islice(dataset, n))


def load_sample(
    config: ParlaMintConfig, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> pd.DataFrame:
    """Stream up to ``sample_size`` records and return them as a DataFrame.

    Returns an empty DataFrame when streaming is not supported for the
    configuration's source format (e.g. TAR archives).
    """
    try:
        streamed = load_parlamint(config, streaming=True)
        records = stream_records(streamed, sample_size)
    except NotImplementedError as error:
        logger.warning(
            "Streaming is unavailable for this configuration's source (%s). "
            "Use load_full() instead.",
            error,
        )
        return pd.DataFrame()
    logger.info("Streamed %d records.", len(records))
    return pd.DataFrame.from_records(records)


def _iso_cutoff(since: str) -> str:
    """Normalize a date or year string (e.g. ``"2016"``) to ``YYYY-MM-DD``."""
    return pd.to_datetime(since).date().isoformat()


def load_full(
    config: ParlaMintConfig,
    limit: int | None = None,
    since: str | None = None,
) -> pd.DataFrame:
    """Download the split and return it as a DataFrame.

    The data ships as one TAR archive per configuration, so the full archive
    is always downloaded and prepared. When ``limit`` is given, only the
    first ``limit`` rows are materialized in memory (split slicing), which
    avoids the RAM cost of the full split.

    ``since`` (e.g. ``"2016"`` or ``"2016-01-01"``) keeps only rows whose
    ``Date`` is on or after that day. The filter is applied before ``limit``
    and while the data is still in the memory-mapped Arrow format, so
    filtered-out rows never reach RAM.
    """
    if since is None:
        split = f"{config.split}[:{limit}]" if limit is not None else None
        dataset = load_parlamint(config, streaming=False, split=split)
    else:
        cutoff = _iso_cutoff(since)
        dataset = load_parlamint(config, streaming=False)
        total = len(dataset)
        # ISO dates compare correctly as strings; guard against missing values.
        dataset = dataset.filter(
            lambda example: str(example["Date"] or "") >= cutoff
        )
        logger.info(
            "Date filter (Date >= %s) kept %d of %d rows.",
            cutoff, len(dataset), total,
        )
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
    data = dataset.to_pandas()
    logger.info("Loaded %d rows and %d columns.", len(data), len(data.columns))
    return data


def load_dataframe(
    config: ParlaMintConfig,
    *,
    full: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    limit: int | None = None,
    since: str | None = None,
) -> pd.DataFrame:
    """Convenience dispatcher: full/sliced download or streamed sample.

    ``limit`` and ``since`` imply non-streaming mode: the archive is
    downloaded, but only matching rows are loaded into memory.
    """
    if full or limit is not None or since is not None:
        return load_full(config, limit=limit, since=since)
    return load_sample(config, sample_size)


def save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a DataFrame to a CSV file, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(data), path)
    return path


def export_to_csv(
    config: ParlaMintConfig,
    output_path: str | Path | None = None,
    *,
    limit: int | None = None,
    since: str | None = None,
) -> Path:
    """Download the configured split and save it as a CSV file.

    The full archive is always downloaded; ``limit`` keeps only the first
    ``limit`` rows and ``since`` (e.g. ``"2016"``) keeps only rows from that
    date onward. Defaults to ``parlmint_<config>.csv`` in the current
    working directory. Returns the path written.
    """
    data = load_full(config, limit=limit, since=since)
    if output_path is None:
        output_path = f"parlmint_{config.resolved_config_name()}.csv"
    return save_csv(data, output_path)


def get_schema_fields(config: ParlaMintConfig) -> list[str]:
    """Return the feature/column names of the configuration.

    Uses the dataset builder metadata, so no data is downloaded.
    """
    builder = load_dataset_builder(
        config.dataset_id,
        config.resolved_config_name(),
        revision=config.revision,
        trust_remote_code=True,
    )
    return list(builder.info.features)


# ---------------------------------------------------------------------------
# Quality checks and exploration
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Result of basic data-quality checks on a ParlaMint DataFrame."""

    row_count: int
    column_count: int
    dtypes: pd.Series
    missing_values: pd.DataFrame  # per QUALITY_COLUMNS: missing + empty strings
    duplicate_ids: int | None  # None when no "ID" column
    unparseable_dates: int | None  # None when no "Date" column
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    speaker_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Rows: {self.row_count:,}   Columns: {self.column_count}",
            "",
            "Data types:",
            self.dtypes.rename("dtype").to_frame().to_string(),
            "",
            "Missing values / empty strings:",
            self.missing_values.to_string(),
        ]
        if self.duplicate_ids is not None:
            lines.append(f"\nDuplicate IDs: {self.duplicate_ids:,}")
        if self.unparseable_dates is not None:
            lines.append(f"Unparseable dates: {self.unparseable_dates:,}")
        if self.date_range is not None:
            lines.append(f"Date range: {self.date_range[0]} to {self.date_range[1]}")
        lines.append(f"\nAvailable speaker metadata: {self.speaker_columns}")
        return "\n".join(lines)


def parse_dates(data: pd.DataFrame) -> pd.Series:
    """Parse the ``Date`` column to datetimes (invalid values become NaT)."""
    return pd.to_datetime(data["Date"], errors="coerce")


def filter_since(data: pd.DataFrame, since: str) -> pd.DataFrame:
    """Return the rows whose ``Date`` is on or after ``since``.

    ``since`` accepts a full date (``"2016-01-01"``) or just a year
    (``"2016"``). Rows with missing or unparseable dates are dropped. Use
    this to trim an already loaded DataFrame (or CSV export) to recent
    years; for new downloads prefer ``load_full(config, since=...)``, which
    filters before rows reach RAM.
    """
    if "Date" not in data.columns:
        logger.warning("No 'Date' column found; returning the data unfiltered.")
        return data
    cutoff = pd.to_datetime(since)
    filtered = data.loc[parse_dates(data) >= cutoff]
    logger.info(
        "Date filter (Date >= %s) kept %d of %d rows.",
        cutoff.date().isoformat(), len(filtered), len(data),
    )
    return filtered


def quality_report(data: pd.DataFrame) -> QualityReport:
    """Run basic data-quality checks on a ParlaMint DataFrame."""
    if data.empty:
        raise ValueError(
            "No row-level data available; load data first, e.g. with "
            "load_full(config, limit=N), or provide a non-empty DataFrame."
        )

    quality_columns = [c for c in QUALITY_COLUMNS if c in data.columns]
    missing_values = pd.DataFrame(
        {
            "missing": data[quality_columns].isna().sum(),
            "empty_strings": (data[quality_columns].fillna("") == "").sum(),
        }
    )

    duplicate_ids = int(data["ID"].duplicated().sum()) if "ID" in data.columns else None

    unparseable_dates = None
    date_range = None
    if "Date" in data.columns:
        parsed_dates = parse_dates(data)
        unparseable_dates = int(parsed_dates.isna().sum())
        date_range = (parsed_dates.min(), parsed_dates.max())

    speaker_columns = [c for c in SPEAKER_METADATA_COLUMNS if c in data.columns]

    return QualityReport(
        row_count=len(data),
        column_count=len(data.columns),
        dtypes=data.dtypes,
        missing_values=missing_values,
        duplicate_ids=duplicate_ids,
        unparseable_dates=unparseable_dates,
        date_range=date_range,
        speaker_columns=speaker_columns,
    )


def top_parties(data: pd.DataFrame, n: int = 10) -> pd.Series:
    """Speech-row counts for the ``n`` most frequent speaker parties."""
    if "Speaker_party_name" not in data.columns:
        return pd.Series(dtype="int64", name="speech_rows")
    return (
        data["Speaker_party_name"]
        .value_counts(dropna=False)
        .head(n)
        .rename("speech_rows")
    )


def rows_from_final_year(data: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    """Return rows dated within ``days`` of the most recent observation."""
    if "Date" not in data.columns:
        raise ValueError("Column 'Date' not present in the DataFrame.")
    parsed_dates = parse_dates(data)
    cutoff = parsed_dates.max() - pd.Timedelta(days=days)
    return data[parsed_dates >= cutoff]


def text_length_stats(data: pd.DataFrame, text_column: str = "text") -> pd.Series:
    """Descriptive statistics of text lengths (in characters)."""
    if text_column not in data.columns:
        raise ValueError(f"Column {text_column!r} not present in the DataFrame.")
    return data[text_column].fillna("").str.len().describe().rename("text_length")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config_name",
        default=None,
        help="Dataset configuration (default: first available).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision (commit hash) for a reproducible snapshot.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of records to stream in sample mode.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full split instead of streaming a sample "
        "(accept the download size and RAM cost first).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download the archive but only load the first N rows "
        "(implies non-streaming mode; keeps RAM usage low).",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help="Only keep rows with Date >= DATE (e.g. 2016 or 2016-01-01). "
        "Implies non-streaming mode; applied before --limit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="",
        default=None,
        metavar="CSV_PATH",
        help="Save the loaded data to a CSV file "
        "(default path: parlmint_<config>.csv). Implies non-streaming mode; "
        "without --limit the full split is exported.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Only list available configurations and exit.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the .env file containing HF_TOKEN.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_hf_token(args.env_file)

    try:
        configs = list_configs(revision=args.revision)
    except Exception as error:
        print(f"Error: could not fetch dataset configurations: {error}", file=sys.stderr)
        return 1

    if args.list_configs:
        for name, version in environment_info().items():
            print(f"{name}: {version}")
        print(f"dataset: {DATASET_ID}")
        print(f"\nFound {len(configs)} configurations: {', '.join(configs)}")
        return 0

    if args.config_name is not None and args.config_name not in configs:
        print(
            f"Error: unknown configuration {args.config_name!r}. "
            f"Choose one of: {', '.join(configs)}",
            file=sys.stderr,
        )
        return 2

    config = ParlaMintConfig(
        config_name=args.config_name or configs[0],
        revision=args.revision,
        split=args.split,
    )
    print(f"Selected configuration: {config.config_name}")

    # --output/--since/--limit/--full all imply a real (non-streaming) load;
    # without --limit that means the full split.
    download = (
        args.full
        or args.limit is not None
        or args.output is not None
        or args.since is not None
    )
    try:
        data = load_dataframe(
            config,
            full=download,
            sample_size=args.sample_size,
            limit=args.limit,
            since=args.since,
        )
    except Exception as error:
        print(f"Error: could not load configuration {config.config_name!r}: {error}", file=sys.stderr)
        return 1

    if data.empty:
        if download:
            print(
                "The query returned no rows. Check your --since and --limit "
                "values (the filter is applied before the limit)."
            )
        else:
            print(
                "No records were materialized. This dataset ships TAR archives, so "
                "streaming never works.\nRe-run with --limit N (downloads the "
                "archive, loads N rows), --full (loads everything) or "
                "--output FILE (downloads and saves to CSV)."
            )
        return 1

    if args.output is not None:
        output_path = (
            Path(args.output)
            if args.output
            else Path(f"parlmint_{config.config_name}.csv")
        )
        save_csv(data, output_path)
        print(f"Saved {len(data):,} rows to {output_path}")

    print(f"\nSchema fields: {get_schema_fields(config)}")

    print("\n--- Quality report ---")
    print(quality_report(data))

    print("\n--- Top parties ---")
    parties = top_parties(data)
    print(parties.to_frame().to_string() if not parties.empty else "n/a (no column)")

    if "Date" in data.columns:
        final_year_rows = rows_from_final_year(data)
        print(f"\nRows from the final observed year: {len(final_year_rows):,}")

    if "text" in data.columns:
        print("\n--- Text length ---")
        print(text_length_stats(data).to_frame().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
