"""why_rag_analysis.py

Data for the thesis section 'Why retrieval helps or hurts' (results chapter 04).

Scope (all subcommands): RAG condition e5/simple (k = 5) against the no-RAG
baseline of the same model, ten models (Llama 3B / 8B / 70B, Ministral-3 3B /
8B / 14B, Qwen 3B / 7B / 32B / 72B). The log pairs are the ones listed in
results/qualitative/<family>/<size>/qualitative_e5_simple_diagnostics.json;
only posts with a parseable score in both runs are kept (inner join on
text_index).

Positions: CHES 2024 lrgen of the author's party on the 1-7 scale
(1 + lrgen / 10 * 6), rounded half-up to two decimals (DIE LINKE 1.85,
Grüne 2.83, SPD 3.08, BSW 3.18, FDP 4.63, CDU 4.95, CSU 5.54, AfD 6.56). The
same values are used for the target and for the retrieved chunks; the
Bundestag faction CDU/CSU gets the CDU value. Chunks of non-attached speakers
('Fraktionslos') or of unknown party get no position and are counted.
Blocs: left < 3.4 <= centre < 5.2 <= right.

Quadrants of the annotated sample (Q1 RAG win, Q2 RAG distraction, Q3 joint
failure, Q4 baseline sufficient) are the ones the texts were SAMPLED in, i.e.
computed with the one-decimal label_ideology of the logs. Sampling weights
(population texts of the quadrant / sampled texts) use the same definition.

Llama-3.2-3B returns scores below 1 for some posts without RAG; every
Llama-3B result is reported twice: on all paired posts and on the posts with a
baseline score >= 1.

Subcommands and outputs (results/thesis/04_why_rag/):

  judge   the LLM judge: human agreement, judge vs humans, review procedure,
          transfer check and final label distributions
          judge_human_agreement.csv     human-human and judge-human agreement
          judge_calibration_labels.csv  label shares on the 125 calibration chunks
          judge_review_summary.csv      label sources, flags, spot checks per model
          judge_transfer_check.csv      identical chunk lists across models
          judge_label_distribution.csv  final labels per model (weighted / unweighted)
          judge_sampling_weights.csv    population / sampled texts per model and quadrant
          Human agreement is reported in its archived state (disagreement CSV
          before commit e58fb1c, read from git); the later correction is a note.

  composition  which parties e5/simple retrieves (computed once per text: the
          chunk lists are identical for all models), against the knowledge base
          kb_party_distribution.csv       knowledge base per party (chunks, or speeches as fallback)
          retrieval_chance_level.csv      observed vs. random-chunk expectation and permutation null
          retrieval_composition_matrix.csv author party x chunk party: share, CI, lift
          retrieval_by_author_party.csv   same-party / same-bloc / position / distance per author party
          retrieval_rank_position.csv     same-party share by rank, rank 1 vs. rank 5
          retrieval_party_mentions.csv    party names in the chunk text
          fig_composition_heatmap.pdf/.png
          04_why_rag_composition.md
          Knowledge base: --kb rag/ingest/artifacts/chunks.jsonl (indexed chunk file)
          or a CSV party,chunks counted from it; without it, the speech counts in
          src/datasets/knowledge_base/kb_speech_counts.csv are used and labelled
          as an approximation.
          fig_judge_labels.pdf/.png     stacked R_top / R_ideo shares per model
          04_why_rag_judge.md           report

Uncertainty: bootstrap 95 % CIs (2000 resamples, seed 42), resampling texts
(stratified by quadrant within a model; clustered by text_index when models
are pooled, because all models judge the same texts).

  direction  more left / right / centrist with RAG, per model and author party
          direction_overall.csv, direction_movement.csv, direction_by_party.csv,
          direction_compression.csv, direction_lmm.csv, fig_direction_party, fig_direction_extremes,
          04_why_rag_direction.md

  extremes  DIE LINKE and AfD posts: evidence use, where the evidence points, judge
          labels, score-vs-justification check and case listings
          extremes_evidence_use.csv (+ _contrasts), extremes_evidence_gap.csv,
          extremes_pull_by_party.csv, extremes_judge_labels.csv, extremes_score_justification.csv,
          extremes_cases_llama70b_afd.csv, extremes_cases_contrast.csv, extremes_cases_linke.csv,
          04_why_rag_extremes.md (examples with hand-written glosses in CASE_GLOSSES)

  labels  judge labels in the annotated sample (quadrant-weighted, descriptive)
          labels_tier_comparison.csv (tier 'all' vs 'reviewed' refitted on model subsets),
          labels_by_chunk_party.csv, labels_relevance_party.csv, labels_vs_error.csv,
          labels_relevance_party_quadrant.csv, 04_why_rag_labels.md
          The chunk_influence_all / chunk_influence_reviewed folders come from
          analyze_chunk_influence.py --labels final --tier all|reviewed --out results/thesis/04_why_rag/...

  explanatory  M1-M3 of evaluate_metrics (abs_error, full main matrix, hyde_hybrid
          excluded, re-run Ministral-8B cells) refitted with crossed random effects for
          text and model (M1/M2 plus a random RAG slope per model), a two-stage check
          (10 model effects, t-based CIs) and the old specification for comparison
          explanatory_models.csv, explanatory_variance_components.csv, fig_config_m3,
          04_why_rag_explanatory.md  (~15 min; --report-only rebuilds from the CSVs)

Usage:  python src/why_rag_analysis.py judge [--n-boot 2000] [--seed 42]
        python src/why_rag_analysis.py direction
        python src/why_rag_analysis.py extremes
        python src/why_rag_analysis.py composition [--kb kb_party_counts.csv] [--n-boot 2000] [--seed 42]
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # evaluate_metrics uses paths relative to the repository root
import evaluate_metrics as em  # noqa: E402
import retrieval_config_analysis as rca  # noqa: E402

RAG_DIR = ROOT / "RAG Analysis"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Log merging, quadrant rule and party normalisation of the chunk-influence analysis.
aci = _load_module("analyze_chunk_influence", RAG_DIR / "analyze_chunk_influence.py")
strata = aci.strata
# Krippendorff's alpha, kappa, rating table and consensus rule of the human comparison.
cha = _load_module(
    "compare_human_annotations",
    RAG_DIR / "human_annotations" / "compare_human_annotations.py",
)

OUT_DIR = ROOT / em.THESIS_ROOT / "04_why_rag"
DIAGNOSTICS = (
    "results/qualitative/{family}/{size}/qualitative_e5_simple_diagnostics.json"
)
FAMILIES_DIR = RAG_DIR / "LLM judge results" / "families"
HUMAN_DIR = RAG_DIR / "human_annotations"
CONSENSUS_FILE = HUMAN_DIR / "revised" / "consensus_reference_annotations.jsonl"
DISAGREEMENT_CSV = HUMAN_DIR / "revised" / "Annotations - annotation_disagreements.csv"
# The revised report, the consensus file, the judge calibration and the final labels
# were built before commit e58fb1c ('fixed wrong annotations', 29 Oct 2026), which
# changed three chunks in the disagreement CSV. The thesis reports the archived
# state; the CSV as it was before that commit reproduces the revised report exactly
# and is read from git for the bootstrap. The effect of the fix is reported as a note.
ARCHIVED_CSV_REV = "e58fb1c^"
HUMAN_REPORT = HUMAN_DIR / "human_agreement_report.json"
HUMAN_REPORT_REVISED = HUMAN_DIR / "human_agreement_report_revised.json"
JUDGE_RUN = "2026-10-23_142551"
JUDGE_RUN_DIR = RAG_DIR / "LLM judge results" / f"{JUDGE_RUN}_ideo_cuts"
RANGE_REPORT = (
    RAG_DIR
    / "LLM calib"
    / "range_restriction_runs"
    / "archive"
    / f"range_restriction_{JUDGE_RUN}_ideo_cuts.json"
)
CUTPOINT_FILE = (
    RAG_DIR / "LLM calib" / "judge_calibration" / f"posthoc_quantile_{JUDGE_RUN}.json"
)
CALIBRATION_LOG = RAG_DIR / "LLM calib" / "judge_calibration" / "CALIBRATION_LOG.md"
REVIEW_SUMMARY = FAMILIES_DIR / "review_summary.json"

# (folder name, family, size, display name), ordered by family and size.
MODELS = [
    ("llama_3b", "llama", "3b", "Llama 3B"),
    ("llama_8b", "llama", "8b", "Llama 8B"),
    ("llama_70b", "llama", "70b", "Llama 70B"),
    ("mistral_3b", "mistral", "3b", "Ministral 3B"),
    ("mistral_8b", "mistral", "8b", "Ministral 8B"),
    ("mistral_14b", "mistral", "14b", "Ministral 14B"),
    ("qwen_3b", "qwen", "3b", "Qwen 3B"),
    ("qwen_7b", "qwen", "7b", "Qwen 7B"),
    ("qwen_32b", "qwen", "32b", "Qwen 32B"),
    ("qwen_72b", "qwen", "72b", "Qwen 72B"),
]
DISPLAY = {m[0]: m[3] for m in MODELS}
LLAMA_3B = "llama_3b"
SCOPE_ALL = "all paired posts"
SCOPE_GE1 = "baseline score >= 1"
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
DIMS = ("R_top", "R_ideo", "A_caus")
LEVELS = (1, 2, 3, 4, 5)
LABEL_SOURCES = (
    "judge",
    "judge_seen",
    "human_review",
    "human_changed",
    "spot_check",
    "human_consensus",
    "transferred",
)
REVIEWED_MODELS = aci.REVIEWED_MODELS

# Ordinal ramp (one hue, light -> dark) for label levels 1-5; validated with the
# dataviz palette validator (--ordinal, light surface).
LEVEL_COLOURS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


# ---------------------------------------------------------------------------
# Positions and paired data (shared by all subcommands)
# ---------------------------------------------------------------------------


def _round2(x: float) -> float:
    return float(
        Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def load_positions() -> Dict[str, float]:
    """CHES std_party -> lrgen on 1-7, rounded half-up to two decimals."""
    return {
        p: _round2(v)
        for p, v in em.load_ches_lrgen(ROOT / em.CHES_PATH, em.CHES_WAVE).items()
    }


POSITIONS = load_positions()


def chunk_party(raw) -> Optional[str]:
    """Canonical CHES party of a chunk (CDU/CSU -> CDU); None = no position."""
    return aci.norm_party(raw)


def load_pairs() -> pd.DataFrame:
    """One row per model x paired post (both scores parseable).

    Columns: model, text_index, input_party, target (2 dp), target_log (1 dp
    label_ideology), B, R, quadrant_sampling (1 dp rule), chunk_parties (list),
    chunk_keys (tuple identifying the retrieved chunk list)."""
    rows, unmapped = [], Counter()
    for folder, family, size, _ in MODELS:
        diag_path = ROOT / DIAGNOSTICS.format(family=family, size=size)
        if not diag_path.exists():
            raise SystemExit(f"Missing diagnostics file {diag_path}")
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        thr = diag.get("thresholds") or {"pass": 0.5, "fail": 1.0}
        for p in diag["baseline_paths"] + diag["rag_paths"]:
            if not (ROOT / p).exists():
                raise SystemExit(f"Missing log {p} ({folder})")
        base = strata._merge_logs([ROOT / p for p in diag["baseline_paths"]])
        rag = strata._merge_logs([ROOT / p for p in diag["rag_paths"]])
        if base.duplicate_ids or rag.duplicate_ids:
            raise SystemExit(f"{folder}: duplicate text_index in the logs")
        # Sorted: set order depends on per-process string hashing, and the bootstrap
        # draws must map to the same posts in every run.
        common = sorted(
            set(base.records) & set(rag.records),
            key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
        )
        for text_index in common:
            b_rec, r_rec = base.records[text_index], rag.records[text_index]
            b, gt_b, _ = strata.extract_scalar_prediction(b_rec, "label_ideology")
            r, gt_r, _ = strata.extract_scalar_prediction(r_rec, "label_ideology")
            if None in (b, gt_b, r, gt_r):
                continue
            if abs(gt_b - gt_r) > 1e-6:
                raise SystemExit(
                    f"{folder}: ground truth differs between runs for text {text_index}"
                )
            raw_party = (r_rec.get("input_metadata") or {}).get("party")
            party = aci.norm_party(raw_party)
            if party not in POSITIONS:
                unmapped[(folder, raw_party)] += 1
                continue
            chunks = (r_rec.get("inputs") or {}).get("retrieved_chunks") or []
            rows.append(
                {
                    "model": folder,
                    "text_index": str(text_index),
                    "input_party": party,
                    "target": POSITIONS[party],
                    "target_log": gt_r,
                    "B": b,
                    "R": r,
                    "quadrant_sampling": aci.quadrant_short(
                        strata.classify_quadrant(
                            abs(b - gt_r), abs(r - gt_r), thr["pass"], thr["fail"]
                        )
                    ),
                    "chunk_parties": [chunk_party(c.get("party")) for c in chunks],
                    "chunk_keys": tuple(rca._ckey(c) for c in chunks),
                }
            )
    if unmapped:
        raise SystemExit(f"Input parties without a CHES position: {dict(unmapped)}")
    return pd.DataFrame(rows)


def rag_chunk_lists() -> Dict[str, Dict[str, tuple]]:
    """model -> text_index -> chunk key tuple, for EVERY record of the RAG log
    (also posts whose score could not be parsed)."""
    out = {}
    for folder, family, size, _ in MODELS:
        diag = json.loads(
            (ROOT / DIAGNOSTICS.format(family=family, size=size)).read_text(
                encoding="utf-8"
            )
        )
        rag = strata._merge_logs([ROOT / p for p in diag["rag_paths"]])
        out[folder] = {
            str(t): tuple(
                rca._ckey(c)
                for c in ((rec.get("inputs") or {}).get("retrieved_chunks") or [])
            )
            for t, rec in rag.records.items()
        }
    return out


def scopes(model: str) -> List[str]:
    return [SCOPE_ALL, SCOPE_GE1] if model == LLAMA_3B else [SCOPE_ALL]


def in_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df[df["B"] >= 1] if scope == SCOPE_GE1 else df


def model_label(model: str, scope: str = SCOPE_ALL) -> str:
    if model == LLAMA_3B:
        return f"{DISPLAY[model]} ({'all' if scope == SCOPE_ALL else 'baseline ≥ 1'})"
    return DISPLAY.get(model, model)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def r4(x):
    return rca.r4(x)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(r4)
    out.to_csv(path, index=False)
    return path


def _is_p_col(name: str) -> bool:
    return (
        name.startswith("p ")
        or " p " in f" {name} "
        or name.endswith(" p")
        or "p (" in name
    )


def md_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]) and _is_p_col(c):
            out[c] = out[c].map(
                lambda v: "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")
            )
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(
                lambda v: (
                    ""
                    if pd.isna(v)
                    else f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else v
                )
            )
    return em._md_table(out)


def save_fig(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def multiplicities(
    cluster: np.ndarray, strat: Optional[np.ndarray], n_boot: int, rng
) -> np.ndarray:
    """(n_boot, n_rows) resampling multiplicities: clusters are drawn with
    replacement (within each stratum if ``strat`` is given); every row inherits
    the count of its cluster."""
    codes, uniq = pd.factorize(pd.Series(cluster))
    n_cl = len(uniq)
    cl_stratum = np.zeros(n_cl, dtype=object)
    if strat is None:
        cl_stratum[:] = 0
    else:
        cl_stratum[codes] = np.asarray(strat, dtype=object)
    counts = np.zeros((n_boot, n_cl))
    rows = np.arange(n_boot)[:, None]
    for s in pd.unique(cl_stratum):
        members = np.flatnonzero(cl_stratum == s)
        draw = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        np.add.at(counts, (np.broadcast_to(rows, draw.shape), draw), 1)
    return counts[:, codes]


def boot_ratio(num, den, w, m) -> np.ndarray:
    """Bootstrap samples of sum(m w num) / sum(m w den)."""
    num, den, w = (np.asarray(a, dtype=float) for a in (num, den, w))
    return (m * (w * num)).sum(1) / (m * (w * den)).sum(1)


def ci(samples: np.ndarray):
    return em._percentile_ci(np.asarray(samples, dtype=float))


# ---------------------------------------------------------------------------
# judge: 1. human agreement
# ---------------------------------------------------------------------------


def _archived_csv() -> Path:
    """The disagreement CSV as of ARCHIVED_CSV_REV (written to a temp file)."""
    import subprocess
    import tempfile

    rel = DISAGREEMENT_CSV.relative_to(ROOT).as_posix()
    res = subprocess.run(
        ["git", "show", f"{ARCHIVED_CSV_REV}:{rel}"], cwd=ROOT, capture_output=True
    )
    if res.returncode != 0:
        raise SystemExit(
            f"Cannot read {rel} at {ARCHIVED_CSV_REV} from git: {res.stderr.decode()}"
        )
    tmp = Path(tempfile.mkdtemp()) / "archived_disagreements.csv"
    tmp.write_bytes(res.stdout)
    return tmp


def _rating_table(revised: bool, csv_path: Optional[Path] = None):
    files = sorted(HUMAN_DIR.glob(cha.DEFAULT_PATTERN))
    if revised and csv_path is None:
        csv_path = _archived_csv()
    table, _ = cha.build_table(
        files, list(cha.DIMENSIONS), "shared", csv_path if revised else None
    )
    return table


def post_fix_note(ch: pd.DataFrame) -> dict:
    """Effect of the later CSV fix: alphas and consensus labels with the current CSV,
    and the final labels (source human_consensus) that carry an outdated value."""
    table = _rating_table(True, DISAGREEMENT_CSV)
    alphas = {
        d: cha.krippendorff_alpha(
            table.matrices[d],
            "ordinal" if cha.DIMENSIONS[d]["kind"] == "ordinal" else "nominal",
            cha.DIMENSIONS[d]["levels"],
        )
        for d in DIMS
    }
    cons = {}
    for line in CONSENSUS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            for c in rec["chunk_annotations"]:
                cons[(str(rec["text_index"]), int(c["chunk_index"]))] = c
    changed = []
    for i, key in enumerate(table.keys):
        k = (str(key[0]), int(key[1]))
        for d in DIMS:
            new = cha.consensus_label(table.matrices[d][i], cha.DIMENSIONS[d]["kind"])
            if new != cons[k][d]:
                hit = ch[
                    (ch["text_index"] == k[0])
                    & (ch["chunk_index"] == k[1])
                    & (ch[f"source_{d}"] == "human_consensus")
                ]
                changed.append(
                    {
                        "text": k[0],
                        "chunk": k[1],
                        "dim": d,
                        "old": cons[k][d],
                        "new": new,
                        "final_models": [DISPLAY[m] for m in hit["model"]],
                    }
                )
    return {"alphas": alphas, "changed": changed}


def _agreement_stats(
    mat: np.ndarray, dim: str, annotators: List[str]
) -> Dict[str, float]:
    cfg = cha.DIMENSIONS[dim]
    level = "ordinal" if cfg["kind"] == "ordinal" else "nominal"
    out = {"alpha": cha.krippendorff_alpha(mat, level, cfg["levels"])}
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            out[f"{annotators[i]}|{annotators[j]}"] = cha.safe_kappa(
                mat[:, i].astype(int),
                mat[:, j].astype(int),
                labels=cfg["levels"],
                weights="quadratic" if cfg["kind"] == "ordinal" else None,
            )
    return out


def human_agreement(n_boot: int, seed: int) -> List[dict]:
    rows = []
    for revised, report_path, block in (
        (False, HUMAN_REPORT, "Humans, first round (independent)"),
        (True, HUMAN_REPORT_REVISED, "Humans, after discussion and revision"),
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        table = _rating_table(revised)
        texts = np.array([k[0] for k in table.keys])
        n_texts = len(set(texts))
        rng = np.random.default_rng(seed)
        m = multiplicities(texts, None, n_boot, rng)
        for dim in DIMS:
            mr = report["per_dimension"][dim]["multirater"]
            mat = table.matrices[dim]
            point = _agreement_stats(mat, dim, table.annotators)
            # Text-clustered bootstrap: repeat each chunk row by its text's multiplicity.
            boots = defaultdict(list)
            for b in range(n_boot):
                idx = np.repeat(np.arange(len(mat)), m[b].astype(int))
                for k, v in _agreement_stats(mat[idx], dim, table.annotators).items():
                    boots[k].append(np.nan if v is None else v)
            if abs(point["alpha"] - mr["alpha"]) > 1e-9:
                raise SystemExit(
                    f"Recomputed alpha for {dim} ({point['alpha']}) differs from {report_path.name}"
                )
            lo, hi = ci(boots["alpha"])
            rows.append(
                {
                    "Block": block,
                    "Dimension": dim,
                    "Comparison": "3 annotators",
                    "Statistic": f"Krippendorff's alpha ({mr['alpha_level']})",
                    "Value": mr["alpha"],
                    "CI low": lo,
                    "CI high": hi,
                    "N chunks": mr["n_items"],
                    "N texts": n_texts,
                    "Source": report_path.name,
                }
            )
            kappa_name = (
                "QWK" if cha.DIMENSIONS[dim]["kind"] == "ordinal" else "Cohen's kappa"
            )
            for pair, pv in mr["pairwise"].items():
                lo, hi = ci(boots[pair])
                rows.append(
                    {
                        "Block": block,
                        "Dimension": dim,
                        "Comparison": pair.replace("|", " vs "),
                        "Statistic": kappa_name,
                        "Value": pv["kappa"],
                        "CI low": lo,
                        "CI high": hi,
                        "N chunks": pv["n"],
                        "N texts": n_texts,
                        "Source": report_path.name,
                    }
                )
                rows.append(
                    {
                        "Block": block,
                        "Dimension": dim,
                        "Comparison": pair.replace("|", " vs "),
                        "Statistic": "exact agreement",
                        "Value": pv["exact_agreement"],
                        "CI low": np.nan,
                        "CI high": np.nan,
                        "N chunks": pv["n"],
                        "N texts": n_texts,
                        "Source": report_path.name,
                    }
                )
            rows.append(
                {
                    "Block": block,
                    "Dimension": dim,
                    "Comparison": "mean of the 3 pairs",
                    "Statistic": kappa_name,
                    "Value": mr["mean_pairwise_kappa"],
                    "CI low": np.nan,
                    "CI high": np.nan,
                    "N chunks": mr["n_items"],
                    "N texts": n_texts,
                    "Source": report_path.name,
                }
            )
    return rows


def consensus_info() -> dict:
    recs = [
        json.loads(l)
        for l in CONSENSUS_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    chunks = [c for r in recs for c in r["chunk_annotations"]]
    # Re-derive every consensus label from the stored revised ratings with the documented rule.
    mismatches = 0
    for c in chunks:
        for dim in DIMS:
            vals = np.array(list(c["human_ratings"][dim].values()), dtype=float)
            if cha.consensus_label(vals, cha.DIMENSIONS[dim]["kind"]) != c[dim]:
                mismatches += 1
    with open(DISAGREEMENT_CSV, encoding="utf-8") as fh:
        disc = list(csv.DictReader(fh))
    revised_report = json.loads(HUMAN_REPORT_REVISED.read_text(encoding="utf-8"))
    return {
        "texts": len(recs),
        "chunks": len(chunks),
        "source_model": sorted(
            {r["source_context"]["baseline"]["parameters"]["llm"] for r in recs}
        ),
        "quadrants": Counter(r["source_context"]["quadrant"][:2] for r in recs),
        "rule_mismatches": mismatches,
        "unanimous_all_dims": sum(bool(c.get("unanimous")) for c in chunks),
        "unanimous_per_dim": {
            d: sum(len(set(c["human_ratings"][d].values())) == 1 for c in chunks)
            for d in DIMS
        },
        "discussed_chunks": len(disc),
        "values_changed": revised_report["diagnostics"]["n_values_changed_by_revision"],
        "counts": {d: Counter(c[d] for c in chunks) for d in DIMS},
    }


# ---------------------------------------------------------------------------
# judge: 2. judge vs humans
# ---------------------------------------------------------------------------


def judge_vs_consensus(n_boot: int, seed: int) -> List[dict]:
    """Recompute judge-vs-consensus agreement (with text-clustered CIs) from the
    per-chunk labels of the recommended run; check against the range report."""
    judge = {}
    for line in (
        (JUDGE_RUN_DIR / "llm_judge_qualitative_annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip():
            rec = json.loads(line)
            for c in rec["chunk_annotations"]:
                judge[(str(rec["text_index"]), int(c["chunk_index"]))] = c
    cons = {}
    for line in CONSENSUS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            for c in rec["chunk_annotations"]:
                cons[(str(rec["text_index"]), int(c["chunk_index"]))] = c
    if set(judge) != set(cons):
        raise SystemExit("Judge run and consensus file cover different chunks")
    keys = sorted(cons)
    texts = np.array([k[0] for k in keys])
    rng = np.random.default_rng(seed)
    m = multiplicities(texts, None, n_boot, rng)
    rr = json.loads(RANGE_REPORT.read_text(encoding="utf-8"))
    rows = []
    for dim in DIMS:
        cfg = cha.DIMENSIONS[dim]
        weights = "quadratic" if cfg["kind"] == "ordinal" else None
        j = np.array([judge[k][dim] for k in keys], dtype=int)
        h = np.array([cons[k][dim] for k in keys], dtype=int)
        point = cha.safe_kappa(j, h, labels=cfg["levels"], weights=weights)
        if abs(point - rr[dim]["judge_vs_consensus_qwk"]) > 1e-6:
            raise SystemExit(
                f"Judge-vs-consensus {dim}: recomputed {point} != range report "
                f"{rr[dim]['judge_vs_consensus_qwk']}"
            )
        boots = []
        for b in range(n_boot):
            idx = np.repeat(np.arange(len(keys)), m[b].astype(int))
            v = cha.safe_kappa(j[idx], h[idx], labels=cfg["levels"], weights=weights)
            boots.append(np.nan if v is None else v)
        lo, hi = ci(boots)
        kname = "QWK" if weights else "Cohen's kappa"
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "judge vs consensus",
                "Statistic": kname,
                "Value": point,
                "CI low": lo,
                "CI high": hi,
                "Source": f"recomputed from {JUDGE_RUN_DIR.name}/llm_judge_qualitative_annotations.jsonl "
                f"(matches {RANGE_REPORT.name})",
            }
        )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "judge vs consensus",
                "Statistic": "exact agreement",
                "Value": float(np.mean(j == h)),
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": f"recomputed from {JUDGE_RUN_DIR.name}",
            }
        )
        if cfg["kind"] == "ordinal":
            rows.append(
                {
                    "Dimension": dim,
                    "Comparison": "judge vs consensus",
                    "Statistic": "within ±1",
                    "Value": float(np.mean(np.abs(j - h) <= 1)),
                    "CI low": np.nan,
                    "CI high": np.nan,
                    "Source": f"recomputed from {JUDGE_RUN_DIR.name}",
                }
            )
    return rows


def judge_vs_humans_from_report() -> List[dict]:
    rr = json.loads(RANGE_REPORT.read_text(encoding="utf-8"))
    src = f"LLM calib/range_restriction_runs/archive/{RANGE_REPORT.name}"
    rows = []
    for dim in DIMS:
        d = rr[dim]
        kname = "QWK" if dim != "A_caus" else "QWK (= Cohen's kappa for 0/1)"
        for name, val in d["judge_vs_each_annotator_qwk"].items():
            rows.append(
                {
                    "Dimension": dim,
                    "Comparison": f"judge vs {name}",
                    "Statistic": kname,
                    "Value": val,
                    "CI low": np.nan,
                    "CI high": np.nan,
                    "Source": src,
                }
            )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "judge vs humans (mean of 3)",
                "Statistic": kname,
                "Value": d["judge_vs_annotators_mean"]["qwk"],
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": src,
            }
        )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "human vs human (mean of 3 pairs)",
                "Statistic": kname,
                "Value": d["human_vs_human_mean"]["qwk"],
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": src,
            }
        )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "ratio judge-human / human-human",
                "Statistic": "QWK ratio",
                "Value": d["ratio_judge_to_human"],
                "CI low": d["ratio_ci95"][0],
                "CI high": d["ratio_ci95"][1],
                "Source": src,
            }
        )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "judge vs humans (mean of 3)",
                "Statistic": "exact agreement",
                "Value": d["judge_vs_annotators_mean"]["exact"],
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": src,
            }
        )
        rows.append(
            {
                "Dimension": dim,
                "Comparison": "human vs human (mean of 3 pairs)",
                "Statistic": "exact agreement",
                "Value": d["human_vs_human_mean"]["exact"],
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": src,
            }
        )
    return rows


def cutpoint_loto() -> List[dict]:
    cp = json.loads(CUTPOINT_FILE.read_text(encoding="utf-8"))
    src = f"LLM calib/judge_calibration/{CUTPOINT_FILE.name}"
    rows = []
    for dim in ("R_top", "R_ideo"):
        cv = cp[dim]["leave_one_text_out_cv"]
        for variant, label in (
            ("uncalibrated", "gate ladder (0.5 thresholds)"),
            ("quantile_cutpoints", "distribution-matched cut points"),
        ):
            for stat in ("alpha", "qwk", "exact", "bias"):
                rows.append(
                    {
                        "Dimension": dim,
                        "Comparison": f"leave-one-text-out, {label}",
                        "Statistic": {
                            "alpha": "Krippendorff's alpha (ordinal) vs consensus",
                            "qwk": "QWK vs consensus",
                            "exact": "exact agreement",
                            "bias": "mean bias (judge - consensus)",
                        }[stat],
                        "Value": cv[variant][stat],
                        "CI low": np.nan,
                        "CI high": np.nan,
                        "Source": src,
                    }
                )
    # The 8-text holdout (fit on the 17 development texts) is only recorded in the calibration log.
    for variant, val in (
        ("gate ladder (0.5 thresholds)", 0.608),
        ("distribution-matched cut points", 0.665),
    ):
        rows.append(
            {
                "Dimension": "R_ideo",
                "Comparison": f"8-text holdout, {variant}",
                "Statistic": "Krippendorff's alpha (ordinal) vs consensus",
                "Value": val,
                "CI low": np.nan,
                "CI high": np.nan,
                "Source": "LLM calib/judge_calibration/CALIBRATION_LOG.md (round 3b)",
            }
        )
    return rows


def calibration_labels(cinfo: dict) -> pd.DataFrame:
    rr = json.loads(RANGE_REPORT.read_text(encoding="utf-8"))
    rows = []
    for dim in DIMS:
        levels = rr[dim]["levels"]
        for rater in ("Awsam", "Jannes", "Neda", "consensus", "judge"):
            shares = rr[dim]["distribution"][rater]
            row = {
                "Dimension": dim,
                "Rater": (
                    "Consensus"
                    if rater == "consensus"
                    else (
                        "Judge (run 2026-10-23_142551, R_ideo cut points)"
                        if rater == "judge"
                        else rater
                    )
                ),
                "N chunks": 125,
            }
            for lv, s in zip(levels, shares):
                row[f"Share level {lv}"] = s
            row["SD of labels"] = rr[dim]["levels_used_sd"][rater]
            row["Source"] = RANGE_REPORT.name
            rows.append(row)
        # Consistency check: consensus shares recomputed from the consensus file.
        counts = cinfo["counts"][dim]
        recomputed = [counts.get(lv, 0) / cinfo["chunks"] for lv in levels]
        if not np.allclose(recomputed, rr[dim]["distribution"]["consensus"]):
            raise SystemExit(
                f"Consensus distribution of {dim} differs between consensus file and range report"
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# judge: 3. review procedure
# ---------------------------------------------------------------------------


def load_final_annotations() -> pd.DataFrame:
    """One row per (model, text, chunk) with final labels, label sources and judge labels."""
    rows = []
    for folder, *_ in MODELS:
        path = FAMILIES_DIR / folder / "final_annotations.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc = rec.get("source_context") or {}
            for c in rec["chunk_annotations"]:
                row = {
                    "model": folder,
                    "text_index": str(rec["text_index"]),
                    "quadrant": (sc.get("quadrant") or "")[:2],
                    "chunk_index": int(c["chunk_index"]),
                    "chunk_text": c.get("chunk_text") or "",
                    "chunk_party_raw": (c.get("chunk_metadata") or {}).get("party"),
                    "input_text": rec.get("input_text") or "",
                }
                for d in DIMS:
                    row[d] = c.get(d)
                    row[f"source_{d}"] = (c.get("label_source") or {}).get(d)
                    row[f"flag_{d}"] = bool((c.get("needs_review") or {}).get(d))
                    row[f"judge_{d}"] = (c.get("judge_labels") or {}).get(d)
                rows.append(row)
    ch = pd.DataFrame(rows)
    bad = ch[~ch["quadrant"].isin(QUADRANTS)]
    if len(bad):
        raise SystemExit(f"{len(bad)} annotated chunks without a sampling quadrant")
    return ch


def review_summary(ch: pd.DataFrame) -> pd.DataFrame:
    summ = json.loads(REVIEW_SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for folder, *_ in MODELS:
        f = summ["folders"][folder]
        sub = ch[ch["model"] == folder]
        # Check: label-source counts in the final file = review_summary.json (summed over dimensions).
        total = Counter()
        for d in DIMS:
            total.update(sub[f"source_{d}"].value_counts().to_dict())
        if dict(total) != f["sources"]:
            raise SystemExit(
                f"{folder}: label sources in final_annotations.jsonl {dict(total)} "
                f"!= review_summary.json {f['sources']}"
            )
        for d in DIMS:
            src_counts = sub[f"source_{d}"].value_counts()
            spot = f["spot"][d]
            row = {
                "Model": DISPLAY[folder],
                "Dimension": d,
                "Texts": f["texts"],
                "Texts selected for review": f["texts_selected"],
                "Texts reviewed": f["texts_reviewed"],
                "Chunks": f["chunks"],
            }
            for s in LABEL_SOURCES:
                row[f"Chunks from {s}"] = int(src_counts.get(s, 0))
            row["Flagged chunks reviewed"] = f["flagged"][d]["n"]
            row["Flagged chunks changed"] = f["flagged"][d]["changed"]
            row["Share of flagged changed"] = (
                f["flagged"][d]["changed"] / f["flagged"][d]["n"]
                if f["flagged"][d]["n"]
                else np.nan
            )
            for kind in ("unflagged", "flagged"):
                sp = spot[kind] or {}
                row[f"Spot check {kind}: n"] = sp.get("n", 0)
                row[f"Spot check {kind}: exact"] = sp.get("exact", np.nan)
                row[f"Spot check {kind}: within ±1"] = (
                    sp.get("within1", np.nan) if d != "A_caus" else np.nan
                )
            rows.append(row)
    df = pd.DataFrame(rows)
    # Pooled over the reviewed models (counts summed, spot-check shares weighted by n).
    pooled = []
    for d in DIMS:
        sub = df[
            (df["Dimension"] == d)
            & df["Model"].isin([DISPLAY[m] for m in REVIEWED_MODELS])
        ]
        row = {"Model": "Reviewed models pooled", "Dimension": d}
        for c in df.columns:
            if (
                c in ("Model", "Dimension")
                or c.startswith("Share")
                or ": exact" in c
                or "within" in c
            ):
                continue
            row[c] = sub[c].sum()
        row["Share of flagged changed"] = (
            row["Flagged chunks changed"] / row["Flagged chunks reviewed"]
        )
        for kind in ("unflagged", "flagged"):
            n = sub[f"Spot check {kind}: n"]
            for stat in ("exact", "within ±1"):
                v = sub[f"Spot check {kind}: {stat}"]
                ok = n > 0
                row[f"Spot check {kind}: {stat}"] = (
                    float((v[ok] * n[ok]).sum() / n[ok].sum())
                    if ok.any() and v[ok].notna().all()
                    else np.nan
                )
        p = summ["pooled"][d]
        if row["Flagged chunks reviewed"] != p["flagged_reviewed"] or (
            row["Spot check unflagged: n"]
            and abs(row["Spot check unflagged: exact"] - p["spot_unflagged"]["exact"])
            > 1e-9
        ):
            raise SystemExit(
                f"Pooled review numbers for {d} do not match review_summary.json"
            )
        pooled.append(row)
    df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
    int_cols = [
        c
        for c in df.columns
        if c.startswith(("Texts", "Chunks", "Flagged chunks")) or c.endswith(": n")
    ]
    df[int_cols] = df[int_cols].astype(int)
    return df


# ---------------------------------------------------------------------------
# judge: 4. transfer check
# ---------------------------------------------------------------------------


def transfer_check(ch: pd.DataFrame) -> pd.DataFrame:
    lists = rag_chunk_lists()
    rows = []
    common = set.intersection(*(set(v) for v in lists.values()))
    identical = sum(len({lists[m][t] for m in lists}) == 1 for t in common)
    rows.append(
        {
            "Check": "Texts in the e5/simple RAG log of all 10 models",
            "N": len(common),
            "Consistent": identical,
            "Share": identical / len(common),
            "Definition": "identical ordered list of 5 chunks (speech id + chunk text) in all 10 models",
            "Source": "RAG logs listed in qualitative_e5_simple_diagnostics.json",
        }
    )
    any_texts = set.union(*(set(v) for v in lists.values()))
    rows.append(
        {
            "Check": "Texts in the RAG log of at least one model",
            "N": len(any_texts),
            "Consistent": len(common),
            "Share": len(common) / len(any_texts),
            "Definition": "text present in all 10 RAG logs",
            "Source": "RAG logs",
        }
    )
    # Annotated chunks = the chunks of that model's RAG log (same order).
    match = n = 0
    for (model, text), g in ch.groupby(["model", "text_index"]):
        n += 1
        log_keys = lists[model].get(text, ())
        ann = tuple(g.sort_values("chunk_index")["chunk_text"])
        log_texts = tuple(k.split("#", 1)[1] for k in log_keys)
        ann_hash = tuple(rca._ckey({"text": t}).split("#", 1)[1] for t in ann)
        match += ann_hash == log_texts
    rows.append(
        {
            "Check": "Annotated (model, text) pairs whose chunks equal the model's RAG log",
            "N": n,
            "Consistent": match,
            "Share": match / n,
            "Definition": "same 5 chunk texts in the same order",
            "Source": "final_annotations.jsonl vs RAG logs",
        }
    )
    # (input, chunk) pairs that occur in the annotation sample of several models.
    pairs = ch.groupby(["text_index", "chunk_index"])["model"].nunique()
    rows.append(
        {
            "Check": "(text, chunk) pairs annotated for more than one model",
            "N": int(pairs.size),
            "Consistent": int((pairs > 1).sum()),
            "Share": float((pairs > 1).mean()),
            "Definition": "'Consistent' = pairs that occur in >= 2 models' samples",
            "Source": "final_annotations.jsonl",
        }
    )
    summ = json.loads(REVIEW_SUMMARY.read_text(encoding="utf-8"))["transfer"]
    for d in ("R_top", "R_ideo"):
        t = summ["repeated_pairs_consistent"][d]
        rows.append(
            {
                "Check": f"Human-reviewed pairs repeated across models, {d} identical",
                "N": t["n"],
                "Consistent": t["consistent"],
                "Share": t["consistent"] / t["n"],
                "Definition": "the reviewer gave the same label to the same (text, chunk) pair in different models",
                "Source": "review_summary.json (transfer)",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# judge: 5. final label distributions
# ---------------------------------------------------------------------------


def sampling_weights(
    pairs: pd.DataFrame, ch_texts: pd.DataFrame, scope: str, model: str
) -> Dict[str, float]:
    """Population texts / sampled texts per sampling quadrant (one-decimal rule)."""
    pop = in_scope(pairs[pairs["model"] == model], scope)
    pop_counts = pop["quadrant_sampling"].value_counts()
    samp_counts = ch_texts["quadrant"].value_counts()
    return {q: pop_counts.get(q, 0) / samp_counts[q] for q in samp_counts.index}


def text_level(ch: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "chunks": ("chunk_index", "size"),
        "sum_R_top": ("R_top", "sum"),
        "sum_R_ideo": ("R_ideo", "sum"),
        "cited": ("A_caus", "sum"),
        "rtop_ge4": ("R_top", lambda s: (s >= 4).sum()),
        "rideo_ge4": ("R_ideo", lambda s: (s >= 4).sum()),
    }
    for lv in LEVELS:
        agg[f"rtop_{lv}"] = ("R_top", lambda s, lv=lv: (s == lv).sum())
        agg[f"rideo_{lv}"] = ("R_ideo", lambda s, lv=lv: (s == lv).sum())
    return ch.groupby(["model", "text_index", "quadrant"], as_index=False).agg(**agg)


def distribution_row(
    t: pd.DataFrame,
    w: np.ndarray,
    m: np.ndarray,
    label: str,
    scope: str,
    weighting: str,
) -> dict:
    row = {
        "Model": label,
        "Posts": scope,
        "Weighting": weighting,
        "Texts": len(t),
        "Chunks": int(t["chunks"].sum()),
    }
    den = (w * t["chunks"]).sum()
    for name, prefix in (("R_top", "rtop"), ("R_ideo", "rideo")):
        for lv in LEVELS:
            row[f"{name} share {lv}"] = (w * t[f"{prefix}_{lv}"]).sum() / den
    for col, name, per_text in (
        ("sum_R_top", "Mean R_top", False),
        ("sum_R_ideo", "Mean R_ideo", False),
        ("rtop_ge4", "Share R_top ≥ 4", False),
        ("rideo_ge4", "Share R_ideo ≥ 4", False),
        ("cited", "Share cited (A_caus = 1)", False),
        ("cited", "Cited chunks per text", True),
    ):
        d = np.ones(len(t)) if per_text else t["chunks"].to_numpy()
        row[name] = (w * t[col]).sum() / (w * d).sum()
        lo, hi = ci(boot_ratio(t[col], d, w, m))
        row[f"{name} CI low"], row[f"{name} CI high"] = lo, hi
    return row


def label_distribution(ch: pd.DataFrame, pairs: pd.DataFrame, n_boot: int, seed: int):
    texts = text_level(ch)
    base = pairs.set_index(["model", "text_index"])["B"]
    texts["B"] = [
        base.get((m, t), np.nan) for m, t in zip(texts["model"], texts["text_index"])
    ]
    if texts["B"].isna().any():
        missing = texts[texts["B"].isna()][["model", "text_index"]].values.tolist()
        raise SystemExit(f"Annotated texts not among the paired posts: {missing[:10]}")
    rows, weights_rows = [], []
    pooled_parts = []
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            t = in_scope(texts[texts["model"] == folder], scope).reset_index(drop=True)
            wq = sampling_weights(pairs, t, scope, folder)
            w_true = t["quadrant"].map(wq).to_numpy()
            for q in QUADRANTS:
                pop_n = (
                    in_scope(pairs[pairs["model"] == folder], scope)[
                        "quadrant_sampling"
                    ]
                    == q
                ).sum()
                weights_rows.append(
                    {
                        "Model": model_label(folder, scope),
                        "Quadrant": q,
                        "Population texts": int(pop_n),
                        "Sampled texts": int((t["quadrant"] == q).sum()),
                        "Weight": wq.get(q, np.nan),
                    }
                )
            for weighting, w in (
                ("unweighted", np.ones(len(t))),
                ("weighted by quadrant base rates", w_true),
            ):
                rng = np.random.default_rng(seed)
                m = multiplicities(
                    t["text_index"].to_numpy(), t["quadrant"].to_numpy(), n_boot, rng
                )
                rows.append(
                    distribution_row(
                        t, w, m, model_label(folder, scope), scope, weighting
                    )
                )
            if scope == SCOPE_ALL:
                pooled_parts.append(t.assign(w=w_true))
    pooled = pd.concat(pooled_parts, ignore_index=True)
    for weighting, w in (
        ("unweighted", np.ones(len(pooled))),
        ("weighted by quadrant base rates", pooled["w"].to_numpy()),
    ):
        rng = np.random.default_rng(seed)
        # All models judge the same texts: resample text_index clusters.
        m = multiplicities(pooled["text_index"].to_numpy(), None, n_boot, rng)
        rows.append(
            distribution_row(
                pooled, w, m, "All models (Llama 3B: all)", SCOPE_ALL, weighting
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(weights_rows)


def check_population_counts(pairs: pd.DataFrame):
    """Population quadrant counts must reproduce the diagnostics files (sampling frame)."""
    for folder, family, size, _ in MODELS:
        diag = json.loads(
            (ROOT / DIAGNOSTICS.format(family=family, size=size)).read_text(
                encoding="utf-8"
            )
        )
        mine = pairs[pairs["model"] == folder]["quadrant_sampling"].value_counts()
        for q_long, n in diag["population_counts"].items():
            if mine.get(q_long[:2], 0) != n:
                raise SystemExit(
                    f"{folder} {q_long}: {mine.get(q_long[:2], 0)} paired texts, diagnostics say {n}"
                )
        if len(pairs[pairs["model"] == folder]) != diag["valid_paired_records"]:
            raise SystemExit(f"{folder}: paired N differs from diagnostics")


def plot_labels(dist: pd.DataFrame):
    d = dist[(dist["Weighting"] == "weighted by quadrant base rates")].reset_index(
        drop=True
    )
    labels = d["Model"].tolist()
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(labels) + 1.4), sharey=True)
    for ax, dim in zip(axes, ("R_top", "R_ideo")):
        left = np.zeros(len(d))
        for lv, colour in zip(LEVELS, LEVEL_COLOURS):
            share = d[f"{dim} share {lv}"].to_numpy()
            ax.barh(
                y,
                share,
                left=left,
                height=0.7,
                color=colour,
                edgecolor="white",
                linewidth=1.2,
                label=str(lv),
            )
            for yi, l0, s in zip(y, left, share):
                if s >= 0.09:
                    ax.text(
                        l0 + s / 2,
                        yi,
                        f"{s * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=INK if lv <= 2 else "white",
                    )
            left += share
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(
            ["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8, color=INK_SECONDARY
        )
        ax.set_title(f"{dim} (share of chunks)", fontsize=10, color=INK, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK_SECONDARY)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5, color=INK)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"level {lv}" for lv in leg_labels],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.55, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, "fig_judge_labels")


# ---------------------------------------------------------------------------
# judge: report
# ---------------------------------------------------------------------------


def _get(df, **kw):
    sub = df
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        raise KeyError(kw)
    return sub.iloc[0]


def write_judge_report(
    agree: pd.DataFrame,
    cal: pd.DataFrame,
    review: pd.DataFrame,
    transfer: pd.DataFrame,
    dist: pd.DataFrame,
    weights: pd.DataFrame,
    cinfo: dict,
    ch: pd.DataFrame,
    note: dict,
    n_boot: int,
    seed: int,
):
    L = []
    a = lambda **kw: _get(agree, **kw)  # noqa: E731
    first, rev = (
        "Humans, first round (independent)",
        "Humans, after discussion and revision",
    )
    jblock = "Judge vs humans (run 2026-10-23_142551, R_top ladder, R_ideo cut points)"
    fmt = lambda v: f"{v:.2f}"  # noqa: E731
    fci = (
        lambda r: f"{r['Value']:.2f} ({r['CI low']:.2f} to {r['CI high']:.2f})"
    )  # noqa: E731

    L.append("# Why retrieval helps or hurts: the LLM judge\n")
    L.append("## Data\n")
    n_texts = ch.groupby("model")["text_index"].nunique()
    L.append(
        f"Human annotation: {cinfo['texts']} texts × 5 chunks = {cinfo['chunks']} chunks, rated by 3 annotators "
        f"(Awsam, Jannes, Neda) on R_top (1-5), R_ideo (1-5) and A_caus (0/1). The texts come from the quadrant "
        f"sample of {', '.join(cinfo['source_model'])} with e5/simple retrieval "
        f"(Q1 {cinfo['quadrants']['Q1']}, Q2 {cinfo['quadrants']['Q2']}, Q3 {cinfo['quadrants']['Q3']}, "
        f"Q4 {cinfo['quadrants']['Q4']} texts). Judge: TypeSafe Jev 1.14, question set v1, run "
        f"{JUDGE_RUN} with the R_ideo cut points. Final labels: `final_annotations.jsonl` of the ten models, "
        f"{len(ch)} chunks of {int(n_texts.sum())} (model, text) pairs "
        f"({', '.join(f'{DISPLAY[m]} {n_texts[m]}' for m, *_ in MODELS)} texts). The quadrants are the ones the "
        "texts were sampled in (one-decimal target of the logs). The weights for the population use the paired "
        "posts of the e5/simple vs. no-RAG logs; they cover Q1-Q4 only, because the buffer texts (error between "
        "0.5 and 1.0) were not sampled. Llama 3B is reported on all sampled texts and on the texts with a "
        f"baseline score ≥ 1. Bootstrap: {n_boot} resamples of texts, seed {seed}. No exclusions apart from these.\n"
    )

    # 1. Human agreement
    L.append("## 1. Human agreement\n")
    tab = []
    for dim in DIMS:
        r1 = a(Block=first, Dimension=dim, Comparison="3 annotators")
        r2 = a(Block=rev, Dimension=dim, Comparison="3 annotators")
        pk = agree[
            (agree["Block"] == rev)
            & (agree["Dimension"] == dim)
            & agree["Comparison"].str.contains(" vs ")
            & agree["Statistic"].isin(["QWK", "Cohen's kappa"])
        ]
        tab.append(
            {
                "Dimension": dim,
                "alpha first round": fci(r1),
                "alpha after revision": fci(r2),
                "pairwise kappa after revision (min-max)": f"{pk['Value'].min():.2f}-{pk['Value'].max():.2f}",
                "kappa type": pk["Statistic"].iloc[0],
            }
        )
    L.append(em._md_table(pd.DataFrame(tab)) + "\n")
    rt1, rt2 = a(Block=first, Dimension="R_top", Comparison="3 annotators"), a(
        Block=rev, Dimension="R_top", Comparison="3 annotators"
    )
    ri2 = a(Block=rev, Dimension="R_ideo", Comparison="3 annotators")
    L.append(
        f"Reading: rated independently with only the help page, the annotators agreed poorly on the two ordinal "
        f"scales (R_top α {fmt(rt1['Value'])}). After they discussed {cinfo['discussed_chunks']} of the "
        f"{cinfo['chunks']} chunks and revised their own ratings "
        f"({', '.join(f'{k} {v}' for k, v in cinfo['values_changed'].items())} values changed), agreement reached "
        f"α {fmt(rt2['Value'])} for R_top and {fmt(ri2['Value'])} for R_ideo. R_ideo stays the weakest "
        f"dimension, and its CI is wide because almost all chunks are rated 3 or 4. The CIs resample the "
        f"{cinfo['texts']} texts.\n"
    )
    L.append(
        f"Consensus (`revised/consensus_reference_annotations.jsonl`): per chunk, the median of the three revised "
        f"ratings for R_top and R_ideo and the majority for A_caus (`consensus_label` in "
        f"`compare_human_annotations.py`; ties rounded up, which cannot occur with three raters). Re-deriving all "
        f"{cinfo['chunks'] * 3} consensus labels with this rule gives {cinfo['rule_mismatches']} mismatches. After "
        f"revision the three annotators agree completely on {cinfo['unanimous_per_dim']['R_top']} (R_top), "
        f"{cinfo['unanimous_per_dim']['R_ideo']} (R_ideo) and {cinfo['unanimous_per_dim']['A_caus']} (A_caus) "
        f"chunks, and on all three dimensions for {cinfo['unanimous_all_dims']}. The chunks discussed are listed in "
        f"`revised/Annotations - annotation_disagreements.csv`.\n"
    )
    changes = "; ".join(
        f"text {c['text']}, chunk {c['chunk']}, {c['dim']} {c['old']} → {c['new']}"
        + (
            f" (final label from the consensus in {', '.join(c['final_models'])})"
            if c["final_models"]
            else " (no final label uses it)"
        )
        for c in note["changed"]
    )
    L.append(
        f"Note on a later correction: the numbers above are the archived state on which the judge was calibrated "
        f"and the final labels were built (revised report of 22 Oct; CIs computed on the disagreement CSV as of "
        f"`{ARCHIVED_CSV_REV}`, which reproduces that report exactly). Commit e58fb1c (29 Oct) later corrected three "
        f"chunks in the CSV. With the corrected CSV, α is {note['alphas']['R_top']:.3f} (R_top), "
        f"{note['alphas']['R_ideo']:.3f} (R_ideo) and {note['alphas']['A_caus']:.3f} (A_caus), and "
        f"{len(note['changed'])} consensus labels would change: {changes}. One of the three corrections "
        f"(text 39, chunk 2, A_caus = 1) was already in the consensus file. The judge-vs-consensus figures and the "
        f"final labels use the consensus file as it is.\n"
    )

    # 2. Judge vs humans
    L.append("## 2. Judge vs. humans (125 calibration chunks)\n")
    tab = []
    for dim in DIMS:
        hh = a(
            Block=jblock,
            Dimension=dim,
            Comparison="human vs human (mean of 3 pairs)",
            Statistic=agree[
                (agree["Block"] == jblock)
                & (agree["Dimension"] == dim)
                & (agree["Comparison"] == "human vs human (mean of 3 pairs)")
            ]["Statistic"].iloc[0],
        )
        jh = a(
            Block=jblock,
            Dimension=dim,
            Comparison="judge vs humans (mean of 3)",
            Statistic=hh["Statistic"],
        )
        ra = a(
            Block=jblock, Dimension=dim, Comparison="ratio judge-human / human-human"
        )
        jc = agree[
            (agree["Block"] == jblock)
            & (agree["Dimension"] == dim)
            & (agree["Comparison"] == "judge vs consensus")
            & agree["Statistic"].isin(["QWK", "Cohen's kappa"])
        ].iloc[0]
        ex = a(
            Block=jblock,
            Dimension=dim,
            Comparison="judge vs consensus",
            Statistic="exact agreement",
        )
        tab.append(
            {
                "Dimension": dim,
                "human vs human QWK": fmt(hh["Value"]),
                "judge vs human QWK": fmt(jh["Value"]),
                "ratio (95 % CI)": fci(ra),
                "judge vs consensus QWK / kappa¹ (95 % CI)": fci(jc),
                "exact agreement with consensus": fmt(ex["Value"]),
            }
        )
    L.append(em._md_table(pd.DataFrame(tab)) + "\n")
    L.append(
        "¹ QWK for R_top and R_ideo, Cohen's kappa for A_caus (for 0/1 labels the two coincide).\n"
    )
    rr = {
        d: a(Block=jblock, Dimension=d, Comparison="ratio judge-human / human-human")[
            "Value"
        ]
        for d in DIMS
    }
    L.append(
        f"Reading: the judge reaches {rr['R_top'] * 100:.0f} % (R_top), {rr['R_ideo'] * 100:.0f} % (R_ideo) and "
        f"{rr['A_caus'] * 100:.0f} % (A_caus) of the agreement two humans reach with each other. The ratios and "
        f"the human/judge QWKs come from `{RANGE_REPORT.name}` (in `LLM calib/range_restriction_runs/archive/`). "
        f"The judge-vs-consensus coefficients are recomputed from the run's per-chunk labels, match that file, "
        f"and get text-clustered CIs here. For R_ideo the cut points were fitted on these same 125 chunks, so its "
        f"row is in-sample; the out-of-sample effect is in the next table.\n"
    )
    loto = agree[agree["Block"] == "Cut points, out of sample"]
    tab = []
    for dim in ("R_top", "R_ideo"):
        for variant in (
            "gate ladder (0.5 thresholds)",
            "distribution-matched cut points",
        ):
            g = loto[
                (loto["Dimension"] == dim)
                & (loto["Comparison"] == f"leave-one-text-out, {variant}")
            ].set_index("Statistic")["Value"]
            tab.append(
                {
                    "Dimension": dim,
                    "labels": variant,
                    "LOTO alpha": fmt(g["Krippendorff's alpha (ordinal) vs consensus"]),
                    "LOTO QWK": fmt(g["QWK vs consensus"]),
                    "LOTO exact": fmt(g["exact agreement"]),
                    "LOTO bias": f"{g['mean bias (judge - consensus)']:+.3f}",
                }
            )
    L.append(em._md_table(pd.DataFrame(tab)) + "\n")
    L.append(
        f"Reading: in leave-one-text-out cross-validation (cut points fitted on 24 texts, applied to the 25th; "
        f"`{CUTPOINT_FILE.name}`) the cut points raise R_ideo α from "
        f"{tab[2]['LOTO alpha']} to {tab[3]['LOTO alpha']} and remove its upward bias. On the 8-text holdout "
        f"(fit on the 17 development texts; `CALIBRATION_LOG.md`, round 3b) R_ideo α rises from 0.608 to 0.665. "
        f"For R_top the cut points do not help (α {tab[0]['LOTO alpha']} → {tab[1]['LOTO alpha']}), so R_top "
        f"is read off the gate ladder.\n"
    )
    caltab = cal[cal["Dimension"].isin(["R_top", "R_ideo"])].copy()
    caltab = caltab[["Dimension", "Rater"] + [f"Share level {lv}" for lv in LEVELS]]
    L.append(md_table(caltab) + "\n")
    jr = _get(
        cal, Dimension="R_top", Rater="Judge (run 2026-10-23_142551, R_ideo cut points)"
    )
    cr = _get(cal, Dimension="R_top", Rater="Consensus")
    ca = _get(cal, Dimension="A_caus", Rater="Consensus")
    ja = _get(
        cal,
        Dimension="A_caus",
        Rater="Judge (run 2026-10-23_142551, R_ideo cut points)",
    )
    L.append(
        f"Reading: R_ideo labels of the judge have the consensus shares by construction (the cut points match "
        f"them); level 1 is never used and level 2 only for 2 % of chunks. The R_top ladder puts more chunks on "
        f"level 2 than the consensus ({jr['Share level 2'] * 100:.0f} % vs {cr['Share level 2'] * 100:.0f} %) and "
        f"fewer on level 3 ({jr['Share level 3'] * 100:.0f} % vs {cr['Share level 3'] * 100:.0f} %). A_caus = 1: "
        f"consensus {ca['Share level 1'] * 100:.0f} %, judge {ja['Share level 1'] * 100:.0f} %. "
        f"Full table incl. A_caus and each annotator: `judge_calibration_labels.csv`.\n"
    )

    # 3. Review
    L.append("## 3. Human review of the judge labels\n")
    pooled = review[review["Model"] == "Reviewed models pooled"].set_index("Dimension")
    tab = []
    for d in DIMS:
        p = pooled.loc[d]
        tab.append(
            {
                "Dimension": d,
                "flagged chunks reviewed": int(p["Flagged chunks reviewed"]),
                "changed": int(p["Flagged chunks changed"]),
                "share changed": fmt(p["Share of flagged changed"]),
                "spot check unflagged: exact (±1), n": (
                    f"{p['Spot check unflagged: exact']:.2f}"
                    + (
                        f" ({p['Spot check unflagged: within ±1']:.2f})"
                        if d != "A_caus"
                        else ""
                    )
                    + f", n = {int(p['Spot check unflagged: n'])}"
                ),
                "spot check flagged: exact (±1), n": (
                    f"{p['Spot check flagged: exact']:.2f}"
                    + (
                        f" ({p['Spot check flagged: within ±1']:.2f})"
                        if d != "A_caus"
                        else ""
                    )
                    + f", n = {int(p['Spot check flagged: n'])}"
                ),
            }
        )
    L.append(em._md_table(pd.DataFrame(tab)) + "\n")
    L.append(
        f"Reading: three models were reviewed by hand ({', '.join(DISPLAY[m] for m in REVIEWED_MODELS)}): all "
        f"Q1/Q2 texts and 10 Q3/Q4 texts each. The reviewer changed "
        f"{pooled.loc['R_top', 'Share of flagged changed'] * 100:.0f} % of the flagged R_top labels, "
        f"{pooled.loc['R_ideo', 'Share of flagged changed'] * 100:.0f} % of the flagged R_ideo labels and "
        f"{pooled.loc['A_caus', 'Share of flagged changed'] * 100:.0f} % of the few flagged A_caus labels. In the "
        f"blind spot check, unflagged R_top labels agree exactly only "
        f"{pooled.loc['R_top', 'Spot check unflagged: exact'] * 100:.0f} % of the time "
        f"({pooled.loc['R_top', 'Spot check unflagged: within ±1'] * 100:.0f} % within one level); R_ideo is "
        f"within ±1 in every spot check and A_caus agrees in "
        f"{pooled.loc['A_caus', 'Spot check unflagged: exact'] * 100:.0f} %. The spot-check samples are small "
        f"(n = 18-44 per cell), so treat unreviewed R_top labels as ±1 level. Source: `review_summary.json`.\n"
    )
    src_cols = [f"Chunks from {s}" for s in LABEL_SOURCES]
    st = review[
        (review["Dimension"].isin(["R_top", "A_caus"]))
        & (review["Model"] != "Reviewed models pooled")
    ]
    st = st[["Model", "Dimension", "Texts", "Texts reviewed"] + src_cols].rename(
        columns={c: c.replace("Chunks from ", "") for c in src_cols}
    )
    L.append(
        "Label sources per model (chunks; R_ideo has the same counts as R_top, A_caus is never transferred):\n"
    )
    L.append(em._md_table(st) + "\n")
    rt = st[st["Dimension"] == "R_top"]
    share_human = 1 - (rt["judge"].sum()) / (
        rt[
            [
                "judge",
                "judge_seen",
                "human_review",
                "human_changed",
                "spot_check",
                "human_consensus",
                "transferred",
            ]
        ]
        .sum(axis=1)
        .sum()
    )
    L.append(
        f"Reading: `judge` = never seen by a human; `judge_seen` = unflagged label in a reviewed text, kept; "
        f"`human_review` = flagged label checked by the reviewer; `human_changed` = unflagged label the reviewer "
        f"changed; `spot_check` = rated blind; `human_consensus` = (text, chunk) pair from the 25 consensus texts; "
        f"`transferred` = R_top / R_ideo copied from a reviewed model for the identical (text, chunk) pair. Pooled "
        f"over all ten models, {share_human * 100:.0f} % of the R_top labels have been seen or set by a human. "
        f"Per-model flags and spot checks: `judge_review_summary.csv`.\n"
    )

    # 4. Transfer
    L.append("## 4. Transfer assumption\n")
    L.append(md_table(transfer[["Check", "N", "Consistent", "Share"]]) + "\n")
    t0 = transfer.iloc[0]
    L.append(
        f"Reading: e5/simple retrieved the identical ordered list of five chunks for all ten models in "
        f"{t0['Consistent']} of {t0['N']} texts ({t0['Share'] * 100:.1f} %), so retrieval does not depend on the "
        f"generator, and every annotated chunk list is the one the model actually saw. That makes it valid to "
        f"copy R_top and R_ideo across models. A human reviewer who met the same pair in two models gave the "
        f"same label in {transfer.iloc[-2]['Share'] * 100:.0f} % (R_top) and {transfer.iloc[-1]['Share'] * 100:.0f} % "
        f"(R_ideo) of the cases; this repeat consistency of the reviewer is an upper bound for how exactly a "
        f"transferred label fits a given model's pair.\n"
    )

    # 5. Final labels
    L.append("## 5. Final label distributions\n")
    w = dist[dist["Weighting"] == "weighted by quadrant base rates"]
    tab = pd.DataFrame(
        {
            "Model": w["Model"],
            "Texts": w["Texts"],
            "Mean R_top (95 % CI)": [
                f"{r['Mean R_top']:.2f} ({r['Mean R_top CI low']:.2f}-{r['Mean R_top CI high']:.2f})"
                for _, r in w.iterrows()
            ],
            "R_top ≥ 4": w["Share R_top ≥ 4"].map(fmt),
            "Mean R_ideo (95 % CI)": [
                f"{r['Mean R_ideo']:.2f} ({r['Mean R_ideo CI low']:.2f}-{r['Mean R_ideo CI high']:.2f})"
                for _, r in w.iterrows()
            ],
            "R_ideo ≥ 4": w["Share R_ideo ≥ 4"].map(fmt),
            "Share cited (95 % CI)": [
                f"{r['Share cited (A_caus = 1)']:.2f} ({r['Share cited (A_caus = 1) CI low']:.2f}-{r['Share cited (A_caus = 1) CI high']:.2f})"
                for _, r in w.iterrows()
            ],
            "Cited per text": w["Cited chunks per text"].map(fmt),
        }
    )
    L.append("Weighted by the quadrant base rates of each model:\n")
    L.append(em._md_table(tab) + "\n")
    pw = _get(
        dist,
        Model="All models (Llama 3B: all)",
        Weighting="weighted by quadrant base rates",
    )
    pu = _get(dist, Model="All models (Llama 3B: all)", Weighting="unweighted")
    cites = w[~w["Model"].str.startswith("All")].set_index("Model")[
        "Cited chunks per text"
    ]
    L.append(
        f"Reading: weighted to the population, a retrieved chunk has a mean R_top of {pw['Mean R_top']:.2f} "
        f"(unweighted sample {pu['Mean R_top']:.2f}); {pw['Share R_top ≥ 4'] * 100:.0f} % of the chunks are on "
        f"the input's specific item (R_top ≥ 4). R_ideo is high almost everywhere (mean {pw['Mean R_ideo']:.2f}). "
        f"The models differ most in how many chunks they cite: from {cites.min():.2f} ({cites.idxmin()}) to "
        f"{cites.max():.2f} ({cites.idxmax()}) of 5 per text. R_top and R_ideo hardly differ between models, as "
        f"expected, because the chunks are the same and part of the labels is transferred. The CIs resample "
        f"texts within quadrants; the pooled row resamples text_index clusters.\n"
    )
    L.append("![fig_judge_labels.png](fig_judge_labels.png)\n")
    L.append(
        "Figure: share of chunks per R_top and R_ideo level, weighted by the quadrant base rates. Numbers inside "
        "the bars are percentages (shown from 9 %).\n"
    )
    L.append("Sampling weights (population texts of the quadrant / sampled texts):\n")
    wt = (
        weights.pivot(index="Model", columns="Quadrant", values="Weight")
        .reindex([model_label(m, s) for m, *_ in MODELS for s in scopes(m)])
        .reset_index()
    )
    L.append(md_table(wt) + "\n")
    main = weights[weights["Model"].isin([model_label(m) for m, *_ in MODELS])]
    largest = main.loc[main.groupby("Model")["Population texts"].idxmax()].set_index(
        "Model"
    )["Quadrant"]
    others = [f"{m} ({q})" for m, q in largest.items() if q != "Q3"]
    L.append(
        f"Reading: a weight above 1 means that a sampled text stands for several population texts. Q3 (joint "
        f"failure) is the largest quadrant in {int((largest == 'Q3').sum())} of {len(largest)} models"
        + (f"; the exceptions are {', '.join(others)}" if others else "")
        + ". Unweighted shares give the sampled texts of each quadrant equal weight. Both versions are in "
        "`judge_label_distribution.csv`; the weights per model and quadrant are in `judge_sampling_weights.csv`.\n"
    )
    (OUT_DIR / "04_why_rag_judge.md").write_text("\n".join(L), encoding="utf-8")


def cmd_judge(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading paired logs ...")
    pairs = load_pairs()
    check_population_counts(pairs)
    print("Human agreement (bootstrap) ...")
    agree_rows = [dict(r) for r in human_agreement(args.n_boot, args.seed)]
    jblock = "Judge vs humans (run 2026-10-23_142551, R_top ladder, R_ideo cut points)"
    for r in judge_vs_humans_from_report() + judge_vs_consensus(args.n_boot, args.seed):
        agree_rows.append({"Block": jblock, **r, "N chunks": 125, "N texts": 25})
    for r in cutpoint_loto():
        agree_rows.append(
            {"Block": "Cut points, out of sample", **r, "N chunks": 125, "N texts": 25}
        )
    agree = pd.DataFrame(agree_rows)[
        [
            "Block",
            "Dimension",
            "Comparison",
            "Statistic",
            "Value",
            "CI low",
            "CI high",
            "N chunks",
            "N texts",
            "Source",
        ]
    ]
    cinfo = consensus_info()
    cal = calibration_labels(cinfo)
    print("Final annotations ...")
    ch = load_final_annotations()
    review = review_summary(ch)
    transfer = transfer_check(ch)
    print("Label distributions (bootstrap) ...")
    dist, weights = label_distribution(ch, pairs, args.n_boot, args.seed)
    for df, name in (
        (agree, "judge_human_agreement.csv"),
        (cal, "judge_calibration_labels.csv"),
        (review, "judge_review_summary.csv"),
        (transfer, "judge_transfer_check.csv"),
        (dist, "judge_label_distribution.csv"),
        (weights, "judge_sampling_weights.csv"),
    ):
        print("  wrote", write_csv(df, name))
    plot_labels(dist)
    note = post_fix_note(ch)
    write_judge_report(
        agree,
        cal,
        review,
        transfer,
        dist,
        weights,
        cinfo,
        ch,
        note,
        args.n_boot,
        args.seed,
    )
    print("  wrote", OUT_DIR / "04_why_rag_judge.md", "and fig_judge_labels.pdf/.png")


# ===========================================================================
# composition: which parties does e5/simple retrieve?
# ===========================================================================

GROUP_OF = {"CDU": "CDU/CSU", "CSU": "CDU/CSU"}  # canonical CHES party -> display group
GROUP_DISPLAY = {
    "DIE LINKE": "DIE LINKE",
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "SPD": "SPD",
    "BSW": "BSW",
    "FDP": "FDP",
    "CDU/CSU": "CDU/CSU",
    "AfD": "AfD",
}
NO_POSITION = "no position"
NO_POSITION_RAW = {
    "",
    "fraktionslos",
    "unknown",
}  # casefolded raw spellings without a CHES position
DEFAULT_KB = (
    ROOT / "rag" / "ingest" / "artifacts" / "chunks.jsonl"
)  # the file indexed into chunks_e5 (HPC)
# Fallback: speeches per party of the indexed corpus (thesis table tab:kb-parties); an
# approximation, because longer speeches yield more chunks.
KB_SPEECHES = ROOT / "src" / "datasets" / "knowledge_base" / "kb_speech_counts.csv"
N_PERM = 1000
KB_NOTE = {
    "chunks": "This is the chunk file indexed into the e5 collection, i.e. the pool e5/simple retrieves from.",
    "speeches": (
        "APPROXIMATION: the knowledge-base shares are shares of speeches (thesis table tab:kb-parties), "
        "not of chunks. Retrieval draws chunks, and longer speeches yield more chunks; on the 1,731-speech "
        "development sample of the chunker the chunk share differs from the speech share by up to 2 "
        "percentage points (CDU/CSU 29.6 % vs 27.5 %, DIE LINKE 8.7 % vs 9.8 %). Expected shares and lifts "
        "are therefore approximate (lifts by up to about 10 %)."
    ),
}
BLOC_NAMES = em.BLOC_NAMES  # Left / Centre / Right

# Party names in chunk text (case-sensitive; 'Union' not after 'Europäische(n)').
MENTION_PATTERNS = {
    "DIE LINKE": r"\b(?:LINKEN?|Linken?)\b",
    "BÜNDNIS 90/DIE GRÜNEN": r"\b(?:GRÜNEN?|Grünen?)\b|BÜNDNIS\s90|Bündnis\s90",
    "SPD": r"\bSPD\b",
    "BSW": r"\bBSW\b|Bündnis\sSahra\sWagenknecht",
    "FDP": r"\bFDP\b",
    "CDU/CSU": r"\bCDU\b|\bCSU\b|(?<!Europäische\s)(?<!Europäischen\s)\bUnion\b|\bUnions(?:fraktion|parteien)\b",
    "AfD": r"\bAfD\b",
}


def party_group(canon: Optional[str]) -> Optional[str]:
    return GROUP_OF.get(canon, canon) if canon else None


def group_position(group: str) -> float:
    """Position of a display group for CHUNKS (the faction CDU/CSU gets the CDU value)."""
    return POSITIONS["CDU" if group == "CDU/CSU" else group]


def group_order(groups) -> List[str]:
    return sorted(groups, key=group_position)


def _raw_to_canon(raw, where: str) -> Optional[str]:
    canon = aci.norm_party(raw)
    if canon is None:
        key = " ".join(str(raw or "").split()).casefold()
        if key not in NO_POSITION_RAW:
            raise SystemExit(
                f"Party spelling {raw!r} ({where}) does not map to a CHES party"
            )
    elif canon not in POSITIONS:
        raise SystemExit(f"No CHES position for {canon!r} ({where})")
    return canon


def load_kb(path: Path) -> pd.DataFrame:
    """Knowledge-base chunks per party group, from chunks.jsonl (the file indexed into
    every Qdrant collection) or from a CSV with columns party,chunks counted from it."""
    if not path.exists():
        raise SystemExit(
            f"Knowledge base not found: {path}\nOn the HPC, count the parties of the indexed chunk file with\n"
            "  python -c \"import json,collections,csv; c=collections.Counter(json.loads(l).get('party') for l in "
            "open('rag/ingest/artifacts/chunks.jsonl')); w=csv.writer(open('kb_party_counts.csv','w',newline=''));"
            " w.writerow(['party','chunks']); [w.writerow([k,v]) for k,v in c.most_common()]\"\n"
            "and pass the CSV with --kb."
        )
    counts = Counter()
    if path.suffix == ".jsonl":
        unit = "chunks"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    counts[json.loads(line).get("party")] += 1
    else:
        kb = pd.read_csv(path, keep_default_na=False)
        unit = "chunks" if "chunks" in kb.columns else "speeches"
        for raw, n in zip(kb["party"], kb[unit]):
            counts[raw] += int(n)
    rows = defaultdict(lambda: {"Count": 0, "Raw spellings": set()})
    for raw, n in counts.items():
        canon = _raw_to_canon(raw, f"knowledge base {path.name}")
        g = party_group(canon) or NO_POSITION
        rows[g]["Count"] += n
        rows[g]["Raw spellings"].add(repr(raw))
    total = sum(r["Count"] for r in rows.values())
    positioned = sum(r["Count"] for g, r in rows.items() if g != NO_POSITION)
    out = []
    for g in group_order([g for g in rows if g != NO_POSITION]) + (
        [NO_POSITION] if NO_POSITION in rows else []
    ):
        r = rows[g]
        pos = group_position(g) if g != NO_POSITION else np.nan
        out.append(
            {
                "Party": GROUP_DISPLAY.get(g, g),
                "group": g,
                "Count": r["Count"],
                "Share": r["Count"] / total,
                "Share with a position": (
                    r["Count"] / positioned if g != NO_POSITION else np.nan
                ),
                "Position (lrgen, 1-7)": pos,
                "Bloc": BLOC_NAMES[em._bloc(pos)] if g != NO_POSITION else "",
                "Raw spellings": "; ".join(sorted(r["Raw spellings"])),
            }
        )
    df = pd.DataFrame(out)
    df.attrs["unit"] = unit
    return df


def kb_output(kb: pd.DataFrame) -> pd.DataFrame:
    """Knowledge-base table with unit-specific column names."""
    u = kb.attrs["unit"]
    return kb.drop(columns=["group"]).rename(
        columns={
            "Count": u.capitalize(),
            "Share": f"Share of all {u}",
            "Share with a position": f"Share of {u} with a position",
        }
    )


def load_retrieval() -> pd.DataFrame:
    """One row per (text, rank) of the e5/simple retrievals. Retrieval does not depend
    on the generator: the chunk lists are checked to be identical across all models."""
    logs = {}
    for folder, family, size, _ in MODELS:
        diag = json.loads(
            (ROOT / DIAGNOSTICS.format(family=family, size=size)).read_text(
                encoding="utf-8"
            )
        )
        logs[folder] = strata._merge_logs([ROOT / p for p in diag["rag_paths"]]).records
    texts = set.intersection(*(set(map(str, r)) for r in logs.values()))
    union = set.union(*(set(map(str, r)) for r in logs.values()))
    if texts != union:
        raise SystemExit(f"{len(union - texts)} texts are missing from some RAG logs")
    ref = {str(k): v for k, v in logs[MODELS[0][0]].items()}
    keyed = {m: {str(k): v for k, v in recs.items()} for m, recs in logs.items()}
    rows = []
    for t in sorted(texts, key=lambda x: int(x) if x.isdigit() else x):
        lists = {
            tuple(
                rca._ckey(c)
                for c in (keyed[m][t].get("inputs") or {}).get("retrieved_chunks") or []
            )
            for m in keyed
        }
        if len(lists) != 1:
            raise SystemExit(
                f"Text {t}: e5/simple retrieved different chunks for different models"
            )
        rec = ref[t]
        meta = rec.get("input_metadata") or {}
        a_canon = _raw_to_canon(meta.get("party"), f"author of text {t}")
        if a_canon is None:
            raise SystemExit(
                f"Text {t}: author party {meta.get('party')!r} has no position"
            )
        a_pos = POSITIONS[a_canon]
        chunks = (rec.get("inputs") or {}).get("retrieved_chunks") or []
        if len(chunks) != 5:
            raise SystemExit(f"Text {t}: {len(chunks)} chunks instead of 5")
        for rank, c in enumerate(chunks, start=1):
            canon = _raw_to_canon(c.get("party"), f"chunk of text {t}")
            g = party_group(canon)
            pos = group_position(g) if g else np.nan
            rows.append(
                {
                    "text_index": t,
                    "author_canon": a_canon,
                    "author_group": party_group(a_canon),
                    "author_pos": a_pos,
                    "author_bloc": em._bloc(a_pos),
                    "rank": rank,
                    "chunk_group": g or NO_POSITION,
                    "chunk_pos": pos,
                    "chunk_bloc": em._bloc(pos) if g else -1,
                    "chunk_text": c.get("text") or "",
                    "chunk_speaker": c.get("speaker") or "",
                    "chunk_party_raw": c.get("party"),
                }
            )
    return pd.DataFrame(rows)


class RetrievalArrays:
    """(n_texts, 5) arrays of the retrieved chunks for vectorised statistics."""

    def __init__(self, ret: pd.DataFrame, groups: List[str]):
        ret = ret.sort_values(["text_index", "rank"])
        self.texts = ret["text_index"].drop_duplicates().to_numpy()
        first = ret.groupby("text_index", sort=False).first().loc[self.texts]
        self.groups = groups
        code = {g: i for i, g in enumerate(groups)}
        n = len(self.texts)
        self.G = (
            ret["chunk_group"].map(lambda g: code.get(g, -1)).to_numpy().reshape(n, 5)
        )
        self.P = ret["chunk_pos"].to_numpy(dtype=float).reshape(n, 5)
        self.B = ret["chunk_bloc"].to_numpy(dtype=int).reshape(n, 5)
        self.a = first["author_group"].map(code).to_numpy()
        self.a_group = first["author_group"].to_numpy()
        self.a_pos = first["author_pos"].to_numpy(dtype=float)
        self.a_bloc = first["author_bloc"].to_numpy(dtype=int)
        opp = np.where(self.a_bloc == 0, 2, np.where(self.a_bloc == 2, 0, -9))
        self.opp = opp

    def per_text(self, perm: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        G, P, B = (
            (self.G, self.P, self.B)
            if perm is None
            else (self.G[perm], self.P[perm], self.B[perm])
        )
        valid = G >= 0
        nv = valid.sum(1).astype(float)
        nv[nv == 0] = np.nan
        with np.errstate(invalid="ignore"):
            out = {
                "same_party": (G == self.a[:, None]).sum(1) / nv,
                "same_bloc": ((B == self.a_bloc[:, None]) & valid).sum(1) / nv,
                "mean_position": np.nanmean(np.where(valid, P, np.nan), 1),
                "distance": np.nanmean(
                    np.where(valid, np.abs(P - self.a_pos[:, None]), np.nan), 1
                ),
                "opposite_bloc": np.where(
                    self.opp >= 0,
                    ((B == self.opp[:, None]) & valid).sum(1) / nv,
                    np.nan,
                ),
            }
        return out


STAT_LABELS = {
    "same_party": "Same-party share",
    "same_bloc": "Same-bloc share",
    "mean_position": "Mean position of retrieved parties",
    "distance": "Mean distance to author position",
    "opposite_bloc": "Opposite-bloc share",
}


def kb_expectation(arr: RetrievalArrays, kb: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per text: expected value of each statistic for 5 chunks drawn at random from
    the knowledge base (chunks with a position)."""
    kbp = kb[kb["group"] != NO_POSITION]
    p = {g: s for g, s in zip(kbp["group"], kbp["Share with a position"])}
    pos = {g: group_position(g) for g in p}
    bloc = {g: em._bloc(pos[g]) for g in p}
    out = {
        "same_party": np.array([p.get(g, 0.0) for g in arr.a_group]),
        "same_bloc": np.array(
            [sum(p[g] for g in p if bloc[g] == ab) for ab in arr.a_bloc]
        ),
        "mean_position": np.full(len(arr.texts), sum(p[g] * pos[g] for g in p)),
        "distance": np.array(
            [sum(p[g] * abs(pos[g] - ap) for g in p) for ap in arr.a_pos]
        ),
        "opposite_bloc": np.array(
            [
                sum(p[g] for g in p if bloc[g] == o) if o >= 0 else np.nan
                for o in arr.opp
            ]
        ),
    }
    return out


def author_scopes(arr: RetrievalArrays) -> List[Tuple[str, np.ndarray]]:
    scopes_ = [("All posts", np.ones(len(arr.texts), dtype=bool))]
    for g in group_order(set(arr.a_group)):
        scopes_.append((GROUP_DISPLAY[g], arr.a_group == g))
    return scopes_


def _mean_ci(values: np.ndarray, n_boot: int, rng) -> Tuple[float, float, float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = rca.paired_boot(v, n_boot, rng)
    return float(v.mean()), lo, hi


def chance_level(
    arr: RetrievalArrays, kb: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    obs = arr.per_text()
    exp = kb_expectation(arr, kb)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(arr.texts)) for _ in range(N_PERM)]
    null = [arr.per_text(p) for p in perms]
    rows = []
    for scope, mask in author_scopes(arr):
        for stat, label in STAT_LABELS.items():
            o = obs[stat][mask]
            if np.all(np.isnan(o)):
                continue
            mean, lo, hi = _mean_ci(o, n_boot, np.random.default_rng(seed))
            e = exp[stat][mask]
            ok = ~np.isnan(o) & ~np.isnan(e)
            null_means = np.array([np.nanmean(nl[stat][mask]) for nl in null])
            if scope == "All posts" and stat == "mean_position":
                null_means = np.full(
                    N_PERM, np.nan
                )  # shuffling across texts cannot change the overall mean
            centre = null_means.mean()
            rows.append(
                {
                    "Author party": scope,
                    "Statistic": label,
                    "Posts": int(ok.sum()),
                    "Observed": mean,
                    "CI low": lo,
                    "CI high": hi,
                    "Knowledge-base expectation": float(np.mean(e[ok])),
                    "Observed - expectation": mean - float(np.mean(e[ok])),
                    "Wilcoxon p (observed vs expectation)": rca.wilcoxon_p(
                        o[ok] - e[ok]
                    ),
                    "Permutation null mean": centre,
                    "Permutation null 2.5 %": float(np.percentile(null_means, 2.5)),
                    "Permutation null 97.5 %": float(np.percentile(null_means, 97.5)),
                    "Permutation p": (
                        np.nan
                        if np.isnan(centre)
                        else float(
                            (
                                1
                                + np.sum(
                                    np.abs(null_means - centre) >= abs(mean - centre)
                                )
                            )
                            / (N_PERM + 1)
                        )
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.insert(
        df.columns.get_loc("Wilcoxon p (observed vs expectation)") + 1,
        "Wilcoxon p (Holm)",
        em._holm(df["Wilcoxon p (observed vs expectation)"].tolist()),
    )
    return df


def chapter03_crosscheck(ret: pd.DataFrame) -> Dict[str, float]:
    """Same-party / same-bloc / distance as defined in chapter 03 (positions =
    mean(lrecon, galtan), em.load_party_positions), for comparison only."""
    pos03 = em.load_party_positions(ROOT / em.CHES_PATH, em.CHES_WAVE)
    out = defaultdict(list)
    for t, g in ret.groupby("text_index"):
        a = g["author_group"].iloc[0]
        mapped = [c for c in g["chunk_group"] if c != NO_POSITION]
        if not mapped:
            continue
        out["same_party"].append(np.mean([c == a for c in mapped]))
        ap = pos03[a]
        out["same_bloc"].append(
            np.mean([em._bloc(pos03[c]) == em._bloc(ap) for c in mapped])
        )
        out["distance"].append(np.mean([abs(pos03[c] - ap) for c in mapped]))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    # Parties whose bloc differs between the two position sets (chunks: group position;
    # authors: CSU keeps its own lrgen position here, chapter 03 uses CDU/CSU).
    moved = []
    for g in group_order([g for g in pos03 if g in GROUP_DISPLAY]):
        b_now, b_03 = em._bloc(group_position(g)), em._bloc(pos03[g])
        if b_now != b_03:
            moved.append(
                f"{GROUP_DISPLAY[g]} ({BLOC_NAMES[b_03].lower()} → {BLOC_NAMES[b_now].lower()})"
            )
    if em._bloc(POSITIONS["CSU"]) != em._bloc(pos03["CDU/CSU"]):
        moved.append(
            f"CSU authors ({BLOC_NAMES[em._bloc(pos03['CDU/CSU'])].lower()} → "
            f"{BLOC_NAMES[em._bloc(POSITIONS['CSU'])].lower()})"
        )
    res["moved"] = moved
    return res


def composition_matrix(
    ret: pd.DataFrame, kb: pd.DataFrame, arr: RetrievalArrays, n_boot: int, seed: int
) -> pd.DataFrame:
    cols = group_order(
        [g for g in set(ret["chunk_group"]) | set(kb["group"]) if g != NO_POSITION]
    ) + [NO_POSITION]
    kb_share = dict(zip(kb["group"], kb["Share"]))
    counts = (
        ret.groupby(["text_index", "chunk_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=arr.texts, columns=cols, fill_value=0)
    )
    rows = []
    for scope, mask in author_scopes(arr):
        sub = counts[mask]
        m = multiplicities(
            sub.index.to_numpy(), None, n_boot, np.random.default_rng(seed)
        )
        den = np.full(len(sub), 5.0)
        for g in cols:
            num = sub[g].to_numpy(dtype=float)
            share = num.sum() / den.sum()
            lo, hi = ci(boot_ratio(num, den, np.ones(len(sub)), m))
            ks = kb_share.get(g, 0.0)
            rows.append(
                {
                    "Author party": scope,
                    "Chunk party": GROUP_DISPLAY.get(g, g),
                    "Posts": int(mask.sum()),
                    "Retrieved chunks": int(num.sum()),
                    "Share": share,
                    "CI low": lo,
                    "CI high": hi,
                    "Knowledge-base share": ks,
                    "Lift": share / ks if ks else np.nan,
                    "Lift CI low": lo / ks if ks else np.nan,
                    "Lift CI high": hi / ks if ks else np.nan,
                }
            )
    for g in cols:
        ks = kb_share.get(g, 0.0)
        rows.append(
            {
                "Author party": "Knowledge base",
                "Chunk party": GROUP_DISPLAY.get(g, g),
                "Posts": np.nan,
                "Retrieved chunks": np.nan,
                "Share": ks,
                "CI low": np.nan,
                "CI high": np.nan,
                "Knowledge-base share": ks,
                "Lift": 1.0 if ks else np.nan,
                "Lift CI low": np.nan,
                "Lift CI high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def by_author_party(
    ret: pd.DataFrame, arr: RetrievalArrays, n_boot: int, seed: int
) -> pd.DataFrame:
    obs = arr.per_text()
    npos = (arr.G < 0).sum(1)
    rows = []
    for scope, mask in author_scopes(arr):
        row = {
            "Author party": scope,
            "Posts": int(mask.sum()),
            "Retrieved chunks": int(mask.sum() * 5),
            "Chunks without position": int(npos[mask].sum()),
            "Posts without any positioned chunk": int(
                np.isnan(obs["same_party"][mask]).sum()
            ),
            "Author position": (
                float(np.mean(arr.a_pos[mask])) if scope != "All posts" else np.nan
            ),
            "Posts with an opposite bloc (left or right authors)": int(
                (arr.opp[mask] >= 0).sum()
            ),
        }
        for stat, label in STAT_LABELS.items():
            mean, lo, hi = _mean_ci(
                obs[stat][mask], n_boot, np.random.default_rng(seed)
            )
            row[label], row[f"{label} CI low"], row[f"{label} CI high"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def rank_position(arr: RetrievalArrays, n_boot: int, seed: int) -> pd.DataFrame:
    same = np.where(arr.G >= 0, (arr.G == arr.a[:, None]).astype(float), np.nan)
    rows = []
    for scope, mask in author_scopes(arr):
        s = same[mask]
        row = {"Author party": scope, "Posts": int(mask.sum())}
        for r in range(5):
            col = s[:, r]
            row[f"Same-party share at rank {r + 1}"] = float(np.nanmean(col))
        both = ~np.isnan(s[:, 0]) & ~np.isnan(s[:, 4])
        d = s[both, 0] - s[both, 4]
        lo, hi = rca.paired_boot(d, n_boot, np.random.default_rng(seed))
        row.update(
            {
                "Posts with rank 1 and 5 positioned": int(both.sum()),
                "Rank 1 - rank 5": float(d.mean()),
                "CI low": lo,
                "CI high": hi,
                "Wilcoxon p": rca.wilcoxon_p(d),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Wilcoxon p (Holm)"] = em._holm(df["Wilcoxon p"].tolist())
    return df


def _mentions(text: str) -> set:
    import re

    return {g for g, pat in MENTION_PATTERNS.items() if re.search(pat, text)}


def party_mentions(
    ret: pd.DataFrame, arr: RetrievalArrays, n_boot: int, seed: int
) -> pd.DataFrame:
    ret = ret.copy()
    ret["named"] = ret["chunk_text"].map(_mentions)
    ret["names_any"] = ret["named"].map(bool)
    ret["names_own"] = [g in n for g, n in zip(ret["chunk_group"], ret["named"])]
    ret["names_other"] = [
        bool(n - {g}) for g, n in zip(ret["chunk_group"], ret["named"])
    ]
    per_text = (
        ret.groupby("text_index")
        .agg(
            n=("rank", "size"),
            any_=("names_any", "sum"),
            own=("names_own", "sum"),
            other=("names_other", "sum"),
        )
        .loc[arr.texts]
    )
    rows = []
    for scope, mask in author_scopes(arr):
        sub = per_text[mask]
        m = multiplicities(
            sub.index.to_numpy(), None, n_boot, np.random.default_rng(seed)
        )
        ones = np.ones(len(sub))
        row = {
            "Author party": scope,
            "Posts": int(mask.sum()),
            "Retrieved chunks": int(sub["n"].sum()),
        }
        for col, label in (
            ("any_", "names any party"),
            ("own", "names its own party"),
            ("other", "names another party"),
        ):
            row[f"Share {label}"] = sub[col].sum() / sub["n"].sum()
            row[f"Share {label} CI low"], row[f"Share {label} CI high"] = ci(
                boot_ratio(sub[col], sub["n"], ones, m)
            )
        if scope != "All posts":
            g = next(k for k, v in GROUP_DISPLAY.items() if v == scope)
            # Chunks of OTHER parties' speakers that name this author party: posts of this party
            # vs. posts of all other parties (reference).
            ref = ret[(ret["author_group"] != g) & (ret["chunk_group"] != g)]
            this = ret[(ret["author_group"] == g) & (ret["chunk_group"] != g)]
            this_t = this.groupby("text_index").agg(
                k=("named", lambda s: sum(g in x for x in s)), n=("rank", "size")
            )
            ref_t = ref.groupby("text_index").agg(
                k=("named", lambda s: sum(g in x for x in s)), n=("rank", "size")
            )
            row["Chunks from other parties"] = int(this_t["n"].sum())
            row["... naming the author party"] = int(this_t["k"].sum())
            row["Share naming the author party"] = this_t["k"].sum() / this_t["n"].sum()
            row["Reference: same share in posts of other parties"] = (
                ref_t["k"].sum() / ref_t["n"].sum()
            )
            rng = np.random.default_rng(seed)
            b1 = boot_ratio(
                this_t["k"],
                this_t["n"],
                np.ones(len(this_t)),
                multiplicities(this_t.index.to_numpy(), None, n_boot, rng),
            )
            b2 = boot_ratio(
                ref_t["k"],
                ref_t["n"],
                np.ones(len(ref_t)),
                multiplicities(ref_t.index.to_numpy(), None, n_boot, rng),
            )
            (
                row["Share naming the author party CI low"],
                row["Share naming the author party CI high"],
            ) = ci(b1)
            diff = b1 - b2
            row["Difference"] = (
                row["Share naming the author party"]
                - row["Reference: same share in posts of other parties"]
            )
            row["Difference CI low"], row["Difference CI high"] = ci(diff)
            row["Bootstrap p (difference)"] = em._boot_p_two_sided(diff)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Bootstrap p (Holm)"] = em._holm(df["Bootstrap p (difference)"].tolist())
    return df


def plot_composition(matrix: pd.DataFrame):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "seq_blue",
        ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"],
    )
    rows = [r for r in matrix["Author party"].drop_duplicates()]
    cols = list(matrix["Chunk party"].drop_duplicates())
    share = matrix.pivot(
        index="Author party", columns="Chunk party", values="Share"
    ).loc[rows, cols]
    lift = matrix.pivot(index="Author party", columns="Chunk party", values="Lift").loc[
        rows, cols
    ]
    vmax = float(np.nanmax(share.to_numpy()))
    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 2.2, 0.62 * len(rows) + 1.4))
    ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            s, lf = share.iloc[i, j], lift.iloc[i, j]
            dark = s > 0.45 * vmax
            txt = f"{s * 100:.0f} %" + (
                "" if r == "Knowledge base" or pd.isna(lf) else f"\n×{lf:.1f}"
            )
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if dark else INK,
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8.5, color=INK, rotation=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_xlabel(
        "Party of the retrieved chunk's speaker",
        fontsize=9,
        color=INK_SECONDARY,
        labelpad=8,
    )
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Author party of the post", fontsize=9, color=INK_SECONDARY)
    ax.axhline(
        0.5, color="white", linewidth=4
    )  # 'All posts' above, 'Knowledge base' below the author rows
    ax.axhline(len(rows) - 1.5, color="white", linewidth=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save_fig(fig, "fig_composition_heatmap")


def write_composition_report(
    kb,
    kb_path,
    ret,
    arr,
    chance,
    cross03,
    matrix,
    by_author,
    ranks,
    mentions,
    n_boot,
    seed,
    readings: Dict[str, str],
):
    L = ["# Why retrieval helps or hurts: what e5/simple retrieves\n", "## Data\n"]
    n_np = int((ret["chunk_group"] == NO_POSITION).sum())
    raw_np = (
        ret.loc[ret["chunk_group"] == NO_POSITION, "chunk_party_raw"]
        .map(lambda x: " ".join(str(x).split()))
        .value_counts()
    )
    mix = pd.Series(arr.a_group).map(GROUP_DISPLAY).value_counts()
    L.append(
        f"Retrieved chunks: the e5/simple RAG logs of the ten models (k = 5). Retrieval does not depend on the "
        f"generator: for all {len(arr.texts)} texts the ten models received the identical ordered chunk list "
        f"(checked), so every retrieval fact is computed once per text ({len(arr.texts)} texts, "
        f"{len(ret)} chunks). Because no model enters, the Llama-3B split by baseline score does not apply. "
        f"Author parties: {', '.join(f'{p} {mix.get(p, 0)}' for p in [GROUP_DISPLAY[g] for g in group_order(set(arr.a_group))])} "
        f"posts (CDU/CSU = CDU, CSU and CDU/CSU authors; CSU authors keep the CSU position 5.54, all others the "
        f"CDU value 4.95). Chunks without a position: {n_np} "
        f"({', '.join(f'{k} {v}' for k, v in raw_np.items())}); they count in the composition matrix as "
        f"'no position' and are left out of all shares, positions and distances. Knowledge base: `{kb_path}` "
        f"({int(kb['Count'].sum())} {kb.attrs['unit']}). {KB_NOTE[kb.attrs['unit']]} Positions: CHES 2024 lrgen, two decimals; blocs left < 3.4 ≤ centre "
        f"< 5.2 ≤ right. Bootstrap {n_boot} resamples of texts, seed {seed}; permutation null {N_PERM} shuffles of "
        f"the chunk lists across texts, seed {seed}.\n"
    )
    L.append("## 1. Knowledge base and chance level\n")
    L.append(md_table(kb_output(kb).drop(columns=["Raw spellings"])) + "\n")
    L.append(readings["kb"] + "\n")
    allp = chance[chance["Author party"] == "All posts"][
        [
            "Statistic",
            "Observed",
            "CI low",
            "CI high",
            "Knowledge-base expectation",
            "Permutation null mean",
            "Permutation null 2.5 %",
            "Permutation null 97.5 %",
            "Wilcoxon p (Holm)",
            "Permutation p",
        ]
    ]
    L.append(md_table(allp) + "\n")
    L.append(readings["chance"] + "\n")
    L.append(
        f"Cross-check with chapter 03: with the chapter-03 definitions (party positions = mean of CHES lrecon and "
        f"galtan, `em.load_party_positions`) the same texts give same-party {cross03['same_party']:.4f}, same-bloc "
        f"{cross03['same_bloc']:.4f} and distance {cross03['distance']:.4f} "
        f"(`party_agreement_by_condition.csv`, e5/simple: 0.3002, 0.5467, 1.2232; there averaged over models). "
        f"Same-party does not depend on positions. Same-bloc and distance differ because this chapter places "
        f"parties by lrgen; this moves {', '.join(cross03['moved']) or 'no party'} to another bloc.\n"
    )
    L.append("## 2. Retrieved party composition by author party\n")
    wide = matrix.pivot(index="Author party", columns="Chunk party", values="Share")
    wide = wide.loc[
        matrix["Author party"].drop_duplicates(),
        matrix["Chunk party"].drop_duplicates(),
    ].reset_index()
    L.append("Share of the retrieved chunks (rows sum to 1):\n")
    L.append(md_table(wide) + "\n")
    L.append(readings["matrix"] + "\n")
    tab = by_author[["Author party", "Posts", "Author position"]].copy()
    for label in STAT_LABELS.values():
        tab[label] = [
            f"{v:.2f} ({lo:.2f}-{hi:.2f})" if not pd.isna(v) else ""
            for v, lo, hi in zip(
                by_author[label],
                by_author[f"{label} CI low"],
                by_author[f"{label} CI high"],
            )
        ]
    L.append(md_table(tab) + "\n")
    L.append(readings["by_author"] + "\n")
    L.append("## 3. Lift (observed share / knowledge-base share)\n")
    lw = matrix[~matrix["Author party"].isin(["Knowledge base"])].pivot(
        index="Author party", columns="Chunk party", values="Lift"
    )
    lw = lw.loc[
        [r for r in matrix["Author party"].drop_duplicates() if r != "Knowledge base"],
        matrix["Chunk party"].drop_duplicates(),
    ].reset_index()
    L.append(md_table(lw) + "\n")
    L.append(readings["lift"] + "\n")
    L.append("![fig_composition_heatmap.png](fig_composition_heatmap.png)\n")
    L.append(
        "Figure: share of the retrieved chunks by author party (rows) and the party of the chunk's speaker "
        "(columns); the second line is the lift over the knowledge-base share. Last row: knowledge base.\n"
    )
    L.append("## 4. Rank position\n")
    L.append(md_table(ranks.drop(columns=["Wilcoxon p"])) + "\n")
    L.append(readings["rank"] + "\n")
    L.append("## 5. Party mentions in the retrieved chunks\n")
    cols = [
        "Author party",
        "Retrieved chunks",
        "Share names any party",
        "Share names its own party",
        "Share names another party",
    ]
    L.append(md_table(mentions[cols]) + "\n")
    L.append(readings["mentions"] + "\n")
    cols = [
        "Author party",
        "Chunks from other parties",
        "... naming the author party",
        "Share naming the author party",
        "Reference: same share in posts of other parties",
        "Difference",
        "Difference CI low",
        "Difference CI high",
        "Bootstrap p (Holm)",
    ]
    L.append(md_table(mentions[mentions["Author party"] != "All posts"][cols]) + "\n")
    L.append(readings["about"] + "\n")
    L.append(
        "Regex (case-sensitive): "
        + "; ".join(f"{GROUP_DISPLAY[g]}: `{p}`" for g, p in MENTION_PATTERNS.items())
        + ". Lower-case adjectives (linke, grüne) are not counted. The chunk text still contains the chair's words "
        "when the chair interrupts a speech (e.g. 'Nächster Redner ist für die Fraktion …'), so a small part of "
        "the mentions are procedural.\n"
    )
    (OUT_DIR / "04_why_rag_composition.md").write_text("\n".join(L), encoding="utf-8")


def composition_readings(
    kb, chance, matrix, by_author, ranks, mentions, arr
) -> Dict[str, str]:
    """Reading paragraphs. Numbers are filled in from the tables; the wording was written
    for the results of 12 Nov 2026 and should be re-checked if the knowledge base changes.
    """
    unit = kb.attrs["unit"]
    pct = lambda v: f"{v * 100:.0f} %"  # noqa: E731
    c = lambda scope, stat: _get(
        chance, **{"Author party": scope, "Statistic": STAT_LABELS[stat]}
    )  # noqa: E731
    sh = lambda a, ch: _get(
        matrix, **{"Author party": a, "Chunk party": ch}
    )  # noqa: E731
    kbs = dict(zip(kb["Party"], kb["Share"]))
    big = kb[kb["group"] != NO_POSITION].sort_values("Share", ascending=False)
    author_mean = float(np.mean(arr.a_pos))
    sp, sb = c("All posts", "same_party"), c("All posts", "same_bloc")
    mp, ob = c("All posts", "mean_position"), c("All posts", "opposite_bloc")
    authors = [r for r in by_author["Author party"] if r != "All posts"]
    lifts = {a: sh(a, a)["Lift"] for a in authors}
    top, low = max(lifts, key=lifts.get), min(lifts, key=lifts.get)
    wil_ns = [
        a
        for a in authors
        if c(a, "same_party")["Wilcoxon p (Holm)"] >= 0.05
        and c(a, "same_party")["Permutation p"] < 0.05
    ]
    afd = _get(mentions, **{"Author party": "AfD"})
    lin = _get(mentions, **{"Author party": "DIE LINKE"})
    allm = _get(mentions, **{"Author party": "All posts"})
    rk = _get(ranks, **{"Author party": "All posts"})
    rk_sig = [
        a
        for a in authors
        if _get(ranks, **{"Author party": a})["Wilcoxon p (Holm)"] < 0.05
    ]
    neg = mentions[
        (mentions["Author party"] != "All posts") & (mentions["Difference CI high"] < 0)
    ]["Author party"].tolist()
    r = {}
    r["kb"] = (
        f"Reading: the pool is dominated by {big.iloc[0]['Party']} ({pct(big.iloc[0]['Share'])}) and "
        f"{big.iloc[1]['Party']} ({pct(big.iloc[1]['Share'])}); DIE LINKE ({pct(kbs['DIE LINKE'])}) and BSW "
        f"({kbs.get('BSW', 0) * 100:.1f} %) are the smallest groups. These shares of {unit} define what five random "
        f"chunks would contain{' (approximately, see Data)' if unit == 'speeches' else ''}."
    )
    r["chance"] = (
        f"Reading: for a random post, {pct(sp['Observed'])} of the retrieved chunks come from the author's own party, "
        f"against {pct(sp['Knowledge-base expectation'])} for random chunks and {pct(sp['Permutation null mean'])} "
        f"under the permutation null, which keeps the overall mix of retrieved parties but breaks the link to the "
        f"post. Same-bloc ({pct(sb['Observed'])} vs {pct(sb['Knowledge-base expectation'])} / "
        f"{pct(sb['Permutation null mean'])}) and opposite-bloc chunks ({pct(ob['Observed'])} vs "
        f"{pct(ob['Knowledge-base expectation'])} / {pct(ob['Permutation null mean'])}) point the same way: retrieval "
        f"is clearly party- and bloc-sensitive, but most chunks still come from other parties. The retrieved parties "
        f"sit at {mp['Observed']:.2f} on average, left of the pool ({mp['Knowledge-base expectation']:.2f}) and right "
        f"of the posts' authors ({author_mean:.2f})."
    )
    diag_max, not_max = [], []
    for a in authors:
        row = matrix[
            (matrix["Author party"] == a) & (matrix["Chunk party"] != NO_POSITION)
        ]
        best = row.loc[row["Share"].idxmax()]
        own = sh(a, a)["Share"]
        if best["Chunk party"] == a or abs(best["Share"] - own) < 0.005:
            diag_max.append(f"{a} ({pct(own)})")
        else:
            not_max.append(
                f"{a} posts get more chunks from {best['Chunk party']} ({pct(best['Share'])}) than from "
                f"their own party ({pct(own)})"
            )
    r["matrix"] = (
        f"Reading: the own party is the largest (or tied largest) column for {', '.join(diag_max)}"
        + (f"; {'; '.join(not_max)}" if not_max else "")
        + f". AfD posts still get {pct(1 - sh('AfD', 'AfD')['Share'] - sh('AfD', 'no position')['Share'])} of their "
        f"chunks from other parties. The CIs per cell are in `retrieval_composition_matrix.csv`."
    )
    r["by_author"] = (
        f"Reading: the same-party share is highest for AfD ({pct(c('AfD', 'same_party')['Observed'])}) and DIE LINKE "
        f"({pct(c('DIE LINKE', 'same_party')['Observed'])}) posts, the two parties at the ends of the scale, and "
        f"lowest for FDP ({pct(c('FDP', 'same_party')['Observed'])}) and Grüne "
        f"({pct(c('Grüne', 'same_party')['Observed'])}). The retrieved parties are far from the author for the "
        f"parties at the ends of the scale (mean distance AfD "
        f"{c('AfD', 'distance')['Observed']:.2f}, DIE LINKE {c('DIE LINKE', 'distance')['Observed']:.2f}) and close "
        f"for SPD ({c('SPD', 'distance')['Observed']:.2f}). For AfD posts {pct(c('AfD', 'opposite_bloc')['Observed'])} "
        f"of the chunks come from the left bloc. The CDU/CSU row mixes CDU authors (centre) and CSU authors (right); "
        f"its opposite-bloc share rests on {int(c('CDU/CSU', 'opposite_bloc')['Posts'])} CSU posts only. Tests "
        f"against chance are in `retrieval_chance_level.csv`; "
        + (
            f"for {', '.join(wil_ns)} the same-party excess is significant in the permutation test but not in the "
            f"Holm-corrected Wilcoxon test, because most of their posts receive no own-party chunk at all (median "
            f"0), so the mean rises without the median."
            if wil_ns
            else "all per-party tests agree."
        )
        + ""
    )
    over = [a for a in authors if lifts[a] > 1]
    afd_other = [sh(a, "AfD")["Lift"] for a in authors if a != "AfD"]
    r["lift"] = (
        (
            "Reading: every party is over-retrieved for its own posts (diagonal lift > 1). "
            if len(over) == len(authors)
            else f"Reading: {', '.join(over)} are over-retrieved for their own posts (diagonal lift > 1). "
        )
        + f"The strongest own-party lift is "
        f"{top} (×{lifts[top]:.1f}), the weakest {low} (×{lifts[low]:.1f}). AfD chunks are retrieved "
        f"{lifts['AfD']:.1f} times as often for AfD posts as their share of the pool, but less often than that share "
        f"for the posts of every other party (lift {min(afd_other):.1f}-{max(afd_other):.1f}). DIE LINKE is "
        f"under-retrieved for CDU/CSU posts (×{sh('CDU/CSU', 'DIE LINKE')['Lift']:.1f}). The lifts of the small "
        f"groups (BSW, no position) rest on few chunks and are not interpreted."
        + (
            " With speech shares as the reference, lifts are approximate (see Data)."
            if unit == "speeches"
            else ""
        )
    )
    r["rank"] = (
        f"Reading: over all posts the author's party is somewhat more frequent at rank 1 "
        f"({pct(rk['Same-party share at rank 1'])}) than at rank 5 ({pct(rk['Same-party share at rank 5'])}); the "
        f"difference is {rk['Rank 1 - rank 5']:.3f} (95 % CI {rk['CI low']:.3f} to {rk['CI high']:.3f}, Holm p "
        f"{rk['Wilcoxon p (Holm)']:.3f}). Within the author parties "
        + (
            f"only {', '.join(rk_sig)} reaches significance after Holm correction"
            if rk_sig
            else "no difference survives the Holm correction"
        )
        + ", so the rank adds little beyond the overall party sensitivity."
    )
    r["mentions"] = (
        f"Reading: {pct(allm['Share names any party'])} of the retrieved chunks name at least one party, mostly another "
        f"party than the speaker's own ({pct(allm['Share names another party'])} vs "
        f"{pct(allm['Share names its own party'])}). Party names are most frequent in chunks retrieved for AfD posts "
        f"({pct(afd['Share names any party'])})."
    )
    r["about"] = (
        f"Reading: for AfD posts, {int(afd['... naming the author party'])} of the {int(afd['Chunks from other parties'])} "
        f"chunks from non-AfD speakers ({pct(afd['Share naming the author party'])}) name the AfD, against "
        f"{pct(afd['Reference: same share in posts of other parties'])} of the non-AfD chunks retrieved for other "
        f"posts: a large part of the 'other-party' evidence for AfD posts is speech about the AfD rather than by it. "
        f"For DIE LINKE the effect is smaller ({int(lin['... naming the author party'])} of "
        f"{int(lin['Chunks from other parties'])}, {pct(lin['Share naming the author party'])} vs "
        f"{pct(lin['Reference: same share in posts of other parties'])})."
        + (
            f" For {', '.join(neg)} posts, chunks of other parties name the author's party less often than for other "
            f"posts."
            if neg
            else ""
        )
        + " p values: two-sided bootstrap test of the difference (texts resampled within each group), Holm-corrected "
        "across the six author parties."
    )
    return r


def cmd_composition(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kb_path = (
        Path(args.kb)
        if args.kb
        else (DEFAULT_KB if DEFAULT_KB.exists() else KB_SPEECHES)
    )
    if kb_path == KB_SPEECHES:
        print(
            f"  NOTE: chunk file not found; using speech counts {KB_SPEECHES.name} as an approximation"
        )
    kb = load_kb(kb_path)
    print("Loading e5/simple retrievals ...")
    ret = load_retrieval()
    groups = group_order(
        [
            g
            for g in set(ret["chunk_group"])
            | set(kb["group"])
            | set(ret["author_group"])
            if g != NO_POSITION
        ]
    )
    arr = RetrievalArrays(ret, groups)
    print("Chance level and permutation null ...")
    chance = chance_level(arr, kb, args.n_boot, args.seed)
    cross03 = chapter03_crosscheck(ret)
    matrix = composition_matrix(ret, kb, arr, args.n_boot, args.seed)
    by_author = by_author_party(ret, arr, args.n_boot, args.seed)
    ranks = rank_position(arr, args.n_boot, args.seed)
    mentions = party_mentions(ret, arr, args.n_boot, args.seed)
    kb_out = kb_output(kb)
    for df, name in (
        (kb_out, "kb_party_distribution.csv"),
        (chance, "retrieval_chance_level.csv"),
        (matrix, "retrieval_composition_matrix.csv"),
        (by_author, "retrieval_by_author_party.csv"),
        (ranks, "retrieval_rank_position.csv"),
        (mentions, "retrieval_party_mentions.csv"),
    ):
        print("  wrote", write_csv(df, name))
    plot_composition(matrix)
    readings = composition_readings(kb, chance, matrix, by_author, ranks, mentions, arr)
    try:
        kb_label = kb_path.relative_to(ROOT)
    except ValueError:
        kb_label = kb_path
    write_composition_report(
        kb,
        kb_label,
        ret,
        arr,
        chance,
        cross03,
        matrix,
        by_author,
        ranks,
        mentions,
        args.n_boot,
        args.seed,
        readings,
    )
    print(
        "  wrote",
        OUT_DIR / "04_why_rag_composition.md",
        "and fig_composition_heatmap.pdf/.png",
    )


# ===========================================================================
# direction: more left, more right or more centrist with RAG?
# ===========================================================================

AUTHOR_GROUPS = ["DIE LINKE", "BÜNDNIS 90/DIE GRÜNEN", "SPD", "FDP", "CDU/CSU", "AfD"]
MOVE_THRESHOLD = 0.5  # |RAG - baseline| >= 0.5 counts as moved (= the pass threshold)
SMALL_N = 30
COL_BASE, COL_RAG = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated)
COL_LINKE, COL_AFD = (
    "#6250d6",
    "#1baf7a",
)  # violet / aqua (validated; aqua < 3:1 -> markers + labels)


def direction_data() -> pd.DataFrame:
    pairs = load_pairs()
    pairs["party"] = pairs["input_party"].map(
        party_group
    )  # CSU merged into CDU/CSU (as in the logs' party labels)
    pairs["se_B"] = pairs["B"] - pairs["target"]
    pairs["se_R"] = pairs["R"] - pairs["target"]
    pairs["ae_B"] = pairs["se_B"].abs()
    pairs["ae_R"] = pairs["se_R"].abs()
    return pairs


def _overall_metrics(T, P, L, A) -> Dict[str, np.ndarray]:
    """Metrics over the last axis (vectorised for bootstrap index matrices)."""
    lo, hi = em.BLOC_THRESHOLDS
    mt, mp = T.mean(-1, keepdims=True), P.mean(-1, keepdims=True)
    slope = ((T - mt) * (P - mp)).sum(-1) / ((T - mt) ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = (P * A).sum(-1) / A.sum(-1) - (P * L).sum(-1) / L.sum(-1)
    return {
        "Mean absolute error": np.abs(P - T).mean(-1),
        "Mean signed error": (P - T).mean(-1),
        "Mean prediction": P.mean(-1),
        "SD of predictions": P.std(-1, ddof=1),
        "Calibration slope": slope,
        "Calibration intercept": mp[..., 0] - slope * mt[..., 0],
        "Spread AfD - DIE LINKE": spread,
        "Share in centre band [3.4, 5.2)": ((P >= lo) & (P < hi)).mean(-1),
    }


def direction_overall(d: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    lo_c, hi_c = em.BLOC_THRESHOLDS
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            g = in_scope(d[d["model"] == folder], scope)
            T, B, R = (g[c].to_numpy(float) for c in ("target", "B", "R"))
            L = (g["party"] == "DIE LINKE").to_numpy(float)
            A = (g["party"] == "AfD").to_numpy(float)
            mb, mr = _overall_metrics(T, B, L, A), _overall_metrics(T, R, L, A)
            idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
            bb, br = _overall_metrics(T[idx], B[idx], L[idx], A[idx]), _overall_metrics(
                T[idx], R[idx], L[idx], A[idx]
            )
            # Per-post paired differences where they exist (Wilcoxon); otherwise bootstrap p.
            per_post = {
                "Mean absolute error": np.abs(R - T) - np.abs(B - T),
                "Mean signed error": R - B,
                "Mean prediction": R - B,
                "Share in centre band [3.4, 5.2)": ((R >= lo_c) & (R < hi_c)).astype(
                    float
                )
                - ((B >= lo_c) & (B < hi_c)).astype(float),
            }
            for stat in mb:
                diff = br[stat] - bb[stat]
                ci_lo, ci_hi = ci(diff)
                if stat in per_post:
                    p, test = (
                        rca.wilcoxon_p(per_post[stat]),
                        "Wilcoxon signed-rank (per post)",
                    )
                else:
                    p, test = em._boot_p_two_sided(diff), "paired bootstrap"
                rows.append(
                    {
                        "Model": model_label(folder, scope),
                        "Posts": scope,
                        "N": len(g),
                        "Statistic": stat,
                        "Baseline": float(mb[stat]),
                        "RAG": float(mr[stat]),
                        "Difference (RAG - baseline)": float(mr[stat] - mb[stat]),
                        "CI low": ci_lo,
                        "CI high": ci_hi,
                        "Test": test,
                        "p": p,
                    }
                )
    df = pd.DataFrame(rows)
    df["p (Holm, within model)"] = np.nan
    for m, g in df.groupby("Model"):
        # 'Mean prediction' duplicates 'Mean signed error' (target fixed per post): test it once.
        fam = g[g["Statistic"] != "Mean prediction"]
        df.loc[fam.index, "p (Holm, within model)"] = em._holm(fam["p"].tolist())
    df.loc[df["Statistic"] == "Mean prediction", "p"] = np.nan
    targets = {"Spread AfD - DIE LINKE": POSITIONS["AfD"] - POSITIONS["DIE LINKE"]}
    df["Target value"] = df["Statistic"].map(targets)
    return df


def _share_ci(x: np.ndarray, n_boot: int, seed: int):
    return rca.paired_boot(x.astype(float), n_boot, np.random.default_rng(seed))


def direction_movement(d: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            g = in_scope(d[d["model"] == folder], scope)
            for party in ["All posts"] + AUTHOR_GROUPS:
                s = g if party == "All posts" else g[g["party"] == party]
                move = (s["R"] - s["B"]).to_numpy(float)
                left, right = move <= -MOVE_THRESHOLD, move >= MOVE_THRESHOLD
                row = {
                    "Model": model_label(folder, scope),
                    "Posts": scope,
                    "Author party": GROUP_DISPLAY.get(party, party),
                    "N": len(s),
                    "Small N (< 30)": len(s) < SMALL_N,
                }
                for name, x in (
                    ("moved left", left),
                    ("moved right", right),
                    ("unchanged", ~left & ~right),
                ):
                    row[f"Share {name}"] = float(x.mean())
                    row[f"Share {name} CI low"], row[f"Share {name} CI high"] = (
                        _share_ci(x, n_boot, seed)
                    )
                row["Mean move (RAG - baseline)"] = float(move.mean())
                rows.append(row)
    return pd.DataFrame(rows)


def direction_by_party(d: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            g = in_scope(d[d["model"] == folder], scope)
            block = []
            for party in AUTHOR_GROUPS:
                s = g[g["party"] == party]
                dae = (s["ae_R"] - s["ae_B"]).to_numpy(float)
                dse = (s["se_R"] - s["se_B"]).to_numpy(float)
                lo1, hi1 = _share_ci(dae, n_boot, seed)
                lo2, hi2 = _share_ci(dse, n_boot, seed)
                block.append(
                    {
                        "Model": model_label(folder, scope),
                        "Posts": scope,
                        "Author party": GROUP_DISPLAY[party],
                        "N": len(s),
                        "of which CSU posts": int((s["input_party"] == "CSU").sum()),
                        "Small N (< 30)": len(s) < SMALL_N,
                        "Mean target": float(s["target"].mean()),
                        "MAE baseline": float(s["ae_B"].mean()),
                        "MAE RAG": float(s["ae_R"].mean()),
                        "dMAE": float(dae.mean()),
                        "dMAE CI low": lo1,
                        "dMAE CI high": hi1,
                        "dMAE Wilcoxon p": rca.wilcoxon_p(dae),
                        "Signed error baseline": float(s["se_B"].mean()),
                        "Signed error RAG": float(s["se_R"].mean()),
                        "d signed error": float(dse.mean()),
                        "d signed error CI low": lo2,
                        "d signed error CI high": hi2,
                        "d signed error Wilcoxon p": rca.wilcoxon_p(dse),
                        "Mean prediction baseline": float(s["B"].mean()),
                        "Mean prediction RAG": float(s["R"].mean()),
                    }
                )
            blk = pd.DataFrame(block)
            blk["dMAE p (Holm, across parties)"] = em._holm(
                blk["dMAE Wilcoxon p"].tolist()
            )
            blk["d signed error p (Holm, across parties)"] = em._holm(
                blk["d signed error Wilcoxon p"].tolist()
            )
            rows.append(blk)
    return pd.concat(rows, ignore_index=True)


def _sign(lo: float, hi: float) -> int:
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


COMPRESSION_RULE = (
    "Rule (d = RAG − baseline signed error, 95 % CI): 'compresses' if d > 0 for DIE LINKE and d < 0 for AfD "
    "(both CIs exclude 0: both ends pulled towards the centre); 'expands' if d < 0 for DIE LINKE and d > 0 for AfD; "
    "'shifts right' if both CIs lie above 0, or one lies above 0 and the other includes 0 with a positive point "
    "estimate; 'shifts left' mirrored; 'mixed' for everything else (both CIs include 0, or one end moves "
    "significantly and the other non-significantly in the opposite direction)."
)


def direction_compression(bp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scope), g in bp.groupby(["Model", "Posts"], sort=False):
        L = g[g["Author party"] == "DIE LINKE"].iloc[0]
        A = g[g["Author party"] == "AfD"].iloc[0]
        sl = _sign(L["d signed error CI low"], L["d signed error CI high"])
        sa = _sign(A["d signed error CI low"], A["d signed error CI high"])
        pl, pa = np.sign(L["d signed error"]), np.sign(A["d signed error"])
        if sl == 1 and sa == -1:
            cls = "compresses"
        elif sl == -1 and sa == 1:
            cls = "expands"
        elif (
            (sl == 1 and sa == 1)
            or (sl == 1 and sa == 0 and pa > 0)
            or (sa == 1 and sl == 0 and pl > 0)
        ):
            cls = "shifts right"
        elif (
            (sl == -1 and sa == -1)
            or (sl == -1 and sa == 0 and pa < 0)
            or (sa == -1 and sl == 0 and pl < 0)
        ):
            cls = "shifts left"
        else:
            cls = "mixed"
        detail = ""
        if cls == "mixed":
            parts = []
            if sl != 0:
                parts.append(
                    f"only DIE LINKE moves significantly ({'right, towards the centre' if sl > 0 else 'left, outwards'})"
                )
            if sa != 0:
                parts.append(
                    f"only AfD moves significantly ({'left, towards the centre' if sa < 0 else 'right, outwards'})"
                )
            detail = "; ".join(parts) or "neither end moves significantly"
        rows.append(
            {
                "Model": model,
                "Posts": scope,
                "DIE LINKE d signed error": L["d signed error"],
                "DIE LINKE CI low": L["d signed error CI low"],
                "DIE LINKE CI high": L["d signed error CI high"],
                "AfD d signed error": A["d signed error"],
                "AfD CI low": A["d signed error CI low"],
                "AfD CI high": A["d signed error CI high"],
                "DIE LINKE dMAE": L["dMAE"],
                "AfD dMAE": A["dMAE"],
                "Spread change": A["d signed error"] - L["d signed error"],
                "Classification": cls,
                "Detail": detail,
            }
        )
    return pd.DataFrame(rows)


def direction_lmm(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in (SCOPE_ALL, SCOPE_GE1):
        sub = d if scope == SCOPE_ALL else d[(d["model"] != LLAMA_3B) | (d["B"] >= 1)]
        long = pd.concat(
            [
                sub.assign(rag=0, abs_error=sub["ae_B"], signed_error=sub["se_B"]),
                sub.assign(rag=1, abs_error=sub["ae_R"], signed_error=sub["se_R"]),
            ],
            ignore_index=True,
        )[["model", "text_index", "party", "rag", "abs_error", "signed_error"]]
        long["party"] = long["party"].map(GROUP_DISPLAY)
        for outcome in ("abs_error", "signed_error"):
            formula = f"{outcome} ~ rag * {em._cat('party', 'SPD', long)} + C(model)"
            res, converged, err = em._fit_lmm(formula, long)
            if res is None:
                raise SystemExit(f"Mixed model failed ({outcome}, {scope}): {err}")
            ci_ = res.conf_int()
            terms = [t for t in res.params.index if t == "rag" or t.startswith("rag:")]
            block = []
            for t in terms:
                party = (
                    "SPD (reference)" if t == "rag" else t.split("[T.")[1].rstrip("]")
                )
                block.append(
                    {
                        "Outcome": outcome.replace("_", " "),
                        "Llama 3B posts": (
                            "all" if scope == SCOPE_ALL else "baseline score >= 1"
                        ),
                        "Term": (
                            "rag (effect for SPD)" if t == "rag" else f"rag x {party}"
                        ),
                        "Coefficient": float(res.params[t]),
                        "SE": float(res.bse[t]),
                        "CI low": float(ci_.loc[t, 0]),
                        "CI high": float(ci_.loc[t, 1]),
                        "p": float(res.pvalues[t]),
                        "N rows": int(res.nobs),
                        "N texts": int(long["text_index"].nunique()),
                        "Converged": converged,
                        "Formula": formula + " + (1 | text_index)",
                    }
                )
            blk = pd.DataFrame(block)
            inter = blk["Term"] != "rag (effect for SPD)"
            blk["p (Holm, interactions)"] = np.nan
            blk.loc[inter, "p (Holm, interactions)"] = em._holm(
                blk.loc[inter, "p"].tolist()
            )
            rows.append(blk)
    return pd.concat(rows, ignore_index=True)


def plot_direction_party(bp: pd.DataFrame):
    panels = list(bp[["Model", "Posts"]].drop_duplicates().itertuples(index=False))
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), sharey=True)
    axes = axes.ravel()
    labels = [GROUP_DISPLAY[p] for p in AUTHOR_GROUPS]
    x = np.arange(len(labels))
    for ax, (model, scope) in zip(axes, panels):
        g = (
            bp[(bp["Model"] == model) & (bp["Posts"] == scope)]
            .set_index("Author party")
            .loc[labels]
        )
        ax.scatter(
            x,
            g["Mean target"],
            marker="_",
            s=260,
            color=INK,
            linewidths=2,
            zorder=3,
            label="target",
        )
        ax.plot(
            x - 0.12,
            g["Mean prediction baseline"],
            "o",
            color=COL_BASE,
            ms=6,
            label="baseline",
            zorder=4,
        )
        ax.plot(
            x + 0.12,
            g["Mean prediction RAG"],
            "s",
            color=COL_RAG,
            ms=6,
            label="RAG (e5/simple)",
            zorder=4,
        )
        for xi, b, r in zip(x, g["Mean prediction baseline"], g["Mean prediction RAG"]):
            ax.plot(
                [xi - 0.12, xi + 0.12], [b, r], color="#c9c8c2", linewidth=1, zorder=2
            )
        ax.set_title(model, fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right", color=INK)
        ax.set_ylim(1, 7)
        ax.axhspan(*em.BLOC_THRESHOLDS, color="#f0efec", zorder=0)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7.5)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("mean score (1-7)", fontsize=8.5, color=INK_SECONDARY)
    handles, leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_fig(fig, "fig_direction_party")


def plot_direction_extremes(bp: pd.DataFrame):
    order = list(bp[["Model"]].drop_duplicates()["Model"])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    for party, col, marker, off in (
        ("DIE LINKE", COL_LINKE, "o", 0.15),
        ("AfD", COL_AFD, "s", -0.15),
    ):
        g = bp[bp["Author party"] == party].set_index("Model").loc[order]
        ax.errorbar(
            g["dMAE"],
            y + off,
            xerr=[g["dMAE"] - g["dMAE CI low"], g["dMAE CI high"] - g["dMAE"]],
            fmt=marker,
            color=col,
            ms=6,
            elinewidth=1.4,
            capsize=0,
            label=party,
        )
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5, color=INK)
    ax.set_xlabel(
        "ΔMAE (RAG − baseline), 95 % CI; < 0 = RAG better",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig_direction_extremes")


def preliminary_check(d: pd.DataFrame) -> dict:
    """Llama-70B AfD signed error on the paired e5/simple posts vs. the preliminary
    unpaired figures in results/evaluation/2026-11-12_full/bias_by_party_label_ideology.csv.
    """
    g = d[(d["model"] == "llama_70b") & (d["party"] == "AfD")]
    out = {
        "n": len(g),
        "base_2dp": g["se_B"].mean(),
        "rag_2dp": g["se_R"].mean(),
        "base_1dp": (g["B"] - g["target_log"]).mean(),
        "rag_1dp": (g["R"] - g["target_log"]).mean(),
    }
    prelim = (
        ROOT / "results/evaluation/2026-11-12_full/bias_by_party_label_ideology.csv"
    )
    if prelim.exists():
        p = pd.read_csv(prelim)
        p = p[
            (p["Model"] == "Meta-Llama-3.1-70B-Instruct-FP8") & (p["Party"] == "AfD")
        ].set_index("Condition")
        out["prelim_base"] = (
            float(p.loc["no_rag", "Mean_signed_error"])
            if "no_rag" in p.index
            else np.nan
        )
        out["prelim_e5"] = (
            float(p.loc["e5/simple", "Mean_signed_error"])
            if "e5/simple" in p.index
            else np.nan
        )
        rag_conds = [c for c in p.index if c != "no_rag"]
        out["prelim_rag_range"] = (
            float(p.loc[rag_conds, "Mean_signed_error"].min()),
            float(p.loc[rag_conds, "Mean_signed_error"].max()),
            len(rag_conds),
        )
    return out


def write_direction_report(
    d, overall, movement, bp, comp, lmm, prelim, n_boot, seed, readings
):
    L = ["# Why retrieval helps or hurts: direction of the RAG shift\n", "## Data\n"]
    n = d.groupby("model").size()
    L.append(
        f"Paired posts of e5/simple (k = 5) and the no-RAG baseline per model (inner join on text_index, both scores "
        f"parseable): {', '.join(f'{DISPLAY[m]} {n[m]}' for m, *_ in MODELS)}. Llama 3B is reported on all paired "
        f"posts and on the {int(((d['model'] == LLAMA_3B) & (d['B'] >= 1)).sum())} posts with a baseline score ≥ 1. "
        f"Target: CHES 2024 lrgen of the author's party, two decimals (DIE LINKE 1.85, Grüne 2.83, SPD 3.08, FDP "
        f"4.63, CDU 4.95, CSU 5.54, AfD 6.56); target spread AfD − DIE LINKE = "
        f"{POSITIONS['AfD'] - POSITIONS['DIE LINKE']:.2f}. CSU posts (about 7 per model) are merged into CDU/CSU, as "
        f"in the party labels of the logs, and are not tested on their own; each post keeps its own target "
        f"(CSU 5.54, CDU and CDU/CSU 4.95). Centre band [3.4, 5.2). Moved = |RAG − baseline| ≥ {MOVE_THRESHOLD}. "
        f"Paired bootstrap {n_boot} resamples of posts, seed {seed}; Wilcoxon signed-rank tests on per-post "
        f"differences, Holm within each model. No other exclusions.\n"
    )
    L.append("## 1. Overall shift per model\n")
    piv = overall[
        overall["Statistic"].isin(
            [
                "Mean signed error",
                "SD of predictions",
                "Calibration slope",
                "Spread AfD - DIE LINKE",
                "Share in centre band [3.4, 5.2)",
            ]
        )
    ]
    tab = piv.assign(
        cell=[
            f"{b:.2f} → {r:.2f} ({dd:+.2f} [{lo:+.2f}, {hi:+.2f}])"
            for b, r, dd, lo, hi in zip(
                piv["Baseline"],
                piv["RAG"],
                piv["Difference (RAG - baseline)"],
                piv["CI low"],
                piv["CI high"],
            )
        ]
    )
    stat_order = [
        "Mean signed error",
        "SD of predictions",
        "Calibration slope",
        "Spread AfD - DIE LINKE",
        "Share in centre band [3.4, 5.2)",
    ]
    tab = tab.pivot(index="Model", columns="Statistic", values="cell").loc[
        overall["Model"].drop_duplicates(), stat_order
    ]
    L.append("Baseline → RAG (difference [95 % CI]):\n")
    L.append(em._md_table(tab.reset_index()) + "\n")
    L.append(readings["overall"] + "\n")
    L.append("## 2. Movement per post\n")
    mv = movement[movement["Author party"].isin(["All posts", "DIE LINKE", "AfD"])]
    mtab = mv.assign(
        cell=[
            f"{l * 100:.0f} / {u * 100:.0f} / {r * 100:.0f}"
            for l, u, r in zip(
                mv["Share moved left"], mv["Share unchanged"], mv["Share moved right"]
            )
        ]
    )
    mtab = mtab.pivot(index="Model", columns="Author party", values="cell").loc[
        movement["Model"].drop_duplicates(), ["All posts", "DIE LINKE", "AfD"]
    ]
    L.append("Share of posts moved left / unchanged / right (%):\n")
    L.append(em._md_table(mtab.reset_index()) + "\n")
    L.append(readings["movement"] + "\n")
    L.append("## 3. By author party\n")
    sel = bp[bp["Author party"].isin(["DIE LINKE", "AfD"])]
    t3 = pd.DataFrame(
        {
            "Model": sel["Model"],
            "Party": sel["Author party"],
            "N": sel["N"],
            "MAE base → RAG": [
                f"{a:.2f} → {b:.2f}"
                for a, b in zip(sel["MAE baseline"], sel["MAE RAG"])
            ],
            "dMAE [95 % CI]": [
                f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                for v, lo, hi in zip(
                    sel["dMAE"], sel["dMAE CI low"], sel["dMAE CI high"]
                )
            ],
            "p (Holm)": sel["dMAE p (Holm, across parties)"],
            "signed error base → RAG": [
                f"{a:+.2f} → {b:+.2f}"
                for a, b in zip(sel["Signed error baseline"], sel["Signed error RAG"])
            ],
        }
    )
    L.append(md_table(t3) + "\n")
    L.append(readings["by_party"] + "\n")
    L.append(
        f"Preliminary figure check (Llama 70B, AfD posts): the unpaired, all-conditions file "
        f"`results/evaluation/2026-11-12_full/bias_by_party_label_ideology.csv` gives a mean signed error of "
        f"{prelim.get('prelim_base', np.nan):+.2f} without RAG and {prelim.get('prelim_e5', np.nan):+.2f} for "
        f"e5/simple (range over its {prelim['prelim_rag_range'][2]} RAG conditions "
        f"{prelim['prelim_rag_range'][0]:+.2f} to {prelim['prelim_rag_range'][1]:+.2f}), against the one-decimal "
        f"target 6.6. On the {prelim['n']} paired posts here the same target gives {prelim['base_1dp']:+.2f} → "
        f"{prelim['rag_1dp']:+.2f}, i.e. the same numbers. With the two-decimal target (6.56) used in this chapter "
        f"the values are {prelim['base_2dp']:+.2f} → {prelim['rag_2dp']:+.2f}: e5/simple shows the same large "
        f"leftward shift for AfD posts.\n"
    )
    L.append("## 4. Compression or shift?\n")
    L.append(COMPRESSION_RULE + "\n")
    ct = pd.DataFrame(
        {
            "Model": comp["Model"],
            "DIE LINKE d [95 % CI]": [
                f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                for v, lo, hi in zip(
                    comp["DIE LINKE d signed error"],
                    comp["DIE LINKE CI low"],
                    comp["DIE LINKE CI high"],
                )
            ],
            "AfD d [95 % CI]": [
                f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                for v, lo, hi in zip(
                    comp["AfD d signed error"], comp["AfD CI low"], comp["AfD CI high"]
                )
            ],
            "Classification": comp["Classification"],
            "Detail": comp["Detail"],
        }
    )
    L.append(em._md_table(ct) + "\n")
    L.append(readings["compression"] + "\n")
    L.append("## 5. Pooled mixed models\n")
    lt = lmm[lmm["Llama 3B posts"] == "all"].copy()
    lt = pd.DataFrame(
        {
            "Outcome": lt["Outcome"],
            "Term": lt["Term"],
            "Coefficient [95 % CI]": [
                f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                for v, lo, hi in zip(lt["Coefficient"], lt["CI low"], lt["CI high"])
            ],
            "p": lt["p"],
            "p (Holm, interactions)": lt["p (Holm, interactions)"],
        }
    )
    L.append(
        f"`{lmm['Formula'].iloc[0]}` (and the same for signed error), ML fit, random intercept per text, "
        f"{int(lmm['N rows'].iloc[0])} rows / {int(lmm['N texts'].iloc[0])} texts; all fits converged: "
        f"{bool(lmm['Converged'].all())}. Llama 3B on all paired posts (the version with baseline ≥ 1 is in "
        f"`direction_lmm.csv`).\n"
    )
    L.append(md_table(lt) + "\n")
    L.append(readings["lmm"] + "\n")
    L.append("![fig_direction_party.png](fig_direction_party.png)\n")
    L.append(
        "Figure: mean prediction per author party, baseline (circle) and RAG (square); black bar = mean "
        "target; grey band = centre band [3.4, 5.2).\n"
    )
    L.append("![fig_direction_extremes.png](fig_direction_extremes.png)\n")
    L.append(
        "Figure: ΔMAE (RAG − baseline) for DIE LINKE and AfD posts per model with 95 % CI.\n"
    )
    (OUT_DIR / "04_why_rag_direction.md").write_text("\n".join(L), encoding="utf-8")


def direction_readings(overall, movement, bp, comp, lmm) -> Dict[str, str]:
    """Reading paragraphs; model lists and numbers come from the tables (main scope:
    Llama 3B on all posts; the baseline >= 1 version is mentioned where it differs)."""
    main = [model_label(m) for m, *_ in MODELS]
    ge1 = model_label(LLAMA_3B, SCOPE_GE1)

    def by_ci(stat, sign, rows=main):
        g = overall[(overall["Statistic"] == stat) & overall["Model"].isin(rows)]
        return g[(g["CI low"] > 0) if sign > 0 else (g["CI high"] < 0)][
            "Model"
        ].tolist()

    def lst(x):
        return ", ".join(x) if x else "none"

    right, left = by_ci("Mean signed error", 1), by_ci("Mean signed error", -1)
    sd_up, sd_dn = by_ci("SD of predictions", 1), by_ci("SD of predictions", -1)
    sp_up, sp_dn = by_ci("Spread AfD - DIE LINKE", 1), by_ci(
        "Spread AfD - DIE LINKE", -1
    )
    c_up, c_dn = by_ci("Share in centre band [3.4, 5.2)", 1), by_ci(
        "Share in centre band [3.4, 5.2)", -1
    )
    same_ge1 = all(
        np.sign(_get(overall, Model=ge1, Statistic=s)["Difference (RAG - baseline)"])
        == np.sign(
            _get(overall, Model=model_label(LLAMA_3B), Statistic=s)[
                "Difference (RAG - baseline)"
            ]
        )
        for s in (
            "Mean signed error",
            "SD of predictions",
            "Spread AfD - DIE LINKE",
            "Share in centre band [3.4, 5.2)",
        )
    )
    r = {}
    r["overall"] = (
        f"Reading: there is no common direction. With RAG, {len(right)} of {len(main)} models predict further right on "
        f"average ({lst(right)}) and {len(left)} further left ({lst(left)}). More informative is the spread: the "
        f"predictions become more dispersed (SD up) in {lst(sd_up)}, and the AfD − DIE LINKE gap widens in "
        f"{lst(sp_up)}, towards the target gap of {POSITIONS['AfD'] - POSITIONS['DIE LINKE']:.2f}. The SD shrinks in "
        f"{lst(sd_dn)}, and the gap narrows in {lst(sp_dn)}. More predictions fall in the centre band in {lst(c_up)} and "
        f"fewer in {lst(c_dn)}. "
        + (
            "The Llama 3B results without the sub-1 baseline scores point the same way."
            if same_ge1
            else "For Llama 3B, excluding the sub-1 baseline scores changes the direction of at least one statistic "
            "(see table)."
        )
    )
    mv = movement[
        (movement["Author party"] == "All posts") & movement["Model"].isin(main)
    ].set_index("Model")
    unch = mv["Share unchanged"]
    dom = [
        f"{m} ({'left' if mv.loc[m, 'Share moved left'] > mv.loc[m, 'Share moved right'] else 'right'})"
        for m in main
        if abs(mv.loc[m, "Share moved left"] - mv.loc[m, "Share moved right"]) >= 0.15
    ]
    r["movement"] = (
        f"Reading: RAG moves many single predictions by half a point or more: the unchanged share ranges from "
        f"{unch.min() * 100:.0f} % ({unch.idxmin()}) to {unch.max() * 100:.0f} % ({unch.idxmax()}). In most models, "
        f"posts move in both directions. Where one direction dominates by at least 15 percentage points, it is: "
        f"{lst(dom)}. The per-party movement is in `direction_movement.csv`."
    )

    def helped(party, sign):
        g = bp[(bp["Author party"] == party) & bp["Model"].isin(main)]
        return g[(g["dMAE CI high"] < 0) if sign < 0 else (g["dMAE CI low"] > 0)][
            "Model"
        ].tolist()

    l_better, l_worse = helped("DIE LINKE", -1), helped("DIE LINKE", 1)
    a_better, a_worse = helped("AfD", -1), helped("AfD", 1)
    ext = bp[bp["Author party"].isin(["DIE LINKE", "AfD"]) & bp["Model"].isin(main)]
    worst = ext.loc[ext["dMAE"].idxmax()]
    small = bp[bp["Small N (< 30)"]]
    r["by_party"] = (
        f"Reading: for DIE LINKE posts RAG lowers the error in {lst(l_better)} and raises it in {lst(l_worse)}; for AfD "
        f"posts it lowers the error in {lst(a_better)} and raises it in {lst(a_worse)} (95 % CIs of ΔMAE). The extremes "
        f"therefore do not move together: a model can improve one end and worsen the other. The largest single "
        f"deterioration is {worst['Model']} for {worst['Author party']} posts (MAE {worst['MAE baseline']:.2f} → "
        f"{worst['MAE RAG']:.2f}). CIs and Holm-corrected p values are computed per model; all parties incl. Grüne, "
        f"SPD, FDP and CDU/CSU are in `direction_by_party.csv` "
        + (
            "(no cell has fewer than 30 posts after merging CSU into CDU/CSU)."
            if small.empty
            else f"(cells with fewer than {SMALL_N} posts are flagged: "
            f"{lst([f'{m} {p}' for m, p in zip(small['Model'], small['Author party'])])})."
        )
    )
    cnt = comp[comp["Model"].isin(main)]["Classification"].value_counts()
    cls_models = {
        c: comp[(comp["Classification"] == c) & comp["Model"].isin(main)][
            "Model"
        ].tolist()
        for c in ("compresses", "expands", "shifts left", "shifts right", "mixed")
    }
    lb = bp[(bp["Author party"] == "DIE LINKE") & bp["Model"].isin(main)][
        "Signed error baseline"
    ]
    ab = bp[(bp["Author party"] == "AfD") & bp["Model"].isin(main)][
        "Signed error baseline"
    ]
    base_compressed = bool((lb > 0).all() and (ab < 0).all())
    r["compression"] = (
        (
            "Reading: no model compresses both ends towards the centre. "
            if not cls_models["compresses"]
            else "Reading: "
        )
        + "; ".join(f"{c}: {lst(m)}" for c, m in cls_models.items() if m)
        + ". "
        + "'Expands' means RAG moves DIE LINKE posts further left and AfD posts further right"
        + (
            ", i.e. it reduces the compression the baselines already have (every baseline places DIE LINKE posts too "
            "far right and AfD posts too far left)."
            if base_compressed
            else "."
        )
        + (
            f" The {len(cls_models['mixed'])} 'mixed' case(s) are described in the detail column."
            if cls_models["mixed"]
            else ""
        )
        + " Classification counts: "
        + ", ".join(f"{k} {v}" for k, v in cnt.items())
        + "."
    )
    la = lmm[lmm["Llama 3B posts"] == "all"]
    sig = la[(la["p (Holm, interactions)"] < 0.05)]
    rag_se = _get(la, Outcome="signed error", Term="rag (effect for SPD)")
    rag_ae = _get(la, Outcome="abs error", Term="rag (effect for SPD)")
    ae_txt = (
        "leaves their absolute error unchanged (95 % CI includes 0)"
        if rag_ae["CI low"] < 0 < rag_ae["CI high"]
        else f"changes their absolute error by {rag_ae['Coefficient']:+.2f}"
    )
    r["lmm"] = (
        f"Reading: pooled over models, RAG moves SPD posts {rag_se['Coefficient']:+.2f} points (signed error) and "
        f"{ae_txt}. After Holm correction the RAG effect differs from SPD for "
        + (
            lst(
                [
                    f"{t.replace('rag x ', '')} ({o})"
                    for t, o in zip(sig["Term"], sig["Outcome"])
                ]
            )
            or "no party"
        )
        + ". The pooled model assumes one RAG effect for all models. Sections 1-4 show that the models move in "
        f"opposite directions, so these pooled interactions average over opposite effects and understate the "
        f"party-specific changes within single models."
    )
    return r


def cmd_direction(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading paired logs ...")
    d = direction_data()
    overall = direction_overall(d, args.n_boot, args.seed)
    movement = direction_movement(d, args.n_boot, args.seed)
    bp = direction_by_party(d, args.n_boot, args.seed)
    comp = direction_compression(bp)
    print("Mixed models ...")
    lmm = direction_lmm(d)
    for df, name in (
        (overall, "direction_overall.csv"),
        (movement, "direction_movement.csv"),
        (bp, "direction_by_party.csv"),
        (comp, "direction_compression.csv"),
        (lmm, "direction_lmm.csv"),
    ):
        print("  wrote", write_csv(df, name))
    plot_direction_party(bp)
    plot_direction_extremes(bp)
    prelim = preliminary_check(d)
    write_direction_report(
        d,
        overall,
        movement,
        bp,
        comp,
        lmm,
        prelim,
        args.n_boot,
        args.seed,
        direction_readings(overall, movement, bp, comp, lmm),
    )
    print("  wrote", OUT_DIR / "04_why_rag_direction.md", "and the two figures")


# ===========================================================================
# extremes: why DIE LINKE and AfD posts behave differently
# ===========================================================================

EXTREME_GROUPS = {"DIE LINKE": "DIE LINKE", "AfD": "AfD"}
SUBSETS = ("DIE LINKE", "AfD", "other parties")
CASE_SHIFT = 1.5  # |RAG - baseline| >= 1.5 for the case listings
N_EXAMPLES = 5
N_CONTRAST = 3
N_LINKE_EXAMPLES = 3
CONTRAST_MODELS = ("qwen_7b", "llama_8b")
SMALL_TEXTS = 10  # judge-label cells with fewer sampled texts are flagged


def _subset(party: str) -> str:
    return party if party in EXTREME_GROUPS else "other parties"


def _judge_module():
    import logging

    logging.getLogger("streamlit").setLevel(logging.ERROR)
    import run_llm_judge as rj  # deterministic citation rules of the judge

    return rj


def text_evidence(ret: pd.DataFrame) -> pd.DataFrame:
    """Per text: E = mean position of the retrieved chunks with a position."""
    g = ret.groupby("text_index")
    return pd.DataFrame(
        {"E": g["chunk_pos"].mean(), "n_positioned": g["chunk_pos"].count()}
    ).reset_index()


def extremes_data():
    d = direction_data()
    ret = load_retrieval()
    ev = text_evidence(ret)
    d = d.merge(ev, on="text_index", how="left")
    d["subset"] = d["party"].map(_subset)
    return d, ret


def _fit_E(data: pd.DataFrame, formula: str = "R ~ B + E"):
    fit, used = aci._ols_cluster(formula, data)
    return fit, used


def extremes_evidence_use(d: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(label, scope, subset, data, fe=False):
        formula = "R ~ B + E" + (" + C(model)" if fe else "")
        fit, used = _fit_E(data, formula)
        ci_ = fit.conf_int()
        rows.append(
            {
                "Model": label,
                "Posts": scope,
                "Subset": subset,
                "N posts": int(fit.nobs),
                "N texts": int(used["text_index"].nunique()),
                "SD of E": float(used["E"].std()),
                "E coefficient": float(fit.params["E"]),
                "CI low": float(ci_.loc["E", 0]),
                "CI high": float(ci_.loc["E", 1]),
                "p": float(fit.pvalues["E"]),
                "B coefficient": float(fit.params["B"]),
                "R2": float(fit.rsquared),
            }
        )

    inter_rows = []

    def add_inter(label, scope, data, fe=False):
        data = data.copy()
        formula = "R ~ (B + E) * C(subset, Treatment('other parties'))" + (
            " + C(model)" if fe else ""
        )
        fit, _ = _fit_E(data, formula)
        ci_ = fit.conf_int()
        block = []
        for grp in ("DIE LINKE", "AfD"):
            t = f"E:C(subset, Treatment('other parties'))[T.{grp}]"
            block.append(
                {
                    "Model": label,
                    "Posts": scope,
                    "Contrast": f"E coefficient, {grp} − other parties",
                    "Difference": float(fit.params[t]),
                    "CI low": float(ci_.loc[t, 0]),
                    "CI high": float(ci_.loc[t, 1]),
                    "p": float(fit.pvalues[t]),
                }
            )
        for b, p in zip(block, em._holm([x["p"] for x in block])):
            b["p (Holm, 2 contrasts)"] = p
        inter_rows.extend(block)

    for folder, *_ in MODELS:
        for scope in scopes(folder):
            g = in_scope(d[d["model"] == folder], scope)
            label = model_label(folder, scope)
            add(label, scope, "all posts", g)
            for s in SUBSETS:
                add(label, scope, s, g[g["subset"] == s])
            add_inter(label, scope, g)
    pooled = d  # Llama 3B on all paired posts
    add("All models (model fixed effects)", SCOPE_ALL, "all posts", pooled, fe=True)
    for s in SUBSETS:
        add(
            "All models (model fixed effects)",
            SCOPE_ALL,
            s,
            pooled[pooled["subset"] == s],
            fe=True,
        )
    add_inter("All models (model fixed effects)", SCOPE_ALL, pooled, fe=True)
    use = pd.DataFrame(rows)
    inter = pd.DataFrame(inter_rows)
    return use, inter


def extremes_evidence_gap(d: pd.DataFrame, n_boot: int, seed: int):
    """(a) per author party, model-independent: where E lies relative to the target;
    (b) per model x party: how far the RAG prediction moves towards E."""
    texts = d.drop_duplicates("text_index")
    rows = []
    for party in ["All posts"] + AUTHOR_GROUPS:
        t = texts if party == "All posts" else texts[texts["party"] == party]
        t = t.dropna(subset=["E"])
        gap = (t["E"] - t["target"]).to_numpy()
        lo, hi = _share_ci(gap, n_boot, seed)
        same_bloc = np.mean(
            [em._bloc(e) == em._bloc(tg) for e, tg in zip(t["E"], t["target"])]
        )
        rows.append(
            {
                "Author party": GROUP_DISPLAY.get(party, party),
                "Posts": len(t),
                "Mean target": float(t["target"].mean()),
                "Mean E": float(t["E"].mean()),
                "SD of E": float(t["E"].std()),
                "E - target": float(gap.mean()),
                "CI low": lo,
                "CI high": hi,
                "MAE of E as prediction": float(np.abs(gap).mean()),
                "Share of posts with E in the target's bloc": float(same_bloc),
            }
        )
    gap_tab = pd.DataFrame(rows)
    rows = []
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            g = in_scope(d[d["model"] == folder], scope).dropna(subset=["E"])
            fit, _ = _fit_E(g)
            g = g.assign(R_hat=fit.predict(g))
            for party in AUTHOR_GROUPS:
                s = g[g["party"] == party]
                toward = (s["E"] - s["B"]).mean()
                moved = (s["R"] - s["B"]).mean()
                rows.append(
                    {
                        "Model": model_label(folder, scope),
                        "Posts": scope,
                        "Author party": GROUP_DISPLAY[party],
                        "N": len(s),
                        "Mean target": float(s["target"].mean()),
                        "Mean baseline": float(s["B"].mean()),
                        "Mean E": float(s["E"].mean()),
                        "Mean RAG": float(s["R"].mean()),
                        "E - baseline": float(toward),
                        "RAG - baseline": float(moved),
                        "Mean RAG predicted by the all-posts fit": float(
                            s["R_hat"].mean()
                        ),
                        "RAG - predicted": float(s["R"].mean() - s["R_hat"].mean()),
                        "MAE baseline": float(s["ae_B"].mean()),
                        "MAE RAG": float(s["ae_R"].mean()),
                        "MAE if prediction = E": float(
                            (s["E"] - s["target"]).abs().mean()
                        ),
                    }
                )
    return gap_tab, pd.DataFrame(rows)


def extremes_judge_labels(d: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    ch = load_final_annotations()
    ch["chunk_group"] = ch["chunk_party_raw"].map(
        lambda r: party_group(_raw_to_canon(r, "annotated chunk")) or NO_POSITION
    )
    ch["chunk_pos"] = ch["chunk_group"].map(
        lambda g: group_position(g) if g != NO_POSITION else np.nan
    )
    meta = d[["model", "text_index", "party", "target", "B", "quadrant_sampling"]]
    ch = ch.merge(meta, on=["model", "text_index"], how="left")
    if ch["party"].isna().any():
        raise SystemExit("Annotated texts without a paired post")
    ch["author_bloc"] = ch["target"].map(em._bloc)
    ch["chunk_bloc"] = ch["chunk_pos"].map(
        lambda p: em._bloc(p) if not pd.isna(p) else -1
    )
    ch["opp"] = ch["author_bloc"].map({0: 2, 2: 0}).fillna(-9).astype(int)
    ch["cited"] = ch["A_caus"] == 1
    ch["cited_own"] = ch["cited"] & (ch["chunk_group"] == ch["party"])
    ch["cited_opp"] = ch["cited"] & (ch["chunk_bloc"] == ch["opp"])
    t = ch.groupby(
        ["model", "text_index", "quadrant", "party", "B", "opp"], as_index=False
    ).agg(
        chunks=("chunk_index", "size"),
        sum_R_top=("R_top", "sum"),
        sum_R_ideo=("R_ideo", "sum"),
        cited=("cited", "sum"),
        cited_own=("cited_own", "sum"),
        cited_opp=("cited_opp", "sum"),
    )
    t["subset"] = t["party"].map(_subset)
    rows = []
    pooled = []
    for folder, *_ in MODELS:
        for scope in scopes(folder):
            tm = in_scope(t[t["model"] == folder], scope).reset_index(drop=True)
            wq = sampling_weights(d.rename(columns={}), tm, scope, folder)
            tm["w"] = tm["quadrant"].map(wq)
            if scope == SCOPE_ALL:
                pooled.append(tm)
            for s in SUBSETS:
                rows.append(
                    _judge_row(
                        tm[tm["subset"] == s].reset_index(drop=True),
                        model_label(folder, scope),
                        scope,
                        s,
                        n_boot,
                        seed,
                        strat=True,
                    )
                )
    allp = pd.concat(pooled, ignore_index=True)
    for s in SUBSETS:
        rows.append(
            _judge_row(
                allp[allp["subset"] == s].reset_index(drop=True),
                "All models (Llama 3B: all)",
                SCOPE_ALL,
                s,
                n_boot,
                seed,
                strat=False,
            )
        )
    return pd.DataFrame(rows)


def _judge_row(
    t: pd.DataFrame,
    label: str,
    scope: str,
    subset: str,
    n_boot: int,
    seed: int,
    strat: bool,
) -> dict:
    row = {
        "Model": label,
        "Posts": scope,
        "Subset": subset,
        "Sampled texts": len(t),
        "Small N (< 10 texts)": len(t) < SMALL_TEXTS,
    }
    if len(t) == 0:
        return row
    m = multiplicities(
        t["text_index"].to_numpy() if strat else t["text_index"].to_numpy(),
        t["quadrant"].to_numpy() if strat else None,
        n_boot,
        np.random.default_rng(seed),
    )
    w = t["w"].to_numpy(float)
    has_opp = (t["opp"] >= 0).to_numpy()
    stats = (
        ("Mean R_top", t["sum_R_top"], t["chunks"], None),
        ("Mean R_ideo", t["sum_R_ideo"], t["chunks"], None),
        ("Share cited (A_caus = 1)", t["cited"], t["chunks"], None),
        ("Cited chunks per text", t["cited"], np.ones(len(t)), None),
        (
            "Cited chunks from the author's party per text",
            t["cited_own"],
            np.ones(len(t)),
            None,
        ),
        (
            "Cited chunks from the opposite bloc per text",
            t["cited_opp"],
            np.ones(len(t)),
            has_opp,
        ),
    )
    for name, num, den, mask in stats:
        num, den = np.asarray(num, float), np.asarray(den, float)
        ww = w if mask is None else np.where(mask, w, 0.0)
        if ww.sum() == 0:
            row[name] = np.nan
            continue
        row[name] = float((ww * num).sum() / (ww * den).sum())
        lo, hi = ci(boot_ratio(num, den, ww, m))
        row[f"{name} CI low"], row[f"{name} CI high"] = lo, hi
    row["Texts with an opposite bloc"] = int(has_opp.sum())
    return row


def _sentences(text: str) -> List[str]:
    import re

    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def decisive_sentence(justification: str, cited: set) -> str:
    """Heuristic: the sentence that cites retrieved chunks and names parties most;
    ties -> the later sentence (justifications usually conclude at the end)."""
    import re

    rj = _judge_module()
    best, best_score = "", -1
    for s in _sentences(justification):
        score = (
            2 * bool(rj.cited_chunk_numbers(s))
            + len(_mentions(s))
            + bool(
                re.search(r"\b(left|right|centr|far-right|extreme|leaning)", s, re.I)
            )
        )
        if score >= best_score:
            best, best_score = s, score
    return best


def _log_records() -> Dict[str, Tuple[dict, dict]]:
    out = {}
    for folder, family, size, _ in MODELS:
        diag = json.loads(
            (ROOT / DIAGNOSTICS.format(family=family, size=size)).read_text(
                encoding="utf-8"
            )
        )
        base = strata._merge_logs([ROOT / p for p in diag["baseline_paths"]]).records
        rag = strata._merge_logs([ROOT / p for p in diag["rag_paths"]]).records
        out[folder] = (
            {str(k): v for k, v in base.items()},
            {str(k): v for k, v in rag.items()},
        )
    return out


def case_rows(
    model: str, posts: pd.DataFrame, logs, ann: pd.DataFrame, author_group: str
) -> pd.DataFrame:
    """One row per post with retrieved / cited chunk parties, 'about the author party'
    chunks and judge labels where the text was annotated for this model."""
    rj = _judge_module()
    base, rag = logs[model]
    rows = []
    for _, p in posts.iterrows():
        rec = rag[p["text_index"]]
        just = (rec.get("output") or {}).get("justification") or ""
        chunks = (rec.get("inputs") or {}).get("retrieved_chunks") or []
        numbers = rj.cited_chunk_numbers(just)
        a = ann[
            (ann["model"] == model) & (ann["text_index"] == p["text_index"])
        ].set_index("chunk_index")
        parts, cited_parties, about, about_cited, labels = [], [], 0, 0, []
        for i, c in enumerate(chunks):
            g = party_group(_raw_to_canon(c.get("party"), "case chunk")) or NO_POSITION
            info = rj.citation_info(just, i + 1, c.get("speaker") or "")
            if len(a):
                cited = bool(a.loc[i, "A_caus"] == 1)
                labels.append(
                    f"[{i + 1}] R_top {a.loc[i, 'R_top']}, R_ideo {a.loc[i, 'R_ideo']}"
                )
            else:
                cited = info["cited_by_number"] or info["speaker_named"]
            name = GROUP_DISPLAY.get(g, g)
            parts.append(f"[{i + 1}] {name}{'*' if cited else ''}")
            is_about = g != author_group and author_group in _mentions(
                c.get("text") or ""
            )
            about += is_about
            if cited:
                cited_parties.append(name)
                about_cited += is_about
        rows.append(
            {
                "Model": DISPLAY[model],
                "text_index": p["text_index"],
                "Author party": GROUP_DISPLAY[author_group],
                "Target": p["target"],
                "Baseline": p["B"],
                "RAG": p["R"],
                "RAG - baseline": p["R"] - p["B"],
                "E": p["E"],
                "Chunks (* = cited)": ", ".join(parts),
                "Retrieved from author party": sum(
                    GROUP_DISPLAY.get(x, x) == GROUP_DISPLAY[author_group]
                    for x in [pp.split("] ")[1].rstrip("*") for pp in parts]
                ),
                "Cited chunks": len(cited_parties),
                "Cited parties": ", ".join(cited_parties),
                "Cited from author party": sum(
                    x == GROUP_DISPLAY[author_group] for x in cited_parties
                ),
                "Retrieved other-party chunks naming the author party": about,
                "Cited other-party chunks naming the author party": about_cited,
                "Citation source": (
                    "final A_caus labels"
                    if len(a)
                    else (
                        "number / speaker citations"
                        if numbers
                        else "speaker names only (no numeric citation)"
                    )
                ),
                "Judge labels": "; ".join(labels),
                "Post": (rec.get("inputs") or {}).get("text") or "",
                "RAG justification": just,
                "Baseline justification": (
                    (base[p["text_index"]].get("output") or {}).get("justification")
                    or ""
                ),
                "Decisive sentence": decisive_sentence(just, numbers),
                "Score sentence": score_sentence(just, p["R"]),
                "Justification describes the post as": describes(just),
                "Score contradicts description": (
                    describes(just) == "right" and p["R"] < em.BLOC_THRESHOLDS[0]
                )
                or (describes(just) == "left" and p["R"] >= em.BLOC_THRESHOLDS[1]),
                "Likens post to author party": likens(just, author_group),
                "Likened but scored other side": likens(just, author_group)
                and other_side(p["R"], author_group),
            }
        )
    return pd.DataFrame(rows)


def typical(cases: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n posts closest to the case-set medians of the shift and of the share of
    cited chunks that are other-party speech about the author's party."""
    c = cases.copy()
    c["about_share"] = c["Cited other-party chunks naming the author party"] / c[
        "Cited chunks"
    ].replace(0, np.nan)
    z = 0.0
    for col in ("RAG - baseline", "about_share"):
        v = c[col]
        sd = v.std() if v.std() > 0 else 1.0
        z = z + ((v - v.median()).abs() / sd).fillna(10)
    return c.assign(typicality=z).sort_values(["typicality", "text_index"]).head(n)


# Short English glosses of the German posts used as examples (written by hand for the
# examples the selection rule picks; missing glosses are marked in the report).
CASE_GLOSSES: Dict[Tuple[str, str], str] = {
    (
        "Llama 70B",
        "102",
    ): "Pandering with fatal consequences: SMEs have suffered for years because big corporations "
    "pandered to the green zeitgeist; now all firms are hit hard (Bosch cuts 13,000 jobs, chemical "
    "and steel industry …).",
    (
        "Llama 70B",
        "108",
    ): "The coalition committee of the black-red government under Chancellor Merz (CDU) is a mere show "
    "without substance: an 'autumn of reforms' with nothing but empty promises while Germany sinks "
    "into recession …",
    (
        "Llama 70B",
        "133",
    ): "In Brazil elections cannot be verified because they are purely electronic; Bolsonaro demands "
    "paper ballots as in Germany, and now a court investigates him: merely demanding verifiable "
    "elections could be punishable.",
    (
        "Llama 70B",
        "136",
    ): "Hairdressers may reopen because the state premiers suffer under their own mops; infection risk "
    "is not the point. Museums and retailers are just unlucky that the 'rulers of the new era' do "
    "not miss them. #lockdown",
    (
        "Llama 70B",
        "153",
    ): "Von der Leyen considers proceedings against Germany over the Constitutional Court's ECB ruling, "
    "i.e. the EU wants the government to disempower the court: an invitation to break the "
    "constitution. Protect the Basic Law from Brussels!",
    (
        "Qwen 7B",
        "176",
    ): "Heading for bankruptcy with flags flying? The corona shutdown keeps about 2.1 million citizens "
    "awake at night; their financial losses threaten their livelihoods. #AfD",
    (
        "Qwen 7B",
        "192",
    ): "SPD, Greens and Linke stir up hatred and distrust against the police and create the climate for "
    "left-extremist attacks on officers; citizens pay the price, an insecure police cannot protect us.",
    (
        "Llama 8B",
        "132",
    ): "Erdoğan jails opponents, wages a war of annihilation against the Kurds, drives migrants to the EU "
    "border, defames Greece and trivialises Nazi crimes. Where is the clear stance against this "
    "aggression?",
    (
        "Llama 3B",
        "417",
    ): "More testing is needed, and slaughterhouse workers and harvest helpers must no longer be ruthlessly "
    "exploited; labour minister Heil must ensure protection, better working and housing conditions "
    "and nationwide inspections.",
    (
        "Llama 3B",
        "449",
    ): "Earth Overshoot Day: humanity uses more resources than the earth can regenerate, so industry "
    "lobbyists need less priority and companies more legal duties to extend product lifetimes.",
    (
        "Llama 3B",
        "451",
    ): "Interesting poll: a majority supports withdrawing US troops, ending nuclear weapons storage and "
    "rejects higher defence spending; the government should finally act for this majority and push "
    "through disarmament.",
}
# Glosses of German justification sentences quoted in the examples.
SENTENCE_GLOSSES: Dict[Tuple[str, str], str] = {
    (
        "Qwen 7B",
        "176",
    ): "The use of #AfD also points to a right-wing perspective, which supports the high bias score.",
    (
        "Qwen 7B",
        "192",
    ): "Comparison with anchors [1] and [4] shows that such accusations are typically made by right-wing "
    "populist parties such as the AfD or right-leaning politicians to attack the other side and "
    "back the police.",
    (
        "Llama 3B",
        "417",
    ): "The FDP speech [4] lies in the middle: it calls for more inspections and digital time recording "
    "for all workers, but less radically than the Greens' or the AfD's speech.",
    (
        "Llama 3B",
        "449",
    ): "Compared with anchor [5], the text is more critical in tone and calls for social and political "
    "change to promote sustainability.",
    (
        "Llama 3B",
        "451",
    ): "The call to end nuclear weapons storage and support disarmament shows a strong link to the ideas "
    "of the SPD and the Greens, which are clear in anchors [3] and [1].",
}

# Keyword heuristic for how a justification describes the post's position. English
# terms case-insensitive; German only as lower-case adjectives (so 'Die Linke' and
# 'Rechtsstaat' do not count).
RIGHT_DESC_EN = (
    r"\bright[- ]?wing\b|\bfar[- ]right\b|\bright of (?:the )?cent(?:er|re)\b|\bright[- ]leaning\b|"
    r"\bcent(?:er|re)[- ]right\b|\bnationalis|\bconservative\b"
)
LEFT_DESC_EN = (
    r"\bleft[- ]?wing\b|\bfar[- ]left\b|\bleft of (?:the )?cent(?:er|re)\b|\bleft[- ]leaning\b|"
    r"\bcent(?:er|re)[- ]left\b|\bprogressive\b|\bsocialis"
)
RIGHT_DESC_DE = (
    r"\brechte[nrs]?\b|\brechts(?:extrem|populist|radikal|orientiert|gerichtet|konservativ)|"
    r"\bnationalistisch|\bkonservativ"
)
LEFT_DESC_DE = (
    r"\blinke[nrs]?\b|\blinks(?:extrem|populist|radikal|orientiert|gerichtet|liberal)|"
    r"\bprogressiv|\bsozialistisch"
)


LIKEN_WORDS = (
    r"(?:align|consistent|similar|reminiscent|mirror|echo|characteristic|typical|in line|resembl|"
    r"ähnlich|entspricht|typisch|vergleichbar|übereinstimm)"
)
PARTY_WORD = {"AfD": r"\bAfD\b", "DIE LINKE": r"\b(?:LINKEN?|Linken?)\b"}


def likens(text: str, party: str) -> bool:
    """The justification likens the post to ``party`` (keyword heuristic)."""
    import re

    pw = PARTY_WORD[party]
    return bool(
        re.search(
            rf"{LIKEN_WORDS}[^.]{{0,100}}{pw}|{pw}[^.]{{0,60}}"
            rf"(?:stance|rhetoric|ideology|position|style|Position|Rhetorik)[^.]{{0,40}}{LIKEN_WORDS}",
            text or "",
            re.I,
        )
    )


def other_side(score: float, party: str) -> bool:
    lo, hi = em.BLOC_THRESHOLDS
    return score < lo if party == "AfD" else score >= hi


def describes(text: str) -> str:
    import re

    t = text or ""
    right = bool(re.search(RIGHT_DESC_EN, t, re.I) or re.search(RIGHT_DESC_DE, t))
    left = bool(re.search(LEFT_DESC_EN, t, re.I) or re.search(LEFT_DESC_DE, t))
    return (
        "right"
        if right and not left
        else ("left" if left and not right else ("both" if left else "none"))
    )


def score_justification(d: pd.DataFrame, logs) -> pd.DataFrame:
    """Share of justifications that describe the post as right / left, and share where
    the score contradicts that description (right but < 3.4, left but >= 5.2)."""
    lo, hi = em.BLOC_THRESHOLDS
    rows = []
    for folder, *_ in MODELS:
        base, rag = logs[folder]
        g = d[d["model"] == folder]
        for cond, recs, col in (("baseline", base, "B"), ("RAG", rag, "R")):
            desc = [
                describes((recs[t].get("output") or {}).get("justification") or "")
                for t in g["text_index"]
            ]
            g2 = g.assign(desc=desc)
            for s in SUBSETS:
                x = g2[g2["subset"] == s]
                inc = ((x["desc"] == "right") & (x[col] < lo)) | (
                    (x["desc"] == "left") & (x[col] >= hi)
                )
                row = {
                    "Model": DISPLAY[folder],
                    "Condition": cond,
                    "Subset": s,
                    "N": len(x),
                    "Described as right": float((x["desc"] == "right").mean()),
                    "Described as left": float((x["desc"] == "left").mean()),
                    "Score contradicts description": float(inc.mean()),
                    "Posts with contradiction": int(inc.sum()),
                }
                if s in PARTY_WORD:
                    lk = np.array(
                        [
                            likens(
                                (recs[t].get("output") or {}).get("justification")
                                or "",
                                s,
                            )
                            for t in x["text_index"]
                        ],
                        dtype=bool,
                    )
                    os_ = np.array([other_side(v, s) for v in x[col]], dtype=bool)
                    row["Likens post to author party"] = float(lk.mean())
                    row["Likens to author party but scores other side"] = float(
                        (lk & os_).mean()
                    )
                    row["Posts likened but scored other side"] = int((lk & os_).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def score_sentence(justification: str, score: float) -> str:
    """The sentence that states the score (mentions 'score' or the score value)."""
    import re

    val = f"{score:g}"
    hits = [
        s
        for s in _sentences(justification)
        if re.search(rf"\bscore\b|Score|Wert|\b{re.escape(val)}\b", s)
    ]
    return hits[-1] if hits else ""


def extremes_cases(d: pd.DataFrame, comp_bp: pd.DataFrame, logs):
    ann = load_final_annotations()
    afd = d[
        (d["model"] == "llama_70b")
        & (d["party"] == "AfD")
        & (d["R"] - d["B"] <= -CASE_SHIFT)
    ]
    afd_cases = case_rows("llama_70b", afd, logs, ann, "AfD")
    afd_rest = case_rows(
        "llama_70b",
        d[
            (d["model"] == "llama_70b")
            & (d["party"] == "AfD")
            & (d["R"] - d["B"] > -CASE_SHIFT)
        ],
        logs,
        ann,
        "AfD",
    )
    contrast = []
    for m in CONTRAST_MODELS:
        g = d[
            (d["model"] == m)
            & (d["party"] == "AfD")
            & (d["ae_R"] <= d["ae_B"] - MOVE_THRESHOLD)
        ]
        contrast.append(case_rows(m, g, logs, ann, "AfD"))
    contrast = pd.concat(contrast, ignore_index=True)
    # Model with the largest dMAE for DIE LINKE (main scope); Llama 3B uses posts with baseline >= 1.
    lk = comp_bp[
        (comp_bp["Author party"] == "DIE LINKE")
        & comp_bp["Model"].isin([model_label(m) for m, *_ in MODELS])
    ]
    worst_label = lk.loc[lk["dMAE"].idxmax(), "Model"]
    worst = next(m for m, *_ in MODELS if model_label(m) == worst_label)
    gl = d[
        (d["model"] == worst)
        & (d["party"] == "DIE LINKE")
        & (d["R"] - d["B"] >= CASE_SHIFT)
    ]
    if worst == LLAMA_3B:
        gl = gl[gl["B"] >= 1]
    linke = case_rows(worst, gl, logs, ann, "DIE LINKE")
    return afd_cases, afd_rest, contrast, linke, worst


def write_extremes_report(
    d,
    use,
    inter,
    gap_tab,
    pull,
    jl,
    afd_cases,
    afd_rest,
    contrast,
    linke,
    linke_model,
    examples,
    sj,
    n_boot,
    seed,
    readings,
):
    L = ["# Why retrieval helps or hurts: the extreme parties\n", "## Data\n"]
    n = d.groupby("model").size()
    L.append(
        f"Paired e5/simple vs. no-RAG posts per model ({', '.join(f'{DISPLAY[m]} {n[m]}' for m, *_ in MODELS)}). "
        f"E = mean position of the retrieved chunks' parties (CHES 2024 lrgen, two decimals; CDU/CSU faction = CDU "
        f"value; chunks without a position left out; {int(d.drop_duplicates('text_index')['E'].isna().sum())} texts "
        f"without any positioned chunk). Subsets: DIE LINKE posts, AfD posts and the posts of all other parties "
        f"(Grüne, SPD, FDP, CDU, CSU). Regressions: OLS, standard errors clustered by text (as in "
        f"`analyze_chunk_influence.py`); pooled fits add model fixed effects. Judge labels: `final_annotations.jsonl`, "
        f"weighted by the quadrant base rates of each model, CIs from {n_boot} bootstrap resamples of texts within "
        f"quadrants (seed {seed}). Llama 3B: all paired posts and posts with a baseline score ≥ 1. Cited chunks in "
        f"the case listings: final A_caus label where the text was annotated for that model, otherwise the "
        f"deterministic citation rule of the judge (chunk number or speaker name in the justification).\n"
    )
    L.append("## 1. Evidence use by subset\n")
    piv = use[use["Subset"] != "all posts"].assign(
        cell=lambda x: [
            f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
            for v, lo, hi in zip(x["E coefficient"], x["CI low"], x["CI high"])
        ]
    )
    piv = piv.pivot(index="Model", columns="Subset", values="cell").loc[
        use["Model"].drop_duplicates(), list(SUBSETS)
    ]
    L.append("Coefficient on E in R ~ B + E [95 % CI], fitted per subset:\n")
    L.append(em._md_table(piv.reset_index()) + "\n")
    it = inter.assign(
        cell=[
            f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}], p_Holm {'<0.001' if p < 0.001 else f'{p:.3f}'}"
            for v, lo, hi, p in zip(
                inter["Difference"],
                inter["CI low"],
                inter["CI high"],
                inter["p (Holm, 2 contrasts)"],
            )
        ]
    )
    it = it.pivot(index="Model", columns="Contrast", values="cell").loc[
        inter["Model"].drop_duplicates()
    ]
    L.append(
        "Difference of the E coefficient from the other parties (interaction model):\n"
    )
    L.append(em._md_table(it.reset_index()) + "\n")
    L.append(readings["use"] + "\n")
    L.append("## 2. Where the evidence points\n")
    L.append(md_table(gap_tab) + "\n")
    L.append(readings["gap"] + "\n")
    pl = pull[pull["Author party"].isin(["DIE LINKE", "AfD"])][
        [
            "Model",
            "Author party",
            "Mean target",
            "Mean baseline",
            "Mean E",
            "Mean RAG",
            "Mean RAG predicted by the all-posts fit",
            "RAG - predicted",
            "MAE baseline",
            "MAE RAG",
            "MAE if prediction = E",
        ]
    ]
    L.append(md_table(pl) + "\n")
    L.append(readings["pull"] + "\n")
    L.append("## 3. Judge labels of the sampled texts\n")
    jt = jl[
        [
            "Model",
            "Subset",
            "Sampled texts",
            "Mean R_top",
            "Mean R_ideo",
            "Share cited (A_caus = 1)",
            "Cited chunks from the author's party per text",
            "Cited chunks from the opposite bloc per text",
        ]
    ]
    L.append(md_table(jt) + "\n")
    L.append(readings["judge"] + "\n")
    L.append(
        f"## 4. Case study: Llama 70B, AfD posts moved ≥ {CASE_SHIFT} points left\n"
    )
    L.append(readings["cases"] + "\n")
    summ = pd.DataFrame(
        [
            _case_summary(afd_cases, f"moved ≥ {CASE_SHIFT} left"),
            _case_summary(afd_rest, "other AfD posts"),
        ]
    )
    L.append(md_table(summ) + "\n")
    L.append(
        "Score vs. justification (keyword heuristics). 'Contradicts': the justification describes the post as "
        "right but the score is < 3.4, or as left but ≥ 5.2. 'Likened to own party but other side': the "
        "justification likens the post to the author's party (aligns with / similar to / characteristic of … "
        "AfD resp. DIE LINKE) but the score lies on the other side of the centre band (AfD < 3.4, DIE LINKE "
        "≥ 5.2). Shares of posts:\n"
    )
    ex = sj[sj["Subset"].isin(["AfD", "DIE LINKE"])]
    a = ex.pivot_table(
        index=["Model", "Subset"],
        columns="Condition",
        values="Score contradicts description",
    )
    b = ex.pivot_table(
        index=["Model", "Subset"],
        columns="Condition",
        values="Likens to author party but scores other side",
    )
    sjt = pd.DataFrame(
        {
            "contradicts, baseline": a["baseline"],
            "contradicts, RAG": a["RAG"],
            "likened to own party but other side, baseline": b["baseline"],
            "likened to own party but other side, RAG": b["RAG"],
        }
    ).reset_index()
    sjt["Model"] = pd.Categorical(sjt["Model"], [m[3] for m in MODELS], ordered=True)
    L.append(md_table(sjt.sort_values(["Subset", "Model"])) + "\n")
    L.append(readings["consistency"] + "\n")
    L.append("### Typical examples\n")
    L.append(
        f"Selection rule: the {N_EXAMPLES} posts closest to the case-set medians of the shift and of the share "
        f"of cited chunks that are other-party speech naming the author's party (sum of absolute deviations in "
        f"SD units). The decisive sentence is the justification sentence that cites chunks and names parties "
        f"most (keyword heuristic); the score sentence is the one stating the score.\n"
    )
    L.extend(examples["afd"])
    L.append("### Contrast: AfD posts that RAG improved (Qwen 7B, Llama 8B)\n")
    L.append(
        md_table(
            pd.DataFrame(
                [
                    _case_summary(contrast[contrast["Model"] == DISPLAY[m]], DISPLAY[m])
                    for m in CONTRAST_MODELS
                ]
            )
        )
        + "\n"
    )
    L.extend(examples["contrast"])
    L.append(readings["contrast"] + "\n")
    L.append(
        f"## 5. DIE LINKE posts moved ≥ {CASE_SHIFT} points right ({DISPLAY[linke_model]})\n"
    )
    L.append(
        md_table(pd.DataFrame([_case_summary(linke, DISPLAY[linke_model])])) + "\n"
    )
    L.extend(examples["linke"])
    L.append(readings["linke"] + "\n")
    (OUT_DIR / "04_why_rag_extremes.md").write_text("\n".join(L), encoding="utf-8")


def _case_summary(c: pd.DataFrame, label: str) -> dict:
    if c.empty:
        return {"Posts": label, "N": 0}
    return {
        "Posts": label,
        "N": len(c),
        "Mean shift (RAG - baseline)": c["RAG - baseline"].mean(),
        "Mean E": c["E"].mean(),
        "Retrieved from author party (of 5)": c["Retrieved from author party"].mean(),
        "Cited chunks": c["Cited chunks"].mean(),
        "Cited from author party": c["Cited from author party"].mean(),
        "Cited other-party chunks naming the author party": c[
            "Cited other-party chunks naming the author party"
        ].mean(),
        "Retrieved other-party chunks naming the author party": c[
            "Retrieved other-party chunks naming the author party"
        ].mean(),
        "Posts without numeric citation": int(
            (c["Citation source"] == "speaker names only (no numeric citation)").sum()
        ),
        "Score contradicts justification": int(
            c["Score contradicts description"].sum()
        ),
        "Likens to author party": int(c["Likens post to author party"].sum()),
        "Likened but scored other side": int(c["Likened but scored other side"].sum()),
    }


def format_examples(sel: pd.DataFrame) -> List[str]:
    out = []
    for _, r in sel.iterrows():
        key = (r["Model"], str(r["text_index"]))
        gloss = CASE_GLOSSES.get(key, "(gloss missing)")
        post = " ".join(r["Post"].split())
        post = post if len(post) <= 280 else post[:277].rsplit(" ", 1)[0] + " …"
        sg = SENTENCE_GLOSSES.get(key)
        txt = (
            f"**{r['Model']}, text {r['text_index']}** ({r['Author party']}, target {r['Target']:.2f}): baseline "
            f"{r['Baseline']:.1f} → RAG {r['RAG']:.1f}; E = {r['E']:.2f}.\n\n"
            f"> {post}\n>\n> *Gloss:* {gloss}\n\n"
            f"Chunks (* = cited): {r['Chunks (* = cited)']}"
            + (f" (judge: {r['Judge labels']})" if r["Judge labels"] else "")
            + ".\n\n"
            f"Decisive sentence of the RAG justification: “{r['Decisive sentence']}”"
            + (f" *(Gloss: {sg})*" if sg else "")
            + "\n"
        )
        if r["Score sentence"] and r["Score sentence"] != r["Decisive sentence"]:
            txt += f"\nScore sentence: “{r['Score sentence']}”\n"
        out.append(txt)
    return out


def extremes_readings(
    use, inter, gap_tab, pull, jl, afd_cases, afd_rest, contrast, linke, linke_model, sj
) -> Dict[str, str]:
    """Reading paragraphs; numbers and model lists from the tables. The interpretation
    (score generated before the justification) refers to the JSON schema in rag/evaluator.py.
    """
    main = [model_label(m) for m, *_ in MODELS]
    BLOC_KEY = "Share of posts with E in the target's bloc"
    OWN_KEY = "Cited chunks from the author's party per text"
    pooled = "All models (model fixed effects)"
    u = lambda sub: _get(use, Model=pooled, Subset=sub)  # noqa: E731
    sig = inter[(inter["p (Holm, 2 contrasts)"] < 0.05) & inter["Model"].isin(main)]
    sig_txt = (
        ", ".join(
            f"{m} ({c.split(', ')[1].split(' −')[0]} {v:+.2f})"
            for m, c, v in zip(sig["Model"], sig["Contrast"], sig["Difference"])
        )
        or "none"
    )
    l70 = _get(use, Model="Llama 70B", Subset="AfD")
    r = {}
    r["use"] = (
        f"Reading: pooled over models, the pull towards the retrieved parties is similar for the three subsets "
        f"(E coefficient DIE LINKE {u('DIE LINKE')['E coefficient']:.2f}, AfD {u('AfD')['E coefficient']:.2f}, other "
        f"parties {u('other parties')['E coefficient']:.2f}; the pooled contrasts are not significant). Within single "
        f"models, the E coefficient of an extreme party differs from the other parties after Holm correction in: "
        f"{sig_txt}. Llama 70B is the only model with a negative coefficient for AfD posts "
        f"({l70['E coefficient']:.2f} [{l70['CI low']:.2f}, {l70['CI high']:.2f}]): among AfD posts, the more AfD-heavy "
        f"the retrieved evidence, the further left its RAG prediction. This is the opposite of evidence following."
    )
    ga = _get(gap_tab, **{"Author party": "AfD"})
    gl = _get(gap_tab, **{"Author party": "DIE LINKE"})
    pm = pull[pull["Model"].isin(main)]
    worse = {
        p: int(
            (
                pm[pm["Author party"] == p]["MAE baseline"]
                < pm[pm["Author party"] == p]["MAE if prediction = E"]
            ).sum()
        )
        for p in ("AfD", "DIE LINKE")
    }
    r["gap"] = (
        f"Reading: the evidence of the extreme parties lies far towards the centre. For AfD posts E averages "
        f"{ga['Mean E']:.2f}, {abs(ga['E - target']):.2f} points left of the target, and lies in the right bloc for only "
        f"{ga[BLOC_KEY] * 100:.0f} % of the posts; for DIE LINKE posts E averages "
        f"{gl['Mean E']:.2f}, {gl['E - target']:.2f} points right of the target (left bloc in "
        f"{gl[BLOC_KEY] * 100:.0f} %). A model that predicted E itself would have "
        f"an MAE of {ga['MAE of E as prediction']:.2f} on AfD and {gl['MAE of E as prediction']:.2f} on DIE LINKE posts, "
        f"worse than the baseline of {worse['AfD']} (AfD) and {worse['DIE LINKE']} (DIE LINKE) of the {len(main)} "
        f"models. Following the evidence therefore has to cost accuracy at the two ends, in "
        f"proportion to the model's evidence use."
    )
    pa = pull[(pull["Author party"] == "AfD") & pull["Model"].isin(main)].set_index(
        "Model"
    )
    worst = pa["MAE RAG"].sub(pa["MAE baseline"]).sort_values(ascending=False)
    r["pull"] = (
        f"Reading: the column 'Mean RAG predicted by the all-posts fit' shows where a model's general evidence use "
        f"(R ~ B + E over all its posts) puts the extreme posts. For AfD posts the two models that lose most are "
        f"{worst.index[0]} (MAE {pa.loc[worst.index[0], 'MAE baseline']:.2f} → {pa.loc[worst.index[0], 'MAE RAG']:.2f}) "
        f"and {worst.index[1]} ({pa.loc[worst.index[1], 'MAE baseline']:.2f} → {pa.loc[worst.index[1], 'MAE RAG']:.2f}). "
        f"For {worst.index[1]} the observed mean ({pa.loc[worst.index[1], 'Mean RAG']:.2f}) is close to the prediction "
        f"of its general evidence use ({pa.loc[worst.index[1], 'Mean RAG predicted by the all-posts fit']:.2f}): it "
        f"follows the centre-leaning evidence. Llama 70B's AfD predictions ({pa.loc['Llama 70B', 'Mean RAG']:.2f}) lie "
        f"{abs(pa.loc['Llama 70B', 'RAG - predicted']):.2f} points left of what its evidence use predicts, and section 1 "
        f"shows that they move against E within the AfD posts. Its loss needs another explanation (section 4)."
    )
    pj = jl[jl["Model"] == "All models (Llama 3B: all)"].set_index("Subset")
    small = jl[jl["Small N (< 10 texts)"] & jl["Model"].isin(main)]
    r["judge"] = (
        f"Reading: pooled over models, the chunks retrieved for AfD and DIE LINKE posts are more ideological (mean "
        f"R_ideo {pj.loc['AfD', 'Mean R_ideo']:.2f} and {pj.loc['DIE LINKE', 'Mean R_ideo']:.2f} vs "
        f"{pj.loc['other parties', 'Mean R_ideo']:.2f}), about as relevant (R_top {pj.loc['AfD', 'Mean R_top']:.2f}, "
        f"{pj.loc['DIE LINKE', 'Mean R_top']:.2f}, {pj.loc['other parties', 'Mean R_top']:.2f}) and cited as often "
        f"(share {pj.loc['AfD', 'Share cited (A_caus = 1)']:.2f}, {pj.loc['DIE LINKE', 'Share cited (A_caus = 1)']:.2f}, "
        f"{pj.loc['other parties', 'Share cited (A_caus = 1)']:.2f}). Per text, the models cite "
        f"{pj.loc['AfD', OWN_KEY]:.2f} AfD chunks for AfD posts and "
        f"{pj.loc['DIE LINKE', OWN_KEY]:.2f} DIE LINKE chunks for DIE LINKE "
        f"posts, but also {pj.loc['AfD', 'Cited chunks from the opposite bloc per text']:.2f} left-bloc chunks for AfD "
        f"posts. The per-model cells are small: {len(small)} cells have fewer than {SMALL_TEXTS} sampled texts (e.g. "
        f"{', '.join(f'{m} {s} ({n})' for m, s, n in zip(small['Model'][:4], small['Subset'][:4], small['Sampled texts'][:4]))}), "
        f"so per-model differences are not interpreted."
    )
    ca = _case_summary(afd_cases, "")
    cr = _case_summary(afd_rest, "")
    top_scores = afd_cases["RAG"].value_counts()
    r["cases"] = (
        f"Reading: {ca['N']} of Llama 70B's AfD posts move at least {CASE_SHIFT} points left with RAG (mean shift "
        f"{ca['Mean shift (RAG - baseline)']:.1f}). {int((afd_cases['Baseline'] == 6.5).sum())} of them had a baseline "
        f"of 6.5, and the RAG scores cluster at {top_scores.index[0]:g} ({top_scores.iloc[0]} posts) and "
        f"{top_scores.index[1]:g} ({top_scores.iloc[1]}). The evidence does not explain the jump: these posts retrieve "
        f"more AfD chunks than the other AfD posts ({ca['Retrieved from author party (of 5)']:.1f} vs "
        f"{cr['Retrieved from author party (of 5)']:.1f} of 5) and cite more of them ({ca['Cited from author party']:.1f} "
        f"vs {cr['Cited from author party']:.1f}). Cited speeches of other parties about the AfD are rare "
        f"({ca['Cited other-party chunks naming the author party']:.2f} per post vs "
        f"{cr['Cited other-party chunks naming the author party']:.2f}). What sets them apart is the justification: "
        f"{ca['Likens to author party']} of {ca['N']} liken the post to the AfD, yet {ca['Likened but scored other side']} "
        f"of these are scored below 3.4."
    )
    l70sj = _get(sj, Model="Llama 70B", Condition="RAG", Subset="AfD")
    l70b = _get(sj, Model="Llama 70B", Condition="baseline", Subset="AfD")
    others = sj[
        (sj["Condition"] == "RAG")
        & (sj["Subset"] == "AfD")
        & (sj["Model"] != "Llama 70B")
    ]
    lk = (
        sj[(sj["Condition"] == "RAG") & (sj["Subset"] == "DIE LINKE")]
        .set_index("Model")["Likens to author party but scores other side"]
        .sort_values(ascending=False)
    )
    r["consistency"] = (
        f"Reading: with RAG, {l70sj['Likens to author party but scores other side'] * 100:.0f} % of Llama 70B's AfD "
        f"justifications liken the post to the AfD but give a score below 3.4 (baseline "
        f"{l70b['Likens to author party but scores other side'] * 100:.0f} %); in every other model this share is at "
        f"most {others['Likens to author party but scores other side'].max() * 100:.0f} %. The score and the reasoning "
        f"come apart: the justification states that the post matches the AfD anchors, while the number lands on the "
        f"left. In at least one case (text 114) the justification even calls the AfD's ideology 'left-wing populist'. "
        f"The output schema in `rag/evaluator.py` "
        f"lists `bias_score` before `justification`, so with constrained decoding the score is most likely generated "
        f"before the reasoning, and the reasoning cannot correct it. For DIE LINKE posts the mirror pattern (likened to "
        f"DIE LINKE but scored ≥ 5.2) appears mainly in {lk.index[0]} ({lk.iloc[0] * 100:.0f} %) and {lk.index[1]} "
        f"({lk.iloc[1] * 100:.0f} %). Both measures are keyword heuristics; a manual check of Llama 70B case "
        f"justifications found the flagged ones to be real contradictions and the unflagged ones to include further "
        f"contradictions, so the shares are lower bounds."
    )
    cq = _case_summary(contrast[contrast["Model"] == "Qwen 7B"], "")
    c8 = _case_summary(contrast[contrast["Model"] == "Llama 8B"], "")
    right_share = (contrast["Justification describes the post as"] == "right").mean()
    own_cited = contrast["Cited from author party"].sum() / max(
        contrast["Cited chunks"].sum(), 1
    )
    r["contrast"] = (
        f"Reading: Qwen 7B improves {cq['N']} AfD posts by at least {MOVE_THRESHOLD} points (mean shift "
        f"{cq['Mean shift (RAG - baseline)']:+.2f}), Llama 8B {c8['N']} (mean {c8['Mean shift (RAG - baseline)']:+.2f}). "
        f"In these posts the evidence and the reading agree: {own_cited * 100:.0f} % of the cited chunks are AfD "
        f"speeches, {right_share * 100:.0f} % of the justifications describe the post as right (and not as left), and "
        f"{int(contrast['Likened but scored other side'].sum())} liken it to the AfD while scoring it below 3.4. Qwen 7B "
        f"cites few chunks by number ({cq['Posts without numeric citation']} of {cq['N']} posts without a numeric "
        f"citation)."
    )
    lc = _case_summary(linke, "")
    ls = linke["RAG"].value_counts()
    r["linke"] = (
        f"Reading: the model with the largest ΔMAE on DIE LINKE posts is {DISPLAY[linke_model]}"
        + (
            " (posts with a baseline score ≥ 1 only, to exclude the sub-1 scores)"
            if linke_model == LLAMA_3B
            else ""
        )
        + f". {lc['N']} of its DIE LINKE posts move at least {CASE_SHIFT} points right (mean "
        f"{lc['Mean shift (RAG - baseline)']:+.2f}); their RAG scores cluster at {ls.index[0]:g} ({ls.iloc[0]} posts). "
        f"They retrieve {lc['Retrieved from author party (of 5)']:.1f} DIE LINKE chunks of 5 but cite only "
        f"{lc['Cited from author party']:.2f}, and {lc['Posts without numeric citation']} of {lc['N']} justifications "
        f"cite no chunk by number. {lc['Likened but scored other side']} liken the post to DIE LINKE but score it "
        f"≥ 5.2. Because this model shifts all posts towards the centre band with RAG (section 1 of "
        f"`04_why_rag_direction.md`), these cases mainly show that general shift rather than DIE LINKE-specific "
        f"evidence."
    )
    return r


def cmd_extremes(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading paired logs and retrievals ...")
    d, ret = extremes_data()
    use, inter = extremes_evidence_use(d)
    gap_tab, pull = extremes_evidence_gap(d, args.n_boot, args.seed)
    print("Judge labels ...")
    jl = extremes_judge_labels(d, args.n_boot, args.seed)
    print("Case studies ...")
    bp = direction_by_party(d, 0, args.seed)
    logs = _log_records()
    afd_cases, afd_rest, contrast, linke, linke_model = extremes_cases(d, bp, logs)
    sj = score_justification(d, logs)
    examples = {
        "afd": format_examples(typical(afd_cases, N_EXAMPLES)),
        "contrast": format_examples(
            pd.concat(
                [
                    typical(contrast[contrast["Model"] == DISPLAY[m]], n)
                    for m, n in zip(CONTRAST_MODELS, (2, 1))
                ]
            )
        ),
        "linke": format_examples(typical(linke, N_LINKE_EXAMPLES)),
    }
    drop = ["Baseline justification"]
    for df, name in (
        (use, "extremes_evidence_use.csv"),
        (inter, "extremes_evidence_use_contrasts.csv"),
        (gap_tab, "extremes_evidence_gap.csv"),
        (pull, "extremes_pull_by_party.csv"),
        (jl, "extremes_judge_labels.csv"),
        (afd_cases.drop(columns=drop), "extremes_cases_llama70b_afd.csv"),
        (contrast.drop(columns=drop), "extremes_cases_contrast.csv"),
        (linke.drop(columns=drop), "extremes_cases_linke.csv"),
        (sj, "extremes_score_justification.csv"),
    ):
        print("  wrote", write_csv(df, name))
    write_extremes_report(
        d,
        use,
        inter,
        gap_tab,
        pull,
        jl,
        afd_cases,
        afd_rest,
        contrast,
        linke,
        linke_model,
        examples,
        sj,
        args.n_boot,
        args.seed,
        extremes_readings(
            use,
            inter,
            gap_tab,
            pull,
            jl,
            afd_cases,
            afd_rest,
            contrast,
            linke,
            linke_model,
            sj,
        ),
    )
    print("  wrote", OUT_DIR / "04_why_rag_extremes.md")


# ===========================================================================
# labels: what the judge labels say (sample, quadrant-weighted, descriptive)
# ===========================================================================

LABEL_SCOPES = ("all", "Llama 3B baseline >= 1")


def tier_comparison() -> pd.DataFrame:
    """Refit the chunk-influence sample analyses (analyze_chunk_influence.py, its own
    positions and functions) on model subsets, to separate the effect of the human
    review (final vs. judge labels on the same models) from the effect of which models
    are in the 'reviewed' tier."""
    pop = aci.add_outcome(aci.load_population())
    t_final = aci.text_table(aci.load_sample(FAMILIES_DIR, "final", pop))
    t_judge = aci.text_table(aci.load_sample(FAMILIES_DIR, "judge", pop))
    reviewed = list(aci.REVIEWED_MODELS)
    subsets = [
        ("All 10 models (tier 'all')", t_final, None),
        ("3 reviewed models (tier 'reviewed')", t_final, reviewed),
        ("3 reviewed models, raw judge labels", t_judge, reviewed),
        ("7 other models", t_final, [m for m, *_ in MODELS if m not in reviewed]),
    ]
    subsets += [(f"{DISPLAY[m]} only", t_final, [m]) for m, *_ in MODELS]
    rows = []
    for name, t, models in subsets:
        tt = t if models is None else t[t["model"].isin(models)]
        row = {
            "Subset": name,
            "Models": len(tt["model"].unique()),
            "Sampled texts": len(tt),
        }
        try:
            ev = aci.evidence_by_label(tt).set_index("term")
            for term in ("E_cited", "E_uncited", "E_relevant"):
                row[f"{term} coef"] = ev.loc[term, "coef"]
                row[f"{term} CI low"], row[f"{term} CI high"] = (
                    ev.loc[term, "ci_low"],
                    ev.loc[term, "ci_high"],
                )
            row["Texts in E_cited fit"] = int(ev.loc["E_cited", "n_texts"])
        except Exception as exc:  # noqa: BLE001 - too few texts in a single model
            row["Note"] = f"evidence fit failed: {type(exc).__name__}"
        for col, label in (
            ("share_same_party", "chunks from input's party"),
            ("cited_same_party", "cited chunks from input's party"),
        ):
            coef, p = aci._q1_vs_q2(tt, col)
            q = tt[tt["quadrant"].isin(["Q1", "Q2"])].groupby("quadrant")[col].mean()
            row[f"Q1 mean, {label}"], row[f"Q2 mean, {label}"] = q.get(
                "Q1", np.nan
            ), q.get("Q2", np.nan)
            row[f"Q1 vs Q2 log-odds per SD, {label}"], row[f"p, {label}"] = coef, p
        row["Q1 texts"] = int((tt["quadrant"] == "Q1").sum())
        row["Q2 texts"] = int((tt["quadrant"] == "Q2").sum())
        rows.append(row)
    # Formal test of the tier difference: one weighted fit on all models with E_cited x
    # (reviewed model), clustered by text (texts recur across models).
    import statsmodels.formula.api as smf

    data = t_final.dropna(subset=["R", "B", "E_cited", "E_uncited"]).copy()
    data["rev"] = data["model"].isin(reviewed).astype(int)
    fit = smf.wls(
        "R ~ B + E_cited * rev + E_uncited * rev + C(model)",
        data,
        weights=data["weight"],
    ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["text_index"])[0]})
    ci_ = fit.conf_int()
    rows.append(
        {
            "Subset": "Difference: 3 reviewed − 7 other models (interaction E_cited × reviewed)",
            "Models": 10,
            "Sampled texts": len(data),
            "E_cited coef": fit.params["E_cited:rev"],
            "E_cited CI low": ci_.loc["E_cited:rev", 0],
            "E_cited CI high": ci_.loc["E_cited:rev", 1],
            "p, E_cited difference": fit.pvalues["E_cited:rev"],
        }
    )
    # Population evidence use per model (from the same script), to show what the
    # reviewed tier selects.
    ev_pop, _ = aci.evidence_use(pop)
    evp = ev_pop.set_index("model")["coef"]
    out = pd.DataFrame(rows)
    out["Population evidence use (E coef)"] = [
        evp.get(next((m for m, *_ in MODELS if f"{DISPLAY[m]} only" == s), ""), np.nan)
        for s in out["Subset"]
    ]
    return out


def label_chunks(d: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Final-annotation chunks with chunk party, author party, errors and sampling weights."""
    ch = load_final_annotations()
    ch["chunk_group"] = ch["chunk_party_raw"].map(
        lambda r: party_group(_raw_to_canon(r, "annotated chunk")) or NO_POSITION
    )
    ch["chunk_pos"] = ch["chunk_group"].map(
        lambda g: group_position(g) if g != NO_POSITION else np.nan
    )
    ch["chunk_bloc"] = ch["chunk_pos"].map(
        lambda p: BLOC_NAMES[em._bloc(p)] if not pd.isna(p) else NO_POSITION
    )
    meta = d[["model", "text_index", "party", "target", "B", "R", "ae_B", "ae_R"]]
    ch = ch.merge(meta, on=["model", "text_index"], how="left")
    if ch["party"].isna().any():
        raise SystemExit("Annotated texts without a paired post")
    if scope != "all":
        ch = ch[(ch["model"] != LLAMA_3B) | (ch["B"] >= 1)]
    ch["same_party"] = ch["chunk_group"] == ch["party"]
    ch["relevant"] = ch["R_top"] >= 4
    ch["cited"] = (ch["A_caus"] == 1).astype(float)
    w = {}
    for folder, *_ in MODELS:
        texts = ch[ch["model"] == folder].drop_duplicates("text_index")
        sc = SCOPE_GE1 if (scope != "all" and folder == LLAMA_3B) else SCOPE_ALL
        w[folder] = sampling_weights(d, texts, sc, folder)
    ch["w"] = [w[m][q] for m, q in zip(ch["model"], ch["quadrant"])]
    return ch.reset_index(drop=True)


def _weighted_stat(
    ch: pd.DataFrame, mask: np.ndarray, value: np.ndarray, m: np.ndarray
):
    mask = mask.astype(float)
    w = ch["w"].to_numpy(float)
    if (mask * w).sum() == 0:
        return np.nan, np.nan, np.nan
    est = float((w * mask * value).sum() / (w * mask).sum())
    lo, hi = ci(boot_ratio(mask * value, mask, w, m))
    return est, lo, hi


def labels_by_chunk_party(ch: pd.DataFrame, scope: str, m: np.ndarray) -> pd.DataFrame:
    rows = []
    groups = [
        ("Chunk party", GROUP_DISPLAY.get(g, g), (ch["chunk_group"] == g).to_numpy())
        for g in group_order(
            [g for g in ch["chunk_group"].unique() if g != NO_POSITION]
        )
    ]
    groups.append(
        ("Chunk party", NO_POSITION, (ch["chunk_group"] == NO_POSITION).to_numpy())
    )
    groups += [
        ("Chunk bloc", b, (ch["chunk_bloc"] == b).to_numpy()) for b in BLOC_NAMES
    ]
    groups += [
        ("Chunk from the author's party", "yes", ch["same_party"].to_numpy()),
        ("Chunk from the author's party", "no", (~ch["same_party"]).to_numpy()),
    ]
    for kind, name, mask in groups:
        row = {
            "Llama 3B posts": scope,
            "Grouping": kind,
            "Group": name,
            "Chunks": int(mask.sum()),
            "Texts": int(
                ch.loc[mask, ["model", "text_index"]].drop_duplicates().shape[0]
            ),
        }
        for col, label in (
            ("R_top", "Mean R_top"),
            ("R_ideo", "Mean R_ideo"),
            ("cited", "Share cited (A_caus = 1)"),
        ):
            row[label], row[f"{label} CI low"], row[f"{label} CI high"] = (
                _weighted_stat(ch, mask, ch[col].to_numpy(float), m)
            )
        row["Share R_ideo >= 4"] = _weighted_stat(
            ch, mask, (ch["R_ideo"] >= 4).to_numpy(float), m
        )[0]
        rows.append(row)
    return pd.DataFrame(rows)


def labels_relevance_party(ch: pd.DataFrame, scope: str, m: np.ndarray) -> pd.DataFrame:
    rows = []
    same = ch["same_party"].to_numpy(float)
    rel = ch["relevant"].to_numpy()
    w = ch["w"].to_numpy(float)
    for author in ["All posts"] + AUTHOR_GROUPS:
        a = (
            np.ones(len(ch), bool)
            if author == "All posts"
            else (ch["party"] == author).to_numpy()
        )
        row = {
            "Llama 3B posts": scope,
            "Author party": GROUP_DISPLAY.get(author, author),
        }
        bs = {}
        for name, mask in (
            ("relevant (R_top >= 4)", a & rel),
            ("other (R_top <= 3)", a & ~rel),
        ):
            mk = mask.astype(float)
            row[f"Chunks, {name}"] = int(mask.sum())
            row[f"Share from author's party, {name}"] = float(
                (w * mk * same).sum() / (w * mk).sum()
            )
            bs[name] = boot_ratio(mk * same, mk, w, m)
            row[f"CI low, {name}"], row[f"CI high, {name}"] = ci(bs[name])
        diff = bs["relevant (R_top >= 4)"] - bs["other (R_top <= 3)"]
        row["Difference (relevant - other)"] = (
            row["Share from author's party, relevant (R_top >= 4)"]
            - row["Share from author's party, other (R_top <= 3)"]
        )
        row["Difference CI low"], row["Difference CI high"] = ci(diff)
        row["Bootstrap p"] = em._boot_p_two_sided(diff)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Bootstrap p (Holm, across author parties)"] = np.nan
    parties = df["Author party"] != "All posts"
    df.loc[parties, "Bootstrap p (Holm, across author parties)"] = em._holm(
        df.loc[parties, "Bootstrap p"].tolist()
    )
    return df


def labels_vs_error(ch: pd.DataFrame, scope: str) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    t = ch.groupby(["model", "text_index", "quadrant", "party"], as_index=False).agg(
        mean_R_top=("R_top", "mean"),
        mean_R_ideo=("R_ideo", "mean"),
        w=("w", "first"),
        ae_B=("ae_B", "first"),
        ae_R=("ae_R", "first"),
    )
    t["dAE"] = t["ae_R"] - t["ae_B"]
    t["party"] = t["party"].map(GROUP_DISPLAY)
    rows = []
    specs = [
        ("mean R_top only", "dAE ~ mean_R_top"),
        ("mean R_ideo only", "dAE ~ mean_R_ideo"),
        ("both", "dAE ~ mean_R_top + mean_R_ideo"),
    ]
    for spec, rhs in specs:
        formula = f"{rhs} + C(party, Treatment('SPD')) + C(model)"
        for weighting, wts in (
            ("weighted by quadrant base rates", t["w"]),
            ("unweighted", None),
        ):
            model = (
                smf.wls(formula, t, weights=wts)
                if wts is not None
                else smf.ols(formula, t)
            )
            fit = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": pd.factorize(t["text_index"])[0]},
            )
            ci_ = fit.conf_int()
            for term in ("mean_R_top", "mean_R_ideo"):
                if term not in fit.params:
                    continue
                rows.append(
                    {
                        "Llama 3B posts": scope,
                        "Specification": spec,
                        "Weighting": weighting,
                        "Term": term.replace("_", " "),
                        "Coefficient (change in dAE per label point)": fit.params[term],
                        "CI low": ci_.loc[term, 0],
                        "CI high": ci_.loc[term, 1],
                        "p": fit.pvalues[term],
                        "Texts": int(fit.nobs),
                        "Formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def labels_relevance_party_quadrant(
    ch: pd.DataFrame, scope: str, n_boot: int, seed: int
) -> pd.DataFrame:
    """Cited chunks in the 2x2 relevant x same party: share of Q1 among the Q1/Q2 texts
    with at least one cited chunk in the cell (weighted), and the chunk-level share."""
    c = ch[ch["quadrant"].isin(["Q1", "Q2"])].copy()
    cited = c[c["cited"] == 1]
    t_all = c.drop_duplicates(["model", "text_index"])[
        ["model", "text_index", "quadrant", "w"]
    ]
    cells = [("any cited chunk (reference)", None, None)]
    cells += [
        (
            f"{'relevant' if r else 'not relevant'} × {'same party' if s else 'other party'}",
            r,
            s,
        )
        for r in (True, False)
        for s in (True, False)
    ]
    rows = []
    for name, r, s in cells:
        sel = (
            cited
            if r is None
            else cited[(cited["relevant"] == r) & (cited["same_party"] == s)]
        )
        keys = sel[["model", "text_index"]].drop_duplicates()
        tt = t_all.merge(keys, on=["model", "text_index"])
        q1 = (tt["quadrant"] == "Q1").to_numpy(float)
        w = tt["w"].to_numpy(float)
        m = multiplicities(
            tt["text_index"].to_numpy(), None, n_boot, np.random.default_rng(seed)
        )
        lo, hi = ci(boot_ratio(q1, np.ones(len(tt)), w, m))
        cq = sel["quadrant"].value_counts()
        rows.append(
            {
                "Llama 3B posts": scope,
                "Cited chunks in cell": name,
                "Q1 texts": int(q1.sum()),
                "Q2 texts": int(len(tt) - q1.sum()),
                "Share Q1 (weighted)": float((w * q1).sum() / w.sum()),
                "CI low": lo,
                "CI high": hi,
                "Share Q1 (unweighted)": float(q1.mean()),
                "Cited chunks in Q1 texts": int(cq.get("Q1", 0)),
                "Cited chunks in Q2 texts": int(cq.get("Q2", 0)),
            }
        )
    return pd.DataFrame(rows)


def write_labels_report(tiers, by_party, relp, vs_err, rpq, ch, n_boot, seed, readings):
    L = ["# Why retrieval helps or hurts: what the chunk labels show\n", "## Data\n"]
    nt = ch.drop_duplicates(["model", "text_index"]).groupby("model").size()
    L.append(
        f"Sample: `final_annotations.jsonl` of the ten models, {len(ch)} chunks of {int(nt.sum())} (model, text) pairs "
        f"({', '.join(f'{DISPLAY[m]} {nt[m]}' for m, *_ in MODELS)}). Quadrant = the quadrant each text was sampled in. "
        f"All shares and means are weighted by the quadrant base rates of each model (population texts / sampled "
        f"texts, Q1-Q4 only; the buffer is not sampled) unless marked unweighted. CIs: {n_boot} bootstrap resamples of "
        f"text_index clusters (the same texts appear in several models), seed {seed}. Positions for sections 2-5: "
        f"CHES 2024 lrgen, two decimals, CDU/CSU faction = CDU value; chunks of non-attached speakers count as 'no "
        f"position' and never as the author's party. Section 1 re-runs `analyze_chunk_influence.py` unchanged (its "
        f"own exact lrgen positions). Every table is also computed without the Llama 3B texts with a baseline score "
        f"< 1 (column 'Llama 3B posts' in the CSVs). The sample is selected on the outcome, so everything here is "
        f"descriptive.\n"
    )
    L.append("## 1. Tier 'all' vs. tier 'reviewed'\n")
    L.append(
        "`analyze_chunk_influence.py --labels final --tier all|reviewed`, output in `chunk_influence_all/` and "
        "`chunk_influence_reviewed/`. The tier 'reviewed' keeps every annotated text of the three models with a "
        "human review (Llama 70B, Ministral 14B, Qwen 3B), including their unreviewed Q3/Q4 texts; it is not "
        "a filter on reviewed labels. The table refits the same analyses on model subsets:\n"
    )
    tt = tiers[
        [
            "Subset",
            "Sampled texts",
            "Population evidence use (E coef)",
            "E_cited coef",
            "E_cited CI low",
            "E_cited CI high",
            "Q1 mean, chunks from input's party",
            "Q2 mean, chunks from input's party",
            "p, chunks from input's party",
            "Q1 texts",
            "Q2 texts",
        ]
    ]
    L.append(md_table(tt) + "\n")
    L.append(readings["tiers"] + "\n")
    L.append(readings["recommendation"] + "\n")
    L.append("## 2. Labels by the chunk's party\n")
    bp = by_party[by_party["Llama 3B posts"] == "all"]
    bt = pd.DataFrame(
        {
            "Grouping": bp["Grouping"],
            "Group": bp["Group"],
            "Chunks": bp["Chunks"],
            "Mean R_top [95 % CI]": [
                f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
                for v, lo, hi in zip(
                    bp["Mean R_top"], bp["Mean R_top CI low"], bp["Mean R_top CI high"]
                )
            ],
            "Mean R_ideo [95 % CI]": [
                f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
                for v, lo, hi in zip(
                    bp["Mean R_ideo"],
                    bp["Mean R_ideo CI low"],
                    bp["Mean R_ideo CI high"],
                )
            ],
            "Share cited [95 % CI]": [
                f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
                for v, lo, hi in zip(
                    bp["Share cited (A_caus = 1)"],
                    bp["Share cited (A_caus = 1) CI low"],
                    bp["Share cited (A_caus = 1) CI high"],
                )
            ],
        }
    )
    L.append(em._md_table(bt) + "\n")
    L.append(readings["by_party"] + "\n")
    L.append("## 3. Relevant chunks and the author's party\n")
    rp = relp[relp["Llama 3B posts"] == "all"]
    rt = pd.DataFrame(
        {
            "Author party": rp["Author party"],
            "relevant: share own party": [
                f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
                for v, lo, hi in zip(
                    rp["Share from author's party, relevant (R_top >= 4)"],
                    rp["CI low, relevant (R_top >= 4)"],
                    rp["CI high, relevant (R_top >= 4)"],
                )
            ],
            "other: share own party": [
                f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
                for v, lo, hi in zip(
                    rp["Share from author's party, other (R_top <= 3)"],
                    rp["CI low, other (R_top <= 3)"],
                    rp["CI high, other (R_top <= 3)"],
                )
            ],
            "difference": [
                f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                for v, lo, hi in zip(
                    rp["Difference (relevant - other)"],
                    rp["Difference CI low"],
                    rp["Difference CI high"],
                )
            ],
            "p (Holm)": rp["Bootstrap p (Holm, across author parties)"],
            "chunks relevant / other": [
                f"{a} / {b}"
                for a, b in zip(
                    rp["Chunks, relevant (R_top >= 4)"],
                    rp["Chunks, other (R_top <= 3)"],
                )
            ],
        }
    )
    L.append(md_table(rt) + "\n")
    L.append(readings["relevance"] + "\n")
    L.append("## 4. Labels and the change in error (text level)\n")
    ve = vs_err[(vs_err["Llama 3B posts"] == "all")]
    vt = pd.DataFrame(
        {
            "Specification": ve["Specification"],
            "Weighting": ve["Weighting"],
            "Term": ve["Term"],
            "Coefficient [95 % CI]": [
                f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                for v, lo, hi in zip(
                    ve["Coefficient (change in dAE per label point)"],
                    ve["CI low"],
                    ve["CI high"],
                )
            ],
            "p": ve["p"],
            "Texts": ve["Texts"],
        }
    )
    L.append(
        f"dAE = |e_RAG| − |e_baseline| per (model, text); `{ve['Formula'].iloc[-1]}`, clustered by text. "
        f"Negative coefficient = higher label goes with a larger error reduction.\n"
    )
    L.append(md_table(vt) + "\n")
    L.append(readings["error"] + "\n")
    L.append("## 5. Cited chunks: relevant × same party, Q1 vs. Q2\n")
    q = rpq[rpq["Llama 3B posts"] == "all"].drop(columns=["Llama 3B posts"])
    L.append(
        "Q1/Q2 texts with at least one cited chunk in the cell; 'Share Q1' = share of these texts that are RAG "
        "wins rather than distractions (weighted by the base rates of Q1 and Q2).\n"
    )
    L.append(md_table(q) + "\n")
    L.append(readings["quadrant"] + "\n")
    (OUT_DIR / "04_why_rag_labels.md").write_text("\n".join(L), encoding="utf-8")


def labels_readings(tiers, by_party, relp, vs_err, rpq) -> Dict[str, str]:
    """Reading paragraphs; numbers from the tables. Checks the Llama-3B-restricted
    version and says whether it changes anything."""
    T = tiers.set_index("Subset")
    P_KEY, Q1_KEY, Q2_KEY = (
        "p, chunks from input's party",
        "Q1 mean, chunks from input's party",
        "Q2 mean, chunks from input's party",
    )
    REL_KEY = "Share from author's party, relevant (R_top >= 4)"
    OTH_KEY = "Share from author's party, other (R_top <= 3)"
    a, rv, rj, o = (
        T.loc[k]
        for k in (
            "All 10 models (tier 'all')",
            "3 reviewed models (tier 'reviewed')",
            "3 reviewed models, raw judge labels",
            "7 other models",
        )
    )
    diff = T.loc[
        "Difference: 3 reviewed − 7 other models (interaction E_cited × reviewed)"
    ]
    per = T[T.index.str.endswith(" only")]
    low = (
        per[per["Population evidence use (E coef)"] < 0.15]
        .index.str.replace(" only", "")
        .tolist()
    )
    rev_names = [DISPLAY[m] for m in aci.REVIEWED_MODELS]
    l70 = T.loc["Llama 70B only"]
    r = {}
    r["tiers"] = (
        f"Reading: the two tiers differ because they contain different models, not because of the human review. "
        f"Tier 'all' has {int(a['Sampled texts'])} texts from 10 models, tier 'reviewed' {int(rv['Sampled texts'])} texts "
        f"from {', '.join(rev_names)}. On the same three models, the raw judge labels give practically the same "
        f"E_cited as the reviewed labels ({rj['E_cited coef']:.2f} vs {rv['E_cited coef']:.2f}) and an identical "
        f"result (the Q1-vs-Q2 share of chunks from the author's party uses no judge label at all, so it is identical "
        f"by construction); the review does not move these results. The difference comes from the model selection: "
        f"{' and '.join(m for m in low if m in rev_names)} are among the "
        f"{len(low)} models with almost no evidence use in the population (E coefficient < 0.15: {', '.join(low)}), "
        f"and two of the three reviewed models are of that kind. The seven other models give E_cited "
        f"{o['E_cited coef']:.2f} [{o['E_cited CI low']:.2f}, {o['E_cited CI high']:.2f}], the reviewed three "
        f"{rv['E_cited coef']:.2f} [{rv['E_cited CI low']:.2f}, {rv['E_cited CI high']:.2f}]. The two CIs of the "
        f"tiers overlap ({a['E_cited coef']:.2f} [{a['E_cited CI low']:.2f}, {a['E_cited CI high']:.2f}] vs "
        f"{rv['E_cited coef']:.2f}), but a direct test of reviewed vs. other models gives a difference of "
        f"{diff['E_cited coef']:+.2f} [{diff['E_cited CI low']:+.2f}, {diff['E_cited CI high']:+.2f}] "
        f"(p {diff['p, E_cited difference']:.3f}). The Q1-vs-Q2 difference in the share of chunks from the author's "
        f"party is likewise carried by the seven other models (p {o[P_KEY]:.3f}); among the "
        f"reviewed models, Llama 70B points the other way (Q1 {l70[Q1_KEY]:.2f} vs Q2 "
        f"{l70[Q2_KEY]:.2f}) and Qwen 3B has only {int(T.loc['Qwen 3B only', 'Q1 texts'])} "
        f"Q1 texts."
    )
    r["recommendation"] = (
        "Recommendation: report tier 'all' as the main result. It covers every model, so it describes the same "
        "population as the evidence-use analysis, and the comparison above shows that its labels are not worse in a "
        "way that matters: the human review changes E_cited by less than 0.01 on the reviewed models. As robustness "
        "checks, report (a) the reviewed models with raw judge vs. reviewed labels (isolates the review) and (b) the "
        "tier 'reviewed' with the note that it is a selection of models, two of which barely use evidence. The "
        "tier 'reviewed' should not be the main result, because its lower E_cited reflects which models were reviewed, "
        "not better labels. This recommendation rests on E_cited and the Q1/Q2 split; R_top-based results (E_relevant) "
        "remain the least reliable label and should still be read with the ±1 caveat of the judge chapter."
    )
    bp = by_party[
        (by_party["Llama 3B posts"] == "all") & (by_party["Grouping"] == "Chunk party")
    ].set_index("Group")
    sp = by_party[
        (by_party["Llama 3B posts"] == "all")
        & (by_party["Grouping"] == "Chunk from the author's party")
    ].set_index("Group")
    main_p = bp.drop(index=[x for x in ("BSW", NO_POSITION) if x in bp.index])
    ideo = main_p["Mean R_ideo"].sort_values(ascending=False)
    r["by_party"] = (
        f"Reading: speeches of the two outer parties are rated most ideological ({ideo.index[0]} "
        f"{ideo.iloc[0]:.2f}, {ideo.index[1]} {ideo.iloc[1]:.2f}), those of {ideo.index[-1]} ({ideo.iloc[-1]:.2f}) and "
        f"{ideo.index[-2]} ({ideo.iloc[-2]:.2f}) least; the CIs of the two outer parties do not overlap with those of "
        f"{ideo.index[-1]} and {ideo.index[-2]}. "
        f"The judge rates R_ideo without seeing the speaker's party, so this reflects the text. R_top differs less "
        f"between parties ({main_p['Mean R_top'].min():.2f}-{main_p['Mean R_top'].max():.2f}). AfD chunks are cited most "
        f"often ({bp.loc['AfD', 'Share cited (A_caus = 1)']:.2f}). Chunks from the author's own party are more relevant "
        f"({sp.loc['yes', 'Mean R_top']:.2f} vs {sp.loc['no', 'Mean R_top']:.2f}), more ideological "
        f"({sp.loc['yes', 'Mean R_ideo']:.2f} vs {sp.loc['no', 'Mean R_ideo']:.2f}) and cited more often "
        f"({sp.loc['yes', 'Share cited (A_caus = 1)']:.2f} vs {sp.loc['no', 'Share cited (A_caus = 1)']:.2f}). BSW (14 "
        f"chunks) and 'no position' (26) are too small to interpret."
    )
    rp = relp[relp["Llama 3B posts"] == "all"].set_index("Author party")
    sig = rp[(rp["Bootstrap p (Holm, across author parties)"] < 0.05)].index.tolist()
    allr = rp.loc["All posts"]
    r["relevance"] = (
        f"Reading: yes, but modestly. Of the relevant chunks (R_top ≥ 4), "
        f"{allr[REL_KEY] * 100:.0f} % come from the author's party, of the "
        f"others {allr[OTH_KEY] * 100:.0f} % (difference "
        f"{allr['Difference (relevant - other)']:+.2f} [{allr['Difference CI low']:+.2f}, {allr['Difference CI high']:+.2f}]). "
        f"After Holm correction across the author parties the difference is significant for "
        f"{', '.join(sig) if sig else 'no single party'}; the other parties point the same way with wide CIs."
    )
    ve = vs_err[
        (vs_err["Llama 3B posts"] == "all")
        & (vs_err["Weighting"] == "weighted by quadrant base rates")
    ]
    vb = ve[ve["Specification"] == "both"].set_index("Term")
    r["error"] = (
        f"Reading: higher labels go with a smaller RAG benefit, not a larger one. With both labels in the model "
        f"(weighted, input party and model controlled), one point of mean R_top goes with a "
        f"{vb.loc['mean R top', 'Coefficient (change in dAE per label point)']:+.3f} change in dAE "
        f"[{vb.loc['mean R top', 'CI low']:+.3f}, {vb.loc['mean R top', 'CI high']:+.3f}] and one point of mean R_ideo "
        f"with {vb.loc['mean R ideo', 'Coefficient (change in dAE per label point)']:+.3f} "
        f"[{vb.loc['mean R ideo', 'CI low']:+.3f}, {vb.loc['mean R ideo', 'CI high']:+.3f}]. The sample is selected on "
        f"dAE (Q1 and Q2 are defined by large changes, the buffer is missing), so this is a description of the error "
        f"cases, not an effect: relevant, ideological chunks are about as common in RAG distractions as in RAG wins, "
        f"which fits the Q1-vs-Q2 profiles in `chunk_influence_all/summary.md` (Q2 has the higher R_top)."
    )
    q = rpq[rpq["Llama 3B posts"] == "all"].set_index("Cited chunks in cell")
    ref = q.loc["any cited chunk (reference)", "Share Q1 (weighted)"]
    cells = q.drop(index="any cited chunk (reference)")[
        "Share Q1 (weighted)"
    ].sort_values()
    r["quadrant"] = (
        f"Reading: among Q1/Q2 texts with any cited chunk, {ref * 100:.0f} % are RAG wins (weighted). Texts that cite a "
        f"chunk from the author's party are slightly more often wins ("
        f"{q.loc['relevant × same party', 'Share Q1 (weighted)'] * 100:.0f} % relevant, "
        f"{q.loc['not relevant × same party', 'Share Q1 (weighted)'] * 100:.0f} % not relevant), texts that cite a "
        f"relevant chunk from another party slightly more often distractions "
        f"({q.loc['relevant × other party', 'Share Q1 (weighted)'] * 100:.0f} %). All CIs overlap with each other and "
        f"with the reference, and a text can fall into several cells, so this is a weak pattern: if anything, the party "
        f"of a cited chunk matters more than its relevance."
    )
    # Llama 3B restriction check
    key = ["Mean R_ideo", "Share cited (A_caus = 1)"]
    big = by_party[(by_party["Llama 3B posts"] == "all") & (by_party["Chunks"] >= 100)][
        ["Grouping", "Group"]
    ]
    b2 = by_party.merge(big, on=["Grouping", "Group"]).pivot_table(
        index=["Grouping", "Group"], columns="Llama 3B posts", values=key
    )
    dmax = max(
        (b2[(k, "all")] - b2[(k, "Llama 3B baseline >= 1")]).abs().max() for k in key
    )
    r["by_party"] += (
        f" Without the Llama 3B texts with a baseline score < 1, the values of the groups with at "
        f"least 100 chunks change by at most {dmax:.2f}."
    )
    return r


def cmd_labels(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Tier comparison (analyze_chunk_influence functions) ...")
    tiers = tier_comparison()
    d = direction_data()
    outs = defaultdict(list)
    ch_main = None
    for scope in LABEL_SCOPES:
        ch = label_chunks(d, scope)
        if scope == "all":
            ch_main = ch
        m = multiplicities(
            ch["text_index"].to_numpy(),
            None,
            args.n_boot,
            np.random.default_rng(args.seed),
        )
        outs["by_party"].append(labels_by_chunk_party(ch, scope, m))
        outs["relp"].append(labels_relevance_party(ch, scope, m))
        outs["vs_err"].append(labels_vs_error(ch, scope))
        outs["rpq"].append(
            labels_relevance_party_quadrant(ch, scope, args.n_boot, args.seed)
        )
    by_party, relp, vs_err, rpq = (
        pd.concat(outs[k], ignore_index=True)
        for k in ("by_party", "relp", "vs_err", "rpq")
    )
    for df, name in (
        (tiers, "labels_tier_comparison.csv"),
        (by_party, "labels_by_chunk_party.csv"),
        (relp, "labels_relevance_party.csv"),
        (vs_err, "labels_vs_error.csv"),
        (rpq, "labels_relevance_party_quadrant.csv"),
    ):
        print("  wrote", write_csv(df, name))
    write_labels_report(
        tiers,
        by_party,
        relp,
        vs_err,
        rpq,
        ch_main,
        args.n_boot,
        args.seed,
        labels_readings(tiers, by_party, relp, vs_err, rpq),
    )
    print("  wrote", OUT_DIR / "04_why_rag_labels.md")


# ===========================================================================
# explanatory: M1-M3 with crossed random effects for text and model
# ===========================================================================

OLD_FACTOR_CSV = (
    ROOT / "results/evaluation/2026-11-12_full/factor_analysis_label_ideology.csv"
)
OLD_M3_CSV = ROOT / em.THESIS_ROOT / "03_retrieval_configs" / "m3_coefficients.csv"
FAM = "C(family, Treatment('Llama'))"
PARTY = "C(party_group, Treatment('SPD'))"
DESIGN = [
    "C(strategy, Treatment('simple'))",
    "hybrid",
    "C(embedding_model, Treatment('bge'))",
]
SPECS = {
    "M1_family_size": f"{FAM} + size_c + rag + {PARTY} + {FAM}:rag + size_c:rag",
    "M2_time_size": f"time_c + size_c + rag + {PARTY} + time_c:rag + size_c:rag",
    "M3_rag_design": f"{FAM} + size_c + {PARTY} + " + " + ".join(DESIGN),
}


def term_label(t: str) -> str:
    rep = {
        "size_c": "size (per doubling)",
        "rag": "RAG",
        "time_c": "time (per year)",
        "hybrid": "hybrid (vs dense)",
        "Intercept": "intercept",
    }
    if t in rep:
        return rep[t]
    if ":" in t:
        return " × ".join(term_label(x) for x in t.split(":"))
    if "[T." in t:
        lvl = t.split("[T.")[1].rstrip("]")
        lvl = {
            "Mistral": "Ministral",
            "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
            "hyde": "HyDE",
            "twostage": "two-stage",
        }.get(lvl, lvl)
        ref = {
            "family": "Llama",
            "party_group": "SPD",
            "strategy": "simple",
            "embedding_model": "bge",
        }
        var = t.split("(")[1].split(",")[0].split(")")[0]
        return f"{lvl} (vs {ref.get(var, '?')})"
    return t


def explanatory_data(scope: str) -> pd.DataFrame:
    rows, _, _ = rca.load_all(em.BASE_DIR, None)
    v = rca.main_matrix(rows)
    # Target: CHES 2024 lrgen of the author's party, two decimals (chapter-04 convention).
    v["true"] = v["party"].map(lambda p: POSITIONS.get(aci.norm_party(p)))
    if v["true"].isna().any():
        raise SystemExit(
            f"Posts without a position: {v.loc[v['true'].isna(), 'party'].unique()}"
        )
    v["signed_error"] = v["pred"] - v["true"]
    v["abs_error"] = v["signed_error"].abs()
    if scope == SCOPE_GE1:
        l3 = v[
            (v["model"] == "Llama-3.2-3B-Instruct") & (~v["is_rag"]) & (v["pred"] < 1)
        ]["text_index"]
        v = v[
            ~((v["model"] == "Llama-3.2-3B-Instruct") & v["text_index"].isin(set(l3)))
        ]
    v["rag"] = v["is_rag"].astype(int)
    v["size_c"] = v["log2_size"] - v["log2_size"].mean()
    v["time_c"] = v["time_cov"] - v["time_cov"].mean()
    v["one"] = 1
    return v.reset_index(drop=True)


def _coef_rows(
    res, model, method, scope, n, extra=None, params=None, bse=None, ci_=None, pv=None
):
    params = res.fe_params if params is None else params
    bse = res.bse_fe if bse is None else bse
    ci_ = res.conf_int().loc[params.index] if ci_ is None else ci_
    pv = res.pvalues if pv is None else pv
    out = []
    for t in params.index:
        out.append(
            {
                "Model": model,
                "Method": method,
                "Llama 3B posts": scope,
                "Term": term_label(t),
                "Raw term": t,
                "Estimate": float(params[t]),
                "SE": float(bse[t]),
                "CI low": float(ci_.loc[t, 0]),
                "CI high": float(ci_.loc[t, 1]),
                "p": float(pv[t]),
                "N rows": n,
                **(extra or {}),
            }
        )
    return out


def fit_crossed(formula: str, data: pd.DataFrame, rag_slope: bool):
    import statsmodels.formula.api as smf

    vc = {"text": "0 + C(text_index)", "model": "0 + C(model)"}
    if rag_slope:
        vc["model_rag"] = "0 + C(model):rag"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = smf.mixedlm(formula, data, groups="one", re_formula="0", vc_formula=vc)
        res = m.fit(reml=True, method=["lbfgs"])
    conv = bool(res.converged) and not any(
        "Convergence" in type(w.message).__name__ for w in caught
    )
    vcomp = dict(zip(m.exog_vc.names, res.vcomp))
    return res, conv, vcomp


def two_stage(v: pd.DataFrame, spec: str, scope: str) -> List[dict]:
    """Stage 1: model fixed effects (and model-specific RAG effects) with a random
    intercept per text; stage 2: OLS of the 10 model effects on size / family / time,
    t-based CIs (small-sample inference for model-level terms)."""
    import statsmodels.formula.api as smf

    rows = []
    data = v if spec != "M3_rag_design" else v[v["is_rag"]]
    rhs1 = f"0 + C(model) + {PARTY}" + (
        " + C(model):rag" if spec != "M3_rag_design" else " + " + " + ".join(DESIGN)
    )
    res, conv, err = em._fit_lmm(f"abs_error ~ {rhs1}", data)
    if res is None:
        raise SystemExit(f"Stage 1 failed ({spec}): {err}")
    fe = res.fe_params
    meta = (
        v.groupby("model")
        .agg(
            log2_size=("log2_size", "first"),
            family=("family", "first"),
            time_cov=("time_cov", "first"),
        )
        .reset_index()
    )
    meta["size_c"] = meta["log2_size"] - meta["log2_size"].mean()
    meta["time_c"] = meta["time_cov"] - meta["time_cov"].mean()
    meta["level"] = [fe[f"C(model)[{m}]"] for m in meta["model"]]
    rhs2 = "size_c + " + (FAM if spec != "M2_time_size" else "time_c")
    targets = [("level", "")]
    if spec != "M3_rag_design":
        meta["rag_eff"] = [fe[f"C(model)[{m}]:rag"] for m in meta["model"]]
        targets.append(("rag_eff", ":rag"))
    for y, suffix in targets:
        fit = smf.ols(f"{y} ~ {rhs2}", meta).fit()
        ci_ = fit.conf_int()
        for t in fit.params.index:
            if t == "Intercept" and suffix:
                name = "rag"
            elif t == "Intercept":
                continue
            else:
                name = t + suffix
            rows.append(
                {
                    "Model": spec,
                    "Method": "two-stage (10 model effects, t-based CI)",
                    "Llama 3B posts": scope,
                    "Term": term_label(name),
                    "Raw term": name,
                    "Estimate": float(fit.params[t]),
                    "SE": float(fit.bse[t]),
                    "CI low": float(ci_.loc[t, 0]),
                    "CI high": float(ci_.loc[t, 1]),
                    "p": float(fit.pvalues[t]),
                    "N rows": len(data),
                    "df (stage 2)": int(fit.df_resid),
                    "Stage 1 converged": conv,
                }
            )
    return rows


def explanatory_fits(scope: str):
    v = explanatory_data(scope)
    coef, var = [], []
    for spec, rhs in SPECS.items():
        data = v if spec != "M3_rag_design" else v[v["is_rag"]].copy()
        print(f"  [{scope}] {spec}: old spec (text RE) ...")
        res, conv, err = em._fit_lmm(f"abs_error ~ {rhs}", data)
        coef += _coef_rows(
            res,
            spec,
            "old spec: random intercept per text",
            scope,
            len(data),
            {"Converged": conv},
        )
        print(f"  [{scope}] {spec}: crossed text + model ...")
        resc, convc, vcomp = fit_crossed(
            f"abs_error ~ {rhs}", data, rag_slope=spec != "M3_rag_design"
        )
        method = "crossed: text + model" + (
            " + model × RAG" if spec != "M3_rag_design" else ""
        )
        coef += _coef_rows(resc, spec, method, scope, len(data), {"Converged": convc})
        var.append(
            {
                "Model": spec,
                "Llama 3B posts": scope,
                "Method": method,
                "N rows": len(data),
                "N texts": data["text_index"].nunique(),
                "N models": data["model"].nunique(),
                **{f"Var {k}": float(val) for k, val in vcomp.items()},
                "Var residual": float(resc.scale),
                "Converged": convc,
                "Formula": f"abs_error ~ {rhs}",
            }
        )
        print(f"  [{scope}] {spec}: two-stage ...")
        coef += two_stage(v, spec, scope)
    return pd.DataFrame(coef), pd.DataFrame(var), v


def stored_old() -> pd.DataFrame:
    rows = []
    if OLD_FACTOR_CSV.exists():
        o = pd.read_csv(OLD_FACTOR_CSV)
        o = o[o["Outcome"] == "abs_error"]
        for _, r in o.iterrows():
            rows.append(
                {
                    "Model": r["Spec"],
                    "Method": "stored: evaluate_metrics 2026-11-12_full (label_ideology)",
                    "Llama 3B posts": SCOPE_ALL,
                    "Term": term_label(r["Term"].replace("is_rag_i", "rag")),
                    "Raw term": r["Term"],
                    "Estimate": r["Estimate"],
                    "SE": r["SE"],
                    "CI low": r["CI_lo"],
                    "CI high": r["CI_hi"],
                    "p": r["p"],
                }
            )
    if OLD_M3_CSV.exists():
        o = pd.read_csv(OLD_M3_CSV)
        o = o[(o["Outcome"] == "abs_error") & (o["Spec"] == "M3_rag_design")]
        for _, r in o.iterrows():
            rows.append(
                {
                    "Model": "M3_rag_design",
                    "Method": "stored: chapter 03 m3_coefficients (exact lrgen)",
                    "Llama 3B posts": SCOPE_ALL,
                    "Term": term_label(r["Term"]),
                    "Raw term": r["Term"],
                    "Estimate": r["Estimate"],
                    "SE": r["SE"],
                    "CI low": r["CI_lo"],
                    "CI high": r["CI_hi"],
                    "p": r["p"],
                }
            )
    return pd.DataFrame(rows)


def plot_design_factors(coef: pd.DataFrame):
    c = coef[
        (coef["Model"] == "M3_rag_design")
        & coef["Method"].str.startswith("crossed")
        & (coef["Llama 3B posts"] == SCOPE_ALL)
    ].set_index("Raw term")
    terms = (
        sorted([t for t in c.index if "embedding_model" in t])
        + sorted([t for t in c.index if "strategy" in t])
        + ["hybrid"]
    )
    labels = [c.loc[t, "Term"] for t in terms]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    for i, t in enumerate(terms):
        r = c.loc[t]
        ax.plot([r["CI low"], r["CI high"]], [i, i], color=COL_BASE, linewidth=1.4)
        ax.scatter(
            [r["Estimate"]],
            [i],
            s=26,
            zorder=3,
            color=COL_BASE,
            facecolor=COL_BASE if r["p"] < 0.05 else "white",
            linewidth=1.2,
        )
        ax.text(
            max(r["CI high"], 0) + 0.003,
            i,
            f"{r['Estimate']:+.3f}",
            va="center",
            fontsize=7,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(len(terms) - 0.5, -0.5)
    for k in (2.5, 4.5):
        ax.axhline(k, color="#e6e5e0", linewidth=0.7)
    ax.set_xlabel(
        "Change in absolute error (scale points), 95 % CI",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=7.5)
    fig.tight_layout()
    save_fig(fig, "fig_config_m3")


KEY_TERMS = [
    "size (per doubling)",
    "Ministral (vs Llama)",
    "Qwen (vs Llama)",
    "time (per year)",
    "RAG",
    "size (per doubling) × RAG",
    "Ministral (vs Llama) × RAG",
    "Qwen (vs Llama) × RAG",
    "time (per year) × RAG",
]


def write_explanatory_report(coef, var, v, readings):
    L = [
        "# Why retrieval helps or hurts: explanatory models with crossed random effects\n",
        "## Data\n",
    ]
    nm = v.groupby("model").size()
    L.append(
        f"Main evaluation matrix as in chapters 02/03 (`retrieval_config_analysis.main_matrix`: party_label logs, k = 5, "
        f"bge/hyde_hybrid excluded, duplicates resolved to the latest run, so the re-run Ministral-8B bge/twostage and "
        f"bge/twostage_hybrid cells of 2026-11-10 replace the superseded 2026-08-08 runs). {len(v)} rows "
        f"({int(v['is_rag'].sum())} RAG, {int((~v['is_rag']).sum())} no-RAG), {v['text_index'].nunique()} texts, "
        f"{v['model'].nunique()} models, {v['condition'].nunique()} conditions; rows per model "
        f"{nm.min()}-{nm.max()} (Qwen 72B has one condition fewer; rows with unparseable scores are dropped). Outcome: "
        f"absolute error against the CHES 2024 lrgen of the author's party, two decimals (the stored estimates used "
        f"label_ideology with one decimal resp. the exact lrgen). Size = log2 of the parameter count, centred, so its "
        f"coefficient is per doubling. Llama 3B: all posts and, in the CSV, without the texts whose Llama-3B baseline "
        f"score is < 1 (all conditions of these texts for Llama 3B).\n"
    )
    L.append("Methods compared:\n")
    L.append(
        "* **old spec**: the specification of `evaluate_metrics.compute_factor_analysis` (random intercept per "
        "text, ML), refitted on the current data;\n"
        "* **crossed**: REML with crossed variance components for text and model; M1 and M2 add a random RAG "
        "slope per model, because size × RAG and family × RAG are also between-model comparisons (statsmodels "
        "`MixedLM`, one group, `vc_formula`);\n"
        "* **two-stage**: stage 1 estimates one level (and one RAG effect) per model with a random intercept per "
        "text; stage 2 regresses the 10 model effects on size and family (M2: size and time) by OLS with t-based "
        "CIs (6 resp. 7 df). This is the small-sample check for the model-level terms;\n"
        "* **stored**: the estimates in `results/evaluation/2026-11-12_full/factor_analysis_label_ideology.csv` "
        "and `03_retrieval_configs/m3_coefficients.csv`.\n"
    )
    for spec in SPECS:
        L.append(f"## {spec}\n")
        c = coef[
            (coef["Model"] == spec)
            & (coef["Llama 3B posts"] == SCOPE_ALL)
            & coef["Term"].isin(
                KEY_TERMS
                + [t for t in coef["Term"].unique() if "(vs SPD)" in t]
                + [
                    t
                    for t in coef["Term"].unique()
                    if "(vs bge)" in t or "(vs simple)" in t or t == "hybrid (vs dense)"
                ]
            )
        ]
        c = c.assign(
            cell=[
                f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                for e, lo, hi in zip(c["Estimate"], c["CI low"], c["CI high"])
            ]
        )
        order = [
            m
            for m in (
                "stored: evaluate_metrics 2026-11-12_full (label_ideology)",
                "stored: chapter 03 m3_coefficients (exact lrgen)",
                "old spec: random intercept per text",
            )
            if m in set(c["Method"])
        ]
        order += sorted(
            [m for m in c["Method"].unique() if m.startswith("crossed")]
        ) + ["two-stage (10 model effects, t-based CI)"]
        piv = c.pivot_table(
            index="Term", columns="Method", values="cell", aggfunc="first"
        )
        piv = piv[[m for m in order if m in piv.columns]]
        terms = [t for t in KEY_TERMS if t in piv.index] + sorted(
            [t for t in piv.index if t not in KEY_TERMS]
        )
        piv = piv.loc[terms].fillna("")
        piv.columns = [
            {
                "stored: evaluate_metrics 2026-11-12_full (label_ideology)": "stored (11-12_full)",
                "stored: chapter 03 m3_coefficients (exact lrgen)": "stored (ch. 03)",
                "old spec: random intercept per text": "old spec, current data",
                "two-stage (10 model effects, t-based CI)": "two-stage",
            }.get(m, "crossed")
            for m in piv.columns
        ]
        L.append(em._md_table(piv.reset_index()) + "\n")
        L.append(readings[spec] + "\n")
    L.append("## Variance components (crossed fits)\n")
    vt = var[var["Llama 3B posts"] == SCOPE_ALL].drop(
        columns=["Llama 3B posts", "Formula"]
    )
    L.append(md_table(vt) + "\n")
    L.append(readings["var"] + "\n")
    L.append("## Design factors (M3, crossed)\n")
    L.append("![fig_config_m3.png](fig_config_m3.png)\n")
    L.append(
        "Figure: design-factor coefficients of M3 (crossed random effects for text and model), absolute error, "
        "95 % CI; filled = p < 0.05. Replaces the chapter-03 figure, which also showed size and family.\n"
    )
    L.append(readings["design"] + "\n")
    (OUT_DIR / "04_why_rag_explanatory.md").write_text("\n".join(L), encoding="utf-8")


def explanatory_readings(coef, var) -> Dict[str, str]:
    """Readings: per model, which model-level terms lose significance once the model
    random effects are added, the CI widening, the two-stage check and the Llama-3B
    restriction."""

    def get(spec, method_prefix, term, scope=SCOPE_ALL):
        g = coef[
            (coef["Model"] == spec)
            & coef["Method"].str.startswith(method_prefix)
            & (coef["Term"] == term)
            & (coef["Llama 3B posts"] == scope)
        ]
        return g.iloc[0] if len(g) else None

    def sig(r):
        return r is not None and (r["CI low"] > 0 or r["CI high"] < 0)

    def fmt(r):
        return f"{r['Estimate']:+.3f} [{r['CI low']:+.3f}, {r['CI high']:+.3f}]"

    r = {}
    for spec in SPECS:
        terms = [t for t in KEY_TERMS if get(spec, "crossed", t) is not None]
        lost = [
            t
            for t in terms
            if sig(get(spec, "old spec", t)) and not sig(get(spec, "crossed", t))
        ]
        kept = [t for t in terms if sig(get(spec, "crossed", t))]
        widen = [
            (get(spec, "crossed", t)["CI high"] - get(spec, "crossed", t)["CI low"])
            / (get(spec, "old spec", t)["CI high"] - get(spec, "old spec", t)["CI low"])
            for t in terms
        ]
        ts_agree = all(
            sig(get(spec, "two-stage", t)) == sig(get(spec, "crossed", t))
            for t in terms
            if get(spec, "two-stage", t) is not None
        )
        ge1 = [
            t
            for t in terms
            if sig(get(spec, "crossed", t)) != sig(get(spec, "crossed", t, SCOPE_GE1))
        ]
        txt = (
            f"Reading: the point estimates hardly change (the design is nearly balanced), but the CIs of the "
            f"model-level terms become {(f'{min(widen):.0f}-{max(widen):.0f}' if round(min(widen)) != round(max(widen)) else f'about {min(widen):.0f}')} "
            f"times wider once model is a random effect. "
            f"Still clearly different from 0: {', '.join(f'{t} {fmt(get(spec, 'crossed', t))}' for t in kept) or 'none'}. "
            f"No longer significant: {', '.join(lost) or 'none'}. "
        )
        txt += (
            "The two-stage check (10 model effects, t-based CIs) agrees on every term. "
            if ts_agree
            else "The two-stage check differs on at least one term (see table); its CIs are wider still. "
        )
        if ge1:
            txt += (
                "Without the Llama-3B posts with a baseline score < 1, the significance changes for "
                + ", ".join(
                    f"{t} ({fmt(get(spec, 'crossed', t, SCOPE_GE1))})" for t in ge1
                )
                + ". "
            )
        else:
            txt += "Without the Llama-3B posts with a baseline score < 1 no term changes its significance. "
        if spec == "M1_family_size":
            rg, rg1 = get(spec, "crossed", "RAG"), get(
                spec, "crossed", "RAG", SCOPE_GE1
            )
            txt += (
                f"The average RAG effect for Llama (the reference family) at mean size is {fmt(rg)} on all posts "
                f"and {fmt(rg1)} without the sub-1 Llama-3B scores: the negative RAG coefficient comes largely from "
                f"those scores. The party terms are within-text contrasts and do not change."
            )
        if spec == "M2_time_size":
            txt += (
                "With 10 models, time and family are partly confounded (the Ministral-3 models are the newest), "
                "so the time coefficient should not be read separately from the family effect of M1."
            )
        if spec == "M3_rag_design":
            txt += (
                "The design factors vary within each model, so their estimates and CIs are practically the same "
                "in all methods; none is significant for the absolute error."
            )
        r[spec] = txt
    v = var[var["Llama 3B posts"] == SCOPE_ALL].set_index("Model")
    m1 = v.loc["M1_family_size"]
    r["var"] = (
        f"Reading: in M1 the model intercept variance is {m1['Var model']:.3f} and the variance of the "
        f"model-specific RAG effect {m1['Var model_rag']:.3f}, against {m1['Var text']:.3f} for texts and "
        f"{m1['Var residual']:.3f} residual. The between-model variation of the RAG effect (SD "
        f"{np.sqrt(m1['Var model_rag']):.2f}) is larger than the average RAG effect, which is why size × RAG "
        f"and family × RAG cannot be distinguished from 0 with 10 models."
    )
    d = coef[
        (coef["Model"] == "M3_rag_design")
        & coef["Method"].str.startswith("crossed")
        & (coef["Llama 3B posts"] == SCOPE_ALL)
    ]
    d = d[d["Term"].str.contains(r"\(vs bge\)|\(vs simple\)|hybrid", regex=True)]
    r["design"] = (
        f"Reading: all design-factor effects lie between {d['Estimate'].min():+.3f} and "
        f"{d['Estimate'].max():+.3f} scale points, with CIs that include 0 "
        f"({int(((d['CI low'] > 0) | (d['CI high'] < 0)).sum())} of {len(d)} exclude it). Embedding model, "
        f"retrieval strategy and hybrid search do not change the absolute error beyond ±0.03 points, an order "
        f"of magnitude less than one doubling of model size."
    )
    return r


def cmd_explanatory(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        # Rebuild report and figure from the saved CSVs (the fits take ~15 min).
        coef = pd.read_csv(OUT_DIR / "explanatory_models.csv")
        coef = pd.concat(
            [coef[~coef["Method"].str.startswith("stored")], stored_old()],
            ignore_index=True,
        )
        var = pd.read_csv(OUT_DIR / "explanatory_variance_components.csv")
        write_csv(coef, "explanatory_models.csv")
        plot_design_factors(coef)
        write_explanatory_report(
            coef, var, explanatory_data(SCOPE_ALL), explanatory_readings(coef, var)
        )
        print("  rebuilt report from CSVs")
        return
    allc, allv, v_main = [], [], None
    for scope in (SCOPE_ALL, SCOPE_GE1):
        c, vv, v = explanatory_fits(scope)
        allc.append(c)
        allv.append(vv)
        if scope == SCOPE_ALL:
            v_main = v
    coef = pd.concat(allc + [stored_old()], ignore_index=True)
    var = pd.concat(allv, ignore_index=True)
    write_csv(coef, "explanatory_models.csv")
    write_csv(var, "explanatory_variance_components.csv")
    plot_design_factors(coef)
    write_explanatory_report(coef, var, v_main, explanatory_readings(coef, var))
    print(
        "  wrote explanatory_models.csv, explanatory_variance_components.csv, fig_config_m3, 04_why_rag_explanatory.md"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "judge",
        help="LLM judge: agreement, review, transfer check, label distributions",
    )
    p.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    p.set_defaults(func=cmd_judge)
    p = sub.add_parser(
        "composition", help="party composition of the e5/simple retrievals vs. chance"
    )
    p.add_argument(
        "--kb",
        default=None,
        help="indexed chunk file (chunks.jsonl), or CSV with columns party,chunks (counted from it) "
        "or party,speeches (approximation). Default: the chunk file if present, else "
        "src/datasets/knowledge_base/kb_speech_counts.csv",
    )
    p.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    p.set_defaults(func=cmd_composition)
    p = sub.add_parser(
        "direction", help="direction of the RAG shift: left / right / centre, by party"
    )
    p.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    p.set_defaults(func=cmd_direction)
    p = sub.add_parser(
        "extremes",
        help="DIE LINKE and AfD posts: evidence use, evidence gap, judge labels, cases",
    )
    p.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    p.set_defaults(func=cmd_extremes)
    p = sub.add_parser(
        "labels",
        help="judge labels by chunk party, relevance, error change; tier comparison",
    )
    p.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    p.set_defaults(func=cmd_labels)
    p = sub.add_parser(
        "explanatory",
        help="M1-M3 refitted with crossed random effects for text and model",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild report/figure from the saved CSVs",
    )
    p.set_defaults(func=cmd_explanatory)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
