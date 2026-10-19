"""evaluate_metrics.py

Single entry point for evaluating the batch-run logs (``logs/batch_runs``).

Sections
--------
1. Metrics per model x condition: MAE, RMSE, Pearson r, Spearman rho, Lin's CCC,
   mean signed error, calibration slope/intercept - all with bootstrap CIs
   (resampling texts).
2. Paired RAG-vs-no_rag deltas (dMAE, dRMSE, dr, drho, dCCC, d|bias|) with paired
   bootstrap CIs and Holm-corrected significance flags.
3. Bias direction (formerly ``analyze_bias_direction.py``): signed-error
   statistics, left/centric/right shares, per-party and per-true-bin signed error.
4. Group comparison across model family, size and time (release / knowledge cutoff).
5. Mixed-effects factor analysis (random intercept per text) on |error| and
   signed error.
6. Optional qualitative per-configuration reports (``--qualitative``).
7. Retrieval-depth (k) ablation for one model/strategy (``--k-ablation``): metrics
   per k, paired pairwise k-vs-k comparison and a trend plot (separate entry point,
   ``main_k_ablation``).
8. Retrieval quality: do the retrieved chunks come from the input party's political
   spectrum? Bloc Macro-F1, bloc precision@k and mean CHES party distance.

Scale: 1.0 (extreme left) -> 7.0 (extreme right); signed error = prediction - truth.
"""

import argparse
import functools
import itertools
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/evaluation")

ALL_TARGETS = ["label_ideology", "label_economic", "label_galtan"]
CENTRIC_THRESHOLD = 0.5  # |signed_error| <= 0.5 counts as centric
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 42
DEFAULT_K = 5  # k_chunks of the main evaluation matrix; other k = ablation runs

# Merge party spellings before per-party analysis (edit if CDU/CSU should be split).
PARTY_ALIASES = {"CDU": "CDU/CSU", "CSU": "CDU/CSU"}

# Retrieval quality (section 8): party position = mean(lrecon, galtan) of the
# CHES wave below (latest earlier wave if a party is missing), converted from the
# CHES 0-10 scale to the project 1-7 scale (see _ches_to_7).
# Blocs on the 1-7 scale: Left < 3.4 <= Centre < 5.2 <= Right (= 4 / 7 on 0-10).
CHES_PATH = Path("src/datasets/ground_truth/ches_germany_1999-2024.csv")
CHES_WAVE = 2024
BLOC_THRESHOLDS = (3.4, 5.2)
BLOC_NAMES = ["Left", "Centre", "Right"]

# Casefolded, whitespace-normalised party spellings -> canonical name. Anything
# not listed (fraktionslos, UNKNOWN, ...) is unmapped and excluded.
CHUNK_PARTY_ALIASES = {
    "die linke": "DIE LINKE",
    "linke": "DIE LINKE",
    "bündnis 90/die grünen": "BÜNDNIS 90/DIE GRÜNEN",
    "die grünen": "BÜNDNIS 90/DIE GRÜNEN",
    "grüne": "BÜNDNIS 90/DIE GRÜNEN",
    "spd": "SPD",
    "fdp": "FDP",
    "afd": "AfD",
    "cdu/csu": "CDU/CSU",
    "cdu": "CDU/CSU",
    "csu": "CDU/CSU",
    "bsw": "BSW",
}

# Model metadata for the family / size / time comparison. Keyed by cleaned model
# name (last path component of ``parameters.llm``). ``cutoff`` is the knowledge
# cutoff (YYYY-MM) and stays None when the vendor does not publish one; the time
# covariate then falls back to the release date (see ``_time_basis``).
# VERIFY every value against the model cards before using it in the thesis.
MODEL_META: Dict[str, Dict[str, Any]] = {
    "Qwen2.5-3B-Instruct": dict(family="Qwen", params_b=3, release="2024-09", cutoff=None),  # VERIFY
    "Qwen2.5-7B-Instruct": dict(family="Qwen", params_b=7, release="2024-09", cutoff=None),  # VERIFY
    "Qwen2.5-32B-Instruct": dict(family="Qwen", params_b=32, release="2024-09", cutoff=None),  # VERIFY
    "Qwen2.5-72B-Instruct-FP8-dynamic": dict(family="Qwen", params_b=72, release="2024-09", cutoff=None),  # VERIFY
    "Llama-3.2-3B-Instruct": dict(family="Llama", params_b=3, release="2024-09", cutoff="2023-12"),  # VERIFY
    "Llama-3.1-8B-Instruct": dict(family="Llama", params_b=8, release="2024-07", cutoff="2023-12"),  # VERIFY
    "Meta-Llama-3.1-70B-Instruct-FP8": dict(family="Llama", params_b=70, release="2024-07", cutoff="2023-12"),  # VERIFY
    "Ministral-3-3B-Instruct-2512": dict(family="Mistral", params_b=3, release="2025-12", cutoff=None),  # VERIFY
    "Ministral-3-8B-Instruct-2512": dict(family="Mistral", params_b=8, release="2025-12", cutoff=None),  # VERIFY
    "Ministral-3-14B-Instruct-2512": dict(family="Mistral", params_b=14, release="2025-12", cutoff=None),  # VERIFY
}

# ---------------------------------------------------------------------------
# Helper Functions & Utilities
# ---------------------------------------------------------------------------


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


@functools.lru_cache(maxsize=None)
def _norm_party(name: Optional[str]) -> Optional[str]:
    """Canonical party name for messy log spellings (NBSP, line breaks, case);
    None for fraktionslos / unknown parties."""
    if not isinstance(name, str):
        return None
    return CHUNK_PARTY_ALIASES.get(" ".join(name.split()).casefold())


def _ches_to_7(score_10):
    """CHES 0-10 -> project 1-7 scale (same formula as the ground-truth dataset,
    see src/datasets/DATASET_DOCUMENTATION.md)."""
    return 1 + (score_10 / 10) * 6


def load_party_positions(
    path: Path = CHES_PATH, wave: int = CHES_WAVE
) -> Dict[str, float]:
    """Party -> mean(lrecon, galtan) on the 1-7 scale from the CHES wave (or the
    latest earlier wave a party appears in). CDU and CSU are averaged into CDU/CSU."""
    ches = pd.read_csv(path)
    ches = ches[ches["year"] <= wave].sort_values("year")
    ches = ches.groupby("std_party").tail(1).copy()
    ches["party"] = ches["std_party"].map(_norm_party)
    ches["pos"] = _ches_to_7((ches["lrecon"] + ches["galtan"]) / 2)
    return ches.dropna(subset=["party"]).groupby("party")["pos"].mean().to_dict()


def _bloc(pos: float) -> int:
    lo, hi = BLOC_THRESHOLDS
    return 0 if pos < lo else (1 if pos < hi else 2)


def _model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "Qwen"
    elif "ministral" in name_lower or "mistral" in name_lower:
        return "Mistral"
    elif "llama" in name_lower:
        return "Llama"
    return "Other"


def _model_size_key(model_name: str) -> int:
    name_lower = model_name.lower()
    if "large" in name_lower:
        return 999
    match = re.search(r"(\d+)b", name_lower)
    return int(match.group(1)) if match else 999


def _decimal_year(ym: Optional[str]) -> float:
    if not isinstance(ym, str) or not ym:
        return np.nan
    year, month = ym.split("-")[:2]
    return int(year) + (int(month) - 1) / 12


def _size_bucket(params_b: float) -> str:
    if params_b <= 3:
        return "S (<=3B)"
    if params_b <= 14:
        return "M (7-14B)"
    return "L (32B+)"


def _meta_for(model: str) -> Dict[str, Any]:
    meta = dict(MODEL_META.get(model, {}))
    meta.setdefault("family", _model_family(model))
    meta.setdefault("params_b", float(_model_size_key(model)))
    meta.setdefault("release", None)
    meta.setdefault("cutoff", None)
    return meta


def _condition_order(conditions: List[str]) -> List[str]:
    """Return conditions sorted with 'no_rag' always first."""
    others = sorted(c for c in conditions if c != "no_rag")
    return ["no_rag"] + others


def _apply_condition_order(df: pd.DataFrame, col: str = "Condition") -> pd.DataFrame:
    """Convert condition column to categorical with no_rag first."""
    if col not in df.columns or df.empty:
        return df
    order = _condition_order(df[col].unique().tolist())
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def _bin_ideology(score: float) -> str:
    if score < 3.0:
        return "Left"
    elif score > 5.0:
        return "Right"
    return "Center"


def _describe_direction(mean_signed_error: float, p_value: float) -> str:
    if p_value >= 0.05:
        return "No systematic bias"
    return "Right-leaning" if mean_signed_error > 0 else "Left-leaning"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _round(x: float, nd: int = 4) -> float:
    return round(float(x), nd) if x is not None and not np.isnan(x) else np.nan


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def _summarise_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "") or ""
        out.append(
            {
                "rank": i,
                "party": chunk.get("party"),
                "speaker": chunk.get("speaker"),
                "score": chunk.get("score"),
                "date": chunk.get("date"),
                "snippet": text[:180] + "..." if len(text) > 180 else text,
            }
        )
    return out


def load_logs_from_directory(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
    keep_details: bool = False,
    models: Optional[List[str]] = None,
    split_by_k: bool = False,
) -> pd.DataFrame:
    """Walk base_dir recursively, read JSONL entries, flatten into a DataFrame,

    and deduplicate re-run entries to prevent sample size inflation.
    ``keep_details`` additionally stores text, justification and retrieved chunks
    (only needed for the qualitative reports; memory heavy).
    ``models`` keeps only these cleaned model names. ``split_by_k`` appends
    ``/k<k_chunks>`` to RAG condition names so runs that differ only in k are
    neither merged nor deduplicated against each other (k-ablation).
    """
    records: List[Dict] = []
    jsonl_files = sorted(base_dir.rglob("party_label_*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No party_label_*.jsonl files found under '{base_dir}'"
        )

    for fp in jsonl_files:
        run_folder = fp.relative_to(base_dir).parts[0]
        if batch_run and run_folder != batch_run:
            continue

        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue

                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue

                params = log.get("parameters", {})
                model_name = _clean_model_name(params.get("llm", "unknown"))
                if models and model_name not in models:
                    continue
                emb = params.get("embedding_model", "none")
                retrieval_mode = params.get("retrieval_mode", "unknown")
                is_rag = params.get("is_rag", True)

                # Determine condition name
                if (
                    not is_rag
                    or emb in (None, "none", "")
                    or retrieval_mode == "no_rag"
                ):
                    condition = "no_rag"
                    emb = "none"
                    retrieval_mode = "no_rag"
                    is_rag = False
                else:
                    condition = f"{emb}/{retrieval_mode}"
                    if split_by_k:
                        condition += f"/k{params.get('k_chunks', 0)}"

                if embedding_models and emb not in embedding_models:
                    continue
                if retrieval_modes and retrieval_mode not in retrieval_modes:
                    continue

                meta = log.get("input_metadata", {})
                gt = log.get("ground_truth", {})
                output = log.get("output", {})
                inputs = log.get("inputs", {})

                hybrid = bool(params.get("hybrid", False)) or retrieval_mode.endswith(
                    "_hybrid"
                )
                record = {
                    "run_id": log.get("run_id", "unknown"),
                    "batch_run": run_folder,
                    "model": model_name,
                    "llm_full": params.get("llm", "unknown"),
                    "llm_region": params.get("llm_region", "unknown"),
                    "embedding_model": emb,
                    "retrieval_mode": retrieval_mode,
                    "strategy": retrieval_mode.replace("_hybrid", ""),
                    "hybrid": int(hybrid),
                    "is_rag": bool(is_rag),
                    "condition": condition,
                    "text_index": str(meta.get("text_index", "")),
                    "party": meta.get("party", ""),
                    "speaker": meta.get("speaker", ""),
                    "source": meta.get("source", ""),
                    "predicted_bias": output.get("bias"),
                    "label_ideology": gt.get("label_ideology"),
                    "label_economic": gt.get("label_economic"),
                    "label_galtan": gt.get("label_galtan"),
                    "k_chunks": params.get("k_chunks", 0),
                    "n_chunks_retrieved": len(inputs.get("retrieved_chunks", [])),
                    "chunk_parties": [
                        c.get("party") for c in inputs.get("retrieved_chunks", [])
                    ],
                }
                if keep_details:
                    record["input_text"] = inputs.get("text")
                    record["justification"] = output.get("justification")
                    record["retrieved_chunks"] = _summarise_chunks(
                        inputs.get("retrieved_chunks", [])
                    )
                records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        print("Warning: Loaded DataFrame is empty.")
        return df

    # Deduplicate runs to prevent sample-size inflation from duplicated executions
    initial_len = len(df)
    df = df.drop_duplicates(subset=["model", "condition", "text_index"], keep="last")
    print(
        f"Loaded {len(df)} unique records (deduplicated from {initial_len}) across {len(jsonl_files)} files."
    )
    print(f"  Batch runs:       {sorted(df['batch_run'].unique())}")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
    print(f"  Conditions:       {sorted(df['condition'].unique())}")
    print(f"  Models:           {sorted(df['model'].unique())}")
    return df


def build_model_meta(models) -> Tuple[pd.DataFrame, str]:
    """Per-model metadata (family, size, release/cutoff, time covariate).
    The time covariate is the knowledge cutoff if known for all models, else the
    release date."""
    meta = pd.DataFrame(
        [{"model": m, **_meta_for(m)} for m in sorted(models)]
    ).rename(columns={"release": "release_date", "cutoff": "cutoff_date"})
    meta["log2_size"] = np.log2(meta["params_b"].astype(float))
    meta["size_bucket"] = meta["params_b"].astype(float).map(_size_bucket)
    meta["release_year"] = meta["release_date"].map(
        lambda s: s[:4] if isinstance(s, str) else "unknown"
    )
    meta["cutoff_year"] = meta["cutoff_date"].map(
        lambda s: s[:4] if isinstance(s, str) else np.nan
    )
    release_dec = meta["release_date"].map(_decimal_year)
    cutoff_dec = meta["cutoff_date"].map(_decimal_year)
    if cutoff_dec.notna().all():
        meta["time_cov"], basis = cutoff_dec, "knowledge cutoff"
    else:
        meta["time_cov"], basis = release_dec, "release date"
    return meta, basis


def prepare_target_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Numeric-coerce prediction/target, drop invalid rows, add derived columns.

    Adds: pred, true, signed_error, abs_error, true_bin, party_group, family,
    params_b, log2_size, size_bucket, release_year, cutoff_year, time_cov.
    """
    valid = df.dropna(subset=["predicted_bias", target_col]).copy()
    valid["pred"] = pd.to_numeric(valid["predicted_bias"], errors="coerce")
    valid["true"] = pd.to_numeric(valid[target_col], errors="coerce")
    valid = valid.dropna(subset=["pred", "true"])
    valid["predicted_bias"] = valid["pred"]
    valid[target_col] = valid["true"]

    dropped = len(df) - len(valid)
    if dropped > 0:
        print(
            f"Note: Omitted {dropped} rows with missing/non-numeric predictions "
            f"for target '{target_col}'."
        )

    valid["signed_error"] = valid["pred"] - valid["true"]
    valid["abs_error"] = valid["signed_error"].abs()
    valid["true_bin"] = valid["true"].map(_bin_ideology)
    valid["party_group"] = valid["party"].replace(PARTY_ALIASES)

    meta, basis = build_model_meta(valid["model"].unique())
    valid = valid.merge(
        meta.drop(columns=["release_date", "cutoff_date"]), on="model", how="left"
    )
    valid.attrs["time_basis"] = basis
    return valid


# ---------------------------------------------------------------------------
# Metric primitives (vectorised over the last axis -> same code for point
# estimates (1-D) and bootstrap samples (2-D))
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "MAE",
    "RMSE",
    "Pearson_r",
    "Spearman_rho",
    "CCC",
    "Mean_signed_error",
    "Calibration_slope",
    "Calibration_intercept",
]


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > 1e-12, a / np.where(b == 0, 1, b), np.nan)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da = a - a.mean(-1, keepdims=True)
    db = b - b.mean(-1, keepdims=True)
    cov = (da * db).mean(-1)
    return _safe_div(cov, np.sqrt((da**2).mean(-1) * (db**2).mean(-1)))


def _vec_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """All point metrics along the last axis. Correlation-type metrics are NaN
    for n < 3 or zero variance (matches the previous behaviour)."""
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    e = p - t
    n = t.shape[-1]
    mt, mp = t.mean(-1), p.mean(-1)
    vt = ((t - mt[..., None]) ** 2).mean(-1)
    vp = ((p - mp[..., None]) ** 2).mean(-1)
    cov = ((t - mt[..., None]) * (p - mp[..., None])).mean(-1)
    slope = _safe_div(cov, vt)
    out = {
        "MAE": np.abs(e).mean(-1),
        "RMSE": np.sqrt((e**2).mean(-1)),
        "Mean_signed_error": e.mean(-1),
        "CCC": _safe_div(2 * cov, vt + vp + (mt - mp) ** 2),
        "Calibration_slope": slope,
        "Calibration_intercept": mp - slope * mt,
    }
    if n >= 3:
        out["Pearson_r"] = _corr(t, p)
        out["Spearman_rho"] = _corr(rankdata(t, axis=-1), rankdata(p, axis=-1))
    else:
        out["Pearson_r"] = np.full(mt.shape, np.nan)
        out["Spearman_rho"] = np.full(mt.shape, np.nan)
    return out


_MOMENT_COLS = ["w", "ae", "e", "e2", "t", "p", "t2", "p2", "tp"]


def _row_moments(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    e = pred - true
    return np.column_stack(
        [np.ones_like(e), np.abs(e), e, e**2, true, pred, true**2, pred**2, true * pred]
    )


def _metrics_from_moments(S: np.ndarray) -> Dict[str, np.ndarray]:
    """Metrics from summed moments (additive over rows -> cheap cluster
    bootstrap for pooled groups). No Spearman (not additive)."""
    S = np.atleast_2d(S)
    w, ae, e, e2, t, p, t2, p2, tp = (S[:, i] for i in range(9))
    mt, mp = _safe_div(t, w), _safe_div(p, w)
    vt = _safe_div(t2, w) - mt**2
    vp = _safe_div(p2, w) - mp**2
    cov = _safe_div(tp, w) - mt * mp
    slope = _safe_div(cov, vt)
    return {
        "MAE": _safe_div(ae, w),
        "RMSE": np.sqrt(_safe_div(e2, w)),
        "Mean_signed_error": _safe_div(e, w),
        "Pearson_r": _safe_div(cov, np.sqrt(vt * vp)),
        "CCC": _safe_div(2 * cov, vt + vp + (mt - mp) ** 2),
        "Calibration_slope": slope,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if len(samples) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(
    stat_fn: Callable[..., np.ndarray],
    arrays: Sequence[np.ndarray],
    n_boot: int = DEFAULT_N_BOOT,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.05,
) -> Tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI. ``stat_fn`` receives the resampled arrays with
    shape (n_boot, n) and must return an array of shape (n_boot,)."""
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.asarray(stat_fn(*[a[idx] for a in arrays]))
    lo, hi = _percentile_ci(samples, alpha)
    return lo, hi, samples


def _add_ci(row: Dict[str, Any], name: str, samples: Optional[np.ndarray]):
    if samples is None:
        return
    lo, hi = _percentile_ci(samples)
    row[f"{name}_lo"] = _round(lo)
    row[f"{name}_hi"] = _round(hi)


def _boot_p_two_sided(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0 from bootstrap deltas."""
    s = samples[~np.isnan(samples)]
    p = 2 * min(
        (np.sum(s <= 0) + 1) / (len(s) + 1),
        (np.sum(s >= 0) + 1) / (len(s) + 1),
    )
    return _round(min(p, 1.0), 4)


def _holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return out.tolist()
    order = np.argsort(p[valid])
    m = valid.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[valid][order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[valid] = res
    return out.tolist()


# ---------------------------------------------------------------------------
# 1. Metrics per Model x Condition
# ---------------------------------------------------------------------------


def compute_evaluation_metrics(
    valid_df: pd.DataFrame,
    target_col: str = "label_ideology",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """MAE, RMSE, Pearson r, Spearman rho, CCC, mean signed error and
    calibration slope/intercept per (model, condition) with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    results = []
    for (model, condition), group in valid_df.groupby(["model", "condition"]):
        if len(group) == 0:
            continue
        y_true = group["true"].to_numpy(float)
        y_pred = group["pred"].to_numpy(float)
        point = _vec_metrics(y_true, y_pred)
        boot = None
        if n_boot > 0 and len(group) >= 3:
            idx = rng.integers(0, len(group), size=(n_boot, len(group)))
            boot = _vec_metrics(y_true[idx], y_pred[idx])

        row: Dict[str, Any] = {
            "Target": target_col,
            "Model": model,
            "Family": group["family"].iloc[0],
            "Condition": condition,
            "N_samples": len(group),
        }
        for name in METRIC_NAMES:
            row[name] = _round(point[name])
            _add_ci(row, name, None if boot is None else boot[name])
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = _apply_condition_order(results_df, "Condition")
    return results_df.sort_values(["Model", "Condition"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Paired RAG delta vs. no_rag
# ---------------------------------------------------------------------------

DELTA_METRICS = [
    ("ΔMAE (RAG-Base)", "MAE"),
    ("ΔRMSE (RAG-Base)", "RMSE"),
    ("ΔPearson_r (RAG-Base)", "Pearson_r"),
    ("ΔSpearman_rho (RAG-Base)", "Spearman_rho"),
    ("ΔCCC (RAG-Base)", "CCC"),
    ("Δ|Bias| (RAG-Base)", "Mean_signed_error"),  # on |mean signed error|
]


def _delta_arrays(mr: Dict, mb: Dict, key: str) -> np.ndarray:
    if key == "Mean_signed_error":
        return np.abs(mr[key]) - np.abs(mb[key])
    return mr[key] - mb[key]


def compute_pairwise_rag_delta(
    valid_df: pd.DataFrame,
    target_col: str = "label_ideology",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """RAG vs no_rag deltas on the paired text_index intersection between EACH
    RAG condition and the no_rag baseline. Paired bootstrap: the same resampled
    texts are used for baseline and RAG. Negative ΔMAE/ΔRMSE/Δ|Bias| = RAG better.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for model, model_df in valid_df.groupby("model"):
        baseline_df = model_df[model_df["condition"] == "no_rag"]
        if baseline_df.empty:
            continue
        model_rows = []

        for cond in [c for c in model_df["condition"].unique() if c != "no_rag"]:
            cond_df = model_df[model_df["condition"] == cond]
            merged = pd.merge(
                baseline_df[["text_index", "true", "pred"]],
                cond_df[["text_index", "pred"]],
                on="text_index",
                suffixes=("_base", "_rag"),
            )
            if len(merged) < 3:
                continue

            y_true = merged["true"].to_numpy(float)
            y_base = merged["pred_base"].to_numpy(float)
            y_rag = merged["pred_rag"].to_numpy(float)
            mb = _vec_metrics(y_true, y_base)
            mr = _vec_metrics(y_true, y_rag)

            boot_b = boot_r = None
            if n_boot > 0:
                idx = rng.integers(0, len(merged), size=(n_boot, len(merged)))
                boot_b = _vec_metrics(y_true[idx], y_base[idx])
                boot_r = _vec_metrics(y_true[idx], y_rag[idx])

            row: Dict[str, Any] = {
                "Target": target_col,
                "Model": model,
                "Family": model_df["family"].iloc[0],
                "Condition": cond,
                "N_paired": len(merged),
            }
            for col, key in DELTA_METRICS:
                row[col] = _round(_delta_arrays(mr, mb, key)[()])
                if boot_b is not None:
                    samples = _delta_arrays(boot_r, boot_b, key)
                    _add_ci(row, col, samples)
                    if key == "MAE":
                        row["ΔMAE_p_boot"] = _boot_p_two_sided(samples)
            model_rows.append(row)

        # Holm correction across the RAG conditions of this model
        if model_rows and "ΔMAE_p_boot" in model_rows[0]:
            adj = _holm([r["ΔMAE_p_boot"] for r in model_rows])
            for r, a in zip(model_rows, adj):
                r["ΔMAE_p_holm"] = _round(a, 4)
                r["ΔMAE_significant"] = bool(a < 0.05)
        rows.extend(model_rows)

    delta_df = pd.DataFrame(rows)
    if not delta_df.empty:
        delta_df = _apply_condition_order(delta_df, "Condition")
        delta_df = delta_df.sort_values(["Model", "Condition"]).reset_index(drop=True)
    return delta_df


# ---------------------------------------------------------------------------
# 3. Bias direction (signed error)
# ---------------------------------------------------------------------------


def compute_bias_direction(
    valid_df: pd.DataFrame, metrics_df: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    """Signed-error statistics per (model, condition): shares of left / centric /
    right-leaning predictions, t-test vs 0, skew, mean signed error per true bin.
    Mean signed error + CI are taken from ``metrics_df``."""
    rows = []
    for (model, condition), g in valid_df.groupby(["model", "condition"]):
        se = g["signed_error"].to_numpy(float)
        n = len(se)
        if n < 3:
            continue
        t_stat, p_val = stats.ttest_1samp(se, popmean=0.0)
        row: Dict[str, Any] = {
            "Target": target_col,
            "Model": model,
            "Family": g["family"].iloc[0],
            "Condition": condition,
            "N_samples": n,
            "Median_signed_error": _round(np.median(se)),
            "Std_signed_error": _round(np.std(se, ddof=1)),
            "Skewness": _round(stats.skew(se)),
            "Pct_left_leaning": _round((se < -CENTRIC_THRESHOLD).mean() * 100, 2),
            "Pct_centric": _round((np.abs(se) <= CENTRIC_THRESHOLD).mean() * 100, 2),
            "Pct_right_leaning": _round((se > CENTRIC_THRESHOLD).mean() * 100, 2),
            "t_stat": _round(t_stat, 3),
            "p_value": _round(p_val, 6),
        }
        row["Direction"] = _describe_direction(se.mean(), p_val)
        for b in ("Left", "Center", "Right"):
            sub = g.loc[g["true_bin"] == b, "signed_error"]
            row[f"MeanSE_true_{b}"] = _round(sub.mean()) if len(sub) else np.nan
            row[f"N_true_{b}"] = len(sub)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keep = [c for c in metrics_df.columns if c.startswith("Mean_signed_error")]
    out = out.merge(
        metrics_df[["Model", "Condition", *keep]].astype(
            {"Condition": str}, errors="ignore"
        ),
        on=["Model", "Condition"],
        how="left",
    )
    out = _apply_condition_order(out, "Condition")
    return out.sort_values(["Model", "Condition"]).reset_index(drop=True)


def compute_bias_by_party(
    valid_df: pd.DataFrame,
    target_col: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Mean signed error per (model, condition, party) with bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, condition, party), g in valid_df.groupby(
        ["model", "condition", "party_group"]
    ):
        se = g["signed_error"].to_numpy(float)
        row: Dict[str, Any] = {
            "Target": target_col,
            "Model": model,
            "Family": g["family"].iloc[0],
            "Condition": condition,
            "Party": party,
            "N_samples": len(se),
            "True_label_mean": _round(g["true"].mean()),
            "Mean_signed_error": _round(se.mean()),
            "Mean_abs_error": _round(g["abs_error"].mean()),
        }
        if n_boot > 0 and len(se) >= 5:
            lo, hi, _ = bootstrap_ci(lambda x: x.mean(-1), [se], n_boot, rng)
            row["Mean_signed_error_lo"], row["Mean_signed_error_hi"] = _round(lo), _round(hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _apply_condition_order(out, "Condition")
        out = out.sort_values(["Model", "Condition", "Party"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 4. Group comparison: family / size / time
# ---------------------------------------------------------------------------

GROUP_METRICS = [
    "MAE",
    "RMSE",
    "Mean_signed_error",
    "Pearson_r",
    "CCC",
    "Calibration_slope",
]


def compute_group_comparison(
    valid_df: pd.DataFrame,
    target_col: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Pooled metrics per model / family / size bucket / release year / cutoff
    year / family x size, separately for scope ``no_rag`` (pure model effect),
    ``rag`` (RAG rows only) and ``all``. CIs come from a cluster bootstrap that
    resamples *texts* (every text appears in many models/conditions)."""
    rng = np.random.default_rng(seed)
    texts = valid_df["text_index"].unique()
    pos = {t: i for i, t in enumerate(texts)}
    n_texts = len(texts)
    counts = (
        rng.multinomial(n_texts, np.full(n_texts, 1.0 / n_texts), size=n_boot).astype(float)
        if n_boot > 0
        else None
    )

    df = valid_df.reset_index(drop=True)
    df["family_size"] = df["family"] + " " + df["size_bucket"]
    df["text_pos"] = df["text_index"].map(pos)
    mom = _row_moments(df["true"].to_numpy(float), df["pred"].to_numpy(float))

    dims = [
        ("model", "model"),
        ("family", "family"),
        ("size_bucket", "size_bucket"),
        ("release_year", "release_year"),
        ("cutoff_year", "cutoff_year"),
        ("family x size", "family_size"),
    ]
    scopes = {
        "no_rag": ~df["is_rag"].to_numpy(bool),
        "rag": df["is_rag"].to_numpy(bool),
        "all": np.ones(len(df), dtype=bool),
    }

    rows = []
    for scope, smask in scopes.items():
        for dim, key in dims:
            sub = df[smask]
            if sub[key].isna().all():
                continue
            for level, g in sub.groupby(key):
                gm = mom[g.index.to_numpy()]
                total = gm.sum(0, keepdims=True)
                point = _metrics_from_moments(total)
                row: Dict[str, Any] = {
                    "Target": target_col,
                    "Scope": scope,
                    "Dimension": dim,
                    "Level": level,
                    "N_rows": len(g),
                    "N_models": g["model"].nunique(),
                    "Family": g["family"].iloc[0] if g["family"].nunique() == 1 else "",
                    "Params_B": g["params_b"].iloc[0] if g["model"].nunique() == 1 else np.nan,
                }
                boot = None
                if counts is not None:
                    per_text = np.zeros((n_texts, 9))
                    np.add.at(per_text, g["text_pos"].to_numpy(int), gm)
                    boot = _metrics_from_moments(counts @ per_text)
                for name in GROUP_METRICS:
                    row[name] = _round(point[name][0])
                    _add_ci(row, name, None if boot is None else boot[name])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Mixed-effects factor analysis
# ---------------------------------------------------------------------------


def _cat(col: str, ref: Optional[str], data: pd.DataFrame) -> str:
    if ref is not None and ref in set(data[col].unique()):
        return f"C({col}, Treatment('{ref}'))"
    return f"C({col})"


def _fit_lmm(formula: str, data: pd.DataFrame):
    """Random-intercept-per-text linear mixed model (ML, for AIC comparison)."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = smf.mixedlm(formula, data, groups=data["text_index"]).fit(
                reml=False, method=["lbfgs"]
            )
        except Exception as exc:  # noqa: BLE001 - logged, not fatal
            return None, False, str(exc)
    conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return res, bool(getattr(res, "converged", True)) and not conv_warn, ""


def compute_factor_analysis(
    valid_df: pd.DataFrame,
    target_col: str,
    seed: int = DEFAULT_SEED,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed models with a random intercept per text on ``abs_error`` and
    ``signed_error``.

    * M1_family_size: family + size + RAG (+ party, family:RAG, size:RAG), all rows
    * M2_time_size:   time covariate (cutoff or release) + size + RAG, all rows
    * M3_rag_design:  family + size + strategy + hybrid + embedding + party,
                      RAG rows only (strategy/embedding are undefined for no_rag)

    Returns (coefficients, model_summary).
    """
    df = valid_df.copy()
    df["is_rag_i"] = df["is_rag"].astype(int)
    df["size_c"] = df["log2_size"] - df["log2_size"].mean()
    df["time_c"] = df["time_cov"] - df["time_cov"].mean()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
        print(f"  [LMM] subsampled to {max_rows} rows")

    n_family = df["family"].nunique()
    has_rag = df["is_rag"].any()
    has_norag = (~df["is_rag"]).any()
    party = _cat("party_group", "SPD", df)
    fam = _cat("family", "Llama", df)

    specs: List[Tuple[str, pd.DataFrame, str]] = []
    rag_term = ["is_rag_i"] if has_rag and has_norag else []
    inter_size = ["size_c:is_rag_i"] if rag_term else []
    multi_family = n_family > 1
    m1 = ([fam] if multi_family else []) + ["size_c", *rag_term, party]
    if rag_term and multi_family:
        m1.append(f"{fam}:is_rag_i")
    m1 += inter_size
    specs.append(("M1_family_size", df, " + ".join(m1)))

    if df["time_c"].nunique() > 1:
        m2 = ["time_c", "size_c", *rag_term, party]
        if rag_term:
            m2 += ["time_c:is_rag_i", *inter_size]
        specs.append(("M2_time_size", df, " + ".join(m2)))

    rag_only = df[df["is_rag"]]
    if has_rag and rag_only["condition"].nunique() > 1:
        m3 = [fam] if n_family > 1 else []
        m3 += ["size_c", party]
        if rag_only["strategy"].nunique() > 1:
            m3.append(_cat("strategy", "simple", rag_only))
        if rag_only["hybrid"].nunique() > 1:
            m3.append("hybrid")
        if rag_only["embedding_model"].nunique() > 1:
            m3.append(_cat("embedding_model", None, rag_only))
        specs.append(("M3_rag_design", rag_only, " + ".join(m3)))

    coef_rows, summary_rows = [], []
    for outcome in ("abs_error", "signed_error"):
        for spec, data, rhs in specs:
            print(f"  [LMM] {target_col} | {outcome} ~ {spec} (n={len(data)}) …")
            res, converged, err = _fit_lmm(f"{outcome} ~ {rhs}", data)
            if res is None:
                print(f"    failed: {err}")
                summary_rows.append(
                    {"Target": target_col, "Outcome": outcome, "Spec": spec,
                     "N": len(data), "Converged": False, "Error": err}
                )
                continue
            ci = res.conf_int().loc[res.fe_params.index]
            for term in res.fe_params.index:
                coef_rows.append(
                    {
                        "Target": target_col,
                        "Outcome": outcome,
                        "Spec": spec,
                        "Term": term,
                        "Estimate": _round(res.fe_params[term]),
                        "SE": _round(res.bse_fe[term]),
                        "CI_lo": _round(ci.loc[term, 0]),
                        "CI_hi": _round(ci.loc[term, 1]),
                        "p": _round(res.pvalues[term], 6),
                    }
                )
            var_text = float(res.cov_re.iloc[0, 0])
            summary_rows.append(
                {
                    "Target": target_col,
                    "Outcome": outcome,
                    "Spec": spec,
                    "N": len(data),
                    "N_texts": data["text_index"].nunique(),
                    "Var_text": _round(var_text),
                    "Var_resid": _round(res.scale),
                    "ICC": _round(var_text / (var_text + res.scale)),
                    "LogLik": _round(res.llf, 2),
                    "AIC": _round(getattr(res, "aic", np.nan), 2),
                    "BIC": _round(getattr(res, "bic", np.nan), 2),
                    "Converged": converged,
                    "Formula": f"{outcome} ~ {rhs}",
                }
            )
    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def write_factor_analysis_report(
    coef_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    out_path: Path,
):
    meta, basis = build_model_meta(valid_df["model"].unique())
    lines = [f"# Factor analysis — {target_col}", ""]
    lines += [
        "Linear mixed models, random intercept per text (`text_index`), fitted by ML.",
        "Outcomes: `abs_error` (accuracy) and `signed_error` (pred − truth; positive = leans right).",
        "Reference levels: family = Llama, party = SPD (where present). `size_c` = centred log2(parameters); "
        f"`time_c` = centred {basis} in years.",
        "",
        "> **Caveat — family and time are nearly collinear.** With only a handful of models, "
        "'later knowledge cutoff / release' is almost identical to 'Mistral family' "
        "(see table below). M1 vs M2 therefore cannot separate a *time* effect from a *family* effect; "
        "treat the comparison as descriptive, not causal.",
        "",
    ]
    if meta is not None:
        show = meta[["model", "family", "params_b", "release_date", "cutoff_date", "time_cov"]].copy()
        show["time_cov"] = show["time_cov"].round(2)
        lines += ["## Models", "", _md_table(show), ""]
    if not summary_df.empty:
        cols = [c for c in ["Outcome", "Spec", "N", "N_texts", "ICC", "AIC", "BIC", "Converged"] if c in summary_df.columns]
        lines += ["## Model comparison", "", _md_table(summary_df[cols]), ""]
    for outcome in ("abs_error", "signed_error"):
        for spec in coef_df["Spec"].unique() if not coef_df.empty else []:
            sub = coef_df[(coef_df["Outcome"] == outcome) & (coef_df["Spec"] == spec)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["sig"] = np.where(sub["p"] < 0.001, "***", np.where(sub["p"] < 0.01, "**", np.where(sub["p"] < 0.05, "*", "")))
            lines += [f"## {outcome} — {spec}", "", _md_table(sub[["Term", "Estimate", "SE", "CI_lo", "CI_hi", "p", "sig"]]), ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Qualitative reports (opt-in; formerly analyze_bias_direction.py)
# ---------------------------------------------------------------------------


def _example_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "text_index": row.get("text_index"),
        "target_party": row.get("party"),
        "target_speaker": row.get("speaker"),
        "input_text": row.get("input_text"),
        "pred_bias": float(row["pred"]),
        "true_bias": float(row["true"]),
        "abs_error": round(float(row["abs_error"]), 4),
        "signed_error": round(float(row["signed_error"]), 4),
        "ideology_bin": row["true_bin"],
        "justification": row.get("justification"),
        "retrieved_chunks": row.get("retrieved_chunks") or [],
    }


def write_qualitative_reports(
    valid_df: pd.DataFrame, out_dir: Path, top_k: int, target_col: str
):
    """Per (model, condition): JSON + Markdown with signed-error statistics and
    the top-K most left-leaning / right-leaning / centric predictions."""
    if "justification" not in valid_df.columns:
        print("Skipping qualitative reports — logs were loaded without details.")
        return

    for (model, condition), g in valid_df.groupby(["model", "condition"]):
        records = [_example_record(r) for _, r in g.iterrows()]
        se = np.array([r["signed_error"] for r in records])
        n = len(se)
        if n < 3:
            continue
        t_stat, p_val = stats.ttest_1samp(se, popmean=0.0)
        by_signed = sorted(records, key=lambda x: x["signed_error"])
        examples = {
            f"Most Left-Leaning Errors (Top {top_k})": by_signed[:top_k],
            f"Most Right-Leaning Errors (Top {top_k})": list(reversed(by_signed[-top_k:])),
            f"Most Centric Predictions (Top {top_k})": sorted(
                records, key=lambda x: abs(x["signed_error"])
            )[:top_k],
        }
        bins = defaultdict(list)
        for r in records:
            bins[r["ideology_bin"]].append(r)

        family = g["family"].iloc[0]
        group_key = f"{model}__{condition}".replace("/", "_")
        path = out_dir / "qualitative" / family / group_key
        path.mkdir(parents=True, exist_ok=True)

        report = {
            "group": group_key,
            "target": target_col,
            "n_total": n,
            "signed_error": {
                "mean": round(float(se.mean()), 4),
                "median": round(float(np.median(se)), 4),
                "std": round(float(se.std(ddof=1)), 4),
                "min": round(float(se.min()), 4),
                "max": round(float(se.max()), 4),
                "skewness": round(float(stats.skew(se)), 4),
            },
            "t_test_against_zero": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_at_05": bool(p_val < 0.05),
            },
            "by_true_ideology_bin": {
                b: {
                    "count": len(rs),
                    "mean_signed_error": round(float(np.mean([r["signed_error"] for r in rs])), 4),
                    "mean_abs_error": round(float(np.mean([r["abs_error"] for r in rs])), 4),
                }
                for b, rs in bins.items()
            },
            "examples": {k: v for k, v in examples.items()},
        }
        with open(path / "bias_analysis.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        with open(path / "bias_analysis.md", "w", encoding="utf-8") as md:
            md.write(f"# Directional Bias Analysis\n\n**Group:** `{group_key}`  \n")
            md.write(f"**Target:** {target_col}  \n**Total Evaluations:** {n}  \n")
            md.write(f"**Centric Threshold:** ±{CENTRIC_THRESHOLD}\n\n---\n\n")
            md.write("## Signed-Error Summary\n\n| Statistic | Value |\n|---|---|\n")
            for k, v in report["signed_error"].items():
                md.write(f"| {k} | {v} |\n")
            verdict = (
                f"systematic **{'left' if se.mean() < 0 else 'right'}** bias"
                if p_val < 0.05
                else "no significant systematic bias"
            )
            md.write(
                f"\n> t = {round(float(t_stat), 3)}, p = {round(float(p_val), 4)} → {verdict}.\n\n"
            )
            md.write("## Mean Signed Error by True Ideology Bin\n\n")
            md.write("| True Ideology | Count | Mean Signed Error | Mean Abs Error |\n|---|---|---|---|\n")
            for b in ("Left", "Center", "Right"):
                s = report["by_true_ideology_bin"].get(b)
                if s:
                    md.write(f"| {b} | {s['count']} | {s['mean_signed_error']} | {s['mean_abs_error']} |\n")
            md.write("\n---\n\n")
            for title, items in examples.items():
                md.write(f"## {title}\n\n")
                for idx, r in enumerate(items, 1):
                    md.write(
                        f"### #{idx} | Signed Error: {r['signed_error']} "
                        f"(Pred: {r['pred_bias']} vs Truth: {r['true_bias']})\n"
                        f"- **Metadata:** Speaker: `{r['target_speaker']}` | Party: `{r['target_party']}` | Text ID: `{r['text_index']}`\n"
                        f"- **Input Text:**\n> {r['input_text']}\n\n"
                        f"- **Model Justification:**\n> {r['justification']}\n\n"
                    )
                    if r["retrieved_chunks"]:
                        md.write(f"**Retrieved Chunks (Top {len(r['retrieved_chunks'])}):**\n")
                        for c in r["retrieved_chunks"]:
                            md.write(
                                f"  - `[Rank {c['rank']}]` **{c['speaker']}** ({c['party']}, {c['date']}) | Score: `{c['score']}`\n"
                                f"    - *Snippet:* {c['snippet']}\n"
                            )
                    md.write("\n---\n\n")
    print(f"Saved qualitative reports under {out_dir / 'qualitative'}")


# ---------------------------------------------------------------------------
# 7. Retrieval-depth (k) ablation
# ---------------------------------------------------------------------------

K_DELTA_METRICS = [
    ("ΔMAE", "MAE"),
    ("ΔRMSE", "RMSE"),
    ("ΔPearson_r", "Pearson_r"),
    ("ΔSpearman_rho", "Spearman_rho"),
    ("ΔCCC", "CCC"),
    ("Δ|Bias|", "Mean_signed_error"),  # on |mean signed error|
]


def _k_label(k: int) -> str:
    return "no_rag" if int(k) == 0 else f"k={int(k)}"


def compute_k_ablation_metrics(
    valid_df: pd.DataFrame,
    target_col: str = "label_ideology",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Metrics per k (k = 0 is the no_rag baseline) with bootstrap CIs.

    ``Subset`` = ``all_available`` uses every text a k has (N may differ when a run
    is incomplete); ``common_texts`` restricts all k to the texts present for every
    k, so the rows are directly comparable."""
    df = valid_df.assign(condition=valid_df["k_chunks"].map(_k_label))
    common = set.intersection(*(set(g["text_index"]) for _, g in df.groupby("k_chunks")))
    k_of = dict(zip(df["condition"], df["k_chunks"].astype(int)))

    parts = []
    for subset, sub in [
        ("all_available", df),
        ("common_texts", df[df["text_index"].isin(common)]),
    ]:
        m = compute_evaluation_metrics(sub, target_col, n_boot, seed)
        m["Condition"] = m["Condition"].astype(str)
        m.insert(3, "K", m["Condition"].map(k_of))
        m.insert(0, "Subset", subset)
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["Subset", "Model", "K"]).reset_index(drop=True)


def compute_k_pairwise(
    valid_df: pd.DataFrame,
    target_col: str = "label_ideology",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """All k-vs-k comparisons on the paired text_index intersection of each pair.

    Deltas are (higher k) - (lower k): negative ΔMAE/ΔRMSE/Δ|Bias| = the larger k is
    better; positive ΔPearson/ΔSpearman/ΔCCC = the larger k is better. Paired
    bootstrap CIs (same resampled texts for both k); ΔMAE p-values are Holm-corrected
    across all pairs of a model."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, mdf in valid_df.groupby("model"):
        ks = sorted(mdf["k_chunks"].astype(int).unique())
        model_rows = []
        for k_lo, k_hi in itertools.combinations(ks, 2):
            lo = mdf[mdf["k_chunks"] == k_lo][["text_index", "true", "pred"]]
            hi = mdf[mdf["k_chunks"] == k_hi][["text_index", "pred"]]
            merged = pd.merge(lo, hi, on="text_index", suffixes=("_lo", "_hi"))
            if len(merged) < 3:
                continue
            y_true = merged["true"].to_numpy(float)
            y_lo = merged["pred_lo"].to_numpy(float)
            y_hi = merged["pred_hi"].to_numpy(float)
            m_lo, m_hi = _vec_metrics(y_true, y_lo), _vec_metrics(y_true, y_hi)

            boot_lo = boot_hi = None
            if n_boot > 0:
                idx = rng.integers(0, len(merged), size=(n_boot, len(merged)))
                boot_lo = _vec_metrics(y_true[idx], y_lo[idx])
                boot_hi = _vec_metrics(y_true[idx], y_hi[idx])

            row: Dict[str, Any] = {
                "Target": target_col,
                "Model": model,
                "K_low": k_lo,
                "K_high": k_hi,
                "Comparison": f"{_k_label(k_hi)} vs {_k_label(k_lo)}",
                "N_paired": len(merged),
            }
            for col, key in K_DELTA_METRICS:
                row[col] = _round(_delta_arrays(m_hi, m_lo, key)[()])
                if boot_lo is not None:
                    samples = _delta_arrays(boot_hi, boot_lo, key)
                    _add_ci(row, col, samples)
                    if key == "MAE":
                        row["ΔMAE_p_boot"] = _boot_p_two_sided(samples)
            model_rows.append(row)

        if model_rows and "ΔMAE_p_boot" in model_rows[0]:
            adj = _holm([r["ΔMAE_p_boot"] for r in model_rows])
            for r, a in zip(model_rows, adj):
                r["ΔMAE_p_holm"] = _round(a, 4)
                r["ΔMAE_significant"] = bool(a < 0.05)
        rows.extend(model_rows)
    return pd.DataFrame(rows)


def plot_k_ablation(metrics_df: pd.DataFrame, target_name: str, output_path: Path):
    """Metric vs. k (ordinal x-axis) with bootstrap CIs on the common-texts subset;
    the no_rag baseline (k = 0), if present, is drawn as a dashed reference line."""
    sub = metrics_df[metrics_df["Subset"] == "common_texts"]
    if sub.empty:
        return
    sns.set_theme(style="whitegrid")
    panels = [
        ("MAE", "MAE (lower = better)"),
        ("RMSE", "RMSE (lower = better)"),
        ("Pearson_r", "Pearson r (higher = better)"),
        ("CCC", "Lin's CCC (higher = better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, ylabel) in zip(axes.flatten(), panels):
        for model, g in sub.groupby("Model"):
            base = g[g["K"] == 0]
            rag = g[g["K"] > 0].sort_values("K")
            x = np.arange(len(rag))
            y = rag[metric].to_numpy(float)
            if f"{metric}_lo" in rag.columns:
                yerr = np.vstack([y - rag[f"{metric}_lo"], rag[f"{metric}_hi"] - y])
                ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=model)
            else:
                ax.plot(x, y, "o-", label=model)
            if not base.empty:
                ax.axhline(float(base[metric].iloc[0]), color="grey", linestyle="--",
                           linewidth=1, label="no_rag")
        ax.set_xticks(np.arange(len(rag)))
        ax.set_xticklabels(rag["K"].astype(int))
        ax.set_xlabel("k (retrieved chunks)")
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=8)
    n = int(sub["N_samples"].iloc[0])
    fig.suptitle(f"k-ablation (n = {n} common texts)  |  Target: {target_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def evaluate_k_ablation(
    df: pd.DataFrame, target_col: str, output_dir: Path, n_boot: int, seed: int
):
    print("\n" + "#" * 80)
    print(f"# K-ABLATION | TARGET: {target_col}")
    print("#" * 80)

    valid_df = prepare_target_frame(df, target_col)
    if valid_df["k_chunks"].nunique() < 2:
        print("Need at least two distinct k values — skipping.")
        return

    counts = valid_df.groupby("k_chunks").agg(
        N=("text_index", "nunique"), batch_run=("batch_run", lambda s: ", ".join(sorted(set(s))))
    )
    print(counts.to_string())
    if counts["N"].nunique() > 1:
        print(
            "WARNING: unequal N across k (incomplete run?). Headline comparison uses the "
            "'common_texts' subset; pairwise tests use each pair's own text intersection."
        )

    metrics_df = compute_k_ablation_metrics(valid_df, target_col, n_boot, seed)
    pair_df = compute_k_pairwise(valid_df, target_col, n_boot, seed)

    def _show(frame: pd.DataFrame) -> str:
        return frame[[c for c in frame.columns if not c.endswith(("_lo", "_hi"))]].to_string(index=False)

    print("\nMETRICS PER k (common texts: identical texts for every k)")
    print(_show(metrics_df[metrics_df["Subset"] == "common_texts"]))
    print("\nPAIRWISE k-vs-k (Δ = higher k - lower k; negative ΔMAE/ΔRMSE = higher k is better)")
    print(_show(pair_df))

    for name, frame in [
        (f"k_ablation_metrics_{target_col}.csv", metrics_df),
        (f"k_ablation_pairwise_{target_col}.csv", pair_df),
    ]:
        if not frame.empty:
            frame.to_csv(output_dir / name, index=False)
            print(f"  → {output_dir / name}")
    plot_k_ablation(metrics_df, target_col, output_dir / f"k_ablation_{target_col}.png")


def main_k_ablation(
    base_dir: Path = BASE_DIR,
    output_root: Path = OUTPUT_ROOT,
    run_id: Optional[str] = None,
    targets: Optional[List[str]] = None,
    model: str = "Ministral-3-14B-Instruct-2512",
    embedding_model: str = "e5",
    retrieval_mode: str = "simple",
    k_values: Optional[List[int]] = None,
    include_baseline: bool = True,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
):
    """Compare different k_chunks for one model + embedding + strategy. Logs of all
    batch runs under ``base_dir`` are scanned, so k values spread over several runs
    (e.g. k=5 from an older batch) are combined automatically. The no_rag run of the
    same model is added as k = 0 when ``include_baseline``."""
    targets = targets or ["label_ideology"]
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_k_ablation"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"k-ablation | model={model} emb={embedding_model} strategy={retrieval_mode}")
    print(f"Output directory: {output_dir.resolve()}\n")

    df = load_logs_from_directory(base_dir, models=[model], split_by_k=True)
    if df.empty:
        print("No valid records found. Exiting.")
        return
    keep = (df["embedding_model"] == embedding_model) & (df["retrieval_mode"] == retrieval_mode)
    if k_values:
        keep &= df["k_chunks"].isin(k_values)
    if include_baseline:
        keep |= ~df["is_rag"]
    df = df[keep]

    print("\nRetrieval quality per k …")
    print_party_blocs()
    rq_df = compute_retrieval_quality(df.assign(condition=df["k_chunks"].map(_k_label)), n_boot, seed)
    if not rq_df.empty:
        rq_df.insert(2, "K", rq_df["Condition"].astype(str).map(lambda c: int(c[2:])))
        rq_df = rq_df.sort_values(["Model", "K"]).reset_index(drop=True)
        show = [c for c in rq_df.columns if not c.endswith(("_lo", "_hi"))]
        print(rq_df[show].to_string(index=False))
        rq_df.to_csv(output_dir / "k_ablation_retrieval_quality.csv", index=False)
        print(f"  → {output_dir / 'k_ablation_retrieval_quality.csv'}")
        plot_retrieval_quality(rq_df, output_dir / "k_ablation_retrieval_quality.png")

    for target_col in targets:
        evaluate_k_ablation(df, target_col, output_dir, n_boot, seed)
    print(f"\nAll outputs written to: {output_dir.resolve()}\nDone.")


# ---------------------------------------------------------------------------
# 8. Retrieval quality (political spectrum of the retrieved chunks)
# ---------------------------------------------------------------------------


def _macro_f1(conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bloc F1 and macro-F1 from flattened 3x3 confusion matrices (..., 9),
    rows = input bloc, columns = chunk bloc. F1 = 2TP / (2TP + FP + FN); blocs that
    occur neither as input nor as chunk are NaN and left out of the macro mean."""
    C = conf.reshape(*conf.shape[:-1], 3, 3)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    f1 = _safe_div(2 * tp, C.sum(-1) + C.sum(-2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return f1, np.nanmean(f1, axis=-1)


def compute_retrieval_quality(
    df: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    positions: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Spectrum match of the retrieved chunks per (model, condition), RAG rows only.

    Each chunk "predicts" the input party's bloc (Left / Centre / Right, see
    BLOC_THRESHOLDS):
    * Macro_F1: bloc F1 averaged over blocs (plus F1_<bloc>).
    * Bloc_precision_at_k: mean per-text share of chunks from the input's bloc.
    * Mean_party_distance: mean |pos(input party) - pos(chunk party)| on the
      project 1-7 scale, i.e. comparable to MAE (graded; lower = better).
    Fraktionslos/unknown chunk parties are excluded (N_chunks_unmapped). CIs:
    cluster bootstrap over texts."""
    positions = positions or load_party_positions()
    rag = df[df["is_rag"]]
    if rag.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    texts = rag["text_index"].unique()
    tpos = {t: i for i, t in enumerate(texts)}
    counts = (
        rng.multinomial(len(texts), np.full(len(texts), 1.0 / len(texts)), size=n_boot).astype(float)
        if n_boot > 0
        else None
    )

    rows = []
    for (model, condition), g in rag.groupby(["model", "condition"]):
        conf = np.zeros((len(texts), 9))
        prec, dist = [], []
        n_chunks = n_unmapped = 0
        for text, party, chunks in zip(g["text_index"], g["party"], g["chunk_parties"]):
            ip = positions.get(_norm_party(party))
            if ip is None or not chunks:
                continue
            ib = _bloc(ip)
            cps = [positions.get(_norm_party(c)) for c in chunks]
            valid = [c for c in cps if c is not None]
            n_chunks += len(cps)
            n_unmapped += len(cps) - len(valid)
            if not valid:
                continue
            cbs = [_bloc(c) for c in valid]
            for cb in cbs:
                conf[tpos[text], 3 * ib + cb] += 1
            prec.append(np.mean([cb == ib for cb in cbs]))
            dist.append(np.mean([abs(c - ip) for c in valid]))
        if not prec:
            continue

        f1, macro = _macro_f1(conf.sum(0))
        prec, dist = np.asarray(prec), np.asarray(dist)
        row: Dict[str, Any] = {
            "Model": model,
            "Condition": condition,
            "N_texts": len(prec),
            "N_chunks": n_chunks,
            "N_chunks_unmapped": n_unmapped,
            "Macro_F1": _round(macro),
            "Bloc_precision_at_k": _round(prec.mean()),
            "Mean_party_distance": _round(dist.mean()),
        }
        if counts is not None:
            _add_ci(row, "Macro_F1", _macro_f1(counts @ conf)[1])
            for name, arr in (("Bloc_precision_at_k", prec), ("Mean_party_distance", dist)):
                lo, hi, _ = bootstrap_ci(lambda x: x.mean(-1), [arr], n_boot, rng)
                row[f"{name}_lo"], row[f"{name}_hi"] = _round(lo), _round(hi)
        for name, v in zip(BLOC_NAMES, f1):
            row[f"F1_{name}"] = _round(v)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _apply_condition_order(out, "Condition")
        out = out.sort_values(["Model", "Condition"]).reset_index(drop=True)
    return out


def print_party_blocs(positions: Optional[Dict[str, float]] = None):
    positions = positions or load_party_positions()
    print(f"Party positions (CHES {CHES_WAVE}, mean of lrecon & galtan, 1-7 scale) and blocs:")
    for party, pos in sorted(positions.items(), key=lambda kv: kv[1]):
        print(f"  {party:<24} {pos:5.2f}  {BLOC_NAMES[_bloc(pos)]}")


def _plot_retrieval_quality_heatmaps(
    rq_df: pd.DataFrame, output_path: Path, y_col: str, levels: List[str], models: List[str]
):
    sns.set_theme(style="white")
    panels = [
        ("Macro_F1", "Bloc Macro-F1 (higher = better)", "RdYlGn"),
        ("Bloc_precision_at_k", "Same-bloc share (higher = better)", "RdYlGn"),
        ("Mean_party_distance", "Mean party distance, 1-7 scale (lower = better)", "RdYlGn_r"),
    ]
    df = rq_df.assign(**{y_col: rq_df[y_col].astype(str)})
    fig, axes = plt.subplots(
        1, 3, figsize=(8 + 1.6 * len(models), max(5, 0.45 * len(levels) + 2.5)), sharey=True
    )
    for ax, (metric, title, cmap) in zip(axes, panels):
        pivot = df.pivot_table(index=y_col, columns="Model", values=metric, aggfunc="first")
        pivot = pivot.reindex(index=levels, columns=models)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.6}, annot_kws={"fontsize": 7})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes[0] else y_col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    fig.suptitle("Retrieval quality: political spectrum of retrieved chunks",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def plot_retrieval_quality(rq_df: pd.DataFrame, output_path: Path, y_col: str = "Condition"):
    """Macro-F1, bloc precision@k and mean party distance per condition. One model:
    horizontal bars with bootstrap CIs; several models: condition x model heatmaps."""
    if rq_df.empty:
        return
    levels = [str(c) for c in rq_df[y_col].astype(str).drop_duplicates()]
    models = sorted(rq_df["Model"].unique(), key=lambda m: (_model_family(m), _model_size_key(m)))
    if len(models) > 1:
        _plot_retrieval_quality_heatmaps(rq_df, output_path, y_col, levels, models)
        return
    sns.set_theme(style="whitegrid")
    height = 0.8 / len(models)
    colors = sns.color_palette("mako", len(models))
    panels = [
        ("Macro_F1", "Bloc Macro-F1 (higher = better)"),
        ("Bloc_precision_at_k", "Same-bloc share of chunks (higher = better)"),
        ("Mean_party_distance", "Mean party distance, 1-7 scale (lower = better)"),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(18, max(5, 0.25 * len(levels) * len(models) + 2)), sharey=True
    )
    for ax, (metric, title) in zip(axes, panels):
        for i, (model, color) in enumerate(zip(models, colors)):
            g = rq_df[rq_df["Model"] == model]
            g = g.set_index(g[y_col].astype(str)).reindex(levels)
            y = np.arange(len(levels)) + (i - (len(models) - 1) / 2) * height
            xerr = None
            if f"{metric}_lo" in g.columns:
                xerr = np.vstack([g[metric] - g[f"{metric}_lo"], g[f"{metric}_hi"] - g[metric]])
            ax.barh(y, g[metric], height=height, xerr=xerr, color=color, label=model,
                    error_kw={"elinewidth": 0.8, "capsize": 2})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yticks(np.arange(len(levels)))
        ax.set_yticklabels(levels)
        ax.invert_yaxis()
    axes[0].legend(title="Model", fontsize=8)
    fig.suptitle("Retrieval quality: political spectrum of retrieved chunks",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


def plot_metric_comparison(
    metrics_df: pd.DataFrame, metric: str, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    lower_better = metric in ("MAE", "RMSE")
    direction = "lower = better" if lower_better else "higher = better"

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

    fig, ax = plt.subplots(figsize=(12, max(6, len(cond_order) * 0.35)))

    sns.barplot(
        data=plot_df,
        x=metric,
        y="Condition",
        hue="Model",
        ax=ax,
        palette="mako",
        order=cond_order,
    )

    ax.set_title(
        f"{metric} by Condition and Model  |  Target: {target_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"{metric}  ({direction})", fontsize=12)
    ax.set_ylabel("Condition", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, output_path)


def plot_mae_rmse_combined(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    """Plots MAE and RMSE side-by-side in a single figure for easy error comparison."""
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, max(6, len(cond_order) * 0.45)), sharey=True
    )

    sns.barplot(
        data=plot_df,
        x="MAE",
        y="Condition",
        hue="Model",
        ax=ax1,
        palette="mako",
        order=cond_order,
    )
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("MAE (lower = better)", fontsize=11)
    ax1.set_ylabel("Condition", fontsize=11)
    ax1.legend(title="Model", fontsize=9)

    sns.barplot(
        data=plot_df,
        x="RMSE",
        y="Condition",
        hue="Model",
        ax=ax2,
        palette="mako",
        order=cond_order,
    )
    ax2.set_title("Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("RMSE (lower = better)", fontsize=11)
    ax2.set_ylabel("")
    ax2.legend(title="Model", fontsize=9)

    fig.suptitle(
        f"Error Metric Comparison (MAE & RMSE)  |  Target: {target_name}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_all_metrics_grid(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

    panels = [
        ("MAE", "Mean Absolute Error (MAE)", "MAE (lower = better)"),
        ("RMSE", "Root Mean Squared Error (RMSE)", "RMSE (lower = better)"),
        ("Pearson_r", "Pearson r", "Pearson r (higher = better)"),
        ("Spearman_rho", "Spearman ρ", "Spearman ρ (higher = better)"),
        ("CCC", "Lin's CCC", "CCC (higher = better)"),
        ("Calibration_slope", "Calibration slope", "slope (1 = perfectly calibrated)"),
    ]

    fig_height = max(14, len(cond_order) * 0.6)
    fig, axes = plt.subplots(3, 2, figsize=(18, fig_height))
    axes = axes.flatten()

    for ax, (metric, title, xlabel) in zip(axes, panels):
        sns.barplot(
            data=plot_df,
            x=metric,
            y="Condition",
            hue="Model",
            ax=ax,
            palette="mako",
            order=cond_order,
        )
        if metric == "Calibration_slope":
            ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Condition", fontsize=10)
        ax.legend(title="Model", fontsize=8)

    fig.suptitle(
        f"Model Evaluation Summary — Target: {target_name}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_model_family_heatmaps(
    delta_df: pd.DataFrame, target_name: str, output_dir: Path
):
    """Generates heatmaps per model family showing pairwise RAG deltas.

    Handles missing models and missing RAG conditions gracefully.
    Green always indicates RAG improvement across all 4 metric tiles.
    """
    if delta_df.empty:
        print("Skipping RAG delta heatmap — no delta data available.")
        return

    sns.set_theme(style="white")
    plot_df = delta_df.copy()
    plot_df["Family"] = plot_df["Model"].apply(_model_family)

    delta_metrics = [
        ("ΔMAE (RAG-Base)", "ΔMAE (Lower is Better)", True),
        ("ΔRMSE (RAG-Base)", "ΔRMSE (Lower is Better)", True),
        ("ΔPearson_r (RAG-Base)", "ΔPearson r (Higher is Better)", False),
        ("ΔSpearman_rho (RAG-Base)", "ΔSpearman ρ (Higher is Better)", False),
    ]

    families = sorted(plot_df["Family"].unique())

    for family in families:
        sub = plot_df[plot_df["Family"] == family]
        if sub.empty:
            continue

        model_variants = sorted(sub["Model"].unique(), key=_model_size_key)
        conditions = sorted(sub["Condition"].unique())

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                max(8, 2.5 * len(model_variants) + 2),
                max(7, 1.8 * len(conditions) + 1),
            ),
        )
        axes = axes.flatten()

        for ax, (col, title, invert) in zip(axes, delta_metrics):
            if col not in sub.columns or sub[col].dropna().empty:
                ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
                ax.set_title(title, fontsize=11, fontweight="bold")
                continue

            pivot = sub.pivot_table(
                index="Condition", columns="Model", values=col, aggfunc="first"
            )
            pivot = pivot.reindex(index=conditions, columns=model_variants)

            # Invert values for error metrics so negative delta displays green (improvement)
            display_data = -pivot if invert else pivot

            # Safe formatting for string matrix with NaNs
            map_func = pivot.map if hasattr(pivot, "map") else pivot.applymap
            annot_matrix = map_func(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")

            sns.heatmap(
                display_data,
                annot=annot_matrix,
                fmt="",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_facecolor("#f0f0f0")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Condition", fontsize=10)
            ax.set_xlabel("Model Variant", fontsize=10)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", fontsize=9
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        fig.suptitle(
            f"{family} Family — Pairwise RAG Delta vs. Baseline\nTarget: {target_name}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        save_path = output_dir / f"rag_delta_heatmap_{family.lower()}_{target_name}.png"
        _save(fig, save_path)


def plot_scatter_predicted_vs_actual(
    df: pd.DataFrame, target_col: str, output_path: Path
):
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    if valid_df.empty:
        return

    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    sns.set_theme(style="whitegrid")
    conditions = _condition_order(sorted(valid_df["condition"].unique().tolist()))
    n_cols = min(len(conditions), 4)
    n_rows = (len(conditions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), sharey=True
    )
    axes_flat = np.array(axes).flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        sub = valid_df[valid_df["condition"] == cond]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.scatterplot(
            data=sub,
            x=target_col,
            y="predicted_bias",
            hue="model",
            alpha=0.7,
            ax=ax,
            palette="tab10",
        )

        min_val = min(sub[target_col].min(), sub["predicted_bias"].min()) - 0.5
        max_val = max(sub[target_col].max(), sub["predicted_bias"].max()) + 0.5
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            linewidth=1,
            label="Identity (Perfect)",
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"Condition: {cond}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Ground Truth ({target_col})", fontsize=10)
        ax.set_ylabel("Predicted Bias Score", fontsize=10)
        ax.legend(title="Model", fontsize=8)

    for idx in range(len(conditions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Predicted Bias vs. Ground Truth ({target_col})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_signed_error_forest(
    group_df: pd.DataFrame, target_name: str, output_path: Path
):
    """Mean signed error per model (no_rag vs. RAG pooled), grouped by family
    and ordered by size. Right of zero = predictions lean right."""
    sub = group_df[(group_df["Dimension"] == "model") & group_df["Scope"].isin(["no_rag", "rag"])]
    if sub.empty:
        return
    sns.set_theme(style="whitegrid")
    sub = sub.copy()
    sub["order"] = sub["Family"].map({"Llama": 0, "Mistral": 1, "Qwen": 2}).fillna(3) * 1000 + sub["Params_B"].fillna(0)
    models = sub.sort_values("order")["Level"].drop_duplicates().tolist()
    ypos = {m: i for i, m in enumerate(models)}
    colors = {"no_rag": "#4C72B0", "rag": "#DD8452"}
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(models) + 1.5)))
    for scope, off in (("no_rag", -0.15), ("rag", 0.15)):
        s = sub[sub["Scope"] == scope]
        y = [ypos[m] + off for m in s["Level"]]
        has_ci = "Mean_signed_error_lo" in s.columns
        xerr = (
            [s["Mean_signed_error"] - s["Mean_signed_error_lo"], s["Mean_signed_error_hi"] - s["Mean_signed_error"]]
            if has_ci
            else None
        )
        ax.errorbar(s["Mean_signed_error"], y, xerr=xerr, fmt="o", color=colors[scope],
                    capsize=3, label="no_rag" if scope == "no_rag" else "RAG (pooled)")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Mean signed error (pred − truth)  ←  left | right  →", fontsize=11)
    ax.set_title(f"Directional bias by model  |  Target: {target_name}", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, output_path)


def plot_party_bias_heatmap(
    valid_df: pd.DataFrame, target_name: str, output_path: Path
):
    """Mean signed error per model x party for no_rag and RAG-pooled.

    Parties are ordered left -> right by their mean ground-truth score."""
    sns.set_theme(style="white")
    party_truth = valid_df.groupby("party_group")["true"].mean().sort_values()
    party_order = party_truth.index.tolist()
    short = {"BÜNDNIS 90/DIE GRÜNEN": "GRÜNE"}  # display only; CSVs keep full names
    party_labels = [f"{short.get(p, p)}\n({party_truth[p]:.1f})" for p in party_order]
    scopes = [("no_rag", ~valid_df["is_rag"]), ("RAG (pooled)", valid_df["is_rag"])]
    models = sorted(valid_df["model"].unique(), key=lambda m: (_model_family(m), _model_size_key(m)))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.5 * len(models) + 2)), sharey=True)
    vmax = valid_df["signed_error"].abs().quantile(0.9)
    for ax, (name, mask) in zip(axes, scopes):
        sub = valid_df[mask]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index="model", columns="party_group", values="signed_error", aggfunc="mean").reindex(index=models, columns=party_order)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                    xticklabels=party_labels)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Party (mean ground-truth score)  ←  left | right  →")
        ax.set_ylabel("")
    fig.suptitle(f"Mean signed error by model and party (red = predicted too far right)  |  {target_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def plot_calibration_slope_vs_size(
    group_df: pd.DataFrame, target_name: str, output_path: Path
):
    sub = group_df[(group_df["Dimension"] == "model") & group_df["Scope"].isin(["no_rag", "rag"])]
    sub = sub.dropna(subset=["Params_B", "Calibration_slope"])
    if sub.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, scope in zip(axes, ("no_rag", "rag")):
        for family, g in sub[sub["Scope"] == scope].groupby("Family"):
            g = g.sort_values("Params_B")
            ax.plot(g["Params_B"], g["Calibration_slope"], marker="o", label=family)
            if "Calibration_slope_lo" in g.columns:
                ax.fill_between(g["Params_B"], g["Calibration_slope_lo"], g["Calibration_slope_hi"], alpha=0.15)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (B, log scale)")
        ax.set_title("no_rag" if scope == "no_rag" else "RAG (pooled)", fontweight="bold")
    axes[0].set_ylabel("Calibration slope (pred ~ truth; <1 = compressed to centre)")
    axes[0].legend(title="Family")
    fig.suptitle(f"Calibration slope vs. model size  |  Target: {target_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def evaluate_target(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    n_boot: int,
    seed: int,
    run_mixed: bool,
    lmm_max_rows: Optional[int],
    qualitative: bool,
    top_k: int,
):
    print("\n" + "#" * 80)
    print(f"# TARGET: {target_col}")
    print("#" * 80)

    valid_df = prepare_target_frame(df, target_col)
    if valid_df.empty:
        print("No valid rows for this target — skipping.")
        return

    print("[2/5] Computing metrics, paired deltas, bias direction, group comparison …")
    metrics_df = compute_evaluation_metrics(valid_df, target_col, n_boot, seed)
    delta_df = compute_pairwise_rag_delta(valid_df, target_col, n_boot, seed)
    direction_df = compute_bias_direction(valid_df, metrics_df, target_col)
    party_df = compute_bias_by_party(valid_df, target_col, n_boot, seed)
    group_df = compute_group_comparison(valid_df, target_col, n_boot, seed)

    print()
    print("=" * 80)
    print(f"EVALUATION METRICS (Target: {target_col})")
    print("=" * 80)
    show_cols = [c for c in metrics_df.columns if not c.endswith(("_lo", "_hi"))]
    print(metrics_df[show_cols].to_string(index=False))
    print()
    if not delta_df.empty:
        print("=" * 80)
        print("PAIRWISE RAG vs. BASELINE DELTA")
        print("  ΔMAE / ΔRMSE / Δ|Bias|: RAG - Baseline (negative = RAG is better)")
        print("  ΔPearson / ΔSpearman / ΔCCC (positive = RAG is better)")
        print("=" * 80)
        show = [c for c in delta_df.columns if not c.endswith(("_lo", "_hi"))]
        print(delta_df[show].to_string(index=False))
        print()

    print("[3/5] Saving CSV outputs …")
    for name, frame in [
        (f"evaluation_metrics_{target_col}.csv", metrics_df),
        (f"rag_delta_pairwise_{target_col}.csv", delta_df),
        (f"bias_direction_{target_col}.csv", direction_df),
        (f"bias_by_party_{target_col}.csv", party_df),
        (f"group_comparison_{target_col}.csv", group_df),
    ]:
        if not frame.empty:
            frame.to_csv(output_dir / name, index=False)
            print(f"  → {output_dir / name}")

    if run_mixed:
        print("[4/5] Mixed-effects factor analysis …")
        try:
            coef_df, summary_df = compute_factor_analysis(valid_df, target_col, seed, lmm_max_rows)
        except ImportError:
            print("  statsmodels not installed — skipping (pip install statsmodels).")
        else:
            if not coef_df.empty:
                coef_df.to_csv(output_dir / f"factor_analysis_{target_col}.csv", index=False)
            if not summary_df.empty:
                summary_df.to_csv(output_dir / f"factor_analysis_summary_{target_col}.csv", index=False)
            write_factor_analysis_report(
                coef_df, summary_df, valid_df, target_col,
                output_dir / f"factor_analysis_{target_col}.md",
            )
            print(f"  → {output_dir / f'factor_analysis_{target_col}.md'}")
    else:
        print("[4/5] Mixed-effects factor analysis skipped (--no-mixed).")

    print("[5/5] Generating plots …")
    plot_metric_comparison(metrics_df, "MAE", target_col, output_dir / f"mae_comparison_{target_col}.png")
    plot_metric_comparison(metrics_df, "RMSE", target_col, output_dir / f"rmse_comparison_{target_col}.png")
    plot_mae_rmse_combined(metrics_df, target_col, output_dir / f"mae_rmse_combined_{target_col}.png")
    plot_all_metrics_grid(metrics_df, target_col, output_dir / f"all_metrics_{target_col}.png")
    plot_model_family_heatmaps(delta_df, target_col, output_dir)
    plot_scatter_predicted_vs_actual(valid_df, target_col, output_dir / f"scatter_predicted_vs_actual_{target_col}.png")
    plot_signed_error_forest(group_df, target_col, output_dir / f"signed_error_forest_{target_col}.png")
    plot_party_bias_heatmap(valid_df, target_col, output_dir / f"party_bias_heatmap_{target_col}.png")
    plot_calibration_slope_vs_size(group_df, target_col, output_dir / f"calibration_slope_vs_size_{target_col}.png")

    if qualitative:
        write_qualitative_reports(valid_df, output_dir / target_col, top_k, target_col)


def main(
    base_dir: Path = BASE_DIR,
    output_root: Path = OUTPUT_ROOT,
    run_id: Optional[str] = None,
    targets: Optional[List[str]] = None,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    run_mixed: bool = True,
    lmm_max_rows: Optional[int] = None,
    qualitative: bool = False,
    top_k: int = 25,
    k_chunks: int = DEFAULT_K,
):
    targets = targets or ["label_ideology"]
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Model Evaluation Metrics")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Targets: {targets} | bootstrap: {n_boot or 'off'} (seed {seed})\n")

    print("[1/5] Loading batch-run JSONL files …")
    df = load_logs_from_directory(
        base_dir, batch_run, embedding_models, retrieval_modes,
        keep_details=qualitative, split_by_k=True,
    )
    if not df.empty:
        df["condition"] = df["condition"].str.replace(r"/k\d+$", "", regex=True)
    if df.empty:
        print("No valid records found. Exiting.")
        return
    # k-ablation runs share model + condition with the main matrix; without this
    # filter the dedup would silently replace the k = DEFAULT_K rows by them.
    other_k = df["is_rag"] & (df["k_chunks"] != k_chunks)
    if other_k.any():
        print(f"  Excluding {int(other_k.sum())} RAG rows with k_chunks != {k_chunks} (use --k-ablation for those).")
        df = df[~other_k]

    print("\nRetrieval quality (political spectrum of retrieved chunks) …")
    print_party_blocs()
    rq_df = compute_retrieval_quality(df, n_boot, seed)
    if not rq_df.empty:
        show = [c for c in rq_df.columns if not c.endswith(("_lo", "_hi"))]
        print(rq_df[show].to_string(index=False))
        rq_df.to_csv(output_dir / "retrieval_quality.csv", index=False)
        print(f"  → {output_dir / 'retrieval_quality.csv'}")
        plot_retrieval_quality(rq_df, output_dir / "retrieval_quality.png")

    for target_col in targets:
        evaluate_target(
            df, target_col, output_dir, n_boot, seed, run_mixed,
            lmm_max_rows, qualitative, top_k,
        )

    print(f"\nAll outputs written to: {output_dir.resolve()}")
    print("Done.")


def _comma_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate batch-run LLM bias prediction logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Root directory with batch run JSONL files.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Root directory for results.")
    parser.add_argument("--run-id", type=str, default=None, help="Name of output subfolder (default: timestamp).")
    parser.add_argument(
        "--target", default="label_ideology", choices=ALL_TARGETS, help="Ground-truth label column."
    )
    parser.add_argument(
        "--targets",
        type=_comma_list,
        default=None,
        help="Comma-separated targets or 'all' (overrides --target).",
    )
    parser.add_argument("--batch-run", type=str, default=None, help="Restrict analysis to a single batch run folder.")
    parser.add_argument("--embedding-models", type=_comma_list, default=None, help="Comma-separated embedding models to filter.")
    parser.add_argument("--retrieval-modes", type=_comma_list, default=None, help="Comma-separated retrieval modes to filter.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap resamples (default 2000).")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CIs (fast dev runs).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for bootstrap / subsampling.")
    parser.add_argument("--no-mixed", action="store_true", help="Skip mixed-effects factor analysis.")
    parser.add_argument("--lmm-max-rows", type=int, default=None, help="Subsample rows for the mixed models.")
    parser.add_argument("--qualitative", action="store_true", help="Write per-configuration bias reports with top-K examples.")
    parser.add_argument("--k", type=int, default=25, help="Examples per category for --qualitative (default 25).")
    parser.add_argument("--k-chunks", type=int, default=DEFAULT_K, help=f"k_chunks of the main evaluation (default {DEFAULT_K}); RAG rows with other k are excluded.")
    parser.add_argument("--k-ablation", action="store_true", help="Run only the k_chunks ablation (one model/embedding/strategy) instead of the full evaluation.")
    parser.add_argument("--k-model", type=str, default="Ministral-3-14B-Instruct-2512", help="Model (cleaned name) for --k-ablation.")
    parser.add_argument("--k-embedding", type=str, default="e5", help="Embedding model for --k-ablation.")
    parser.add_argument("--k-strategy", type=str, default="simple", help="Retrieval strategy for --k-ablation.")
    parser.add_argument("--k-values", type=lambda v: [int(x) for x in _comma_list(v)], default=None, help="Comma-separated k values to include (default: all found).")
    parser.add_argument("--k-no-baseline", action="store_true", help="Do not add the no_rag run (k = 0) to --k-ablation.")
    args = parser.parse_args()

    if args.targets and args.targets == ["all"]:
        chosen_targets = ALL_TARGETS
    else:
        chosen_targets = args.targets or [args.target]
    unknown = [t for t in chosen_targets if t not in ALL_TARGETS]
    if unknown:
        parser.error(f"Unknown target(s): {unknown}. Choose from {ALL_TARGETS}.")

    if args.k_ablation:
        main_k_ablation(
            base_dir=args.base_dir,
            output_root=args.output_root,
            run_id=args.run_id,
            targets=chosen_targets,
            model=args.k_model,
            embedding_model=args.k_embedding,
            retrieval_mode=args.k_strategy,
            k_values=args.k_values,
            include_baseline=not args.k_no_baseline,
            n_boot=0 if args.no_bootstrap else args.n_boot,
            seed=args.seed,
        )
        raise SystemExit(0)

    main(
        base_dir=args.base_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        targets=chosen_targets,
        batch_run=args.batch_run,
        embedding_models=args.embedding_models,
        retrieval_modes=args.retrieval_modes,
        n_boot=0 if args.no_bootstrap else args.n_boot,
        seed=args.seed,
        run_mixed=not args.no_mixed,
        lmm_max_rows=args.lmm_max_rows,
        qualitative=args.qualitative,
        top_k=args.k,
        k_chunks=args.k_chunks,
    )
