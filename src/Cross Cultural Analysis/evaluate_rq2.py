"""RQ2 analysis: metrics, condition contrasts and uncertainty for the C0-C5 runs.

Reads every ``*.jsonl`` with an ``rq2`` block under the given run directories
(written by ``src/run_rq2.py``) and writes CSV tables plus ``summary.md``:

* ``metrics.csv``      MAE, RMSE, directional accuracy, Pearson/Spearman and the
                       within-country Spearman of party means, per model x
                       country x condition x config x target x party_cue split.
* ``contrasts.csv``    paired differences in absolute error (same tweets) for
                       C2-C3 (RQ2b), C1-C0, C2-C1 (RQ2a), C5-C2, C2-C0, with a
                       bootstrap CI clustered by party, per country and pooled.
* ``c4_country_share.csv``  share of retrieved chunks from the target country (C4).
* ``family_correlation.csv``  within a CHES party family, correlation of predicted
                       party means with CHES scores across countries.
* ``mixed_model.csv``  abs_error ~ condition with a country random intercept
                       (needs ``statsmodels``; skipped otherwise).

Negative contrast values mean the first condition has the LOWER error.
Directional accuracy: predicted and CHES score fall on the same side of the
scale centre (4.0); tweets of parties exactly at 4.0 are left out.

Usage (from the repo root)::

    python "src/Cross Cultural Analysis/evaluate_rq2.py" logs/rq2_runs/<run> \\
        --out results/rq2 --target label_ideology
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_metrics import _calculate_array_stats, _clean_model_name  # noqa: E402

TARGETS = ["label_ideology", "label_economic", "label_galtan"]
CENTRE = 4.0
# (first, second, needs same config). C0/C1 have no retrieval config.
CONTRASTS = [
    ("C2", "C3", True),
    ("C1", "C0", False),
    ("C2", "C1", False),
    ("C5", "C2", True),
    ("C2", "C0", False),
]
CUE_SPLITS = {"all": None, "cue": True, "no_cue": False}
NO_RAG = {"C0", "C1"}


# --------------------------------------------------------------------------- loading
def load_runs(dirs: list[Path]) -> pd.DataFrame:
    records = []
    for base in dirs:
        for path in sorted(base.rglob("*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        log = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rq2 = log.get("rq2")
                    if not rq2:
                        continue
                    params = log.get("parameters", {})
                    chunks = log.get("inputs", {}).get("retrieved_chunks", [])
                    target = rq2["target_country"]
                    is_rag = params.get("is_rag", False)
                    records.append(
                        {
                            "model": _clean_model_name(params.get("llm", "unknown")),
                            "config": (
                                f"{params.get('embedding_model')}/{params.get('retrieval_mode')}"
                                if is_rag
                                else "none"
                            ),
                            "condition": rq2["condition"],
                            "country": target,
                            "text_id": str(log["input_metadata"]["text_index"]),
                            "party": rq2["ches_party_id"],
                            "party_name": rq2.get("canonical_party", ""),
                            "family": rq2.get("party_family", ""),
                            "party_cue": bool(rq2.get("party_cue", False)),
                            "pred": log.get("output", {}).get("bias"),
                            **{t: log.get("ground_truth", {}).get(t) for t in TARGETS},
                            "n_chunks": len(chunks),
                            "share_target": (
                                np.mean([c.get("country_code") == target for c in chunks])
                                if chunks
                                else np.nan
                            ),
                        }
                    )
    df = pd.DataFrame(records)
    if df.empty:
        sys.exit("No RQ2 log entries found.")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    before = len(df)
    df = df.drop_duplicates(["model", "config", "condition", "country", "text_id"], keep="last")
    print(f"Loaded {len(df)} predictions ({before - len(df)} duplicates dropped).")
    missing = df["pred"].isna().sum()
    if missing:
        print(f"  {missing} entries without a numeric prediction are ignored.")
    return df.dropna(subset=["pred"])


def _cue_subset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    flag = CUE_SPLITS[split]
    return df if flag is None else df[df["party_cue"] == flag]


# --------------------------------------------------------------------------- metrics
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != CENTRE
    if not mask.any():
        return np.nan
    return float(np.mean(np.sign(y_true[mask] - CENTRE) == np.sign(y_pred[mask] - CENTRE)))


def party_mean_spearman(group: pd.DataFrame, target: str) -> float:
    means = group.groupby("party").agg(pred=("pred", "mean"), true=(target, "first"))
    if len(means) < 3 or means["true"].nunique() < 2 or means["pred"].nunique() < 2:
        return np.nan
    return float(spearmanr(means["true"], means["pred"])[0])


def metrics_table(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        data = df.dropna(subset=[target])
        for split in CUE_SPLITS:
            sub = _cue_subset(data, split)
            for keys, group in sub.groupby(["model", "country", "condition", "config"]):
                y_true, y_pred = group[target].to_numpy(float), group["pred"].to_numpy(float)
                if len(group) == 0:
                    continue
                mae, rmse, pr, sr = _calculate_array_stats(y_true, y_pred)
                rows.append(
                    dict(
                        zip(["model", "country", "condition", "config"], keys),
                        target=target,
                        cue_split=split,
                        n=len(group),
                        n_parties=group["party"].nunique(),
                        mae=mae,
                        rmse=rmse,
                        directional_acc=directional_accuracy(y_true, y_pred),
                        pearson_r=pr,
                        spearman_rho=sr,
                        party_mean_spearman=party_mean_spearman(group, target),
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- contrasts
def _paired(df: pd.DataFrame, a: str, b: str, same_config: bool, target: str) -> pd.DataFrame:
    """One row per (model, country, config, text_id): abs error of a minus b.

    C0/C1 have no retrieval config ("none"); against a retrieval condition they
    are paired with every config of that condition.
    """
    data = df.dropna(subset=[target]).assign(abs_err=lambda d: (d["pred"] - d[target]).abs())
    keep = ["model", "country", "config", "text_id", "party", "party_cue", "abs_err"]
    left = data[data["condition"] == a][keep]
    right = data[data["condition"] == b][keep].drop(columns=["party", "party_cue"])
    on = ["model", "country", "text_id"]
    if same_config:
        on.append("config")
    elif b in NO_RAG:
        right = right.drop(columns="config")
    else:
        left = left.drop(columns="config")
    merged = left.merge(right, on=on, suffixes=("_a", "_b"))
    merged["diff"] = merged["abs_err_a"] - merged["abs_err_b"]
    return merged[["model", "country", "config", "text_id", "party", "party_cue", "diff"]]


def cluster_bootstrap(
    paired: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean paired difference with a 95% CI; parties resampled within each country."""
    if paired.empty:
        return np.nan, np.nan, np.nan
    # Per country: arrays of per-party (sum of diffs, count).
    strata = []
    for _, grp in paired.groupby("country"):
        agg = grp.groupby("party")["diff"].agg(["sum", "count"])
        strata.append((agg["sum"].to_numpy(), agg["count"].to_numpy()))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        total = count = 0.0
        for sums, counts in strata:
            idx = rng.integers(0, len(sums), len(sums))
            total += sums[idx].sum()
            count += counts[idx].sum()
        stats[i] = total / count
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(paired["diff"].mean()), float(lo), float(hi)


def contrast_table(df: pd.DataFrame, targets: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        for a, b, same in CONTRASTS:
            paired = _paired(df, a, b, same, target)
            if paired.empty:
                continue
            for split in CUE_SPLITS:
                sub = _cue_subset(paired, split)
                for (model, config), grp in sub.groupby(["model", "config"]):
                    scopes = [(c, g) for c, g in grp.groupby("country")] + [("pooled", grp)]
                    for country, g in scopes:
                        mean, lo, hi = cluster_bootstrap(g, n_boot, rng)
                        rows.append(
                            {
                                "contrast": f"{a}-{b}",
                                "target": target,
                                "cue_split": split,
                                "model": model,
                                "config": config,
                                "country": country,
                                "n": len(g),
                                "n_parties": g["party"].nunique(),
                                "mean_abs_err_diff": mean,
                                "ci_low": lo,
                                "ci_high": hi,
                                "significant": bool(lo > 0 or hi < 0),
                            }
                        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- C4 / families
def c4_share(df: pd.DataFrame) -> pd.DataFrame:
    c4 = df[(df["condition"] == "C4") & (df["n_chunks"] > 0)]
    if c4.empty:
        return pd.DataFrame()
    return (
        c4.groupby(["model", "config", "country"])
        .agg(n=("text_id", "size"), share_target=("share_target", "mean"))
        .reset_index()
    )


def family_correlation(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Within one family: do predicted party means follow CHES across countries?"""
    rows = []
    for target in targets:
        data = df.dropna(subset=[target])
        means = (
            data.groupby(["model", "config", "condition", "family", "country", "party"])
            .agg(pred=("pred", "mean"), true=(target, "first"))
            .reset_index()
        )
        for keys, grp in means.groupby(["model", "config", "condition", "family"]):
            if len(grp) < 3 or grp["country"].nunique() < 2 or grp["true"].std() == 0:
                continue
            rows.append(
                dict(
                    zip(["model", "config", "condition", "family"], keys),
                    target=target,
                    n_parties=len(grp),
                    n_countries=grp["country"].nunique(),
                    pearson_r=float(pearsonr(grp["true"], grp["pred"])[0]),
                    spearman_rho=float(spearmanr(grp["true"], grp["pred"])[0]),
                )
            )
    return pd.DataFrame(rows)


def mixed_models(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed: mixed model skipped (pip install statsmodels).")
        return pd.DataFrame()
    rows = []
    for target in targets:
        data = df.dropna(subset=[target]).assign(abs_err=lambda d: (d["pred"] - d[target]).abs())
        for (model, config), grp in data[data["config"] != "none"].groupby(["model", "config"]):
            sub = grp[grp["condition"].isin(["C2", "C3", "C4", "C5"])]
            if sub["country"].nunique() < 2 or sub["condition"].nunique() < 2:
                continue
            ref = "C3" if "C3" in set(sub["condition"]) else sorted(sub["condition"].unique())[0]
            formula = f"abs_err ~ C(condition, Treatment('{ref}'))"
            try:
                fit = smf.mixedlm(formula, sub, groups=sub["country"]).fit(reml=True)
            except Exception as e:  # noqa: BLE001 - few countries can fail to converge
                print(f"Mixed model failed for {model}/{config}/{target}: {e}")
                continue
            for term, coef in fit.params.items():
                if not term.startswith("C(condition"):
                    continue
                lo, hi = fit.conf_int().loc[term]
                rows.append(
                    {
                        "model": model,
                        "config": config,
                        "target": target,
                        "term": term.split("]")[0].split("T.")[-1] + f" vs {ref}",
                        "coef": coef,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p": fit.pvalues[term],
                        "n": len(sub),
                        "n_countries": sub["country"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def _md_table(table: pd.DataFrame) -> str:
    """Plain markdown table (avoids the optional ``tabulate`` dependency)."""
    named = any(name is not None for name in table.index.names)
    table = table.reset_index(drop=not named)
    cells = [[str(c) for c in table.columns]] + [
        ["" if pd.isna(v) else str(v) for v in row] for row in table.itertuples(index=False)
    ]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


def summary_md(metrics: pd.DataFrame, contrasts: pd.DataFrame, share: pd.DataFrame, target: str) -> str:
    out = [f"# RQ2 summary ({target})", ""]
    m = metrics[(metrics["target"] == target) & (metrics["cue_split"] == "all")]
    if not m.empty:
        pivot = m.pivot_table(
            index=["model", "country", "config"], columns="condition", values="mae"
        ).round(3)
        out += ["## MAE per condition", "", _md_table(pivot), ""]
    c = contrasts[(contrasts["target"] == target) & (contrasts["cue_split"] == "all")]
    if not c.empty:
        cols = ["contrast", "model", "config", "country", "n", "mean_abs_err_diff", "ci_low", "ci_high", "significant"]
        out += [
            "## Paired contrasts (negative = first condition better)",
            "",
            _md_table(c[cols].round(3)),
            "",
        ]
    if not share.empty:
        out += ["## C4: share of retrieved chunks from the target country", "", _md_table(share.round(3)), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/rq2"))
    parser.add_argument("--target", default="label_ideology", choices=TARGETS, help="Target for summary.md")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = load_runs(args.run_dirs)
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = metrics_table(df, TARGETS)
    contrasts = contrast_table(df, TARGETS, args.n_boot, args.seed)
    share = c4_share(df)
    families = family_correlation(df, TARGETS)
    mixed = mixed_models(df, TARGETS)

    for name, table in [
        ("metrics", metrics),
        ("contrasts", contrasts),
        ("c4_country_share", share),
        ("family_correlation", families),
        ("mixed_model", mixed),
    ]:
        if not table.empty:
            table.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "summary.md").write_text(summary_md(metrics, contrasts, share, args.target), encoding="utf-8")
    print(f"Wrote tables and summary.md to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
