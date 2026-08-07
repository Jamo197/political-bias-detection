# Qualitative Analysis — Action Plan & Output Reference

This document describes every analysis module, what question it answers, what the
script produces, and how to run it. Use it as a roadmap for thesis write-up.

---

## Motivating Example

One concrete failure illustrates why all these analyses exist:

- **Target Text:** SPD politician (Karl Lauterbach) thanking firefighters in Cologne.
- **Retrieved Chunks:** 5 parliamentary speeches about the Ahrtal flood disaster
  from SPD, FDP, AfD, and CDU/CSU.
- **What Went Wrong:**
  1. **Topical Attraction** — The embedding matched on keywords like
     *"Feuerwehr"*, *"Katastrophe"*, *"Helfer"* rather than on political ideology.
  2. **Context Pollution** — The LLM cited CDU/CSU and AfD chunks to classify an
     SPD tweet (ground truth: **2.7 left**) as "centrist/conservative" (**4.5**).
  3. **Score Compression** — All five cosine similarities were between **0.9118**
     and **0.9155** (Δ = 0.0037) — too tight to distinguish helpful from
     misleading chunks.

---

## Module 1: Retrieval Score Dynamics

**Script:** `src/results/analyze_retrieval_scores.py`

**What we measure:** For every query, we look at the scores assigned to each
retrieved chunk (cosine similarity, RRF fusion score, or cross-encoder logit).
We compute per-query statistics (mean, variance, range = top1 − top5) and then
check: *Do higher retrieval scores correlate with lower prediction error?* The
answer tells us whether a simple score threshold (e.g., "only keep chunks with
similarity > 0.85") could filter out misleading context.

**The complication:** Different retrieval strategies produce scores on
incompatible scales — cosine similarity is ~[0.4, 1.0], RRF fusion is an
arbitrary rank-blend score, and cross-encoder logits are unbounded ~[−9, 10].
Raw scores are only compared within a score type; for cross-condition comparison
we min-max normalize per `(embedding_model, retrieval_mode)` group.

**Score type mapping:**

| Retrieval modes | Score type | Native range |
|---|---|---|
| `simple`, `hyde`, `hyde_hybrid` | cosine similarity | ~[0.4, 1.0] |
| `simple_hybrid` | RRF fusion | arbitrary |
| `twostage`, `twostage_hybrid` | cross-encoder logit | ~[−9, 10] |

**Outputs (inside `results/qualitative/retrieval_scores/<run_id>/`):**

| File | What it contains |
|---|---|
| `retrieval_score_per_query.csv` | One row per query: all score stats, prediction, ground truth, error |
| `retrieval_score_summary.csv` | Aggregated stats per `(embedding_model, retrieval_mode)` |
| `retrieval_score_correlations.csv` | Pearson & Spearman between mean score and absolute error (per condition + global + per score type) |
| `boxplot_all_scores_{cosine,rrf,cross}.png` | Box plots of raw chunk scores across conditions (one per score type) |
| `boxplot_score_{range,variance}_{type}.png` | Per-query range / variance box plots |
| `scatter_score_vs_error.png` | Raw mean score vs. absolute error (all conditions) |
| `boxplot_*_normalized.png` | Same box plots on min-max normalized scores (cross-type comparable) |
| `scatter_score_vs_error_normalized.png` | Normalized mean score vs. absolute error |

**Run:**
```
python src/results/analyze_retrieval_scores.py
python src/results/analyze_retrieval_scores.py --batch-run 2026-08-04_eval_matrix_20260804_191413
python src/results/analyze_retrieval_scores.py --embedding-models bge,jina --retrieval-modes twostage,twostage_hybrid
```

---

## Module 2: Retrieval Party Alignment

**Script:** `src/results/analyze_retrieval_party_alignment.py`

**What we measure:** What parties are actually being retrieved, and does having
the *same party* as the target text improve or hurt prediction accuracy? This
module has three sub-analyses:

### 2a. Party Distribution per Target Party

For each target party (SPD, CDU/CSU, AfD, FDP, Greens, Left), we look at the
proportion of retrieved chunks that belong to each party. A stacked bar chart
shows the pattern: e.g., when a CDU/CSU speech is the target, how often are
SPD chunks retrieved? This reveals *cross-party retrieval patterns* — especially
whether CDU/CSU acts as a "gravity attractor" across all target parties.

### 2b. In-Party vs. Out-Party Retrieval Impact

We split every query into two groups:
- **In-Party Match:** at least one retrieved chunk matches the target's party.
- **Out-Party Noise:** all retrieved chunks are from opposing parties.

We then compare MAE, Pearson r, and Spearman r between these two groups —
overall and for every retrieval condition. If In-Party Match has consistently
lower error, party-aligned retrieval is a strong predictor of accuracy.

### 2c. In-Party Ratio vs. Error

For each query we compute `inparty_ratio` (fraction of retrieved chunks that
match the target's party, 0.0 to 1.0) and correlate it with absolute error
using Pearson & Spearman. A scatter grid per condition shows the trend with
regression lines. The question: *Does a higher proportion of same-party chunks
monotonically reduce error?*

**Outputs (inside `results/qualitative/retrieval_party_alignment/<run_id>/`):**

| File | What it contains |
|---|---|
| `party_alignment_per_query.csv` | One row per query: target party, n_chunks, n_inparty, inparty_ratio, retrieval type, prediction, error |
| `party_distribution_overall.csv` | Crosstab: target party × retrieved party (proportions, all conditions pooled) |
| `party_distribution_by_condition.csv` | Same crosstab split by condition (long format) |
| `inparty_outparty_metrics.csv` | MAE, Pearson r, Spearman r per `(condition, retrieval_type)` + overall |
| `inparty_ratio_correlations.csv` | Pearson & Spearman between inparty_ratio and abs_error (global + per condition) |
| `party_distribution_stacked.png` | Overall stacked bar chart |
| `party_distribution_by_condition.png` | Per-condition grid of stacked bar charts |
| `inparty_ratio_by_condition.png` | Bar chart of mean in-party ratio per condition (colored by embedding model) |
| `inparty_vs_outparty_metrics.png` | Overall: MAE and Pearson r comparing In-Party vs. Out-Party |
| `inparty_outparty_by_condition.png` | Per-condition grouped bars (MAE + Pearson panels) |
| `scatter_inparty_ratio_vs_error.png` | Scatter grid: in-party ratio vs. abs_error per condition |

**Run:**
```
python src/results/analyze_retrieval_party_alignment.py
python src/results/analyze_retrieval_party_alignment.py --retrieval-modes simple,hyde
```

---

## Module 3: Error Taxonomy (Extreme Cases)

**Script:** `src/results/analyze_error_taxonomy.py`

**What we measure:** We isolate the worst and best predictions for manual
qualitative review. Within each `(run_id, model, embedding, retrieval_mode)`
group, we tag:
- **Worst 10%:** top 10% highest absolute error (plus any residual > 1.5)
- **Best 10%:** bottom 10% lowest absolute error (plus any residual < 0.2)

The goal is to build a qualitative failure taxonomy for the thesis — read
through a sample of extreme cases and categorize *why* the model failed or
succeeded (topical attraction, neutral target, off-topic context, etc.).

**Outputs (inside `results/qualitative/error_taxonomy/<run_id>/`):**

| File | What it contains |
|---|---|
| `error_taxonomy_per_query.csv` | All queries before binning (11 columns: metadata + prediction + residual) |
| `extreme_cases.jsonl` | Every worst/best case as a JSON object with nested chunk data (text, party, speaker, score) |
| `extreme_cases.csv` | Same data in flat CSV form (chunks expanded to columns) — ready for spreadsheet review |
| `summary.json` | Per-run summary: total samples, MAE, worst/best counts and preview text indices |
| `residual_distribution_overall.png` | Histogram of all residuals with threshold lines (0.2, 1.5, 10th/90th percentile) |
| `residual_distribution_by_condition.png` | Per-condition histogram grid |
| `extreme_cases_by_condition.png` | Grouped bar: worst vs. best counts per condition |
| `mae_by_condition.png` | MAE bar chart per condition (colored by embedding model) |

**Run:**
```
python src/results/analyze_error_taxonomy.py
python src/results/analyze_error_taxonomy.py --batch-run 2026-08-04_eval_matrix_20260804_191413
```

### Manual Coding Schema

For the thesis, take a stratified sample of ~100 cases (50 worst, 50 best) and
fill the `manual_failure_mode` column in `extreme_cases.csv` using these
categories:

| Category | Description |
|---|---|
| **Topical Attraction / Party Confusion** | Target + chunks match on topic keywords (e.g., "Feuerwehr") but chunks belong to opposing parties dragging the rating away from ground truth |
| **Neutral Target Drag** | Target text expresses generic sentiment/sympathy without clear policy stance; LLM over-interprets context to force a non-neutral score |
| **Off-Topic Context Pollution** | Retrieved chunks match neither topic nor ideology — purely distracting information |
| **Correct In-Party Anchoring (Success)** | Target + chunks match on both topic and party; chunks successfully anchor the correct score |
| **Out-of-Domain Stance Mismatch** | Target text uses informal/social media phrasing that doesn't align with formal parliamentary speech style |

---

## Module 4: Evaluation Metrics (All Conditions)

**Script:** `src/evaluate_metrics.py`

**What we measure:** This is the top-level summary — how well does each
(LLM model × retrieval condition) combination predict ideology scores overall?
It computes four standard regression metrics per group and compares every RAG
condition against a non-RAG baseline.

Unlike the other modules which analyze *why* retrieval helps or hurts, this
module answers the basic question: *Does RAG improve predictions compared to
no retrieval at all, and by how much?*

**Key design feature:** Non-RAG baseline records (`is_rag=False`) are included
as a `no_rag` condition and sorted to the top of every output — the first row
for each model is always the baseline, making it easy to compare.

**The four metrics:**

| Metric | Direction | What it tells you |
|---|---|---|
| **MAE** | lower = better | Average absolute error in ideology score (1–10 scale) |
| **RMSE** | lower = better | Like MAE but penalizes large errors more heavily |
| **Pearson r** | higher = better | Linear correlation between predicted and true scores |
| **Spearman ρ** | higher = better | Rank correlation — robust to non-linear relationships |

**Outputs (inside `results/evaluation/<run_id>/`):**

| File | What it contains |
|---|---|
| `evaluation_metrics_{target}.csv` | MAE, RMSE, Pearson r, Spearman ρ per (Model, Condition) — `no_rag` first within each model |
| `rag_delta_{target}.csv` | Delta from `no_rag` baseline for every RAG condition (ΔMAE, ΔRMSE, ΔPearson, ΔSpearman) |
| `mae_comparison_{target}.png` | Horizontal bar chart: MAE per condition, colored by model |
| `rmse_comparison_{target}.png` | Same for RMSE |
| `all_metrics_{target}.png` | 2×2 grid showing all four metrics side by side |
| `rag_delta_heatmap_{target}.png` | Heatmap per model: RAG improvement (green) vs. regression (red) |
| `scatter_predicted_vs_actual_{target}.png` | Per-condition scatter plot with identity line |

Each plot uses `(lower = better)` / `(higher = better)` labels on the x-axis
so the direction is immediately clear.

**Run:**
```
python src/evaluate_metrics.py
python src/evaluate_metrics.py --target label_economic
python src/evaluate_metrics.py --batch-run 2026-08-04_eval_matrix_20260804_191413
```

---

## Module 6 (Planned): Topical vs. Ideological Alignment

**Not yet implemented.** The hypothesis is that RAG primarily retrieves *topical
matches* rather than *ideological matches*. To test this:

1. Run **BERTopic** (or similar) across the full speech corpus to assign topic
   labels (e.g., *Disaster Relief*, *Tax Policy*, *Immigration*).
2. Measure **Topic Overlap Rate**: how often target and top-1 retrieved chunk
   share the same topic cluster.
3. Test the **Topical Attraction Hypothesis**: high topic overlap + opposing
   party match → highest error, proving that RAG retrieves topical twins rather
   than ideological anchors.

---

## Module 7 (Planned): LLM-as-a-Judge Evaluation

**Not yet implemented.** Use a larger LLM (e.g., Llama-3.1-70B or Mistral-Large)
to score the quality of retrieved chunks in the extreme-cases sample:

| Dimension | Scale | Question |
|---|---|---|
| Topical Relevance | 1–5 | Is the chunk about the same underlying topic? |
| Ideological Utility | 1–5 | Does the chunk help identify the author's political alignment? |
| Context Noise | 1–5 | Does the chunk contain conflicting political views? |

This would provide quantitative proof that vector search scores high on topical
relevance but low on ideological utility.

---

## Quick-Start Command Reference

All four scripts share the same CLI interface:

```
python src/results/analyze_{retrieval_scores,retrieval_party_alignment,error_taxonomy}.py [options]
python src/evaluate_metrics.py [options]
```

| Option | Default | Effect |
|---|---|---|
| `--base-dir` | `logs/batch_runs` | Where to find batch-run JSONL folders |
| `--output-root` | `results/qualitative/{script_name}` | Where to write results |
| `--run-id` | `YYYYMMDD_HHMMSS` | Names the output subfolder |
| `--batch-run` | (all) | Restrict to one batch-run folder |
| `--embedding-models` | (all) | Comma-separated filter, e.g. `bge,jina` |
| `--retrieval-modes` | (all) | Comma-separated filter, e.g. `simple,twostage` |

To see all conditions available in your data, run any script once — it prints
the list of discovered batch runs, embedding models, retrieval modes, and
conditions.

---

## Output Structure

Every script creates a single timestamped folder per execution:

```
results/
  evaluation/                      ← overall prediction metrics
    20260807_110403/
      evaluation_metrics_label_ideology.csv
      rag_delta_label_ideology.csv
      mae_comparison_label_ideology.png
      all_metrics_label_ideology.png
      rag_delta_heatmap_label_ideology.png
      scatter_predicted_vs_actual_label_ideology.png
  qualitative/
    retrieval_scores/               ← score dynamics & variance
      20260806_151455/
        retrieval_score_per_query.csv
        retrieval_score_summary.csv
        ...
    retrieval_party_alignment/      ← party composition & alignment
      20260806_151455/
        party_alignment_per_query.csv
        party_distribution_stacked.png
        ...
    error_taxonomy/                 ← extreme cases for manual coding
      20260806_153128/
        error_taxonomy_per_query.csv
        extreme_cases.jsonl
        extreme_cases.csv
        summary.json
        ...
```

Use `--run-id my-descriptive-name` to override the auto-timestamp.

Note: `evaluate_metrics.py` is at `src/evaluate_metrics.py` (top-level src/, not src/results/), while the other three are under `src/results/`.

---

## Common Data Sources

All scripts read the same `party_label_*.jsonl` files under
`logs/batch_runs/<folder>/`. Each file records one evaluation run for a specific
`(model, embedding_model, retrieval_mode)` combination. The analysis scripts
filter `is_rag: true` records and skip those without retrieved chunks. The
evaluation script (`evaluate_metrics.py`) keeps non-RAG baseline records as a
`no_rag` condition.

Current data inventory (as of August 2026):
- **11 batch runs** covering 16 conditions (15 RAG + no_rag baseline)
- **4 embedding models:** bge, e5, jina, qwen3
- **6 retrieval modes:** simple, hyde, simple_hybrid, hyde_hybrid, twostage,
  twostage_hybrid (bge uses all 6; e5/jina/qwen3 use simple, hyde, twostage)
- **7 LLM models:** Ministral-3-8B, Ministral-3-3B, Llama-3.1-8B, Qwen2.5-7B,
  llama-3.1-70b, mistral-large-2512, qwen-2.5-72b (last three only have no_rag
  baseline data in the current logs)
- **~35,000 total records** including baseline (evaluate_metrics); ~29,000–30,000
  RAG-specific records for the analysis scripts

---

## Key Findings So Far

### 1. The "Cosine Compression" Trap

Retrieval scores across e5 and jina are severely compressed (Δ < 0.01 between
Top-1 and Top-5), and similarity scores show near-zero correlation with
prediction error. Two-stage retrieval spreads scores slightly, but not enough
to serve as a quality signal.

**Thesis insight:** Dense embeddings vectorize text along *topical* axes, not
*ideological* axes. Two speeches about flood disaster response sit next to each
other regardless of party. Cosine distance indicates topical relevance, not
ideological utility. Standard score thresholding cannot filter misleading
context.

### 2. The Party Alignment Principle

Retrieving at least one chunk from the target's own party drops MAE from ~1.81
to ~1.19 (34% reduction) and increases Pearson r from ~0.35 to ~0.65. Out-party
context acts as toxic noise.

**Cross-retrieval pattern:** CDU/CSU text is retrieved ~25–31% of the time
across *all* target parties. Self-retrieval is highest for AfD (~34%), Left
(~32%), and Greens (~23%).

**Thesis insight:** CDU/CSU acts as an "attractor" in vector space because
mainstream governing parties use formal, institutional language that matches a
wide range of policy queries. When an LLM receives out-party chunks (especially
centrist CDU/CSU text), its bias estimate gets pulled toward the center.

### 3. Recommendations for Thesis Narrative

1. **Party-Aware Filtering:** Political RAG should filter the search space by
   target party (or political spectrum) before similarity search, rather than
   relying on pure dense vector retrieval.
2. **Critique of Standard Distance Metrics:** Score compression and near-zero
   score-to-error correlation directly challenge standard RAG evaluation
   techniques that rely on vector similarity cutoffs.
