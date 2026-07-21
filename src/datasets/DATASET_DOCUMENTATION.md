# Dataset Documentation: Political Social-Media Posts and Linked News Articles

Last reviewed: 2026-07-15

This document describes the current rebuilt dataset release in this repository. It is written for researchers who want to interpret or reuse the final CSV without reading the source code.

## Dataset Overview

### Purpose

The dataset links German politicians' social-media posts to the news articles shared in those posts. Each row is one post-article sample and contains:

- the politician's post text;
- the cleaned linked article text;
- the social-media platform and account handle;
- the politician's party;
- media and party ideology scores;
- one stance label describing the post's relation to the article; and
- three final stance-adjusted labels.

The current research-ready file is:

```text
V.1.2/merged_cleaned_dataset_with_stance.csv
```

Current file shape:

```text
820 rows
17 columns
```

Platform distribution in the current final file:

| Platform | Rows |
|---|---:|
| X | 725 |
| LinkedIn | 78 |
| Bluesky | 12 |
| Threads | 5 |

The final file intentionally contains 820 rows. During the rebuilt pipeline, one row without a usable media label was removed before final stance export.

### Data Sources

The rebuilt dataset combines three cleaned source datasets:

| Source component | Cleaned input used for merge |
|---|---|
| LinkedIn posts | `New_posts_LinkedIn/output/linkedin_cleaned_articles.csv` |
| X, Bluesky, and Threads posts | `New_posts_X_Bluesky_and_Threads/output/x_bluesky_threads_cleaned_articles.csv` |
| Older thesis dataset from Ben | `Old_posts_thesis_Ben/output/ben_old_posts_cleaned_standardized.csv` |

The merged pre-label file is:

```text
V.1.2/merged_cleaned_dataset.csv
```

The label-filled pre-stance file is:

```text
V.1.2/merged_cleaned_dataset_with_labels.csv
```

The final stance file is:

```text
V.1.2/merged_cleaned_dataset_with_stance.csv
```

## Final File Format

The final dataset is a standard CSV:

- Encoding: UTF-8.
- Delimiter: comma.
- Header row: present.
- Every row has 17 fields.
- The file loads with plain pandas:

```python
import pandas as pd

df = pd.read_csv("V.1.2/merged_cleaned_dataset_with_stance.csv")
```

The CSV is exported with:

```python
import csv

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL,
    lineterminator="\n",
)
```

## Data Generation Pipeline

### Step 1: Clean Each Source Collection

Each source folder has its own cleaning stage. The source-specific cleaning scripts standardize text fields, article URLs, article titles, outlet names, platform names, politician handles, and party fields.

The cleaned source outputs used for the rebuilt merge are:

- `New_posts_LinkedIn/output/linkedin_cleaned_articles.csv`
- `New_posts_X_Bluesky_and_Threads/output/x_bluesky_threads_cleaned_articles.csv`
- `Old_posts_thesis_Ben/output/ben_old_posts_cleaned_standardized.csv`

Each of these files is expected to be readable by `pd.read_csv()` before merging.

### Step 2: Merge Cleaned Datasets

The merge script is:

```text
merge_cleaned_datasets.py
```

It loads the three cleaned datasets, maps source-specific columns into the shared schema, creates missing target columns as empty values, concatenates rows, resets the `index` column, and writes:

```text
V.1.2/merged_cleaned_dataset.csv
```

No media labels, party labels, or stance labels are computed in this step.

### Step 3: Fill Media and Party Labels

The label-filling script is:

```text
fill_cleaned_dataset_labels.py
```

It fills these columns:

- `media_label`
- `party_label_ideology`
- `party_label_economic`
- `party_label_galtan`

It does not compute stance and does not modify article text, post text, URLs, or titles.

The output is:

```text
V.1.2/merged_cleaned_dataset_with_labels.csv
```

### Step 4: Media Ideology Scores

Media scores are assigned from `article_source`.

The current code uses:

```text
media_ideology_scores.py
```

The intended policy is:

1. Normalize the outlet name.
2. Look up a fixed continuous score from the German media table's `Parteilichkeit` values where available.
3. If the outlet is absent from that table, use the existing legacy integer/class fallback only where the code has an explicit fallback.

The resulting `media_label` is numeric. It can contain a mixture of:

- continuous media-table values such as `3.6`, `4.8`, `5.2`; and
- legacy integer-like values such as `1.0`, `2.0`, `3.0`.

The current final file also contains some `0.0` label values. These are part of the current artifact and should be treated cautiously as legacy or unresolved mapping values rather than continuous `Parteilichkeit` scores.

### Step 5: Party Ideology Scores

Party scores are assigned from `party`.

The current code uses:

```text
party_ideology_dimensions.py
```

The project supports three party-level dimensions:

| Final column | Meaning |
|---|---|
| `party_label_ideology` | General left-right party ideology score |
| `party_label_economic` | Economic left-right score |
| `party_label_galtan` | GAL-TAN score |

The latest fixed party tables are stored on a 1-10 scale and converted to the project 1-7 scale with:

```python
score_7 = 1 + ((score_10 - 1) / 9) * 6
```

Converted values are rounded to one decimal place.

The intended latest converted values are:

| Party | Ideology | Economic | GAL-TAN |
|---|---:|---:|---:|
| LINKE | 1.3 | 1.3 | 1.9 |
| Grüne | 2.4 | 2.6 | 1.4 |
| SPD | 2.7 | 2.7 | 2.7 |
| BSW | 2.7 | 2.2 | 5.1 |
| FDP | 4.3 | 5.4 | 2.5 |
| CDU | 4.7 | 4.7 | 4.7 |
| FW | 4.8 | 4.6 | 4.8 |
| CSU | 5.4 | 4.9 | 5.3 |
| AfD | 6.5 | 5.4 | 6.6 |

Party aliases are normalized in code. Examples include `Grüne`, `Bündnis 90 Die Grünen`, `Die Linke`, `LINKE`, `CDU`, `CSU`, `CDU/CSU`, `SPD`, `FDP`, `AfD`, `BSW`, and `FW`.

Note: the current final file still contains some legacy-looking values in `party_label_ideology`, including `0.0` and `1.0`. Researchers should verify whether they want to use the current artifact as-is or rerun the label and stance stages after confirming the intended mapping tables.

### Step 6: Stance Classification and Final Labels

The stance computation is implemented in:

```text
recompute_stance_labels.py
party_ideology_dimensions.py
```

The final dataset contains one stance value per row:

```text
pro
contra
neutral
```

The final labels are computed separately for the three party dimensions:

- `final_label_ideology`
- `final_label_economic`
- `final_label_galtan`

Let:

- \(m\) = `media_label`
- \(p\) = `party_label_ideology`
- \(e\) = `party_label_economic`
- \(g\) = `party_label_galtan`
- \(x\) = one of \(p\), \(e\), or \(g\)

For each dimension \(x\):

```text
if stance is pro or neutral:
    final_label_x = (x + m) / 2

if stance is contra:
    final_label_x = ((8 - x) + m) / 2
```

The `contra` case uses the mirrored 1-7 scale value `8 - x`.

The final score columns are rounded to one decimal in the current final export.

### Step 7: Final Schema Export

After stance computation, the final export keeps only the 17 public project columns. Internal helper columns, debugging columns, original source-specific columns, and intermediate stance flags are removed.

The current final file is:

```text
V.1.2/merged_cleaned_dataset_with_stance.csv
```

## Party-Media Compatibility and Stance Score Generation

Earlier versions of the project used a distance rule to decide whether a stance classifier was needed. In the current rebuilt final dataset, every retained row has a stance label.

The distance logic remains useful conceptually:

```text
party distance    = |party_label_ideology - media_label|
economic distance = |party_label_economic - media_label|
GAL-TAN distance  = |party_label_galtan - media_label|
```

However, the final public dataset does not keep the internal distance-helper columns.

### Decision Tree

```text
Start
│
├── Load one merged post-article row
│
├── Assign media score:
│     m = media_label
│
├── Assign party scores:
│     p = party_label_ideology
│     e = party_label_economic
│     g = party_label_galtan
│
├── Classify stance for the post/article pair:
│     stance ∈ {pro, contra, neutral}
│
└── Compute final labels:
      │
      ├── For ideology x = p
      │     ├── pro/neutral: (x + m) / 2
      │     └── contra:      ((8 - x) + m) / 2
      │
      ├── For economic x = e
      │     ├── pro/neutral: (x + m) / 2
      │     └── contra:      ((8 - x) + m) / 2
      │
      └── For GAL-TAN x = g
            ├── pro/neutral: (x + m) / 2
            └── contra:      ((8 - x) + m) / 2
```

### Worked Examples

| Example | Party score \(x\) | Media score \(m\) | Stance | Final score |
|---|---:|---:|---|---:|
| SPD-like score sharing SPON positively | 2.7 | 3.5 | `pro` | `(2.7 + 3.5) / 2 = 3.1` |
| AfD-like score sharing BILD neutrally | 6.5 | 5.2 | `neutral` | `(6.5 + 5.2) / 2 = 5.9` |
| Linke-like score opposing NTV | 1.3 | 4.3 | `contra` | `((8 - 1.3) + 4.3) / 2 = 5.5` |
| FDP economic score sharing taz | 5.4 | 2.8 | `pro` | `(5.4 + 2.8) / 2 = 4.1` |
| Green GAL-TAN score opposing Welt | 1.4 | 4.8 | `contra` | `((8 - 1.4) + 4.8) / 2 = 5.7` |

## Column Dictionary

The current final dataset contains exactly these columns, in this order:

| # | Column | Type | Possible values / range | Description | Computation / source | Assumptions and limitations |
|---:|---|---|---|---|---|---|
| 1 | `index` | Integer | 0 to 819 in the current file | Sequential row identifier for this release. | Reset during merge/final export. | Not stable across reruns or filtering. Do not use as a permanent sample ID. |
| 2 | `article_source` | String | Outlet keys such as `spiegel`, `welt`, `tagesschau`, `t-online`, `Spiegel` | News outlet linked in the post. | Mapped from source-specific outlet columns or inherited from Ben's cleaned dataset. | Naming is mostly standardized but not perfectly canonical; capitalization variants can remain. |
| 3 | `media_label` | Float | Current file includes values from `0.0` to `5.2` | Media ideology score used as \(m\). | Filled from `article_source` using fixed media table plus explicit fallback logic. | Mixed scale values can occur; `0.0` values should be reviewed before substantive analysis. |
| 4 | `party_label_ideology` | Float | Current file includes values from `0.0` to `6.5` | General party ideology score. | Filled from `party` using project party mappings. | Intended latest mapping uses 1-10 to 1-7 conversion, but current artifact includes legacy-looking values. |
| 5 | `party_label_economic` | Float | Approximately 1.3 to 5.4 | Party economic ideology score. | Filled from `party` using the economic party mapping. | Party-level, not politician-specific. |
| 6 | `party_label_galtan` | Float | Approximately 1.4 to 6.6 | Party GAL-TAN score. | Filled from `party` using the GAL-TAN party mapping. | Party-level, not politician-specific. |
| 7 | `final_label_ideology` | Float | Current file includes values from `0.0` to `5.8` | Stance-adjusted ideology score. | Computed from `party_label_ideology`, `media_label`, and `stance`. | Rounded to one decimal in final export. Values are constructed labels, not observed ideology. |
| 8 | `final_label_economic` | Float | Current file includes values from `0.6` to `5.3` | Stance-adjusted economic score. | Computed from `party_label_economic`, `media_label`, and `stance`. | Combines a party economic score with a general media score; interpret cautiously. |
| 9 | `final_label_galtan` | Float | Current file includes values from `0.7` to `5.9` | Stance-adjusted GAL-TAN score. | Computed from `party_label_galtan`, `media_label`, and `stance`. | Combines a party GAL-TAN score with a general media score; interpret cautiously. |
| 10 | `site_content` | String | Non-empty article text | Cleaned article body. | Scraped and cleaned from linked article pages. | Can contain residual boilerplate or missing content from paywalls/dynamic pages. |
| 11 | `social_media_handle` | String | Platform account handle or author identifier | Politician account/handle. | Mapped from source-specific user/handle columns. | Handle formats differ by platform. |
| 12 | `social_media` | String | `X`, `LinkedIn`, `Bluesky`, `Threads` | Platform where the post appeared. | Mapped from source platform metadata. | Platform coverage is imbalanced. |
| 13 | `party` | String | Examples: `SPD`, `Linke`, `B90Grune`, `Bündnis 90 Die Grünen`, `FDP`, `AfD`, `CDU`, `CDU/CSU`, `CSU` | Party affiliation of the account. | Provided by source datasets and account mappings. | String variants are retained; normalize aliases before party-level grouping if needed. |
| 14 | `post_content` | String | Non-empty post text | Text written in the social-media post. | Mapped from source-specific post text fields. | Can include hashtags, mentions, links, emojis, quotes, and platform artifacts. |
| 15 | `article_url` | String URL | HTTP(S) URL | Linked article URL. | Resolved/cleaned URL from source datasets. | URLs can expire or redirect after collection. |
| 16 | `article_title` | String | Non-empty title | Linked article headline. | Extracted from source metadata or scraping. | Headlines may change after collection. |
| 17 | `stance` | String | `pro`, `contra`, `neutral` | Post's stance toward the linked article. | Assigned by the stance classifier / existing stance logic. | Machine-generated; irony, mixed stances, or ambiguous posts can be misclassified. |

## Stance Score Interpretation

The final label columns are continuous constructed scores. Lower values generally correspond to the left/GAL side of the scale; higher values correspond to the right/TAN side. They are not probabilities, sentiment scores, media-factuality ratings, or direct estimates of an individual politician's ideology.

Interpretation guidance:

- Use `media_label` and the three party label columns directly when analyzing source-party compatibility.
- Use `stance` when analyzing whether the politician appears to support, oppose, or neutrally share the article.
- Use `final_label_ideology`, `final_label_economic`, and `final_label_galtan` when the analysis needs a stance-adjusted score.
- Report which label dimension is used; the three final labels are not interchangeable.
- Treat `neutral` as a modeling decision: it uses the same averaging formula as `pro`.

## Known Limitations

- The dataset is not representative of all German political communication.
- Platform coverage and time windows differ across the three source collections.
- Party scores are party-level labels and ignore individual politician variation.
- Media scores are outlet-level labels and ignore article-level variation.
- Some labels in the current artifact are legacy-looking values, especially `0.0` and `1.0` in `media_label` and `party_label_ideology`; review these before analysis.
- Stance labels are machine-generated and can be wrong for sarcasm, irony, quotations, mixed agreement, or issue-specific disagreement.
- Article scraping can fail or include residual boilerplate.
- The same article can be shared by multiple politicians, although the current final file has no duplicate `article_url` values after the rebuilt cleanup.
- `index` is release-specific and should not be used as a stable identifier.

## Reproducibility Notes

Important scripts:

| Stage | Script |
|---|---|
| Merge cleaned source datasets | `merge_cleaned_datasets.py` |
| Fill media and party labels | `fill_cleaned_dataset_labels.py` |
| Compute/recompute stance labels | `recompute_stance_labels.py` |
| Round score columns | `round_stance_score_columns.py` |
| Validate final dataset structure | `scripts/check_dataset.py` |

Important reports in `V.1.2/`:

| Report | Purpose |
|---|---|
| `merge_cleaned_dataset_report.csv` | Merge-stage row and duplicate report |
| `label_mapping_report.csv` | Row-level label mapping report |
| `stance_computation_report.csv` | Stance computation report |
| `recomputed_stance_report.csv` | Recomputed stance value comparison |
| `final_dataset_integrity_audit_report.md` | Final integrity audit |


Example validation command:

```bash
.venv/bin/python scripts/check_dataset.py \
  --input V.1.2/merged_cleaned_dataset_with_stance.csv
```

## Recommended Release Checks

Before using or sharing the final dataset, verify:

1. `pd.read_csv("V.1.2/merged_cleaned_dataset_with_stance.csv")` succeeds.
2. The dataset has 820 rows and 17 columns.
3. The columns match the documented final schema exactly.
4. Every CSV record has exactly 17 fields.
5. All score columns are numeric and rounded to at most one decimal.
6. No required text or URL fields are missing.
7. Any `0.0` or other legacy-looking label values are acceptable for the intended analysis, or the label/stance stages are rerun after mapping review.

