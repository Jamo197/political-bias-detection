# How the LLM-as-a-judge works

Status: 12 Nov 2026, question set `v1` (built-in questions hash `17202cbdadf221ae`), model `jev-1.14.0`.

Code:
* `src/run_llm_judge.py`: runner, label rules, review flags, reports.
* `src/judge_question_sets.py`: the questions (named sets `v1`, `v2`) and the rules that turn answers into labels.
* `src/run_streamlit.py` (`--review`): human review of flagged chunks.
* `src/merge_judge_review.py`: merge of judge and human labels into the final annotations.
* `LLM calib/judge_calibration/` (`calibrate.py`, `compare_arms.py`, `CALIBRATION_LOG.md`): calibration tools and history.

## 0. The whole pipeline in one view

| Step | What happens | Where |
|---|---|---|
| 1. Human annotation | Three annotators (Awsam, Jannes, Neda) independently rate 125 chunks (25 texts × 5 chunks) with the rubric in `help page.md` | `human_annotations/*_qualitative_annotations.jsonl` |
| 2. Consensus and revision | The ratings are compared, the disagreements are discussed, and every annotator revises their own ratings. A consensus (median for R_top / R_ideo, majority for A_caus, ties rounded up) is computed from the revised ratings | `human_annotations/revised/` |
| 3. Calibration | The judge is built and tuned against the revised consensus until the agreement is acceptable (rounds 0-3, section 10) | `LLM calib/judge_calibration/` |
| 4. Judge at scale | The calibrated judge rates all chunks of all 10 model families (968 texts, 4,840 chunks), and the 25 consensus texts are part of this data | `LLM judge results/families/` |
| 5. Human review | The calibrated judge alone was not good enough for R_top and R_ideo, so the chunks it marks `needs_review` (plus a blind spot-check sample) are checked and corrected by hand | Streamlit `--review` |
| 6. Merge | Judge, human review, transferred labels and human consensus become one final label set per model | `final_annotations.jsonl` |
| 7. Analysis | The final labels feed the chunk-influence analysis of the thesis | `analyze_chunk_influence.py` |

### Human annotation and revision

The annotators only received the help page (`help page.md`). They rated independently, so the first round measures how well the rubric alone defines the labels. The agreement is low for the ordinal scales:

| Krippendorff's α | R_top | R_ideo | A_caus |
|---|---|---|---|
| First round (independent) | 0.33 | 0.31 | 0.68 |
| After discussion and revision | 0.82 | 0.63 | 0.88 |

(`human_agreement_report.json` and `human_agreement_report_revised.json`; pairwise quadratic kappa after revision: R_top 0.77-0.91, R_ideo 0.58-0.70, A_caus 0.87-0.89.)

After the discussion, every annotator revised their own ratings. The **revised** ratings are the basis of everything below. The consensus per chunk is `revised/consensus_reference_annotations.jsonl` (`consensus_label` in `compare_human_annotations.py`). The chunks where the annotators still differ are listed in `revised/Annotations - annotation_disagreements.csv`. R_ideo remains the weakest dimension even between humans (α 0.63), because nearly all chunks are rated 3 or 4 (range restriction, section 7).

### Why a human review step was added

After calibration, R_top and especially R_ideo still agreed with the humans only at about 80-90 % of the human-human level. Such errors cannot be tuned away with 125 chunks. The judge therefore marks the chunks where it is unsure, and a human corrects them (section 9). The 25 consensus texts are not judged blindly in the final data: where the judge rated the identical (input, chunk) pair the human consensus wins.

## 1. What the judge is for

The RAG classifier retrieves passages ("chunks") from German Bundestag speeches and uses them to rate the
political bias of a short text. To find out *why* RAG helps or hurts, every retrieved chunk is rated on the
same three dimensions the human annotators used (see `help page.md`):

| Dimension | Scale | Question |
|---|---|---|
| **R_top**, topical relevance | 1-5 | How closely does the chunk match the specific subject of the input text? |
| **R_ideo**, ideological grounding | 1-5 | How much ideological expression does the chunk itself contain? |
| **A_caus**, causal use | 0/1 | Did the generator's justification actually draw on this chunk? |

The 125 human-rated chunks come from four quadrants of the baseline-vs-RAG error change (25 texts, 5 chunks each):
Q1 RAG win, Q2 RAG distraction, Q3 joint failure, Q4 baseline sufficiency. The judge reproduces the human
ratings automatically so that the whole corpus can be rated, and it is checked against those 125 chunks.

## 2. Design: yes/no questions instead of a 1-5 score

The judge does **not** ask the model "rate this from 1 to 5". That was tested: with the help-page wording the
model over-rated by about +0.3 to +0.4 levels and agreed less with the humans (CALIBRATION_LOG, round 2,
experiment 3).

Instead, each level boundary gets **one yes/no question with an explicit test** (a "gate"). The model is
TypeSafe's Jev decision model, called through the TypeSafe Python SDK (`system_one`). It returns a probability
of "yes" (`noul`, 0-1) for each question independently. The label is then derived from these probabilities
with fixed rules. This has three advantages:

* every label can be traced back to concrete answers ("level 4 because the chunk discusses the same item");
* each boundary can be checked and reworded on its own (gate AUC);
* the probabilities give a confidence signal (`needs_review`, section 6).

**Question sets.** The questions live in `judge_question_sets.py` as named sets. `v1` is the calibrated set
that produced all results in this document and all final labels. `v2` (per-gate focus, holistic Score
questions, split A_caus) is an experimental alternative; it was not adopted. `--list-question-sets` prints the
sets and their hashes. Every result row stores the set name and hash.

## 3. One chunk, step by step

```
input text + chunk + generator justification
        │
        ├─ request 1: content questions (R_top + R_ideo gates)
        │             state WITHOUT the chunk's party / speaker / source ("nometa")
        │
        ├─ request 2: A_caus question
        │             full state (needs chunk number, speaker, party)
        │
        ├─ deterministic citation check of the justification (regex)
        │
        └─ label rules ──► R_top, R_ideo, A_caus  + needs_review flags + spot_check flag
```

**Why two requests.** The annotators were told to ignore speaker metadata. When the model saw the party name,
it judged the party label rather than the text (the "speaks for own party" gate fired for 49 instead of 37
chunks). Removing party, speaker and source for the content questions raised R_ideo agreement from 0.49 to
0.56 without fitting anything. A_caus *needs* that metadata, because justifications cite chunks by number or
speaker, so it gets its own request with the full state.

The state sent to the model is a JSON object (`build_state` / `build_nometa_state`):

```json
{
  "evaluation_task": "Evaluating evidence quality in a RAG pipeline ...",
  "input_text": "<German post / quote>",
  "retrieved_chunk": {"text": "<chunk>"},          // nometa: text only
  "generator_justification": "<the classifier's rationale>"
}
```

For A_caus, `retrieved_chunk` additionally carries `chunk_identifier` ("[2]"), `citation_forms`, `party`,
`speaker`, `source` and (when present) `date`. No retrieval or similarity scores are sent.

`--state-mode` switches this: `nometa` (default), `full` (every question sees the metadata) and `minimal`
(only the two texts, no justification).

## 4. The questions (gates)

All `v1` gates are in `judge_question_sets.py`. A gate counts as **yes** when its probability is ≥ 0.5.

### R_top: topical relevance

| Gate | Plain-language test (yes when …) | Separates |
|---|---|---|
| `T_domain` | input and chunk are in the same concrete policy field (criminal law, pandemic, migration, …); shared words or "both are politics" do not count | 1 vs ≥2 |
| `T_subarea` | the chunk argues about the same narrow policy question (same kind of decision or problem) | ≤2 vs ≥3 |
| `T_item` | the chunk discusses the specific bill, measure, event or figure the input takes a position on | ≤3 vs ≥4 |
| `T_central` | that item is the chunk's *main* subject, and the chunk speaks to the point the input makes | ≤4 vs 5 |

All four share a focus instruction: judge subject matter only (opposite positions on the same measure are the
same topic), and first identify the specific item the input text is about.

### R_ideo: ideological grounding

| Gate | Plain-language test (yes when …) | Separates |
|---|---|---|
| `I_procedural` | the chunk is only floor management (thanks, agenda, votes) | 1 vs 2 (never fired in the data) |
| `I_evaluative` | the passage uses normative or politically loaded language, selective emphasis or one-sided framing; **no** for neutral reporting, balanced overviews, consensual or cross-party appeals | 2 vs ≥3 |
| `I_stance` | the speaker takes a strong, partisan side (charged language, sweeping judgements, a camp-typical programmatic demand); *burden of proof: no unless clearly shown* | 3 vs ≥4 |
| `I_party` | the text explicitly names the speaker's own party or faction **and** states its programmatic position; *burden of proof* | 4 vs 5 |

`I_evaluative` was reworded in round 3. The old version ("does the speaker argue or evaluate?") said yes for
all 125 chunks, so the judge could never give a 2. Its AUC rose from 0.81 to 0.91-0.92.

### A_caus: causal use

`A_caus` asks whether the justification demonstrably draws on *this* chunk: by number in any form ("[2]",
"[2, 3]", "anchor 2"), by speaker name, or by invoking this chunk's party as comparison evidence.

## 5. From answers to labels

### R_top: ladder

```
R_top = 1 + (position of the highest gate answered yes)      none → 1
        T_domain → 2,  T_subarea → 3,  T_item → 4,  T_central → 5
```

Example: `T_domain 0.96, T_subarea 0.34, T_item 0.62, T_central 0.31` → the highest yes is `T_item` → **4**.

### R_ideo: ladder, or cut points (default)

Ladder (`derive_labels`):

| Condition | Label |
|---|---|
| `I_party` and `I_stance` yes | 5 |
| `I_stance` and `I_evaluative` yes | 4 |
| `I_evaluative` yes | 3 |
| otherwise: `I_procedural` yes / no | 1 / 2 |

**Distribution-matched cut points** replace the hard 0.5 thresholds. They are applied **by default**
(`--posthoc-model`, `DEFAULT_POSTHOC_MODEL`; `--posthoc-model none` switches them off) and only for R_ideo:

1. Each gate probability becomes a logit, `ln(p / (1−p))`, with p clipped to [0.02, 0.98], so each gate
   contributes between −3.9 and +3.9.
2. The logits of `I_evaluative`, `I_stance` and `I_party` are **summed** into one score.
3. Four fixed cut points on that score give the level. They were placed so that, on the 125 calibration
   chunks, the label shares equal the consensus shares (0 % / 2 % / 22 % / 57 % / 19 %). Only these shares
   are fitted, not which chunk gets which label.

Current R_ideo cut points (`posthoc_quantile_2026-10-23_142551_R_ideo.json`): level 2 is always reached;
level 3 needs score ≥ −5.73, level 4 needs ≥ 0.33, and level 5 needs ≥ 7.07.

Example (text 510, chunk 1, a consensual cross-party sports appeal; the humans rated it 2):
`I_evaluative 0.38, I_stance 0.08, I_party 0.03` → logits −0.49, −2.44, −3.48 → score −6.41 < −5.73 →
**2**.

Why it helps: a single gate no longer decides alone. A chunk with a confident "loaded framing" but a
borderline stance is weighed as a whole, and the systematic over-rating of R_ideo (bias +0.1) disappears.

The cut points live on the scale of the gate probabilities, so **they belong to one question wording**. If a
gate is reworded, refit them on a run of the new questions (`calibrate.py --export-quantile`).
`run_llm_judge.py` skips a mapping whose questions hash differs from that of the set and prints a warning.
Applying the cut points fitted for the reworded `T_subarea` to the old `T_subarea` shifted R_top down by a
whole level (run `2026-10-23_142308`).

For R_top, cut points are **not** used: they reproduce the human shares but lose slightly in agreement on two
separate runs (section 7). The labels of a stored run can be re-derived without API calls
(`--rederive-from`); the ladder-only labels stay in `uncalibrated_labels`.

### A_caus: citation rules first, model second

Human A_caus labels follow the citations in the justification very consistently, so the explicit cases are
decided by a deterministic check (`citation_info`, which also understands lists, ranges and keywords such as
"chunk 2" / "anchors 1 and 3"):

| Situation | A_caus |
|---|---|
| the justification cites this chunk's number (any form, lists and ranges included), or names its speaker | 1 |
| it cites other chunks by number, but not this one | 0 |
| no numeric citations at all (e.g. only "in line with the SPD speakers") | model probability ≥ 0.5 |

## 6. Confidence: `needs_review` and `spot_check`

Each chunk gets a per-dimension `needs_review` flag that marks labels a human should check:

* **R_top and A_caus** (gate ladder / rules): a gate of that dimension lies close to 0.5 (band ±0.10 for
  R_top, ±0.20 for A_caus), or the ladder contradicts itself (a higher gate yes while a lower gate is no, e.g.
  `T_item` yes with `T_subarea` no).
* **R_ideo** (read through cut points): the summed logit lies within 1.0 of a cut point (`REVIEW_CUT_BAND`).
  On the 125 calibration chunks this flags 31 % and catches 20 of 43 errors, against 34 % and 17 of 43 for
  the old gate-near-0.5 rule.
* A_caus decided by the citation rules is never flagged.

`needs_review.any` is true when any dimension is flagged.

Independent of the flags, about **5 % of all chunks** are marked `spot_check` (a fixed hash of source, text
and chunk, so it is stable across re-runs). A human rates these blind, i.e. without seeing the judge's label.
They estimate the error of the labels nobody reviews.

On the 125 calibration chunks about 37-40 % (R_top), 31-38 % (R_ideo) and 10-13 % (A_caus) are flagged. On the
unflagged chunks agreement is clearly higher (R_top α ≈ 0.78 vs 0.69 overall). On the 4,840 chunks of the
model families the shares are higher for R_top: about 36-49 % R_top, 27-31 % R_ideo, 0-11 % A_caus, and about
56-65 % with at least one flag (`families/summary.json`).

## 7. How good is it? (calibration set, 125 chunks)

Reference: the adjudicated consensus of the three annotators
(`human_annotations/revised/consensus_reference_annotations.jsonl`). The judge is compared with each
annotator and with the consensus. The fairest yardstick is the **ratio** of judge-vs-human agreement to
human-vs-human agreement, because both suffer equally from the narrow label range of R_ideo (range
restriction).

Recommended configuration: run `2026-10-23_142551`, R_top from the ladder, R_ideo with cut points
(range report `range_restriction_2026-10-23_142551_ideo_cuts.json`):

| | Human vs human QWK | Judge vs human QWK | Ratio (95 % CI) | Judge vs consensus QWK |
|---|---|---|---|---|
| R_top | 0.82 | 0.65 | 0.79 (0.61-0.90) | 0.68 |
| R_ideo | 0.65 | 0.57 | 0.88 (0.75-1.02) | 0.61 |
| A_caus (kappa) | 0.88 | 0.73 | 0.82 (0.61-1.00) | 0.74 |

The judge reaches about 80-90 % of the agreement two humans reach with each other. The R_ideo cut points are
fitted on these same chunks. Their honest, out-of-sample effect (leave-one-text-out: fitted on 24 texts,
scored on the 25th) is **α 0.563 → 0.579**. On the 8 held-out texts it is **0.608 → 0.665**.

Label distributions (counts of 125, levels 1-5):

| | R_top | R_ideo |
|---|---|---|
| consensus | 10 / 37 / 23 / 29 / 26 | 0 / 2 / 28 / 71 / 24 |
| judge | 8 / 46 / 14 / 25 / 32 | 0 / 2 / 28 / 71 / 24 |

### Alt-test: can the judge replace a human annotator?

`alt_test/RESULTS.md` applies the Alternative Annotator Test (Calderon, Reichart & Dror, 2025) to run
`2026-10-22_132304` (leave-one-annotator-out, ε = 0.2, revised ratings):

| Dimension | Winning rate ω | Verdict |
|---|---|---|
| R_top | 0.00 | fail |
| R_ideo | 0.33 | fail (passes from ε = 0.22) |
| A_caus | 1.00 | pass |

Conclusion drawn there and used here: the judge may annotate A_caus on its own, whereas R_top and R_ideo need
a human in the loop. This is the reason for step 5 of the pipeline. (The run predates the final cut points, and
the thresholds were tuned on the same 125 chunks, so the ω values are optimistic.)

**Known limits**

* **R_top 2 vs 3** remains the weakest boundary: the judge still gives 2 too often. A looser `T_subarea`
  (round 3, `questions_v5.json`) ranks better (AUC 0.90 vs 0.88) but answers yes too often, and only reaches
  α 0.707 together with cut points. That is within noise, and worse on the held-out texts, so it was not
  adopted.
* **Quadrant Q2 (RAG distraction)** has clearly lower R_top agreement (α ≈ 0.5) than Q1 (≈ 0.86).
* **A_caus** errors come from justifications that only name a party. Annotators themselves disagreed there.
* **Run-to-run noise** is small (gate correlation ≥ 0.995, 0-4 label flips per dimension). A_caus kappa
  varies by about ±0.03 between identical runs, so smaller differences are not meaningful.
* Only 25 texts: all confidence intervals are wide.

## 8. How to run it

From the project root. Results go to `RAG Analysis/LLM judge results/<run_id>/`.

```bash
# calibration set (default input: the revised consensus chunks); R_ideo through the cut points
python src/run_llm_judge.py --concurrency 8

# without the cut points (pure gate ladder)
python src/run_llm_judge.py --posthoc-model none

# another input file (annotation format from run_streamlit.py; labels may be absent)
python src/run_llm_judge.py --annotations <file.jsonl>

# re-apply label rules / cut points to a stored run (no API calls)
python src/run_llm_judge.py --rederive-from "RAG Analysis/LLM judge results/<run>/llm_judge_annotations.jsonl" \
  --posthoc-model <...> --run-id <new_id>

# try reworded questions, or compare sets in the same requests
python src/run_llm_judge.py --questions-file <questions.json> --run-id <id>
python src/run_llm_judge.py --question-set v1,v2 --run-id v1_vs_v2
```

The API key (`TYPESAFE_API_KEY`, fallback `OPENROUTER_API_KEY_ME`) is read from `.env.local`.

After changing a question:

```bash
python "RAG Analysis/LLM calib/judge_calibration/calibrate.py" "LLM judge results/<run>" --export-quantile R_ideo
python "RAG Analysis/LLM calib/judge_calibration/compare_arms.py" "LLM judge results/<old>" "LLM judge results/<new>"
python "RAG Analysis/LLM calib/range_restriction_report.py" --run "RAG Analysis/LLM judge results/<run>"
```

The first command prints the gate AUCs and the leave-one-text-out comparison, and writes the new cut-point
file. Then rederive with it.

Output files per run:

* `llm_judge_annotations.jsonl`: one line per chunk, with all gate probabilities, the citation evidence,
  `derived_labels` (final), `uncalibrated_labels` (ladder only, when cut points were applied),
  `needs_review`, `spot_check` and token usage;
* `llm_judge_qualitative_annotations.jsonl`: the same, grouped per text in the human annotation format;
* `llm_judge_report.json`: agreement with the reference labels (when the chunks carry labels), confusion
  matrices and the share flagged for review.

## 9. Judge-assisted annotation of all model families

The calibrated judge pre-annotates the generator results of every model family and size. A human checks the
chunks the judge is unsure about, plus a blind random sample.

**1. Judge everything.** This covers 10 files (3 sizes or more of llama, mistral and qwen), 968 texts and
4,840 chunks, with one folder per model and size.

```bash
python src/run_llm_judge.py --samples all --run-id families --concurrency 8
python src/run_llm_judge.py --samples all --run-id families --resume   # after a crash or failed chunks
```

* Question set `v1` and the R_ideo cut points are the defaults; `--samples` takes only one set.
* `families/summary.json` lists the flagged chunks per dimension and file.
* The 25 consensus texts are part of these files.

**2. Review in Streamlit**, one folder at a time:

```bash
streamlit run src/run_streamlit.py -- --review "RAG Analysis/LLM judge results/families/llama_70b" --annotator Jannes
```

* **Which texts:**
  - All Q1 (RAG win) and Q2 (RAG distraction) texts, plus 10 random texts each from Q3 and Q4
    (`REVIEW_QUOTAS`, seed 42, drawn per folder). That is about 70 texts (350 chunks) per model and size.
  - The draw is stored in `review_selection.json`, so it never changes.
  - Other quotas: `--quotas Q1=all,Q2=all,Q3=5,Q4=5`. Every text: `--all`.
  - Texts outside the selection keep the judge's labels (source `judge`).
* **⚠ review R_top / R_ideo / A_caus**: the judge was unsure on this dimension. It is pre-filled with the
  judge's label; check it. The other dimensions of the chunk are pre-filled too and count as "seen".
* **🎯 spot check**: about 5 % of all chunks. They are empty and must be rated from scratch, so the judge's
  label cannot anchor the rating.
* The reviews are saved to `reviewed_annotations.jsonl` in the same folder. Re-opening the app continues
  where you left off.

**Which models were reviewed.** Only three folders were reviewed by hand: **llama_70b, mistral_14b and
qwen_3b** (67, 70 and 51 texts). The other seven keep the judge's labels, supplemented by transferred labels
and the human consensus (see below).

**3. Merge** into the final labels:

```bash
python src/merge_judge_review.py "RAG Analysis/LLM judge results/families"
# llama_8b: let the human consensus win for the texts the annotators rated
python src/merge_judge_review.py "RAG Analysis/LLM judge results/families/llama_8b" \
  --human "RAG Analysis/human_annotations/revised/consensus_reference_annotations.jsonl"
```

* Each folder gets `final_annotations.jsonl`, in the human annotation format.
* Each label carries its source: `judge`, `judge_seen`, `human_review`, `human_changed`, `spot_check`,
  `transferred` or `human_consensus`.
* `families/review_summary.json` reports, per folder and pooled:
  - how often a reviewed flagged label was changed;
  - the spot-check agreement of the judge with the blind human rating, split into unflagged and flagged
    chunks. The unflagged value is the error estimate for every label with source `judge`.

**Transfer to unreviewed models.** Retrieval does not depend on the generator: the same input text retrieves
the same chunks for every model.

* R_top and R_ideo depend only on the input text and the chunk. A pair reviewed for one model is copied to
  every identical pair (`--transfer-from`, label source `transferred`; the median if reviewed several times).
* A_caus is never transferred, because it depends on each model's own justification.
* Consensus labels (`--human`) are copied the same way and win over the judge. Their A_caus is used only
  where the justification is identical.

```bash
F="RAG Analysis/LLM judge results/families"
python src/merge_judge_review.py "$F" --transfer-from "$F/llama_70b" "$F/mistral_14b" "$F/qwen_3b" \
  --human "RAG Analysis/human_annotations/revised/consensus_reference_annotations.jsonl"
```

### What the review showed (`review_summary.json`, 2 Nov 2026)

Pooled over the three reviewed models:

| | R_top | R_ideo | A_caus |
|---|---|---|---|
| Flagged chunks reviewed | 418 | 248 | 26 |
| Share of flagged labels the reviewer changed | 46 % | 25 % | 19 % |
| Spot check, unflagged: exact (within ±1) | 36 % (84 %), n=25 | 59 % (100 %), n=27 | 98 %, n=44 |
| Spot check, flagged: exact (within ±1) | 25 % (80 %), n=20 | 33 % (100 %), n=18 | n=1 |

* A_caus is reliable, as in calibration. R_ideo is within ±1 in all spot checks.
* **R_top is the weak point.** The flagged labels were changed almost every second time, and even the
  unflagged spot-check chunks agree exactly only 36 % of the time (84 % within one level). This is below the
  59 % exact agreement the flag achieved on the calibration chunks. The spot-check samples are small (n=18-44
  per cell), so these figures are rough. Treat R_top labels with source `judge` as ±1 level.
* The reviewer rated pairs that occur in several models identically in 71 % (R_top) and 89 % (R_ideo) of
  cases (110 repeated pairs). That is about the level the judge reaches against the reviewed labels.
* The transfer covers 15-29 % of the chunks of each unreviewed model; the rest keeps the judge's labels.
* A correction mapping refitted on the 940 reviewed chunks did not beat the judge's own labels on a held-out
  model, so no learned correction is applied.

**Calibration expectation (125 chunks) for comparison.**

| | R_top | R_ideo | A_caus |
|---|---|---|---|
| Chunks flagged | 40 % | 31 % | 13 % |
| Judge errors caught by the flag | 52 % | 47 % | 56 % |
| Exact agreement on unflagged chunks | 59 % (~95 % within ±1) | 73 % | 94 % |

About half of the judge's errors are not flagged and remain in the final data. In the method section, report
the spot-check error estimate and describe the procedure as "LLM pre-annotation with human review of
uncertain cases and a blind random check".

## 10. How it got here (short)

| Round | Change | Effect |
|---|---|---|
| 0 | gate ladder as found | R_top α 0.68, R_ideo 0.49 |
| 1 | stricter `T_domain`, `T_central` about "same item and point" | R_top diagnostics better, α unchanged |
| 2 | chunk metadata withheld for content questions (`nometa`) | R_ideo 0.49 → 0.56; examples and graded 1-5 questions tested and rejected |
| 3 | `I_evaluative` reworded to "loaded vs neutral/balanced" (adopted); distribution-matched cut points (adopted for R_ideo); looser `T_subarea` (not adopted) | R_ideo 0.56 → 0.58 out-of-sample (0.61 vs consensus in-sample), first level-2 labels; R_top unchanged |
| later | `v2` question set (holistic Score questions, split A_caus) tried as an alternative | not adopted; `v1` stays the default |
| review | judge run on all families, flagged chunks and spot checks reviewed by hand | see section 9 |

The details, all runs and confidence intervals are in `judge_calibration/CALIBRATION_LOG.md`.

## 11. How the final labels are used downstream

The final labels (`final_annotations.jsonl`, one per model folder) are the chunk ratings of the thesis. They
feed `RAG Analysis/analyze_chunk_influence.py`, which asks how the retrieved chunks shape the RAG predictions:

```bash
python "RAG Analysis/analyze_chunk_influence.py"                               # final labels, all texts
python "RAG Analysis/analyze_chunk_influence.py" --tier reviewed               # only human-checked texts
python "RAG Analysis/analyze_chunk_influence.py" --labels judge --out "RAG Analysis/chunk_influence_judge"
```

* `--labels final` (default) uses the merged labels; `--labels judge` uses the unreviewed judge labels, as a
  robustness check of the human review (`chunk_influence_judge/`).
* `--tier all` uses every text with chunk labels; `--tier reviewed` restricts to the texts a human has
  actually seen (`chunk_influence/` and `chunk_influence_reviewed/`).

What it does with the labels (outputs in `RAG Analysis/chunk_influence/`, summary in `summary.md`):

1. **Quadrants and base rates:** the sample has 25 texts per quadrant, so pooled numbers are weighted by the
   quadrant base rates of the full population (about 775 texts per model).
2. **Evidence use (population, needs no labels):** a regression of the RAG prediction on the baseline
   prediction and the mean party position of the retrieved chunks (CHES 2024 `lrgen`, 1-7 scale), with a
   permutation null.
3. **Which chunks carry the effect (labels needed):** the same regression with the positions of cited vs.
   uncited chunks (A_caus) and of relevant (R_top ≥ 4) vs. other chunks.
4. **Quadrant descriptions:** mean R_top / R_ideo, citation counts and the party composition of the chunks per
   quadrant, with Q1-vs-Q2 tests.
5. **Utilisation:** the share of chunks cited by R_top / R_ideo level.
6. **Helped or hurt:** per model, the share of texts where RAG reduced or increased the error.

Consequences for interpretation: R_top is the least reliable label (section 9), so conclusions that rest on
R_top thresholds (e.g. "relevant = R_top ≥ 4") should be backed by the `reviewed` tier or the `judge` robustness
run. A_caus is reliable, and R_ideo is stable within ±1 level.
