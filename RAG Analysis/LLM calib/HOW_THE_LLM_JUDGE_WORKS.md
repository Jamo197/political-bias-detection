# How the LLM-as-a-judge works

Status: 23 Oct 2026, built-in questions hash `17202cbdadf221ae`, model `jev-1.14.0`.
Code: `src/run_llm_judge.py`. Calibration tools and history: `LLM calib/judge_calibration/`
(`calibrate.py`, `compare_arms.py`, `CALIBRATION_LOG.md`).

## 1. What the judge is for

The RAG classifier retrieves passages ("chunks") from German Bundestag speeches and uses them to rate the
political bias of a short text. To find out *why* RAG helps or hurts, every retrieved chunk is rated on the
same three dimensions the human annotators used (see `help page.md`):

| Dimension | Scale | Question |
|---|---|---|
| **R_top**, topical relevance | 1-5 | How closely does the chunk match the specific subject of the input text? |
| **R_ideo**, ideological grounding | 1-5 | How much ideological expression does the chunk itself contain? |
| **A_caus**, causal use | 0/1 | Did the generator's justification actually draw on this chunk? |

Three annotators rated 125 chunks (25 texts × 5 chunks). The judge reproduces these ratings automatically, so
the whole corpus can be rated without a human, and it is checked against those 125 chunks.

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
        └─ label rules ──► R_top, R_ideo, A_caus  + needs_review flags
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
`speaker`, `source` and `date`.

## 4. The questions (gates)

All gates are in `JEV_QUESTIONS` in `run_llm_judge.py`. A gate counts as **yes** when its probability is
≥ 0.5.

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

### R_ideo: ladder, or optionally cut points

Ladder (`derive_labels`):

| Condition | Label |
|---|---|
| `I_party` and `I_stance` yes | 5 |
| `I_stance` and `I_evaluative` yes | 4 |
| `I_evaluative` yes | 3 |
| otherwise: `I_procedural` yes / no | 1 / 2 |

**Distribution-matched cut points** are optional (`--posthoc-model`) and recommended for R_ideo. They replace
the hard 0.5 thresholds:

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
`run_llm_judge.py` prints a warning when the questions hash stored in the file differs from that of the
answers. Applying the cut points fitted for the reworded `T_subarea` to the old `T_subarea` shifted R_top
down by a whole level (run `2026-10-23_142308`).

For R_top, cut points are **not** recommended: they reproduce the human shares but lose slightly in agreement
on two separate runs (section 7).

### A_caus: citation rules first, model second

Human A_caus labels follow the citations in the justification very consistently, so the explicit cases are
decided by a deterministic check (`citation_info`):

| Situation | A_caus |
|---|---|
| the justification cites this chunk's number (any form, lists and ranges included), or names its speaker | 1 |
| it cites other chunks by number, but not this one | 0 |
| no numeric citations at all (e.g. only "in line with the SPD speakers") | model probability ≥ 0.5 |

## 6. Confidence: `needs_review`

Each chunk gets a per-dimension flag that marks labels a human should check:

* a gate of that dimension lies close to 0.5 (band ±0.10 for R_top, ±0.20 for R_ideo and A_caus), or
* the ladder contradicts itself (a higher gate yes while a lower gate is no, e.g. `T_item` yes with
  `T_subarea` no).

A_caus decided by the citation rules is never flagged. Currently about 37 % (R_top), 34-38 % (R_ideo) and
10-12 % (A_caus) of chunks are flagged. On the unflagged chunks agreement is clearly higher (R_top α ≈ 0.78
vs 0.69 overall).

## 7. How good is it?

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
# recommended: built-in questions, R_ideo through the cut points
python src/run_llm_judge.py --concurrency 8 \
  --posthoc-model "RAG Analysis/LLM calib/judge_calibration/posthoc_quantile_2026-10-23_142551_R_ideo.json"

# another input file (annotation format from run_streamlit.py; labels may be absent)
python src/run_llm_judge.py --annotations <file.jsonl> --posthoc-model <...>

# re-apply label rules / cut points to a stored run (no API calls)
python src/run_llm_judge.py --rederive-from "RAG Analysis/LLM judge results/<run>/llm_judge_annotations.jsonl" \
  --posthoc-model <...> --run-id <new_id>

# try reworded questions
python src/run_llm_judge.py --questions-file <questions.json> --run-id <id>
```

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
  `needs_review` and token usage;
* `llm_judge_qualitative_annotations.jsonl`: the same, grouped per text in the human annotation format;
* `llm_judge_report.json`: agreement with the reference labels, confusion matrices and the share flagged
  for review.

## 9. How it got here (short)

| Round | Change | Effect |
|---|---|---|
| 0 | gate ladder as found | R_top α 0.68, R_ideo 0.49 |
| 1 | stricter `T_domain`, `T_central` about "same item and point" | R_top diagnostics better, α unchanged |
| 2 | chunk metadata withheld for content questions (`nometa`) | R_ideo 0.49 → 0.56; examples and graded 1-5 questions tested and rejected |
| 3 | `I_evaluative` reworded to "loaded vs neutral/balanced" (adopted); distribution-matched cut points (adopted for R_ideo); looser `T_subarea` (not adopted) | R_ideo 0.56 → 0.58 out-of-sample (0.61 vs consensus in-sample), first level-2 labels; R_top unchanged |

The details, all runs and confidence intervals are in `judge_calibration/CALIBRATION_LOG.md`.
