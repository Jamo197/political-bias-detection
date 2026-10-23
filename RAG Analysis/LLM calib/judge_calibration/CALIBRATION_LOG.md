# Jev judge calibration log

Reference: adjudicated consensus (`human_annotations/revised/consensus_reference_annotations.jsonl`,
125 chunks / 25 texts, three annotators per chunk). Judge: `jev-1.14.0`. Agreement is
Krippendorff's alpha (ordinal for R_top / R_ideo, nominal for A_caus) between judge and consensus;
kappa is quadratic-weighted for R_top / R_ideo and Cohen's for A_caus. Dev/holdout split is by text
(every third text held out: 17 dev texts / 8 holdout texts, so the holdout is noisy).

Reproduce any row: `python "RAG Analysis/judge_calibration/calibrate.py" --summary <run> ...`
(full report: drop `--summary`; add `--posthoc` for the calibration CV, `--json` to save).

## Final configuration (frozen, round 2)

* Questions: built-in `JEV_QUESTIONS` in `src/run_llm_judge.py` = `questions_v4.json` (hash `7c4f5961ce892ca3`).
  **Round 3 update:** `I_evaluative` replaced by its v5 wording (built-in hash now `17202cbdadf221ae`);
  see "Round 3" below for the optional cut-point layer and the candidate `T_subarea`.
* State: `--state-mode nometa` (default). R_top / R_ideo are asked with the full state minus the
  chunk's party / speaker / source; A_caus keeps the full state (two requests per chunk).
* Labels read straight off the gate answers, nothing fitted, plus a `needs_review` flag.
* Run: `final_nometa`. The earlier `final` / `final_posthoc` runs (full state) are kept as the round-1 result.

| Dimension | Human-only alpha | Judge vs consensus alpha (95 % CI by text) | kappa | Judge as 4th rater alpha |
|---|---|---|---|---|
| R_top | 0.823 | 0.695 (0.53-0.79) | 0.689 | 0.738 |
| R_ideo | 0.634 | 0.560 (0.41-0.68) | 0.562 | 0.570 |
| A_caus | 0.883 | 0.743 (0.54-0.92) | 0.743 | 0.799 |

R_top kappa 0.689 is just under the 0.70 rule; R_ideo is below it, but human R_ideo alpha is itself
only 0.63 (pairwise quadratic kappa 0.58-0.70), so 0.70 against the consensus may not be attainable.
The judge as 4th rater lowers the human alpha from 0.634 to 0.570 for R_ideo and from 0.823 to 0.738
for R_top.

### Trust-or-escalate (`needs_review`, `final_nometa`)

| Dimension | Routed to human | Alpha on kept chunks | Exact on kept |
|---|---|---|---|
| R_top (band 0.10) | 40 % | 0.784 | 0.57 |
| R_ideo (band 0.20) | 33 % | 0.648 | 0.69 |
| A_caus (band 0.20) | 11 % | 0.838 | 0.92 |

Bands were chosen on these same 125 chunks (one scalar per dimension), so treat the kept-chunk alphas
as optimistic.

### Round 1 result (full state, superseded)

`final`: R_top 0.681, R_ideo 0.490, A_caus 0.758 (alpha). An optional ordinal-logistic mapping for
R_ideo (`posthoc_final.json`, fitted on `final`) raised R_ideo to 0.547 out-of-sample. Round 2 showed
that this gain only compensated for the speaker metadata: on the `nometa` run the same kind of
mapping gives 0.489 out-of-sample, below the unfitted 0.560. The mapping is therefore not
recommended; the code stays (`--posthoc-model`) but is off by default, and `posthoc_final.json`
belongs to the old full-state probabilities and must not be applied to `nometa` runs.

## Round 3: label shape (R_top 2/3, R_ideo 2)

Problem (range-restriction report on `2026-10-22_132304`): R_top gave 2 too often and 3 too rarely
(judge 8/45/15/24/33 vs consensus 10/37/23/29/26); R_ideo never gave 2. Diagnosis on the stored
probabilities: `T_subarea` (the 2/3 gate) asked for "the same *narrow policy question*", stricter than
help-page level 3; `I_evaluative` (the 2/3 gate) said yes for every chunk (5th percentile 0.94) because
every speech "argues or evaluates", while the help page draws 2/3 at loaded framing vs neutral /
balanced / consensual.

Two changes, tested separately (run `v5_subarea_eval`, `questions_v5.json`, hash `42b4039d5921f935`):

1. **Rewording** both gates along the help page (`T_subarea`: same sub-area or regulatory mechanism, with
   the heating-law example; `I_evaluative`: loaded / one-sided framing vs neutral, balanced, consensual).
2. **Distribution-matched cut points** (`calibrate.py --export-quantile`, applied with
   `run_llm_judge.py --posthoc-model`): the level is read off the *sum of the gate logits* with four fixed
   cut points placed so the label shares equal the consensus shares. Only the marginal distribution is
   fitted (4 numbers per dimension), not which chunk gets which label.

Gate AUC: `T_subarea` 0.876 -> 0.902, `I_evaluative` 0.813 -> 0.923 (other gates unchanged, A_caus
identical: noise check passed).

Alpha vs consensus, leave-one-text-out for the cut points (`--posthoc`):

| Set-up | R_top | R_ideo |
|---|---|---|
| current questions, ladder | 0.697 | 0.551 |
| current questions + cut points | 0.685 | 0.598 |
| v5 questions, ladder | 0.648 | 0.576 |
| v5 questions + cut points | **0.707** | 0.576 (qwk 0.597) |

Paired delta v5 vs current (ladder, `compare_arms.py`): R_top -0.049 (-0.117, +0.017), R_ideo
**+0.025 (+0.004, +0.057)**, holdout R_ideo +0.031.

* **`I_evaluative` v5 adopted** into `JEV_QUESTIONS` (passes the round-2 rule on its own: alpha up, CI
  above 0, holdout up; it now yields level 2 for a consensus-2 chunk without any fitting). New built-in
  hash `17202cbdadf221ae`.
* **`T_subarea` v5 not adopted on its own**: it ranks better but is too lenient at 0.5 (yes for 101
  chunks vs 78 human), so the ladder shifts R_top up (bias +0.26). Only with the cut points does it give
  the best R_top (0.707 LOTO, shape equal to the consensus). Kept in `questions_v5.json`.
* **Cut points** fix the shape for both dimensions (range report, in-sample: R_top 10/37/23/29/26,
  R_ideo 0/2/28/71/24, both exactly the consensus). For R_ideo they also remove the +0.1 bias. For R_top
  they need the v5 `T_subarea`: on the old gate they cost 0.012.
* All differences except the R_ideo rewording are within the bootstrap CIs. The cut points are fitted on
  the evaluation set, so only the LOTO figures count; on the 8-text holdout (fit on dev texts only), R_top
  with cut points is below the uncalibrated ladder (0.570 vs 0.603).

Files: `posthoc_quantile_2026-10-22_132304.json` and `posthoc_quantile_v5_subarea_eval.json` (cut points
per run). Rederived runs: `LLM judge results/2026-10-22_132304_quantile` and `v5_subarea_eval_quantile`.
Apply a cut-point file only to runs whose gate wording matches the run it was fitted on.

### Round 3b: re-run with the new built-in questions (`17202cbdadf221ae`)

Two fresh runs: `2026-10-23_142551` (no post-hoc) and `2026-10-23_142308` (with
`posthoc_quantile_v5_subarea_eval.json`). The two runs agree closely: gate correlation 0.995-0.999,
0 R_top and 4 R_ideo ladder flips; A_caus kappa 0.74 vs 0.79 from 3 flips near 0.5.

* **The 142308 R_top result is a mismatch, not a property of the cut points.** The v5 file was fitted for
  the lenient v5 `T_subarea`; on the stricter built-in gate the logit sums are lower, so R_top slid down
  (judge 20/41/21/26/17 vs consensus 10/37/23/29/26, bias -0.36). `run_llm_judge.py` now warns when a
  post-hoc file's `fitted_on.questions_hash` differs from the answers' hash.
* **The `I_evaluative` gain replicates.** R_ideo ladder alpha 0.551 -> 0.563 (v5 run 0.576); AUC 0.913.
* **Refitted on 142551** (`posthoc_quantile_2026-10-23_142551.json`), LOTO alpha:
  - R_top: ladder 0.690, with cut points 0.686.
  - R_ideo: ladder 0.563, with cut points 0.579. Holdout (fit on dev texts): R_ideo 0.608 -> 0.665,
    R_top 0.581 -> 0.513.
* **Recommended configuration:** built-in questions, R_top read off the ladder, R_ideo through the cut
  points (`posthoc_quantile_2026-10-23_142551_R_ideo.json`). On both runs, R_top cut points match the
  label shares but lose a little agreement (LOTO -0.004 / -0.012, holdout -0.07 / -0.06). Range report
  for this configuration (`2026-10-23_142551_ideo_cuts`): judge/human QWK ratio R_top 0.79, R_ideo 0.88
  (in-sample for R_ideo), A_caus 0.82.
* How the whole judge works now is described in `../HOW_THE_LLM_JUDGE_WORKS.md`.

## Round 2: asking Jev differently (three pre-planned experiments)

Question: would asking Jev more like the annotators were asked improve agreement? Adoption rule, fixed
before experiments 2 and 3: adopt a change for a dimension only if alpha on all chunks rises by at
least 0.05, holdout alpha does not fall by more than 0.03, and any fitted layer holds up
out-of-sample. Delta alpha below is B minus A with a paired 95 % CI that resamples texts
(`compare_arms.py`). Run-to-run noise of an identical set-up is about 0.015 (`exp1b_nometa` vs
`final_nometa`: R_top +0.015, R_ideo +0.001, A_caus +0.001).

| # | Change | Baseline | R_top delta alpha (95 % CI) | R_ideo delta alpha (95 % CI) | Verdict |
|---|---|---|---|---|---|
| 1a | `minimal`: content questions see only the two texts (no metadata, no justification, new task text) | `final` | -0.093 (-0.217, -0.007) | +0.052 (-0.013, +0.126) | rejected: hurts R_top |
| 1b | `nometa`: only the chunk's party / speaker / source removed | `final` | -0.002 (-0.038, +0.053) | **+0.069 (+0.002, +0.147)** | adopted (holdout R_ideo +0.103) |
| 2 | Real example pairs inside the yes/no criteria (dev texts only, 4 texts left out of scoring) | `exp1b_nometa` | -0.018 (-0.087, +0.029) | -0.004 (-0.039, +0.030) | no effect |
| 3 | Graded 1-5 questions with the help-page wording, as a second arm | `exp1b_nometa` | -0.109 (-0.196, -0.046) as label | -0.256 (-0.451, -0.067) as label | rejected |

* **1 (state).** Jev does use fields a question does not point to: removing metadata flips up to 13
  labels per gate. Metadata mainly inflated the party gate (`I_party` yes 49 to 37 chunks) and R_ideo
  bias (+0.22 to +0.11). Stripping the justification as well (1a) hurts R_top, so that framing is useful
  there. Only `nometa` passes the rule; its CI lower bound is +0.002, so the effect is real but small,
  and it is one of two variants tried on the same data.
* **2 (examples).** Twelve rule-selected examples (script `build_examples_questions.py`, list in
  `examples_used.json`) change nothing. The A_caus alpha moved by +0.02 in the same comparison although
  its request is identical, which is the noise floor.
* **3 (graded questions).** With the help-page wording verbatim, Jev over-rates: bias +0.32 (R_top) and
  +0.44 (R_ideo); 35 chunks the annotators call 4 get a 5. Expected score, ordinal mapping on the
  score, and score plus gates (leave-one-text-out) are all no better than the gates alone (best:
  combination, R_top -0.025, R_ideo -0.066). Adding the score questions did not change the gate answers
  (correlation at least 0.996).

Open (next step if the remaining gap matters): the 3-vs-4 boundary of R_ideo and the middle levels of
R_top (level 3 recall 13 %, level 4 recall 24 % in round 1) do not move with wording, examples, or
the graded format, so the remaining error looks like annotator disagreement or a construct difference
rather than a prompt problem.

## Version history

All rows: whole set / holdout alpha.

| Version | What changed | R_top | R_ideo | A_caus | Verdict |
|---|---|---|---|---|---|
| baseline (`2026-10-20_154122`, `fb516249cee42c73`) | ladder as found | .680 / .539 | .487 / .504 | .759 / .789 | starting point |
| v1 (`calib_v1`) | T_domain stricter, T_central about "same item and point", I_stance loosened | .683 / .603 | .280 / .269 | .742 / .789 | I_stance now yes for 121/125 chunks, rejected |
| v2 (`calib_v2`, `calib_v2_ladder`) | v1 R_top, original I_stance, plus a single 5-level R_ideo Score question as second arm | .662 / .528 | Score arm .101, ladder .486 | .727 / .789 | Score arm clearly worse (Spearman 0.36 vs 0.62), dropped |
| v3 (`calib_v3`) | I_stance reworded as "substantive policy position" | .664 / .525 | .413 / .416 | .759 / .789 | I_stance AUC .80 to .70, rejected |
| v4 (`calib_v4`) | v1 R_top with the T_domain fix for debt/tax topics, original R_ideo gates | .684 / .520 | .487 / .504 | .727 / .789 | adopted as built-in |
| final (`final`, `7c4f5961ce892ca3`) | v4 re-run with the built-in questions | .681 / .534 | .490 / .504 | .758 / .789 | frozen |
| final_posthoc | final + ordinal mapping for R_ideo (in-sample here) | .681 / .534 | .561 / .693 | .758 / .789 | superseded, see round 2 |
| final_nometa (`7c4f5961ce892ca3`, `--state-mode nometa`) | same questions, chunk metadata removed for R_top / R_ideo | .695 / .596 | .560 / .577 | .743 / .789 | **current** |

## What we learned

* **Wording moved diagnostics, not agreement.** v4 makes `T_domain` discriminative (it said yes for
  125/125 chunks before; AUC .905 to .935), so R_top = 1 is now predicted (recall 0.5, 11 predicted vs
  10 human) and R_top bias drops from +0.10 to +0.06, but alpha stays at 0.68 (CI 0.50-0.77).
* **The 3-vs-4 boundary of R_ideo is the limit.** Three differently worded `I_stance` gates all
  rank chunks the same way (AUC 0.70-0.80). Annotators give 3 to attacks on opponents, emotional
  appeals and government reporting, and 4 to substantive policy positions even in a calm tone, but
  they disagree among themselves there (alpha 0.63).
* **Run-to-run noise is small.** Identical wording gives gate probabilities that correlate at
  0.997-0.998 across runs (0 `I_stance` label flips). A_caus is the exception (5 flips near 0.5,
  kappa 0.727-0.759), so differences below about 0.03 are noise there.
* **Post-hoc calibration helps only R_ideo.** Out-of-sample (leave-one-text-out): R_top 0.68 stays
  best uncalibrated (global threshold 0.66, Platt 0.65, ordinal 0.66); R_ideo ordinal-logistic 0.55
  vs 0.49, and per-gate Platt / one global threshold give nothing.
* **A_caus errors are not systematic.** 17 errors (12 judge 1 / human 0, 5 the other way) come from
  5 texts whose justification names only a party ("aligns with DIE LINKE's reference points") without a
  chunk number. Annotators labelled chunks of the same party differently within a text (text 534:
  1, 0, 0), so no extra rule was added; the `needs_review` flag catches most of them.
* **Quadrants.** R_top agreement is much lower for Q2 (RAG Distraction, alpha about 0.5) than Q1
  (0.86): the judge is weakest exactly on the cases the study cares about.

## Caveats

* Only 25 texts: CIs are wide (R_top 0.50-0.77), the holdout has 8 texts, and the post-hoc mapping
  and review bands are fitted on the same consensus chunks that are used for evaluation (hence the
  leave-one-text-out figures).
* The earlier calibration folders (`calib_v0..calib_final_default`, `judge_calibration/questions_*.json`
  from before this session) are no longer on disk or in the git index, so they could not be
  re-scored. Their remembered Score-design numbers (R_ideo QWK about 0.61) could not be reproduced with
  the Score question that is still in `git show HEAD:src/run_llm_judge.py` (QWK 0.19, argmax collapses
  to level 4); the wording that produced 0.61 is lost.
