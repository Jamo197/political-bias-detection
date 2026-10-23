# Alt-Test: can the Jev judge replace a human annotator?

Alternative Annotator Test (Calderon, Reichart & Dror, 2025) for the final calibrated judge run
`2026-10-22_132304` (`jev-1.14.0`, questions `7c4f5961ce892ca3`, `nometa`) against the three
annotators (Awsam, Jannes, Neda) on 125 chunks / 25 texts.

Reproduce: `python "RAG Analysis/alt_test/alt_test.py"` (`--run`, `--epsilon`). Outputs in
`2026-10-22_132304/`: `alt_test_report.json` (all numbers), `alt_test_output.txt` (console),
`alt_test_items.csv` (per chunk × left-out annotator: labels, alignment scores, W^f / W^h).

## Set-up

* Leave-one-out over the 3 annotators. Alignment score: negative RMSE to the two remaining annotators
  for R_top / R_ideo (Likert), accuracy for A_caus (binary). Ties count as a win for both sides.
* One-sided paired t-test on d_i = W^f_i − W^h_i against H0: ρ^f ≤ ρ^h − ε (n = 125 ≥ 30, so t-test).
* Benjamini-Yekutieli over the 3 annotators, q = 0.05; winning rate ω = share of rejected H0;
  pass if ω ≥ 0.5.
* ε = 0.2 (cost-benefit margin the paper suggests for expert annotators; 0.1 for crowd-workers).
* Human labels: **revised** per-annotator ratings after the discussion round (same table as
  `human_annotations/human_agreement_report_revised.json`). Judge labels: `derived_labels`.

## Main result (revised ratings, ε = 0.2)

| Dimension | ω | Verdict | avg ρ^f | ρ^f / ρ^h per annotator (Awsam, Jannes, Neda) | Smallest passing ε |
|---|---|---|---|---|---|
| R_top | 0.00 | fail | 0.699 | .70/.91, .70/.90, .70/.84 | 0.32 |
| R_ideo | 0.33 | fail | 0.821 | **.84/.79**, .81/.95, .82/.94 | 0.22 |
| A_caus | 1.00 | **pass** | 0.896 | .90/.98, .90/.98, .90/.98 | 0.15 |

* **R_top:** for every annotator, the excluded human matches the other two better than the judge does
  (gap 0.14–0.22). The judge does not pass for any ε ≤ 0.3.
* **R_ideo:** the judge beats Awsam (ρ^f > ρ^h) but not Jannes or Neda (gap ≈ 0.12–0.14). It just
  misses at ε = 0.2 and passes from ε = 0.22 (ω = 1.0 at 0.25).
* **A_caus:** passes for all three annotators (p_BY = 0.0003). The gap is ≈ 0.08–0.09, mostly from
  13 chunks where the two remaining annotators agree and the judge differs.

## Robustness (ε = 0.2)

| Variant | R_top ω | R_ideo ω | A_caus ω | Note |
|---|---|---|---|---|
| Main (125 chunks, −RMSE / ACC) | 0.00 | 0.33 | 1.00 | |
| Exact-match (ACC) for Likert dims | 0.00 | 0.33 | – | same picture |
| Text as unit (25 texts, Wilcoxon) | 0.00 | 0.33 | 1.00 | chunks of a text are not independent; same verdicts |
| Calibration holdout only (8 texts, 40 chunks) | 0.00 | 0.00 | 0.00 | A_caus gap unchanged (ρ^f .875 vs ρ^h .975) but p_BY = 0.09; n = 40 is underpowered |
| **Original (pre-discussion) ratings** | **1.00** | **1.00** | **1.00** | passes even at ε = 0 (R_top, R_ideo) / ε = 0.09 (A_caus) |

## Interpretation and caveats

1. **The verdict depends on which human ratings are used.** The original ratings agree poorly
   (Krippendorff α: R_top 0.33, R_ideo 0.31, A_caus 0.68). After the discussion round they agree
   much better (0.82 / 0.63 / 0.88). Against the independent originals, the judge beats every single
   annotator. Against the revised ratings, it only replaces annotators for A_caus.
   * The Alt-Test assumes independent annotators. Revised ratings are not independent, because each
     annotator saw the others' views. That inflates ρ^h and biases the test **against** the LLM.
   * The original ratings contain errors that the discussion fixed. That flatters the LLM.
   * Reported here: the revised ratings as the conservative primary result, the originals as an
     upper bound. A defensible wording: the judge is at least as good as a single annotator working
     alone on all three dimensions, but only matches discussed and adjudicated annotations on A_caus.
2. **In-sample.** The judge's questions were calibrated on these 125 chunks against the consensus
   (see `judge_calibration/CALIBRATION_LOG.md`), so all ω values are optimistic. On the 40 holdout
   chunks the A_caus effect has the same size but is not significant. That holdout is too small to
   confirm or refute the result.
3. **3 annotators** is the minimum the procedure needs. With m = 3, ω only takes the values
   0, 0.33, 0.67 and 1, and each leave-one-out comparison has just two humans as the reference.
4. Suggested use: let the judge annotate **A_caus** alone. For **R_top** and **R_ideo**, keep a
   human in the loop, e.g. the `needs_review` routing from the calibration log.
