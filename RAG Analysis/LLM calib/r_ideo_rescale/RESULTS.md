# R_ideo scale collapse

Question: human R_ideo agreement is α = 0.63. Levels 1 and 2 are almost never used. Does α improve if
the scale is collapsed accordingly? The existing ratings are mapped onto the new scales; nothing is
re-annotated.

Reproduce: `python "RAG Analysis/r_ideo_rescale/r_ideo_rescale.py"`. Numbers are in
`2026-10-22_132304/r_ideo_rescale_report.json` and the console output next to it.

## Rating distribution (3 annotators × 125 chunks = 375 ratings)

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Revised | 0 | 8 | 93 | 198 | 76 |
| Original | 0 | 28 | 98 | 175 | 74 |
| Judge (125) | 0 | 0 | 27 | 65 | 33 |

Disagreeing annotator pairs (revised): 3↔4: 80, 4↔5: 42, 2↔3: 12, 3↔5: 2.

## Results

The α column is human Krippendorff α with a 95 % CI that resamples texts. Ordinal α is used for 3 or
more levels and nominal α for 2. The judge is compared with the median of the mapped ratings.

| Scheme | Human α (CI) | Human mean κ_w | Human α, original ratings | Judge vs consensus α | Judge exact | Alt-Test ω (ε = 0.2) |
|---|---|---|---|---|---|---|
| 5-level (as annotated) | 0.634 (0.51–0.73) | 0.646 | 0.314 | 0.551 | 0.62 | 0.33 |
| 4-level: 1+2 | 0.634 (0.50–0.74) | 0.646 | 0.314 | 0.551 | 0.62 | 0.33 |
| 3-level: 1+2+3 / 4 / 5 | 0.623 (0.50–0.72) | 0.633 | 0.304 | 0.546 | 0.63 | 0.33 |
| Binary: ≤3 vs 4+5 | 0.446 (0.30–0.59) | 0.454 | 0.201 | 0.298 | 0.75 | 1.00 |
| Binary: ≤4 vs 5 | 0.638 (0.48–0.77) | 0.638 | 0.278 | 0.615 | 0.86 | 1.00 |
| 3-level: 1+2 / 3+4 / 5 | 0.621 (0.46–0.74) | 0.612 | 0.284 | 0.605 | 0.85 | 1.00 |

(The judge's 5-level α of 0.551 differs slightly from the 0.560 in the calibration log, because that
uses the stored consensus file, which differs from the median in one chunk.)

## Interpretation

1. **Dropping the unused low end changes nothing.** Ordinal α depends only on the rank order and
   the observed frequency of each level. Level 1 is empty and level 2 holds 2 % of the ratings, so
   merging them leaves α at 0.634 (1+2) or 0.623 (1+2+3). The scale already behaves like a 3-level
   scale, and α measures it as one.
2. **Agreement is limited by the 3↔4 and 4↔5 boundaries, and merging them does not help either.**
   Merging 3+4 removes most observed disagreement. But 78 % of all ratings then fall into one
   category, so the disagreement expected by chance drops by the same amount. α is chance-corrected,
   so it stays at 0.62. Splitting at 3 | 4+5 is worse (0.45): that is exactly where annotators
   disagree most. Splitting at ≤4 | 5 ("explicit party grounding") is as good as the full scale
   (0.64) but no better.
3. **No scheme reaches 0.70.** All CIs overlap with the 5-level one. Every variant was also chosen
   after seeing the data, so even a small gain would not count as evidence.
4. **The higher exact agreement and Alt-Test ω on coarse scales are an artefact.** With 2–3 levels
   and one dominant category, ties become much more frequent. Ties count as a win for both sides in
   the Alt-Test, so the gap between ρ^f and ρ^h shrinks below ε. This does not show that the judge
   has become a better annotator.

Conclusion: the R_ideo reliability problem is not caused by unused scale levels. It comes from the
substantive 3 (implicit framing) vs 4 (clear stance) distinction, and to a lesser extent 4 vs 5,
which the annotators apply inconsistently. Improving it would need a clearer definition or
decision rule for that boundary and a new annotation round. The defensible way to report this: R_ideo
was used on a de-facto 3-level range (3–5), and its reliability (α = 0.63) is below the 0.70 rule
regardless of the level coding.
