# Chunk influence on the RAG predictions

Labels: **final**, sample tier: **reviewed**. Quadrants: correct = |error| ≤ 0.5, wrong = |error| ≥ 1.0, texts in between = buffer (not sampled).

Party positions (CHES 2024 lrgen, 1-7 scale; = ground truth definition): DIE LINKE 1.85, BÜNDNIS 90/DIE GRÜNEN 2.83, SPD 3.08, BSW 3.17, FDP 4.63, CDU 4.95, CSU 5.54, AfD 6.56

## 1. Base rates (full population)

The sample has 25 texts per quadrant, so it over-represents rare quadrants (above all Q2). Pooled sample numbers below are weighted by these base rates; buffer texts are missing from the sample entirely.

| model | texts | Q1 share | Q2 share | Q3 share | Q4 share | buffer share |
|---|---|---|---|---|---|---|
| llama_3b | 710 | 0.09 | 0.19 | 0.38 | 0.11 | 0.23 |
| llama_70b | 775 | 0.03 | 0.09 | 0.15 | 0.23 | 0.49 |
| llama_8b | 774 | 0.09 | 0.03 | 0.26 | 0.20 | 0.41 |
| mistral_14b | 770 | 0.05 | 0.06 | 0.18 | 0.35 | 0.37 |
| mistral_3b | 761 | 0.10 | 0.15 | 0.29 | 0.13 | 0.33 |
| mistral_8b | 760 | 0.12 | 0.06 | 0.35 | 0.21 | 0.26 |
| qwen_32b | 775 | 0.10 | 0.05 | 0.29 | 0.12 | 0.44 |
| qwen_3b | 773 | 0.01 | 0.03 | 0.54 | 0.02 | 0.40 |
| qwen_72b | 775 | 0.07 | 0.04 | 0.23 | 0.17 | 0.49 |
| qwen_7b | 773 | 0.12 | 0.07 | 0.37 | 0.17 | 0.28 |
| pooled | 7646 | 0.08 | 0.08 | 0.30 | 0.17 | 0.37 |

### Party composition per quadrant (share of the quadrant's texts)

Parties close to the scale centre are 'correct' for a constant centrist prediction, so quadrant membership partly reflects the party mix:

| input_party | Q1 | Q2 | Q3 | Q4 | buffer | all texts | baseline correct (\|err\| ≤ 0.5) | position |
|---|---|---|---|---|---|---|---|---|
| DIE LINKE | 0.12 | 0.12 | 0.31 | 0.10 | 0.28 | 0.23 | 0.15 | 1.85 |
| BÜNDNIS 90/DIE GRÜNEN | 0.14 | 0.27 | 0.18 | 0.21 | 0.21 | 0.20 | 0.35 | 2.83 |
| SPD | 0.29 | 0.28 | 0.20 | 0.27 | 0.35 | 0.28 | 0.33 | 3.08 |
| FDP | 0.14 | 0.08 | 0.09 | 0.04 | 0.06 | 0.07 | 0.22 | 4.63 |
| CDU | 0.07 | 0.03 | 0.05 | 0.06 | 0.07 | 0.06 | 0.31 | 4.95 |
| CSU | 0.01 | 0.01 | 0.02 | 0.00 | 0.00 | 0.01 | 0.10 | 5.54 |
| AfD | 0.23 | 0.21 | 0.16 | 0.31 | 0.04 | 0.15 | 0.48 | 6.56 |

| constant prediction | counted correct for |
|---|---|
| 1.00 | — |
| 1.50 | DIE LINKE |
| 2.00 | DIE LINKE |
| 2.50 | BÜNDNIS 90/DIE GRÜNEN |
| 3.00 | BÜNDNIS 90/DIE GRÜNEN, SPD, BSW |
| 3.50 | SPD, BSW |
| 4.00 | — |
| 4.50 | FDP, CDU |
| 5.00 | FDP, CDU |
| 5.50 | CSU |
| 6.00 | CSU |
| 6.50 | AfD |
| 7.00 | AfD |

## 2. Evidence use (full population, not selected on outcome)

R ~ B + E (+ model / + input party), E = mean position of the retrieved chunks' parties; SE clustered by text. The permutation null shuffles E across texts within each model (1000×). `naive_pull_corr` is the earlier corr(R − B, E − B); its null mean shows how much of it appears without any evidence use.

| model | coef | ci_low | ci_high | p | perm_null_mean | perm_null_95 | perm_p | B_coef | naive_pull_corr | naive_pull_corr_null_mean | n | r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 0.43 | 0.39 | 0.48 | <0.001 | 0.00 | 0.02 | <0.001 | 0.48 | 0.57 | 0.44 | 7646 | 0.46 |
| pooled + input party | 0.32 | 0.27 | 0.37 | <0.001 | -0.00 | 0.02 | <0.001 | 0.41 | 0.57 | 0.44 | 7646 | 0.49 |
| llama_3b | 0.12 | 0.04 | 0.20 | 0.004 | 0.00 | 0.07 | 0.004 | 0.32 | 0.79 | 0.78 | 710 | 0.36 |
| llama_70b | 0.06 | -0.06 | 0.18 | 0.321 | -0.00 | 0.07 | 0.073 | 0.48 | 0.47 | 0.48 | 775 | 0.31 |
| llama_8b | 0.65 | 0.53 | 0.76 | <0.001 | -0.00 | 0.09 | <0.001 | 0.35 | 0.72 | 0.60 | 774 | 0.44 |
| mistral_14b | 0.60 | 0.49 | 0.72 | <0.001 | -0.00 | 0.07 | <0.001 | 0.75 | 0.34 | 0.09 | 770 | 0.70 |
| mistral_3b | 0.66 | 0.54 | 0.78 | <0.001 | -0.00 | 0.10 | <0.001 | 0.33 | 0.67 | 0.53 | 761 | 0.35 |
| mistral_8b | 0.57 | 0.42 | 0.72 | <0.001 | 0.00 | 0.09 | <0.001 | 0.72 | 0.30 | 0.15 | 760 | 0.58 |
| qwen_32b | 0.48 | 0.40 | 0.56 | <0.001 | -0.00 | 0.06 | <0.001 | 0.56 | 0.49 | 0.19 | 775 | 0.48 |
| qwen_3b | 0.04 | 0.00 | 0.08 | 0.040 | -0.00 | 0.03 | 0.015 | 0.15 | 0.50 | 0.47 | 773 | 0.06 |
| qwen_72b | 0.43 | 0.36 | 0.49 | <0.001 | -0.00 | 0.05 | <0.001 | 0.81 | 0.36 | 0.02 | 775 | 0.73 |
| qwen_7b | 0.31 | 0.18 | 0.43 | <0.001 | -0.00 | 0.08 | <0.001 | 0.62 | 0.30 | 0.16 | 773 | 0.24 |

Reading: holding the baseline prediction fixed, a one-point shift of the retrieved parties' mean position moves the RAG prediction by 0.43 points (95 % CI 0.39 to 0.48); the permutation null is centred at 0.00. The naive pull correlation is 0.57, of which 0.44 arises with shuffled evidence.

## 3. Which chunks carry the effect (sample, weighted, descriptive)

Same regression on the sampled texts with the positions of cited vs. uncited and relevant vs. other chunks. The sample is selected on the outcome and lacks the buffer, so read this as descriptive.

| comparison | term | coef | ci_low | ci_high | p | n_texts |
|---|---|---|---|---|---|---|
| cited vs. uncited chunks | E_cited | 0.18 | 0.04 | 0.32 | 0.014 | 162 |
| cited vs. uncited chunks | E_uncited | -0.03 | -0.19 | 0.12 | 0.692 | 162 |
| relevant (R_top ≥ 4) vs. other chunks | E_relevant | 0.24 | 0.08 | 0.40 | 0.003 | 137 |
| relevant (R_top ≥ 4) vs. other chunks | E_irrelevant | 0.08 | -0.07 | 0.23 | 0.296 | 137 |

## 4. Quadrant descriptions (sample)

Quadrants are defined by the errors, so they describe the error cases, not different mechanisms (Q1 / Q2 are selected for large moves, Q4 for none). The last columns test Q1 vs. Q2 with input party and model as controls.

| feature | Q1 | Q2 | Q3 | Q4 | all (weighted, no buffer) | Q1 vs Q2: log-odds per SD | p (party + model controlled) |
|---|---|---|---|---|---|---|---|
| mean R_top | 2.60 | 2.92 | 2.83 | 2.99 | 3.03 | -0.37 | 0.083 |
| share chunks R_top ≥ 4 | 0.24 | 0.34 | 0.34 | 0.36 | 0.38 | -0.36 | 0.092 |
| mean R_ideo | 3.85 | 3.87 | 3.79 | 3.82 | 3.90 | -0.21 | 0.415 |
| share chunks R_ideo ≥ 4 | 0.74 | 0.75 | 0.67 | 0.70 | 0.74 | -0.11 | 0.678 |
| chunks cited (of 5) | 3.04 | 2.55 | 3.07 | 3.03 | 2.96 | -0.15 | 0.626 |
| R_top of cited chunks | 2.80 | 3.08 | 2.90 | 3.14 | 3.11 | -0.27 | 0.209 |
| R_ideo of cited chunks | 3.90 | 3.84 | 3.78 | 3.89 | 3.88 | 0.35 | 0.228 |
| share chunks from input's party | 0.28 | 0.30 | 0.28 | 0.33 | 0.33 | 0.02 | 0.937 |
| share cited chunks from input's party | 0.38 | 0.36 | 0.26 | 0.42 | 0.37 | 0.02 | 0.945 |
| |cited chunks' position − ground truth| | 1.00 | 0.95 | 1.44 | 0.85 | 1.07 | 0.02 | 0.916 |

### Share of cited chunks from the input's own party, by input party (Q1 vs. Q2)

| input_party | Q1 mean | Q2 mean | Q1 count | Q2 count | position |
|---|---|---|---|---|---|
| DIE LINKE | 0.30 | — | 5.00 | — | 1.85 |
| BÜNDNIS 90/DIE GRÜNEN | 0.31 | 0.14 | 9.00 | 20.00 | 2.83 |
| SPD | 0.34 | 0.09 | 4.00 | 9.00 | 3.08 |
| FDP | 0.33 | 0.40 | 9.00 | 6.00 | 4.63 |
| CDU | 0.23 | 0.46 | 5.00 | 4.00 | 4.95 |
| CSU | 0.00 | — | 3.00 | — | 5.54 |
| AfD | 0.60 | 0.68 | 15.00 | 19.00 | 6.56 |

### Utilisation: share of chunks cited by label level

| dimension | level | n chunks | cited (weighted) | cited Q1 | cited Q2 | cited Q3 | cited Q4 |
|---|---|---|---|---|---|---|---|
| R_top | 1 | 207 | 0.48 | 0.58 | 0.27 | 0.63 | 0.24 |
| R_top | 2 | 472 | 0.60 | 0.50 | 0.51 | 0.61 | 0.62 |
| R_top | 3 | 227 | 0.60 | 0.65 | 0.59 | 0.57 | 0.67 |
| R_top | 4 | 189 | 0.63 | 0.79 | 0.57 | 0.69 | 0.64 |
| R_top | 5 | 245 | 0.60 | 0.70 | 0.58 | 0.56 | 0.71 |
| R_ideo | 2 | 14 | 0.65 | — | 0.50 | 0.60 | 0.20 |
| R_ideo | 3 | 367 | 0.61 | 0.61 | 0.54 | 0.64 | 0.51 |
| R_ideo | 4 | 789 | 0.60 | 0.59 | 0.50 | 0.63 | 0.63 |
| R_ideo | 5 | 170 | 0.53 | 0.68 | 0.48 | 0.49 | 0.75 |

## 5. Noise floor

Not available: every model has one baseline run (greedy decoding, temperature 0). Re-run the no-RAG baseline once per model and pass it with `--baseline-rerun <model>=<log>`; the share of texts that land in 'Q1' / 'Q2' between two baseline runs is the part of the RAG quadrants explained by run-to-run variation (batching / FP8 non-determinism).

![base_rates.png](base_rates.png)
![evidence_use.png](evidence_use.png)
![labels_by_quadrant.png](labels_by_quadrant.png)
