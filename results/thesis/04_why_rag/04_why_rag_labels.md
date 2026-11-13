# Why retrieval helps or hurts: what the chunk labels show

## Data

Sample: `final_annotations.jsonl` of the ten models, 4840 chunks of 968 (model, text) pairs (Llama 3B 100, Llama 8B 100, Llama 70B 97, Ministral 3B 100, Ministral 8B 100, Ministral 14B 100, Qwen 3B 71, Qwen 7B 100, Qwen 32B 100, Qwen 72B 100). Quadrant = the quadrant each text was sampled in. All shares and means are weighted by the quadrant base rates of each model (population texts / sampled texts, Q1-Q4 only; the buffer is not sampled) unless marked unweighted. CIs: 2000 bootstrap resamples of text_index clusters (the same texts appear in several models), seed 42. Positions for sections 2-5: CHES 2024 lrgen, two decimals, CDU/CSU faction = CDU value; chunks of non-attached speakers count as 'no position' and never as the author's party. Section 1 re-runs `analyze_chunk_influence.py` unchanged (its own exact lrgen positions). Every table is also computed without the Llama 3B texts with a baseline score < 1 (column 'Llama 3B posts' in the CSVs). The sample is selected on the outcome, so everything here is descriptive.

## 1. Tier 'all' vs. tier 'reviewed'

`analyze_chunk_influence.py --labels final --tier all|reviewed`, output in `chunk_influence_all/` and `chunk_influence_reviewed/`. The tier 'reviewed' keeps every annotated text of the three models with a human review (Llama 70B, Ministral 14B, Qwen 3B), including their unreviewed Q3/Q4 texts; it is not a filter on reviewed labels. The table refits the same analyses on model subsets:

| Subset | Sampled texts | Population evidence use (E coef) | E_cited coef | E_cited CI low | E_cited CI high | Q1 mean, chunks from input's party | Q2 mean, chunks from input's party | p, chunks from input's party | Q1 texts | Q2 texts |
|---|---|---|---|---|---|---|---|---|---|---|
| All 10 models (tier 'all') | 968 |  | 0.3485 | 0.2546 | 0.4423 | 0.327 | 0.2645 | 0.0297 | 230 | 248 |
| 3 reviewed models (tier 'reviewed') | 268 |  | 0.1807 | 0.0365 | 0.325 | 0.2836 | 0.3014 | 0.9367 | 55 | 73 |
| 3 reviewed models, raw judge labels | 268 |  | 0.1887 | 0.0438 | 0.3335 | 0.2836 | 0.3014 | 0.9367 | 55 | 73 |
| 7 other models | 700 |  | 0.4101 | 0.295 | 0.5251 | 0.3406 | 0.2491 | 0.0029 | 175 | 175 |
| Llama 3B only | 100 | 0.1193 | 0.1379 | -0.095 | 0.3707 | 0.24 | 0.28 | 0.094 | 25 | 25 |
| Llama 8B only | 100 | 0.6478 | 0.3784 | 0.0586 | 0.6981 | 0.24 | 0.128 | 0.251 | 25 | 25 |
| Llama 70B only | 97 | 0.0597 | 0.0397 | -0.2572 | 0.3366 | 0.2273 | 0.392 | 0.9559 | 22 | 25 |
| Ministral 3B only | 100 | 0.6599 | 0.6193 | 0.3604 | 0.8783 | 0.288 | 0.352 | 0.5761 | 25 | 25 |
| Ministral 8B only | 100 | 0.5689 | 0.805 | 0.3784 | 1.2317 | 0.296 | 0.272 | 0.4186 | 25 | 25 |
| Ministral 14B only | 100 | 0.605 | 0.547 | 0.0538 | 1.0401 | 0.36 | 0.184 | 0.0172 | 25 | 25 |
| Qwen 3B only | 71 | 0.0395 | 0.106 | -0.0215 | 0.2335 | 0.2 | 0.3304 | 0.291 | 8 | 23 |
| Qwen 7B only | 100 | 0.3073 | 0.097 | -0.1312 | 0.3252 | 0.424 | 0.248 | 0.4592 | 25 | 25 |
| Qwen 32B only | 100 | 0.4779 | 0.4344 | 0.2196 | 0.6492 | 0.472 | 0.24 | 0.1357 | 25 | 25 |
| Qwen 72B only | 100 | 0.4274 | 0.3667 | 0.2034 | 0.5299 | 0.424 | 0.224 | 0.0595 | 25 | 25 |
| Difference: 3 reviewed − 7 other models (interaction E_cited × reviewed) | 636 |  | -0.1641 | -0.3244 | -0.0039 |  |  |  |  |  |

Reading: the two tiers differ because they contain different models, not because of the human review. Tier 'all' has 968 texts from 10 models, tier 'reviewed' 268 texts from Llama 70B, Ministral 14B, Qwen 3B. On the same three models, the raw judge labels give practically the same E_cited as the reviewed labels (0.19 vs 0.18) and an identical result (the Q1-vs-Q2 share of chunks from the author's party uses no judge label at all, so it is identical by construction); the review does not move these results. The difference comes from the model selection: Llama 70B and Qwen 3B are among the 3 models with almost no evidence use in the population (E coefficient < 0.15: Llama 3B, Llama 70B, Qwen 3B), and two of the three reviewed models are of that kind. The seven other models give E_cited 0.41 [0.30, 0.53], the reviewed three 0.18 [0.04, 0.32]. The two CIs of the tiers overlap (0.35 [0.25, 0.44] vs 0.18), but a direct test of reviewed vs. other models gives a difference of -0.16 [-0.32, -0.00] (p 0.045). The Q1-vs-Q2 difference in the share of chunks from the author's party is likewise carried by the seven other models (p 0.003); among the reviewed models, Llama 70B points the other way (Q1 0.23 vs Q2 0.39) and Qwen 3B has only 8 Q1 texts.

Recommendation: report tier 'all' as the main result. It covers every model, so it describes the same population as the evidence-use analysis, and the comparison above shows that its labels are not worse in a way that matters: the human review changes E_cited by less than 0.01 on the reviewed models. As robustness checks, report (a) the reviewed models with raw judge vs. reviewed labels (isolates the review) and (b) the tier 'reviewed' with the note that it is a selection of models, two of which barely use evidence. The tier 'reviewed' should not be the main result, because its lower E_cited reflects which models were reviewed, not better labels. This recommendation rests on E_cited and the Q1/Q2 split; R_top-based results (E_relevant) remain the least reliable label and should still be read with the ±1 caveat of the judge chapter.

## 2. Labels by the chunk's party

| Grouping | Group | Chunks | Mean R_top [95 % CI] | Mean R_ideo [95 % CI] | Share cited [95 % CI] |
|---|---|---|---|---|---|
| Chunk party | DIE LINKE | 563 | 3.47 [3.25, 3.67] | 4.21 [4.11, 4.30] | 0.60 [0.53, 0.68] |
| Chunk party | Grüne | 870 | 2.90 [2.73, 3.10] | 3.76 [3.69, 3.82] | 0.50 [0.45, 0.55] |
| Chunk party | SPD | 1127 | 2.78 [2.61, 2.96] | 3.55 [3.49, 3.62] | 0.57 [0.52, 0.62] |
| Chunk party | BSW | 14 | 2.92 [1.40, 4.64] | 4.06 [3.31, 4.81] | 0.43 [0.07, 0.80] |
| Chunk party | FDP | 546 | 3.13 [2.87, 3.37] | 3.82 [3.75, 3.90] | 0.57 [0.51, 0.64] |
| Chunk party | CDU/CSU | 1015 | 2.81 [2.64, 2.98] | 3.61 [3.53, 3.68] | 0.55 [0.50, 0.59] |
| Chunk party | AfD | 679 | 2.98 [2.74, 3.22] | 4.27 [4.18, 4.34] | 0.66 [0.59, 0.72] |
| Chunk party | no position | 26 | 2.83 [2.35, 3.33] | 3.90 [3.74, 4.00] | 0.35 [0.14, 0.57] |
| Chunk bloc | Left | 2574 | 3.00 [2.86, 3.14] | 3.79 [3.73, 3.84] | 0.56 [0.52, 0.59] |
| Chunk bloc | Centre | 1561 | 2.93 [2.76, 3.08] | 3.68 [3.62, 3.74] | 0.56 [0.52, 0.60] |
| Chunk bloc | Right | 679 | 2.98 [2.74, 3.22] | 4.27 [4.18, 4.34] | 0.66 [0.59, 0.72] |
| Chunk from the author's party | yes | 1437 | 3.21 [3.05, 3.36] | 3.97 [3.89, 4.04] | 0.66 [0.62, 0.70] |
| Chunk from the author's party | no | 3403 | 2.87 [2.74, 3.00] | 3.76 [3.71, 3.80] | 0.53 [0.50, 0.56] |

Reading: speeches of the two outer parties are rated most ideological (AfD 4.27, DIE LINKE 4.21), those of SPD (3.55) and CDU/CSU (3.61) least; the CIs of the two outer parties do not overlap with those of SPD and CDU/CSU. The judge rates R_ideo without seeing the speaker's party, so this reflects the text. R_top differs less between parties (2.78-3.47). AfD chunks are cited most often (0.66). Chunks from the author's own party are more relevant (3.21 vs 2.87), more ideological (3.97 vs 3.76) and cited more often (0.66 vs 0.53). BSW (14 chunks) and 'no position' (26) are too small to interpret. Without the Llama 3B texts with a baseline score < 1, the values of the groups with at least 100 chunks change by at most 0.01.

## 3. Relevant chunks and the author's party

| Author party | relevant: share own party | other: share own party | difference | p (Holm) | chunks relevant / other |
|---|---|---|---|---|---|
| All posts | 0.35 [0.30, 0.39] | 0.27 [0.23, 0.30] | +0.08 [+0.03, +0.13] |  | 1751 / 3089 |
| DIE LINKE | 0.44 [0.35, 0.53] | 0.26 [0.18, 0.36] | +0.17 [+0.06, +0.29] | 0.042 | 271 / 324 |
| Grüne | 0.26 [0.19, 0.34] | 0.23 [0.17, 0.31] | +0.03 [-0.07, +0.14] | 1.000 | 426 / 569 |
| SPD | 0.36 [0.27, 0.44] | 0.30 [0.25, 0.35] | +0.06 [-0.03, +0.16] | 0.804 | 370 / 865 |
| FDP | 0.26 [0.13, 0.40] | 0.16 [0.09, 0.24] | +0.11 [-0.03, +0.25] | 0.604 | 185 / 345 |
| CDU/CSU | 0.31 [0.14, 0.54] | 0.20 [0.12, 0.29] | +0.11 [-0.06, +0.31] | 0.804 | 86 / 299 |
| AfD | 0.38 [0.27, 0.49] | 0.33 [0.26, 0.41] | +0.04 [-0.08, +0.17] | 1.000 | 413 / 687 |

Reading: yes, but modestly. Of the relevant chunks (R_top ≥ 4), 35 % come from the author's party, of the others 27 % (difference +0.08 [+0.03, +0.13]). After Holm correction across the author parties the difference is significant for DIE LINKE; the other parties point the same way with wide CIs.

## 4. Labels and the change in error (text level)

dAE = |e_RAG| − |e_baseline| per (model, text); `dAE ~ mean_R_top + mean_R_ideo + C(party, Treatment('SPD')) + C(model)`, clustered by text. Negative coefficient = higher label goes with a larger error reduction.

| Specification | Weighting | Term | Coefficient [95 % CI] | p | Texts |
|---|---|---|---|---|---|
| mean R_top only | weighted by quadrant base rates | mean R top | +0.098 [+0.025, +0.171] | 0.008 | 968 |
| mean R_top only | unweighted | mean R top | +0.107 [+0.010, +0.204] | 0.031 | 968 |
| mean R_ideo only | weighted by quadrant base rates | mean R ideo | +0.221 [+0.026, +0.416] | 0.026 | 968 |
| mean R_ideo only | unweighted | mean R ideo | +0.274 [+0.008, +0.540] | 0.044 | 968 |
| both | weighted by quadrant base rates | mean R top | +0.088 [+0.016, +0.160] | 0.017 | 968 |
| both | weighted by quadrant base rates | mean R ideo | +0.180 [-0.012, +0.373] | 0.067 | 968 |
| both | unweighted | mean R top | +0.096 [-0.000, +0.193] | 0.050 | 968 |
| both | unweighted | mean R ideo | +0.239 [-0.025, +0.503] | 0.076 | 968 |

Reading: higher labels go with a smaller RAG benefit, not a larger one. With both labels in the model (weighted, input party and model controlled), one point of mean R_top goes with a +0.088 change in dAE [+0.016, +0.160] and one point of mean R_ideo with +0.180 [-0.012, +0.373]. The sample is selected on dAE (Q1 and Q2 are defined by large changes, the buffer is missing), so this is a description of the error cases, not an effect: relevant, ideological chunks are about as common in RAG distractions as in RAG wins, which fits the Q1-vs-Q2 profiles in `chunk_influence_all/summary.md` (Q2 has the higher R_top).

## 5. Cited chunks: relevant × same party, Q1 vs. Q2

Q1/Q2 texts with at least one cited chunk in the cell; 'Share Q1' = share of these texts that are RAG wins rather than distractions (weighted by the base rates of Q1 and Q2).

| Cited chunks in cell | Q1 texts | Q2 texts | Share Q1 (weighted) | CI low | CI high | Share Q1 (unweighted) | Cited chunks in Q1 texts | Cited chunks in Q2 texts |
|---|---|---|---|---|---|---|---|---|
| any cited chunk (reference) | 202 | 200 | 0.5227 | 0.4617 | 0.5769 | 0.5025 | 656 | 646 |
| relevant × same party | 65 | 53 | 0.5615 | 0.4633 | 0.6602 | 0.5508 | 100 | 78 |
| relevant × other party | 82 | 96 | 0.4729 | 0.3865 | 0.5601 | 0.4607 | 134 | 195 |
| not relevant × same party | 101 | 68 | 0.5855 | 0.5026 | 0.6651 | 0.5976 | 169 | 104 |
| not relevant × other party | 125 | 123 | 0.5373 | 0.4616 | 0.6038 | 0.504 | 253 | 269 |

Reading: among Q1/Q2 texts with any cited chunk, 52 % are RAG wins (weighted). Texts that cite a chunk from the author's party are slightly more often wins (56 % relevant, 59 % not relevant), texts that cite a relevant chunk from another party slightly more often distractions (47 %). All CIs overlap with each other and with the reference, and a text can fall into several cells, so this is a weak pattern: if anything, the party of a cited chunk matters more than its relevance.
