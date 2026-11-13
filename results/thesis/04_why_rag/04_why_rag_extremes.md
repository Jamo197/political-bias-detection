# Why retrieval helps or hurts: the extreme parties

## Data

Paired e5/simple vs. no-RAG posts per model (Llama 3B 710, Llama 8B 774, Llama 70B 775, Ministral 3B 761, Ministral 8B 760, Ministral 14B 770, Qwen 3B 773, Qwen 7B 773, Qwen 32B 775, Qwen 72B 775). E = mean position of the retrieved chunks' parties (CHES 2024 lrgen, two decimals; CDU/CSU faction = CDU value; chunks without a position left out; 0 texts without any positioned chunk). Subsets: DIE LINKE posts, AfD posts and the posts of all other parties (Grüne, SPD, FDP, CDU, CSU). Regressions: OLS, standard errors clustered by text (as in `analyze_chunk_influence.py`); pooled fits add model fixed effects. Judge labels: `final_annotations.jsonl`, weighted by the quadrant base rates of each model, CIs from 2000 bootstrap resamples of texts within quadrants (seed 42). Llama 3B: all paired posts and posts with a baseline score ≥ 1. Cited chunks in the case listings: final A_caus label where the text was annotated for that model, otherwise the deterministic citation rule of the judge (chunk number or speaker name in the justification).

## 1. Evidence use by subset

Coefficient on E in R ~ B + E [95 % CI], fitted per subset:

| Model | DIE LINKE | AfD | other parties |
|---|---|---|---|
| Llama 3B (all) | 0.02 [-0.13, 0.17] | 0.10 [-0.01, 0.20] | 0.09 [-0.07, 0.25] |
| Llama 3B (baseline ≥ 1) | 0.02 [-0.14, 0.17] | 0.10 [-0.01, 0.20] | 0.17 [0.00, 0.34] |
| Llama 8B | 0.56 [0.35, 0.76] | 0.51 [0.23, 0.79] | 0.38 [0.20, 0.55] |
| Llama 70B | 0.24 [0.11, 0.36] | -0.58 [-0.98, -0.18] | 0.22 [0.10, 0.33] |
| Ministral 3B | 0.76 [0.61, 0.92] | 0.22 [-0.15, 0.59] | 0.57 [0.37, 0.77] |
| Ministral 8B | 0.74 [0.48, 1.00] | 0.25 [0.03, 0.46] | 0.36 [0.14, 0.58] |
| Ministral 14B | 0.45 [0.30, 0.60] | 0.68 [0.43, 0.94] | 0.46 [0.32, 0.59] |
| Qwen 3B | -0.00 [-0.07, 0.06] | 0.15 [0.03, 0.26] | 0.01 [-0.03, 0.06] |
| Qwen 7B | 0.03 [-0.22, 0.28] | 0.56 [0.32, 0.79] | 0.08 [-0.09, 0.25] |
| Qwen 32B | 0.45 [0.35, 0.55] | 0.31 [0.07, 0.54] | 0.32 [0.23, 0.42] |
| Qwen 72B | 0.49 [0.37, 0.60] | 0.35 [0.17, 0.53] | 0.32 [0.24, 0.39] |
| All models (model fixed effects) | 0.42 [0.35, 0.49] | 0.25 [0.12, 0.38] | 0.31 [0.24, 0.39] |

Difference of the E coefficient from the other parties (interaction model):

| Model | E coefficient, AfD − other parties | E coefficient, DIE LINKE − other parties |
|---|---|---|
| Llama 3B (all) | +0.01 [-0.18, +0.20], p_Holm 1.000 | -0.07 [-0.29, +0.15], p_Holm 1.000 |
| Llama 3B (baseline ≥ 1) | -0.08 [-0.28, +0.12], p_Holm 0.458 | -0.15 [-0.38, +0.08], p_Holm 0.394 |
| Llama 8B | +0.14 [-0.19, +0.46], p_Holm 0.418 | +0.18 [-0.09, +0.45], p_Holm 0.375 |
| Llama 70B | -0.79 [-1.20, -0.38], p_Holm <0.001 | +0.02 [-0.15, +0.19], p_Holm 0.796 |
| Ministral 3B | -0.35 [-0.77, +0.07], p_Holm 0.213 | +0.20 [-0.05, +0.45], p_Holm 0.213 |
| Ministral 8B | -0.11 [-0.42, +0.19], p_Holm 0.470 | +0.38 [+0.04, +0.72], p_Holm 0.056 |
| Ministral 14B | +0.23 [-0.06, +0.51], p_Holm 0.255 | -0.01 [-0.21, +0.20], p_Holm 0.951 |
| Qwen 3B | +0.13 [+0.01, +0.26], p_Holm 0.065 | -0.02 [-0.10, +0.06], p_Holm 0.695 |
| Qwen 7B | +0.47 [+0.19, +0.76], p_Holm 0.002 | -0.06 [-0.36, +0.24], p_Holm 0.714 |
| Qwen 32B | -0.02 [-0.27, +0.23], p_Holm 0.881 | +0.12 [-0.01, +0.26], p_Holm 0.155 |
| Qwen 72B | +0.03 [-0.16, +0.23], p_Holm 0.725 | +0.17 [+0.03, +0.31], p_Holm 0.032 |
| All models (model fixed effects) | -0.08 [-0.22, +0.07], p_Holm 0.303 | +0.08 [-0.02, +0.18], p_Holm 0.252 |

Reading: pooled over models, the pull towards the retrieved parties is similar for the three subsets (E coefficient DIE LINKE 0.42, AfD 0.25, other parties 0.31; the pooled contrasts are not significant). Within single models, the E coefficient of an extreme party differs from the other parties after Holm correction in: Llama 70B (AfD -0.79), Qwen 7B (AfD +0.47), Qwen 72B (DIE LINKE +0.17). Llama 70B is the only model with a negative coefficient for AfD posts (-0.58 [-0.98, -0.18]): among AfD posts, the more AfD-heavy the retrieved evidence, the further left its RAG prediction. This is the opposite of evidence following.

## 2. Where the evidence points

| Author party | Posts | Mean target | Mean E | SD of E | E - target | CI low | CI high | MAE of E as prediction | Share of posts with E in the target's bloc |
|---|---|---|---|---|---|---|---|---|---|
| All posts | 775 | 3.5019 | 3.828 | 0.8519 | 0.3261 | 0.2266 | 0.4233 | 1.1618 | 0.4155 |
| DIE LINKE | 179 | 1.85 | 3.3901 | 0.9403 | 1.5401 | 1.4042 | 1.6737 | 1.5401 | 0.5475 |
| Grüne | 156 | 2.83 | 3.6954 | 0.6524 | 0.8654 | 0.7577 | 0.965 | 0.9124 | 0.3397 |
| SPD | 217 | 3.08 | 3.7884 | 0.6508 | 0.7084 | 0.6228 | 0.7985 | 0.7661 | 0.2719 |
| FDP | 58 | 4.63 | 3.9094 | 0.6349 | -0.7206 | -0.8867 | -0.5526 | 0.798 | 0.7241 |
| CDU/CSU | 51 | 5.031 | 4.0056 | 0.554 | -1.0254 | -1.1968 | -0.8602 | 1.0254 | 0.7255 |
| AfD | 114 | 6.56 | 4.6515 | 0.9015 | -1.9085 | -2.0804 | -1.7538 | 1.9085 | 0.2895 |

Reading: the evidence of the extreme parties lies far towards the centre. For AfD posts E averages 4.65, 1.91 points left of the target, and lies in the right bloc for only 29 % of the posts; for DIE LINKE posts E averages 3.39, 1.54 points right of the target (left bloc in 55 %). A model that predicted E itself would have an MAE of 1.91 on AfD and 1.54 on DIE LINKE posts, worse than the baseline of 8 (AfD) and 5 (DIE LINKE) of the 10 models. Following the evidence therefore has to cost accuracy at the two ends, in proportion to the model's evidence use.

| Model | Author party | Mean target | Mean baseline | Mean E | Mean RAG | Mean RAG predicted by the all-posts fit | RAG - predicted | MAE baseline | MAE RAG | MAE if prediction = E |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | DIE LINKE | 1.85 | 3.658 | 3.3874 | 5.0136 | 4.8758 | 0.1378 | 2.0975 | 3.1636 | 1.5374 |
| Llama 3B (all) | AfD | 6.56 | 5.4383 | 4.6722 | 5.957 | 5.591 | 0.366 | 1.1665 | 0.603 | 1.8878 |
| Llama 3B (baseline ≥ 1) | DIE LINKE | 1.85 | 3.8732 | 3.3538 | 5.0614 | 4.9107 | 0.1508 | 2.1121 | 3.2114 | 1.5038 |
| Llama 3B (baseline ≥ 1) | AfD | 6.56 | 5.4383 | 4.6722 | 5.957 | 5.6243 | 0.3327 | 1.1665 | 0.603 | 1.8878 |
| Llama 8B | DIE LINKE | 1.85 | 3.2326 | 3.3988 | 2.8854 | 3.0466 | -0.1612 | 1.5601 | 1.1315 | 1.5488 |
| Llama 8B | AfD | 6.56 | 5.95 | 4.6515 | 5.7526 | 4.8061 | 0.9465 | 0.7216 | 0.8116 | 1.9085 |
| Llama 70B | DIE LINKE | 1.85 | 2.7095 | 3.3901 | 2.7849 | 3.049 | -0.264 | 0.883 | 0.9701 | 1.5401 |
| Llama 70B | AfD | 6.56 | 5.6912 | 4.6515 | 4.1754 | 4.5414 | -0.366 | 0.873 | 2.3888 | 1.9085 |
| Ministral 3B | DIE LINKE | 1.85 | 3.3017 | 3.3901 | 2.5324 | 2.9242 | -0.3918 | 1.6556 | 1.0506 | 1.5401 |
| Ministral 3B | AfD | 6.56 | 5.9434 | 4.646 | 4.877 | 4.6326 | 0.2444 | 0.9855 | 1.8458 | 1.914 |
| Ministral 8B | DIE LINKE | 1.85 | 2.8506 | 3.3818 | 2.627 | 2.888 | -0.261 | 1.3584 | 1.4197 | 1.5318 |
| Ministral 8B | AfD | 6.56 | 6.0496 | 4.6524 | 6.4673 | 5.9055 | 0.5618 | 0.8892 | 0.7515 | 1.9076 |
| Ministral 14B | DIE LINKE | 1.85 | 2.4486 | 3.3884 | 2.2955 | 2.4458 | -0.1503 | 0.7477 | 0.7438 | 1.5384 |
| Ministral 14B | AfD | 6.56 | 5.5044 | 4.6515 | 5.9711 | 5.5159 | 0.4551 | 1.2391 | 0.8879 | 1.9085 |
| Qwen 3B | DIE LINKE | 1.85 | 4.0419 | 3.3901 | 4.1536 | 4.1371 | 0.0165 | 2.1919 | 2.3036 | 1.5401 |
| Qwen 3B | AfD | 6.56 | 4.0746 | 4.6515 | 4.2412 | 4.192 | 0.0493 | 2.4854 | 2.3188 | 1.9085 |
| Qwen 7B | DIE LINKE | 1.85 | 3.5826 | 3.3815 | 3.8697 | 3.8415 | 0.0281 | 1.7365 | 2.1107 | 1.5315 |
| Qwen 7B | AfD | 6.56 | 4.6544 | 4.6515 | 5.6886 | 4.895 | 0.7936 | 1.9098 | 0.9612 | 1.9085 |
| Qwen 32B | DIE LINKE | 1.85 | 3.3101 | 3.3901 | 2.7709 | 3.0923 | -0.3213 | 1.4601 | 1.0539 | 1.5401 |
| Qwen 32B | AfD | 6.56 | 4.7491 | 4.6515 | 4.8175 | 4.5044 | 0.3131 | 1.8151 | 1.7425 | 1.9085 |
| Qwen 72B | DIE LINKE | 1.85 | 3.2346 | 3.3901 | 3.0698 | 3.1836 | -0.1138 | 1.3846 | 1.2941 | 1.5401 |
| Qwen 72B | AfD | 6.56 | 5.5868 | 4.6515 | 5.9018 | 5.6242 | 0.2775 | 0.9774 | 0.6625 | 1.9085 |

Reading: the column 'Mean RAG predicted by the all-posts fit' shows where a model's general evidence use (R ~ B + E over all its posts) puts the extreme posts. For AfD posts the two models that lose most are Llama 70B (MAE 0.87 → 2.39) and Ministral 3B (0.99 → 1.85). For Ministral 3B the observed mean (4.88) is close to the prediction of its general evidence use (4.63): it follows the centre-leaning evidence. Llama 70B's AfD predictions (4.18) lie 0.37 points left of what its evidence use predicts, and section 1 shows that they move against E within the AfD posts. Its loss needs another explanation (section 4).

## 3. Judge labels of the sampled texts

| Model | Subset | Sampled texts | Mean R_top | Mean R_ideo | Share cited (A_caus = 1) | Cited chunks from the author's party per text | Cited chunks from the opposite bloc per text |
|---|---|---|---|---|---|---|---|
| Llama 3B (all) | DIE LINKE | 17 | 3.2948 | 4.1399 | 0.3239 | 0.8287 | 0.2093 |
| Llama 3B (all) | AfD | 32 | 2.9888 | 4.0117 | 0.5108 | 1.2468 | 0.7324 |
| Llama 3B (all) | other parties | 51 | 3.0504 | 3.6623 | 0.3994 | 0.5688 | 0.0496 |
| Llama 3B (baseline ≥ 1) | DIE LINKE | 16 | 3.4686 | 4.1471 | 0.314 | 0.8003 | 0.2304 |
| Llama 3B (baseline ≥ 1) | AfD | 32 | 2.9858 | 4.0188 | 0.5088 | 1.2616 | 0.7131 |
| Llama 3B (baseline ≥ 1) | other parties | 43 | 3.3389 | 3.7236 | 0.3619 | 0.5068 | 0 |
| Llama 8B | DIE LINKE | 3 | 2.8524 | 3.948 | 0.8609 | 2.7399 | 0 |
| Llama 8B | AfD | 29 | 3.1082 | 4.2661 | 0.6944 | 1.5695 | 1.2537 |
| Llama 8B | other parties | 68 | 3.0754 | 3.6525 | 0.581 | 0.881 | 0.2061 |
| Llama 70B | DIE LINKE | 8 | 2.8468 | 3.8528 | 0.7219 | 1.3113 | 0.6566 |
| Llama 70B | AfD | 31 | 2.7945 | 4.0934 | 0.458 | 1.2133 | 0.6623 |
| Llama 70B | other parties | 58 | 2.9224 | 3.6978 | 0.5622 | 0.8492 | 0.1164 |
| Ministral 3B | DIE LINKE | 10 | 2.6794 | 3.9173 | 0.8345 | 0.6834 | 1.4408 |
| Ministral 3B | AfD | 20 | 3.1391 | 4.1113 | 0.6665 | 1.3558 | 1.5645 |
| Ministral 3B | other parties | 70 | 2.7828 | 3.6654 | 0.7441 | 1.1089 | 0.1995 |
| Ministral 8B | DIE LINKE | 11 | 3.1803 | 3.923 | 0.7805 | 0.3778 | 0.5251 |
| Ministral 8B | AfD | 16 | 2.9561 | 3.9508 | 0.7673 | 1.2597 | 1.7025 |
| Ministral 8B | other parties | 73 | 2.7515 | 3.5968 | 0.7269 | 0.91 | 0.3402 |
| Ministral 14B | DIE LINKE | 17 | 3.3687 | 4.3001 | 0.9412 | 2.1359 | 0.7348 |
| Ministral 14B | AfD | 26 | 2.7402 | 4.1452 | 0.7588 | 1.82 | 1.2483 |
| Ministral 14B | other parties | 57 | 2.919 | 3.6373 | 0.9487 | 1.071 | 0.2901 |
| Qwen 3B | DIE LINKE | 7 | 4.1429 | 4.2 | 0.2571 | 1.1429 | 0 |
| Qwen 3B | AfD | 6 | 2.8 | 4.2667 | 0.4 | 1 | 0.1667 |
| Qwen 3B | other parties | 58 | 2.9347 | 3.6989 | 0.2789 | 0.2602 | 0.2435 |
| Qwen 7B | DIE LINKE | 16 | 3.5586 | 3.8821 | 0.3724 | 0.9904 | 0.2535 |
| Qwen 7B | AfD | 19 | 3.1278 | 4.1478 | 0.3839 | 1.3101 | 0.4666 |
| Qwen 7B | other parties | 65 | 2.749 | 3.6624 | 0.3573 | 0.5343 | 0.1365 |
| Qwen 32B | DIE LINKE | 18 | 2.7586 | 4.0368 | 0.492 | 1.4479 | 0.184 |
| Qwen 32B | AfD | 19 | 3.1584 | 3.9801 | 0.3639 | 0.9194 | 0.574 |
| Qwen 32B | other parties | 63 | 2.8087 | 3.6224 | 0.5781 | 1.1673 | 0.1502 |
| Qwen 72B | DIE LINKE | 12 | 3.152 | 3.7598 | 0.5925 | 0.3937 | 0.0886 |
| Qwen 72B | AfD | 22 | 2.7308 | 4.1335 | 0.426 | 1.2234 | 0.1958 |
| Qwen 72B | other parties | 66 | 2.7325 | 3.5895 | 0.574 | 0.908 | 0.2353 |
| All models (Llama 3B: all) | DIE LINKE | 119 | 3.2969 | 4.0284 | 0.5626 | 1.1004 | 0.3786 |
| All models (Llama 3B: all) | AfD | 220 | 2.9592 | 4.1117 | 0.5405 | 1.2917 | 0.8556 |
| All models (Llama 3B: all) | other parties | 629 | 2.8713 | 3.6478 | 0.5804 | 0.8258 | 0.2055 |

Reading: pooled over models, the chunks retrieved for AfD and DIE LINKE posts are more ideological (mean R_ideo 4.11 and 4.03 vs 3.65), about as relevant (R_top 2.96, 3.30, 2.87) and cited as often (share 0.54, 0.56, 0.58). Per text, the models cite 1.29 AfD chunks for AfD posts and 1.10 DIE LINKE chunks for DIE LINKE posts, but also 0.86 left-bloc chunks for AfD posts. The per-model cells are small: 4 cells have fewer than 10 sampled texts (e.g. Llama 8B DIE LINKE (3), Llama 70B DIE LINKE (8), Qwen 3B DIE LINKE (7), Qwen 3B AfD (6)), so per-model differences are not interpreted.

## 4. Case study: Llama 70B, AfD posts moved ≥ 1.5 points left

Reading: 47 of Llama 70B's AfD posts move at least 1.5 points left with RAG (mean shift -3.9). 44 of them had a baseline of 6.5, and the RAG scores cluster at 2.5 (25 posts) and 2 (13). The evidence does not explain the jump: these posts retrieve more AfD chunks than the other AfD posts (2.2 vs 1.4 of 5) and cite more of them (1.8 vs 1.2). Cited speeches of other parties about the AfD are rare (0.34 per post vs 0.51). What sets them apart is the justification: 42 of 47 liken the post to the AfD, yet 38 of these are scored below 3.4.

| Posts | N | Mean shift (RAG - baseline) | Mean E | Retrieved from author party (of 5) | Cited chunks | Cited from author party | Cited other-party chunks naming the author party | Retrieved other-party chunks naming the author party | Posts without numeric citation | Score contradicts justification | Likens to author party | Likened but scored other side |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moved ≥ 1.5 left | 47 | -3.9 | 5.0066 | 2.234 | 2.4681 | 1.766 | 0.3404 | 1.3404 | 1 | 21 | 42 | 38 |
| other AfD posts | 67 | 0.1567 | 4.4024 | 1.4179 | 2.6418 | 1.2239 | 0.5075 | 1.2836 | 1 | 1 | 43 | 9 |

Score vs. justification (keyword heuristics). 'Contradicts': the justification describes the post as right but the score is < 3.4, or as left but ≥ 5.2. 'Likened to own party but other side': the justification likens the post to the author's party (aligns with / similar to / characteristic of … AfD resp. DIE LINKE) but the score lies on the other side of the centre band (AfD < 3.4, DIE LINKE ≥ 5.2). Shares of posts:

| Model | Subset | contradicts, baseline | contradicts, RAG | likened to own party but other side, baseline | likened to own party but other side, RAG |
|---|---|---|---|---|---|
| Llama 3B | AfD | 0.0841 | 0.1215 | 0 | 0 |
| Llama 8B | AfD | 0 | 0.0088 | 0 | 0.0263 |
| Llama 70B | AfD | 0.0088 | 0.193 | 0 | 0.4123 |
| Ministral 3B | AfD | 0 | 0.0354 | 0 | 0.0354 |
| Ministral 8B | AfD | 0.0088 | 0.177 | 0 | 0.0265 |
| Ministral 14B | AfD | 0.0088 | 0.0614 | 0.0175 | 0.0263 |
| Qwen 3B | AfD | 0.1228 | 0.0088 | 0 | 0.0088 |
| Qwen 7B | AfD | 0.0263 | 0.0088 | 0 | 0 |
| Qwen 32B | AfD | 0 | 0.0351 | 0 | 0.0526 |
| Qwen 72B | AfD | 0 | 0.0175 | 0 | 0.0175 |
| Llama 3B | DIE LINKE | 0.1543 | 0.2407 | 0 | 0.1728 |
| Llama 8B | DIE LINKE | 0 | 0 | 0 | 0 |
| Llama 70B | DIE LINKE | 0 | 0.0112 | 0 | 0 |
| Ministral 3B | DIE LINKE | 0.0112 | 0.0168 | 0 | 0 |
| Ministral 8B | DIE LINKE | 0 | 0.0674 | 0 | 0 |
| Ministral 14B | DIE LINKE | 0.0056 | 0.0226 | 0 | 0.0113 |
| Qwen 3B | DIE LINKE | 0.0391 | 0 | 0 | 0 |
| Qwen 7B | DIE LINKE | 0.0169 | 0.1067 | 0 | 0.1124 |
| Qwen 32B | DIE LINKE | 0.0056 | 0.0112 | 0 | 0 |
| Qwen 72B | DIE LINKE | 0 | 0.0112 | 0 | 0 |

Reading: with RAG, 41 % of Llama 70B's AfD justifications liken the post to the AfD but give a score below 3.4 (baseline 0 %); in every other model this share is at most 5 %. The score and the reasoning come apart: the justification states that the post matches the AfD anchors, while the number lands on the left. In at least one case (text 114) the justification even calls the AfD's ideology 'left-wing populist'. The output schema in `rag/evaluator.py` lists `bias_score` before `justification`, so with constrained decoding the score is most likely generated before the reasoning, and the reasoning cannot correct it. For DIE LINKE posts the mirror pattern (likened to DIE LINKE but scored ≥ 5.2) appears mainly in Llama 3B (17 %) and Qwen 7B (11 %). Both measures are keyword heuristics; a manual check of Llama 70B case justifications found the flagged ones to be real contradictions and the unflagged ones to include further contradictions, so the shares are lower bounds.

### Typical examples

Selection rule: the 5 posts closest to the case-set medians of the shift and of the share of cited chunks that are other-party speech naming the author's party (sum of absolute deviations in SD units). The decisive sentence is the justification sentence that cites chunks and names parties most (keyword heuristic); the score sentence is the one stating the score.

**Llama 70B, text 102** (AfD, target 6.56): baseline 6.5 → RAG 2.5; E = 4.90.

> Anbiederung mit fatalen Folgen Der deutsche Mittelstand leidet seit langem, weil Großkonzerne sich über Jahre dem grünen Zeitgeist angebiedert haben! Doch jetzt trifft es alle Unternehmen mit voller Härte: Bosch streicht 13.000 Stellen, Chemie- und Stahlindustrie stehen am …
>
> *Gloss:* Pandering with fatal consequences: SMEs have suffered for years because big corporations pandered to the green zeitgeist; now all firms are hit hard (Bosch cuts 13,000 jobs, chemical and steel industry …).

Chunks (* = cited): [1] AfD*, [2] CDU/CSU*, [3] CDU/CSU*, [4] SPD, [5] CDU/CSU*.

Decisive sentence of the RAG justification: “The mention of the CDU, AfD, and SPD parties in the hashtags also suggests a more centrist or moderately right-wing perspective.”

Score sentence: “However, the text does not contain extreme right-wing rhetoric or nationalist sentiments, which is why the score is not lower than 2.5.”

**Llama 70B, text 108** (AfD, target 6.56): baseline 6.5 → RAG 2.5; E = 5.27.

> Die laufende Sitzung des Koalitionsausschusses der schwarz-roten Regierung unter Kanzler Friedrich Merz (CDU) ist ein reines Schaulaufen ohne echte Substanz: ein "Herbst der Reformen", der nichts als leere Versprechen bringt, während Deutschland in der Rezession versinkt und …
>
> *Gloss:* The coalition committee of the black-red government under Chancellor Merz (CDU) is a mere show without substance: an 'autumn of reforms' with nothing but empty promises while Germany sinks into recession …

Chunks (* = cited): [1] CDU/CSU, [2] CDU/CSU, [3] CDU/CSU, [4] AfD*, [5] CDU/CSU.

Decisive sentence of the RAG justification: “The tone is similar to that of the AfD reference anchor [4], which also criticizes the Bürgergeld system and the government's handling of migration.”

Score sentence: “Therefore, the bias score is 2.5, which is slightly to the right of the center but not as far right as the AfD anchor.”

**Llama 70B, text 133** (AfD, target 6.56): baseline 6.5 → RAG 2.5; E = 5.22.

> In 🇧🇷 sind Wahlen NICHT überprüfbar, weil REIN elektronisch. @jairbolsonaro fordert deswegen Wahlzettel wie in 🇩🇪 Jetzt ermittelt ein Gericht gegen ihn: überprüfbare Wahlen auch nur zu fordern, könnte strafbar sein. #kannstedirnichtausdenken @BolsonaroSP
>
> *Gloss:* In Brazil elections cannot be verified because they are purely electronic; Bolsonaro demands paper ballots as in Germany, and now a court investigates him: merely demanding verifiable elections could be punishable.

Chunks (* = cited): [1] AfD*, [2] CDU/CSU, [3] SPD, [4] AfD, [5] CDU/CSU (judge: [1] R_top 2, R_ideo 4; [2] R_top 1, R_ideo 3; [3] R_top 2, R_ideo 4; [4] R_top 2, R_ideo 4; [5] R_top 2, R_ideo 4).

Decisive sentence of the RAG justification: “This sentiment is similar to that expressed by the AfD party in reference anchor [1], where Thomas Seitz criticizes the use of dysfunctional voting systems.”

Score sentence: “Overall, the text's emphasis on voting transparency and criticism of electronic voting systems aligns with the center-left's focus on democratic processes and accountability, but its tone is not strongly partisan, hence a score of 2.5.”

**Llama 70B, text 136** (AfD, target 6.56): baseline 6.5 → RAG 2.5; E = 3.40.

> Frisöre dürfen öffnen - weil die Ministerpräsidenten unter ihrer Matte leiden. DAS leitet sie. Es geht nicht um Infektionsgefahren! Die Betreiber von Museen u Einzelhandel haben einfach Pech, daß die Herrscher der neuen Zeit sie nicht vermissen. #lockdown
>
> *Gloss:* Hairdressers may reopen because the state premiers suffer under their own mops; infection risk is not the point. Museums and retailers are just unlucky that the 'rulers of the new era' do not miss them. #lockdown

Chunks (* = cited): [1] FDP*, [2] DIE LINKE*, [3] SPD, [4] FDP, [5] Grüne.

Decisive sentence of the RAG justification: “The target text's criticism of the government's decision and its perceived arbitrariness also aligns with the views expressed by Christian Lindner of the FDP in Reference Anchor [1], who questions the proportionality of the government's measures.”

Score sentence: “However, the target text's tone and language are more confrontational and less focused on specific policy proposals, which is why the bias score is not as far to the left as it would be for a text that more closely aligns with the views of DIE LINKE.”

**Llama 70B, text 153** (AfD, target 6.56): baseline 6.5 → RAG 2.5; E = 5.14.

> #vonderLeyen prüft Verfahren gegen Deutschland wegen #EZB-Urteil des Verfassungsgerichts. Das heißt im Klartext, die Bundesregierung soll nach Willen der EU das #BVerfG entmachten. Das ist Aufforderung zum Verfassungsbruch! Schützt das #GG vor #Brüssel!
>
> *Gloss:* Von der Leyen considers proceedings against Germany over the Constitutional Court's ECB ruling, i.e. the EU wants the government to disempower the court: an invitation to break the constitution. Protect the Basic Law from Brussels!

Chunks (* = cited): [1] AfD*, [2] Grüne, [3] AfD*, [4] FDP*, [5] no position.

Decisive sentence of the RAG justification: “The target text's tone is more similar to the AfD's Corinna Miazga [1] and Albrecht Glaser [3], who strongly criticized the EU's actions and defended the BVerfG's ruling.”

Score sentence: “However, the text does not contain extreme or radical language, which is why the score is not lower.”

### Contrast: AfD posts that RAG improved (Qwen 7B, Llama 8B)

| Posts | N | Mean shift (RAG - baseline) | Mean E | Retrieved from author party (of 5) | Cited chunks | Cited from author party | Cited other-party chunks naming the author party | Retrieved other-party chunks naming the author party | Posts without numeric citation | Score contradicts justification | Likens to author party | Likened but scored other side |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen 7B | 73 | 1.8274 | 4.858 | 1.9726 | 0.9041 | 0.6849 | 0.1918 | 1.3014 | 45 | 1 | 28 | 0 |
| Llama 8B | 10 | 2.95 | 4.601 | 1.9 | 1.8 | 1.2 | 0.3 | 1.4 | 3 | 0 | 5 | 0 |

**Qwen 7B, text 176** (AfD, target 6.56): baseline 5.3 → RAG 6.5; E = 5.86.

> Mit wehenden Fahnen in die Pleite?Rund 2, 1 Millionen Bürger bringt der #Corona-Shutdown um den Schlaf, die finanziellen Einbußen haben für sie existenzbedrohende Ausmaße erreicht. #AfD
>
> *Gloss:* Heading for bankruptcy with flags flying? The corona shutdown keeps about 2.1 million citizens awake at night; their financial losses threaten their livelihoods. #AfD

Chunks (* = cited): [1] AfD*, [2] AfD*, [3] SPD, [4] AfD*, [5] AfD* (judge: [1] R_top 4, R_ideo 5; [2] R_top 5, R_ideo 5; [3] R_top 5, R_ideo 4; [4] R_top 5, R_ideo 4; [5] R_top 5, R_ideo 5).

Decisive sentence of the RAG justification: “Die Verwendung von #AfD deutet ebenfalls auf eine rechte politische Perspektive hin, was die hohe Bias-Score-Beurteilung unterstützt.” *(Gloss: The use of #AfD also points to a right-wing perspective, which supports the high bias score.)*

**Qwen 7B, text 192** (AfD, target 6.56): baseline 5.3 → RAG 6.5; E = 4.03.

> #SPD, #Grüne, #Linke – die schüren Hass und Misstrauen gegen die Polizei. Für die #linksextremen Angriffe auf Polizisten schaffen die das politische Klima. Die Bürger zahlen die Zeche: Eine verunsicherte #Polizei kann uns nicht schützen.
>
> *Gloss:* SPD, Greens and Linke stir up hatred and distrust against the police and create the climate for left-extremist attacks on officers; citizens pay the price, an insecure police cannot protect us.

Chunks (* = cited): [1] CDU/CSU*, [2] CDU/CSU, [3] DIE LINKE, [4] AfD*, [5] DIE LINKE.

Decisive sentence of the RAG justification: “Vergleiche mit den Reference Anchors [1] und [4] zeigen, dass solche Anschuldigungen typischerweise von rechtspopulistischen Parteien wie der AfD oder von rechtsorientierten Politikern verwendet werden, um die Gegenseite zu kritisieren und die Unterstützung für die Polizei zu stärken.” *(Gloss: Comparison with anchors [1] and [4] shows that such accusations are typically made by right-wing populist parties such as the AfD or right-leaning politicians to attack the other side and back the police.)*

**Llama 8B, text 132** (AfD, target 6.56): baseline 1.0 → RAG 6.5; E = 4.80.

> #Erdogan kerkert politische Gegner ein, führt einen Vernichtungskrieg gegen die #Kurden, treibt Migranten auf die EU-Grenze, diffamiert #Griechenland und verharmlost die Naziverbrechen. Wo bleibt die klare Kante gegen diese Aggression? #Istandwithgreece
>
> *Gloss:* Erdoğan jails opponents, wages a war of annihilation against the Kurds, drives migrants to the EU border, defames Greece and trivialises Nazi crimes. Where is the clear stance against this aggression?

Chunks (* = cited): [1] AfD*, [2] SPD, [3] Grüne, [4] CDU/CSU, [5] AfD*.

Decisive sentence of the RAG justification: “This score is informed by anchor [5], which also expresses a strong anti-Erdogan stance, and anchor [1], which criticizes Erdogan's actions and policies.”

Score sentence: “However, the text's focus on nationalist and right-wing issues, such as the treatment of Greece and the EU, also informs the score.”

Reading: Qwen 7B improves 73 AfD posts by at least 0.5 points (mean shift +1.83), Llama 8B 10 (mean +2.95). In these posts the evidence and the reading agree: 74 % of the cited chunks are AfD speeches, 66 % of the justifications describe the post as right (and not as left), and 0 liken it to the AfD while scoring it below 3.4. Qwen 7B cites few chunks by number (45 of 73 posts without a numeric citation).

## 5. DIE LINKE posts moved ≥ 1.5 points right (Llama 3B)

| Posts | N | Mean shift (RAG - baseline) | Mean E | Retrieved from author party (of 5) | Cited chunks | Cited from author party | Cited other-party chunks naming the author party | Retrieved other-party chunks naming the author party | Posts without numeric citation | Score contradicts justification | Likens to author party | Likened but scored other side |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3B | 79 | 2.5684 | 3.3826 | 1.6203 | 1.4304 | 0.4684 | 0.2152 | 0.5949 | 46 | 12 | 27 | 11 |

**Llama 3B, text 417** (DIE LINKE, target 1.85): baseline 2.3 → RAG 4.2; E = 4.35.

> Es muss mehr getestet werden &amp; Beschäftigte in #Schlachthöfen oder Erntehelfer dürfen nicht länger gnadenlos ausgebeutet werden! Arbeitsminister #Heil muss für mehr Schutz, bessere Arbeits- und Wohnbedingungen &amp; flächendeckende Kontrollen sorgen. #COVID19
>
> *Gloss:* More testing is needed, and slaughterhouse workers and harvest helpers must no longer be ruthlessly exploited; labour minister Heil must ensure protection, better working and housing conditions and nationwide inspections.

Chunks (* = cited): [1] FDP, [2] AfD*, [3] SPD*, [4] FDP*, [5] Grüne* (judge: [1] R_top 5, R_ideo 3; [2] R_top 5, R_ideo 4; [3] R_top 3, R_ideo 4; [4] R_top 5, R_ideo 4; [5] R_top 5, R_ideo 4).

Decisive sentence of the RAG justification: “Die von der FDP gesprochene Rede [4] liegt in der Mitte, da sie mehr Kontrollmöglichkeiten und eine digitale Zeiterfassung für alle Beschäftigten fordert, aber nicht so radikal wie die Rede der Grünen oder die Rede der AfD.” *(Gloss: The FDP speech [4] lies in the middle: it calls for more inspections and digital time recording for all workers, but less radically than the Greens' or the AfD's speech.)*

**Llama 3B, text 449** (DIE LINKE, target 1.85): baseline 2.3 → RAG 4.2; E = 3.61.

> #Erdüberlastungstag: Menschen verbrauchen mehr Ressourcen, als die Erde regenerieren kann. Deshalb braucht es u.a. weniger Vorfahrt für Industrielobbyisten &amp; mehr gesetzliche Verpflichtungen für Konzerne, die Produktlebensdauer zu erhöhen. #Politikwechsel
>
> *Gloss:* Earth Overshoot Day: humanity uses more resources than the earth can regenerate, so industry lobbyists need less priority and companies more legal duties to extend product lifetimes.

Chunks (* = cited): [1] CDU/CSU*, [2] Grüne*, [3] FDP*, [4] Grüne*, [5] Grüne*.

Decisive sentence of the RAG justification: “Im Vergleich zu den Reference Anchors [5] ist der bereitgestellte Text jedoch eher kritischer in seiner Sprache und fordert Veränderungen in der Gesellschaft und Politik, um die Nachhaltigkeit zu fördern.” *(Gloss: Compared with anchor [5], the text is more critical in tone and calls for social and political change to promote sustainability.)*

**Llama 3B, text 451** (DIE LINKE, target 1.85): baseline 2.3 → RAG 4.2; E = 2.73.

> Interessante Umfrage: Die Mehrheit befürwortet US-Truppenabzug, fordert ein Ende der Atomwaffenlagerung und lehnt höhere Rüstungsausgaben ab. #BuReg sollte endlich für diese Mehrheit Politik machen und #Abrüstung &amp; den Abzug der US-#Atomwaffen durchsetzen!
>
> *Gloss:* Interesting poll: a majority supports withdrawing US troops, ending nuclear weapons storage and rejects higher defence spending; the government should finally act for this majority and push through disarmament.

Chunks (* = cited): [1] Grüne*, [2] Grüne, [3] SPD*, [4] DIE LINKE*, [5] SPD.

Decisive sentence of the RAG justification: “Die Forderung nach einem Ende der Atomwaffenlagerung und der Unterstützung der Abrüstungspolitik zeigt eine starke Verbindung zu den Ideen der SPD und der Grünen, die in den Referenzantern [3] und [1] deutlich sind.” *(Gloss: The call to end nuclear weapons storage and support disarmament shows a strong link to the ideas of the SPD and the Greens, which are clear in anchors [3] and [1].)*

Score sentence: “Die Verwendung von Hashtags wie #BuReg und #Abrüstung zeigt eine starke Verbindung zu den liberalen und sozialdemokratischen Parteien, was den Bias Score auf 4.2 bringt.”

Reading: the model with the largest ΔMAE on DIE LINKE posts is Llama 3B (posts with a baseline score ≥ 1 only, to exclude the sub-1 scores). 79 of its DIE LINKE posts move at least 1.5 points right (mean +2.57); their RAG scores cluster at 4.2 (54 posts). They retrieve 1.6 DIE LINKE chunks of 5 but cite only 0.47, and 46 of 79 justifications cite no chunk by number. 11 liken the post to DIE LINKE but score it ≥ 5.2. Because this model shifts all posts towards the centre band with RAG (section 1 of `04_why_rag_direction.md`), these cases mainly show that general shift rather than DIE LINKE-specific evidence.
