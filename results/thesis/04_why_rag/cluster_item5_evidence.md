# Item 5: evidence use R ~ B + E with account-level inference

OLS as before (pooled fits with model fixed effects). OLD = SEs clustered by text (one row per text and model, so per model this is the heteroskedasticity-robust SE). NEW = SEs clustered by author account (CR1, normal reference as before; the t(G−1) column uses the number of accounts minus 1 as df). Account bootstrap = pairs cluster bootstrap with the shared draws (weighted LS per draw). Permutation: E is permuted within each model by account blocks (see the script docstring), 1000 permutations, seed 42.

## Population fits (chunk_influence_all/evidence_use.csv)

| model | N | Accounts | E [old CI] | E [account-clustered CI] | E [account bootstrap CI] | Old perm p | Account-perm null 95 % | Account-perm p (one-sided) | Changed |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 7646 | 72 | +0.43 [+0.39, +0.48] | +0.43 [+0.38, +0.48] | +0.43 [+0.36, +0.47] | <0.001 | 0.056 | 0.001 | no |
| pooled + input party | 7646 | 72 | +0.32 [+0.27, +0.37] | +0.32 [+0.25, +0.39] | +0.32 [+0.25, +0.38] | <0.001 | 0.030 | 0.001 | no |
| llama_3b | 710 | 68 | +0.12 [+0.04, +0.20] | +0.12 [+0.03, +0.21] | +0.12 [+0.02, +0.22] | 0.002 | 0.108 | 0.031 | no |
| llama_70b | 775 | 72 | +0.06 [-0.06, +0.18] | +0.06 [-0.23, +0.35] | +0.06 [-0.27, +0.26] | 0.099 | 0.102 | 0.201 | no |
| llama_8b | 774 | 72 | +0.65 [+0.53, +0.76] | +0.65 [+0.46, +0.83] | +0.65 [+0.45, +0.82] | <0.001 | 0.218 | 0.001 | no |
| mistral_14b | 770 | 72 | +0.60 [+0.49, +0.72] | +0.60 [+0.43, +0.78] | +0.60 [+0.43, +0.78] | <0.001 | 0.128 | 0.001 | no |
| mistral_3b | 761 | 68 | +0.66 [+0.54, +0.78] | +0.66 [+0.49, +0.83] | +0.66 [+0.46, +0.78] | <0.001 | 0.184 | 0.001 | no |
| mistral_8b | 760 | 72 | +0.57 [+0.42, +0.72] | +0.57 [+0.40, +0.73] | +0.57 [+0.32, +0.70] | <0.001 | 0.146 | 0.001 | no |
| qwen_32b | 775 | 72 | +0.48 [+0.40, +0.56] | +0.48 [+0.38, +0.58] | +0.48 [+0.36, +0.55] | <0.001 | 0.124 | 0.001 | no |
| qwen_3b | 773 | 72 | +0.04 [+0.00, +0.08] | +0.04 [-0.02, +0.10] | +0.04 [-0.01, +0.11] | 0.011 | 0.031 | 0.023 | yes |
| qwen_72b | 775 | 72 | +0.43 [+0.36, +0.49] | +0.43 [+0.35, +0.50] | +0.43 [+0.35, +0.51] | <0.001 | 0.074 | 0.001 | no |
| qwen_7b | 773 | 72 | +0.31 [+0.18, +0.43] | +0.31 [+0.02, +0.59] | +0.31 [+0.06, +0.64] | <0.001 | 0.146 | 0.001 | no |

## By subset (extremes_evidence_use.csv)

| Model | Subset | N posts | Accounts | E [old CI] | E [account-clustered CI] | E [account bootstrap CI] | Old sig | New sig (CR1) | New sig (t G-1) | Boot sig | New sig (robust) | Changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | DIE LINKE | 162 | 6 | +0.02 [-0.13, +0.17] | +0.02 [-0.06, +0.10] | +0.02 [-0.16, +0.50] | no | no | no | no | no | no |
| Llama 3B (all) | AfD | 107 | 11 | +0.10 [-0.01, +0.20] | +0.10 [+0.04, +0.15] | +0.10 [+0.03, +0.33] | no | yes | yes | yes | yes | yes |
| Llama 3B (all) | other parties | 441 | 51 | +0.09 [-0.07, +0.25] | +0.09 [-0.05, +0.23] | +0.09 [-0.05, +0.28] | no | no | no | no | no | no |
| Llama 3B (baseline ≥ 1) | DIE LINKE | 153 | 6 | +0.02 [-0.14, +0.17] | +0.02 [-0.07, +0.11] | +0.02 [-0.18, +0.50] | no | no | no | no | no | no |
| Llama 3B (baseline ≥ 1) | AfD | 107 | 11 | +0.10 [-0.01, +0.20] | +0.10 [+0.04, +0.15] | +0.10 [+0.03, +0.33] | no | yes | yes | yes | yes | yes |
| Llama 3B (baseline ≥ 1) | other parties | 351 | 46 | +0.17 [+0.00, +0.34] | +0.17 [+0.05, +0.29] | +0.17 [+0.05, +0.36] | yes | yes | yes | yes | yes | no |
| Llama 8B | DIE LINKE | 178 | 7 | +0.56 [+0.35, +0.76] | +0.56 [+0.50, +0.61] | +0.56 [+0.43, +1.18] | yes | yes | yes | yes | yes | no |
| Llama 8B | AfD | 114 | 12 | +0.51 [+0.23, +0.79] | +0.51 [+0.32, +0.70] | +0.51 [+0.21, +0.99] | yes | yes | yes | yes | yes | no |
| Llama 8B | other parties | 482 | 53 | +0.38 [+0.20, +0.55] | +0.38 [+0.21, +0.54] | +0.38 [+0.26, +0.69] | yes | yes | yes | yes | yes | no |
| Llama 70B | DIE LINKE | 179 | 7 | +0.24 [+0.11, +0.36] | +0.24 [+0.21, +0.27] | +0.24 [-0.00, +0.25] | yes | yes | yes | no | no | yes |
| Llama 70B | AfD | 114 | 12 | -0.58 [-0.98, -0.18] | -0.58 [-1.18, +0.02] | -0.58 [-0.94, +0.28] | yes | no | no | no | no | yes |
| Llama 70B | other parties | 482 | 53 | +0.22 [+0.10, +0.33] | +0.22 [+0.11, +0.32] | +0.22 [+0.14, +0.39] | yes | yes | yes | yes | yes | no |
| Ministral 3B | DIE LINKE | 179 | 7 | +0.76 [+0.61, +0.92] | +0.76 [+0.70, +0.83] | +0.76 [+0.06, +0.88] | yes | yes | yes | yes | yes | no |
| Ministral 3B | AfD | 113 | 11 | +0.22 [-0.15, +0.59] | +0.22 [-0.07, +0.51] | +0.22 [+0.01, +0.80] | no | no | no | yes | no | no |
| Ministral 3B | other parties | 469 | 50 | +0.57 [+0.37, +0.77] | +0.57 [+0.45, +0.68] | +0.57 [+0.47, +0.77] | yes | yes | yes | yes | yes | no |
| Ministral 8B | DIE LINKE | 178 | 7 | +0.74 [+0.48, +1.00] | +0.74 [+0.59, +0.89] | +0.74 [-0.58, +0.83] | yes | yes | yes | no | no | yes |
| Ministral 8B | AfD | 113 | 12 | +0.25 [+0.03, +0.46] | +0.25 [+0.06, +0.43] | +0.25 [-0.12, +0.51] | yes | yes | yes | no | no | yes |
| Ministral 8B | other parties | 469 | 53 | +0.36 [+0.14, +0.58] | +0.36 [+0.09, +0.63] | +0.36 [-0.03, +0.54] | yes | yes | yes | no | no | yes |
| Ministral 14B | DIE LINKE | 177 | 7 | +0.45 [+0.30, +0.60] | +0.45 [+0.34, +0.56] | +0.45 [+0.13, +0.77] | yes | yes | yes | yes | yes | no |
| Ministral 14B | AfD | 114 | 12 | +0.68 [+0.43, +0.94] | +0.68 [+0.39, +0.97] | +0.68 [+0.04, +1.20] | yes | yes | yes | yes | yes | no |
| Ministral 14B | other parties | 479 | 53 | +0.46 [+0.32, +0.59] | +0.46 [+0.33, +0.58] | +0.46 [+0.28, +0.58] | yes | yes | yes | yes | yes | no |
| Qwen 3B | DIE LINKE | 179 | 7 | -0.00 [-0.07, +0.06] | -0.00 [-0.04, +0.03] | -0.00 [-0.03, +0.43] | no | no | no | no | no | no |
| Qwen 3B | AfD | 114 | 12 | +0.15 [+0.03, +0.26] | +0.15 [+0.04, +0.25] | +0.15 [-0.04, +0.22] | yes | yes | yes | no | no | yes |
| Qwen 3B | other parties | 480 | 53 | +0.01 [-0.03, +0.06] | +0.01 [-0.02, +0.05] | +0.01 [-0.03, +0.06] | no | no | no | no | no | no |
| Qwen 7B | DIE LINKE | 178 | 7 | +0.03 [-0.22, +0.28] | +0.03 [-0.17, +0.22] | +0.03 [-1.57, +1.19] | no | no | no | no | no | no |
| Qwen 7B | AfD | 114 | 12 | +0.56 [+0.32, +0.79] | +0.56 [+0.39, +0.72] | +0.56 [+0.33, +1.14] | yes | yes | yes | yes | yes | no |
| Qwen 7B | other parties | 481 | 53 | +0.08 [-0.09, +0.25] | +0.08 [-0.15, +0.32] | +0.08 [-0.09, +0.36] | no | no | no | no | no | no |
| Qwen 32B | DIE LINKE | 179 | 7 | +0.45 [+0.35, +0.55] | +0.45 [+0.37, +0.53] | +0.45 [+0.00, +2.86] | yes | yes | yes | no | no | yes |
| Qwen 32B | AfD | 114 | 12 | +0.31 [+0.07, +0.54] | +0.31 [+0.22, +0.39] | +0.31 [+0.21, +0.49] | yes | yes | yes | yes | yes | no |
| Qwen 32B | other parties | 482 | 53 | +0.32 [+0.23, +0.42] | +0.32 [+0.25, +0.40] | +0.32 [+0.25, +0.44] | yes | yes | yes | yes | yes | no |
| Qwen 72B | DIE LINKE | 179 | 7 | +0.49 [+0.37, +0.60] | +0.49 [+0.46, +0.51] | +0.49 [+0.46, +2.18] | yes | yes | yes | yes | yes | no |
| Qwen 72B | AfD | 114 | 12 | +0.35 [+0.17, +0.53] | +0.35 [+0.11, +0.60] | +0.35 [+0.10, +0.79] | yes | yes | yes | yes | yes | no |
| Qwen 72B | other parties | 482 | 53 | +0.32 [+0.24, +0.39] | +0.32 [+0.25, +0.39] | +0.32 [+0.27, +0.44] | yes | yes | yes | yes | yes | no |
| All models (model fixed effects) | DIE LINKE | 1768 | 7 | +0.42 [+0.35, +0.49] | +0.42 [+0.39, +0.44] | +0.42 [+0.27, +0.61] | yes | yes | yes | yes | yes | no |
| All models (model fixed effects) | AfD | 1131 | 12 | +0.25 [+0.12, +0.38] | +0.25 [+0.11, +0.40] | +0.25 [+0.15, +0.55] | yes | yes | yes | yes | yes | no |
| All models (model fixed effects) | other parties | 4747 | 53 | +0.31 [+0.24, +0.39] | +0.31 [+0.27, +0.36] | +0.31 [+0.27, +0.40] | yes | yes | yes | yes | yes | no |

DIE LINKE posts come from 7 accounts and AfD posts from 12, and one account (SWagenknecht, 121 of 179 posts) dominates DIE LINKE. With so few clusters the CR1 SE is biased downward: it comes out narrower than the old text-clustered SE in several DIE LINKE rows. The percentile cluster bootstrap is unstable as well, because a replicate can contain zero or seven copies of the dominant account. For the subsets and the contrasts, a result therefore counts as significant ('robust') only if the CR1 t(G−1) CI AND the account bootstrap CI both exclude 0. 'Changed' compares this robust rule with the old result.

## Interaction contrasts (extremes_evidence_use_contrasts.csv)

| Model | Contrast | Diff [old CI] | Old p (Holm, 2) | Diff [account-clustered CI] | New p (Holm, 2) | New p t (Holm, 2) | Diff [account bootstrap CI] | Boot p (Holm, 2) | New sig (robust) | Changed |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | E coefficient, DIE LINKE − other parties | -0.07 [-0.29, +0.15] | 1.000 | -0.07 [-0.23, +0.09] | 0.803 | 0.809 | -0.07 [-0.31, +0.40] | 0.952 | no | no |
| Llama 3B (all) | E coefficient, AfD − other parties | +0.01 [-0.18, +0.20] | 1.000 | +0.01 [-0.14, +0.16] | 0.938 | 0.938 | +0.01 [-0.19, +0.27] | 0.952 | no | no |
| Llama 3B (baseline ≥ 1) | E coefficient, DIE LINKE − other parties | -0.15 [-0.38, +0.08] | 0.394 | -0.15 [-0.30, -0.01] | 0.083 | 0.092 | -0.15 [-0.42, +0.31] | 0.363 | no | no |
| Llama 3B (baseline ≥ 1) | E coefficient, AfD − other parties | -0.08 [-0.28, +0.12] | 0.458 | -0.08 [-0.21, +0.06] | 0.261 | 0.266 | -0.08 [-0.28, +0.18] | 0.398 | no | no |
| Llama 8B | E coefficient, DIE LINKE − other parties | +0.18 [-0.09, +0.45] | 0.375 | +0.18 [+0.01, +0.35] | 0.073 | 0.080 | +0.18 [-0.20, +0.80] | 0.452 | no | no |
| Llama 8B | E coefficient, AfD − other parties | +0.14 [-0.19, +0.46] | 0.418 | +0.14 [-0.11, +0.38] | 0.276 | 0.280 | +0.14 [-0.31, +0.58] | 0.582 | no | no |
| Llama 70B | E coefficient, DIE LINKE − other parties | +0.02 [-0.15, +0.19] | 0.796 | +0.02 [-0.08, +0.13] | 0.676 | 0.677 | +0.02 [-0.28, +0.09] | 0.969 | no | no |
| Llama 70B | E coefficient, AfD − other parties | -0.79 [-1.20, -0.38] | 0.000 | -0.79 [-1.38, -0.21] | 0.015 | 0.019 | -0.79 [-1.19, +0.06] | 0.180 | no | yes |
| Ministral 3B | E coefficient, DIE LINKE − other parties | +0.20 [-0.05, +0.45] | 0.213 | +0.20 [+0.07, +0.33] | 0.006 | 0.008 | +0.20 [-0.51, +0.35] | 0.396 | no | no |
| Ministral 3B | E coefficient, AfD − other parties | -0.35 [-0.77, +0.07] | 0.213 | -0.35 [-0.65, -0.04] | 0.025 | 0.028 | -0.35 [-0.66, +0.21] | 0.396 | no | no |
| Ministral 8B | E coefficient, DIE LINKE − other parties | +0.38 [+0.04, +0.72] | 0.056 | +0.38 [+0.08, +0.68] | 0.028 | 0.033 | +0.38 [-0.98, +0.77] | 0.546 | no | no |
| Ministral 8B | E coefficient, AfD − other parties | -0.11 [-0.42, +0.19] | 0.470 | -0.11 [-0.43, +0.21] | 0.489 | 0.491 | -0.11 [-0.51, +0.37] | 0.660 | no | no |
| Ministral 14B | E coefficient, DIE LINKE − other parties | -0.01 [-0.21, +0.20] | 0.951 | -0.01 [-0.17, +0.16] | 0.939 | 0.940 | -0.01 [-0.38, +0.28] | 0.977 | no | no |
| Ministral 14B | E coefficient, AfD − other parties | +0.23 [-0.06, +0.51] | 0.255 | +0.23 [-0.08, +0.53] | 0.296 | 0.305 | +0.23 [-0.42, +0.77] | 0.472 | no | no |
| Qwen 3B | E coefficient, DIE LINKE − other parties | -0.02 [-0.10, +0.06] | 0.695 | -0.02 [-0.07, +0.03] | 0.526 | 0.528 | -0.02 [-0.08, +0.10] | 0.732 | no | no |
| Qwen 3B | E coefficient, AfD − other parties | +0.13 [+0.01, +0.26] | 0.065 | +0.13 [+0.03, +0.24] | 0.027 | 0.032 | +0.13 [-0.06, +0.23] | 0.356 | no | no |
| Qwen 7B | E coefficient, DIE LINKE − other parties | -0.06 [-0.36, +0.24] | 0.714 | -0.06 [-0.35, +0.24] | 0.712 | 0.713 | -0.06 [-1.56, +1.08] | 0.683 | no | no |
| Qwen 7B | E coefficient, AfD − other parties | +0.47 [+0.19, +0.76] | 0.002 | +0.47 [+0.19, +0.76] | 0.002 | 0.003 | +0.47 [+0.12, +1.02] | 0.024 | yes | no |
| Qwen 32B | E coefficient, DIE LINKE − other parties | +0.12 [-0.01, +0.26] | 0.155 | +0.12 [+0.01, +0.23] | 0.054 | 0.060 | +0.12 [-0.32, +2.51] | 0.330 | no | no |
| Qwen 32B | E coefficient, AfD − other parties | -0.02 [-0.27, +0.23] | 0.881 | -0.02 [-0.13, +0.09] | 0.740 | 0.741 | -0.02 [-0.18, +0.18] | 0.817 | no | no |
| Qwen 72B | E coefficient, DIE LINKE − other parties | +0.17 [+0.03, +0.31] | 0.032 | +0.17 [+0.10, +0.25] | 0.000 | 0.000 | +0.17 [+0.04, +1.83] | 0.046 | yes | no |
| Qwen 72B | E coefficient, AfD − other parties | +0.03 [-0.16, +0.23] | 0.725 | +0.03 [-0.21, +0.28] | 0.781 | 0.782 | +0.03 [-0.25, +0.47] | 0.786 | no | no |
| All models (model fixed effects) | E coefficient, DIE LINKE − other parties | +0.08 [-0.02, +0.18] | 0.252 | +0.08 [+0.02, +0.13] | 0.010 | 0.013 | +0.08 [-0.11, +0.17] | 0.388 | no | no |
| All models (model fixed effects) | E coefficient, AfD − other parties | -0.08 [-0.22, +0.07] | 0.303 | -0.08 [-0.22, +0.07] | 0.289 | 0.293 | -0.08 [-0.20, +0.20] | 0.479 | no | no |

## Cited/uncited and relevant/other (weighted quadrant sample, chunk_influence_all/evidence_by_label.csv)

| comparison | term | n_texts | Accounts | Estimate | Old p | New CI low (account-clustered) | New CI high (account-clustered) | New p (account-clustered, normal) | Boot CI low (account) | Boot CI high (account) | Boot p (account) | Changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cited vs. uncited chunks | E_cited | 636 | 61 | 0.348 | <0.001 | 0.296 | 0.401 | 0.000 | 0.286 | 0.405 | 0.001 | no |
| cited vs. uncited chunks | E_uncited | 636 | 61 | -0.049 | 0.345 | -0.126 | 0.028 | 0.212 | -0.126 | 0.045 | 0.281 | no |
| cited vs. uncited chunks | E_cited − E_uncited | 636 | 61 | 0.398 | <0.001 |  |  | 0.000 | 0.295 | 0.498 | 0.001 | no |
| relevant (R_top ≥ 4) vs. other chunks | E_relevant | 526 | 45 | 0.285 | <0.001 | 0.158 | 0.413 | 0.000 | 0.129 | 0.411 | 0.001 | no |
| relevant (R_top ≥ 4) vs. other chunks | E_irrelevant | 526 | 45 | 0.047 | 0.345 | -0.076 | 0.171 | 0.451 | -0.113 | 0.157 | 0.541 | no |
| relevant (R_top ≥ 4) vs. other chunks | E_relevant − E_irrelevant | 526 | 45 | 0.238 | 0.003 |  |  | 0.038 | 0.039 | 0.494 | 0.016 | no |

## Sensitivity: without Karl_Lauterbach (71 accounts)

Population fits: account-clustered CI. Subsets and contrasts: the robust rule.

Conclusions (account-clustered) that change without the largest account: 2

| Table | Row | Full | Without largest account |
|---|---|---|---|
| subset | Llama 3B (baseline ≥ 1) / other parties | +0.17 [+0.05, +0.29] robust=True | +0.19 [+0.00, +0.38] robust=False |
| contrast | Qwen 72B / E coefficient, DIE LINKE − other parties | +0.17, p_Holm CR1 0.000 / boot 0.046 | +0.13, p_Holm CR1 0.021 / boot 0.078 |