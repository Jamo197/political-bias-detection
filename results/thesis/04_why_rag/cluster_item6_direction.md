# Item 6: direction of the RAG shift with account-level inference

Paired e5/simple vs no-RAG posts per model, the same posts and point estimates as 04_why_rag_direction.md. NEW = paired account cluster bootstrap with the shared draws. Every statistic is recomputed on the resampled accounts (weights = account multiplicities), with a two-sided bootstrap p and Holm within model, as before. OLD = post bootstrap CI, with Wilcoxon or post bootstrap p.

## Overall statistics (difference RAG − baseline)

| Model | Statistic | Old [post CI] | Old p (Holm) | New [account CI] | p (Holm, within model) | Old CI direction | New CI direction | Changed (CI) |
|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | Mean absolute error | -0.08 [-0.20, +0.03] | 0.217 | -0.08 [-0.58, +0.35] | 0.686 | n.s. | n.s. | no |
| Llama 3B (all) | Mean signed error | +1.60 [+1.47, +1.73] | 0.000 | +1.60 [+1.24, +1.94] | 0.007 | up | up | no |
| Llama 3B (all) | SD of predictions | -0.97 [-1.05, -0.88] | 0.005 | -0.97 [-1.08, -0.78] | 0.007 | down | down | no |
| Llama 3B (all) | Calibration slope | -0.18 [-0.26, -0.10] | 0.005 | -0.18 [-0.42, +0.10] | 0.460 | down | n.s. | yes |
| Llama 3B (all) | Spread AfD - DIE LINKE | -0.84 [-1.20, -0.43] | 0.005 | -0.84 [-2.66, -0.13] | 0.072 | down | down | no |
| Llama 3B (all) | Share in centre band [3.4, 4.6) | +0.49 [+0.45, +0.53] | 0.000 | +0.49 [+0.35, +0.59] | 0.007 | up | up | no |
| Llama 3B (baseline ≥ 1) | Mean absolute error | +0.23 [+0.11, +0.34] | 0.000 | +0.23 [-0.23, +0.59] | 0.291 | up | n.s. | yes |
| Llama 3B (baseline ≥ 1) | Mean signed error | +1.23 [+1.10, +1.35] | 0.000 | +1.23 [+0.96, +1.52] | 0.007 | up | up | no |
| Llama 3B (baseline ≥ 1) | SD of predictions | -0.72 [-0.78, -0.64] | 0.004 | -0.72 [-0.83, -0.53] | 0.007 | down | down | no |
| Llama 3B (baseline ≥ 1) | Calibration slope | -0.16 [-0.24, -0.09] | 0.004 | -0.16 [-0.35, +0.01] | 0.177 | down | n.s. | yes |
| Llama 3B (baseline ≥ 1) | Spread AfD - DIE LINKE | -0.67 [-1.03, -0.28] | 0.004 | -0.67 [-2.66, +0.03] | 0.177 | down | n.s. | yes |
| Llama 3B (baseline ≥ 1) | Share in centre band [3.4, 4.6) | +0.45 [+0.41, +0.49] | 0.000 | +0.45 [+0.31, +0.56] | 0.007 | up | up | no |
| Llama 8B | Mean absolute error | -0.51 [-0.60, -0.42] | 0.000 | -0.51 [-0.66, -0.30] | 0.007 | down | down | no |
| Llama 8B | Mean signed error | -0.66 [-0.78, -0.55] | 0.000 | -0.66 [-0.89, -0.43] | 0.007 | down | down | no |
| Llama 8B | SD of predictions | -0.44 [-0.49, -0.38] | 0.004 | -0.44 [-0.63, -0.20] | 0.008 | down | down | no |
| Llama 8B | Calibration slope | -0.01 [-0.09, +0.06] | 0.820 | -0.01 [-0.25, +0.12] | 0.815 | n.s. | n.s. | no |
| Llama 8B | Spread AfD - DIE LINKE | +0.15 [-0.22, +0.52] | 0.820 | +0.15 [-0.56, +0.50] | 0.748 | n.s. | n.s. | no |
| Llama 8B | Share in centre band [3.4, 4.6) | +0.22 [+0.19, +0.25] | 0.000 | +0.22 [+0.13, +0.31] | 0.007 | up | up | no |
| Llama 70B | Mean absolute error | +0.28 [+0.20, +0.36] | 0.000 | +0.28 [+0.08, +0.61] | 0.008 | up | up | no |
| Llama 70B | Mean signed error | -0.10 [-0.19, -0.00] | 0.826 | -0.10 [-0.48, +0.13] | 0.586 | down | n.s. | yes |
| Llama 70B | SD of predictions | -0.16 [-0.21, -0.10] | 0.005 | -0.16 [-0.26, -0.01] | 0.082 | down | down | no |
| Llama 70B | Calibration slope | -0.32 [-0.39, -0.24] | 0.005 | -0.32 [-0.46, -0.11] | 0.009 | down | down | no |
| Llama 70B | Spread AfD - DIE LINKE | -1.59 [-2.02, -1.19] | 0.005 | -1.59 [-2.09, -1.22] | 0.007 | down | down | no |
| Llama 70B | Share in centre band [3.4, 4.6) | +0.09 [+0.06, +0.12] | 0.000 | +0.09 [+0.07, +0.14] | 0.007 | up | up | no |
| Ministral 3B | Mean absolute error | -0.06 [-0.16, +0.03] | 0.435 | -0.06 [-0.32, +0.25] | 1.000 | n.s. | n.s. | no |
| Ministral 3B | Mean signed error | -0.51 [-0.63, -0.39] | 0.000 | -0.51 [-0.80, -0.26] | 0.007 | down | down | no |
| Ministral 3B | SD of predictions | -0.20 [-0.28, -0.12] | 0.005 | -0.20 [-0.30, -0.13] | 0.007 | down | down | no |
| Ministral 3B | Calibration slope | -0.09 [-0.17, -0.01] | 0.140 | -0.09 [-0.25, +0.08] | 0.792 | down | n.s. | yes |
| Ministral 3B | Spread AfD - DIE LINKE | -0.30 [-0.71, +0.13] | 0.435 | -0.30 [-0.88, +0.10] | 0.488 | n.s. | n.s. | no |
| Ministral 3B | Share in centre band [3.4, 4.6) | +0.08 [+0.04, +0.12] | 0.001 | +0.08 [+0.04, +0.15] | 0.007 | up | up | no |
| Ministral 8B | Mean absolute error | +0.07 [-0.01, +0.15] | 0.143 | +0.07 [-0.07, +0.18] | 0.766 | n.s. | n.s. | no |
| Ministral 8B | Mean signed error | -0.04 [-0.14, +0.06] | 0.497 | -0.04 [-0.20, +0.12] | 0.766 | n.s. | n.s. | no |
| Ministral 8B | SD of predictions | +0.31 [+0.25, +0.37] | 0.006 | +0.31 [+0.15, +0.39] | 0.007 | up | up | no |
| Ministral 8B | Calibration slope | +0.10 [+0.04, +0.17] | 0.008 | +0.10 [-0.03, +0.16] | 0.488 | up | n.s. | yes |
| Ministral 8B | Spread AfD - DIE LINKE | +0.64 [+0.32, +0.97] | 0.006 | +0.64 [-0.44, +0.82] | 0.488 | up | n.s. | yes |
| Ministral 8B | Share in centre band [3.4, 4.6) | -0.13 [-0.17, -0.10] | 0.000 | -0.13 [-0.21, -0.05] | 0.007 | down | down | no |
| Ministral 14B | Mean absolute error | +0.05 [-0.01, +0.11] | 0.018 | +0.05 [-0.10, +0.17] | 0.934 | n.s. | n.s. | no |
| Ministral 14B | Mean signed error | +0.06 [-0.01, +0.13] | 0.685 | +0.06 [-0.08, +0.21] | 0.934 | n.s. | n.s. | no |
| Ministral 14B | SD of predictions | +0.24 [+0.19, +0.30] | 0.007 | +0.24 [+0.15, +0.29] | 0.007 | up | up | no |
| Ministral 14B | Calibration slope | +0.11 [+0.05, +0.18] | 0.007 | +0.11 [+0.02, +0.16] | 0.090 | up | up | no |
| Ministral 14B | Spread AfD - DIE LINKE | +0.62 [+0.29, +0.96] | 0.007 | +0.62 [+0.07, +0.86] | 0.120 | up | up | no |
| Ministral 14B | Share in centre band [3.4, 4.6) | +0.04 [+0.01, +0.07] | 0.019 | +0.04 [+0.01, +0.06] | 0.090 | up | up | no |
| Qwen 3B | Mean absolute error | +0.07 [+0.02, +0.12] | 0.040 | +0.07 [-0.06, +0.16] | 0.879 | up | n.s. | yes |
| Qwen 3B | Mean signed error | +0.17 [+0.12, +0.22] | 0.000 | +0.17 [+0.08, +0.23] | 0.010 | up | up | no |
| Qwen 3B | SD of predictions | -0.21 [-0.26, -0.16] | 0.005 | -0.21 [-0.27, -0.16] | 0.007 | down | down | no |
| Qwen 3B | Calibration slope | +0.00 [-0.04, +0.04] | 1.000 | +0.00 [-0.08, +0.04] | 1.000 | n.s. | n.s. | no |
| Qwen 3B | Spread AfD - DIE LINKE | +0.05 [-0.14, +0.25] | 1.000 | +0.05 [-0.53, +0.22] | 1.000 | n.s. | n.s. | no |
| Qwen 3B | Share in centre band [3.4, 4.6) | +0.09 [+0.06, +0.13] | 0.000 | +0.09 [+0.05, +0.15] | 0.007 | up | up | no |
| Qwen 7B | Mean absolute error | +0.07 [-0.01, +0.15] | 0.080 | +0.07 [-0.26, +0.26] | 0.820 | n.s. | n.s. | no |
| Qwen 7B | Mean signed error | +0.34 [+0.26, +0.43] | 0.000 | +0.34 [+0.18, +0.52] | 0.007 | up | up | no |
| Qwen 7B | SD of predictions | +0.46 [+0.40, +0.52] | 0.005 | +0.46 [+0.37, +0.56] | 0.007 | up | up | no |
| Qwen 7B | Calibration slope | +0.15 [+0.09, +0.21] | 0.005 | +0.15 [+0.01, +0.27] | 0.148 | up | up | no |
| Qwen 7B | Spread AfD - DIE LINKE | +0.75 [+0.41, +1.10] | 0.005 | +0.75 [+0.10, +2.50] | 0.148 | up | up | no |
| Qwen 7B | Share in centre band [3.4, 4.6) | -0.17 [-0.21, -0.14] | 0.000 | -0.17 [-0.24, -0.12] | 0.007 | down | down | no |
| Qwen 32B | Mean absolute error | -0.15 [-0.21, -0.09] | 0.000 | -0.15 [-0.25, -0.01] | 0.032 | down | down | no |
| Qwen 32B | Mean signed error | -0.27 [-0.33, -0.20] | 0.000 | -0.27 [-0.36, -0.09] | 0.015 | down | down | no |
| Qwen 32B | SD of predictions | +0.20 [+0.15, +0.26] | 0.004 | +0.20 [+0.09, +0.29] | 0.007 | up | up | no |
| Qwen 32B | Calibration slope | +0.12 [+0.07, +0.18] | 0.004 | +0.12 [+0.06, +0.17] | 0.007 | up | up | no |
| Qwen 32B | Spread AfD - DIE LINKE | +0.61 [+0.33, +0.91] | 0.004 | +0.61 [+0.31, +1.18] | 0.015 | up | up | no |
| Qwen 32B | Share in centre band [3.4, 4.6) | -0.13 [-0.17, -0.10] | 0.000 | -0.13 [-0.19, -0.07] | 0.007 | down | down | no |
| Qwen 72B | Mean absolute error | -0.03 [-0.08, +0.01] | 0.192 | -0.03 [-0.15, +0.07] | 1.000 | n.s. | n.s. | no |
| Qwen 72B | Mean signed error | -0.00 [-0.06, +0.05] | 0.902 | -0.00 [-0.12, +0.09] | 1.000 | n.s. | n.s. | no |
| Qwen 72B | SD of predictions | +0.21 [+0.17, +0.25] | 0.006 | +0.21 [+0.15, +0.27] | 0.007 | up | up | no |
| Qwen 72B | Calibration slope | +0.09 [+0.05, +0.12] | 0.006 | +0.09 [+0.03, +0.13] | 0.035 | up | up | no |
| Qwen 72B | Spread AfD - DIE LINKE | +0.48 [+0.29, +0.66] | 0.006 | +0.48 [+0.16, +0.94] | 0.039 | up | up | no |
| Qwen 72B | Share in centre band [3.4, 4.6) | -0.09 [-0.12, -0.06] | 0.000 | -0.09 [-0.11, -0.05] | 0.018 | down | down | no |

## Share of posts moving ≥ 0.5 (all posts)

| Model | N | Moved | Right − left | Bootstrap p (right - left) |
|---|---|---|---|---|
| Llama 3B (all) | 710 | +0.75 [+0.65, +0.83] | +0.56 [+0.42, +0.69] | 0.001 |
| Llama 3B (baseline ≥ 1) | 611 | +0.71 [+0.61, +0.79] | +0.49 [+0.36, +0.63] | 0.001 |
| Llama 8B | 774 | +0.55 [+0.44, +0.64] | -0.25 [-0.37, -0.12] | 0.001 |
| Llama 70B | 775 | +0.41 [+0.30, +0.49] | +0.08 [-0.09, +0.21] | 0.378 |
| Ministral 3B | 761 | +0.65 [+0.58, +0.70] | -0.21 [-0.32, -0.11] | 0.001 |
| Ministral 8B | 760 | +0.50 [+0.44, +0.55] | -0.05 [-0.12, +0.02] | 0.142 |
| Ministral 14B | 770 | +0.44 [+0.40, +0.48] | +0.02 [-0.05, +0.10] | 0.588 |
| Qwen 3B | 773 | +0.44 [+0.40, +0.50] | +0.19 [+0.09, +0.25] | 0.001 |
| Qwen 7B | 773 | +0.54 [+0.47, +0.64] | +0.18 [+0.09, +0.29] | 0.002 |
| Qwen 32B | 775 | +0.57 [+0.52, +0.60] | -0.27 [-0.35, -0.11] | 0.001 |
| Qwen 72B | 775 | +0.52 [+0.48, +0.55] | +0.02 [-0.11, +0.13] | 0.788 |

DIE LINKE and AfD movement shares are in `direction_movement_cluster.csv`.

## Per-party changes, DIE LINKE and AfD

| Model | Author party | N | Accounts | dMAE old | Old dMAE p (Holm, across parties) | dMAE new | dMAE p (Holm, across parties) | dSE old | dSE new | dSE CR1-t | Changed dMAE (CI) | Changed d signed error (CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | DIE LINKE | 162 | 6 | +1.07 [+0.83, +1.30] | 0.000 | +1.07 [+0.97, +3.17] | 0.006 | +1.36 [+1.09, +1.62] | +1.36 [+1.16, +3.17] | +1.36 [+0.91, +1.80] | no | no |
| Llama 3B (all) | AfD | 107 | 11 | -0.56 [-0.81, -0.33] | 0.000 | -0.56 [-1.16, -0.30] | 0.042 | +0.52 [+0.28, +0.78] | +0.52 [+0.24, +1.16] | +0.52 [+0.06, +0.98] | no | no |
| Llama 3B (baseline ≥ 1) | DIE LINKE | 153 | 6 | +1.10 [+0.85, +1.35] | 0.000 | +1.10 [+0.99, +3.17] | 0.006 | +1.19 [+0.93, +1.45] | +1.19 [+1.02, +3.17] | +1.19 [+0.81, +1.57] | no | no |
| Llama 3B (baseline ≥ 1) | AfD | 107 | 11 | -0.56 [-0.81, -0.33] | 0.000 | -0.56 [-1.16, -0.30] | 0.084 | +0.52 [+0.28, +0.78] | +0.52 [+0.24, +1.16] | +0.52 [+0.06, +0.98] | no | no |
| Llama 8B | DIE LINKE | 178 | 7 | -0.43 [-0.66, -0.20] | 0.000 | -0.43 [-0.53, +0.32] | 0.294 | -0.35 [-0.58, -0.11] | -0.35 [-0.71, +0.22] | -0.35 [-0.39, -0.30] | yes | yes |
| Llama 8B | AfD | 114 | 12 | +0.09 [-0.20, +0.38] | 0.512 | +0.09 [-0.08, +0.61] | 0.554 | -0.20 [-0.49, +0.10] | -0.20 [-0.64, +0.03] | -0.20 [-0.40, +0.00] | no | no |
| Llama 70B | DIE LINKE | 179 | 7 | +0.09 [-0.01, +0.19] | 0.060 | +0.09 [+0.00, +0.26] | 0.358 | +0.08 [-0.02, +0.18] | +0.08 [+0.00, +0.23] | +0.08 [-0.10, +0.25] | no | no |
| Llama 70B | AfD | 114 | 12 | +1.52 [+1.15, +1.92] | 0.000 | +1.52 [+1.14, +1.98] | 0.006 | -1.52 [-1.92, -1.14] | -1.52 [-1.98, -1.14] | -1.52 [-1.76, -1.28] | no | no |
| Ministral 3B | DIE LINKE | 179 | 7 | -0.61 [-0.80, -0.41] | 0.000 | -0.61 [-0.71, -0.17] | 0.165 | -0.77 [-1.00, -0.55] | -0.77 [-0.90, -0.24] | -0.77 [-1.08, -0.46] | no | no |
| Ministral 3B | AfD | 113 | 11 | +0.86 [+0.54, +1.17] | 0.000 | +0.86 [+0.50, +1.14] | 0.006 | -1.07 [-1.42, -0.70] | -1.07 [-1.34, -0.65] | -1.07 [-1.29, -0.84] | no | no |
| Ministral 8B | DIE LINKE | 178 | 7 | +0.06 [-0.09, +0.22] | 0.458 | +0.06 [-0.02, +0.82] | 1.000 | -0.22 [-0.41, -0.03] | -0.22 [-0.33, +0.73] | -0.22 [-0.46, +0.02] | no | yes |
| Ministral 8B | AfD | 113 | 12 | -0.14 [-0.36, +0.07] | 0.654 | -0.14 [-0.24, +0.22] | 1.000 | +0.42 [+0.17, +0.67] | +0.42 [-0.04, +0.54] | +0.42 [+0.19, +0.65] | no | yes |
| Ministral 14B | DIE LINKE | 177 | 7 | -0.00 [-0.10, +0.10] | 0.655 | -0.00 [-0.09, +0.60] | 0.957 | -0.15 [-0.26, -0.05] | -0.15 [-0.43, +0.16] | -0.15 [-0.22, -0.09] | no | yes |
| Ministral 14B | AfD | 114 | 12 | -0.35 [-0.64, -0.09] | 0.068 | -0.35 [-0.48, +0.06] | 0.798 | +0.47 [+0.18, +0.79] | +0.47 [-0.02, +0.62] | +0.47 [+0.15, +0.78] | yes | yes |
| Qwen 3B | DIE LINKE | 179 | 7 | +0.11 [+0.03, +0.19] | 0.160 | +0.11 [+0.00, +0.50] | 0.370 | +0.11 [+0.03, +0.19] | +0.11 [+0.00, +0.50] | +0.11 [+0.07, +0.15] | yes | yes |
| Qwen 3B | AfD | 114 | 12 | -0.17 [-0.33, +0.00] | 0.255 | -0.17 [-0.30, +0.31] | 1.000 | +0.17 [-0.00, +0.33] | +0.17 [-0.31, +0.30] | +0.17 [-0.08, +0.41] | no | no |
| Qwen 7B | DIE LINKE | 178 | 7 | +0.37 [+0.19, +0.55] | 0.001 | +0.37 [-0.78, +0.47] | 0.680 | +0.29 [+0.08, +0.49] | +0.29 [-1.41, +0.40] | +0.29 [+0.11, +0.47] | yes | yes |
| Qwen 7B | AfD | 114 | 12 | -0.95 [-1.21, -0.68] | 0.000 | -0.95 [-1.12, -0.50] | 0.045 | +1.03 [+0.76, +1.29] | +1.03 [+0.50, +1.22] | +1.03 [+0.75, +1.32] | no | no |
| Qwen 32B | DIE LINKE | 179 | 7 | -0.41 [-0.50, -0.31] | 0.000 | -0.41 [-0.53, -0.20] | 0.072 | -0.54 [-0.65, -0.43] | -0.54 [-1.00, -0.50] | -0.54 [-0.61, -0.47] | no | no |
| Qwen 32B | AfD | 114 | 12 | -0.07 [-0.32, +0.19] | 1.000 | -0.07 [-0.31, +0.22] | 1.000 | +0.07 [-0.20, +0.32] | +0.07 [-0.22, +0.31] | +0.07 [-0.10, +0.23] | no | no |
| Qwen 72B | DIE LINKE | 179 | 7 | -0.09 [-0.18, +0.01] | 0.345 | -0.09 [-0.32, +0.07] | 1.000 | -0.16 [-0.27, -0.06] | -0.16 [-0.62, -0.03] | -0.16 [-0.29, -0.04] | no | no |
| Qwen 72B | AfD | 114 | 12 | -0.31 [-0.47, -0.15] | 0.000 | -0.31 [-0.54, -0.05] | 0.210 | +0.31 [+0.15, +0.47] | +0.31 [+0.05, +0.54] | +0.31 [+0.16, +0.47] | no | no |

'new' = account percentile bootstrap. 'CR1-t' = mean with a cluster-robust SE and a t(G−1) interval. DIE LINKE rests on 7 accounts, and SWagenknecht holds 121 of its 179 posts. In a bootstrap replicate that account appears 0 to 4 times, so the percentile bounds are erratic (for example, an upper bound of +3.17 for Llama 3B). With an effective sample of 1-2 accounts, neither interval is reliable for DIE LINKE. SPD (5 accounts, 193 of 217 posts from Karl_Lauterbach) and FDP (4 accounts) are worse still.

Parties with ≥ 10 accounts (Die Grünen, CDU/CSU, AfD): 4 of 33 dMAE CI directions change, 3 signed-error CI directions change and 6 dMAE Holm decisions change. Parties with fewer accounts (DIE LINKE, SPD, FDP): 14 of 33 dMAE CI directions, 11 signed-error CI directions and 16 Holm decisions change (all rows: `direction_by_party_cluster.csv`).

## Compress / expand / shift classification

The rule is unchanged (`why_rag_analysis.COMPRESSION_RULE`). It is applied once to the account bootstrap CIs and once to the CR1-t CIs of the DIE LINKE and AfD signed-error changes.

| Model | Old Classification | DIE LINKE d [new CI] | AfD d [new CI] | Classification | DIE LINKE d [CR1-t] | AfD d [CR1-t] | Classification (CR1-t) | Class changed | Class changed (CR1-t) |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3B (all) | shifts right | +1.36 [+1.16, +3.17] | +0.52 [+0.24, +1.16] | shifts right | +1.36 [+0.91, +1.80] | +0.52 [+0.06, +0.98] | shifts right | no | no |
| Llama 3B (baseline ≥ 1) | shifts right | +1.19 [+1.02, +3.17] | +0.52 [+0.24, +1.16] | shifts right | +1.19 [+0.81, +1.57] | +0.52 [+0.06, +0.98] | shifts right | no | no |
| Llama 8B | shifts left | -0.35 [-0.71, +0.22] | -0.20 [-0.64, +0.03] | mixed | -0.35 [-0.39, -0.30] | -0.20 [-0.40, +0.00] | shifts left | yes | no |
| Llama 70B | mixed | +0.08 [+0.00, +0.23] | -1.52 [-1.98, -1.14] | mixed | +0.08 [-0.10, +0.25] | -1.52 [-1.76, -1.28] | mixed | no | no |
| Ministral 3B | shifts left | -0.77 [-0.90, -0.24] | -1.07 [-1.34, -0.65] | shifts left | -0.77 [-1.08, -0.46] | -1.07 [-1.29, -0.84] | shifts left | no | no |
| Ministral 8B | expands | -0.22 [-0.33, +0.73] | +0.42 [-0.04, +0.54] | mixed | -0.22 [-0.46, +0.02] | +0.42 [+0.19, +0.65] | mixed | yes | yes |
| Ministral 14B | expands | -0.15 [-0.43, +0.16] | +0.47 [-0.02, +0.62] | mixed | -0.15 [-0.22, -0.09] | +0.47 [+0.15, +0.78] | expands | yes | no |
| Qwen 3B | shifts right | +0.11 [+0.00, +0.50] | +0.17 [-0.31, +0.30] | mixed | +0.11 [+0.07, +0.15] | +0.17 [-0.08, +0.41] | shifts right | yes | no |
| Qwen 7B | shifts right | +0.29 [-1.41, +0.40] | +1.03 [+0.50, +1.22] | shifts right | +0.29 [+0.11, +0.47] | +1.03 [+0.75, +1.32] | shifts right | no | no |
| Qwen 32B | mixed | -0.54 [-1.00, -0.50] | +0.07 [-0.22, +0.31] | mixed | -0.54 [-0.61, -0.47] | +0.07 [-0.10, +0.23] | mixed | no | no |
| Qwen 72B | expands | -0.16 [-0.62, -0.03] | +0.31 [+0.05, +0.54] | expands | -0.16 [-0.29, -0.04] | +0.31 [+0.16, +0.47] | expands | no | no |

Class counts (Llama 3B counted once, all posts, as in the old report). OLD: shifts right 3, expands 3, shifts left 2, mixed 2. NEW (bootstrap): mixed 6, shifts right 2, shifts left 1, expands 1. NEW (CR1-t): shifts right 3, mixed 3, shifts left 2, expands 2.

## Sensitivity: without Karl_Lauterbach (71 accounts)

Overall statistics whose account-CI direction changes: 2

| Model | Statistic | Full | Sens |
|---|---|---|---|
| Qwen 32B | Mean absolute error | -0.15 [-0.25, -0.01] | -0.14 [-0.25, +0.02] |
| Qwen 72B | Mean absolute error | -0.03 [-0.15, +0.07] | -0.10 [-0.18, -0.01] |

Classes (bootstrap or CR1-t) that change without the largest account: 0


Per-party dMAE whose account-CI direction changes: 4

| Model | Author party | N_s | Full | Sens |
|---|---|---|---|---|
| Llama 3B (all) | SPD | 24 | -0.30 [-1.18, -0.25] | -0.65 [-1.96, +0.34] |
| Ministral 3B | Die Grünen | 149 | -0.18 [-0.50, +0.01] | -0.18 [-0.50, -0.01] |
| Qwen 3B | SPD | 24 | +0.24 [+0.21, +0.65] | +0.42 [+0.00, +1.50] |
| Qwen 32B | SPD | 24 | -0.19 [-0.29, -0.05] | -0.19 [-0.34, +0.00] |
