# Factor analysis — label_ideology

Linear mixed models, random intercept per text (`text_index`), fitted by ML.
Outcomes: `abs_error` (accuracy) and `signed_error` (pred − truth; positive = leans right).
Reference levels: family = Llama, party = SPD (where present). `size_c` = centred log2(parameters); `time_c` = centred release date in years.

> **Caveat — family and time are nearly collinear.** With only a handful of models, 'later knowledge cutoff / release' is almost identical to 'Mistral family' (see table below). M1 vs M2 therefore cannot separate a *time* effect from a *family* effect; treat the comparison as descriptive, not causal.

## Models

| model | family | params_b | release_date | cutoff_date | time_cov |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | Llama | 8 | 2024-07 | 2023-12 | 2024.5 |
| Llama-3.2-3B-Instruct | Llama | 3 | 2024-09 | 2023-12 | 2024.67 |
| Meta-Llama-3.1-70B-Instruct-FP8 | Llama | 70 | 2024-07 | 2023-12 | 2024.5 |
| Ministral-3-14B-Instruct-2512 | Mistral | 14 | 2025-12 |  | 2025.92 |
| Ministral-3-3B-Instruct-2512 | Mistral | 3 | 2025-12 |  | 2025.92 |
| Ministral-3-8B-Instruct-2512 | Mistral | 8 | 2025-12 |  | 2025.92 |
| Qwen2.5-32B-Instruct | Qwen | 32 | 2024-09 |  | 2024.67 |
| Qwen2.5-3B-Instruct | Qwen | 3 | 2024-09 |  | 2024.67 |
| Qwen2.5-72B-Instruct-FP8-dynamic | Qwen | 72 | 2024-09 |  | 2024.67 |
| Qwen2.5-7B-Instruct | Qwen | 7 | 2024-09 |  | 2024.67 |

## Model comparison

| Outcome | Spec | N | N_texts | ICC | AIC | BIC | Converged |
|---|---|---|---|---|---|---|---|
| abs_error | M1_family_size | 121351 | 775 | 0.1968 | 320476.22 | 320621.82 | True |
| abs_error | M2_time_size | 121351 | 775 | 0.1965 | 320675.97 | 320802.15 | True |
| abs_error | M3_rag_design | 113603 | 775 | 0.2002 | 299479.74 | 299643.63 | True |
| signed_error | M1_family_size | 121351 | 775 | 0.2144 | 371482.07 | 371627.66 | True |
| signed_error | M2_time_size | 121351 | 775 | 0.2136 | 372066.58 | 372192.76 | True |
| signed_error | M3_rag_design | 113603 | 775 | 0.2192 | 346108.76 | 346272.65 | True |

## abs_error — M1_family_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.1988 | 0.0356 | 1.129 | 1.2686 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.4049 | 0.0267 | -0.4573 | -0.3526 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | -0.1533 | 0.0247 | -0.2016 | -0.105 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.3771 | 0.0519 | 0.2753 | 0.4789 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.077 | 0.0471 | -0.0154 | 0.1694 | 0.102231 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1134 | 0.0699 | -0.0236 | 0.2503 | 0.104658 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.491 | 0.0453 | 0.4021 | 0.5798 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0636 | 0.0664 | -0.0665 | 0.1937 | 0.338257 |  |
| size_c | -0.1553 | 0.0063 | -0.1676 | -0.143 | 0.0 | *** |
| is_rag_i | -0.1162 | 0.0192 | -0.1539 | -0.0786 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral]:is_rag_i | 0.1736 | 0.0276 | 0.1194 | 0.2277 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen]:is_rag_i | 0.1003 | 0.0255 | 0.0504 | 0.1503 | 8.2e-05 | *** |
| size_c:is_rag_i | 0.0355 | 0.0065 | 0.0228 | 0.0483 | 0.0 | *** |

## abs_error — M2_time_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.0155 | 0.032 | 0.9527 | 1.0783 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.3771 | 0.0519 | 0.2753 | 0.4789 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.077 | 0.0471 | -0.0154 | 0.1694 | 0.102467 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1133 | 0.0699 | -0.0236 | 0.2503 | 0.104788 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.4909 | 0.0453 | 0.402 | 0.5797 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0635 | 0.0664 | -0.0666 | 0.1936 | 0.338638 |  |
| time_c | -0.2497 | 0.0178 | -0.2845 | -0.2149 | 0.0 | *** |
| size_c | -0.1611 | 0.0063 | -0.1735 | -0.1488 | 0.0 | *** |
| is_rag_i | -0.024 | 0.0105 | -0.0446 | -0.0034 | 0.022597 | * |
| time_c:is_rag_i | 0.1011 | 0.0184 | 0.0652 | 0.1371 | 0.0 | *** |
| size_c:is_rag_i | 0.0397 | 0.0065 | 0.0269 | 0.0525 | 0.0 | *** |

## abs_error — M3_rag_design

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.0765 | 0.0317 | 1.0144 | 1.1386 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.2315 | 0.007 | -0.2451 | -0.2178 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | -0.0528 | 0.0064 | -0.0654 | -0.0402 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.383 | 0.0524 | 0.2804 | 0.4857 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.0812 | 0.0475 | -0.0119 | 0.1744 | 0.087354 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1046 | 0.0704 | -0.0334 | 0.2427 | 0.137488 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.4954 | 0.0457 | 0.4058 | 0.585 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0567 | 0.0669 | -0.0744 | 0.1879 | 0.396468 |  |
| C(strategy, Treatment('simple'))[T.hyde] | 0.0021 | 0.0065 | -0.0107 | 0.0148 | 0.751523 |  |
| C(strategy, Treatment('simple'))[T.twostage] | 0.0046 | 0.0065 | -0.008 | 0.0173 | 0.471771 |  |
| C(embedding_model)[T.e5] | -0.0006 | 0.0083 | -0.0169 | 0.0157 | 0.945446 |  |
| C(embedding_model)[T.jina] | 0.0152 | 0.0085 | -0.0014 | 0.0318 | 0.072287 |  |
| C(embedding_model)[T.qwen3] | -0.0093 | 0.0083 | -0.0256 | 0.007 | 0.263178 |  |
| size_c | -0.1198 | 0.0016 | -0.123 | -0.1165 | 0.0 | *** |
| hybrid | 0.0061 | 0.0083 | -0.0102 | 0.0224 | 0.463575 |  |

## signed_error — M1_family_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.5345 | 0.0457 | 0.4449 | 0.624 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | 0.0139 | 0.0329 | -0.0507 | 0.0784 | 0.673936 |  |
| C(family, Treatment('Llama'))[T.Qwen] | 0.167 | 0.0304 | 0.1074 | 0.2266 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.9062 | 0.0675 | -2.0385 | -1.7738 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0372 | 0.0613 | -0.1573 | 0.0829 | 0.54366 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3359 | 0.0909 | -1.514 | -1.1578 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6542 | 0.059 | 0.5386 | 0.7697 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3114 | 0.0863 | -1.4805 | -1.1422 | 0.0 | *** |
| size_c | -0.0066 | 0.0077 | -0.0218 | 0.0085 | 0.391636 |  |
| is_rag_i | 0.2413 | 0.0237 | 0.1948 | 0.2877 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral]:is_rag_i | -0.4815 | 0.0341 | -0.5483 | -0.4148 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen]:is_rag_i | -0.152 | 0.0314 | -0.2136 | -0.0904 | 1e-06 | *** |
| size_c:is_rag_i | -0.1275 | 0.008 | -0.1432 | -0.1118 | 0.0 | *** |

## signed_error — M2_time_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.6053 | 0.0415 | 0.524 | 0.6865 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.9062 | 0.0675 | -2.0386 | -1.7738 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0372 | 0.0613 | -0.1573 | 0.0829 | 0.543664 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3359 | 0.0909 | -1.514 | -1.1578 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6541 | 0.0589 | 0.5386 | 0.7696 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3114 | 0.0863 | -1.4806 | -1.1423 | 0.0 | *** |
| time_c | -0.0775 | 0.022 | -0.1206 | -0.0345 | 0.000413 | *** |
| size_c | -0.0055 | 0.0078 | -0.0208 | 0.0098 | 0.480846 |  |
| is_rag_i | 0.0352 | 0.013 | 0.0097 | 0.0607 | 0.006796 | ** |
| time_c:is_rag_i | -0.264 | 0.0227 | -0.3085 | -0.2196 | 0.0 | *** |
| size_c:is_rag_i | -0.1294 | 0.0081 | -0.1452 | -0.1136 | 0.0 | *** |

## signed_error — M3_rag_design

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.7667 | 0.041 | 0.6864 | 0.847 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.4679 | 0.0086 | -0.4847 | -0.4512 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | 0.0146 | 0.0079 | -0.0009 | 0.03 | 0.064341 |  |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.916 | 0.068 | -2.0492 | -1.7827 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0388 | 0.0617 | -0.1596 | 0.0821 | 0.529835 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3447 | 0.0914 | -1.5239 | -1.1654 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6433 | 0.0593 | 0.527 | 0.7596 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3205 | 0.0869 | -1.4908 | -1.1503 | 0.0 | *** |
| C(strategy, Treatment('simple'))[T.hyde] | -0.0272 | 0.008 | -0.0428 | -0.0115 | 0.000669 | *** |
| C(strategy, Treatment('simple'))[T.twostage] | 0.031 | 0.0079 | 0.0155 | 0.0465 | 9.2e-05 | *** |
| C(embedding_model)[T.e5] | 0.0266 | 0.0102 | 0.0065 | 0.0466 | 0.00931 | ** |
| C(embedding_model)[T.jina] | 0.0302 | 0.0104 | 0.0098 | 0.0505 | 0.003639 | ** |
| C(embedding_model)[T.qwen3] | -0.0128 | 0.0102 | -0.0328 | 0.0072 | 0.210644 |  |
| size_c | -0.1342 | 0.002 | -0.1382 | -0.1303 | 0.0 | *** |
| hybrid | 0.0235 | 0.0102 | 0.0035 | 0.0435 | 0.021366 | * |
