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
| abs_error | M1_family_size | 121334 | 775 | 0.1966 | 321162.87 | 321308.47 | True |
| abs_error | M2_time_size | 121334 | 775 | 0.1963 | 321359.07 | 321485.25 | True |
| abs_error | M3_rag_design | 113586 | 775 | 0.2001 | 300144.01 | 300307.89 | True |
| signed_error | M1_family_size | 121334 | 775 | 0.2138 | 372328.84 | 372474.44 | True |
| signed_error | M2_time_size | 121334 | 775 | 0.213 | 372910.03 | 373036.21 | True |
| signed_error | M3_rag_design | 113586 | 775 | 0.2187 | 346941.36 | 347105.24 | True |

## abs_error — M1_family_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.2001 | 0.0357 | 1.1301 | 1.2701 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.4049 | 0.0268 | -0.4575 | -0.3524 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | -0.1533 | 0.0247 | -0.2017 | -0.1048 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.3728 | 0.0521 | 0.2707 | 0.4748 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.0741 | 0.0472 | -0.0185 | 0.1667 | 0.11686 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1136 | 0.07 | -0.0237 | 0.2509 | 0.104791 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.4898 | 0.0454 | 0.4007 | 0.5789 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0654 | 0.0665 | -0.065 | 0.1958 | 0.325499 |  |
| size_c | -0.1553 | 0.0063 | -0.1677 | -0.143 | 0.0 | *** |
| is_rag_i | -0.1162 | 0.0193 | -0.154 | -0.0784 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral]:is_rag_i | 0.178 | 0.0277 | 0.1237 | 0.2323 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen]:is_rag_i | 0.1003 | 0.0256 | 0.0502 | 0.1504 | 8.7e-05 | *** |
| size_c:is_rag_i | 0.0356 | 0.0065 | 0.0229 | 0.0484 | 0.0 | *** |

## abs_error — M2_time_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.0168 | 0.0321 | 0.9539 | 1.0798 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.3728 | 0.0521 | 0.2707 | 0.4748 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.074 | 0.0472 | -0.0186 | 0.1666 | 0.117119 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1136 | 0.07 | -0.0237 | 0.2509 | 0.104921 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.4897 | 0.0454 | 0.4007 | 0.5788 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0654 | 0.0665 | -0.065 | 0.1958 | 0.325861 |  |
| time_c | -0.2497 | 0.0178 | -0.2846 | -0.2148 | 0.0 | *** |
| size_c | -0.1611 | 0.0063 | -0.1735 | -0.1487 | 0.0 | *** |
| is_rag_i | -0.0227 | 0.0106 | -0.0434 | -0.002 | 0.031671 | * |
| time_c:is_rag_i | 0.1046 | 0.0184 | 0.0685 | 0.1407 | 0.0 | *** |
| size_c:is_rag_i | 0.0399 | 0.0065 | 0.027 | 0.0527 | 0.0 | *** |

## abs_error — M3_rag_design

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 1.0776 | 0.0318 | 1.0154 | 1.1399 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.2271 | 0.007 | -0.2408 | -0.2133 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | -0.0529 | 0.0064 | -0.0655 | -0.0403 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | 0.3784 | 0.0525 | 0.2755 | 0.4814 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.0781 | 0.0477 | -0.0153 | 0.1715 | 0.101273 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1049 | 0.0707 | -0.0336 | 0.2433 | 0.137755 |  |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.4942 | 0.0458 | 0.4043 | 0.584 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | 0.0587 | 0.0671 | -0.0728 | 0.1903 | 0.381632 |  |
| C(strategy, Treatment('simple'))[T.hyde] | 0.002 | 0.0065 | -0.0108 | 0.0148 | 0.757197 |  |
| C(strategy, Treatment('simple'))[T.twostage] | 0.0086 | 0.0065 | -0.0041 | 0.0213 | 0.185654 |  |
| C(embedding_model)[T.e5] | -0.0028 | 0.0084 | -0.0192 | 0.0135 | 0.733376 |  |
| C(embedding_model)[T.jina] | 0.0129 | 0.0085 | -0.0038 | 0.0295 | 0.129412 |  |
| C(embedding_model)[T.qwen3] | -0.0116 | 0.0083 | -0.0279 | 0.0048 | 0.165365 |  |
| size_c | -0.1197 | 0.0016 | -0.1229 | -0.1164 | 0.0 | *** |
| hybrid | 0.0081 | 0.0083 | -0.0083 | 0.0244 | 0.334763 |  |

## signed_error — M1_family_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.5351 | 0.0458 | 0.4454 | 0.6248 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | 0.0139 | 0.0331 | -0.051 | 0.0787 | 0.675049 |  |
| C(family, Treatment('Llama'))[T.Qwen] | 0.167 | 0.0305 | 0.1072 | 0.2268 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.9021 | 0.0677 | -2.0347 | -1.7695 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0382 | 0.0614 | -0.1585 | 0.0821 | 0.533882 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3391 | 0.091 | -1.5175 | -1.1607 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6519 | 0.0591 | 0.5361 | 0.7676 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3151 | 0.0865 | -1.4845 | -1.1456 | 0.0 | *** |
| size_c | -0.0066 | 0.0078 | -0.0219 | 0.0086 | 0.393226 |  |
| is_rag_i | 0.2413 | 0.0238 | 0.1947 | 0.2879 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral]:is_rag_i | -0.4823 | 0.0342 | -0.5493 | -0.4153 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen]:is_rag_i | -0.152 | 0.0315 | -0.2138 | -0.0902 | 1e-06 | *** |
| size_c:is_rag_i | -0.1275 | 0.008 | -0.1433 | -0.1118 | 0.0 | *** |

## signed_error — M2_time_size

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.6059 | 0.0415 | 0.5245 | 0.6873 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.9022 | 0.0677 | -2.0348 | -1.7696 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0382 | 0.0614 | -0.1585 | 0.0821 | 0.533911 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3391 | 0.091 | -1.5175 | -1.1607 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6518 | 0.0591 | 0.536 | 0.7675 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3151 | 0.0865 | -1.4846 | -1.1457 | 0.0 | *** |
| time_c | -0.0775 | 0.022 | -0.1207 | -0.0343 | 0.000433 | *** |
| size_c | -0.0055 | 0.0078 | -0.0208 | 0.0098 | 0.482302 |  |
| is_rag_i | 0.035 | 0.0131 | 0.0094 | 0.0605 | 0.007367 | ** |
| time_c:is_rag_i | -0.2646 | 0.0228 | -0.3093 | -0.22 | 0.0 | *** |
| size_c:is_rag_i | -0.1294 | 0.0081 | -0.1453 | -0.1136 | 0.0 | *** |

## signed_error — M3_rag_design

| Term | Estimate | SE | CI_lo | CI_hi | p | sig |
|---|---|---|---|---|---|---|
| Intercept | 0.7673 | 0.0411 | 0.6868 | 0.8478 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Mistral] | -0.4688 | 0.0086 | -0.4856 | -0.4519 | 0.0 | *** |
| C(family, Treatment('Llama'))[T.Qwen] | 0.0146 | 0.0079 | -0.0009 | 0.0301 | 0.065313 |  |
| C(party_group, Treatment('SPD'))[T.AfD] | -1.9116 | 0.0681 | -2.0452 | -1.7781 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0398 | 0.0618 | -0.161 | 0.0814 | 0.519857 |  |
| C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.3481 | 0.0917 | -1.5278 | -1.1685 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6408 | 0.0595 | 0.5243 | 0.7574 | 0.0 | *** |
| C(party_group, Treatment('SPD'))[T.FDP] | -1.3245 | 0.0871 | -1.4951 | -1.1539 | 0.0 | *** |
| C(strategy, Treatment('simple'))[T.hyde] | -0.0272 | 0.008 | -0.0429 | -0.0115 | 0.000699 | *** |
| C(strategy, Treatment('simple'))[T.twostage] | 0.0303 | 0.008 | 0.0147 | 0.0459 | 0.000138 | *** |
| C(embedding_model)[T.e5] | 0.0271 | 0.0103 | 0.007 | 0.0472 | 0.008199 | ** |
| C(embedding_model)[T.jina] | 0.0307 | 0.0104 | 0.0103 | 0.0511 | 0.003213 | ** |
| C(embedding_model)[T.qwen3] | -0.0122 | 0.0103 | -0.0323 | 0.0079 | 0.232536 |  |
| size_c | -0.1342 | 0.002 | -0.1382 | -0.1303 | 0.0 | *** |
| hybrid | 0.0234 | 0.0103 | 0.0033 | 0.0435 | 0.022405 | * |
