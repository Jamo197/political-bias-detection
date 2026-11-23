# Item 7: chunk ablations with account-level inference

Paired ΔMAE (condition − reference) on the posts shared with the reference, as in chunk_ablation.md and chunk_ablation_rtop.md. NEW = account cluster bootstrap with the shared draws, two-sided bootstrap p, Holm within model over the comparisons of a table (2 on all posts, 5 on the R_top posts). OLD = post bootstrap. Its Holm family had one comparison per call, so the old 'p_holm' is the unadjusted bootstrap p.

| Table | Model | Condition | Reference | N_paired | Accounts | dMAE [post CI] | Old p_holm | dMAE [account CI] | p_holm_cluster | d|Bias| [account CI] | dr [account CI] | Changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all posts | Llama 8B | e5pool/random5 | no_rag | 775 | 72 | -0.527 [-0.625, -0.429] | 0.001 | -0.527 [-0.748, -0.229] | 0.004 | -0.65 [-0.91, -0.14] | +0.12 [-0.01, +0.21] | no |
| all posts | Llama 8B | e5pool/random5 | e5/simple | 774 | 72 | -0.018 [-0.089, +0.055] | 0.667 | -0.018 [-0.122, +0.120] | 0.792 | +0.01 [-0.15, +0.20] | +0.00 [-0.06, +0.10] | no |
| all posts | Ministral 8B | e5pool/random5 | no_rag | 774 | 71 | +0.173 [+0.101, +0.241] | 0.001 | +0.173 [-0.004, +0.277] | 0.120 | +0.25 [+0.14, +0.35] | -0.03 [-0.08, +0.03] | yes |
| all posts | Ministral 8B | e5pool/random5 | e5/simple | 759 | 71 | +0.102 [+0.018, +0.189] | 0.013 | +0.102 [-0.018, +0.191] | 0.120 | +0.30 [+0.17, +0.45] | -0.01 [-0.07, +0.09] | yes |
| R_top posts | Llama 8B | e5/rtop4plus | no_rag | 65 | 25 | -0.127 [-0.538, +0.300] | 0.530 | -0.127 [-0.752, +0.374] | 1.000 | -0.27 [-1.11, +0.64] | -0.03 [-0.28, +0.25] | no |
| R_top posts | Llama 8B | e5/rtop4plus | e5/simple | 65 | 25 | +0.011 [-0.266, +0.260] | 0.929 | +0.011 [-0.192, +0.288] | 1.000 | -0.20 [-0.42, +0.20] | +0.00 [-0.15, +0.09] | no |
| R_top posts | Llama 8B | e5/rtop4plus | e5pool/random5 | 65 | 25 | +0.192 [-0.129, +0.543] | 0.249 | +0.192 [-0.066, +0.440] | 0.740 | -0.00 [-0.31, +0.35] | -0.13 [-0.29, +0.09] | no |
| R_top posts | Llama 8B | e5pool/random5 | no_rag | 65 | 25 | -0.319 [-0.677, +0.037] | 0.094 | -0.319 [-0.869, +0.157] | 0.764 | -0.26 [-1.12, +0.59] | +0.10 [-0.13, +0.31] | no |
| R_top posts | Llama 8B | e5pool/random5 | e5/simple | 65 | 25 | -0.180 [-0.478, +0.116] | 0.241 | -0.180 [-0.428, +0.188] | 0.935 | -0.19 [-0.50, +0.19] | +0.13 [-0.12, +0.29] | no |
| R_top posts | Ministral 8B | e5/rtop4plus | no_rag | 67 | 18 | +0.173 [-0.220, +0.565] | 0.362 | +0.173 [-0.197, +0.549] | 1.000 | -0.33 [-0.75, +0.34] | -0.02 [-0.42, +0.19] | no |
| R_top posts | Ministral 8B | e5/rtop4plus | e5/simple | 67 | 18 | +0.107 [-0.152, +0.388] | 0.434 | +0.107 [-0.102, +0.374] | 1.000 | -0.35 [-0.55, +0.44] | -0.16 [-0.36, +0.01] | no |
| R_top posts | Ministral 8B | e5/rtop4plus | e5pool/random5 | 67 | 18 | -0.018 [-0.386, +0.384] | 0.933 | -0.018 [-0.387, +0.265] | 1.000 | -0.64 [-1.09, +0.12] | -0.08 [-0.37, +0.13] | no |
| R_top posts | Ministral 8B | e5pool/random5 | no_rag | 67 | 18 | +0.192 [-0.131, +0.507] | 0.246 | +0.192 [-0.165, +0.666] | 1.000 | +0.31 [-0.17, +0.71] | +0.06 [-0.18, +0.21] | no |
| R_top posts | Ministral 8B | e5pool/random5 | e5/simple | 67 | 18 | +0.125 [-0.205, +0.444] | 0.452 | +0.125 [-0.113, +0.452] | 1.000 | +0.29 [-0.11, +0.81] | -0.08 [-0.24, +0.14] | no |

Accounts in the relevant-only (R_top ≥ 4) subsets: Llama 8B 65 posts from 25 accounts, Ministral 8B 67 posts from 18 accounts.

## Sensitivity: without Karl_Lauterbach (71 accounts)

Comparisons whose Holm decision changes: 0


| Table | Model | Condition | Reference | N_paired | Accounts | dMAE | Cluster CI low | Cluster CI high | p_holm_cluster |
|---|---|---|---|---|---|---|---|---|---|
| all posts | Llama 8B | e5pool/random5 | no_rag | 582 | 71 | -0.381 | -0.572 | -0.178 | 0.002 |
| all posts | Llama 8B | e5pool/random5 | e5/simple | 581 | 71 | 0.028 | -0.108 | 0.150 | 0.670 |
| all posts | Ministral 8B | e5pool/random5 | no_rag | 581 | 70 | 0.106 | -0.047 | 0.217 | 0.334 |
| all posts | Ministral 8B | e5pool/random5 | e5/simple | 573 | 70 | 0.098 | -0.045 | 0.209 | 0.334 |
| R_top posts | Llama 8B | e5/rtop4plus | no_rag | 61 | 24 | -0.042 | -0.642 | 0.446 | 1.000 |
| R_top posts | Llama 8B | e5/rtop4plus | e5/simple | 61 | 24 | -0.004 | -0.215 | 0.287 | 1.000 |
| R_top posts | Llama 8B | e5/rtop4plus | e5pool/random5 | 61 | 24 | 0.193 | -0.070 | 0.449 | 0.705 |
| R_top posts | Llama 8B | e5pool/random5 | no_rag | 61 | 24 | -0.235 | -0.762 | 0.242 | 1.000 |
| R_top posts | Llama 8B | e5pool/random5 | e5/simple | 61 | 24 | -0.197 | -0.447 | 0.174 | 1.000 |
| R_top posts | Ministral 8B | e5/rtop4plus | no_rag | 47 | 17 | 0.137 | -0.293 | 0.634 | 1.000 |
| R_top posts | Ministral 8B | e5/rtop4plus | e5/simple | 47 | 17 | 0.145 | -0.130 | 0.439 | 1.000 |
| R_top posts | Ministral 8B | e5/rtop4plus | e5pool/random5 | 47 | 17 | -0.074 | -0.483 | 0.341 | 1.000 |
| R_top posts | Ministral 8B | e5pool/random5 | no_rag | 47 | 17 | 0.211 | -0.268 | 0.761 | 1.000 |
| R_top posts | Ministral 8B | e5pool/random5 | e5/simple | 47 | 17 | 0.219 | -0.106 | 0.511 | 0.875 |