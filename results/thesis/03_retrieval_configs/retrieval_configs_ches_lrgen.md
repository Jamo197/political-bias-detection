# Retrieval configurations (RQ1b, part 1) — target `ches_lrgen`

Data: party_label_* logs, k = 5, bge/hyde_hybrid excluded (same pipeline as bge/hyde: HyDERetrieval ignores the hybrid flag). Paired bootstrap CIs (2000 resamples, seed 42), Wilcoxon signed-rank, Holm within model × factor. ΔMAE = MAE(A) − MAE(B); negative = A better.

## 1. Noise floor (run-to-run)

Cells that were run more than once (≥ 700 shared posts). *fixed context*: ≥ 95% of posts got the identical chunk list, so only LLM sampling differs. *HyDE rerun*: hypothetical texts regenerated. The superseded 2026-08-08 Ministral-8B bge/twostage runs are excluded. SD_run_to_run = RMS(ΔMAE)/√2.

| Kind | N_pairs | Mean_abs_dMAE | Max_abs_dMAE | RMS_dMAE | Mean_pred_identical | Mean_chunks_identical | SD_run_to_run |
|---|---|---|---|---|---|---|---|
| HyDE rerun | 3 | 0.0241 | 0.0657 | 0.038 | 0.4934 | 0.0004 | 0.0269 |
| HyDE rerun (hyde_hybrid) | 7 | 0.0177 | 0.0653 | 0.0271 | 0.6174 | 0.0148 | 0.0192 |
| fixed context | 3 | 0.0111 | 0.0181 | 0.0122 | 0.9486 | 0.9917 | 0.0086 |
| retrieval varies | 3 | 0.0143 | 0.0165 | 0.0144 | 0.8362 | 0.5598 | 0.0102 |
| HyDE (pooled) | 10 | 0.0196 | 0.0657 | 0.0308 | 0.5802 | 0.0105 | 0.0218 |
| all | 16 | 0.017 | 0.0657 | 0.0257 | 0.6973 | 0.2974 | 0.0182 |

| Kind | Model | Condition | N | MAE_a | MAE_b | dMAE | dMAE_lo | dMAE_hi | Pred_identical | Chunks_identical |
|---|---|---|---|---|---|---|---|---|---|---|
| HyDE rerun | Llama-3.1-8B-Instruct | bge/hyde_hybrid | 774 | 1.0661 | 1.0634 | 0.0027 | -0.0649 | 0.0694 | 0.6421 | 0.0013 |
| HyDE rerun | Ministral-3-3B-Instruct-2512 | bge/hyde | 769 | 1.2296 | 1.2335 | -0.0038 | -0.0827 | 0.0746 | 0.4096 | 0.0 |
| HyDE rerun | Ministral-3-3B-Instruct-2512 | bge/hyde_hybrid | 761 | 1.2664 | 1.2007 | 0.0657 | -0.0085 | 0.1373 | 0.4284 | 0.0 |
| HyDE rerun (hyde_hybrid) | Llama-3.1-8B-Instruct | bge/hyde vs bge/hyde_hybrid | 775 | 1.0705 | 1.0629 | 0.0076 | -0.0619 | 0.0777 | 0.6568 | 0.0026 |
| HyDE rerun (hyde_hybrid) | Meta-Llama-3.1-70B-Instruct-FP8 | bge/hyde vs bge/hyde_hybrid | 775 | 1.0884 | 1.1032 | -0.0148 | -0.0753 | 0.0533 | 0.6994 | 0.0026 |
| HyDE rerun (hyde_hybrid) | Ministral-3-14B-Instruct-2512 | bge/hyde vs bge/hyde_hybrid | 769 | 0.9804 | 0.9845 | -0.0041 | -0.0589 | 0.0514 | 0.4369 | 0.0013 |
| HyDE rerun (hyde_hybrid) | Ministral-3-3B-Instruct-2512 | bge/hyde vs bge/hyde_hybrid | 775 | 1.2307 | 1.207 | 0.0238 | -0.0563 | 0.099 | 0.4452 | 0.0 |
| HyDE rerun (hyde_hybrid) | Qwen2.5-3B-Instruct | bge/hyde vs bge/hyde_hybrid | 774 | 1.5137 | 1.509 | 0.0047 | -0.0306 | 0.0401 | 0.7649 | 0.0 |
| HyDE rerun (hyde_hybrid) | Qwen2.5-72B-Instruct-FP8-dynamic | bge/hyde vs bge/hyde_hybrid | 775 | 0.9336 | 0.937 | -0.0034 | -0.0433 | 0.0321 | 0.6968 | 0.0039 |
| HyDE rerun (hyde_hybrid) | Qwen2.5-7B-Instruct | bge/hyde vs bge/hyde_hybrid | 775 | 1.2354 | 1.3006 | -0.0653 | -0.1256 | -0.0091 | 0.6219 | 0.0929 |
| fixed context | Llama-3.1-8B-Instruct | bge/simple | 775 | 1.0668 | 1.0487 | 0.0181 | 0.0043 | 0.035 | 0.9665 | 1.0 |
| fixed context | Llama-3.1-8B-Instruct | bge/twostage | 775 | 1.0356 | 1.0441 | -0.0085 | -0.0224 | 0.0059 | 0.9626 | 1.0 |
| fixed context | Ministral-3-3B-Instruct-2512 | bge/simple | 767 | 1.2168 | 1.2236 | -0.0068 | -0.0255 | 0.0125 | 0.9166 | 0.9752 |
| identical copy (excluded) | Qwen2.5-32B-Instruct | bge/hyde vs bge/hyde_hybrid | 775 | 1.0156 | 1.0156 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| retrieval varies | Llama-3.1-8B-Instruct | bge/simple_hybrid | 775 | 1.0243 | 1.0408 | -0.0165 | -0.0555 | 0.0213 | 0.8465 | 0.369 |
| retrieval varies | Llama-3.1-8B-Instruct | bge/twostage_hybrid | 774 | 1.0624 | 1.0743 | -0.0119 | -0.0439 | 0.0196 | 0.9251 | 0.9302 |
| retrieval varies | Ministral-3-3B-Instruct-2512 | bge/simple_hybrid | 768 | 1.2068 | 1.2212 | -0.0144 | -0.0627 | 0.0325 | 0.737 | 0.3802 |

## 2. Spread of the conditions

Two-way decomposition of the MAE matrix (10 models × 14 conditions, paired CSV of chapter 02).

| Source | df | SS | Share_of_SS | F | p |
|---|---|---|---|---|---|
| Model | 9.0 | 8.204 | 0.9859 | 1179.3838 | 0.0 |
| Condition | 13.0 | 0.0145 | 0.0017 | 1.4478 | 0.1483 |
| Residual (model x condition) | 116.0 | 0.0897 | 0.0108 |  |  |
| SD of model means |  | 0.2566 |  |  |  |
| SD of condition means |  | 0.0149 |  |  |  |
| Mean within-model SD across conditions |  | 0.0271 |  |  |  |
| Run-to-run SD (noise floor) |  | 0.0086 |  |  |  |

| Model | MAE_no_rag | MAE_rag_mean | MAE_rag_min | MAE_rag_max | Range_conditions | SD_conditions | Range_over_noise_SD | Best_condition | Worst_condition |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 1.8041 | 1.7156 | 1.6504 | 1.7794 | 0.129 | 0.0372 | 15.0 | bge/hyde | bge/twostage_hybrid |
| Ministral-3-3B-Instruct-2512 | 1.3165 | 1.2369 | 1.1712 | 1.3145 | 0.1433 | 0.0361 | 16.6628 | qwen3/hyde | jina/simple |
| Qwen2.5-3B-Instruct | 1.4346 | 1.5029 | 1.4817 | 1.5192 | 0.0375 | 0.0115 | 4.3605 | jina/hyde | bge/twostage_hybrid |
| Qwen2.5-7B-Instruct | 1.2021 | 1.2898 | 1.2365 | 1.3299 | 0.0934 | 0.0257 | 10.8605 | bge/hyde | bge/twostage |
| Llama-3.1-8B-Instruct | 1.5488 | 1.0419 | 0.9647 | 1.0946 | 0.1299 | 0.0392 | 15.1047 | qwen3/hyde | jina/simple |
| Ministral-3-8B-Instruct-2512 | 1.1414 | 1.2523 | 1.2118 | 1.3014 | 0.0896 | 0.0296 | 10.4186 | qwen3/simple | jina/hyde |
| Ministral-3-14B-Instruct-2512 | 0.8649 | 0.9317 | 0.8996 | 0.9872 | 0.0876 | 0.0265 | 10.186 | bge/twostage_hybrid | qwen3/hyde |
| Qwen2.5-32B-Instruct | 1.1589 | 1.011 | 0.9555 | 1.0529 | 0.0974 | 0.0259 | 11.3256 | qwen3/hyde | jina/twostage |
| Meta-Llama-3.1-70B-Instruct-FP8 | 0.8117 | 1.0875 | 1.039 | 1.1167 | 0.0777 | 0.0205 | 9.0349 | qwen3/simple | bge/simple |
| Qwen2.5-72B-Instruct-FP8-dynamic | 0.9411 | 0.9219 | 0.8943 | 0.9544 | 0.0601 | 0.0188 | 6.9884 | qwen3/simple | bge/twostage |

## 3. Embedder

Effect = balanced marginal mean of the embedder minus the model mean (additive fit per model on the 12 non-hybrid cells). jina (2048 → 1024) and qwen3 (4096 → 1024) are Matryoshka-truncated; bge and e5 are native 1024.

| Embedder | Mean_effect | SD_effect | Mean_rank | N_best | N_worst | Native_dim | MRL_truncated_to_1024 |
|---|---|---|---|---|---|---|---|
| qwen3 | -0.0104 | 0.0225 | 2.0 | 6 | 2 | 4096 | True |
| e5 | -0.0015 | 0.009 | 2.2 | 2 | 1 | 1024 | False |
| bge | -0.0002 | 0.0153 | 2.8 | 1 | 3 | 1024 | False |
| jina | 0.0122 | 0.0209 | 3.0 | 1 | 4 | 2048 | True |

Post-level, marginal over strategies:

| Model_short | A | B | N | dMAE | dMAE_lo | dMAE_hi | p_holm | significant |
|---|---|---|---|---|---|---|---|---|
| Llama 3B | bge | e5 | 317 | -0.0524 | -0.11 | 0.0053 | 0.1982 | False |
| Llama 3B | bge | jina | 317 | -0.012 | -0.0712 | 0.0488 | 0.8593 | False |
| Llama 3B | bge | qwen3 | 317 | -0.0714 | -0.1311 | -0.0159 | 0.0344 | True |
| Llama 3B | e5 | jina | 317 | 0.0404 | -0.0202 | 0.1006 | 0.2758 | False |
| Llama 3B | e5 | qwen3 | 317 | -0.0189 | -0.0726 | 0.0344 | 0.6055 | False |
| Llama 3B | jina | qwen3 | 317 | -0.0593 | -0.112 | -0.0044 | 0.0851 | False |
| Llama 8B | bge | e5 | 672 | 0.0238 | -0.0224 | 0.0705 | 0.5745 | False |
| Llama 8B | bge | jina | 672 | -0.0198 | -0.065 | 0.0227 | 0.8197 | False |
| Llama 8B | bge | qwen3 | 672 | 0.053 | 0.0089 | 0.0961 | 0.0174 | True |
| Llama 8B | e5 | jina | 672 | -0.0436 | -0.0909 | 0.0026 | 0.1531 | False |
| Llama 8B | e5 | qwen3 | 672 | 0.0292 | -0.0163 | 0.0806 | 0.1531 | False |
| Llama 8B | jina | qwen3 | 672 | 0.0728 | 0.0251 | 0.119 | 0.0053 | True |
| Llama 70B | bge | e5 | 774 | 0.0103 | -0.0306 | 0.0543 | 1.0 | False |
| Llama 70B | bge | jina | 774 | 0.006 | -0.0368 | 0.0483 | 0.7105 | False |
| Llama 70B | bge | qwen3 | 774 | 0.021 | -0.0221 | 0.0649 | 0.1056 | False |
| Llama 70B | e5 | jina | 774 | -0.0044 | -0.0447 | 0.0375 | 1.0 | False |
| Llama 70B | e5 | qwen3 | 774 | 0.0107 | -0.0324 | 0.0527 | 0.7105 | False |
| Llama 70B | jina | qwen3 | 774 | 0.015 | -0.028 | 0.0582 | 0.7105 | False |
| Ministral 3B | bge | e5 | 562 | 0.0001 | -0.0553 | 0.0562 | 1.0 | False |
| Ministral 3B | bge | jina | 562 | -0.0655 | -0.1181 | -0.015 | 0.1125 | False |
| Ministral 3B | bge | qwen3 | 562 | 0.0027 | -0.0514 | 0.059 | 1.0 | False |
| Ministral 3B | e5 | jina | 562 | -0.0656 | -0.1198 | -0.0115 | 0.0936 | False |
| Ministral 3B | e5 | qwen3 | 562 | 0.0025 | -0.0554 | 0.0621 | 1.0 | False |
| Ministral 3B | jina | qwen3 | 562 | 0.0682 | 0.0112 | 0.1204 | 0.0936 | False |
| Ministral 8B | bge | e5 | 635 | 0.0284 | -0.0222 | 0.082 | 0.5963 | False |
| Ministral 8B | bge | jina | 635 | -0.0261 | -0.0776 | 0.0283 | 0.5933 | False |
| Ministral 8B | bge | qwen3 | 635 | 0.0456 | -0.005 | 0.0971 | 0.4218 | False |
| Ministral 8B | e5 | jina | 635 | -0.0545 | -0.1118 | 0.0026 | 0.0905 | False |
| Ministral 8B | e5 | qwen3 | 635 | 0.0172 | -0.0286 | 0.0656 | 0.9753 | False |
| Ministral 8B | jina | qwen3 | 635 | 0.0717 | 0.0219 | 0.1215 | 0.2154 | False |
| Ministral 14B | bge | e5 | 726 | 0.0069 | -0.0308 | 0.047 | 1.0 | False |
| Ministral 14B | bge | jina | 726 | 0.0067 | -0.0357 | 0.0518 | 1.0 | False |
| Ministral 14B | bge | qwen3 | 726 | -0.0123 | -0.053 | 0.0325 | 0.622 | False |
| Ministral 14B | e5 | jina | 726 | -0.0002 | -0.041 | 0.0387 | 1.0 | False |
| Ministral 14B | e5 | qwen3 | 726 | -0.0192 | -0.0583 | 0.0189 | 0.7244 | False |
| Ministral 14B | jina | qwen3 | 726 | -0.0191 | -0.0656 | 0.0263 | 1.0 | False |
| Qwen 3B | bge | e5 | 542 | 0.0083 | -0.0156 | 0.0325 | 1.0 | False |
| Qwen 3B | bge | jina | 542 | 0.0146 | -0.0121 | 0.0401 | 1.0 | False |
| Qwen 3B | bge | qwen3 | 542 | 0.0052 | -0.0203 | 0.0308 | 1.0 | False |
| Qwen 3B | e5 | jina | 542 | 0.0063 | -0.0168 | 0.0288 | 1.0 | False |
| Qwen 3B | e5 | qwen3 | 542 | -0.0031 | -0.0272 | 0.0198 | 1.0 | False |
| Qwen 3B | jina | qwen3 | 542 | -0.0094 | -0.0322 | 0.0142 | 1.0 | False |
| Qwen 7B | bge | e5 | 767 | -0.0008 | -0.0434 | 0.042 | 1.0 | False |
| Qwen 7B | bge | jina | 767 | -0.0086 | -0.0517 | 0.0358 | 1.0 | False |
| Qwen 7B | bge | qwen3 | 767 | -0.0035 | -0.049 | 0.0416 | 1.0 | False |
| Qwen 7B | e5 | jina | 767 | -0.0078 | -0.0545 | 0.0373 | 1.0 | False |
| Qwen 7B | e5 | qwen3 | 767 | -0.0027 | -0.0481 | 0.0412 | 1.0 | False |
| Qwen 7B | jina | qwen3 | 767 | 0.0051 | -0.0401 | 0.0504 | 1.0 | False |
| Qwen 32B | bge | e5 | 775 | -0.0089 | -0.0392 | 0.0226 | 1.0 | False |
| Qwen 32B | bge | jina | 775 | -0.0066 | -0.0405 | 0.0243 | 1.0 | False |
| Qwen 32B | bge | qwen3 | 775 | 0.0167 | -0.0153 | 0.049 | 1.0 | False |
| Qwen 32B | e5 | jina | 775 | 0.0023 | -0.0327 | 0.034 | 1.0 | False |
| Qwen 32B | e5 | qwen3 | 775 | 0.0256 | -0.0049 | 0.0577 | 0.6399 | False |
| Qwen 32B | jina | qwen3 | 775 | 0.0233 | -0.0093 | 0.0583 | 1.0 | False |
| Qwen 72B | bge | e5 | 775 | 0.0336 | 0.0013 | 0.0655 | 0.254 | False |
| Qwen 72B | bge | jina | 775 | 0.0156 | -0.0158 | 0.0476 | 0.8852 | False |
| Qwen 72B | bge | qwen3 | 775 | 0.0405 | 0.0114 | 0.0713 | 0.0629 | False |
| Qwen 72B | e5 | jina | 775 | -0.0179 | -0.052 | 0.0162 | 0.8852 | False |
| Qwen 72B | e5 | qwen3 | 775 | 0.0069 | -0.0199 | 0.0335 | 0.8852 | False |
| Qwen 72B | jina | qwen3 | 775 | 0.0248 | -0.0049 | 0.0554 | 0.5152 | False |

Chunk overlap between embedders (mean per-post Jaccard, averaged over models):

| Factor | Fixed | A | B | N_post_model | Jaccard_chunks | Jaccard_speeches | Mean_shared_of_5 | Share_disjoint | Share_identical_set | Share_same_top1 |
|---|---|---|---|---|---|---|---|---|---|---|
| embedder | simple | bge/simple | e5/simple | 7592 | 0.0861 | 0.1021 | 0.7163 | 0.4658 | 0.0 | 0.1087 |
| embedder | simple | bge/simple | jina/simple | 7544 | 0.1133 | 0.1284 | 0.9096 | 0.4105 | 0.0 | 0.1312 |
| embedder | simple | bge/simple | qwen3/simple | 7608 | 0.1145 | 0.1296 | 0.9156 | 0.4017 | 0.0 | 0.1175 |
| embedder | simple | e5/simple | jina/simple | 7507 | 0.0817 | 0.0965 | 0.67 | 0.5218 | 0.0 | 0.0717 |
| embedder | simple | e5/simple | qwen3/simple | 7574 | 0.1074 | 0.1265 | 0.866 | 0.4222 | 0.0013 | 0.1307 |
| embedder | simple | jina/simple | qwen3/simple | 7528 | 0.128 | 0.1466 | 1.0214 | 0.3478 | 0.0013 | 0.1267 |
| embedder | hyde | bge/hyde | e5/hyde | 7594 | 0.0598 | 0.0719 | 0.5007 | 0.6196 | 0.0001 | 0.0545 |
| embedder | hyde | bge/hyde | jina/hyde | 6388 | 0.0956 | 0.1085 | 0.7711 | 0.4834 | 0.0002 | 0.0931 |
| embedder | hyde | bge/hyde | qwen3/hyde | 7617 | 0.0799 | 0.0936 | 0.6529 | 0.5421 | 0.0 | 0.0786 |
| embedder | hyde | e5/hyde | jina/hyde | 6362 | 0.0591 | 0.0738 | 0.4958 | 0.6176 | 0.0003 | 0.0553 |
| embedder | hyde | e5/hyde | qwen3/hyde | 7589 | 0.0766 | 0.0921 | 0.6301 | 0.5483 | 0.0008 | 0.0766 |
| embedder | hyde | jina/hyde | qwen3/hyde | 6381 | 0.0924 | 0.1096 | 0.7493 | 0.4907 | 0.0006 | 0.0878 |
| embedder | twostage | bge/twostage | e5/twostage | 7591 | 0.1416 | 0.1539 | 1.1068 | 0.3343 | 0.0026 | 0.2112 |
| embedder | twostage | bge/twostage | jina/twostage | 7492 | 0.1695 | 0.1803 | 1.2906 | 0.2842 | 0.004 | 0.2571 |
| embedder | twostage | bge/twostage | qwen3/twostage | 7605 | 0.158 | 0.1694 | 1.2163 | 0.3007 | 0.0026 | 0.2515 |
| embedder | twostage | e5/twostage | jina/twostage | 7459 | 0.1234 | 0.137 | 0.9709 | 0.3848 | 0.0 | 0.2054 |
| embedder | twostage | e5/twostage | qwen3/twostage | 7573 | 0.163 | 0.1766 | 1.2464 | 0.3011 | 0.0 | 0.2413 |
| embedder | twostage | jina/twostage | qwen3/twostage | 7474 | 0.1735 | 0.1893 | 1.32 | 0.2766 | 0.0013 | 0.2705 |

## 4. Retrieval strategy

Post-level, marginal over embedders:

| Model_short | A | B | N | dMAE | dMAE_lo | dMAE_hi | p_holm | significant |
|---|---|---|---|---|---|---|---|---|
| Llama 3B | hyde | twostage | 317 | -0.0212 | -0.0627 | 0.021 | 1.0 | False |
| Llama 3B | simple | hyde | 317 | 0.0189 | -0.0283 | 0.0673 | 1.0 | False |
| Llama 3B | simple | twostage | 317 | -0.0023 | -0.0459 | 0.043 | 1.0 | False |
| Llama 8B | hyde | twostage | 672 | -0.019 | -0.0556 | 0.0186 | 0.2736 | False |
| Llama 8B | simple | hyde | 672 | 0.0199 | -0.0152 | 0.0566 | 0.2736 | False |
| Llama 8B | simple | twostage | 672 | 0.0008 | -0.0342 | 0.0356 | 0.8147 | False |
| Llama 70B | hyde | twostage | 774 | 0.0187 | -0.0165 | 0.057 | 0.834 | False |
| Llama 70B | simple | hyde | 774 | -0.0037 | -0.0363 | 0.0292 | 1.0 | False |
| Llama 70B | simple | twostage | 774 | 0.015 | -0.0244 | 0.0515 | 1.0 | False |
| Ministral 3B | hyde | twostage | 562 | 0.0376 | -0.0063 | 0.0818 | 0.1144 | False |
| Ministral 3B | simple | hyde | 562 | 0.0062 | -0.0359 | 0.0512 | 0.838 | False |
| Ministral 3B | simple | twostage | 562 | 0.0438 | -0.0007 | 0.0893 | 0.1144 | False |
| Ministral 8B | hyde | twostage | 635 | -0.0043 | -0.0555 | 0.0445 | 1.0 | False |
| Ministral 8B | simple | hyde | 635 | -0.0047 | -0.0512 | 0.0412 | 1.0 | False |
| Ministral 8B | simple | twostage | 635 | -0.009 | -0.0507 | 0.0359 | 1.0 | False |
| Ministral 14B | hyde | twostage | 726 | 0.0376 | -0.0009 | 0.0756 | 0.0376 | True |
| Ministral 14B | simple | hyde | 726 | -0.0308 | -0.0629 | -0.0001 | 0.2509 | False |
| Ministral 14B | simple | twostage | 726 | 0.0069 | -0.0231 | 0.0372 | 0.2509 | False |
| Qwen 3B | hyde | twostage | 542 | 0.0108 | -0.0095 | 0.0313 | 1.0 | False |
| Qwen 3B | simple | hyde | 542 | 0.0027 | -0.0172 | 0.023 | 1.0 | False |
| Qwen 3B | simple | twostage | 542 | 0.0135 | -0.0036 | 0.0306 | 0.2526 | False |
| Qwen 7B | hyde | twostage | 767 | -0.0371 | -0.0767 | 0.0016 | 1.0 | False |
| Qwen 7B | simple | hyde | 767 | 0.0093 | -0.0226 | 0.04 | 1.0 | False |
| Qwen 7B | simple | twostage | 767 | -0.0278 | -0.066 | 0.0083 | 0.9276 | False |
| Qwen 32B | hyde | twostage | 775 | -0.0258 | -0.0518 | -0.001 | 0.0808 | False |
| Qwen 32B | simple | hyde | 775 | -0.0051 | -0.0291 | 0.0211 | 0.4649 | False |
| Qwen 32B | simple | twostage | 775 | -0.0309 | -0.0578 | -0.0049 | 0.0808 | False |
| Qwen 72B | hyde | twostage | 775 | 0.0047 | -0.0222 | 0.0309 | 0.821 | False |
| Qwen 72B | simple | hyde | 775 | -0.0162 | -0.0386 | 0.0058 | 0.3798 | False |
| Qwen 72B | simple | twostage | 775 | -0.0114 | -0.0358 | 0.0138 | 0.3798 | False |

HyDE − simple against model size (cell MAEs, mean over embedders):

| Model | Family | Params_B | dMAE_hyde_simple | dMAE_twostage_simple | log2_params | dHyDE_bge | dHyDE_e5 | dHyDE_jina | dHyDE_qwen3 |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | Llama | 3 | -0.0015 | 0.0374 | 1.585 | -0.0142 | 0.0247 | 0.0018 | -0.0181 |
| Llama-3.1-8B-Instruct | Llama | 8 | -0.0214 | 0.0031 | 3.0 | 0.0227 | -0.0537 | -0.0203 | -0.0343 |
| Meta-Llama-3.1-70B-Instruct-FP8 | Llama | 70 | 0.0037 | -0.0147 | 6.1293 | -0.0283 | 0.0049 | -0.0235 | 0.0618 |
| Ministral-3-3B-Instruct-2512 | Mistral | 3 | -0.0186 | -0.0392 | 1.585 | 0.0096 | 0.005 | -0.0198 | -0.0691 |
| Ministral-3-8B-Instruct-2512 | Mistral | 8 | 0.0098 | 0.0088 | 3.0 | 0.0053 | 0.0011 | 0.0042 | 0.0286 |
| Ministral-3-14B-Instruct-2512 | Mistral | 14 | 0.0387 | -0.0083 | 3.8074 | 0.0686 | 0.0399 | -0.0048 | 0.0511 |
| Qwen2.5-3B-Instruct | Qwen | 3 | -0.007 | -0.0047 | 1.585 | 0.0121 | -0.0139 | -0.0206 | -0.0055 |
| Qwen2.5-7B-Instruct | Qwen | 7 | -0.0085 | 0.0282 | 2.8074 | -0.0503 | 0.0179 | -0.0029 | 0.0011 |
| Qwen2.5-32B-Instruct | Qwen | 32 | 0.0051 | 0.0309 | 5.0 | 0.0175 | 0.023 | 0.0101 | -0.0303 |
| Qwen2.5-72B-Instruct-FP8-dynamic | Qwen | 72 | 0.0162 | 0.0108 | 6.1699 | 0.0096 | 0.0306 |  | 0.0083 |

| Contrast | N_models | Spearman_rho_vs_size | p_spearman | Slope_per_doubling | Slope_lo | Slope_hi | p_slope |
|---|---|---|---|---|---|---|---|
| dMAE_hyde_simple | 10 | 0.6278 | 0.052 | 0.0048 | -0.0012 | 0.0108 | 0.1557 |
| dMAE_twostage_simple | 10 | 0.0123 | 0.9731 | 0.0008 | -0.0083 | 0.0099 | 0.868 |

Does the reranker / HyDE change the chunks? (same embedder):

| Factor | Fixed | A | B | N_post_model | Jaccard_chunks | Jaccard_speeches | Mean_shared_of_5 | Share_disjoint | Share_identical_set | Share_same_top1 |
|---|---|---|---|---|---|---|---|---|---|---|
| strategy | bge | bge/simple | bge/hyde | 7624 | 0.2415 | 0.2511 | 1.7492 | 0.1804 | 0.0094 | 0.2671 |
| strategy | bge | bge/simple | bge/twostage | 7625 | 0.2667 | 0.2787 | 1.9949 | 0.0548 | 0.0013 | 0.1297 |
| strategy | bge | bge/hyde | bge/twostage | 7625 | 0.1472 | 0.1593 | 1.1559 | 0.3103 | 0.0003 | 0.0775 |
| strategy | e5 | e5/simple | e5/hyde | 7559 | 0.2947 | 0.3041 | 2.0714 | 0.1144 | 0.0177 | 0.3 |
| strategy | e5 | e5/simple | e5/twostage | 7557 | 0.2578 | 0.2765 | 1.941 | 0.0413 | 0.004 | 0.1257 |
| strategy | e5 | e5/hyde | e5/twostage | 7564 | 0.1518 | 0.1669 | 1.2054 | 0.2587 | 0.0003 | 0.0801 |
| strategy | jina | jina/simple | jina/hyde | 6363 | 0.2913 | 0.3036 | 2.0369 | 0.1355 | 0.0167 | 0.3093 |
| strategy | jina | jina/simple | jina/twostage | 7409 | 0.2684 | 0.2902 | 1.9961 | 0.0424 | 0.0013 | 0.1389 |
| strategy | jina | jina/hyde | jina/twostage | 6257 | 0.1708 | 0.1889 | 1.3268 | 0.2343 | 0.0014 | 0.0965 |
| strategy | qwen3 | qwen3/simple | qwen3/hyde | 7599 | 0.3455 | 0.3558 | 2.3395 | 0.097 | 0.0309 | 0.3727 |
| strategy | qwen3 | qwen3/simple | qwen3/twostage | 7591 | 0.2612 | 0.2805 | 1.9573 | 0.0516 | 0.0 | 0.1425 |
| strategy | qwen3 | qwen3/hyde | qwen3/twostage | 7598 | 0.1785 | 0.1952 | 1.3821 | 0.2247 | 0.0016 | 0.0909 |

Determinism of retrieval across the ten models:

| Condition | N_posts | Share_identical_list_across_models | Share_identical_set_across_models |
|---|---|---|---|
| bge/hyde | 775 | 0.0 | 0.0 |
| bge/simple | 775 | 0.9948 | 0.9948 |
| bge/simple_hybrid | 775 | 0.1716 | 0.5213 |
| bge/twostage | 775 | 0.9961 | 0.9961 |
| bge/twostage_hybrid | 775 | 0.8516 | 0.8516 |
| e5/hyde | 775 | 0.0 | 0.0 |
| e5/simple | 775 | 1.0 | 1.0 |
| e5/twostage | 775 | 0.9974 | 0.9974 |
| jina/hyde | 775 | 0.0026 | 0.0026 |
| jina/simple | 775 | 0.68 | 0.8826 |
| jina/twostage | 775 | 0.8555 | 0.8555 |
| qwen3/hyde | 775 | 0.0 | 0.0 |
| qwen3/simple | 775 | 0.8387 | 0.9303 |
| qwen3/twostage | 775 | 0.9174 | 0.9174 |

## 5. Hybrid search (bge only)

| Model_short | Scope | N | dMAE | dMAE_lo | dMAE_hi | p_holm | significant |
|---|---|---|---|---|---|---|---|
| Llama 3B | bge/simple | 664 | 0.0229 | -0.0366 | 0.084 | 0.3583 | False |
| Llama 3B | bge/twostage | 664 | 0.0443 | -0.0146 | 0.1062 | 0.3076 | False |
| Llama 8B | bge/simple | 775 | -0.0079 | -0.0728 | 0.0568 | 0.6332 | False |
| Llama 8B | bge/twostage | 775 | 0.0296 | -0.0323 | 0.089 | 0.6332 | False |
| Llama 70B | bge/simple | 775 | -0.0065 | -0.0723 | 0.0602 | 1.0 | False |
| Llama 70B | bge/twostage | 774 | 0.0128 | -0.0431 | 0.0685 | 1.0 | False |
| Ministral 3B | bge/simple | 768 | -0.0053 | -0.0762 | 0.0659 | 0.8361 | False |
| Ministral 3B | bge/twostage | 765 | 0.0417 | -0.0248 | 0.1088 | 0.6729 | False |
| Ministral 8B | bge/simple | 744 | 0.0216 | -0.0497 | 0.0957 | 1.0 | False |
| Ministral 8B | bge/twostage | 752 | 0.0047 | -0.0615 | 0.0743 | 1.0 | False |
| Ministral 14B | bge/simple | 767 | 0.0133 | -0.0286 | 0.0541 | 0.0942 | False |
| Ministral 14B | bge/twostage | 765 | -0.0134 | -0.0666 | 0.0377 | 0.804 | False |
| Qwen 3B | bge/simple | 774 | -0.0183 | -0.0511 | 0.0147 | 0.4826 | False |
| Qwen 3B | bge/twostage | 775 | 0.012 | -0.0191 | 0.0433 | 0.497 | False |
| Qwen 7B | bge/simple | 775 | 0.024 | -0.0388 | 0.0797 | 0.6072 | False |
| Qwen 7B | bge/twostage | 774 | -0.0093 | -0.0636 | 0.0491 | 0.622 | False |
| Qwen 32B | bge/simple | 775 | 0.0108 | -0.034 | 0.0548 | 0.3517 | False |
| Qwen 32B | bge/twostage | 775 | 0.0408 | -0.0025 | 0.0845 | 0.0865 | False |
| Qwen 72B | bge/simple | 775 | 0.0048 | -0.0306 | 0.0403 | 1.0 | False |
| Qwen 72B | bge/twostage | 775 | -0.0075 | -0.043 | 0.0295 | 1.0 | False |

| Factor | Fixed | A | B | N_post_model | Jaccard_chunks | Jaccard_speeches | Mean_shared_of_5 | Share_disjoint | Share_identical_set | Share_same_top1 |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | bge | bge/simple | bge/simple_hybrid | 7592 | 0.3916 | 0.3993 | 2.7551 | 0.0005 | 0.0058 | 0.5042 |
| hybrid | bge | bge/twostage | bge/twostage_hybrid | 7594 | 0.3751 | 0.3834 | 2.562 | 0.0338 | 0.0273 | 0.51 |
| hybrid | bge | bge/simple_hybrid | bge/twostage_hybrid | 7563 | 0.2667 | 0.2801 | 1.9914 | 0.0516 | 0.0013 | 0.131 |

Chunks only in the hybrid list vs only in the dense list (same post). Coverage = share of the post's content words (≥ 3 letters, no stop words) that occur in the chunk.

| Strategy | N_posts | Share_posts_changed | Mean_chunks_replaced | Coverage_shared | Coverage_only_dense | Coverage_only_hybrid | dCoverage_hybrid_minus_dense | p_wilcoxon_coverage | SameParty_only_dense | SameParty_only_hybrid | dSameParty_hybrid_minus_dense | p_wilcoxon_sameparty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simple | 775 | 0.9948 | 2.2451 | 0.1273 | 0.1035 | 0.1667 | 0.0632 | 0.0 | 0.2729 | 0.2594 | -0.0141 | 0.3459 |
| twostage | 775 | 0.9729 | 2.5146 | 0.1446 | 0.1166 | 0.1667 | 0.0502 | 0.0 | 0.2684 | 0.2417 | -0.0277 | 0.061 |

## 6. Party agreement of the chunks

| Condition | Same_party | Same_party_sd_models | Same_bloc | Party_distance | MAE_mean_models |
|---|---|---|---|---|---|
| bge/hyde | 0.2879 | 0.0136 | 0.524 | 1.2728 | 1.1976 |
| bge/simple | 0.2965 | 0.0008 | 0.5276 | 1.2736 | 1.1924 |
| bge/simple_hybrid | 0.2902 | 0.0013 | 0.5215 | 1.2986 | 1.1997 |
| bge/twostage | 0.291 | 0.0003 | 0.5286 | 1.2878 | 1.2018 |
| bge/twostage_hybrid | 0.2808 | 0.0014 | 0.5123 | 1.3279 | 1.2186 |
| e5/hyde | 0.2693 | 0.0085 | 0.5392 | 1.2369 | 1.1998 |
| e5/simple | 0.3002 | 0.0005 | 0.5467 | 1.2232 | 1.1918 |
| e5/twostage | 0.3032 | 0.0007 | 0.5365 | 1.2156 | 1.1969 |
| jina/hyde | 0.2855 | 0.0107 | 0.5099 | 1.3192 | 1.2389 |
| jina/simple | 0.2926 | 0.0007 | 0.5044 | 1.341 | 1.2145 |
| jina/twostage | 0.2883 | 0.0028 | 0.5079 | 1.329 | 1.2063 |
| qwen3/hyde | 0.3066 | 0.0121 | 0.5515 | 1.1851 | 1.1816 |
| qwen3/simple | 0.3398 | 0.0009 | 0.5655 | 1.1634 | 1.1825 |
| qwen3/twostage | 0.3158 | 0.0015 | 0.5427 | 1.219 | 1.1977 |

Within-model Spearman correlation across conditions: same-party share vs MAE.

| Model | N_conditions | Spearman_sameparty_MAE | p |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 14 | -0.4462 | 0.1098 |
| Llama-3.2-3B-Instruct | 14 | 0.2923 | 0.3105 |
| Meta-Llama-3.1-70B-Instruct-FP8 | 14 | -0.2659 | 0.3581 |
| Ministral-3-14B-Instruct-2512 | 14 | 0.3407 | 0.2333 |
| Ministral-3-3B-Instruct-2512 | 14 | -0.3319 | 0.2464 |
| Ministral-3-8B-Instruct-2512 | 14 | -0.4462 | 0.1098 |
| Qwen2.5-32B-Instruct | 14 | -0.5429 | 0.0449 |
| Qwen2.5-3B-Instruct | 14 | 0.1692 | 0.563 |
| Qwen2.5-72B-Instruct-FP8-dynamic | 13 | -0.9066 | 0.0 |
| Qwen2.5-7B-Instruct | 14 | -0.2703 | 0.3499 |

## 7. What explains the variation — mixed model M3

Random intercept per post, RAG rows only. size_c = centred log2(parameters): its coefficient is the change per doubling of model size. References: Llama, SPD, simple, dense, bge.

| Outcome | Spec | Term | Estimate | SE | CI_lo | CI_hi | p |
|---|---|---|---|---|---|---|---|
| abs_error | M3_rag_design | Intercept | 1.0915 | 0.0318 | 1.0291 | 1.1539 | 0.0 |
| abs_error | M3_rag_design | C(family, Treatment('Llama'))[T.Mistral] | -0.231 | 0.0073 | -0.2453 | -0.2168 | 0.0 |
| abs_error | M3_rag_design | C(family, Treatment('Llama'))[T.Qwen] | -0.0575 | 0.0067 | -0.0706 | -0.0444 | 0.0 |
| abs_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.AfD] | 0.333 | 0.0526 | 0.23 | 0.4361 | 0.0 |
| abs_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.0612 | 0.0477 | -0.0323 | 0.1547 | 0.1994 |
| abs_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1227 | 0.0707 | -0.0159 | 0.2613 | 0.0828 |
| abs_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.5103 | 0.0459 | 0.4204 | 0.6002 | 0.0 |
| abs_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.FDP] | 0.0697 | 0.0672 | -0.062 | 0.2014 | 0.2994 |
| abs_error | M3_rag_design | C(strategy, Treatment('simple'))[T.hyde] | 0.0032 | 0.0071 | -0.0108 | 0.0172 | 0.6514 |
| abs_error | M3_rag_design | C(strategy, Treatment('simple'))[T.twostage] | 0.0087 | 0.0065 | -0.004 | 0.0214 | 0.1791 |
| abs_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.e5] | -0.0025 | 0.0084 | -0.0189 | 0.0138 | 0.7617 |
| abs_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.jina] | 0.0134 | 0.0085 | -0.0033 | 0.03 | 0.1152 |
| abs_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.qwen3] | -0.0116 | 0.0084 | -0.0279 | 0.0048 | 0.1664 |
| abs_error | M3_rag_design | size_c | -0.1197 | 0.0017 | -0.123 | -0.1163 | 0.0 |
| abs_error | M3_rag_design | hybrid | 0.0097 | 0.0096 | -0.0091 | 0.0285 | 0.3099 |
| abs_error | M3_no_design | Intercept | 1.0967 | 0.0311 | 1.0357 | 1.1577 | 0.0 |
| abs_error | M3_no_design | C(family, Treatment('Llama'))[T.Mistral] | -0.231 | 0.0073 | -0.2453 | -0.2168 | 0.0 |
| abs_error | M3_no_design | C(family, Treatment('Llama'))[T.Qwen] | -0.0577 | 0.0067 | -0.0708 | -0.0446 | 0.0 |
| abs_error | M3_no_design | C(party_group, Treatment('SPD'))[T.AfD] | 0.333 | 0.0526 | 0.23 | 0.4361 | 0.0 |
| abs_error | M3_no_design | C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | 0.0612 | 0.0477 | -0.0323 | 0.1547 | 0.1994 |
| abs_error | M3_no_design | C(party_group, Treatment('SPD'))[T.CDU/CSU] | 0.1227 | 0.0707 | -0.0159 | 0.2613 | 0.0828 |
| abs_error | M3_no_design | C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.5103 | 0.0459 | 0.4204 | 0.6002 | 0.0 |
| abs_error | M3_no_design | C(party_group, Treatment('SPD'))[T.FDP] | 0.0697 | 0.0672 | -0.062 | 0.2014 | 0.2995 |
| abs_error | M3_no_design | size_c | -0.1197 | 0.0017 | -0.1231 | -0.1164 | 0.0 |
| signed_error | M3_rag_design | Intercept | 0.785 | 0.0411 | 0.7044 | 0.8656 | 0.0 |
| signed_error | M3_rag_design | C(family, Treatment('Llama'))[T.Mistral] | -0.4767 | 0.0089 | -0.4941 | -0.4593 | 0.0 |
| signed_error | M3_rag_design | C(family, Treatment('Llama'))[T.Qwen] | 0.0122 | 0.0082 | -0.0039 | 0.0282 | 0.1376 |
| signed_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.AfD] | -1.8802 | 0.0681 | -2.0137 | -1.7466 | 0.0 |
| signed_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0914 | 0.0618 | -0.2126 | 0.0298 | 0.1395 |
| signed_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.4126 | 0.0917 | -1.5922 | -1.2329 | 0.0 |
| signed_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6702 | 0.0595 | 0.5536 | 0.7868 | 0.0 |
| signed_error | M3_rag_design | C(party_group, Treatment('SPD'))[T.FDP] | -1.3763 | 0.0871 | -1.547 | -1.2057 | 0.0 |
| signed_error | M3_rag_design | C(strategy, Treatment('simple'))[T.hyde] | -0.021 | 0.0088 | -0.0381 | -0.0038 | 0.0166 |
| signed_error | M3_rag_design | C(strategy, Treatment('simple'))[T.twostage] | 0.0303 | 0.008 | 0.0148 | 0.0459 | 0.0001 |
| signed_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.e5] | 0.0271 | 0.0103 | 0.007 | 0.0472 | 0.0082 |
| signed_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.jina] | 0.0309 | 0.0104 | 0.0105 | 0.0513 | 0.003 |
| signed_error | M3_rag_design | C(embedding_model, Treatment('bge'))[T.qwen3] | -0.0122 | 0.0102 | -0.0323 | 0.0078 | 0.232 |
| signed_error | M3_rag_design | size_c | -0.1353 | 0.0021 | -0.1394 | -0.1312 | 0.0 |
| signed_error | M3_rag_design | hybrid | 0.0335 | 0.0118 | 0.0105 | 0.0566 | 0.0043 |
| signed_error | M3_no_design | Intercept | 0.8042 | 0.0403 | 0.7252 | 0.8832 | 0.0 |
| signed_error | M3_no_design | C(family, Treatment('Llama'))[T.Mistral] | -0.4765 | 0.0089 | -0.4939 | -0.459 | 0.0 |
| signed_error | M3_no_design | C(family, Treatment('Llama'))[T.Qwen] | 0.0126 | 0.0082 | -0.0035 | 0.0286 | 0.1252 |
| signed_error | M3_no_design | C(party_group, Treatment('SPD'))[T.AfD] | -1.8802 | 0.0681 | -2.0137 | -1.7466 | 0.0 |
| signed_error | M3_no_design | C(party_group, Treatment('SPD'))[T.BÜNDNIS 90/DIE GRÜNEN] | -0.0914 | 0.0618 | -0.2126 | 0.0298 | 0.1396 |
| signed_error | M3_no_design | C(party_group, Treatment('SPD'))[T.CDU/CSU] | -1.4125 | 0.0917 | -1.5922 | -1.2329 | 0.0 |
| signed_error | M3_no_design | C(party_group, Treatment('SPD'))[T.DIE LINKE] | 0.6702 | 0.0595 | 0.5536 | 0.7868 | 0.0 |
| signed_error | M3_no_design | C(party_group, Treatment('SPD'))[T.FDP] | -1.3763 | 0.0871 | -1.547 | -1.2057 | 0.0 |
| signed_error | M3_no_design | size_c | -0.1352 | 0.0021 | -0.1394 | -0.1311 | 0.0 |

| Outcome | Spec | N | N_texts | Var_text | Var_resid | ICC | LogLik | AIC | BIC | Converged | Formula |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abs_error | M3_rag_design | 105913.0 | 775.0 | 0.2006 | 0.8033 | 0.1999 | -140061.27 | 280156.54 | 280319.23 | True | abs_error ~ C(family, Treatment('Llama')) + size_c + C(party_group, Treatment('SPD')) + C(strategy, Treatment('simple')) + hybrid + C(embedding_model, Treatment('bge')) |
| abs_error | M3_no_design | 105913.0 | 775.0 | 0.2006 | 0.8033 | 0.1998 | -140067.49 | 280156.99 | 280262.26 | True | abs_error ~ C(family, Treatment('Llama')) + size_c + C(party_group, Treatment('SPD')) |
| abs_error | LR test design factors |  |  |  |  |  | 12.45 | 6.0 | 0.0526 |  | LR = 2ΔlogLik (LogLik col), df (AIC col), p (BIC col) |
| signed_error | M3_rag_design | 105913.0 | 775.0 | 0.3382 | 1.2096 | 0.2185 | -161781.9 | 323597.81 | 323760.5 | True | signed_error ~ C(family, Treatment('Llama')) + size_c + C(party_group, Treatment('SPD')) + C(strategy, Treatment('simple')) + hybrid + C(embedding_model, Treatment('bge')) |
| signed_error | M3_no_design | 105913.0 | 775.0 | 0.3382 | 1.2104 | 0.2184 | -161818.96 | 323659.91 | 323765.19 | True | signed_error ~ C(family, Treatment('Llama')) + size_c + C(party_group, Treatment('SPD')) |
| signed_error | LR test design factors |  |  |  |  |  | 74.1 | 6.0 | 0.0 |  | LR = 2ΔlogLik (LogLik col), df (AIC col), p (BIC col) |

## 8. Number of chunks k (e5/simple)

k = 0 is the no-RAG run, k = 5 comes from the main matrix, the other k from the k-ablation runs (2026-10-14, 2026-10-15).

| Subset | Model | K | Source | N | MAE | MAE_lo | MAE_hi | Mean_signed_error | Calibration_slope | Same_party_share |
|---|---|---|---|---|---|---|---|---|---|---|
| all_available | Llama-3.1-8B-Instruct | 0 | 2026-07-15_eval_20260715_222803 | 775 | 1.5488 | 1.4523 | 1.6454 | 0.884 | 0.6006 |  |
| all_available | Llama-3.1-8B-Instruct | 1 | 2026-10-15_eval_matrix_20261015_173302 | 775 | 1.209 | 1.1216 | 1.2962 | 0.2793 | 0.5939 | 0.3281 |
| all_available | Llama-3.1-8B-Instruct | 3 | 2026-10-15_eval_matrix_20261015_173302 | 775 | 1.0496 | 0.9734 | 1.1271 | 0.1464 | 0.6027 | 0.3146 |
| all_available | Llama-3.1-8B-Instruct | 5 | 2026-08-03_eval_matrix_20260803_224143 | 774 | 1.0395 | 0.9687 | 1.1115 | 0.2241 | 0.5847 | 0.2994 |
| all_available | Llama-3.1-8B-Instruct | 10 | 2026-10-15_eval_matrix_20261015_173302 | 774 | 0.9943 | 0.9257 | 1.0634 | 0.215 | 0.5824 | 0.2946 |
| all_available | Llama-3.1-8B-Instruct | 20 | 2026-10-15_eval_matrix_20261015_173302 | 774 | 1.1006 | 1.0239 | 1.1769 | 0.2831 | 0.5584 | 0.2812 |
| common_texts | Llama-3.1-8B-Instruct | 0 | 2026-07-15_eval_20260715_222803 | 772 | 1.5451 | 1.4442 | 1.6402 | 0.8777 | 0.6015 |  |
| common_texts | Llama-3.1-8B-Instruct | 1 | 2026-10-15_eval_matrix_20261015_173302 | 772 | 1.2081 | 1.1172 | 1.2995 | 0.2766 | 0.5935 | 0.3281 |
| common_texts | Llama-3.1-8B-Instruct | 3 | 2026-10-15_eval_matrix_20261015_173302 | 772 | 1.0451 | 0.9694 | 1.124 | 0.1393 | 0.6036 | 0.3146 |
| common_texts | Llama-3.1-8B-Instruct | 5 | 2026-08-03_eval_matrix_20260803_224143 | 772 | 1.037 | 0.9662 | 1.1109 | 0.2203 | 0.5856 | 0.2992 |
| common_texts | Llama-3.1-8B-Instruct | 10 | 2026-10-15_eval_matrix_20261015_173302 | 772 | 0.9956 | 0.9288 | 1.0626 | 0.2152 | 0.5819 | 0.294 |
| common_texts | Llama-3.1-8B-Instruct | 20 | 2026-10-15_eval_matrix_20261015_173302 | 772 | 1.0978 | 1.026 | 1.1773 | 0.2782 | 0.5592 | 0.2808 |
| all_available | Ministral-3-14B-Instruct-2512 | 0 | 2026-08-24_eval_matrix_20260824_165851 | 775 | 0.8649 | 0.8011 | 0.9309 | -0.2187 | 0.6288 |  |
| all_available | Ministral-3-14B-Instruct-2512 | 1 | 2026-10-14_eval_matrix_20261014_162114 | 774 | 0.9934 | 0.9182 | 1.0674 | -0.2892 | 0.6628 | 0.3281 |
| all_available | Ministral-3-14B-Instruct-2512 | 3 | 2026-10-14_eval_matrix_20261014_162114 | 769 | 0.9694 | 0.9002 | 1.0408 | -0.2893 | 0.6699 | 0.3136 |
| all_available | Ministral-3-14B-Instruct-2512 | 5 | 2026-08-24_eval_matrix_20260824_183010 | 770 | 0.914 | 0.8502 | 0.9791 | -0.1601 | 0.7466 | 0.3007 |
| all_available | Ministral-3-14B-Instruct-2512 | 10 | 2026-10-14_eval_matrix_20261014_162114 | 765 | 0.9079 | 0.845 | 0.9759 | -0.1839 | 0.7269 | 0.2939 |
| all_available | Ministral-3-14B-Instruct-2512 | 20 | 2026-10-14_eval_matrix_20261014_162114 | 596 | 0.8706 | 0.7982 | 0.9451 | -0.2423 | 0.7394 | 0.2817 |
| common_texts | Ministral-3-14B-Instruct-2512 | 0 | 2026-08-24_eval_matrix_20260824_165851 | 576 | 0.8444 | 0.7729 | 0.9215 | -0.2629 | 0.6477 |  |
| common_texts | Ministral-3-14B-Instruct-2512 | 1 | 2026-10-14_eval_matrix_20261014_162114 | 576 | 0.9484 | 0.8691 | 1.035 | -0.2765 | 0.7065 | 0.3357 |
| common_texts | Ministral-3-14B-Instruct-2512 | 3 | 2026-10-14_eval_matrix_20261014_162114 | 576 | 0.9324 | 0.8538 | 1.0124 | -0.3036 | 0.7003 | 0.3215 |
| common_texts | Ministral-3-14B-Instruct-2512 | 5 | 2026-08-24_eval_matrix_20260824_183010 | 576 | 0.8734 | 0.8002 | 0.9464 | -0.1914 | 0.7735 | 0.3035 |
| common_texts | Ministral-3-14B-Instruct-2512 | 10 | 2026-10-14_eval_matrix_20261014_162114 | 576 | 0.8613 | 0.7941 | 0.935 | -0.2221 | 0.7573 | 0.2968 |
| common_texts | Ministral-3-14B-Instruct-2512 | 20 | 2026-10-14_eval_matrix_20261014_162114 | 576 | 0.8796 | 0.8061 | 0.9549 | -0.2431 | 0.7359 | 0.2809 |

| Model | Scope | A | B | N | dMAE | dMAE_lo | dMAE_hi | p_holm | significant |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | vs k=0 | k=1 | k=0 | 775 | -0.3398 | -0.4431 | -0.2397 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=0 | k=3 | k=0 | 775 | -0.4992 | -0.6019 | -0.4016 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=0 | k=5 | k=0 | 774 | -0.5104 | -0.6026 | -0.4168 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=0 | k=10 | k=0 | 774 | -0.5524 | -0.6482 | -0.4526 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=0 | k=20 | k=0 | 774 | -0.4455 | -0.5468 | -0.3488 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=5 | k=0 | k=5 | 774 | 0.5104 | 0.4154 | 0.6089 | 0.0 | True |
| Llama-3.1-8B-Instruct | vs k=5 | k=1 | k=5 | 774 | 0.1706 | 0.099 | 0.2421 | 0.0001 | True |
| Llama-3.1-8B-Instruct | vs k=5 | k=3 | k=5 | 774 | 0.011 | -0.0429 | 0.0683 | 0.9769 | False |
| Llama-3.1-8B-Instruct | vs k=5 | k=10 | k=5 | 773 | -0.0413 | -0.1008 | 0.0183 | 0.2905 | False |
| Llama-3.1-8B-Instruct | vs k=5 | k=20 | k=5 | 773 | 0.0607 | -0.0044 | 0.1242 | 0.2905 | False |
| Ministral-3-14B-Instruct-2512 | vs k=0 | k=1 | k=0 | 774 | 0.1284 | 0.0538 | 0.2005 | 0.0004 | True |
| Ministral-3-14B-Instruct-2512 | vs k=0 | k=3 | k=0 | 769 | 0.1025 | 0.0417 | 0.1644 | 0.0003 | True |
| Ministral-3-14B-Instruct-2512 | vs k=0 | k=5 | k=0 | 770 | 0.0525 | -0.0133 | 0.1158 | 0.035 | True |
| Ministral-3-14B-Instruct-2512 | vs k=0 | k=10 | k=0 | 765 | 0.0383 | -0.0228 | 0.1013 | 0.0918 | False |
| Ministral-3-14B-Instruct-2512 | vs k=0 | k=20 | k=0 | 596 | 0.0322 | -0.0387 | 0.1052 | 0.4957 | False |
| Ministral-3-14B-Instruct-2512 | vs k=5 | k=0 | k=5 | 770 | -0.0525 | -0.1141 | 0.0093 | 0.035 | True |
| Ministral-3-14B-Instruct-2512 | vs k=5 | k=1 | k=5 | 769 | 0.079 | 0.0223 | 0.1376 | 0.0664 | False |
| Ministral-3-14B-Instruct-2512 | vs k=5 | k=3 | k=5 | 765 | 0.0516 | 0.0061 | 0.0987 | 0.0937 | False |
| Ministral-3-14B-Instruct-2512 | vs k=5 | k=10 | k=5 | 760 | -0.0121 | -0.0442 | 0.0182 | 0.4957 | False |
| Ministral-3-14B-Instruct-2512 | vs k=5 | k=20 | k=5 | 592 | 0.0073 | -0.0397 | 0.0574 | 0.9865 | False |
