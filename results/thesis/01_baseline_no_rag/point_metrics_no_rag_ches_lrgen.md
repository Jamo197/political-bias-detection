# Baseline performance (No-RAG) — target `ches_lrgen`

Target: CHES 2024 `lrgen` of the author's party, converted to 1–7 (`1 + lrgen/10*6`, unrounded); CDU/CSU-tagged posts use the CDU value. Prediction: `output.bias`. Signed error = prediction − target (negative = left of target).
Bootstrap: 2000 resamples of posts, seed 42, percentile 95% CIs.

| Model | N_samples | MAE | RMSE | Pearson_r | Spearman_rho | CCC | Mean_signed_error | Calibration_slope | Calibration_intercept |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 775 | 1.804 [1.708, 1.901] | 2.291 [2.192, 2.389] | 0.273 [0.206, 0.342] | 0.148 [0.076, 0.222] | 0.256 [0.192, 0.321] | -0.307 [-0.471, -0.153] | 0.377 [0.286, 0.469] | 1.875 [1.500, 2.232] |
| Llama-3.1-8B-Instruct | 775 | 1.549 [1.452, 1.645] | 2.087 [1.982, 2.192] | 0.462 [0.402, 0.520] | 0.497 [0.431, 0.558] | 0.398 [0.342, 0.452] | 0.884 [0.748, 1.017] | 0.601 [0.526, 0.674] | 2.283 [1.965, 2.583] |
| Meta-Llama-3.1-70B-Instruct-FP8 | 775 | 0.812 [0.758, 0.870] | 1.143 [1.052, 1.238] | 0.704 [0.650, 0.752] | 0.638 [0.584, 0.687] | 0.700 [0.645, 0.747] | 0.084 [0.004, 0.162] | 0.635 [0.577, 0.692] | 1.362 [1.178, 1.547] |
| Ministral-3-3B-Instruct-2512 | 774 | 1.317 [1.230, 1.394] | 1.750 [1.641, 1.853] | 0.483 [0.421, 0.545] | 0.441 [0.379, 0.503] | 0.463 [0.402, 0.526] | 0.431 [0.306, 0.549] | 0.551 [0.481, 0.619] | 2.003 [1.736, 2.266] |
| Ministral-3-8B-Instruct-2512 | 775 | 1.141 [1.076, 1.213] | 1.515 [1.419, 1.613] | 0.608 [0.547, 0.660] | 0.587 [0.529, 0.641] | 0.577 [0.516, 0.630] | 0.506 [0.410, 0.606] | 0.655 [0.584, 0.720] | 1.715 [1.468, 1.977] |
| Ministral-3-14B-Instruct-2512 | 775 | 0.865 [0.803, 0.926] | 1.237 [1.140, 1.336] | 0.671 [0.613, 0.724] | 0.577 [0.517, 0.632] | 0.663 [0.602, 0.716] | -0.219 [-0.304, -0.132] | 0.629 [0.561, 0.692] | 1.082 [0.882, 1.290] |
| Qwen2.5-3B-Instruct | 775 | 1.435 [1.372, 1.498] | 1.695 [1.625, 1.764] | 0.062 [-0.026, 0.146] | -0.003 [-0.081, 0.073] | 0.039 [-0.017, 0.092] | 0.469 [0.362, 0.582] | 0.025 [-0.011, 0.058] | 3.887 [3.780, 4.000] |
| Qwen2.5-7B-Instruct | 774 | 1.202 [1.142, 1.268] | 1.492 [1.411, 1.577] | 0.364 [0.279, 0.443] | 0.235 [0.157, 0.310] | 0.309 [0.235, 0.376] | 0.222 [0.119, 0.325] | 0.209 [0.157, 0.257] | 2.997 [2.838, 3.162] |
| Qwen2.5-32B-Instruct | 775 | 1.159 [1.106, 1.214] | 1.404 [1.331, 1.475] | 0.497 [0.419, 0.565] | 0.526 [0.463, 0.585] | 0.408 [0.343, 0.466] | 0.392 [0.301, 0.484] | 0.284 [0.234, 0.330] | 2.902 [2.752, 3.059] |
| Qwen2.5-72B-Instruct-FP8-dynamic | 775 | 0.941 [0.897, 0.987] | 1.151 [1.093, 1.210] | 0.735 [0.689, 0.775] | 0.644 [0.592, 0.695] | 0.634 [0.587, 0.674] | 0.455 [0.378, 0.528] | 0.479 [0.442, 0.514] | 2.282 [2.159, 2.412] |

## Target per party (1–7)

| Party | Target |
|---|---|
| DIE LINKE | 1.8526 |
| BÜNDNIS 90/DIE GRÜNEN | 2.8333 |
| SPD | 3.0842 |
| FDP | 4.6333 |
| CDU | 4.9474 |
| CDU/CSU | 4.9474 |
| CSU | 5.5429 |
| AfD | 6.5579 |

## Source logs

- Llama-3.1-8B-Instruct: `logs/batch_runs/2026-07-15_eval_20260715_222803/none/no_rag/party_label_llama_no_rag_evaluation_logs.jsonl`
- Llama-3.2-3B-Instruct: `logs/batch_runs/2026-08-10_eval_matrix_20260810_150811/none/no_rag/party_label_llama-3B_no_rag_evaluation_logs.jsonl`
- Meta-Llama-3.1-70B-Instruct-FP8: `logs/batch_runs/2026-08-18_eval_matrix_20260818_170035/none/no_rag/party_label_llama-70B_no_rag_evaluation_logs.jsonl`
- Ministral-3-14B-Instruct-2512: `logs/batch_runs/2026-08-24_eval_matrix_20260824_165851/none/no_rag/party_label_mistral-14B_no_rag_evaluation_logs.jsonl`
- Ministral-3-3B-Instruct-2512: `logs/batch_runs/2026-08-03_eval_matrix_20260803_182344/none/no_rag/party_label_ministral-3b_no_rag_evaluation_logs.jsonl`
- Ministral-3-8B-Instruct-2512: `logs/batch_runs/2026-07-15_eval_20260715_182937/none/no_rag/party_label_ministral_no_rag_evaluation_logs.jsonl`
- Qwen2.5-32B-Instruct: `logs/batch_runs/2026-08-12_eval_matrix_20260812_183445/none/no_rag/party_label_qwen-32B_no_rag_evaluation_logs.jsonl`, `logs/batch_runs/2026-08-12_eval_matrix_20260812_183808/none/no_rag/party_label_qwen-32B_no_rag_evaluation_logs.jsonl`
- Qwen2.5-3B-Instruct: `logs/batch_runs/2026-08-08_eval_matrix_20260808_234122/none/no_rag/party_label_qwen-3B_no_rag_evaluation_logs.jsonl`
- Qwen2.5-72B-Instruct-FP8-dynamic: `logs/batch_runs/2026-08-25_eval_matrix_20260825_092456/none/no_rag/party_label_qwen-72B_no_rag_evaluation_logs.jsonl`
- Qwen2.5-7B-Instruct: `logs/batch_runs/2026-07-15_eval_20260715_211414/none/no_rag/party_label_qwen_no_rag_evaluation_logs.jsonl`
