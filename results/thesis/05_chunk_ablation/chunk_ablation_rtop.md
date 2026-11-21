# R_top >= 4 chunk ablation (target: CHES lrgen)

All conditions are restricted to the posts for which at least one logged e5/simple chunk was rated R_top >= 4 by the LLM judge. Only those chunks are shown to the model (variable k, 1-5).

## Point metrics on these posts

| Model | Condition | N_samples | MAE | RMSE | Pearson_r | Mean_signed_error | Calibration_slope |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | no_rag | 65 | 1.3556 | 1.9069 | 0.5096 | 0.5673 | 0.6132 |
| Llama-3.1-8B-Instruct | e5/rtop4plus | 65 | 1.2283 | 1.8303 | 0.4746 | -0.3004 | 0.5354 |
| Llama-3.1-8B-Instruct | e5/simple | 65 | 1.2169 | 1.7943 | 0.4719 | -0.4973 | 0.4889 |
| Llama-3.1-8B-Instruct | e5pool/random5 | 65 | 1.0368 | 1.5006 | 0.6051 | -0.305 | 0.6096 |
| Ministral-3-8B-Instruct-2512 | no_rag | 67 | 1.1599 | 1.734 | 0.4203 | 0.5341 | 0.4295 |
| Ministral-3-8B-Instruct-2512 | e5/rtop4plus | 67 | 1.3332 | 1.9356 | 0.4016 | 0.2042 | 0.5148 |
| Ministral-3-8B-Instruct-2512 | e5/simple | 67 | 1.2259 | 1.7472 | 0.5639 | 0.558 | 0.7213 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | 67 | 1.3514 | 1.9328 | 0.4818 | 0.8445 | 0.589 |

## Paired deltas (first condition minus reference)

| Model | Condition | Reference | N_paired | ΔMAE (RAG-Base) | ΔMAE (RAG-Base)_lo | ΔMAE (RAG-Base)_hi | ΔMAE_p_holm | ΔPearson_r (RAG-Base) | Δ|Bias| (RAG-Base) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | e5/rtop4plus | no_rag | 65 | -0.1272 | -0.538 | 0.3001 | 0.5297 | -0.0349 | -0.267 |
| Ministral-3-8B-Instruct-2512 | e5/rtop4plus | no_rag | 67 | 0.1733 | -0.2203 | 0.565 | 0.3618 | -0.0187 | -0.3299 |
| Llama-3.1-8B-Instruct | e5/rtop4plus | e5/simple | 65 | 0.0114 | -0.2661 | 0.2596 | 0.9295 | 0.0027 | -0.1969 |
| Ministral-3-8B-Instruct-2512 | e5/rtop4plus | e5/simple | 67 | 0.1073 | -0.1524 | 0.3881 | 0.4338 | -0.1622 | -0.3537 |
| Llama-3.1-8B-Instruct | e5/rtop4plus | e5pool/random5 | 65 | 0.1916 | -0.1293 | 0.5431 | 0.2489 | -0.1304 | -0.0046 |
| Ministral-3-8B-Instruct-2512 | e5/rtop4plus | e5pool/random5 | 67 | -0.0182 | -0.386 | 0.3835 | 0.9335 | -0.0802 | -0.6403 |
| Llama-3.1-8B-Instruct | e5pool/random5 | no_rag | 65 | -0.3188 | -0.677 | 0.0374 | 0.094 | 0.0955 | -0.2624 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | no_rag | 67 | 0.1915 | -0.1313 | 0.5075 | 0.2459 | 0.0615 | 0.3104 |
| Llama-3.1-8B-Instruct | e5pool/random5 | e5/simple | 65 | -0.1801 | -0.478 | 0.116 | 0.2409 | 0.1332 | -0.1923 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | e5/simple | 67 | 0.1255 | -0.2046 | 0.4439 | 0.4518 | -0.082 | 0.2866 |

## Quadrant of the posts

| model | quadrant | n_posts |
|---|---|---|
| Llama-3.1-8B-Instruct | baseline sufficiency | 19 |
| Llama-3.1-8B-Instruct | RAG win | 17 |
| Llama-3.1-8B-Instruct | joint failure | 15 |
| Llama-3.1-8B-Instruct | RAG distraction | 14 |
| Ministral-3-8B-Instruct-2512 | RAG distraction | 19 |
| Ministral-3-8B-Instruct-2512 | RAG win | 18 |
| Ministral-3-8B-Instruct-2512 | baseline sufficiency | 16 |
| Ministral-3-8B-Instruct-2512 | joint failure | 14 |

## Chunks per post (R_top condition)

| model | condition | count | mean | min | max |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | e5/rtop4plus | 65 | 3.246153846153846 | 1 | 5 |
| Ministral-3-8B-Instruct-2512 | e5/rtop4plus | 67 | 2.970149253731343 | 1 | 5 |
