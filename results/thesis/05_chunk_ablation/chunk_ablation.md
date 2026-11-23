# Chunk ablation (target: CHES lrgen)

## Point metrics

| Model | Condition | N_samples | MAE | RMSE | Pearson_r | Mean_signed_error | Calibration_slope |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | no_rag | 775 | 1.5488 | 2.0865 | 0.4615 | 0.884 | 0.6006 |
| Llama-3.1-8B-Instruct | e5/simple | 774 | 1.0395 | 1.4608 | 0.5733 | 0.2241 | 0.5847 |
| Llama-3.1-8B-Instruct | e5pool/random5 | 775 | 1.0214 | 1.4284 | 0.5776 | 0.2343 | 0.5671 |
| Ministral-3-8B-Instruct-2512 | no_rag | 775 | 1.1414 | 1.5152 | 0.6075 | 0.506 | 0.6549 |
| Ministral-3-8B-Instruct-2512 | e5/simple | 760 | 1.2149 | 1.7046 | 0.5961 | 0.4709 | 0.7627 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | 774 | 1.3151 | 1.8144 | 0.5752 | 0.7566 | 0.7241 |

## Paired deltas (ablation minus reference)

| Model | Condition | Reference | N_paired | ΔMAE (RAG-Base) | ΔMAE (RAG-Base)_lo | ΔMAE (RAG-Base)_hi | ΔMAE_p_holm | ΔPearson_r (RAG-Base) | Δ|Bias| (RAG-Base) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | e5pool/random5 | no_rag | 775 | -0.5274 | -0.6251 | -0.4294 | 0.001 | 0.1161 | -0.6497 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | no_rag | 774 | 0.1733 | 0.1011 | 0.2409 | 0.001 | -0.0317 | 0.251 |
| Llama-3.1-8B-Instruct | e5pool/random5 | e5/simple | 774 | -0.0177 | -0.0893 | 0.0554 | 0.6667 | 0.0039 | 0.0097 |
| Ministral-3-8B-Instruct-2512 | e5pool/random5 | e5/simple | 759 | 0.1021 | 0.0181 | 0.1887 | 0.013 | -0.0137 | 0.302 |
