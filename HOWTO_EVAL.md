# HOWTO: Run the RAG Evaluation Test Suite

This guide explains how to run the evaluation pipeline that tests different
embedding models, retrieval strategies, and LLMs against the political bias
dataset.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running on the HPC Cluster (SLURM)](#running-on-the-hpc-cluster-slurm)
- [Running Locally (Manual)](#running-locally-manual)
- [Computing Metrics After a Run](#computing-metrics-after-a-run)
- [Log Structure](#log-structure)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The evaluation tests a matrix of configurations:

| Embedding Model | Simple | HyDE | TwoStage | no-RAG |
|-----------------|:------:|:----:|:--------:|:------:|
| e5              | x      | x    | x        | —      |
| bge             | x      | x    | x        | —      |
| jina            | x      | x    | x        | —      |
| qwen3           | x      | x    | x        | —      |
| none            | —      | —    | —        | x      |

Each condition is evaluated against every LLM in `LLM_MODELS` (currently
`x-ai/grok-4.3` and `deepseek/deepseek-v4-flash`).

### Query-side embedding backends

| Model  | Local (GPU)             | OpenRouter (CPU)                |
|--------|-------------------------|---------------------------------|
| e5     | SentenceTransformers    | not available                   |
| bge    | FlagEmbedding           | `baai/bge-m3`                   |
| jina   | SentenceTransformers    | not available                   |
| qwen3  | vLLM server             | `qwen/qwen3-embedding-8b`       |

Using OpenRouter for bge and qwen3 eliminates GPU and vLLM server
requirements — those eval jobs run entirely on CPU.

### Resource summary

| Job               | Partition | GPU  | Dependencies                     |
|-------------------|-----------|------|-----------------------------------|
| `20_eval_norag`   | standard  | none | none                              |
| `21_eval_e5`      | gpu       | 1    | Qdrant                            |
| `22_eval_bge`     | standard  | none | Qdrant + OpenRouter API key       |
| `23_eval_jina`    | gpu       | 1    | Qdrant                            |
| `24_eval_qwen3`   | standard  | none | Qdrant + OpenRouter API key       |

**Peak concurrent GPUs: 1** (only e5 or jina at a time).

---

## Prerequisites

### On the HPC cluster

1. **Ingestion complete.** The four Qdrant collections must be populated.
   See `HOWTO_HPC.md` for the ingestion pipeline.

2. **HPC venv created:**
   ```bash
   bash slurm/setup_venv.sh
   ```

3. **`.env.local`** in the project root with at least:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...
   HF_TOKEN=hf_...
   ```

4. **Qdrant running** (from ingestion, or re-started):
   ```bash
   cat logs/slurm/qdrant_host.txt   # check it's still up
   # If not, the eval scripts auto-start it via qdrant_ensure.sh
   ```

5. **Dataset present** at `src/datasets/political_bias_articles_dataset.csv`.

### Locally

1. Python 3.12+ with `pip install -r requirements.txt`
2. Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
3. Collections populated (run ingestion first, or connect to HPC Qdrant via tunnel)
4. `.env.local` with `OPENROUTER_API_KEY`

---

## Running on the HPC Cluster (SLURM)

### Quick start: submit all jobs at once

All jobs share a `RUN_ID` so logs land in the same directory. Set it once
before submitting:

```bash
cd "$PROJECT_ROOT"
export RUN_ID="eval_$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}"

# CPU jobs — submit immediately, no GPU needed
sbatch slurm/20_eval_norag.sbatch       # no-RAG baseline
sbatch slurm/22_eval_bge.sbatch         # bge via OpenRouter
sbatch slurm/24_eval_qwen3.sbatch       # qwen3 via OpenRouter

# GPU jobs — submit one at a time if GPU quota is tight
sbatch slurm/21_eval_e5.sbatch          # e5 (local, GPU)
# wait for e5 to finish, then:
sbatch slurm/23_eval_jina.sbatch        # jina (local, GPU)
```

If you have enough GPU quota, submit e5 and jina simultaneously — they're
independent.

### Monitoring

```bash
squeue --me                            # see your running jobs
tail -f logs/slurm/eval_bge_*.out      # live log for bge
tail -f logs/slurm/eval_e5_*.out       # live log for e5
```

### Tuning parameters

Override defaults via environment variables before `sbatch`:

```bash
SAMPLE_SIZE=100 K_CHUNKS=10 RANDOM_SEED=42 \
    sbatch slurm/21_eval_e5.sbatch
```

| Variable       | Default | Description                          |
|----------------|---------|--------------------------------------|
| `RUN_ID`       | auto    | Shared run identifier                |
| `RUN_DIR`      | auto    | Shared log directory                 |
| `SAMPLE_SIZE`  | 50      | Number of dataset rows to evaluate   |
| `K_CHUNKS`     | 5       | Top-K chunks to retrieve             |
| `RANDOM_SEED`  | 33      | Random seed for sampling             |
| `HYDE_MODEL`   | Qwen/Qwen2.5-0.5B-Instruct | In-process HyDE LLM      |

### Canceling jobs

```bash
scancel <job_id>
```

### After all jobs complete

```bash
# Compute metrics and generate plots
python -m src.evaluate_metrics --base-dir "$RUN_DIR"

# Results are written to results/
#   - evaluation_metrics.csv
#   - rag_delta.csv
#   - mae_comparison.png, rmse_comparison.png
#   - all_metrics.png, rag_delta_heatmap.png
#   - scatter_predicted_vs_actual.png
```

---

## Running Locally (Manual)

### No-RAG baseline (no Qdrant needed)

```bash
python -m src.run_batch \
    --no_rag \
    --sample_size 50
```

### e5 with all strategies (needs Qdrant + GPU)

```bash
python -m src.run_batch \
    --embedding_model e5 \
    --strategies simple,hyde,twostage \
    --device cuda \
    --k_chunks 5 \
    --sample_size 50
```

### bge via OpenRouter (needs Qdrant, no GPU)

```bash
python -m src.run_batch \
    --embedding_model bge \
    --strategies simple,hyde,twostage \
    --query_backend openrouter \
    --k_chunks 5 \
    --sample_size 50
```

### qwen3 via OpenRouter (needs Qdrant, no GPU, no vLLM)

```bash
python -m src.run_batch \
    --embedding_model qwen3 \
    --strategies simple,hyde,twostage \
    --query_backend openrouter \
    --k_chunks 5 \
    --sample_size 50
```

### qwen3 via local vLLM server (needs Qdrant + vLLM running)

```bash
python -m src.run_batch \
    --embedding_model qwen3 \
    --strategies simple,hyde,twostage \
    --query_backend local \
    --vllm_base_url "http://localhost:8000/v1" \
    --k_chunks 5 \
    --sample_size 50
```

### jina with all strategies (needs Qdrant + GPU)

```bash
python -m src.run_batch \
    --embedding_model jina \
    --strategies simple,hyde,twostage \
    --device cuda \
    --k_chunks 5 \
    --sample_size 50
```

### Using a local vLLM server for bias prediction (HPC without internet)

```bash
python -m src.run_batch \
    --embedding_model bge \
    --strategies simple,hyde,twostage \
    --query_backend openrouter \
    --llm_base_url "http://localhost:8001/v1" \
    --k_chunks 5 \
    --sample_size 50
```

> Note: when using `--llm_base_url` with a vLLM server, the `--embedding_model`
> LLM IDs in `LLM_MODELS` must match model names served by your vLLM instance.
> You may need to edit `LLM_MODELS` in `src/run_batch.py` accordingly.

### Single strategy only

```bash
python -m src.run_batch \
    --embedding_model e5 \
    --strategies simple \
    --device cuda
```

### Sharing a run ID across manual invocations

```bash
RUN_ID="my_manual_run"
RUN_DIR="logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}"

python -m src.run_batch --no_rag --run_id "$RUN_ID" --run_dir "$RUN_DIR"
python -m src.run_batch --embedding_model bge --strategies simple \
    --query_backend openrouter --run_id "$RUN_ID" --run_dir "$RUN_DIR"
```

---

## Computing Metrics After a Run

```bash
# Default: reads from logs/batch_runs/
python -m src.evaluate_metrics

# Specific run directory
python -m src.evaluate_metrics --base-dir "logs/batch_runs/2026-07-15_eval_12345678"
```

Output files (written to `results/`):

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `evaluation_metrics.csv`      | MAE, RMSE, Pearson r, Spearman rho per condition |
| `rag_delta.csv`               | RAG-vs-baseline delta per condition        |
| `mae_comparison.png`          | Grouped bar chart (MAE)                    |
| `rmse_comparison.png`         | Grouped bar chart (RMSE)                   |
| `all_metrics.png`             | 2x2 grid: all four metrics                 |
| `rag_delta_heatmap.png`       | Heatmap of RAG improvement per condition   |
| `scatter_predicted_vs_actual` | Predicted vs ground-truth ideology         |

---

## Log Structure

Each evaluation run produces JSONL logs in this directory layout:

```
logs/batch_runs/<date>_<run_id>/
├── logs_info.md                         # run summary
├── none/
│   └── no_rag/
│       ├── xai_no_rag_evaluation_logs.jsonl
│       └── deepseek_no_rag_evaluation_logs.jsonl
├── e5/
│   ├── simple/
│   │   ├── xai_simple_evaluation_logs.jsonl
│   │   └── deepseek_simple_evaluation_logs.jsonl
│   ├── hyde/
│   │   └── ...
│   └── twostage/
│       └── ...
├── bge/
│   └── ...
├── jina/
│   └── ...
└── qwen3/
    └── ...
```

Each JSONL line is one prediction:

```json
{
  "run_id": "abc12345",
  "timestamp": "2026-07-15T14:30:00.123456",
  "parameters": {
    "llm": "x-ai/grok-4.3",
    "llm_region": "Americas",
    "embedding_model": "e5",
    "retrieval_mode": "simple",
    "hybrid": false,
    "is_rag": true,
    "k_chunks": 5
  },
  "input_metadata": {
    "party": "SPD",
    "speaker": "Karl_Lauterbach",
    "source": "spiegel"
  },
  "inputs": {
    "text": "...",
    "hyde_docs": [],
    "retrieved_chunks": [...]
  },
  "output": {
    "bias": 3.2,
    "justification": "..."
  },
  "ground_truth": {
    "label_ideology": 3.1,
    "label_economic": 2.7,
    "label_galtan": 2.7
  }
}
```

---

## Configuration Reference

### `src/run_batch.py` CLI arguments

| Argument             | Default     | Description                                      |
|----------------------|-------------|--------------------------------------------------|
| `--embedding_model`  | `e5`        | One of: e5, bge, jina, qwen3                     |
| `--strategies`       | `simple`    | Comma-separated: simple, simple_dense, simple_hybrid, hyde, twostage |
| `--no_rag`           | false       | Run no-RAG baseline only                         |
| `--query_backend`    | `local`     | `local` (GPU/vLLM) or `openrouter` (CPU, bge/qwen3 only) |
| `--device`           | auto        | `cuda`, `cpu`, or auto-detect                    |
| `--vllm_base_url`    | from env    | vLLM server URL (qwen3 local backend)            |
| `--llm_base_url`     | OpenRouter  | Base URL for bias prediction LLM                 |
| `--k_chunks`         | 5           | Top-K chunks to retrieve                         |
| `--sample_size`      | 5           | Number of dataset rows                           |
| `--random_seed`      | 33          | Sampling seed                                    |
| `--hyde_model`       | Qwen2.5-0.5B | In-process HyDE LLM model                       |
| `--qdrant_url`       | localhost   | Qdrant server URL                                |
| `--run_id`           | auto        | Shared run ID (for multi-job runs)               |
| `--run_dir`          | auto        | Shared log directory                             |

### Strategy reference

| Strategy         | Description                                              | Hybrid? |
|------------------|----------------------------------------------------------|---------|
| `simple`         | Dense vector retrieval (baseline)                        | No      |
| `simple_dense`   | Same as `simple` (explicit)                              | No      |
| `simple_hybrid`  | Dense + sparse RRF fusion (bge only, needs local backend)| Yes     |
| `hyde`           | HyDE: generate hypothetical docs, average embeddings     | No      |
| `twostage`       | Bi-encoder retrieval + cross-encoder reranking           | No      |

> `simple_hybrid` is only available for bge with `--query_backend local`
> (requires FlagEmbedding for sparse vectors). It is automatically skipped
> when using OpenRouter.

### LLM models

Defined in `src/run_batch.py` → `LLM_MODELS`:

```python
LLM_MODELS = {
    "xai":      {"region": "Americas", "id": "x-ai/grok-4.3"},
    "deepseek": {"region": "China",    "id": "deepseek/deepseek-v4-flash"},
}
```

To test additional LLMs, add entries here. The `id` must be a valid
OpenRouter model slug (or vLLM model name if using `--llm_base_url`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `venv not found at .venv-hpc/bin/python` | Run `bash slurm/setup_venv.sh` on a login node |
| `Could not connect to Qdrant` | Check `logs/slurm/qdrant_host.txt`; the ensure script auto-starts it |
| `OpenRouter API error` | Verify `OPENROUTER_API_KEY` in `.env.local` |
| `Model 'e5' is not available on OpenRouter` | e5 and jina require `--query_backend local` (GPU). Only bge and qwen3 support OpenRouter. |
| `simple_hybrid skipped` | Hybrid requires local FlagEmbedding + GPU. Use `--query_backend local --device cuda`. |
| HyDE generates empty docs | The 0.5B model loads in-process; check `logs/slurm/eval_*_*.err` for load errors |
| Cross-encoder fails to load | `pip install sentence-transformers` in the venv |
| Metrics script finds no logs | Check `--base-dir` path matches your `RUN_DIR` |
| OOM on jina eval | jina-v4 needs ~6GB VRAM; increase `--mem` in the sbatch if running on a shared node |
| `curl: connection refused` to Qdrant | Cluster firewall may block inter-node ports. Run Qdrant and eval on the same node via an interactive allocation. |

### Interactive debugging on HPC

```bash
# Get an interactive GPU allocation
salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=01:00:00

# Run the debug script
.venv-hpc/bin/python src/debug_rag_pipeline.py \
    --mode simple --embedding_model e5 --k 3

# Or run a small batch manually
.venv-hpc/bin/python -m src.run_batch \
    --embedding_model e5 \
    --strategies simple \
    --device cuda \
    --sample_size 5
```
