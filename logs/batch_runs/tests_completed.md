# Tested Models

## Models
| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Questions | Status | Folder | Ground Truth |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Qwen/Qwen2.5-7B-Instruct | China | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_211414 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | hyde | 513 | Partial (stopped at 513/775) | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | hyde_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | simple_hybrid | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | twostage_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | hyde | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | twostage | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | hyde | 262 | Partial (stopped at 262/775) | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_222803 | final_label_ideology |
| meta-llama/llama-3.1-70b-instruct | Americas | No | None | no_rag | 775 | Complete | 2026-07-24_eval_20260724_192028 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | None | no_rag | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | hyde | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | hyde_hybrid | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | simple_hybrid | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | twostage | 580 | Partial (stopped at 580/775) | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | e5 | hyde | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | e5 | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | e5 | twostage | 775 | Complete | 2026-08-03_eval_matrix_20260803_182344 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_182937 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | e5 | hyde | 775 | Complete | 2026-07-22_eval_20260722_233914 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | e5 | simple | 775 | Complete | 2026-07-22_eval_20260722_233914 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | e5 | twostage | 775 | Complete | 2026-07-22_eval_20260722_233914 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | jina | hyde | 775 | Complete | 2026-07-23_eval_20260723_115807 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | jina | simple | 775 | Complete | 2026-07-23_eval_20260723_115807 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | jina | twostage | 775 | Complete | 2026-07-23_eval_20260723_115807 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | qwen3 | hyde | 775 | Complete | 2026-07-23_eval_20260723_143404 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | qwen3 | simple | 775 | Complete | 2026-07-23_eval_20260723_143404 | final_label_ideology |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | qwen3 | twostage | 655 | Partial (stopped at 655/775) | 2026-07-23_eval_20260723_143404 | final_label_ideology |
| mistralai/mistral-large-2512 | Europe | No | None | no_rag | 775 | Complete | 2026-07-24_eval_20260724_192028 | final_label_ideology |
| qwen/qwen-2.5-72b-instruct | China | No | None | no_rag | 775 | Complete | 2026-07-25_eval_20260725_133532 | final_label_ideology |


## Remaining Tests
| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Hosting |
|----------|--------|--------|--------|--------|--------|
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | bge | simple, hyde, twostage | 775 |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | simple, hyde, twostage | Local |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | qwen3 | simple, hyde, twostage | Local |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | simple, hyde, twostage | Local |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | e5 | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | jina | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | qwen3 | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | bge | simple, hyde, twostage | OpenRouter |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | jina | simple, hyde, twostage | local |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | local |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | simple, hyde, twostage | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | No | None | no_rag | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | e5 | simple, hyde, twostage | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | jina | simple, hyde, twostage | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | simple, hyde, twostage | local |
| mistralai/mistral-large-2512 | Europe | Yes | e5 | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | jina | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | bge | simple, hyde, twostage | OpenRouter |
| Qwen/Qwen2.5-3B-Instruct | China | No | None | no_rag | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | e5 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | jina | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | qwen3 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | bge | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | e5 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | qwen3 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-32B-Instruct | China | No | None | no_rag | OpenRouter or Local |
| Qwen/Qwen2.5-32B-Instruct | China | Yes | e5 | simple, hyde, twostage | OpenRouter or Local |
| Qwen/Qwen2.5-32B-Instruct | China | Yes | jina | simple, hyde, twostage | OpenRouter or Local |
| Qwen/Qwen2.5-32B-Instruct | China | Yes | qwen3 | simple, hyde, twostage | OpenRouter or Local |
| Qwen/Qwen2.5-32B-Instruct | China | Yes | bge | simple, hyde, twostage | OpenRouter or Local |
| Qwen/Qwen2.5-72B-Instruct | China | No | None | no_rag | OpenRouter |
| Qwen/Qwen2.5-72B-Instruct | China | Yes | e5 | simple, hyde, twostage | OpenRouter |
| Qwen/Qwen2.5-72B-Instruct | China | Yes | jina | simple, hyde, twostage | OpenRouter |
| Qwen/Qwen2.5-72B-Instruct | China | Yes | qwen3 | simple, hyde, twostage | OpenRouter |
| Qwen/Qwen2.5-72B-Instruct | China | Yes | bge | simple, hyde, twostage | OpenRouter |