# Tested Models

## Models
| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Questions | Status | Folder | Ground Truth |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Qwen/Qwen2.5-7B-Instruct | China | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_211414 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | hyde | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | hyde_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | simple_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | bge | twostage_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | e5 | hyde | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | e5 | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | e5 | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | hyde | 42 | Partial (42/775 - needs redo, < 500) | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | qwen3 | hyde | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | qwen3 | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | qwen3 | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_191413 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | hyde | 513 | Partial (stopped at 513/775) | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | hyde_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | simple_hybrid | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | twostage | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | bge | twostage_hybrid | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | hyde | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | simple | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | e5 | twostage | 775 | Complete | 2026-08-03_eval_matrix_20260803_224143 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | hyde | 248 | Partial (248/775 - needs redo, < 500) | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | simple | 775 | Complete | 2026-08-04_eval_matrix_20260804_082443 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | hyde | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | hyde_hybrid | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | simple | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | simple_hybrid | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | twostage | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | bge | twostage_hybrid | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | jina | hyde | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | jina | twostage | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | qwen3 | hyde | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | qwen3 | simple | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | China | Yes | qwen3 | twostage | 775 | Complete | 2026-08-06_eval_matrix_20260806_091309 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | USA | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_222803 | final_label_ideology |
| meta-llama/Llama-3.1-8B-Instruct | USA | No | None | no_rag | 775 | Complete | 2026-07-15_eval_20260715_222803 | final_label_ideology |
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
| mistralai/mistral-large-2512 | Europe | No | None | no_rag | 775 | Complete | 2026-07-24_eval_20260724_192028 | final_label_ideology |
| qwen/qwen-2.5-72b-instruct | China | No | None | no_rag | 440 | Partial (440/775 - needs redo, < 500) | 2026-07-25_eval_20260725_133532 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | No | None | no_rag | 775 | Complete | 2026-08-06_eval_matrix_20260806_210845 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | e5 | simple | 770 | Complete | 2026-08-06_eval_matrix_20260806_210845 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | e5 | hyde | 764 | Complete | 2026-08-06_eval_matrix_20260806_210845 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | simple | 772 | Complete | 2026-08-06_eval_matrix_20260806_211059 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | hyde | 773 | Complete | 2026-08-06_eval_matrix_20260806_211059 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | twostage | 770 | Complete | 2026-08-06_eval_matrix_20260806_211059 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | simple_hybrid | 770 | Complete | 2026-08-06_eval_matrix_20260806_211059 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | hyde_hybrid | 771 | Complete | 2026-08-06_eval_matrix_20260806_211059 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | bge | twostage_hybrid | 770 | Complete | 2026-08-07_eval_matrix_20260807_123902 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | jina | simple | 771 | Complete | 2026-08-07_eval_matrix_20260807_123902 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | qwen3 | simple | 773 | Complete | 2026-08-07_eval_matrix_20260807_094228 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | qwen3 | hyde | 770 | Complete | 2026-08-07_eval_matrix_20260807_094228 | final_label_ideology |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | qwen3 | twostage | 772 | Complete | 2026-08-07_eval_matrix_20260807_094228 | final_label_ideology |


## Redo Needed (incomplete runs with < 500 samples)
| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Reason |
|----------|--------|--------|--------|--------|--------|
| meta-llama/Llama-3.1-8B-Instruct | Americas | Yes | jina | hyde | Only 248/775 completed |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | hyde | Only 42/775 completed |
| qwen/qwen-2.5-72b-instruct | China | No | None | no_rag | Only 440/775 completed |

## Remaining Tests
| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Hosting |
|----------|--------|--------|--------|--------|--------|
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | bge | simple, hyde, twostage | 775 |
| mistralai/Ministral-3-8B-Instruct-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | local |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | e5 | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | jina | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | qwen3 | simple, hyde, twostage | OpenRouter |
| meta-llama/llama-3.1-70b-instruct | Americas | Yes | bge | simple, hyde, twostage | OpenRouter |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | jina | simple, hyde, twostage | local |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | local |
| mistralai/Ministral-3-3B-Instruct-2512 | Europe | Yes | bge | simple, hyde, twostage | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | e5 | twostage (redo eval) | local |
| mistralai/Ministral-3-14B-Instruct-2512 | Europe | Yes | jina | hyde, twostage | local |
| mistralai/mistral-large-2512 | Europe | Yes | e5 | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | jina | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | qwen3 | simple, hyde, twostage | OpenRouter |
| mistralai/mistral-large-2512 | Europe | Yes | bge | simple, hyde, twostage | OpenRouter |
| Qwen/Qwen2.5-3B-Instruct | China | No | None | no_rag | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | e5 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | jina | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | qwen3 | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-3B-Instruct | China | Yes | bge | simple, hyde, twostage | Local |
| Qwen/Qwen2.5-7B-Instruct | China | Yes | jina | simple, hyde, twostage | Local |
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


### Missing qwen32b tests
```jsonl
{"llm": "Qwen/Qwen2.5-32B-Instruct", "llm_region": "China", "embedding_model": "bge", "retrieval_mode": "hyde", "hybrid": false, "is_rag": true, "k_chunks": 5}
{"llm": "Qwen/Qwen2.5-32B-Instruct", "llm_region": "China", "embedding_model": "bge", "retrieval_mode": "hyde_hybrid", "hybrid": true, "is_rag": true, "k_chunks": 5}
{"llm": "Qwen/Qwen2.5-32B-Instruct", "llm_region": "China", "embedding_model": "bge", "retrieval_mode": "twostage", "hybrid": false, "is_rag": true, "k_chunks": 5}
{"llm": "Qwen/Qwen2.5-32B-Instruct", "llm_region": "China", "embedding_model": "bge", "retrieval_mode": "simple_hybrid", "hybrid": true, "is_rag": true, "k_chunks": 5}
```
```python
text_indices = [10,102,109,110,111,119,121,132,133,134,136,137,138,140,141,146,149,156,159,165,166,169,175,183,193,194,197,199,2,200,205,209,210,211,212,213,214,216,219,221,224,23,232,236,24,240,245,248,251,255,260,261,265,266,276,28,282,286,287,29,291,292,293,295,297,30,300,301,303,307,31,312,315,317,320,324,327,328,329,33,330,332,333,334,335,336,337,339,343,345,351,352,353,356,357,358,362,363,364,366,369,376,381,382,383,389,39,394,395,396,399,405,406,41,410,421,423,426,427,429,432,434,435,437,440,441,444,445,446,448,45,451,454,457,458,463,465,478,483,487,488,494,495,496,498,50,500,502,507,515,518,519,52,522,523,528,530,532,533,534,535,537,538,539,542,544,545,546,548,55,550,553,558,56,568,569,57,571,577,579,582,583,584,590,598,6,600,603,605,607,609,61,610,614,617,62,621,622,625,634,635,637,64,651,653,658,66,660,663,666,668,67,671,674,678,68,680,683,689,692,7,70,706,708,71,712,713,718,722,73,731,732,734,739,740,741,744,745,749,751,752,754,757,758,759,765,769,77,771,772,775,776,777,778,78,782,783,785,786,80,800,801,804,807,808,816,817,819,82,83,84,85,87,91,97,98]
```