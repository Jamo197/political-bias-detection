# HOWTO: Cross-cultural RAG dataset (ParlaMint speeches + EU tweets)

Which script do I call to get which result? This guide covers the whole path for
countries other than Germany:

```
 ParlaMint (HuggingFace)   Bundestag corpus           Twitter Parliamentarian DB (local, >1 GB CSVs)
        │                        │                               │
 [A] 30_parlamint_export   [A2] bundestag_to_parlamint_dir  [E] extract_more_tweets.py plan / fetch
        │  speech JSONs 2020–21 (Bundestag schema, DE incl.)     │  EU_tweets_extra.csv
 [B] party_mapping.py build → review → apply              [F] clean_tweets.py → EU_tweets_clean.csv
        │  + ches_party_id on every speech                        │  + country_code, ches_party_id
 [C] 00_chunk.sbatch  (chunk ONCE)                        [G] label_tweets.py → EU_tweets_rq2.csv
 [C2] balance_chunks.py  (same chunk budget per country)         │  CHES 2019 labels, family, party_cue
 [D] 10–13_ingest_*.sbatch (embed 4 models → Qdrant)             │
        ▼                                                        ▼
   chunks_<model>_parlamint  (ALL countries incl. DE) ──► [H] run_rq2.py C0–C5 → [I] evaluate_rq2.py
```

Everything under *Speeches* and *RQ2 runs* runs on the **HPC**. Everything under
*Tweets* runs on your **laptop** (the source CSVs live there).

**RQ2 shared window: 2020-01-01 – 2021-12-31.** The tweet database ends in 2021,
so speeches (ParlaMint and Bundestag) are cut to the same two years. The general cluster setup (images,
`.venv-hpc`, storage rules, Qdrant/vLLM servers) is in [HOWTO_HPC.md](HOWTO_HPC.md);
this file only adds what is specific to the cross-cultural data.

All paths are relative to the repo root (`$PROJECT_ROOT` on the cluster).

---

## 0. One-time prerequisites

1. HPC basics from [HOWTO_HPC.md](HOWTO_HPC.md) steps 0–5 are done (`.venv-hpc`
   created with `bash slurm/setup_venv.sh`, `qdrant.sif`, `vllm.sif`, `.env.local`).
2. The venv needs the `datasets` package for the ParlaMint download. It is now in
   `requirements-hpc.txt`; if the venv already exists:
   ```bash
   .venv-hpc/bin/pip install -r requirements-hpc.txt
   ```
3. `.env.local` contains `HF_TOKEN=...` (ParlaMint3 on HuggingFace is read with it).
4. Files this workflow needs on the cluster (they travel with the normal `rsync` of the repo):
   - `src/datasets/EU_tweets_clean.csv` — cleaned tweets incl. CHES ids
   - `src/datasets/ground_truth/1999-2024_CHES.csv`
   - `src/Cross Cultural Analysis/*.py`

> **Coverage caveat.** ParlaMint3 has *no* Spain (only `es-ct`/`es-ga`), Finland or
> Ireland. Those countries have tweets but cannot get speech data. Available
> configs: `python "src/Cross Cultural Analysis/parlamint_to_speeches.py" --list-configs`.
> The default country list is `at be dk fr gb gr it lv nl pl pt se si`.

---

## Part 1 — Speeches (on the HPC)

### 1.1 Download + extract: `slurm/30_parlamint_export.sbatch`

Downloads ParlaMint3 per country, keeps speeches from `SINCE` to `UNTIL`
(default 2020 – 2021-12-31), drops chair speeches (`Speaker_role=Chairperson`,
procedural only) and writes one JSON per sitting day in the same schema as the
Bundestag data:
`extraction/datasets/parlamint_data/<CC>/TERM_<term>/<YYYY-MM>/speeches/<CC>_<date>_speeches_cleaned.json`.

```bash
cd "$PROJECT_ROOT"; export PROJECT_ROOT="$PWD"

# smoke test first: one country, few rows (a minute or two)
LIMIT=200 COUNTRIES="pl" OUT=extraction/datasets/parlamint_test \
  sbatch slurm/30_parlamint_export.sbatch

# the real run (default 13 countries, 2020–2021)
sbatch slurm/30_parlamint_export.sbatch

# optional coarse pre-cap (seeded sample) so huge parliaments do not dominate chunking
MAX_SPEECHES=20000 COUNTRIES="at pl" sbatch slurm/30_parlamint_export.sbatch

# or a subset / other start year
COUNTRIES="pl gb" SINCE=2018 sbatch slurm/30_parlamint_export.sbatch
```

Each country is a full archive download (several GB, loaded into pandas), hence
64 GB RAM in the job. Check `logs/slurm/parlamint_export_*.out`.

**Compute nodes offline?** Run the identical command on a login node (inside `tmux`/`screen`):

```bash
.venv-hpc/bin/python "src/Cross Cultural Analysis/parlamint_to_speeches.py" \
    --config pl gb --since 2020 --until 2021-12-31 --out extraction/datasets/parlamint_data
# quick test: add  --limit 200
```

Job variables: `COUNTRIES`, `SINCE`, `UNTIL` (`none` = no end), `OUT`, `LIMIT`,
`MAX_SPEECHES`, `MAPPING`. Direct script, all options: `--config`, `--since`,
`--until`, `--limit`, `--max-speeches`, `--seed`, `--out`,
`--mapping [party_mapping.csv]` (adds CHES ids while exporting, see 1.2), `--list-configs`.

Every speech gets: `country`, `country_code`, `party` (ParlaMint code, e.g. `PiS`),
plus after 1.2: `ches_party_id`, `party_canonical`.

### 1.1b Add Germany: `bundestag_to_parlamint_dir.py`

RQ2 keeps all countries in ONE collection per model, so the Bundestag speeches
(not in ParlaMint3) are copied into the same tree as `DE`, cut to the window and
given `country_code="de"`, `party_canonical` and `ches_party_id` (via the
`de,bundestag,…` rows of `party_mapping.csv`):

```bash
.venv-hpc/bin/python "src/Cross Cultural Analysis/bundestag_to_parlamint_dir.py" \
    --out extraction/datasets/parlamint_data      # → parlamint_data/DE/WP_xx/...
```

About 12k speeches for 2020–21; `Fraktionslos` stays without a CHES id.

### 1.2 Attach CHES ids to the speeches: `party_mapping.py`

ParlaMint knows parties only by its own codes, so a reviewable table maps them to
CHES ids. Tweet parties are pre-filled from the tweet dataset (their ids are
already CHES ids); the ParlaMint codes have to be checked by you.

```bash
PY=.venv-hpc/bin/python
MAP="src/Cross Cultural Analysis/party_mapping.py"

# 1. list every ParlaMint party code found in the export; fuzzy-suggest CHES ids
$PY "$MAP" build  --speeches-dir extraction/datasets/parlamint_data

# 2. see what is still unmapped (speeches per country with / without CHES id)
$PY "$MAP" coverage --speeches-dir extraction/datasets/parlamint_data

# 3. EDIT "src/Cross Cultural Analysis/party_mapping.csv" (see below), then:
$PY "$MAP" apply  --speeches-dir extraction/datasets/parlamint_data
$PY "$MAP" coverage --speeches-dir extraction/datasets/parlamint_data   # aim: all big parties mapped
```

How to edit `party_mapping.csv` (columns `country_code, source, raw_label,
canonical_party, ches_party_id, status`):

| `status` | meaning | what you do |
|---|---|---|
| `auto` | tweet label, CHES id from the dataset | nothing |
| `seed` / `suggested` | my guess / fuzzy match for a ParlaMint code | **verify**, then set `status` to `manual` |
| `unmapped` | no match | look the party up in `src/datasets/ground_truth/1999-2024_CHES.csv` (`party_id`, `party`), fill `ches_party_id`, set `status` to `manual` |
| `manual` | your decision | never overwritten by `build` |

Rows with an empty `ches_party_id` are ignored (those speeches get
`party_canonical="UNKNOWN"`, `ches_party_id=null`). Independents/small parties not
in CHES stay unmapped on purpose. `apply` rewrites the JSONs in place and can be
repeated after every mapping edit — no re-download needed. (Alternative: pass
`--mapping` to the export job to do it in one step, but then every mapping change
means a re-download.)

Optional sanity check of all CHES ids in the tweets and the mapping:
`$PY "$MAP" validate` (flags ids missing in CHES, wrong country, parties that
ended before 2019, and several different parties sharing one id).

### 1.3 Chunk once: `slurm/00_chunk.sbatch`

Same script as for Germany, pointed at the ParlaMint folder. Artifacts go to their
own directory so the German `chunks.jsonl` is not overwritten:

```bash
DATA_DIR=extraction/datasets/parlamint_data \
ARTIFACT_DIR=rag/ingest/artifacts/parlamint \
  sbatch slurm/00_chunk.sbatch

ls -lh rag/ingest/artifacts/parlamint/      # chunks.jsonl + full_speeches.jsonl
```

Then give every country the same chunk budget (whole speeches, seeded sample;
default budget = the smallest country):

```bash
.venv-hpc/bin/python "src/Cross Cultural Analysis/balance_chunks.py" \
    --artifact-dir rag/ingest/artifacts/parlamint   # [--budget N] [--require-ches]
# → chunks_balanced.jsonl + full_speeches_balanced.jsonl, prints chunks per country
```

### 1.4 Embed + save to Qdrant: `slurm/10–13_ingest_*.sbatch`

`COLLECTION_SUFFIX=_parlamint` (new, read in `rag/ingest/config.py`) appends a
suffix to **every** collection name, so you get `chunks_e5_parlamint`,
`chunks_bge_parlamint`, `chunks_jina_parlamint`, `chunks_qwen3_parlamint` and
`bundestag_speeches_parlamint` — and the `--reset` in the ingest scripts cannot
touch the German collections.

Export both variables in the shell once, then submit exactly as in HOWTO_HPC §6
(`sbatch` forwards the environment):

```bash
export COLLECTION_SUFFIX=_parlamint
export ARTIFACT_DIR=rag/ingest/artifacts/parlamint
export CHUNKS_FILE=$ARTIFACT_DIR/chunks_balanced.jsonl        # from 1.3
export SPEECHES_FILE=$ARTIFACT_DIR/full_speeches_balanced.jsonl

# Qdrant is auto-started by the ingest jobs (slurm/qdrant_ensure.sh) if not running.
sbatch slurm/10_ingest_e5.sbatch      # uploads parents + chunks_e5_parlamint (run first)
sbatch slurm/11_ingest_bge.sbatch     # dense + sparse
sbatch slurm/12_ingest_jina.sbatch

# Qwen3 needs the vLLM embedding server first:
sbatch slurm/02_vllm_qwen3.sbatch
sbatch slurm/13_ingest_qwen3.sbatch   # waits for /health; scancel the 02 job afterwards
```

New payload fields available for filtering in every collection:
`country`, `country_code`, `party_canonical`, `ches_party_id` (indexed). The
indexes only exist for *newly created* collections, which is the case here.

Verify (collections and point counts; all four chunk collections must match):

```bash
QURL="http://$(cat logs/qdrant_host.txt)"
curl -s "$QURL/collections" | python3 -m json.tool
for c in chunks_e5_parlamint chunks_bge_parlamint chunks_jina_parlamint chunks_qwen3_parlamint bundestag_speeches_parlamint; do
  echo -n "$c: "; curl -s "$QURL/collections/$c" | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"
done
```

(`slurm/01_qdrant.sbatch` writes the host to `logs/qdrant_host.txt`; HOWTO_HPC
still mentions `logs/slurm/qdrant_host.txt`.)

Rebuilding one model: re-submit its ingest job (it resets only its own
`…_parlamint` collection). Re-chunking: re-run 1.3, then all ingest jobs.

### 1.5 Using the collections in the evaluation

`src/run_rq2.py` (Part 3) addresses `chunks_<model>_parlamint` directly
(`--collection_suffix`, default `_parlamint`), so `COLLECTION_SUFFIX` is not
needed for evaluation. `PoliticalRAGRetriever(country_code="pl")` restricts every
retrieval strategy to one country (payload filter on `country_code`, also inside
the bge hybrid prefetch); `country_code=None` searches all countries.

---

## Part 2 — Tweets (on your laptop)

Source data: `…/Twitter Parliamentarian Database/ches_integrated_tweets_members_parties.csv`
(~2 GB, 9.5M tweets with CHES ids; built in `src/datasets/extract_twitter_data.ipynb`).
It is never loaded whole: scripts read it in chunks (~1 min per pass, small RAM).
Run from the repo root with the project venv; fetching needs `beautifulsoup4`
(`pip install beautifulsoup4`).

```bash
cd "src/Cross Cultural Analysis"      # scripts import each other: run from here or with the full path
```

### 2.1 What you have

| File | Content |
|---|---|
| `src/datasets/EU_tweets_with_parties_dataset.csv` | raw sample (hand-corrected CHES ids) |
| `src/datasets/EU_tweets_clean.csv` | + `country_code`, resolved country, `canonical_party` (CHES abbreviation) |

Rebuild the clean file after any change to the raw one:

```bash
python clean_tweets.py
```

It resolves region / "European Parliament" countries (via the CHES id prefix),
adds `country_code` and `canonical_party`, and prints countries per row,
conflicts and unresolved rows.

### 2.2 Extract more tweets: `extract_more_tweets.py`

Three steps; nothing is sent to Twitter before step 3.

```bash
# (a) one-off: turn your manual CHES fixes into rules (ches_overrides.csv)
python extract_more_tweets.py learn-overrides

# (b) plan: top every (country, party) up to N tweets; writes
#     src/datasets/EU_tweets_extra_plan.csv  and  party_review.csv
python extract_more_tweets.py plan            # default: 75 per party, incl. de
#   options: --per-party 75  --min-available 50  --countries at de ...
#            --oversample 1.6  --seed 42  --source <csv>

# (c) REVIEW party_review.csv: ches_abbr must match the party, "id_matches_country"
#     must be True, check parties with ches_last_year < 2019.
#     Wrong id? add a row to ches_overrides.csv (country_code, party, party_official,
#     ches_party_id) and re-run (b).

# (d) fetch the texts (resumable, ~1 request/s, safe to Ctrl-C and re-run)
python extract_more_tweets.py fetch            # --limit 100 for a test run
```

`plan` already counts what `EU_tweets_clean.csv` holds, skips known tweet ids and
oversamples because deleted / URL-only / <25 character tweets are dropped.
`fetch` writes `src/datasets/EU_tweets_extra.csv` (same columns as the raw dataset,
country already resolved) and stops fetching a party once it has enough valid tweets.

### 2.3 Merge and re-clean

```bash
# append the new tweets to the raw dataset (keep the header only once)
tail -n +2 ../datasets/EU_tweets_extra.csv >> ../datasets/EU_tweets_with_parties_dataset.csv
python clean_tweets.py
python party_mapping.py validate          # every CHES id against the CHES file
python party_mapping.py build             # refresh tweet rows in party_mapping.csv (manual rows kept)
```

### 2.4 Labels for RQ2: `label_tweets.py`

```bash
python label_tweets.py                       # → src/datasets/EU_tweets_rq2.csv
#   options: --countries at de  --max-per-party 75  --min-party-tweets 20  --seed 42
```

Adds per tweet: `label_lrgen`, `label_lrecon`, `label_galtan` (CHES wave nearest
to the tweet year, i.e. 2019, rescaled `1 + 0.6·x` from 0–10 to 1–7),
`ches_wave`, `ches_family` and `party_cue` (tweet names a party of its country:
names/abbreviations from `party_mapping.csv` and CHES 2019+, plus party handles /
hashtags in `EXTRA_CUES`; a heuristic, extend `EXTRA_CUES` for new countries).
Germany uses these CHES labels too (not the RQ1 `party_label_*` table, which
comes from a different 1–10 source).

Then re-upload `EU_tweets_rq2.csv` and `party_mapping.csv` to the cluster.

---

## Part 3 — RQ2 runs (on the HPC)

### 3.1 Conditions: `src/run_rq2.py` / `slurm/40_eval_rq2.sbatch`

| ID | Retrieval (filter on `country_code`) | Country in prompt |
|---|---|---|
| C0 | none | no (byte-identical RQ1 no-RAG prompt) |
| C1 | none | country + CHES LRGEN definition |
| C2 | target | yes |
| C3 | donor (`de`; `at` for Germany) | yes |
| C4 | none (pooled) | yes |
| C5 | target, `chunks_<model>_parlamint_en` + `tweet_text_en` | yes |

The model still returns one `bias_score` (as in RQ1); it is compared with lrgen
(main), lrecon and galtan. HyDE writes its hypothetical speeches "from" the
retrieval-source country.

```bash
# generator server first, e.g.
sbatch slurm/vllm_server.sh Qwen/Qwen2.5-32B-Instruct

# Austria pilot, one generator, the two RQ1 configs
COUNTRY=at CONDITIONS=C0,C1,C2,C3,C4 EMB=bge STRATEGIES=simple_hybrid,twostage \
  LLM=qwen-32B sbatch slurm/40_eval_rq2.sbatch
# smoke test: add SAMPLE_SIZE=10
```

Logs: `logs/rq2_runs/<date>_<RUN_ID>/<emb>/<strategy|no_rag>/<llm>_<cc>_<Cx>_<strategy>.jsonl`
with an `rq2` block (condition, target/retrieval country, CHES id, family,
party_cue). Re-submitting with the same `RUN_ID`/`RUN_DIR` resumes.
Generators: `python -m src.run_rq2 --help` lists the keys (`qwen-32B`, `llama-8B`,
`ministral-14B`, …).

### 3.2 English condition C5: `translate_corpus.py`

```bash
TR="src/Cross Cultural Analysis/translate_corpus.py"
VLLM="http://$(cat logs/slurm/vllm_active_host.txt)/v1"
.venv-hpc/bin/python "$TR" chunks --input rag/ingest/artifacts/parlamint/chunks_balanced.jsonl \
    --base-url "$VLLM" --model Qwen/Qwen2.5-32B-Instruct
.venv-hpc/bin/python "$TR" tweets --base-url "$VLLM" --model Qwen/Qwen2.5-32B-Instruct  # adds tweet_text_en
# ingest the English chunks into their own collections (repeat for 11–13)
COLLECTION_SUFFIX=_parlamint_en \
CHUNKS_FILE=rag/ingest/artifacts/parlamint/chunks_en.jsonl \
SPEECHES_FILE=rag/ingest/artifacts/parlamint/full_speeches_balanced.jsonl \
  sbatch slurm/10_ingest_e5.sbatch
```

Both are resumable; failed translations are retried on the next run.

### 3.3 Analysis: `evaluate_rq2.py`

```bash
python "src/Cross Cultural Analysis/evaluate_rq2.py" logs/rq2_runs/<run> [<run> ...] \
    --out results/rq2 --target label_ideology
```

Writes `metrics.csv` (MAE, RMSE, directional accuracy, within-country Spearman of
party means; all / cue / no_cue), `contrasts.csv` (paired abs-error differences
C2−C3, C1−C0, C2−C1, C5−C2, C2−C0 with a party-clustered bootstrap CI, per
country and pooled), `c4_country_share.csv`, `family_correlation.csv`,
`mixed_model.csv` (needs `statsmodels`) and `summary.md`.

> **Known weak spot of the source CSV.** The notebook matches parties by substring
> (e.g. "SD" is found inside "SDS"), which assigns wrong CHES ids to some parties
> (Slovenian Social Democrats → SDS, Danish Alternative → Liberal Alliance, the
> Greek Solution → ANEL, …). That is what `party_review.csv`, `validate` and
> `ches_overrides.csv` are for — look at them before trusting new tweets.

---

## Command cheat sheet

| Goal | Command |
|---|---|
| download + extract ParlaMint | `sbatch slurm/30_parlamint_export.sbatch` |
| list / map ParlaMint parties to CHES | `party_mapping.py build`, edit CSV, `party_mapping.py apply` |
| check mapping coverage | `party_mapping.py coverage` |
| chunk (once) | `DATA_DIR=… ARTIFACT_DIR=… sbatch slurm/00_chunk.sbatch` |
| embed 4 models → Qdrant | `COLLECTION_SUFFIX=_parlamint ARTIFACT_DIR=… sbatch slurm/10…13_ingest_*.sbatch` |
| clean tweet countries / CHES | `clean_tweets.py` |
| more tweets | `extract_more_tweets.py learn-overrides` → `plan` → review → `fetch` |
| validate all CHES ids | `party_mapping.py validate` |
| add Germany to the export tree | `bundestag_to_parlamint_dir.py` |
| equal chunks per country | `balance_chunks.py --artifact-dir …` |
| RQ2 tweet labels + party_cue | `label_tweets.py` |
| run C0–C5 | `COUNTRY=at … sbatch slurm/40_eval_rq2.sbatch` |
| translate for C5 | `translate_corpus.py chunks / tweets` |
| RQ2 metrics + bootstrap | `evaluate_rq2.py logs/rq2_runs/<run>` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Unknown configuration(s)` | Country not in ParlaMint3 (no `es`, `fi`, `ie`). Use `--list-configs`. |
| Export job: connection errors | Compute nodes offline → run the python command on a login node. |
| Export killed (OOM) | Raise `--mem` in `30_parlamint_export.sbatch` or export fewer countries per job. |
| `ModuleNotFoundError: datasets` | `.venv-hpc/bin/pip install -r requirements-hpc.txt`. |
| Many speeches `UNKNOWN` / no CHES id | `party_mapping.py coverage` lists the unmapped codes; fill them, then `apply`. |
| German collections changed | `COLLECTION_SUFFIX` was not exported in the shell that ran `sbatch`. |
| `fetch`: `No module named bs4` | `pip install beautifulsoup4`. |
