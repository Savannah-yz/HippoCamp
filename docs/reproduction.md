# Reproduction Notes

This document collects the longer command sequences that support the public release README.

## Shared setup

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the benchmark subsystem:

```bash
pip install -e ./benchmark --no-deps
```

If you see matplotlib or fontconfig cache warnings, set:

```bash
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

## Data release map

The authoritative data release lives on Hugging Face:

- <https://huggingface.co/datasets/MMMem-org/HippoCamp>

Use the released data pieces as follows:

- The six source directories under `HippoCamp/Adam/{Subset,Fullset}/...`, `HippoCamp/Bei/{Subset,Fullset}/...`, and `HippoCamp/Victoria/{Subset,Fullset}/...` store the raw personal-computing-environment files.
- The six annotation JSON files such as `Adam.json`, `Adam_Subset.json`, `Victoria.json`, and `Victoria_Subset.json` store the released QA pairs and explicit annotations.
- `HippoCamp_Gold` stores parsed text JSON files with the schema `{file_info, summary, segments}`.
- The `*_files.xlsx` spreadsheets store creation time, modification time, and location-oriented metadata. The Hugging Face release also includes `HippoCamp/update_metadata_from_xlsx.py`.

For local use in this repository:

- Place the parsed text release under `benchmark/HippoCamp_Gold/`.
- Download the fullset analysis inputs into `benchmark/analysis/data/` if you want to reproduce the analysis figures.
- Use one of the official annotation JSONs directly as `--questions-file` for terminal-agent batch evaluation.

Concrete analysis-input placement:

```bash
mkdir -p benchmark/analysis/data

cp /path/to/HippoCamp/Adam/Fullset/Adam.json benchmark/analysis/data/Adam.json
cp /path/to/HippoCamp/Bei/Fullset/Bei.json benchmark/analysis/data/Bei.json
cp /path/to/HippoCamp/Victoria/Fullset/Victoria.json benchmark/analysis/data/Victoria.json

cp /path/to/HippoCamp/Adam/Fullset/Adam_files.xlsx benchmark/analysis/data/Adam_files.xlsx
cp /path/to/HippoCamp/Bei/Fullset/Bei_files.xlsx benchmark/analysis/data/Bei_files.xlsx
cp /path/to/HippoCamp/Victoria/Fullset/Victoria_files.xlsx benchmark/analysis/data/Victoria_files.xlsx
```

`benchmark/sample_questions.json` is only a lightweight smoke-test file. It is not the full benchmark release.

## RAG / search-agent pipeline

Run all commands from `benchmark/`.

### 1. Configure services

```bash
cp ../.env.example ../.env
cp configs/services.yaml.example configs/services.yaml
```

Start Qdrant locally if needed:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$PWD/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

### 2. Index the parsed text release

```bash
python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
```

### 3. Start the retriever server when required

```bash
python3 scripts/retriever_server.py -e hippo -p 18000
```

### 4. Run the released methods

For smoke tests you can use `sample_questions.json`. For full runs, replace it with one of the official Hugging Face annotation JSONs.

Standard RAG:

```bash
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate
```

Self RAG:

```bash
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval self_rag --generator gemini --evaluate
```

ReAct (Gemini-2.5-flash):

```bash
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --generator gemini_react --evaluate
```

ReAct (Qwen3-30B-A3B):

```bash
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --generator qwen_react --evaluate
```

Search-R1:

```bash
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --generator search_r1 --evaluate
```

### 5. Standalone evaluation

```bash
python3 scripts/run_evaluation.py /path/to/your_saved_query_results.json \
  --metrics rouge bleu retrieval_precision retrieval_recall retrieval_f1 \
  --limit 1 --no-save
```

## Benchmark analysis

Run the analysis scripts from the repository root after downloading the six fullset files into `benchmark/analysis/data/`.

Difficulty statistics and figure generation:

```bash
python3 benchmark/analysis/difficulty/generate_difficulty_reports.py
```

Timestamp distribution figure:

```bash
python3 benchmark/analysis/file_time/file_combined_boxplot.py
```

Difficulty-vs-performance requires your own local evaluation outputs:

```bash
python3 benchmark/analysis/difficulty_vs_performance/difficulty_performance_plot.py \
  --results-root /path/to/your/local_eval_results
```

All generated analysis figures are written to `benchmark/analysis/outputs/figs/`.

## Terminal-agent pipeline

Run the terminal-agent commands from the repository root.

### 1. Load Docker images

```bash
docker load -i hippocamp_adam_subset.tar
docker load -i hippocamp_adam_fullset.tar
docker load -i hippocamp_bei_subset.tar
docker load -i hippocamp_bei_fullset.tar
docker load -i hippocamp_victoria_subset.tar
docker load -i hippocamp_victoria_fullset.tar
```

### 2. Start a container

```bash
docker run -it -p 18082:8080 --name hippocamp-adam-subset hippocamp/adam_subset:latest
```

The container command layer is documented in [`docs/docker_api.md`](docker_api.md). The mapped host port `18082` serves the WebUI and the mirrored HTTP API. The prompt-based agent wrappers still execute the benchmark commands inside the container via `docker exec`, but the same file operations are also visible through the WebUI backend.

The image metadata also exposes `5000/tcp`. The public workflow does not require mapping it, because the released WebUI and agent wrappers use `8080`. If you want full parity with the declared image ports for debugging, you can add an extra mapping such as `-p 58082:5000`.

Quick WebUI/API smoke checks after the container is running:

```bash
curl http://localhost:18082/api/files/list
curl http://localhost:18082/api/history
curl "http://localhost:18082/api/return_metadata/Guide%20to%20attending%20court.pdf"
```

### 3. Run a terminal agent

Gemini:

```bash
python3 agent/gemini.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --ensure-webui \
  --log-json result/gemini_docker_session.json
```

GPT-5.2:

```bash
python3 agent/chatgpt.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --ensure-webui \
  --log-json result/chatgpt_docker_session.json
```

OpenAI-compatible / vLLM:

```bash
python3 agent/vllm.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --api-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --ensure-webui \
  --log-json result/vllm_docker_session.json
```

### 4. Batch evaluation with official annotation JSONs

Use one of the released annotation JSONs from Hugging Face as `--questions-file`. This is the recommended public workflow because those files already contain the QA pairs and the associated evidence annotations that the evaluators and analysis scripts expect.

```bash
python3 agent/chatgpt_batch.py \
  --container hippocamp-adam-subset \
  --questions-file /path/to/Adam_Subset.json \
  --ensure-webui \
  --log-dir log/chatgpt_batch \
  --result-dir result/chatgpt_batch
```

The batch runners preserve fields such as `question`, `answer`, `file_path`, `evidence`, `rationale`, `agent_cap`, `QA_type`, and `profiling_type` when present, and derive `agent_file_list` from tool traces.

With `--ensure-webui`, the wrappers start the WebUI automatically and post command logs to `/api/log_command`, so the browser view can follow the agent trajectory in real time.

### 5. Top-level evaluation

```bash
python3 evaluate.py \
  --input-dataset /tmp/hippocamp_eval_sample.json \
  --print-results
```

The input records for `evaluate.py` should contain:

- `query_id`
- `query` or `question`
- `answer`
- `ground_truth`
- `ground_file_list`
- `agent_file_list`
- `time_ms`
