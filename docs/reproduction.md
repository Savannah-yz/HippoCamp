# Reproduction Notes

This document collects the step-by-step commands for the public release. Use the root [`README.md`](../README.md) for the release map and high-level workflow selection. Use the [`Docker interface reference`](./docker_api.md) for the container interface itself.

## Before You Start

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install for the benchmark subsystem:

```bash
pip install -e ./benchmark --no-deps
```

If you see matplotlib or fontconfig cache warnings, set:

```bash
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

Create `.env` once:

```bash
cp .env.example .env
```

## External Inputs

The public workflows rely on a few external assets:

| Asset | Source | Local path | Used by |
| --- | --- | --- | --- |
| Parsed-text release (`HippoCamp_Gold`) | Hugging Face | `benchmark/HippoCamp_Gold/` | RAG indexing and `return_txt`-based workflows |
| Official annotation JSON files | Hugging Face | any local path | RAG batch runs and terminal-agent batch runs |
| Fullset annotation JSON files and spreadsheets | Hugging Face | `benchmark/analysis/data/` | analysis scripts |
| Docker archives | release download links | any local path before `docker load` | terminal-agent pipeline |

Browse the Hugging Face release from the `Files and versions` tab:

- <https://huggingface.co/datasets/MMMem-org/HippoCamp>

`benchmark/sample_questions.json` is only a lightweight smoke-test input. It is not the full benchmark release.

## RAG / Search-Agent Pipeline

Run all commands from `benchmark/`.

### Inputs

- `benchmark/HippoCamp_Gold/`
- `benchmark/sample_questions.json` for smoke tests, or an official annotation JSON for full runs
- provider keys in `../.env`

### Setup

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

Index the parsed-text release:

```bash
python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
```

Start the retriever server when required by ReAct or Search-R1:

```bash
python3 scripts/retriever_server.py -e hippo -p 18000
```

### Run the Released Methods

For smoke tests you can use `sample_questions.json`. For full runs, replace it with one of the official Hugging Face annotation JSON files.

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

### Outputs

If you pass `--output-dir`, `scripts/run_query.py` writes:

- one per-query result JSON named `<query_id>_<timestamp>.json`
- `summary_<timestamp>.json` for the whole batch
- `evaluation_<timestamp>.json` when `--evaluate` is enabled

Run standalone evaluation on a saved result file:

```bash
python3 scripts/run_evaluation.py /path/to/your_saved_query_results.json \
  --metrics rouge bleu retrieval_precision retrieval_recall retrieval_f1 \
  --limit 1 --no-save
```

To see all supported standalone metrics:

```bash
python3 scripts/run_evaluation.py --list-metrics
```

## Terminal-Agent Pipeline

Run the terminal-agent commands from the repository root.

### Inputs

- a loaded HippoCamp Docker image
- a running named container
- either a single `--question` string or an official annotation JSON passed through `--questions-file`

### Load and Start a Container

```bash
docker load -i hippocamp_adam_subset.tar
docker run -it -p 18082:8080 --name hippocamp-adam-subset hippocamp/adam_subset:latest
```

At the in-container prompt, start the WebUI if you want browser inspection:

```bash
webui
```

The container interface, WebUI, host-port mapping, and HTTP routes are documented in [`docs/docker_api.md`](./docker_api.md).

### Single-Question Runs

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

### Batch Evaluation

Use one of the released annotation JSONs from Hugging Face as `--questions-file`. This is the canonical public workflow because the JSON already contains the QA pairs and evidence annotations expected downstream.

```bash
python3 agent/chatgpt_batch.py \
  --container hippocamp-adam-subset \
  --questions-file /path/to/Adam_Subset.json \
  --ensure-webui \
  --log-dir log/chatgpt_batch \
  --result-dir result/chatgpt_batch
```

The batch runners preserve fields such as `question`, `answer`, `file_path`, `evidence`, `rationale`, `agent_cap`, `QA_type`, and `profiling_type` when present, and derive `agent_file_list` from tool traces.

### Outputs

Single-question wrappers write one session JSON to the path passed through `--log-json`.

Batch wrappers write:

- stdout and stderr logs under `--log-dir`
- one per-question result JSON under `--result-dir`
- `summary.jsonl` under `--result-dir`
- `aggregate.json` under `--result-dir`, unless overridden by `--aggregate-json`

With `--ensure-webui`, the wrappers start the WebUI automatically and mirror command traces to `/api/log_command`.

## Prompt-Based Agent Evaluation

Run `evaluate.py` from the repository root:

```bash
python3 evaluate.py \
  --input-dataset /tmp/hippocamp_eval_sample.json \
  --print-results
```

### Required Input Schema

Each record should contain:

- `query_id`
- `query` or `question`
- `answer`
- `ground_truth`
- `ground_file_list`
- `agent_file_list`
- `time_ms`

### Outputs

- `--per-query-results-json` writes the per-query judge results
- `--aggregate-metrics-json` writes the aggregate summary
- `--print-results` prints both to stdout

This evaluator is intended for terminal-agent outputs and other custom agent outputs that follow the simplified result schema. It is different from `benchmark/scripts/run_evaluation.py`, which evaluates `run_query.py` outputs.

## Benchmark Analysis

Download the six fullset analysis files into `benchmark/analysis/data/` first. Exact copy commands are documented in [`benchmark/analysis/data/README.md`](../benchmark/analysis/data/README.md).

Difficulty statistics and figure generation:

```bash
python3 benchmark/analysis/difficulty/generate_difficulty_reports.py
```

Timestamp distribution figure:

```bash
python3 benchmark/analysis/file_time/file_combined_boxplot.py
```

Difficulty-vs-performance with your own local evaluation outputs:

```bash
python3 benchmark/analysis/difficulty_vs_performance/difficulty_performance_plot.py \
  --results-root /path/to/your/local_eval_results
```

All generated analysis figures are written to `benchmark/analysis/outputs/`.
