# Reproduction Notes

This document collects longer command sequences that support the public release README.

## Shared setup

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the RAG subsystem:

```bash
pip install -e ./benchmark
```

If you see matplotlib or fontconfig cache warnings, set:

```bash
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

## RAG / search-agent pipeline

Run all commands from `benchmark/`.

### 1. Place benchmark data

- Download `HippoCamp_Gold` from Hugging Face and place it under `benchmark/HippoCamp_Gold/`.
- Place `Adam.json`, `Bei.json`, and `Victoria.json` under `benchmark/analysis/data/` if you
  want to reproduce the analysis figures.

### 2. Configure services

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

### 3. Index data

```bash
python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
```

### 4. Start retriever server when required

```bash
python3 scripts/retriever_server.py -e hippo -p 18000
```

### 5. Run methods

Standard RAG:

```bash
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate
```

Self RAG:

```bash
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --retrieval self_rag --generator gemini --evaluate
```

ReAct (Gemini-2.5-flash):

```bash
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator gemini_react --evaluate
```

ReAct (Qwen3-30B-A3B):

```bash
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator qwen_react --evaluate
```

Search-R1:

```bash
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator search_r1 --evaluate
```

### 6. Standalone evaluation

```bash
python3 scripts/run_evaluation.py analysis/result/finance_standardrag/evaluation_results.json \
  --metrics rouge bleu retrieval_precision retrieval_recall retrieval_f1 \
  --limit 1 --no-save
```

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
docker run -it -p 8082:8080 --name hippocamp-adam-subset hippocamp/adam_subset:latest
```

### 3. Run a terminal agent

Gemini:

```bash
python3 agent/gemini.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --log-json result/gemini_docker_session.json
```

GPT-5.2:

```bash
python3 agent/chatgpt.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --log-json result/chatgpt_docker_session.json
```

OpenAI-compatible / vLLM:

```bash
python3 agent/vllm.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --api-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --log-json result/vllm_docker_session.json
```

### 4. Batch evaluation

```bash
python3 agent/chatgpt_batch.py \
  --container hippocamp-adam-subset \
  --questions-file benchmark/benchmark_example.json \
  --log-dir log/chatgpt_batch \
  --result-dir result/chatgpt_batch
```

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
