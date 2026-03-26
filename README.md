# HippoCamp: Benchmarking Contextual Agents on Personal Computers

HippoCamp is a benchmark for evaluating contextual agents on realistic personal-computing environments. It focuses on personalized multimodal memory: agents must search, perceive, and reason over long-lived file systems that contain heterogeneous user-specific evidence spread across documents, images, audio, video, emails, calendars, and other everyday digital artifacts.

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb)](https://savannah-yz.github.io/project_page/HippoCamp/)
[![Data Visualization](https://img.shields.io/badge/Data-Visualization-0a7ea4)](https://savannah-yz.github.io/data_visualization/HippoCamp/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-f59e0b)](https://huggingface.co/datasets/MMMem-org/HippoCamp/tree/main/HippoCamp_Gold)
[![Paper](https://img.shields.io/badge/Paper-PDF-b91c1c)](docs/paper/HippoCamp.pdf)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Project%20Page-16a34a)](https://savannah-yz.github.io/project_page/HippoCamp/)
[![Video](https://img.shields.io/badge/Video-XXXXXXX-6b7280)](XXXXXXX)
[![Docker Images](https://img.shields.io/badge/Docker-Images-2496ed)](XXXXXXX)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](#install)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ed)](#install)

![HippoCamp teaser](assets/figs/0_overview.png)

## News

- `[XX/XX/2026]`: Submitted to ECCV.
- `[03/26/2026]`: GitHub repository released.

## Overview

HippoCamp instantiates three archetypal personal-computing profiles and evaluates two core task families:

- **Factual retention**: retrieve and verify user-specific facts from multimodal files.
- **Profiling**: infer stable user patterns, preferences, workflows, and habits from long-horizon evidence.

The released benchmark covers:

- **42.4 GB** of data
- **2K+** real-world files
- **581** QA pairs
- **46.1K** structured trajectory annotations
- **3** user profiles
- **2** task families (Profiling & Factual Retention)

## What Is Released

This public release separates code, data, and Docker artifacts cleanly:

- **GitHub**: evaluation code, agent code, configs, public documentation, paper PDF, public figures, and result assets.
- **Hugging Face**: benchmark data, including `HippoCamp_Gold` and the released benchmark annotations used by the analysis scripts.
- **Google Drive**: Docker image archives for the six benchmark environments.
- **Project Page**: leaderboard and additional presentation material.

## Public Repository Structure

The public release surface is organized as:

```text
.
├── README.md
├── .env.example
├── requirements.txt
├── requirements-local.txt
├── evaluate.py
├── CITATION.cff
├── assets/
│   ├── figs/
│   └── tables/
├── docs/
│   ├── docker_api.md
│   ├── reproduction.md
│   └── paper/HippoCamp.pdf
├── HippoCamp/
│   ├── README.md
│   ├── benchmark_example.json
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   ├── analysis/
│   └── HippoCamp_Gold/README.md
└── agent/
    ├── README.md
    ├── gemini.py
    ├── chatgpt.py
    ├── claude.py
    ├── vllm.py
    └── *_batch.py
```

## Install

### 1. Clone and create a Python environment

```bash
git clone <your-public-repo-url>
cd release

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional local-model extras for Qwen / Search-R1 experiments:

```bash
pip install -r requirements-local.txt
```

Optional editable install for the HippoCamp RAG subsystem:

```bash
pip install -e ./HippoCamp
```

### 2. Configure runtime caches

The evaluation and analysis scripts may touch matplotlib and fontconfig caches. To keep
the runtime self-contained:

```bash
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

The same defaults are also documented in `.env.example`.

### 3. Create `.env`

```bash
cp .env.example .env
```

The merged root `.env` is loaded by the public scripts. It groups:

- terminal-agent API keys
- RAG / generator API keys
- evaluation / judge settings
- optional vector DB and Mongo settings
- optional local model service ports

### 4. Download benchmark data

Download the released benchmark data from Hugging Face:

- [https://huggingface.co/datasets/MMMem-org/HippoCamp/tree/main/HippoCamp_Gold](https://huggingface.co/datasets/MMMem-org/HippoCamp/tree/main/HippoCamp_Gold)

Place the extracted gold data under:

```text
HippoCamp/HippoCamp_Gold/
├── Adam/
├── Bei/
└── Victoria/
```

For analysis reproduction, place these annotation JSON files under `HippoCamp/analysis/data/`:

- `Adam.json`
- `Bei.json`
- `Victoria.json`

The lightweight `*_files.xlsx` spreadsheets are already included in the repo.

### 5. Install Docker Desktop

Install Docker Desktop for your platform before using the benchmark images:

- macOS / Windows: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Linux: follow your distribution-specific Docker Engine setup

### 6. Download Docker images

The Docker archives are intentionally not hosted in GitHub. Replace `XXXXXXX` with your
Google Drive links.

| Image                              | Profile  | Variant | Download    |
| ---------------------------------- | -------- | ------- | ----------- |
| `hippocamp_bei_subset.tar`       | Bei      | Subset  | `XXXXXXX` |
| `hippocamp_bei_fullset.tar`      | Bei      | Fullset | `XXXXXXX` |
| `hippocamp_adam_subset.tar`      | Adam     | Subset  | `XXXXXXX` |
| `hippocamp_adam_fullset.tar`     | Adam     | Fullset | `XXXXXXX` |
| `hippocamp_victoria_subset.tar`  | Victoria | Subset  | `XXXXXXX` |
| `hippocamp_victoria_fullset.tar` | Victoria | Fullset | `XXXXXXX` |

Load the images:

```bash
docker load -i hippocamp_bei_subset.tar
docker load -i hippocamp_bei_fullset.tar
docker load -i hippocamp_adam_subset.tar
docker load -i hippocamp_adam_fullset.tar
docker load -i hippocamp_victoria_subset.tar
docker load -i hippocamp_victoria_fullset.tar
```

Run the containers:

```bash
# Subsets
docker run -it -p 8081:8080 --name hippocamp-bei-subset hippocamp/bei_subset:latest
docker run -it -p 8082:8080 --name hippocamp-adam-subset hippocamp/adam_subset:latest
docker run -it -p 8083:8080 --name hippocamp-victoria-subset hippocamp/victoria_subset:latest

# Fullsets
docker run -it -p 8084:8080 --name hippocamp-bei-fullset hippocamp/bei_fullset:latest
docker run -it -p 8085:8080 --name hippocamp-adam-fullset hippocamp/adam_fullset:latest
docker run -it -p 8086:8080 --name hippocamp-victoria-fullset hippocamp/victoria_fullset:latest
```

## End-to-End Evaluation

HippoCamp exposes two complementary reproduction paths:

- a **RAG / search-agent** pipeline under `HippoCamp/`
- a **terminal-agent** pipeline under `agent/`

Longer command cookbooks are collected in [`docs/reproduction.md`](docs/reproduction.md).

### A. RAG / Search-Agent Pipeline

Run these commands from `HippoCamp/`.

Start Qdrant if you use the default local setup:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$PWD/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Index the benchmark data:

```bash
python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
```

Start the retriever server when using ReAct or Search-R1:

```bash
python3 scripts/retriever_server.py -e hippo -p 18000
```

Run the methods:

```bash
# Standard RAG
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate

# Self RAG
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --retrieval self_rag --generator gemini --evaluate

# ReAct (Gemini-2.5-flash)
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator gemini_react --evaluate

# ReAct (Qwen3-30B-A3B)
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator qwen_react --evaluate

# Search-R1
python3 scripts/run_query.py --batch benchmark_example.json -e hippo \
  --generator search_r1 --evaluate
```

Run standalone evaluation on saved results:

```bash
python3 scripts/run_evaluation.py analysis/result/finance_standardrag/evaluation_results.json \
  --metrics rouge bleu retrieval_precision retrieval_recall retrieval_f1 \
  --limit 1 --no-save
```

Method notes:

- **Standard RAG / Self RAG**: require embeddings, vector store, and generator APIs.
- **ReAct (Gemini-2.5-flash)**: requires a running retriever server and Gemini API.
- **ReAct (Qwen3-30B-A3B)**: requires a running retriever server and local GPU-backed model serving.
- **Search-R1**: requires a running retriever server and local model support.

### B. Terminal-Agent Pipeline

Run these commands from the repository root.

Single-question examples:

```bash
# Gemini terminal agent
python3 agent/gemini.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --log-json result/gemini_docker_session.json

# GPT-5.2 terminal agent
python3 agent/chatgpt.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --log-json result/chatgpt_docker_session.json

# OpenAI-compatible / vLLM terminal agent
python3 agent/vllm.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --api-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --log-json result/vllm_docker_session.json
```

Batch example:

```bash
python3 agent/chatgpt_batch.py \
  --container hippocamp-adam-subset \
  --questions-file HippoCamp/benchmark_example.json \
  --log-dir log/chatgpt_batch \
  --result-dir result/chatgpt_batch
```

Top-level evaluation:

```bash
python3 evaluate.py \
  --input-dataset /tmp/hippocamp_eval_sample.json \
  --print-results
```

The expected result schema for `evaluate.py` is:

- `query_id`
- `query` or `question`
- `answer`
- `ground_truth`
- `ground_file_list`
- `agent_file_list`
- `time_ms`

## Develop Your Prompt-Based Agent

The `agent/` directory is designed to be extensible. A public extension should follow the
same interaction contract as the existing terminal agents:

```text
<think>...</think>
<tool>{"name":"terminal","arguments":{"command":"..."}}</tool>
<answer>...</answer>
<end>TERMINATE</end>
```

Recommended development workflow:

1. Start from `agent/gemini.py` or `agent/vllm.py`.
2. Keep the terminal tool contract unchanged.
3. Use the Docker-exposed commands (`list_files`, `return_txt`, `return_img`, `return_ori`,
   `return_metadata`, `set_flags`) as the only environment interface.
4. Preserve the output JSON shape expected by the batch runners.
5. Ensure the batch result records expose `agent_file_list`, since the evaluators compute
   file-level metrics from those touched paths.

The batch runners already extract file paths from command traces, including direct calls to:

- `return_txt`
- `return_img`
- `return_ori`
- `return_metadata`

After you run your own prompt-based agent, evaluate it with `evaluate.py`, then send your
results to `zhe012@e.ntu.edu.sg`. The leaderboard lives on the project page.

## Docker Interface

The Docker images expose a small, stable command surface documented in detail in
[`docs/docker_api.md`](docs/docker_api.md).

Verified locally on `hippocamp/adam_subset:latest`:

```bash
docker run --rm hippocamp/adam_subset:latest hhelp
docker run --rm hippocamp/adam_subset:latest list_files '*.pdf'
docker run --rm hippocamp/adam_subset:latest return_txt 'Guide to attending court.pdf'
docker run --rm hippocamp/adam_subset:latest return_metadata 'Guide to attending court.pdf'
```

Use quotes for file paths with spaces:

```bash
return_txt 'Guide to attending court.pdf'
```

## Results and Analysis

### Table 1

![Table 1](assets/tables/table1.png)

**Main results on HippoCamp across user profiles.** We evaluate representative MLLMs and
agent methods on profiling and factual retention, reporting F1 and accuracy (Acc) for each
archetypal profile and the overall average. Values are percentages (one decimal; % omitted).
Best is highlighted; second-best is underlined.

### Table 2

![Table 2](assets/tables/table2.png)

**Agent capability-wise analysis on HippoCamp.** For the methods in Table 1, we report F1
and LLM-judge accuracy (Acc) aggregated by agent capability labels, decomposed into search,
perception, and reasoning, for profiling and factual retention as well as the overall
average. Values are percentages (one decimal; % omitted). Best is highlighted; second-best
is underlined.

### Analysis figures

#### Evidence breadth

![Evidence breadth](assets/figs/15_Evidence.png)

`15_Evidence` measures evidence breadth: the number of ground-truth supporting files per
query. In the released code, this is read from `file_number` when present and otherwise
falls back to `len(file_path)`. It directly captures retrieval breadth and shows the
benchmark’s multi-file heavy tail.

#### Modality breadth

![Modality breadth](assets/figs/16_Modality.png)

`16_Modality` measures modality breadth: the number of distinct values in `file_modality`
for a query. This is the benchmark’s direct proxy for cross-modal grounding burden.

#### Reasoning depth

![Reasoning depth](assets/figs/17_Reasoning_steps.png)

`17_Reasoning_steps` measures reasoning depth: the maximum `step_id` in `rationale`, or the
rationale length when explicit IDs are absent. It captures the depth of annotated multi-step
integration required by a query.

#### Difficulty distribution

![Difficulty distribution](assets/figs/18_Difficulty.png)

`18_Difficulty` aggregates eight factors into a scalar difficulty score: evidence files,
modalities, file types, evidence items, reasoning steps, question length, answer length, and
time span. The code applies weighted normalization, interaction terms, a hard-case bonus,
and a sigmoid mapping to place the final score on a 0-100 scale.

#### Difficulty vs performance

![Difficulty vs performance](assets/figs/19_difficulty_vs_performance.png)

`19_difficulty_vs_performance` bins question difficulty in 5-point intervals and aligns each
bin with the per-question `judge.llm_as_a_judge_score` across nine evaluated methods. It
shows how performance degrades as retrieval, perception, and reasoning constraints co-occur.

## Leaderboard and Result Submission

The current leaderboard is hosted on the project page:

- [https://savannah-yz.github.io/project_page/HippoCamp/](https://savannah-yz.github.io/project_page/HippoCamp/)

If you evaluate a new prompt-based agent or baseline, email your result package to:

- `zhe012@e.ntu.edu.sg`

Please include:

- method name
- model name
- settings summary
- result JSON or aggregate evaluation output

## Video

An agent trajectory demo video will be linked here when it is public:

- `XXXXXXX`

## Citation

```bibtex
@misc{yang2026hippocamp,
  title={HippoCamp: Benchmarking Contextual Agents on Personal Computers},
  author={Zhe Yang and Shulin Tian and Kairui Hu and Shuai Liu and Hoang-Nhat Nguyen and Yichi Zhang and Zujin Guo and Mengying Yu and Zinan Zhang and Jingkang Yang and Chen Change Loy and Ziwei Liu},
  year={2026},
  note={ECCV submission},
  url={https://savannah-yz.github.io/project_page/HippoCamp/}
}
```
