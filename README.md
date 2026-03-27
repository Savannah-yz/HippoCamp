# HippoCamp: Benchmarking Contextual Agents on Personal Computers

HippoCamp is a benchmark for evaluating contextual agents on realistic personal-computing environments. It covers multimodal file management across documents, images, audio, video, emails, calendars, and other everyday artifacts, with 42.4 GB of data across more than 2K files. On top of these environments, HippoCamp provides 581 QA pairs and 46.1K structured trajectory annotations for analyzing search, perception, and multi-step reasoning failures.

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb)](https://savannah-yz.github.io/project_page/HippoCamp/)
[![Data Visualization](https://img.shields.io/badge/Data-Visualization-0a7ea4)](https://savannah-yz.github.io/data_visualization/HippoCamp/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-f59e0b)](https://huggingface.co/datasets/MMMem-org/HippoCamp)
[![Paper](https://img.shields.io/badge/Paper-PDF-b91c1c)](docs/paper/HippoCamp.pdf)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Project%20Page-16a34a)](https://savannah-yz.github.io/project_page/HippoCamp/)
[![Video Link](https://img.shields.io/badge/Video-Link-coming_soon-6b7280)](#video)
[![Docker Images](https://img.shields.io/badge/Docker-Images-coming_soon-2496ed)](#docker-images)

![HippoCamp teaser](assets/figs/teaser_overview.png)

## Release Status

- HippoCamp was submitted to ECCV on February 15, 2026.
- Public release dates for the repository, project page, dataset, and visualization assets will be finalized later.
- The demo video link, Docker archive download links, and final citation are not finalized yet.

## Overview

HippoCamp instantiates three archetypal personal-computing environments and evaluates two task families:

- **Factual Retention**: retrieve, comprehend, and reason over factual information grounded in multimodal files.
- **Profiling**: aggregate weak, distributed evidence across files and time to infer a coherent user model.

The current release includes:

- **42.4 GB** of benchmark data
- **2K+** real-world files
- **581** QA pairs
- **46.1K** structured trajectory annotations
- **3** user profiles
- **2** task families

The released annotation JSONs follow the hierarchy below.

![Annotation hierarchy](assets/figs/annotation_hierarchy.png)

## Release Assets

| Asset | Status | Location | Contents |
| --- | --- | --- | --- |
| GitHub repository | Available | this repository | code, configs, docs, figures, evaluation scripts, sample assets |
| Hugging Face dataset | Available | <https://huggingface.co/datasets/MMMem-org/HippoCamp> | raw environments, official annotation JSONs, `HippoCamp_Gold`, metadata spreadsheets |
| Project page | Available | <https://savannah-yz.github.io/project_page/HippoCamp/> | benchmark overview, examples, leaderboard |
| Data visualization | Available | <https://savannah-yz.github.io/data_visualization/HippoCamp/> | interactive environment visualization |
| Docker archives | Pending | to be added at release | six prebuilt benchmark images |
| Demo video | Pending | to be added at release | end-to-end WebUI and agent demo |
| Citation | Pending | to be finalized after release | final BibTeX and `CITATION.cff` |

## Data Layout

The Hugging Face dataset is the authoritative data release. Its main structure is:

```text
HippoCamp/
├── Adam/
│   ├── Subset/
│   │   ├── Adam_Subset/
│   │   ├── Adam_Subset.json
│   │   └── Adam_Subset.xlsx
│   └── Fullset/
│       ├── Adam/
│       ├── Adam.json
│       └── Adam_files.xlsx
├── Bei/
│   ├── Subset/
│   │   ├── Bei_Subset/
│   │   ├── Bei_Subset.json
│   │   └── Bei_Subset.xlsx
│   └── Fullset/
│       ├── Bei/
│       ├── Bei.json
│       └── Bei_files.xlsx
└── Victoria/
    ├── Subset/
    │   ├── Victoria_Subset/
    │   ├── Victoria_Subset.json
    │   └── Victoria_Subset.xlsx
    └── Fullset/
        ├── Victoria/
        ├── Victoria.json
        └── Victoria_files.xlsx
```

These artifacts serve different roles:

- The six source directories store the raw personal-computing files.
- The six annotation JSON files store released QA pairs together with annotations such as `file_path`, `file_number`, `file_modality`, `file_type`, `evidence`, `rationale`, `agent_cap`, `QA_type`, and `profiling_type`.
- `HippoCamp_Gold` stores parsed-text JSON files with the schema `{file_info, summary, segments}`.
- The `*_files.xlsx` spreadsheets store explicit metadata such as creation time, modification time, and location fields.

The Hugging Face Dataset Viewer exposes six configs, each with `profiling` and `factual_retention` splits:

| Config | Profile | Scope | Raw files | Total QA | Profiling | Factual retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `adam_fullset` | Adam | Full | 344 | 123 | 20 | 103 |
| `adam_subset` | Adam | Subset | 158 | 18 | 6 | 12 |
| `bei_fullset` | Bei | Full | 875 | 235 | 20 | 215 |
| `bei_subset` | Bei | Subset | 147 | 27 | 4 | 23 |
| `victoria_fullset` | Victoria | Full | 711 | 223 | 20 | 203 |
| `victoria_subset` | Victoria | Subset | 137 | 11 | 6 | 5 |

## What To Download And Why

All public benchmark data is distributed from the Hugging Face dataset page:

- <https://huggingface.co/datasets/MMMem-org/HippoCamp>

On that page, open the `Files and versions` tab to browse and download the released directories and files.

| If you want to... | Download this | Why it is needed | Local destination |
| --- | --- | --- | --- |
| run the RAG / search-agent pipeline | `HippoCamp_Gold/` | it stores the parsed-text JSON used for indexing and retrieval | `benchmark/HippoCamp_Gold/` |
| run terminal-agent batch evaluation | one official annotation JSON such as `Adam.json` or `Adam_Subset.json` | it provides the released questions, answers, and evidence annotations used as `--questions-file` | any local path |
| reproduce the analysis figures | `Adam.json`, `Bei.json`, `Victoria.json`, `Adam_files.xlsx`, `Bei_files.xlsx`, `Victoria_files.xlsx` | the analysis scripts read the fullset annotations and metadata spreadsheets directly | `benchmark/analysis/data/` |
| inspect or study the raw benchmark environments | the six source directories under `Adam/`, `Bei/`, and `Victoria/` | they contain the original personal-computing files | any local path |

`HippoCamp_Gold` is not just an optional extra. It is the parsed-text release that powers the public RAG workflow and the Docker-side `return_txt` interface. If you only want to browse the raw files in Docker, you do not need it locally. If you want to run the released retrieval pipeline, you do.

## Repository Structure

```text
.
├── README.md
├── .env.example
├── requirements.txt
├── evaluate.py
├── CITATION.cff
├── assets/
│   ├── figs/
│   └── tables/
├── docs/
│   ├── docker_api.md
│   ├── reproduction.md
│   └── paper/HippoCamp.pdf
├── benchmark/
│   ├── README.md
│   ├── sample_questions.json
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   ├── analysis/
│   │   ├── README.md
│   │   └── data/README.md
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

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/Savannah-yz/HippoCamp.git
cd HippoCamp

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install for the benchmark subsystem:

```bash
pip install -e ./benchmark --no-deps
```

`requirements.txt` already includes the merged dependency set used by the public release.

### 2. Configure local caches

```bash
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

### 3. Create `.env`

```bash
cp .env.example .env
```

The root `.env` covers terminal-agent keys, RAG provider keys, judge settings, and optional local-service configuration.

### 4. Download benchmark data

Use the Hugging Face dataset pieces as follows:

- **RAG / search-agent pipeline**: place the parsed-text release under `benchmark/HippoCamp_Gold/`.
- **Terminal-agent batch evaluation**: use an official annotation JSON such as `Adam.json`, `Adam_Subset.json`, `Bei.json`, or `Victoria_Subset.json` as `--questions-file`.
- **Analysis reproduction**: place the three fullset annotation JSON files and the three fullset metadata spreadsheets under `benchmark/analysis/data/`.

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

If you are unsure which Hugging Face asset corresponds to your workflow, use the `What To Download And Why` table above first.

### 5. Install Docker Desktop

- macOS / Windows: <https://www.docker.com/products/docker-desktop/>
- Linux: follow your distribution-specific Docker Engine setup

## Docker Images

The public workflow uses six prebuilt Docker archives. Their download links are not finalized yet, but the expected archive names, image names, and host-port mappings are fixed:

| Archive | Image | Container name | Host port | Download |
| --- | --- | --- | --- | --- |
| `hippocamp_bei_subset.tar` | `hippocamp/bei_subset:latest` | `hippocamp-bei-subset` | `18081` | To be added at release |
| `hippocamp_adam_subset.tar` | `hippocamp/adam_subset:latest` | `hippocamp-adam-subset` | `18082` | To be added at release |
| `hippocamp_victoria_subset.tar` | `hippocamp/victoria_subset:latest` | `hippocamp-victoria-subset` | `18083` | To be added at release |
| `hippocamp_bei_fullset.tar` | `hippocamp/bei_fullset:latest` | `hippocamp-bei-fullset` | `18084` | To be added at release |
| `hippocamp_adam_fullset.tar` | `hippocamp/adam_fullset:latest` | `hippocamp-adam-fullset` | `18085` | To be added at release |
| `hippocamp_victoria_fullset.tar` | `hippocamp/victoria_fullset:latest` | `hippocamp-victoria-fullset` | `18086` | To be added at release |

Load an archive once you have it:

```bash
docker load -i hippocamp_adam_subset.tar
```

Start a container:

```bash
docker run -it -p 18082:8080 --name hippocamp-adam-subset hippocamp/adam_subset:latest
```

The `docker run -it ...` command gives you the interactive shell. Start the browser WebUI inside the container with:

```bash
webui
```

For detailed container, WebUI, and HTTP-route behavior, see [`docs/docker_api.md`](docs/docker_api.md).

## Inputs and Outputs

The main workflows use different inputs and produce different artifacts:

| Workflow | Main inputs | Required external assets | Main outputs |
| --- | --- | --- | --- |
| RAG / search-agent pipeline | `benchmark/sample_questions.json` for smoke tests, or an official annotation JSON via `--batch` | `benchmark/HippoCamp_Gold/` | per-query result JSONs in `--output-dir`, plus `summary_*.json` and `evaluation_*.json` |
| Terminal agent, single question | a Docker container plus `--question` | Docker image archive | one session log JSON via `--log-json` |
| Terminal agent, batch | `--questions-file` pointing to an official annotation JSON | Docker image archive | `summary.jsonl`, per-question result JSON files, `aggregate.json`, and stdout/stderr logs |
| Top-level evaluator | JSON or JSONL file via `evaluate.py --input-dataset` | none | per-query judge results JSON and aggregate metrics JSON |
| Analysis scripts | fullset annotation JSON files and `*_files.xlsx` spreadsheets | Hugging Face fullset assets | figures and reports under `benchmark/analysis/outputs/` |

If you are unsure which files to feed into which script, start with [`docs/reproduction.md`](docs/reproduction.md), [`benchmark/README.md`](benchmark/README.md), and [`agent/README.md`](agent/README.md).

## Reproduction Paths

HippoCamp exposes two complementary evaluation paths:

- a **RAG / search-agent** pipeline under `benchmark/`
- a **terminal-agent** pipeline under `agent/`

Longer command sequences live in [`docs/reproduction.md`](docs/reproduction.md).

### A. RAG / Search-Agent Pipeline

Run these commands from `benchmark/`.

1. Copy the parsed-text release into `benchmark/HippoCamp_Gold/`.
2. Copy `.env` from the repository root and configure `configs/services.yaml` if needed.
3. Start Qdrant if you use the default local vector-store setup.
4. Build the local index.
5. Run a baseline.

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$PWD/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant

python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo

python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate
```

Use `sample_questions.json` only for smoke tests. For full evaluation, replace it with one of the official Hugging Face annotation JSON files.

Important outputs:

- `--output-dir` writes one per-query JSON file named `<query_id>_<timestamp>.json`
- `--output-dir` also writes `summary_<timestamp>.json`
- `--evaluate` together with `--output-dir` writes `evaluation_<timestamp>.json`

### B. Terminal-Agent Pipeline

Run the terminal-agent commands from the repository root.

Single-question example:

```bash
python3 agent/chatgpt.py \
  --container hippocamp-adam-subset \
  --question "What does the guide say about court dress code?" \
  --ensure-webui \
  --log-json result/chatgpt_docker_session.json
```

Batch example:

```bash
python3 agent/chatgpt_batch.py \
  --container hippocamp-adam-subset \
  --questions-file /path/to/Adam_Subset.json \
  --ensure-webui \
  --log-dir log/chatgpt_batch \
  --result-dir result/chatgpt_batch
```

The canonical batch input is an official annotation JSON from Hugging Face, not `HippoCamp_Gold`.

Important outputs:

- `--log-json` stores the single-question session trace
- `--log-dir` stores stdout and stderr logs
- `--result-dir` stores one per-question result JSON, `summary.jsonl`, and `aggregate.json`

#### Prompt Configs For Docker-Based Agent Evaluation

The terminal-agent wrappers expose `--prompt-config` so you can control whether the agent may use:

- `return_ori`: original source files
- `return_txt`: parsed-text JSON backed by `HippoCamp_Gold`
- `return_img`: rendered visual assistance

The released wrappers map `--prompt-config` to the Docker-side feature flags as follows:

| Config | `return_ori` | `return_txt` | `return_img` | Recommended use |
| --- | --- | --- | --- | --- |
| `config0` | on | on | on | Full auxiliary interface |
| `config1` | on | off | off | Source-only setting |
| `config2` | on | off | on | Image-enabled, text-disabled |
| `config3` | on | on | off | Text-enabled, image-disabled |

### C. Prompt-Based Agent Output Evaluation

For terminal-agent outputs and other custom agent results, use `evaluate.py`.

See the `Evaluation` section below for:

- the exact input schema
- the supported metrics
- example commands
- the difference between `evaluate.py` and `benchmark/scripts/run_evaluation.py`

## Evaluation

HippoCamp currently exposes two distinct evaluation entrypoints. They are meant for different output formats.

### 1. RAG / Search-Agent Evaluation

Use these when your outputs come from `benchmark/scripts/run_query.py`.

Code paths:

- `benchmark/scripts/run_query.py --evaluate`
- `benchmark/scripts/run_evaluation.py`

`run_query.py --evaluate` is the integrated path during generation. `run_evaluation.py` is the standalone evaluator for saved `summary_*.json` files.

RAG evaluation metrics currently include:

- answer-quality metrics: `rouge`, `bleu`, `exact_match`, `covered_exact_match`, `semantic_similarity`, `bert_score`
- chunk-retrieval metrics: `retrieval_precision`, `retrieval_recall`, `retrieval_f1`
- optional LLM judge: `llm_judge`
- file-level retrieval metrics: precision / recall / F1 computed from `file_list` and `retrieved_file_list`

Examples:

```bash
# Evaluate while running the query pipeline
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate

# Re-evaluate an existing result directory
python3 benchmark/scripts/run_evaluation.py /path/to/output_dir

# Inspect available metrics
python3 benchmark/scripts/run_evaluation.py --list-metrics

# Run specific metrics explicitly
python3 benchmark/scripts/run_evaluation.py /path/to/output_dir \
  --metrics rouge bleu semantic_similarity llm_judge
```

`benchmark/scripts/run_evaluation.py` accepts either:

- a directory containing `summary_*.json`
- or a specific `summary_*.json` file

If you request `llm_judge`, configure Azure judge credentials through `.env` or environment variables such as `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`.

Typical `run_query.py` result record:

```json
{
  "timestamp": "20260327_120000",
  "provider": "gemini",
  "bench": "hippo",
  "query_id": "1",
  "query": "What does the guide say about court dress code?",
  "ground_truth": "Dress neatly and appropriately for court.",
  "answer": "The guide says court attendees should dress neatly and appropriately.",
  "retrieved_chunks": [
    {
      "rank": 1,
      "content": "You should dress neatly and appropriately for court...",
      "score": 0.91,
      "id": "chunk_001",
      "metadata": {
        "file_info": {
          "file_path": "Guide to attending court.pdf",
          "file_type": "pdf",
          "file_name": "Guide to attending court.pdf"
        }
      }
    }
  ],
  "file_list": [
    "Guide to attending court.pdf"
  ],
  "retrieved_file_list": [
    "Guide to attending court.pdf"
  ],
  "execution_time_ms": 4231
}
```

Typical `benchmark/scripts/run_evaluation.py` output record:

```json
{
  "query_id": "1",
  "query": "What does the guide say about court dress code?",
  "answer": "The guide says court attendees should dress neatly and appropriately.",
  "ground_truth": "Dress neatly and appropriately for court.",
  "judge": {
    "llm_as_a_judge_score": 4,
    "pred": "yes",
    "score_0_5": 4,
    "score_normalized": 0.8,
    "api_status": "success"
  },
  "simple_metrics": {
    "rouge": {
      "score": 0.71
    },
    "semantic_similarity": {
      "score": 0.89
    }
  },
  "file_list_metrics": {
    "f1_score": 1.0,
    "recall": 1.0,
    "precision": 1.0
  },
  "timestamp": "20260327_120530"
}
```

### 2. Prompt-Based Agent Evaluation

Use [`evaluate.py`](evaluate.py) for terminal-agent outputs and other custom agent outputs that follow the simplified result schema.

This evaluator currently computes:

- LLM-as-a-judge answer quality
- file-list precision / recall / F1 from `ground_file_list` versus `agent_file_list`

Expected input fields per record:

- `query_id`
- `query` or `question`
- `answer`
- `ground_truth`
- `ground_file_list`
- `agent_file_list`
- `time_ms`

Typical input sources:

- `result/<batch_name>/aggregate.json` produced by `agent/*_batch.py`
- any custom JSON or JSONL file that follows the schema above

Example:

```bash
python3 evaluate.py \
  --input-dataset result/chatgpt_batch/aggregate.json \
  --per-query-results-json result/chatgpt_batch/judge_results.json \
  --aggregate-metrics-json result/chatgpt_batch/judge_summary.json
```

You can also pass judge credentials explicitly:

```bash
python3 evaluate.py \
  --input-dataset result/chatgpt_batch/aggregate.json \
  --judge-api-url "$AZURE_OPENAI_ENDPOINT" \
  --judge-api-key "$AZURE_OPENAI_API_KEY" \
  --print-results
```

If no judge API credentials are configured, `evaluate.py` still runs, but the LLM-judge portion falls back to zero-score outputs.

Typical terminal-agent batch record from `agent/*_batch.py`:

```json
{
  "agent": "ChatGPT",
  "query_id": "1",
  "question": "What does the guide say about court dress code?",
  "answer": "The guide says court attendees should dress neatly and appropriately.",
  "steps": {
    "step_1": {
      "command": "return_txt 'Guide to attending court.pdf'"
    }
  },
  "ground_file_list": [
    "Guide to attending court.pdf"
  ],
  "agent_file_list": [
    "Guide to attending court.pdf"
  ],
  "ground_truth": "Dress neatly and appropriately for court.",
  "evidence": [
    "Guide to attending court.pdf"
  ],
  "agent_cap": "search+reasoning",
  "QA_type": "factual_retention",
  "time_ms": 6842
}
```

Typical per-query result from `evaluate.py`:

```json
{
  "query_id": "1",
  "query": "What does the guide say about court dress code?",
  "answer": "The guide says court attendees should dress neatly and appropriately.",
  "ground_truth": "Dress neatly and appropriately for court.",
  "time_ms": 6842,
  "judge": {
    "llm_as_a_judge_score": 4,
    "pred": "yes",
    "score_0_5": 4,
    "score_normalized": 0.8,
    "api_status": "success"
  },
  "ground_file_list": [
    "Guide to attending court.pdf"
  ],
  "agent_file_list": [
    "Guide to attending court.pdf"
  ],
  "file_list_metrics": {
    "f1_score": 1.0,
    "recall": 1.0,
    "precision": 1.0
  }
}
```

Typical aggregate summary from `evaluate.py`:

```json
{
  "total_queries": 100,
  "metrics": {
    "llm_judge": {
      "mean": 3.42,
      "min": 0,
      "max": 5,
      "count": 100,
      "yes_count": 68,
      "no_count": 32,
      "pass_rate": 0.68,
      "avg_latency_ms": 5120.4
    }
  },
  "file_list_metrics": {
    "total_evaluated": 100,
    "average_f1_score": 0.57,
    "average_recall": 0.61,
    "average_precision": 0.55,
    "file_hit_rate": 0.61
  }
}
```

## Develop Your Prompt-Based Agent

The `agent/` directory is designed to be extensible. The released wrappers use a tag-based interaction contract centered on `<tool>` and `<answer>`:

```text
<think>...</think>
<tool>{"name":"terminal","arguments":{"command":"..."}}</tool>
<answer>...</answer>
```

To build your own prompt-based agent:

- Start from `agent/gemini.py` or `agent/vllm.py`.
- Keep the same terminal-tool contract and JSON command shape.
- Treat `/hippocamp/data` as the working directory root for benchmark file paths.
- Use the released Docker commands as the environment interface: `list_files`, `return_txt`, `return_img`, `return_ori`, `return_metadata`, `set_flags`, `webui`, `webui_status`, and `webui_stop`.
- Preserve the batch output schema so that `evaluate.py` can score your results without extra adapters.

## Results and Analysis

### Table 1

![Table 1](assets/tables/table1.png)

**Main results on HippoCamp across user profiles.** We evaluate representative MLLMs and agent methods on profiling and factual retention, reporting F1 and accuracy (Acc) for each profile and the overall average.

### Table 2

![Table 2](assets/tables/table2.png)

**Agent capability-wise analysis on HippoCamp.** We report F1 and LLM-judge accuracy aggregated by capability labels, decomposed into search, perception, and reasoning.

### Analysis Figures

#### Number of Supporting Files Per Question

![Number of supporting files per question](assets/figs/evidence_breadth.png)

This figure shows how many ground-truth supporting files each question requires. It is the benchmark's direct view of evidence breadth.

#### Number of Evidence Modalities Per Question

![Number of evidence modalities per question](assets/figs/modality_breadth.png)

This figure shows how many distinct file modalities each question spans, such as documents, images, audio, or other file types.

#### Annotated Reasoning Depth Per Question

![Annotated reasoning depth per question](assets/figs/reasoning_depth.png)

This figure shows the number of reasoning steps required by the released rationale annotations.

#### Overall Difficulty Distribution

![Overall difficulty distribution](assets/figs/difficulty_distribution.png)

This figure summarizes the released scalar difficulty score, which combines evidence breadth, modality breadth, file types, evidence items, reasoning steps, question length, answer length, and time span.

#### Performance As Question Difficulty Increases

![Performance as question difficulty increases](assets/figs/difficulty_vs_performance.png)

This figure aligns question difficulty with per-question judge scores across released methods, showing how performance changes as questions become harder.

See [`benchmark/analysis/README.md`](benchmark/analysis/README.md) for the scripts that reproduce these figures.

## Leaderboard and Result Submission

The public leaderboard is hosted on the project page:

- <https://savannah-yz.github.io/project_page/HippoCamp/>

If you evaluate a new prompt-based agent or baseline, email your result package to `zhe012@e.ntu.edu.sg`. Include the method name, model name, settings summary, and either the result JSON or the aggregate evaluation output.

## Video

The demo video link will be added after the release assets are finalized.

## Citation

The final citation will be added after the public release is finalized. The current `CITATION.cff` file should be treated as provisional until then.
