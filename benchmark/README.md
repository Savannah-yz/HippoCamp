# Benchmark Code

`benchmark/` contains the released RAG, search-agent, and analysis code for HippoCamp. This directory is intentionally code-first: benchmark data, Docker archives, and evaluation outputs are external to the repository workflow and should be placed locally as needed.

## Main Entry Points

- `scripts/run_offline.py`: build the local retrieval index from `HippoCamp_Gold/`
- `scripts/retriever_server.py`: retriever service used by ReAct and Search-R1
- `scripts/run_query.py`: main query runner for RAG and search-agent baselines
- `scripts/run_evaluation.py`: metric runner for saved query results
- `analysis/`: analysis scripts and data-placement instructions

## Required Inputs

This directory expects three kinds of inputs:

- `benchmark/HippoCamp_Gold/`: parsed-text JSON files used by indexing and retrieval
- a question file passed to `scripts/run_query.py --batch`
- `benchmark/analysis/data/`: fullset annotation JSON files and metadata spreadsheets for analysis scripts

Use `sample_questions.json` only for smoke tests. For full benchmark evaluation, replace it with one of the official Hugging Face annotation JSON files.

## Outputs

When you run `scripts/run_query.py` with `--output-dir`, it writes:

- one per-query JSON file named `<query_id>_<timestamp>.json`
- `summary_<timestamp>.json` for the batch
- `evaluation_<timestamp>.json` when `--evaluate` is enabled

Analysis scripts write figures and reports under `benchmark/analysis/outputs/`.

## Evaluation

The RAG / search-agent pipeline uses:

- `scripts/run_query.py --evaluate` for integrated evaluation during generation
- `scripts/run_evaluation.py` for standalone evaluation of saved `summary_*.json` files

Supported metrics include:

- `rouge`, `bleu`, `exact_match`, `covered_exact_match`
- `semantic_similarity`, `bert_score`
- `retrieval_precision`, `retrieval_recall`, `retrieval_f1`
- `llm_judge`
- file-list precision / recall / F1 from `file_list` and `retrieved_file_list`

To inspect the available metric names from the CLI:

```bash
python3 scripts/run_evaluation.py --list-metrics
```

## Quick Start

Run these commands from inside `benchmark/` after creating the environment from the repository root:

```bash
cp ../.env.example ../.env
cp configs/services.yaml.example configs/services.yaml

python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate
```

## Related Docs

- [`../README.md`](../README.md): release overview and workflow map
- [`../docs/reproduction.md`](../docs/reproduction.md): step-by-step commands
- [`./analysis/README.md`](./analysis/README.md): analysis inputs, commands, and outputs
- [`./HippoCamp_Gold/README.md`](./HippoCamp_Gold/README.md): expected parsed-text directory layout
