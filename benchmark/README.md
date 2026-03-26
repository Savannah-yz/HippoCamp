# Benchmark Code

`benchmark/` contains the released RAG, search-agent, and analysis code for HippoCamp. This directory is intentionally kept code-first: it does not ship benchmark data, analysis inputs, or evaluation results.

## Layout

- `configs/`: service, provider, indexing, ingestion, and evaluation configs
- `scripts/run_offline.py`: build the local retrieval index from `HippoCamp_Gold/`
- `scripts/retriever_server.py`: retriever service used by ReAct and Search-R1
- `scripts/run_query.py`: main batch query runner for RAG and search-agent baselines
- `scripts/run_evaluation.py`: metric runner for saved query results
- `src/`: Python implementation for ingestion, chunking, retrieval, generation, and evaluation
- `sample_questions.json`: minimal smoke-test query file
- `HippoCamp_Gold/README.md`: explains where to place the parsed text release from Hugging Face
- `analysis/`: analysis scripts and data-placement instructions

## Required external inputs

This directory expects external benchmark assets downloaded from Hugging Face:

- `benchmark/HippoCamp_Gold/`: parsed text JSON files used by the RAG pipeline
- `benchmark/analysis/data/`: fullset annotation JSON files and fullset metadata spreadsheets used by the analysis scripts

See:

- [`../README.md`](../README.md) for the public release overview
- [`../docs/reproduction.md`](../docs/reproduction.md) for longer command sequences
- [`analysis/README.md`](./analysis/README.md) for analysis-specific setup and commands

## Quick start

Create the environment from the repository root, then run the benchmark pipeline from inside `benchmark/`:

```bash
cp ../.env.example ../.env
cp configs/services.yaml.example configs/services.yaml

python3 scripts/run_offline.py HippoCamp_Gold/ --all -e hippo
python3 scripts/run_query.py --batch sample_questions.json -e hippo \
  --retrieval standard_rag --generator gemini --evaluate
```

Replace `sample_questions.json` with one of the official Hugging Face annotation JSON files for full benchmark evaluation.
