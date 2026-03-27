# Evaluation Reference

This document contains the detailed input/output schemas and JSON examples for both evaluation entrypoints. For a high-level overview, see the root [`README.md`](../README.md). For end-to-end reproduction commands, see [`reproduction.md`](./reproduction.md).

## 1. RAG / Search-Agent Evaluation

Use these when your outputs come from `benchmark/scripts/run_query.py`.

Code paths:

- `benchmark/scripts/run_query.py --evaluate`
- `benchmark/scripts/run_evaluation.py`

`run_query.py --evaluate` is the integrated path during generation. `run_evaluation.py` is the standalone evaluator for saved `summary_*.json` files.

### Metrics

- answer-quality metrics: `rouge`, `bleu`, `exact_match`, `covered_exact_match`, `semantic_similarity`, `bert_score`
- chunk-retrieval metrics: `retrieval_precision`, `retrieval_recall`, `retrieval_f1`
- optional LLM judge: `llm_judge`
- file-level retrieval metrics: precision / recall / F1 computed from `file_list` and `retrieved_file_list`

### Commands

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

### Typical `run_query.py` Result Record

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

### Typical `run_evaluation.py` Output Record

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

## 2. Prompt-Based Agent Evaluation

Use [`evaluate.py`](../evaluate.py) for terminal-agent outputs and other custom agent outputs that follow the simplified result schema.

### Metrics

- LLM-as-a-judge answer quality
- file-list precision / recall / F1 from `ground_file_list` versus `agent_file_list`

### Expected Input Fields Per Record

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

### Commands

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

### Typical Terminal-Agent Batch Record

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

### Typical Per-Query Result

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

### Typical Aggregate Summary

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
