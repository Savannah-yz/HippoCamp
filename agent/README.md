# Terminal Agent Subsystem

`agent/` contains the prompt-based terminal-agent wrappers for HippoCamp Docker environments.

## Main Scripts

- `gemini.py` / `gemini_batch.py`
- `chatgpt.py` / `chatgpt_batch.py`
- `claude.py` / `claude_batch.py`
- `vllm.py` / `vllm_batch.py`

## Inputs

The wrappers expect:

- a running HippoCamp Docker container passed through `--container`
- either a single `--question` string or a batch `--questions-file`
- provider credentials from the repository-root `.env`

For batch runs, the recommended `--questions-file` is an official annotation JSON from Hugging Face.

## Outputs

Single-question wrappers write one session trace to the file passed through `--log-json`.

Batch wrappers write:

- stdout and stderr logs under `--log-dir`
- one per-question result JSON under `--result-dir`
- `summary.jsonl` under `--result-dir`
- `aggregate.json` under `--result-dir`, unless overridden by `--aggregate-json`

## Evaluation

For terminal-agent batch outputs, use the repository-root [`evaluate.py`](../evaluate.py).

The most common input is:

- `result/<batch_name>/aggregate.json`

This evaluator computes:

- LLM-as-a-judge answer quality
- file-list precision / recall / F1 from `ground_file_list` and `agent_file_list`

Example:

```bash
python3 evaluate.py \
  --input-dataset result/chatgpt_batch/aggregate.json \
  --per-query-results-json result/chatgpt_batch/judge_results.json \
  --aggregate-metrics-json result/chatgpt_batch/judge_summary.json
```

## Related Docs

- [`../README.md`](../README.md): release overview and workflow map
- [`../docs/reproduction.md`](../docs/reproduction.md): step-by-step single and batch commands
- [`../docs/docker_api.md`](../docs/docker_api.md): container commands, WebUI, and HTTP routes
