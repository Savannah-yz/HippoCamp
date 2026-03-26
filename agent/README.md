# Terminal Agent Subsystem

The public release now uses the repository root [`README.md`](../README.md) as the
single authoritative guide.

Use the root README for:

- Docker image setup
- terminal-agent evaluation
- prompt-based agent extension guidance
- result submission instructions

The `agent/` directory continues to host the runnable terminal-agent backends,
including:

- `gemini.py` / `gemini_batch.py`
- `chatgpt.py` / `chatgpt_batch.py`
- `claude.py` / `claude_batch.py`
- `vllm.py` / `vllm_batch.py`

Detailed container interface notes live in [`docs/docker_api.md`](../docs/docker_api.md).
