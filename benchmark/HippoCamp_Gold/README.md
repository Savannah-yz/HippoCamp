# HippoCamp_Gold

`HippoCamp_Gold/` stores the parsed-text release used by the public RAG pipeline and by Docker-side `return_txt` workflows.

## Expected Layout

Download the parsed-text release from:

- <https://huggingface.co/datasets/MMMem-org/HippoCamp>

Place the extracted profile folders under this directory:

```text
benchmark/HippoCamp_Gold/
├── Adam/
├── Bei/
└── Victoria/
```

## File Schema

Each parsed file follows the high-level schema:

```json
{
  "file_info": {...},
  "summary": "",
  "segments": [...]
}
```

At a high level:

- `file_info` records file identity, user, modality, timestamps, location metadata, and QA linkage
- `summary` stores an optional file-level summary
- `segments` store modality-specific parsed content such as page-level document text or timestamped audio transcription

## Used By

- `benchmark/scripts/run_offline.py`
- `benchmark/scripts/run_query.py`
- Docker-side `return_txt`

See [`../README.md`](../README.md) and [`../../docs/reproduction.md`](../../docs/reproduction.md) for the workflows that consume this directory.
