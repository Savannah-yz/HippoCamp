# HippoCamp_Gold

`HippoCamp_Gold/` is intentionally not shipped as part of the public GitHub release.

Download the parsed text release from:

- <https://huggingface.co/datasets/MMMem-org/HippoCamp>

Place the extracted profile folders under this directory so the released scripts keep their public paths unchanged:

```text
benchmark/HippoCamp_Gold/
├── Adam/
├── Bei/
└── Victoria/
```

`HippoCamp_Gold` stores the parsed text version of the benchmark files as JSON. Each file follows the high-level schema:

```json
{
  "file_info": {...},
  "summary": "",
  "segments": [...]
}
```

At a high level:

- `file_info` records file identity, user, modality, timestamps, location metadata, and QA linkage.
- `summary` stores an optional file-level summary string.
- `segments` store modality-specific parsed content, such as page-level document text or timestamped audio transcription.

The benchmark README and reproduction notes document the commands that consume this path.
