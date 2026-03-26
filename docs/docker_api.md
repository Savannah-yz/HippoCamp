# Docker Interface Reference

HippoCamp ships benchmark environments as Docker images. Each image exposes a small,
stable terminal interface that prompt-based agents can call through `docker exec` or
through the terminal-agent wrappers in `agent/`.

The commands below were verified locally against `hippocamp/adam_subset:latest`.

## Core commands

### `hhelp`

Print the in-container help banner, including dataset name, available commands, and
the expected directory layout under `/hippocamp/`.

### `list_files [pattern]`

List files in the benchmark environment.

Verified response fields:

- `success`
- `count`
- `data`
- `error`

Example:

```bash
docker run --rm hippocamp/adam_subset:latest list_files '*.pdf'
```

Notes:

- Quote patterns containing `*`.
- File paths can contain spaces, parentheses, and punctuation.

### `return_txt <file_path>`

Return the benchmark gold text JSON for a file.

Verified response fields:

- `success`
- `data.segments`
- `data.file_info`
- `data.summary`
- `error`

Example:

```bash
docker run --rm hippocamp/adam_subset:latest return_txt 'Guide to attending court.pdf'
```

The verified response includes:

- `data.segments`: page-level content blocks
- `data.file_info.id`
- `data.file_info.file_path`
- `data.file_info.file_type`
- `data.file_info.file_modality`
- `data.file_info.creation_date`
- `data.file_info.modification_date`
- `data.file_info.QAID`

### `return_metadata <file_path>`

Return lightweight metadata for a file.

Verified response fields:

- `success`
- `metadata`
- `error`

Example:

```bash
docker run --rm hippocamp/adam_subset:latest return_metadata 'Guide to attending court.pdf'
```

The verified metadata object includes:

- `id`
- `file_path`
- `file_type`
- `file_modality`
- `creation_date`
- `modification_date`
- `latitude`
- `longitude`
- `location`

### `return_img <file_path> [--page N]`

Render a file as page images for multimodal inspection.

Use this when the answer depends on layout, tables, scanned pages, or figures that are
not fully captured in text extraction.

### `return_ori <file_path> [output_path]`

Return the original file payload. This is useful when your model or toolchain wants the
source file instead of extracted text or rendered page images.

### `set_flags <return_txt 0|1> <return_img 0|1>`

Toggle whether `return_txt` and `return_img` are exposed during the current container
session.

Example:

```bash
set_flags 1 1
```

## Quoting rules

Always quote file paths with spaces:

```bash
return_txt 'Guide to attending court.pdf'
return_metadata 'Guide to attending court.pdf'
```

## Recommended agent behavior

- Start with `list_files` to discover candidate evidence.
- Use `return_txt` for direct content lookup.
- Use `return_img` when text extraction is insufficient.
- Use `return_metadata` for timestamps and modality checks.
- Keep track of touched file paths so they can be written into `agent_file_list` for
  downstream evaluation.
