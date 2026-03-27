# Docker Interface Reference

The public GitHub release distributes prebuilt Docker images rather than the image-build pipeline. This document therefore focuses on the runtime surface of the released images: the in-container CLI commands, the WebUI and HTTP API, the shell behavior, and the ports that matter for reproduction.

HippoCamp exposes two user-facing access layers:

- a **terminal command layer** inside the container, used by `docker run`, `docker exec`, and the prompt-based agent wrappers in `agent/`
- a **WebUI and HTTP API layer** on `HIPPOCAMP_PORT` (default `8080`), used for browser-based inspection, mirrored file operations, and command-history synchronization

The runtime details below were verified from the released `hippocamp/adam_subset:latest` image. In particular:

- `docker image inspect` shows the image exposes both `5000/tcp` and `8080/tcp`
- the image default user is `hippocamp_user`
- the image entrypoint is `/hippocamp/api/entrypoint.sh`
- the released runtime files inside the image include `/hippocamp/api/*`, `/hippocamp/webui/app.py`, and the `webui` helper scripts

## Runtime layout

The image runtime is organized under `/hippocamp/`:

- `/hippocamp/data` stores the raw benchmark files
- `/hippocamp/gold` stores the parsed text JSON release used by `return_txt`
- `/hippocamp/metadata` stores metadata, feature flags, and path aliases
- `/hippocamp/tools` stores conversion helpers
- `/hippocamp/api` stores the CLI entrypoints and Python runtime
- `/hippocamp/webui` stores the Flask-SocketIO WebUI
- `/hippocamp/output` stores generated images, WebUI logs, and transient outputs

The image entrypoint `/hippocamp/api/entrypoint.sh` resets feature flags on container start and then executes the requested command.

## Execution model

The same benchmark operations are exposed through both CLI and HTTP:

- For one-off smoke tests, you can invoke the CLI directly with `docker run --rm <image> <command>`.
- For a long-running named container, the released prompt-based agent wrappers use `docker exec <container> ...`.
- The WebUI backend mirrors these operations over HTTP on the mapped `8080` port.
- File paths are interpreted relative to `/hippocamp/data`.

Examples:

```bash
docker run --rm hippocamp/adam_subset:latest hhelp
docker run --rm hippocamp/adam_subset:latest list_files '*.pdf'

docker exec -it hippocamp-adam-subset bash -lc 'list_files "*.pdf"'
docker exec -it hippocamp-adam-subset bash -lc 'return_txt "Guide to attending court.pdf"'
```

## Port mapping and WebUI

The public release recommends these host-port mappings:

| Container | Image | Host port | Container port |
| --- | --- | --- | --- |
| `hippocamp-bei-subset` | `hippocamp/bei_subset:latest` | `18081` | `8080` |
| `hippocamp-adam-subset` | `hippocamp/adam_subset:latest` | `18082` | `8080` |
| `hippocamp-victoria-subset` | `hippocamp/victoria_subset:latest` | `18083` | `8080` |
| `hippocamp-bei-fullset` | `hippocamp/bei_fullset:latest` | `18084` | `8080` |
| `hippocamp-adam-fullset` | `hippocamp/adam_fullset:latest` | `18085` | `8080` |
| `hippocamp-victoria-fullset` | `hippocamp/victoria_fullset:latest` | `18086` | `8080` |

Image metadata also exposes `5000/tcp`, but the released runtime files inspected from the image do not route the documented WebUI or the released agent wrappers through `5000`. For public reproduction:

- publish `8080` because it serves the WebUI and HTTP API
- publish `5000` only if you want full parity with the image's declared ports for debugging

Container port `8080` serves:

- the benchmark WebUI for visual inspection
- the WebUI HTTP API, including mirrored file operations such as `/api/return_txt/...`
- the `/api/log_command` endpoint used by the released agent wrappers to sync command and output logs into the WebUI
- the `/api/bash_notify` and `/api/terminal_notify` endpoints used for terminal-to-WebUI synchronization
- the WebSocket channel used by the Flask-SocketIO frontend

The wrappers auto-detect the mapped host port through `docker port <container> 8080/tcp`. You can override it manually with `--webui-base http://localhost:18082` or the corresponding port for another container.

To manage the WebUI on a running container:

```bash
docker exec -it hippocamp-adam-subset bash -lc 'webui'
docker exec -it hippocamp-adam-subset bash -lc 'webui_status'
docker exec -it hippocamp-adam-subset bash -lc 'webui_stop'
```

Starting the container with `docker run -it -p <host>:8080 ...` only gives you the interactive shell. The browser WebUI does not appear on `http://localhost:<host>` until `webui` has been started inside the container.

If Docker reports that the container name is already in use, reuse the existing named container instead of running `docker run` again:

```bash
docker start -ai hippocamp-adam-fullset
```

To replace it with a fresh container using the same name:

```bash
docker rm -f hippocamp-adam-fullset
docker run -it -p 18085:8080 --name hippocamp-adam-fullset hippocamp/adam_fullset:latest
```

Concrete example for Adam fullset:

1. Start the container:

```bash
docker run -it -p 18085:8080 --name hippocamp-adam-fullset hippocamp/adam_fullset:latest
```

   If the container already exists, use `docker start -ai hippocamp-adam-fullset` instead.

2. Start the WebUI at the in-container prompt:

```bash
webui
```

3. Keep that terminal open, then open <http://localhost:18085> in your browser.

Browser URLs after `webui` starts:

- `hippocamp-bei-subset` -> <http://localhost:18081>
- `hippocamp-adam-subset` -> <http://localhost:18082>
- `hippocamp-victoria-subset` -> <http://localhost:18083>
- `hippocamp-bei-fullset` -> <http://localhost:18084>
- `hippocamp-adam-fullset` -> <http://localhost:18085>
- `hippocamp-victoria-fullset` -> <http://localhost:18086>

You can also let the released agent wrappers start it automatically:

```bash
python3 agent/gemini.py --container hippocamp-adam-subset --question "..." --ensure-webui
python3 agent/chatgpt.py --container hippocamp-adam-subset --question "..." --ensure-webui
python3 agent/vllm.py --container hippocamp-adam-subset --question "..." --ensure-webui
```

If `http://localhost:<host-port>` still does not open after `webui` starts:

1. Verify the container is running and still owns the published port:

```bash
docker ps --format '{{.Names}}\t{{.Ports}}\t{{.Status}}' | grep hippocamp-adam-fullset
```

2. Verify the WebUI success banner appeared inside the container after running `webui`.

3. Remember that `docker start -ai <container>` reuses the container's original port bindings. If the container was first created without `-p 18085:8080`, remove it and recreate it with the published port.

4. If the container has the expected mapping and the WebUI is listening on `0.0.0.0:8080` inside the container, but the host still cannot open the page, restart Docker Desktop and recreate the container. That failure mode is in Docker host-port forwarding, not the HippoCamp runtime.

Useful host-side HTTP examples after mapping `18082:8080`:

```bash
curl http://localhost:18082/api/files/list
curl http://localhost:18082/api/history
curl http://localhost:18082/api/feature_flags
curl "http://localhost:18082/api/return_img/Guide%20to%20attending%20court.pdf?page=2"
```

Use URL encoding for spaces and other reserved characters in file paths.

## Shell behavior inside the container

The released image ships a customized bash environment in `/hippocamp/api/bashrc_additions`:

- the shell starts in `/hippocamp/data`
- `cd` is restricted to stay inside `/hippocamp/data`
- `ls` is wrapped to hide the `gold` directory from default listings
- short aliases are provided: `txt`, `img`, `ori`, `ls_files`
- full aliases are also provided: `return_txt`, `return_img`, `return_ori`, `return_metadata`, `list_files`, `set_flags`
- `webui`, `webui_stop`, and `webui_status` are exposed as shell aliases
- executed bash commands are mirrored to the WebUI through `/api/bash_notify`

This means the container behaves like an ordinary Ubuntu shell for standard commands such as `pwd`, `ls`, `cat`, `grep`, `find`, `head`, `tail`, `sed`, `awk`, `python3`, and `curl`, while adding the HippoCamp-specific aliases above.

## Core terminal commands

### `hhelp`

Print the in-container help banner, including dataset name, user, available commands, and the directory layout under `/hippocamp/`.

Verified example:

```bash
docker run --rm hippocamp/adam_subset:latest hhelp
```

### `list_files [pattern]`

List files in the benchmark environment.

Verified response fields:

- `success`
- `count`
- `data`
- `error`

Verified example:

```bash
docker run --rm hippocamp/adam_subset:latest list_files '*.pdf'
```

Notes:

- Quote patterns containing `*`.
- File paths may contain spaces, parentheses, ampersands, and punctuation.
- This is the right starting point before calling `return_txt`, `return_img`, `return_ori`, or `return_metadata`.

### `return_txt <file_path>`

Return the parsed gold text JSON for a file.

Verified response fields:

- `success`
- `data.segments`
- `data.file_info`
- `data.summary`
- `error`

Verified example:

```bash
docker run --rm hippocamp/adam_subset:latest return_txt 'Guide to attending court.pdf'
```

The returned `data.file_info` includes fields such as:

- `id`
- `user`
- `file_path`
- `file_name`
- `file_type`
- `file_modality`
- `creation_date`
- `modification_date`
- `QAID`
- `QANum`

The returned `data.segments` are modality-specific parsed content blocks. For documents, the segments typically correspond to pages; for audio they may carry timestamped transcription ranges.

### `return_metadata <file_path>`

Return lightweight metadata for a file.

Verified response fields:

- `success`
- `metadata`
- `error`

Verified example:

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

### `return_img <file_path> [output_path] [--page N]`

Render a file as page images for multimodal inspection.

Use this when the answer depends on layout, charts, scanned pages, or other visual structure that may not be fully preserved in `return_txt`.

The released prompt configuration documents these output keys:

- `success`
- `image_path`
- `image_paths`
- `image_b64`
- `image_b64_list`
- `page_count`
- `error`

Recommended command forms:

```bash
return_img "Documents/report.pdf"
return_img "Documents/report.pdf" --page 2
return_img "Documents/report.pdf" "/hippocamp/output/report_page2.png" --page 2
```

### `return_ori <file_path> [output_path]`

Return the original file payload.

Use this when your toolchain needs the exact original bytes rather than parsed text or rendered page images.

The released prompt configuration documents these output keys:

- `success`
- `file_path`
- `file_b64`
- `error`

### `set_flags <return_txt 0|1> <return_img 0|1>`

Toggle whether `return_txt` and `return_img` are exposed during the current container session.

Example:

```bash
set_flags 1 1
```

### `webui`

Start the in-container WebUI service.

The released prompt configuration documents these output keys:

- `success`
- `data`
- `url`
- `log`
- `error`

Example:

```bash
webui
```

The documented success payload points to:

- `url`: `http://localhost:8080` inside the container contract; after port mapping, open the corresponding host URL such as `http://localhost:18082`
- `log`: `/hippocamp/output/.webui/webui.log`

### `webui_status`

Check whether the WebUI is running.

The released prompt configuration documents the response keys:

- `success`
- `data`
- `error`

### `webui_stop`

Stop the WebUI service.

The released prompt configuration documents the response keys:

- `success`
- `data`
- `error`

## WebUI HTTP routes

The Flask backend inside `/hippocamp/webui/app.py` exposes the following routes on the mapped `8080` port.

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Main WebUI page |
| `/api/files` | `GET` | File tree view; supports `?path=<subdir>` |
| `/api/files/list` | `GET` | Flat file list; supports `?pattern=<glob-or-prefix>` |
| `/api/return_txt/<path:file_path>` | `GET` | Return parsed text JSON; supports preview options |
| `/api/return_txt_full/<path:file_path>` | `GET` | Return full parsed text JSON without preview truncation |
| `/api/return_img/<path:file_path>` | `GET` | Render image; supports `?page=N` |
| `/api/return_ori/<path:file_path>` | `GET` | Return original-file payload or preview |
| `/api/return_ori_full/<path:file_path>` | `GET` | Return full original-file payload |
| `/api/return_metadata/<path:file_path>` | `GET` | Return file metadata |
| `/api/serve_image/<path:image_path>` | `GET` | Serve generated image from `/hippocamp/output` |
| `/api/serve_file/<path:file_path>` | `GET` | Serve source file from `/hippocamp/data` |
| `/api/history` | `GET` | Return command history |
| `/api/terminal_notify` | `POST` | Terminal-to-WebUI command sync |
| `/api/log_command` | `POST` | Agent or external-client command logging |
| `/api/feature_flags` | `GET` | Read current feature flags |
| `/api/feature_flags` | `POST` | Always returns read-only error |
| `/api/bash_notify` | `POST` | Bash-command sync for ordinary shell commands |

The WebUI also uses Flask-SocketIO for live updates, including file-operation broadcasts and command-history events.

Examples:

```bash
curl http://localhost:18082/api/files
curl http://localhost:18082/api/files/list?pattern=*.pdf
curl http://localhost:18082/api/feature_flags
curl "http://localhost:18082/api/return_txt/Guide%20to%20attending%20court.pdf"
curl "http://localhost:18082/api/return_img/Guide%20to%20attending%20court.pdf?page=2"
curl "http://localhost:18082/api/return_metadata/Guide%20to%20attending%20court.pdf"
```

Notes:

- `return_txt` rejects `--page`; use `return_img <file_path> --page N` or the HTTP `page` query for page-level rendering.
- `return_txt` and `return_ori` support preview-oriented variants in the WebUI backend; use the `_full` routes when you need the full payload.
- `GET /api/feature_flags` is read-only. Update the flags from the terminal with `set_flags`.

## Environment variables and permissions

From the released image metadata and inspected runtime code, the key environment variables are:

- `HIPPOCAMP_DATA_DIR=/hippocamp/data`
- `HIPPOCAMP_GOLD_DIR=/hippocamp/gold`
- `HIPPOCAMP_METADATA_DIR=/hippocamp/metadata`
- `HIPPOCAMP_TOOLS_DIR=/hippocamp/tools`
- `HIPPOCAMP_OUTPUT_DIR=/hippocamp/output`
- `HIPPOCAMP_PORT=8080`
- `HIPPOCAMP_FEATURE_FLAGS=/hippocamp/metadata/feature_flags.json`
- `HIPPOCAMP_RETURN_IMG_BYTES=1` by default in the runtime code
- `HIPPOCAMP_RETURN_ORI_BYTES=1` by default in the runtime code
- `HIPPOCAMP_PDF_DPI=200` unless overridden

The default image user is `hippocamp_user`. The shell aliases call the benchmark file APIs through `sudo -u hippocamp_api ...`, which is why the prompt-based agent wrappers can use simple commands like `return_txt` while the actual file access is delegated to the API user.

## Quoting rules

Always quote file paths with spaces:

```bash
return_txt 'Guide to attending court.pdf'
return_metadata 'Guide to attending court.pdf'
return_img 'Guide to attending court.pdf' --page 2
```

## Recommended agent behavior

- Start with `list_files` to discover candidate evidence and confirm exact paths.
- Use `return_txt` for text-grounded reading and `return_metadata` for timestamps, modality, and location checks.
- Use `return_img` when layout or visual detail matters.
- Keep track of touched file paths, because the batch runners extract `agent_file_list` from these tool calls for downstream evaluation.
- Use `--ensure-webui` when you want the browser-based visualization to mirror the agent trajectory during execution.
