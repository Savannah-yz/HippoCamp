#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal agent that interacts with a Docker container via terminal commands.

- Uses Claude Messages API chat session
- Sends a strict prompt that forces the model to output tagged tool calls
- Executes commands inside the container
- Feeds terminal output back to the model
- For return_img/return_ori, restores native files then sends multimodal parts
- Stops when model returns an answer + end signal
- Logs every step to a JSON file
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.parse
import urllib.request
import random
import tempfile
import warnings
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from datetime import datetime, timezone
from pathlib import Path

from prompt_modules.config import AVAILABLE_CONFIGS, get_auxiliary_functions_block
from prompt_modules.prompt_body import AVAILABLE_PROMPT_VERSIONS, build_prompt


CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"


class ClaudeAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: str = "",
        retry_after_sec: Optional[float] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after_sec = retry_after_sec


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        return


def _find_tag_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    end_tag = re.compile(rf"</{tag}>", re.IGNORECASE)
    end_match = end_tag.search(text, start)
    if end_match:
        return text[start:end_match.start()]
    next_tag = re.compile(r"</?(think|tool|answer|end)\\b", re.IGNORECASE)
    next_match = next_tag.search(text, start)
    if next_match:
        return text[start:next_match.start()]
    return text[start:]


def _extract_first_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return ""


def _loads_relaxed(obj_text: str) -> dict:
    if not obj_text:
        return {}
    if "\\|" in obj_text:
        obj_text = obj_text.replace("\\|", "|")
    try:
        return json.loads(obj_text)
    except Exception:
        try:
            import ast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return ast.literal_eval(obj_text)
        except Exception:
            return {}


def parse_tagged_response(text: str) -> dict:
    tool_block = _find_tag_block(text, "tool")
    answer_block = _find_tag_block(text, "answer")

    has_tool = bool(tool_block.strip())
    has_answer = bool(answer_block.strip())

    if has_tool and has_answer:
        tool_pos = re.search(r"<tool>", text, re.IGNORECASE)
        answer_pos = re.search(r"<answer>", text, re.IGNORECASE)
        if tool_pos and answer_pos and tool_pos.start() < answer_pos.start():
            has_answer = False
        else:
            has_tool = False

    if has_tool:
        json_text = _extract_first_json(tool_block)
        obj = _loads_relaxed(json_text)
        if not obj:
            raise ValueError("Tool JSON not found or invalid")
        args = obj.get("arguments") or obj.get("args") or {}
        tool_name = obj.get("name") or obj.get("tool") or ""

        # Strict format expects {"name":"terminal","arguments":{"command":"..."}}
        # but allow a small compatibility bridge for common mistakes.
        if tool_name and tool_name != "terminal":
            raise ValueError("Tool JSON must use name=terminal and provide arguments.command")

        if "command" not in args and "command" in obj:
            args = {"command": obj.get("command")}
        command = args.get("command", "")
        return {"action": "terminal", "command": command}

    if has_answer:
        answer = answer_block.strip()
        return {"action": "answer", "answer": answer}

    json_text = _extract_first_json(text)
    obj = _loads_relaxed(json_text)
    if obj:
        if "command" in obj:
            return {"action": "terminal", "command": obj.get("command", "")}
        if obj.get("name") == "terminal" and isinstance(obj.get("arguments"), dict):
            return {"action": "terminal", "command": obj["arguments"].get("command", "")}

    raise ValueError("No tool or answer found")


def fallback_parse(text: str) -> dict:
    # Try explicit "command:" patterns
    cmd_match = re.search(r"command\\s*[:=]\\s*(.+)", text, re.IGNORECASE)
    if cmd_match:
        cmd = cmd_match.group(1).strip()
        if cmd:
            return {"action": "terminal", "command": cmd}

    # Try lines that look like shell commands
    command_prefixes = (
        "list_files", "return_txt", "return_img", "return_ori",
        "return_metadata", "ls", "find", "grep", "cat", "head", "tail", "pwd", "cd"
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\\d\\.\\)\\s\\-*]+", "", line).strip()
        line = re.sub(r"^(command|cmd)\\s*[:=]\\s*", "", line, flags=re.IGNORECASE).strip()
        if line.startswith("$"):
            line = line[1:].strip()
        for prefix in command_prefixes:
            if line.startswith(prefix + " ") or line == prefix:
                return {"action": "terminal", "command": line}
    return {}


def parse_response(text: str, relaxed: bool = False) -> dict:
    try:
        return parse_tagged_response(text)
    except Exception:
        if not relaxed:
            raise
        obj = fallback_parse(text)
        if obj:
            return obj
    raise


def normalize_command(command: str) -> str:
    cmd = command.strip()
    if not cmd:
        return cmd
    if has_shell_composition(cmd):
        return cmd
    # If user already quoted, do not modify.
    if '"' in cmd or "'" in cmd:
        return cmd
    try:
        parts = shlex.split(cmd)
    except Exception:
        return cmd
    if not parts:
        return cmd

    head = parts[0]
    if head in ("return_txt", "return_metadata", "list_files"):
        if len(parts) >= 2:
            path = " ".join(parts[1:])
            if " " in path or "&" in path:
                return f'{head} "{path}"'
    elif head in ("return_ori",):
        if len(parts) == 2:
            path = parts[1]
            if " " in path or "&" in path:
                return f'{head} "{path}"'
    elif head in ("return_img",):
        if len(parts) >= 2:
            if "--page" in parts:
                idx = parts.index("--page")
                if idx > 1:
                    path = " ".join(parts[1:idx])
                    if " " in path or "&" in path:
                        return f'{head} "{path}" ' + " ".join(parts[idx:])
            else:
                path = " ".join(parts[1:])
                if " " in path or "&" in path:
                    return f'{head} "{path}"'
    return cmd


SHELL_COMPOSITION_RE = re.compile(r"(?<!\\)(\|\||&&|\|&|[|;<>])")


def has_shell_composition(command: str) -> bool:
    """
    Detect shell composition syntax (pipeline, redirection, chaining).
    These commands should bypass token-level command rewriting.
    """
    if not isinstance(command, str):
        return False
    cmd = command.strip()
    if not cmd:
        return False
    return bool(SHELL_COMPOSITION_RE.search(cmd))


def _list_files_pipeline_without_json_parse(command: str) -> bool:
    """
    list_files returns JSON payload, not plain newline paths.
    Piping directly into head/grep/xargs/while without parsing .data is usually invalid.
    """
    if not isinstance(command, str):
        return False
    cmd = command.strip()
    if not cmd or "list_files" not in cmd or "|" not in cmd:
        return False
    # Allow explicit JSON parsing pipelines.
    if re.search(r"\b(jq|python|python3|node|ruby|perl)\b", cmd):
        return False
    return True


def _grep_targets_pdf(parts: list) -> bool:
    if not parts:
        return False
    if os.path.basename(parts[0]).lower() != "grep":
        return False

    # Parse grep shape: grep [options] PATTERN [FILE...]
    i = 1
    while i < len(parts):
        token = parts[i]
        if token == "--":
            i += 1
            break
        if token in ("-e", "--regexp", "-f", "--file"):
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        break

    # No pattern found
    if i >= len(parts):
        return False

    # Skip pattern token
    i += 1

    for token in parts[i:]:
        if token.startswith("-"):
            continue
        low = token.lower()
        if low.endswith(".pdf") or "*.pdf" in low:
            return True
    return False


def _extract_grep_pattern(command: str) -> str:
    """Best-effort extraction of the first grep pattern argument."""
    if not isinstance(command, str):
        return ""
    try:
        parts = shlex.split(command)
    except Exception:
        return ""
    if not parts:
        return ""

    i = 0
    while i < len(parts):
        token = os.path.basename(parts[i]).lower()
        if token != "grep":
            i += 1
            continue
        i += 1
        # Parse options until the first non-option token (pattern).
        while i < len(parts):
            opt = parts[i]
            if opt == "--":
                i += 1
                break
            if opt in ("-e", "--regexp"):
                if i + 1 < len(parts):
                    return parts[i + 1]
                return ""
            if opt in ("-f", "--file"):
                i += 2
                continue
            if opt.startswith("-"):
                i += 1
                continue
            return opt
        if i < len(parts):
            return parts[i]
        return ""
    return ""


def _is_grep_no_match(command: str, output: str, exit_code: int) -> bool:
    if exit_code != 1:
        return False
    if not isinstance(command, str) or "grep" not in command:
        return False
    text = (output or "").strip().lower()
    if not text:
        return True
    return text == "(no matches)"


def _is_inline_terminal_blob_search(command: str) -> bool:
    """
    Detect anti-pattern commands that paste huge terminal JSON/text blobs
    into echo/subshells and then grep them. These are brittle and often
    break quoting; enforce source-grounded search instead.
    """
    if not isinstance(command, str):
        return False
    cmd = command.strip()
    if not cmd:
        return False

    low = cmd.lower()
    if "grep" not in low:
        return False

    likely_inline_blob = (
        len(cmd) > 1200
        and ("echo " in low or "<<<" in cmd or "$(" in cmd)
    )
    terminal_blob_markers = (
        "this is output of terminal",
        '"success"',
        "'success'",
        '"segments"',
        "'segments'",
        '"file_info"',
        "'file_info'",
    )
    has_blob_markers = any(marker in low for marker in terminal_blob_markers)
    has_nested_echo = bool(re.search(r"\$\(\s*echo\b", cmd, re.IGNORECASE))

    return bool(likely_inline_blob and (has_blob_markers or has_nested_echo))


def command_policy_error(command: str) -> str:
    if _list_files_pipeline_without_json_parse(command):
        return (
            "list_files returns JSON, not newline file paths. "
            "Do not pipe raw list_files output into head/grep/xargs/while. "
            "Parse `.data` first (e.g., with python/jq), then iterate file paths."
        )

    if _is_inline_terminal_blob_search(command):
        return (
            "Do not grep pasted terminal JSON/text blobs via echo/subshells. "
            "Run a source-grounded command instead, e.g. "
            'return_txt "<file_path>" | grep -iE "<pattern>".'
        )

    contains_shell_composition = has_shell_composition(command)
    try:
        parts = shlex.split(command)
    except Exception as exc:
        detail = str(exc).strip()
        if detail:
            return (
                f"Shell syntax/quoting appears invalid ({detail}). "
                "Use balanced quotes; for quote-rich payloads prefer: cat <<'EOF' ... EOF."
            )
        return (
            "Shell syntax/quoting appears invalid. "
            "Use balanced quotes; for quote-rich payloads prefer: cat <<'EOF' ... EOF."
        )
    if not parts:
        return ""

    head = os.path.basename(parts[0]).lower()

    if head in ("list_files", "ls_files"):
        args = parts[1:]
        if not args:
            return (
                "Avoid unbounded list_files. Provide a focused filename pattern, "
                'for example: list_files "*.pptx" or list_files "*keyword*".'
            )
        if len(args) == 1 and args[0].strip() == "*":
            return (
                "Avoid list_files \"*\". Use a narrower pattern by extension or keyword, "
                'for example: list_files "*.pdf" or list_files "*slide*".'
            )

    if head in ("return_txt", "txt") and "--page" in parts[1:]:
        return (
            'return_txt does not support --page. '
            'Use return_txt "<file_path>" or return_img "<file_path>" --page N.'
        )
    if not contains_shell_composition and head in ("return_txt", "txt") and len(parts) != 2:
        return 'return_txt accepts exactly one argument: return_txt "<file_path>".'
    if not contains_shell_composition and head == "return_metadata" and len(parts) != 2:
        return 'return_metadata accepts exactly one argument: return_metadata "<file_path>".'

    if _grep_targets_pdf(parts):
        return (
            'Direct grep on PDF files is not allowed. '
            'Use return_txt "<file_path>" and then search in the returned text.'
        )

    return ""


def validate_action(obj: dict) -> None:
    action = obj.get("action")
    if action == "terminal":
        if not obj.get("command"):
            raise ValueError("Missing command in tool call")
    elif action == "answer":
        if not obj.get("answer"):
            raise ValueError("Missing answer")
    else:
        raise ValueError("Invalid action")


def run_docker_command(container: str, cmd: str, user: str = "") -> tuple:
    # Ensure .bashrc is loaded for aliases and WebUI sync
    effective_cmd = cmd.strip()
    contains_shell_composition = has_shell_composition(effective_cmd)
    privileged_map = {
        "return_txt": "/hippocamp/api/return_txt",
        "return_img": "/hippocamp/api/return_img",
        "return_ori": "/hippocamp/api/return_ori",
        "return_metadata": "/hippocamp/api/return_metadata",
        "list_files": "/hippocamp/api/list_files",
        "set_flags": "/hippocamp/api/set_flags",
        "txt": "/hippocamp/api/return_txt",
        "img": "/hippocamp/api/return_img",
        "ori": "/hippocamp/api/return_ori",
        "ls_files": "/hippocamp/api/list_files",
    }
    if (
        effective_cmd
        and not contains_shell_composition
        and not effective_cmd.lstrip().startswith("sudo ")
    ):
        try:
            parts = shlex.split(effective_cmd)
        except Exception:
            parts = []
        if parts:
            head = parts[0]
            target = privileged_map.get(head)
            if not target and head.startswith("/hippocamp/api/"):
                base = os.path.basename(head)
                if base in (
                    "return_txt",
                    "return_img",
                    "return_ori",
                    "return_metadata",
                    "list_files",
                    "set_flags",
                ):
                    target = head
            if target:
                tail = " ".join(shlex.quote(p) for p in parts[1:])
                effective_cmd = f"sudo -u hippocamp_api {target} {tail}".strip()
    quoted_cmd = shlex.quote(effective_cmd)
    privileged_fn_shims = (
        'return_txt(){ sudo -u hippocamp_api /hippocamp/api/return_txt "$@"; }; '
        'return_img(){ sudo -u hippocamp_api /hippocamp/api/return_img "$@"; }; '
        'return_ori(){ sudo -u hippocamp_api /hippocamp/api/return_ori "$@"; }; '
        'return_metadata(){ sudo -u hippocamp_api /hippocamp/api/return_metadata "$@"; }; '
        'list_files(){ sudo -u hippocamp_api /hippocamp/api/list_files "$@"; }; '
        'set_flags(){ sudo -u hippocamp_api /hippocamp/api/set_flags "$@"; }; '
        'txt(){ return_txt "$@"; }; '
        'img(){ return_img "$@"; }; '
        'ori(){ return_ori "$@"; }; '
        'ls_files(){ list_files "$@"; }; '
    )
    bash_cmd = (
        "set -o pipefail >/dev/null 2>&1 || true; "
        "shopt -s expand_aliases >/dev/null 2>&1; "
        "source /home/hippocamp_user/.bashrc >/dev/null 2>&1; "
        f"{privileged_fn_shims}"
        f"eval {quoted_cmd}"
    )
    base = ["docker", "exec"]
    if user:
        base += ["--user", user]
    base += [container, "/bin/bash", "-lc", bash_cmd]

    start = time.time()
    proc = subprocess.run(
        base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    duration = time.time() - start
    output = (proc.stdout or "") + (proc.stderr or "")
    return output, proc.returncode, duration


def detect_webui_base(container: str, override: str = "") -> str:
    if override:
        return override.rstrip("/")
    try:
        out = subprocess.check_output(
            ["docker", "port", container, "8080/tcp"],
            text=True
        ).strip()
        if not out:
            return ""
        line = out.splitlines()[0].strip()
        port = line.rsplit(":", 1)[-1]
        if port.isdigit():
            return f"http://localhost:{port}"
    except Exception:
        return ""
    return ""


def post_webui_log(base: str, command: str, output: str, exit_code: int) -> None:
    if not base or not command:
        return
    payload = {
        "command": command,
        "source": "agent",
        "result": {
            "success": exit_code == 0,
            "output": output,
            "exit_code": exit_code
        },
        "is_error": exit_code != 0
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/log_command",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def extract_files_from_command(cmd: str) -> list:
    try:
        parts = shlex.split(cmd)
    except Exception:
        return []
    if not parts:
        return []
    head = parts[0]
    if head in ("return_txt", "return_img", "return_ori", "return_metadata"):
        if len(parts) >= 2:
            return [parts[1]]
    return []


IMAGE_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

VIDEO_EXT_TO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
}


def _guess_mime_type(path: str) -> str:
    try:
        mime_type, _ = mimetypes.guess_type(path)
    except Exception:
        mime_type = None
    if mime_type:
        return mime_type

    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT_TO_MIME:
        return IMAGE_EXT_TO_MIME[ext]
    if ext in VIDEO_EXT_TO_MIME:
        return VIDEO_EXT_TO_MIME[ext]
    return "application/octet-stream"


def _extract_response_text(response_obj: Dict[str, Any]) -> str:
    if not isinstance(response_obj, dict):
        return ""
    texts: List[str] = []
    for block in response_obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        txt = block.get("text")
        if isinstance(txt, str) and txt:
            texts.append(txt)
    return "\n".join(texts).strip()


def _parse_json_output(raw_output: str) -> Optional[Dict[str, Any]]:
    text = (raw_output or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        snippet = _extract_first_json(text)
        if not snippet:
            return None
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _decode_b64_to_temp_file(b64_payload: str, suffix: str) -> Optional[str]:
    try:
        payload = base64.b64decode(b64_payload, validate=False)
    except Exception:
        return None
    mm_dir = Path("/tmp/hippocamp_mm_assets")
    mm_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="mm_", suffix=suffix, dir=str(mm_dir))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        return tmp_path
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def _call_claude_api(
    api_key: str,
    api_url: str,
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    def _parse_retry_after_sec(raw_retry_after: Any) -> Optional[float]:
        if raw_retry_after is None:
            return None
        text = str(raw_retry_after).strip()
        if not text:
            return None
        try:
            sec = float(text)
            if sec > 0:
                return sec
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(text)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            wait = (dt - datetime.now(timezone.utc)).total_seconds()
            return wait if wait > 0 else None
        except Exception:
            return None

    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    raw_body = ""
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, timeout_sec)) as resp:
            raw_body = (resp.read() or b"").decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = (exc.read() or b"").decode("utf-8", errors="replace")
        snippet = body.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        retry_after = _parse_retry_after_sec(
            getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None
        )
        if retry_after is not None and retry_after > 0:
            snippet = f"{snippet} (retry_after={retry_after:.1f}s)"
        raise ClaudeAPIError(
            f"Anthropic API HTTP {exc.code}: {snippet}",
            status_code=exc.code,
            body=body,
            retry_after_sec=retry_after,
        ) from exc
    except urllib.error.URLError as exc:
        raise ClaudeAPIError(f"Anthropic API connection error: {exc}") from exc

    try:
        obj = json.loads(raw_body)
    except Exception as exc:
        raise ClaudeAPIError("Anthropic API returned non-JSON response") from exc
    if not isinstance(obj, dict):
        raise ClaudeAPIError("Anthropic API returned invalid response shape")
    if obj.get("type") == "error":
        err = obj.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(obj)
        raise ClaudeAPIError(f"Anthropic API error: {msg}", body=raw_body)
    return obj


def _count_tokens_url_from_messages_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    path = parsed.path or ""
    if path.endswith("/messages"):
        new_path = f"{path}/count_tokens"
    elif path.endswith("/messages/"):
        new_path = f"{path}count_tokens"
    else:
        marker = "/messages"
        idx = path.find(marker)
        if idx >= 0:
            new_path = f"{path[:idx + len(marker)]}/count_tokens"
        else:
            new_path = f"{path.rstrip('/')}/count_tokens"
    return urllib.parse.urlunparse(parsed._replace(path=new_path))


def _count_claude_input_tokens(
    api_key: str,
    api_url: str,
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    timeout_sec: float,
) -> Optional[int]:
    payload = {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
    }
    count_url = _count_tokens_url_from_messages_url(api_url)
    req = urllib.request.Request(
        count_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, timeout_sec)) as resp:
            raw = (resp.read() or b"").decode("utf-8", errors="replace")
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("input_tokens"), int):
            return obj.get("input_tokens")
    except Exception:
        return None
    return None


def _fallback_estimate_input_tokens(system_prompt: str, messages: List[Dict[str, Any]]) -> int:
    try:
        packed = json.dumps({"system": system_prompt, "messages": messages}, ensure_ascii=False)
    except Exception:
        packed = f"{system_prompt}\n{messages}"
    # Practical approximation for mixed text/json payloads.
    return max(1, int(len(packed) / 3.8))


def _prepare_claude_content(
    user_content: Any,
    max_media_items: int,
    max_b64_chars: int,
) -> List[Dict[str, Any]]:
    # Kept for CLI compatibility. Claude path no longer skips by b64 size.
    _ = max_b64_chars
    if not isinstance(user_content, dict) or user_content.get("kind") != "multimodal_followup":
        return [{"type": "text", "text": str(user_content)}]

    text = str(user_content.get("text") or "")
    image_items = user_content.get("image_files") or []
    video_items = user_content.get("video_files") or []
    uploaded_items = user_content.get("uploaded_files") or []
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    attached = 0
    skipped_non_image: List[str] = []

    for item in image_items:
        if max_media_items > 0 and attached >= max_media_items:
            break
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path:
            continue
        mime_type = str(item.get("mime_type") or "image/png")
        try:
            data = Path(path).read_bytes()
        except Exception:
            continue
        b64_payload = base64.b64encode(data).decode("ascii")
        if not b64_payload:
            continue
        parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_payload,
                },
            }
        )
        attached += 1

    for item in list(video_items) + list(uploaded_items):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        mime_type = str(item.get("mime_type") or _guess_mime_type(path))
        if not path:
            continue
        ext = Path(path).suffix.lower()
        # Claude supports image blocks well. For non-image media we keep
        # a textual hint to avoid request failures due unsupported MIME.
        if ext in IMAGE_EXT_TO_MIME:
            continue
        if len(skipped_non_image) < 8:
            skipped_non_image.append(f"{Path(path).name} ({mime_type})")

    if skipped_non_image:
        note = "Non-image media restored locally but not attached to Claude API: " + ", ".join(skipped_non_image)
        parts[0]["text"] = f"{parts[0]['text']}\n{note}"

    return parts


def _build_assistant_message_for_history(response_obj: Dict[str, Any], text: str) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    for block in response_obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        t = block.get("text")
        if isinstance(t, str) and t:
            blocks.append({"type": "text", "text": t})
    if not blocks:
        blocks = [{"type": "text", "text": text}]
    return {"role": "assistant", "content": blocks}


def _build_format_error_feedback(error: Exception) -> str:
    detail = str(error).strip() or "Invalid response format."
    if len(detail) > 240:
        detail = detail[:240] + "..."
    return (
        f"FORMAT_ERROR: {detail}\n"
        "Please respond with strict tags only and a valid JSON tool command when using <tool>.\n"
        "Allowed output:\n"
        "<think>...</think><tool>{\"name\":\"terminal\",\"arguments\":{\"command\":\"...\"}}</tool>\n"
        "OR\n"
        "<think>...</think><answer>...</answer><end>TERMINATE</end>\n"
        "No extra text."
    )


def _build_multimodal_followup(
    command: str,
    raw_output: str,
    text_output: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    try:
        parts = shlex.split(command)
    except Exception:
        return None, []
    if not parts:
        return None, []

    head = os.path.basename(parts[0]).lower()
    if head not in ("return_img", "img", "return_ori", "ori"):
        return None, []

    obj = _parse_json_output(raw_output)
    if not obj or not obj.get("success"):
        return None, []

    payload: Dict[str, Any] = {
        "kind": "multimodal_followup",
        "text": f"This is output of terminal: {text_output}",
        "image_files": [],
        "video_files": [],
        "uploaded_files": [],
    }
    restored_files: List[str] = []

    # Runtime may provide local asset references instead of large base64 payloads.
    local_asset_refs = obj.get("local_asset_refs") or []
    if isinstance(local_asset_refs, list):
        for ref in local_asset_refs:
            if not isinstance(ref, dict):
                continue
            local_path = str(ref.get("path") or "")
            if not local_path:
                continue
            if not os.path.exists(local_path):
                continue
            restored_files.append(local_path)
            ext = Path(local_path).suffix.lower()
            kind = str(ref.get("kind") or "").lower()
            if kind == "image" or ext in IMAGE_EXT_TO_MIME:
                payload["image_files"].append({
                    "path": local_path,
                    "mime_type": IMAGE_EXT_TO_MIME.get(ext, "image/png"),
                })
            elif kind == "video" or ext in VIDEO_EXT_TO_MIME:
                payload["video_files"].append({
                    "path": local_path,
                    "mime_type": VIDEO_EXT_TO_MIME.get(ext, "video/mp4"),
                })
            else:
                payload["uploaded_files"].append({
                    "path": local_path,
                    "mime_type": _guess_mime_type(local_path),
                })

    local_paths = obj.get("local_paths") or []
    if isinstance(local_paths, list):
        for local_path in local_paths:
            if not isinstance(local_path, str) or not local_path:
                continue
            if not os.path.exists(local_path):
                continue
            restored_files.append(local_path)
            ext = Path(local_path).suffix.lower()
            if ext in IMAGE_EXT_TO_MIME:
                payload["image_files"].append({
                    "path": local_path,
                    "mime_type": IMAGE_EXT_TO_MIME.get(ext, "image/png"),
                })
            elif ext in VIDEO_EXT_TO_MIME:
                payload["video_files"].append({
                    "path": local_path,
                    "mime_type": VIDEO_EXT_TO_MIME.get(ext, "video/mp4"),
                })
            else:
                payload["uploaded_files"].append({
                    "path": local_path,
                    "mime_type": _guess_mime_type(local_path),
                })

    if head in ("return_img", "img"):
        b64_list = obj.get("image_b64_list") or []
        if not b64_list and obj.get("image_b64"):
            b64_list = [obj.get("image_b64")]
        image_paths = obj.get("image_paths") or []

        for i, b64_payload in enumerate(b64_list):
            if not isinstance(b64_payload, str) or not b64_payload:
                continue
            src_path = image_paths[i] if i < len(image_paths) else ""
            suffix = Path(str(src_path)).suffix.lower() or ".png"
            local_file = _decode_b64_to_temp_file(b64_payload, suffix)
            if not local_file:
                continue
            restored_files.append(local_file)
            payload["image_files"].append({
                "path": local_file,
                "mime_type": IMAGE_EXT_TO_MIME.get(suffix, "image/png"),
            })

    if head in ("return_ori", "ori"):
        b64_payload = obj.get("file_b64")
        src_path = str(obj.get("file_path") or "")
        ext = Path(src_path).suffix.lower()
        if isinstance(b64_payload, str) and b64_payload:
            local_file = _decode_b64_to_temp_file(b64_payload, ext or ".bin")
            if local_file:
                restored_files.append(local_file)
                if ext in IMAGE_EXT_TO_MIME:
                    payload["image_files"].append({
                        "path": local_file,
                        "mime_type": IMAGE_EXT_TO_MIME[ext],
                    })
                elif ext in VIDEO_EXT_TO_MIME:
                    payload["video_files"].append({
                        "path": local_file,
                        "mime_type": VIDEO_EXT_TO_MIME[ext],
                    })
                else:
                    payload["uploaded_files"].append({
                        "path": local_file,
                        "mime_type": _guess_mime_type(src_path or local_file),
                    })

    if not payload["image_files"] and not payload["video_files"] and not payload["uploaded_files"]:
        return None, []
    return payload, restored_files


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _command_head(command: str) -> str:
    try:
        parts = shlex.split(command or "")
        if parts:
            return os.path.basename(parts[0]).lower()
    except Exception:
        pass
    return ""


def _transport_probe() -> Dict[str, Any]:
    return {
        "mode": "auto_chunk",
        "chunk_index": 99999,
        "chunk_total": 99999,
        "has_more": True,
        "repeat_same_command_for_next_chunk": True,
    }


def _attach_transport(obj: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    out = dict(obj)
    out["transport"] = {
        "mode": "auto_chunk",
        "chunk_index": idx,
        "chunk_total": total,
        "has_more": idx < total,
        "repeat_same_command_for_next_chunk": idx < total,
    }
    if idx < total:
        out["transport"]["next_hint"] = "repeat the exact same command to fetch next chunk"
    else:
        out["transport"]["next_hint"] = "all chunks delivered; choose next action"
    return out


def _fit_list_slice_under_budget(
    base_obj: Dict[str, Any],
    key: str,
    items: List[Any],
    start: int,
    budget: int,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    remaining = len(items) - start
    if remaining <= 0:
        return 0, None
    lo, hi = 1, remaining
    best_n = 0
    best_obj: Optional[Dict[str, Any]] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = dict(base_obj)
        cand[key] = items[start:start + mid]
        cand["transport"] = _transport_probe()
        cand_text = _json_pretty(cand)
        if len(cand_text) <= budget:
            best_n = mid
            best_obj = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best_n, best_obj


def _build_auto_chunks_for_large_output(
    command: str,
    raw_output: str,
    budget_chars: int,
    list_files_hard_cap: int = 0,
) -> List[str]:
    budget = max(256, int(budget_chars))
    text = raw_output if isinstance(raw_output, str) else str(raw_output)
    if len(text) <= budget:
        return []

    head = _command_head(command)
    if head not in ("list_files", "ls_files", "return_txt", "txt", "return_img", "img", "return_ori", "ori"):
        return []

    try:
        obj = json.loads(text)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []

    chunk_objs: List[Dict[str, Any]] = []

    # list_files: split .data list into max-fit chunks.
    if head in ("list_files", "ls_files"):
        data = obj.get("data")
        if not isinstance(data, list):
            return []
        all_items = data[: int(list_files_hard_cap)] if list_files_hard_cap and list_files_hard_cap > 0 else data
        start = 0
        while start < len(all_items):
            n, cand = _fit_list_slice_under_budget(obj, "data", all_items, start, budget)
            if n <= 0 or cand is None:
                one = str(all_items[start])
                cand = dict(obj)
                cand["data"] = [one[: max(16, budget // 6)] + ("..." if len(one) > max(16, budget // 6) else "")]
                n = 1
            cand["data_truncated"] = (start + n) < len(all_items)
            cand["data_omitted"] = len(all_items) - (start + n)
            cand["data_start_index"] = start
            cand["data_end_index"] = start + n - 1
            chunk_objs.append(cand)
            start += n

    # return_txt: split data.segments into max-fit chunks.
    elif head in ("return_txt", "txt"):
        data_obj = obj.get("data")
        segments = data_obj.get("segments") if isinstance(data_obj, dict) else None
        if not isinstance(segments, list):
            return []
        start = 0
        while start < len(segments):
            remaining = len(segments) - start
            lo, hi = 1, remaining
            best_n = 0
            best_obj: Optional[Dict[str, Any]] = None
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = dict(obj)
                cand_data = dict(data_obj)
                cand_data["segments"] = segments[start:start + mid]
                cand["data"] = cand_data
                cand["transport"] = _transport_probe()
                if len(_json_pretty(cand)) <= budget:
                    best_n = mid
                    best_obj = cand
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_n <= 0 or best_obj is None:
                seg = segments[start]
                seg_obj = dict(seg) if isinstance(seg, dict) else {"content": str(seg)}
                content = str(seg_obj.get("content") or "")
                # Fit partial content of one segment.
                lo, hi = 0, len(content)
                best_seg = ""
                while lo <= hi:
                    mid = (lo + hi) // 2
                    trial = dict(obj)
                    trial_data = dict(data_obj)
                    s = dict(seg_obj)
                    s["content"] = content[:mid] + ("...[segment truncated]" if mid < len(content) else "")
                    trial_data["segments"] = [s]
                    trial["data"] = trial_data
                    trial["transport"] = _transport_probe()
                    if len(_json_pretty(trial)) <= budget:
                        best_seg = s["content"]
                        lo = mid + 1
                    else:
                        hi = mid - 1
                seg_obj["content"] = best_seg if best_seg else content[: max(32, budget // 8)] + "...[segment truncated]"
                best_obj = dict(obj)
                cand_data = dict(data_obj)
                cand_data["segments"] = [seg_obj]
                best_obj["data"] = cand_data
                best_n = 1

            cand_data = best_obj.get("data") if isinstance(best_obj.get("data"), dict) else {}
            if isinstance(cand_data, dict):
                cand_data["segments_truncated"] = (start + best_n) < len(segments)
                cand_data["segments_omitted"] = len(segments) - (start + best_n)
                cand_data["segments_start_index"] = start
                cand_data["segments_end_index"] = start + best_n - 1
                best_obj["data"] = cand_data
            chunk_objs.append(best_obj)
            start += best_n

    # return_img: split images; if single b64 too large, replace with local asset ref.
    elif head in ("return_img", "img"):
        image_paths = obj.get("image_paths")
        image_b64_list = obj.get("image_b64_list")
        if not isinstance(image_paths, list):
            image_paths = [obj.get("image_path")] if obj.get("image_path") else []
        if not isinstance(image_b64_list, list):
            image_b64_list = [obj.get("image_b64")] if obj.get("image_b64") else []
        total_images = max(len(image_paths), len(image_b64_list))
        if total_images <= 0:
            return []
        start = 0
        while start < total_images:
            remaining = total_images - start
            lo, hi = 1, remaining
            best_n = 0
            best_obj = None
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = dict(obj)
                cand["image_paths"] = image_paths[start:start + mid]
                if image_b64_list:
                    cand["image_b64_list"] = image_b64_list[start:start + mid]
                cand["transport"] = _transport_probe()
                if len(_json_pretty(cand)) <= budget:
                    best_n = mid
                    best_obj = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_n <= 0 or best_obj is None:
                cand = dict(obj)
                one_path = image_paths[start] if start < len(image_paths) else ""
                one_b64 = image_b64_list[start] if start < len(image_b64_list) else ""
                suffix = Path(str(one_path)).suffix.lower() or ".png"
                local_ref = None
                if isinstance(one_b64, str) and one_b64:
                    local_file = _decode_b64_to_temp_file(one_b64, suffix)
                    if local_file:
                        local_ref = {"kind": "image", "path": local_file}
                cand["image_paths"] = [one_path] if one_path else []
                cand["image_b64_list"] = [f"<base64:{len(one_b64)} chars>"] if isinstance(one_b64, str) and one_b64 else []
                if local_ref:
                    cand["local_asset_refs"] = [local_ref]
                best_n = 1
                best_obj = cand
            best_obj["images_truncated"] = (start + best_n) < total_images
            best_obj["images_omitted"] = total_images - (start + best_n)
            best_obj["images_start_index"] = start
            best_obj["images_end_index"] = start + best_n - 1
            chunk_objs.append(best_obj)
            start += best_n

    # return_ori: single payload; if too large, replace b64 with local ref.
    elif head in ("return_ori", "ori"):
        one = dict(obj)
        b64_payload = one.get("file_b64")
        src_path = str(one.get("file_path") or "")
        ext = Path(src_path).suffix.lower() or ".bin"
        if isinstance(b64_payload, str) and b64_payload and len(_json_pretty(one)) > budget:
            local_file = _decode_b64_to_temp_file(b64_payload, ext)
            one["file_b64"] = f"<base64:{len(b64_payload)} chars>"
            if local_file:
                one["local_asset_refs"] = [{"kind": "file", "path": local_file}]
        chunk_objs.append(one)

    if not chunk_objs:
        return []

    texts: List[str] = []
    total = len(chunk_objs)
    for i, c in enumerate(chunk_objs, start=1):
        with_transport = _attach_transport(c, i, total)
        cand_text = _json_pretty(with_transport)
        if len(cand_text) <= budget:
            texts.append(cand_text)
            continue

        # Try to preserve useful payload by shrinking arrays/content first.
        if head in ("list_files", "ls_files"):
            tuned = dict(with_transport)
            data = tuned.get("data")
            if isinstance(data, list):
                while len(data) > 1 and len(_json_pretty(tuned)) > budget:
                    data = data[:-1]
                    tuned["data"] = data
                    tuned["data_truncated"] = True
                if data and len(_json_pretty(tuned)) > budget:
                    one = str(data[0])
                    lo, hi = 0, len(one)
                    best = ""
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        trial = dict(tuned)
                        trial["data"] = [one[:mid] + ("..." if mid < len(one) else "")]
                        if len(_json_pretty(trial)) <= budget:
                            best = trial["data"][0]
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    if best:
                        tuned["data"] = [best]
                tuned_text = _json_pretty(tuned)
                if len(tuned_text) <= budget:
                    texts.append(tuned_text)
                    continue

        if head in ("return_txt", "txt"):
            tuned = dict(with_transport)
            data_obj = tuned.get("data")
            if isinstance(data_obj, dict):
                segments = data_obj.get("segments")
                if isinstance(segments, list):
                    while len(segments) > 1 and len(_json_pretty(tuned)) > budget:
                        segments = segments[:-1]
                        data_obj["segments"] = segments
                        data_obj["segments_truncated"] = True
                        tuned["data"] = data_obj
                    if segments and len(_json_pretty(tuned)) > budget:
                        seg0 = segments[0]
                        segd = dict(seg0) if isinstance(seg0, dict) else {"content": str(seg0)}
                        raw_content = str(segd.get("content") or "")
                        lo, hi = 0, len(raw_content)
                        best = ""
                        while lo <= hi:
                            mid = (lo + hi) // 2
                            trial = dict(tuned)
                            trial_data = dict(data_obj)
                            s = dict(segd)
                            s["content"] = raw_content[:mid] + ("...[segment truncated]" if mid < len(raw_content) else "")
                            trial_data["segments"] = [s]
                            trial["data"] = trial_data
                            if len(_json_pretty(trial)) <= budget:
                                best = s["content"]
                                lo = mid + 1
                            else:
                                hi = mid - 1
                        if best:
                            segd["content"] = best
                            data_obj["segments"] = [segd]
                            tuned["data"] = data_obj
                tuned_text = _json_pretty(tuned)
                if len(tuned_text) <= budget:
                    texts.append(tuned_text)
                    continue

        # Final safety fallback per chunk.
        minimal = {
            "success": bool(with_transport.get("success", True)),
            "transport": with_transport.get("transport"),
            "truncated": True,
            "note": "chunk still exceeds budget; narrow command or fetch next chunk",
        }
        if isinstance(with_transport, dict):
            for k in ("file_path", "image_paths", "local_asset_refs"):
                if k in with_transport:
                    minimal[k] = with_transport[k]
        texts.append(_json_pretty(minimal))
    return texts


def _compact_terminal_output_for_prompt(
    command: str,
    raw_output: str,
    max_chars: int,
    list_files_items: int = 0,
) -> str:
    cmd_head = ""
    try:
        parts = shlex.split(command or "")
        if parts:
            cmd_head = os.path.basename(parts[0]).lower()
    except Exception:
        cmd_head = ""
    is_list_files_cmd = cmd_head in ("list_files", "ls_files")
    is_return_txt_cmd = cmd_head in ("return_txt", "txt")

    text = raw_output if isinstance(raw_output, str) else str(raw_output)
    stripped = text.strip()
    budget = max(256, int(max_chars))
    if not stripped or (not stripped.startswith("{") and not stripped.startswith("[")):
        if len(text) <= budget:
            return text
        return text[:budget] + "\n...[truncated]"
    try:
        obj = json.loads(text)
    except Exception:
        if len(text) <= budget:
            return text
        return text[:budget] + "\n...[truncated]"

    changed = False

    def _replace_large_b64(v: Any) -> Tuple[Any, bool]:
        if isinstance(v, str) and len(v) > 512:
            return f"<base64:{len(v)} chars>", True
        if isinstance(v, list):
            large = [len(x) for x in v if isinstance(x, str) and len(x) > 512]
            if large:
                return [f"<base64:{n} chars>" for n in large[:8]], True
        return v, False

    if isinstance(obj, dict):
        for b64_key in ("image_b64", "file_b64", "image_b64_list"):
            if b64_key in obj:
                new_val, did = _replace_large_b64(obj.get(b64_key))
                if did:
                    obj[b64_key] = new_val
                    changed = True

        # 1) list_files: maximize number of paths under current budget.
        if is_list_files_cmd:
            data = obj.get("data")
            if isinstance(data, list):
                if list_files_items and list_files_items > 0:
                    data = data[: int(list_files_items)]
                lo, hi = 0, len(data)
                best_text = ""
                best_n = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand = dict(obj)
                    cand["data"] = data[:mid]
                    if mid < len(data):
                        cand["data_truncated"] = True
                        cand["data_omitted"] = len(data) - mid
                    cand_text = _json_pretty(cand)
                    if len(cand_text) <= budget:
                        best_text = cand_text
                        best_n = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best_text:
                    if best_n < len(data):
                        changed = True
                    return best_text

        # 2) return_txt: maximize segments under budget, then partially include next segment.
        if is_return_txt_cmd:
            data_obj = obj.get("data")
            segments = data_obj.get("segments") if isinstance(data_obj, dict) else None
            if isinstance(segments, list):
                lo, hi = 0, len(segments)
                best_obj = None
                best_n = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand = dict(obj)
                    cand_data = dict(data_obj)
                    cand_data["segments"] = segments[:mid]
                    if mid < len(segments):
                        cand_data["segments_truncated"] = True
                        cand_data["segments_omitted"] = len(segments) - mid
                    cand["data"] = cand_data
                    cand_text = _json_pretty(cand)
                    if len(cand_text) <= budget:
                        best_obj = cand
                        best_n = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1

                if best_obj is not None:
                    # Try to fill remaining budget with partial content of next segment.
                    if best_n < len(segments):
                        next_seg = segments[best_n]
                        if isinstance(next_seg, dict) and isinstance(next_seg.get("content"), str):
                            base_data = dict(best_obj.get("data") or {})
                            base_segments = list(base_data.get("segments") or [])
                            raw_content = next_seg.get("content") or ""
                            lo2, hi2 = 0, len(raw_content)
                            best_partial = None
                            while lo2 <= hi2:
                                mid2 = (lo2 + hi2) // 2
                                seg_obj = dict(next_seg)
                                if mid2 < len(raw_content):
                                    seg_obj["content"] = raw_content[:mid2] + "...[segment truncated]"
                                else:
                                    seg_obj["content"] = raw_content
                                cand = dict(best_obj)
                                cand_data = dict(base_data)
                                cand_data["segments"] = base_segments + [seg_obj]
                                cand_data["segments_truncated"] = True
                                cand_data["segments_omitted"] = len(segments) - (best_n + 1)
                                cand["data"] = cand_data
                                cand_text = _json_pretty(cand)
                                if len(cand_text) <= budget:
                                    best_partial = cand_text
                                    lo2 = mid2 + 1
                                else:
                                    hi2 = mid2 - 1
                            if best_partial:
                                return best_partial
                    return _json_pretty(best_obj)

        # 3) return_img/return_ori: keep as many asset references as budget allows.
        if cmd_head in ("return_img", "img"):
            image_paths = obj.get("image_paths")
            image_b64_list = obj.get("image_b64_list")
            if isinstance(image_paths, list):
                b64_list = image_b64_list if isinstance(image_b64_list, list) else None
                lo, hi = 0, len(image_paths)
                best_text = ""
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand = dict(obj)
                    cand["image_paths"] = image_paths[:mid]
                    if b64_list is not None:
                        cand["image_b64_list"] = b64_list[:mid]
                    if mid < len(image_paths):
                        cand["images_truncated"] = True
                        cand["images_omitted"] = len(image_paths) - mid
                    cand_text = _json_pretty(cand)
                    if len(cand_text) <= budget:
                        best_text = cand_text
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best_text:
                    return best_text

    rendered = _json_pretty(obj) if changed else text
    if len(rendered) <= budget:
        return rendered

    # Final fallback: keep valid JSON rather than raw broken truncation.
    fallback_obj = {
        "success": isinstance(obj, dict) and bool(obj.get("success", True)),
        "truncated": True,
        "original_chars": len(text),
        "note": "terminal output exceeds budget; rerun narrower command or use paging/filters.",
    }
    fallback_text = _json_pretty(fallback_obj)
    if len(fallback_text) <= budget:
        return fallback_text
    return fallback_text[:budget]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True, help="Docker container name or ID")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--model", default=CLAUDE_DEFAULT_MODEL, help="Claude model name")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--log-json", default="result/claude_docker_session.json")
    parser.add_argument("--ensure-webui", action="store_true", help="Start WebUI before loop")
    parser.add_argument("--docker-user", default="", help="Optional docker exec user")
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--format-retries", type=int, default=2, help="Retries for format violations")
    parser.add_argument("--strict-parse", action="store_true", help="Disable relaxed parsing fallback")
    parser.add_argument("--webui-base", default="", help="Override WebUI base URL, e.g. http://localhost:8083")
    parser.add_argument("--api-retries", type=int, default=4, help="Retries on 429/temporary API errors")
    parser.add_argument("--api-retry-base", type=float, default=1.5, help="Base backoff seconds")
    parser.add_argument("--api-retry-max", type=float, default=20.0, help="Max backoff seconds")
    parser.add_argument("--api-timeout-sec", type=float, default=120.0, help="Claude API timeout seconds")
    parser.add_argument("--tpm-budget", type=int, default=9500, help="Safety input-token budget per rolling window.")
    parser.add_argument(
        "--tpm-wait-threshold",
        type=int,
        default=8500,
        help="Wait before sending if rolling window + next request exceeds this threshold.",
    )
    parser.add_argument("--tpm-window-sec", type=float, default=60.0, help="Rolling window size in seconds.")
    parser.add_argument(
        "--disable-count-tokens",
        action="store_true",
        help="Disable /messages/count_tokens and use fallback estimation only.",
    )
    parser.add_argument("--api-url", default="", help="Override Claude API URL")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Claude max output tokens")
    parser.add_argument(
        "--list-files-preview-items",
        type=int,
        default=0,
        help="Optional hard cap for list_files candidates before budget fitting (0 = no hard cap).",
    )
    parser.add_argument(
        "--disable-output-compaction",
        action="store_true",
        help="Disable terminal JSON compaction before feeding back to model.",
    )
    parser.add_argument(
        "--disable-context-compaction",
        action="store_true",
        help="Disable context compaction by TPM threshold (quality-first, may increase 429 risk).",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=8,
        help="Keep at most N previous user/assistant turns in request context (0 = unlimited).",
    )
    parser.add_argument(
        "--max-mm-items",
        type=int,
        default=2,
        help="Max media items to attach in one follow-up (0 = attach all restored items).",
    )
    parser.add_argument(
        "--max-mm-b64-chars",
        type=int,
        default=150000000,
        help="Retained for CLI compatibility; Claude path does not skip by base64 size.",
    )
    parser.add_argument(
        "--media-upload-wait-sec",
        type=int,
        default=180,
        help="Retained for CLI compatibility; ignored by Claude API path.",
    )
    parser.add_argument(
        "--prompt-config",
        default="config0",
        choices=AVAILABLE_CONFIGS,
        help="Auxiliary-function prompt config to use (config0/config1/config2/config3).",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=AVAILABLE_PROMPT_VERSIONS,
        help="Main prompt version (e.g. v1, v2).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not found in .env or environment")
    model_name = args.model
    env_model = os.environ.get("CLAUDE_MODEL", "")
    if env_model and args.model == CLAUDE_DEFAULT_MODEL:
        model_name = env_model
    api_url = (args.api_url or os.environ.get("ANTHROPIC_API_URL", "") or CLAUDE_DEFAULT_API_URL).strip()

    flags_by_config = {
        "config0": (1, 1),
        "config1": (0, 0),
        "config2": (0, 1),
        "config3": (1, 0),
    }
    forced_flags = flags_by_config.get(args.prompt_config)

    prompt = build_prompt(
        version=args.prompt_version,
        auxiliary_functions_block=get_auxiliary_functions_block(args.prompt_config),
    )
    conversation: List[Dict[str, Any]] = []
    token_window: Deque[Tuple[float, int]] = deque()
    auto_chunk_sessions: Dict[str, Dict[str, Any]] = {}

    def _is_retryable_error(err: Exception) -> bool:
        if isinstance(err, ClaudeAPIError):
            if err.status_code in (408, 409, 425, 429):
                return True
            if err.status_code is not None and err.status_code >= 500:
                return True
            msg = (err.body or str(err)).lower()
            if any(k in msg for k in ("rate", "overload", "timeout", "tempor", "unavailable", "connection")):
                return True
            return False
        msg = str(err).lower()
        if "rate" in msg or "quota" in msg:
            return True
        if "429" in msg:
            return True
        if "timeout" in msg or "tempor" in msg or "unavailable" in msg or "connection" in msg:
            return True
        return False

    def _window_threshold() -> int:
        budget = max(1, int(args.tpm_budget))
        wait_threshold = max(1, int(args.tpm_wait_threshold))
        return min(budget, wait_threshold)

    def _compact_messages_to_threshold(messages: List[Dict[str, Any]], threshold: int) -> Tuple[List[Dict[str, Any]], int, int]:
        # Keep the original question (first user message) and latest user feedback,
        # compacting middle history first.
        if not messages:
            return messages, 0, 0
        trimmed = list(messages)
        estimated = _fallback_estimate_input_tokens(prompt, trimmed)
        dropped = 0
        protect_first_user = (
            len(trimmed) >= 2
            and isinstance(trimmed[0], dict)
            and trimmed[0].get("role") == "user"
        )
        while estimated > threshold:
            if protect_first_user and len(trimmed) > 2:
                trimmed.pop(1)
                dropped += 1
            elif not protect_first_user and len(trimmed) > 1:
                trimmed.pop(0)
                dropped += 1
            else:
                break
            estimated = _fallback_estimate_input_tokens(prompt, trimmed)
        return trimmed, estimated, dropped

    def _prune_token_window(now_ts: float) -> None:
        cutoff = now_ts - max(1.0, float(args.tpm_window_sec))
        while token_window and token_window[0][0] <= cutoff:
            token_window.popleft()

    def _token_window_sum(now_ts: Optional[float] = None) -> int:
        ts = now_ts if now_ts is not None else time.time()
        _prune_token_window(ts)
        return sum(tok for _, tok in token_window)

    def _record_input_tokens(tokens: int) -> None:
        tok = int(tokens) if isinstance(tokens, int) else 0
        if tok <= 0:
            return
        now_ts = time.time()
        _prune_token_window(now_ts)
        token_window.append((now_ts, tok))

    def _wait_for_tpm_window(next_tokens: int) -> None:
        next_tok = max(1, int(next_tokens))
        threshold = _window_threshold()
        if next_tok > threshold:
            # Single request already exceeds local threshold.
            # Wait until local rolling window drains to avoid compounding bursts.
            while True:
                now_ts = time.time()
                used = _token_window_sum(now_ts)
                if used <= 0:
                    return
                oldest_ts = token_window[0][0]
                sleep_for = max(0.2, oldest_ts + float(args.tpm_window_sec) - now_ts)
                time.sleep(sleep_for)
            return
        while True:
            now_ts = time.time()
            used = _token_window_sum(now_ts)
            if used + next_tok <= threshold:
                return
            if not token_window:
                sleep_for = max(0.2, float(args.tpm_window_sec))
            else:
                oldest_ts = token_window[0][0]
                sleep_for = max(0.2, oldest_ts + float(args.tpm_window_sec) - now_ts)
            time.sleep(sleep_for)

    def _feedback_output_char_budget() -> int:
        # Dynamic budget per turn: use as much as possible without likely exceeding
        # the local request budget. args.max_output_chars remains the hard cap.
        hard_cap = max(256, int(args.max_output_chars))
        try:
            threshold = _window_threshold()
            if args.max_history_turns > 0:
                max_history_messages = max(0, int(args.max_history_turns) * 2)
                history = conversation[-max_history_messages:] if max_history_messages else []
            else:
                history = conversation
            probe_user = {"role": "user", "content": [{"type": "text", "text": "This is output of terminal: "}]}
            base_tokens = _fallback_estimate_input_tokens(prompt, history + [probe_user])
            # Reserve room for structural tokens + model completion.
            reserve_tokens = max(256, int(max(128, args.max_tokens) * 0.75))
            available_tokens = threshold - base_tokens - reserve_tokens
            if available_tokens <= 64:
                return min(hard_cap, 800)
            dynamic_chars = int(available_tokens * 3.8)
            return max(800, min(hard_cap, dynamic_chars))
        except Exception:
            return hard_cap

    def _auto_chunk_key(command: str) -> str:
        return (command or "").strip()

    def _next_auto_chunk_if_any(command: str) -> Optional[str]:
        key = _auto_chunk_key(command)
        if not key:
            return None
        session = auto_chunk_sessions.get(key)
        if not session:
            return None
        chunks = session.get("chunks") or []
        next_idx = int(session.get("next_idx") or 0)
        if next_idx < len(chunks):
            out = chunks[next_idx]
            session["next_idx"] = next_idx + 1
            auto_chunk_sessions[key] = session
            return out
        # Repeating after completion returns an explicit done signal to avoid loops.
        return _json_pretty({
            "success": True,
            "transport": {
                "mode": "auto_chunk",
                "has_more": False,
                "already_complete": True,
                "repeat_same_command_for_next_chunk": False,
                "next_hint": "all chunks already delivered; choose next action",
            },
        })

    def _prime_auto_chunks_if_needed(command: str, raw_output: str, budget_chars: int) -> Tuple[str, bool]:
        chunks = _build_auto_chunks_for_large_output(
            command=command,
            raw_output=raw_output,
            budget_chars=budget_chars,
            list_files_hard_cap=args.list_files_preview_items,
        )
        if not chunks:
            # Reset stale session once fresh full payload is available.
            auto_chunk_sessions.pop(_auto_chunk_key(command), None)
            return raw_output, False
        key = _auto_chunk_key(command)
        auto_chunk_sessions[key] = {"chunks": chunks, "next_idx": 1}
        return chunks[0], True

    def send_to_model(user_content: Any) -> Tuple[str, float, Dict[str, Any], Dict[str, Any]]:
        attempt = 0
        start = time.time()
        user_message = {
            "role": "user",
            "content": _prepare_claude_content(
                user_content=user_content,
                max_media_items=args.max_mm_items,
                max_b64_chars=args.max_mm_b64_chars,
            ),
        }
        if args.max_history_turns > 0:
            max_history_messages = max(0, int(args.max_history_turns) * 2)
            history = conversation[-max_history_messages:] if max_history_messages else []
        else:
            history = conversation
        messages_payload = history + [user_message]
        estimated_input_tokens = (
            _fallback_estimate_input_tokens(prompt, messages_payload)
            if args.disable_count_tokens
            else _count_claude_input_tokens(
                api_key=api_key,
                api_url=api_url,
                model_name=model_name,
                system_prompt=prompt,
                messages=messages_payload,
                timeout_sec=max(1.0, args.api_timeout_sec),
            )
        )
        if not isinstance(estimated_input_tokens, int) or estimated_input_tokens <= 0:
            estimated_input_tokens = _fallback_estimate_input_tokens(prompt, messages_payload)
        threshold = _window_threshold()
        if (
            not args.disable_context_compaction
            and estimated_input_tokens > threshold
            and len(messages_payload) > 1
        ):
            original = estimated_input_tokens
            messages_payload, compacted_estimate, dropped = _compact_messages_to_threshold(messages_payload, threshold)
            if dropped > 0:
                estimated_input_tokens = compacted_estimate
                print(
                    f"[claude] compacted context dropped_messages={dropped} est_input_tokens={original}->{estimated_input_tokens}",
                    file=sys.stderr,
                    flush=True,
                )
        while True:
            try:
                _wait_for_tpm_window(estimated_input_tokens)
                response_obj = _call_claude_api(
                    api_key=api_key,
                    api_url=api_url,
                    model_name=model_name,
                    system_prompt=prompt,
                    messages=messages_payload,
                    max_tokens=max(128, args.max_tokens),
                    timeout_sec=max(1.0, args.api_timeout_sec),
                )
                usage = response_obj.get("usage") or {}
                actual_input_tokens = usage.get("input_tokens")
                if not isinstance(actual_input_tokens, int) or actual_input_tokens <= 0:
                    actual_input_tokens = estimated_input_tokens
                _record_input_tokens(actual_input_tokens)
                text = _extract_response_text(response_obj)
                assistant_message = _build_assistant_message_for_history(response_obj, text)
                return text, time.time() - start, user_message, assistant_message
            except Exception as e:
                if isinstance(e, ClaudeAPIError) and e.status_code == 429:
                    # Conservatively account attempted input to avoid rapid repeated bursts.
                    _record_input_tokens(estimated_input_tokens)
                if attempt >= args.api_retries or not _is_retryable_error(e):
                    raise
                backoff = min(args.api_retry_max, args.api_retry_base * (2 ** attempt))
                retry_after = getattr(e, "retry_after_sec", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    backoff = max(backoff, min(args.api_retry_max, float(retry_after)))
                backoff *= 1.0 + random.uniform(0, 0.25)
                print(
                    f"[claude] retry attempt={attempt + 1}/{args.api_retries} sleep={backoff:.1f}s error={type(e).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(backoff)
                attempt += 1

    start_ts = time.time()
    log = {
        "question": args.question,
        "answer": "",
        "prompt_config": args.prompt_config,
        "prompt_version": args.prompt_version,
        "tpm_budget": int(args.tpm_budget),
        "tpm_wait_threshold": int(args.tpm_wait_threshold),
        "tpm_window_sec": float(args.tpm_window_sec),
        "files_touched": [],
        "steps": {},
        "start_time": datetime.now(timezone.utc).isoformat(),
        "total_time_sec": None,
    }
    files_touched = set()
    webui_base = detect_webui_base(args.container, args.webui_base)

    if args.ensure_webui:
        _out, _rc, _t = run_docker_command(args.container, "webui", user=args.docker_user)
        if not webui_base:
            webui_base = detect_webui_base(args.container, args.webui_base)

    if forced_flags is not None:
        _out, _rc, _t = run_docker_command(
            args.container,
            f"set_flags {forced_flags[0]} {forced_flags[1]}",
            user=args.docker_user,
        )

    pending_user_text = f"Question: {args.question}"
    pending_user_content: Any = pending_user_text
    grep_no_match_streak = 0
    last_grep_pattern = ""

    for step_idx in range(1, args.max_steps + 1):
        step_key = f"step_{step_idx}"
        sent_msg = pending_user_text
        sent_content = pending_user_content
        agent_text = ""
        agent_time = 0.0
        action_obj = None
        format_errors = []
        parsed_user_message: Optional[Dict[str, Any]] = None
        parsed_assistant_message: Optional[Dict[str, Any]] = None

        for attempt in range(args.format_retries + 1):
            agent_text, agent_time, candidate_user_message, candidate_assistant_message = send_to_model(sent_content)
            try:
                action_obj = parse_response(agent_text, relaxed=not args.strict_parse)
                validate_action(action_obj)
                parsed_user_message = candidate_user_message
                parsed_assistant_message = candidate_assistant_message
                break
            except Exception as e:
                format_errors.append({
                    "attempt": attempt + 1,
                    "agent_response": agent_text,
                    "error": str(e),
                })
                feedback = _build_format_error_feedback(e)
                if attempt >= args.format_retries:
                    log["steps"][step_key] = {
                        "agent_response": agent_text,
                        "agent_time_sec": agent_time,
                        "user_input": sent_msg,
                        "error": f"parse_error: {e}",
                        "format_errors": format_errors,
                    }
                    pending_user_text = feedback
                    pending_user_content = feedback
                    action_obj = None
                    break
                sent_msg = feedback
                sent_content = feedback

        if action_obj is None:
            continue

        # Commit conversation turn only after parse + validate succeeds.
        if parsed_user_message is not None and parsed_assistant_message is not None:
            conversation.append(parsed_user_message)
            conversation.append(parsed_assistant_message)

        if action_obj["action"] == "answer":
            log["answer"] = action_obj.get("answer", "")
            log["steps"][step_key] = {
                "agent_response": agent_text,
                "agent_time_sec": agent_time,
                "user_input": sent_msg,
                "command": "",
                "docker_time_sec": 0,
                "terminal_output": "",
                "exit_code": 0,
                "format_errors": format_errors,
            }
            break

        cmd_raw = action_obj.get("command", "").strip()
        cmd = normalize_command(cmd_raw)
        if cmd.startswith("set_flags"):
            correction = "set_flags is not allowed. Select prompt config instead (config0/config1/config2/config3)."
            log["steps"][step_key] = {
                "agent_response": agent_text,
                "agent_time_sec": agent_time,
                "user_input": sent_msg,
                "command": cmd,
                "command_raw": cmd_raw if cmd_raw and cmd_raw != cmd else "",
                "docker_time_sec": 0,
                "terminal_output": "ERROR: set_flags is not allowed",
                "exit_code": 1,
                "error": "set_flags_not_allowed",
            }
            pending_user_text = correction
            pending_user_content = correction
            continue

        policy_error = command_policy_error(cmd)
        if policy_error:
            log["steps"][step_key] = {
                "agent_response": agent_text,
                "agent_time_sec": agent_time,
                "user_input": sent_msg,
                "command": cmd,
                "command_raw": cmd_raw if cmd_raw and cmd_raw != cmd else "",
                "docker_time_sec": 0,
                "terminal_output": f"ERROR: {policy_error}",
                "exit_code": 1,
                "error": "command_policy_blocked",
            }
            pending_user_text = policy_error
            pending_user_content = policy_error
            continue

        if forced_flags is not None:
            try:
                parts = shlex.split(cmd)
            except Exception:
                parts = []
            if parts:
                head = parts[0]
                if forced_flags[0] == 0 and head in ("return_txt", "txt"):
                    correction = "return_txt is disabled by config. Choose a config that enables it."
                    log["steps"][step_key] = {
                        "agent_response": agent_text,
                        "agent_time_sec": agent_time,
                        "user_input": sent_msg,
                        "command": cmd,
                        "command_raw": cmd_raw if cmd_raw and cmd_raw != cmd else "",
                        "docker_time_sec": 0,
                        "terminal_output": "ERROR: return_txt is disabled by config",
                        "exit_code": 1,
                        "error": "return_txt_disabled",
                    }
                    pending_user_text = correction
                    pending_user_content = correction
                    continue
                if forced_flags[1] == 0 and head in ("return_img", "img"):
                    correction = "return_img is disabled by config. Choose a config that enables it."
                    log["steps"][step_key] = {
                        "agent_response": agent_text,
                        "agent_time_sec": agent_time,
                        "user_input": sent_msg,
                        "command": cmd,
                        "command_raw": cmd_raw if cmd_raw and cmd_raw != cmd else "",
                        "docker_time_sec": 0,
                        "terminal_output": "ERROR: return_img is disabled by config",
                        "exit_code": 1,
                        "error": "return_img_disabled",
                    }
                    pending_user_text = correction
                    pending_user_content = correction
                    continue

        touched = extract_files_from_command(cmd)
        for f in touched:
            files_touched.add(f)

        output_budget_chars = _feedback_output_char_budget()
        from_auto_chunk = False
        continued_chunk = _next_auto_chunk_if_any(cmd)
        if continued_chunk is not None:
            output_raw = continued_chunk
            exit_code = 0
            docker_time = 0.0
            from_auto_chunk = True
        else:
            output_raw, exit_code, docker_time = run_docker_command(args.container, cmd, user=args.docker_user)
            if not args.disable_output_compaction:
                output_raw, from_auto_chunk = _prime_auto_chunks_if_needed(cmd, output_raw, output_budget_chars)
        is_grep_no_match = _is_grep_no_match(cmd, output_raw, exit_code)
        if is_grep_no_match and not (output_raw or "").strip():
            output_raw = "(no matches)"

        cmd_head_for_output = _command_head(cmd)
        if from_auto_chunk:
            output_for_log = output_raw if isinstance(output_raw, str) else str(output_raw)
        elif (
            cmd_head_for_output in ("list_files", "ls_files", "return_txt", "txt", "return_img", "img", "return_ori", "ori")
            and isinstance(output_raw, str)
            and len(output_raw) <= output_budget_chars
        ):
            # For supported high-volume commands, keep complete raw output when it already fits.
            output_for_log = output_raw
        elif args.disable_output_compaction:
            output_for_log = output_raw if isinstance(output_raw, str) else str(output_raw)
            if len(output_for_log) > output_budget_chars:
                output_for_log = output_for_log[:output_budget_chars] + "\n...[truncated]"
        else:
            output_for_log = _compact_terminal_output_for_prompt(
                cmd,
                output_raw,
                max_chars=output_budget_chars,
                list_files_items=args.list_files_preview_items,
            )
            if len(output_for_log) > output_budget_chars:
                output_for_log = output_for_log[:output_budget_chars] + "\n...[truncated]"

        if is_grep_no_match:
            grep_pattern = _extract_grep_pattern(cmd) or "__grep__"
            if grep_pattern == last_grep_pattern:
                grep_no_match_streak += 1
            else:
                last_grep_pattern = grep_pattern
                grep_no_match_streak = 1
        else:
            grep_no_match_streak = 0
            last_grep_pattern = ""

        mm_content, restored_files = _build_multimodal_followup(
            command=cmd,
            raw_output=output_raw,
            text_output=output_for_log,
        )

        log["steps"][step_key] = {
            "agent_response": agent_text,
            "agent_time_sec": agent_time,
            "user_input": sent_msg,
            "command": cmd,
            "command_raw": cmd_raw if cmd_raw and cmd_raw != cmd else "",
            "docker_time_sec": docker_time,
            "terminal_output": output_for_log,
            "exit_code": exit_code,
            "format_errors": format_errors,
            "restored_multimodal_files": restored_files,
        }

        post_webui_log(webui_base, cmd, output_for_log, exit_code)

        pending_user_text = f"This is output of terminal: {output_for_log}"
        if is_grep_no_match and grep_no_match_streak >= 4:
            notice = (
                "\nNotice: repeated grep no-match results for the same pattern. "
                "Do not continue brute-force retries. "
                "Revise strategy by narrowing candidate files via filenames/path hints, "
                "broadening pattern variants (case-insensitive, spacing/punctuation variants), "
                "or inspecting one likely file with return_txt before batch searching."
            )
            pending_user_text += notice
            if mm_content and isinstance(mm_content, dict):
                mm_content["text"] = f"{mm_content.get('text', '')}{notice}"

        if mm_content:
            pending_user_content = mm_content
        else:
            pending_user_content = pending_user_text

    if not log.get("answer"):
        forced_final_user = (
            "You have reached the step limit. Based on gathered evidence, provide your best final answer now. "
            "Do not call tools. Output strict tags only:\n"
            "<think>...</think><answer>...</answer><end>TERMINATE</end>"
        )
        try:
            final_text, final_time, _final_user_msg, _final_assistant_msg = send_to_model(forced_final_user)
            forced_obj = parse_response(final_text, relaxed=not args.strict_parse)
            validate_action(forced_obj)
            if forced_obj.get("action") == "answer":
                log["answer"] = forced_obj.get("answer", "")
                log["steps"]["step_final"] = {
                    "agent_response": final_text,
                    "agent_time_sec": final_time,
                    "user_input": forced_final_user,
                    "command": "",
                    "docker_time_sec": 0,
                    "terminal_output": "",
                    "exit_code": 0,
                    "forced_final": True,
                }
            else:
                log["steps"]["step_final"] = {
                    "agent_response": final_text,
                    "agent_time_sec": final_time,
                    "user_input": forced_final_user,
                    "error": "forced_final_not_answer",
                    "forced_final": True,
                }
        except Exception as e:
            log["steps"]["step_final"] = {
                "user_input": forced_final_user,
                "error": f"forced_final_error: {e}",
                "forced_final": True,
            }

    log["files_touched"] = sorted(files_touched)
    log["total_time_sec"] = round(time.time() - start_ts, 3)

    os.makedirs(os.path.dirname(args.log_json) or ".", exist_ok=True)
    with open(args.log_json, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    if log["answer"]:
        print("Final Answer:", log["answer"])
    else:
        print("No final answer produced. See log for details:", args.log_json)


if __name__ == "__main__":
    main()
