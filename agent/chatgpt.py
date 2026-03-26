#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal agent that interacts with a Docker container via terminal commands.

- Uses Azure OpenAI Chat/Responses API (ChatGPT deployment)
- Sends a strict prompt that forces the model to output tagged tool calls
- Executes commands inside the container
- Feeds terminal output back to the model
- For return_img/return_ori, restores native files then forwards multimodal content
- Stops when model returns an answer + end signal
- Logs every step to a JSON file
"""
import argparse
import base64
import json
import os
import re
import socket
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.parse
import shlex
import subprocess
import time
import urllib.request
import random
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timezone
from pathlib import Path

from prompt_modules.config import AVAILABLE_CONFIGS, get_auxiliary_functions_block
from prompt_modules.prompt_body import AVAILABLE_PROMPT_VERSIONS, build_prompt


class ChatGPTAPIError(RuntimeError):
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


def _append_api_version(url: str, api_version: str) -> str:
    if not api_version:
        return url
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if "api-version" in qs:
        return url
    qs["api-version"] = [api_version]
    new_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _normalize_openai_v1_base(endpoint: str) -> str:
    e = (endpoint or "").strip().rstrip("/")
    if not e:
        return e
    if "/openai/v1" in e:
        idx = e.find("/openai/v1")
        return e[: idx + len("/openai/v1")]
    if e.endswith("/openai"):
        return f"{e}/v1"
    if "openai.azure.com" in e and "/openai" not in e:
        return f"{e}/openai/v1"
    return e


def _detect_api_mode(endpoint: str, api_mode: str) -> str:
    if api_mode != "auto":
        return api_mode
    low = (endpoint or "").lower()
    if "/openai/responses" in low:
        return "responses"
    if "/openai/deployments/" in low and "/chat/completions" in low:
        return "azure_chat"
    return "chat"


def _extract_text_from_chat_obj(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_text_from_responses_obj(obj: Dict[str, Any]) -> str:
    output_text = obj.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = obj.get("output") or []
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    return _extract_text_from_chat_obj(obj)


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
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

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    timeout = max(1.0, timeout_sec)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = (resp.read() or b"").decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = (exc.read() or b"").decode("utf-8", errors="replace")
        snippet = body.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        retry_after = _parse_retry_after_sec(getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None)
        if retry_after is not None and retry_after > 0:
            snippet = f"{snippet} (retry_after={retry_after:.1f}s)"
        raise ChatGPTAPIError(
            f"Azure OpenAI HTTP {exc.code}: {snippet}",
            status_code=exc.code,
            body=body,
            retry_after_sec=retry_after,
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ChatGPTAPIError(f"Azure OpenAI request timed out after {timeout:.1f}s: {exc}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise ChatGPTAPIError(f"Azure OpenAI request timed out after {timeout:.1f}s: {reason}") from exc
        raise ChatGPTAPIError(f"Azure OpenAI connection error: {exc}") from exc

    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise ChatGPTAPIError("Azure OpenAI returned non-JSON response") from exc
    if not isinstance(obj, dict):
        raise ChatGPTAPIError("Azure OpenAI returned invalid response shape")
    return obj


def _call_openai_v1_chat(
    endpoint: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float],
    max_completion_tokens: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    base_url = _normalize_openai_v1_base(endpoint)
    chat_url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_completion_tokens > 0:
        payload["max_completion_tokens"] = max_completion_tokens

    obj: Dict[str, Any]
    try:
        from openai import OpenAI
    except Exception:
        headers = {"Content-Type": "application/json"}
        if "azure.com" in base_url.lower():
            headers["api-key"] = api_key
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        obj = _post_json(
            url=chat_url,
            headers=headers,
            payload=payload,
            timeout_sec=timeout_sec,
        )
    else:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=max(1.0, timeout_sec))
        kwargs = dict(payload)
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            if "max_completion_tokens" not in kwargs:
                raise
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            resp = client.chat.completions.create(**kwargs)
        if hasattr(resp, "model_dump"):
            obj = resp.model_dump()
        else:
            obj = json.loads(resp.to_json())

    return {
        "answer": _extract_text_from_chat_obj(obj),
        "usage": obj.get("usage"),
        "raw": obj,
        "endpoint_used": base_url,
        "model_used": obj.get("model") or model,
    }


def _call_azure_chat_completions_url(
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float],
    max_completion_tokens: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    url = _append_api_version(endpoint, api_version)
    payload: Dict[str, Any] = {
        "messages": messages,
    }
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_completion_tokens > 0:
        payload["max_tokens"] = max_completion_tokens

    obj = _post_json(
        url=url,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        payload=payload,
        timeout_sec=timeout_sec,
    )
    return {
        "answer": _extract_text_from_chat_obj(obj),
        "usage": obj.get("usage"),
        "raw": obj,
        "endpoint_used": url,
        "model_used": obj.get("model") or model,
    }


def _to_responses_input_content(content: Any, role: str = "user") -> List[Dict[str, Any]]:
    normalized_role = (role or "user").lower()
    text_block_type = "output_text" if normalized_role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_block_type, "text": content}]
    if not isinstance(content, list):
        return [{"type": text_block_type, "text": str(content or "")}]

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("text", "input_text", "output_text") and isinstance(item.get("text"), str):
            blocks.append({"type": text_block_type, "text": item.get("text", "")})
            continue
        if normalized_role == "assistant" and item_type == "refusal" and isinstance(item.get("refusal"), str):
            blocks.append({"type": "refusal", "refusal": item.get("refusal", "")})
            continue
        if normalized_role != "assistant" and item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    blocks.append({"type": "input_image", "image_url": url})
                    continue
            if isinstance(image_url, str) and image_url:
                blocks.append({"type": "input_image", "image_url": image_url})
                continue
    if not blocks:
        return [{"type": text_block_type, "text": str(content)}]
    return blocks


def _to_responses_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").lower()
        if role not in ("system", "user", "assistant", "developer"):
            role = "user"
        content = _to_responses_input_content(msg.get("content"), role=role)
        converted.append({"role": role, "content": content})
    return converted


def _call_azure_responses_url(
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float],
    max_completion_tokens: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    url = _append_api_version(endpoint, api_version)
    payload: Dict[str, Any] = {
        "model": model,
        "input": _to_responses_input_messages(messages),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_completion_tokens > 0:
        payload["max_output_tokens"] = max_completion_tokens

    obj = _post_json(
        url=url,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        payload=payload,
        timeout_sec=timeout_sec,
    )
    return {
        "answer": _extract_text_from_responses_obj(obj),
        "usage": obj.get("usage"),
        "raw": obj,
        "endpoint_used": url,
        "model_used": obj.get("model") or model,
    }


def _is_retryable_error(err: Exception) -> bool:
    if isinstance(err, (TimeoutError, socket.timeout)):
        return True
    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True

    status = getattr(err, "status_code", None)
    if isinstance(status, int):
        if status in (408, 409, 425, 429):
            return True
        if status >= 500:
            return True
    msg = str(err).lower()
    if any(token in msg for token in (
        "rate",
        "quota",
        "timeout",
        "timed out",
        "tempor",
        "unavailable",
        "connection",
        "429",
        "500",
        "503",
        "504",
    )):
        return True
    return False


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


def _build_multimodal_followup(
    command: str,
    raw_output: str,
    text_output: str,
    max_media_items: int,
    max_b64_chars: int,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
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

    content: List[Dict[str, Any]] = [{"type": "text", "text": f"This is output of terminal: {text_output}"}]
    restored_files: List[str] = []

    if head in ("return_img", "img"):
        b64_list = obj.get("image_b64_list") or []
        if not b64_list and obj.get("image_b64"):
            b64_list = [obj.get("image_b64")]
        image_paths = obj.get("image_paths") or []

        for i, b64_payload in enumerate(b64_list):
            if not isinstance(b64_payload, str) or not b64_payload:
                continue
            if len(b64_payload) > max_b64_chars:
                continue
            src_path = image_paths[i] if i < len(image_paths) else ""
            suffix = Path(str(src_path)).suffix.lower() or ".png"
            local_file = _decode_b64_to_temp_file(b64_payload, suffix)
            if not local_file:
                continue
            restored_files.append(local_file)
            mime_type = IMAGE_EXT_TO_MIME.get(suffix, "image/png")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_payload}"},
            })
            if len(restored_files) >= max_media_items:
                break

    if head in ("return_ori", "ori"):
        b64_payload = obj.get("file_b64")
        src_path = str(obj.get("file_path") or "")
        ext = Path(src_path).suffix.lower()
        if isinstance(b64_payload, str) and b64_payload and len(b64_payload) <= max_b64_chars:
            local_file = _decode_b64_to_temp_file(b64_payload, ext or ".bin")
            if local_file:
                restored_files.append(local_file)
                if ext in IMAGE_EXT_TO_MIME:
                    mime_type = IMAGE_EXT_TO_MIME[ext]
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_payload}"},
                    })

    if len(content) == 1:
        return None, []
    return content, restored_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True, help="Docker container name or ID")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--endpoint", default="", help="Azure endpoint/base URL")
    parser.add_argument("--api-key", default="", help="Azure API key")
    parser.add_argument("--model", default="gpt-5.2", help="Model/deployment name")
    parser.add_argument("--api-version", default="", help="Azure API version (for URL-based modes)")
    parser.add_argument("--api-mode", default="auto", choices=("auto", "chat", "azure_chat", "responses"))
    parser.add_argument("--api-timeout-sec", type=int, default=120, help="API timeout seconds")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--log-json", default="result/chatgpt_docker_session.json")
    parser.add_argument("--ensure-webui", action="store_true", help="Start WebUI before loop")
    parser.add_argument("--docker-user", default="", help="Optional docker exec user")
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--format-retries", type=int, default=2, help="Retries for format violations")
    parser.add_argument("--strict-parse", action="store_true", help="Disable relaxed parsing fallback")
    parser.add_argument("--webui-base", default="", help="Override WebUI base URL, e.g. http://localhost:8083")
    parser.add_argument("--api-retries", type=int, default=4, help="Retries on 429/temporary API errors")
    parser.add_argument("--api-retry-base", type=float, default=1.5, help="Base backoff seconds")
    parser.add_argument("--api-retry-max", type=float, default=20.0, help="Max backoff seconds")
    parser.add_argument("--max-mm-items", type=int, default=2, help="Max media items to attach in one follow-up")
    parser.add_argument("--max-mm-b64-chars", type=int, default=150000000, help="Skip media if base64 exceeds this size")
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
    endpoint = (
        args.endpoint
        or os.environ.get("CHATGPT_AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT", "https://synvoprod.openai.azure.com/openai/v1")
    )
    api_key = (
        args.api_key
        or os.environ.get("CHATGPT_AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY", "")
    )
    model_name = args.model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")
    api_version = args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_mode = _detect_api_mode(endpoint, args.api_mode)
    if not api_key:
        raise SystemExit("CHATGPT_AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_KEY not found in .env or environment")

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
    messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt}]

    def send_to_model(user_content: Any) -> Tuple[str, float]:
        attempt = 0
        start = time.time()
        while True:
            try:
                payload_messages = messages + [{"role": "user", "content": user_content}]
                if api_mode == "chat":
                    out = _call_openai_v1_chat(
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model_name,
                        messages=payload_messages,
                        temperature=args.temperature,
                        max_completion_tokens=args.max_completion_tokens,
                        timeout_sec=args.api_timeout_sec,
                    )
                elif api_mode == "azure_chat":
                    out = _call_azure_chat_completions_url(
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model_name,
                        api_version=api_version,
                        messages=payload_messages,
                        temperature=args.temperature,
                        max_completion_tokens=args.max_completion_tokens,
                        timeout_sec=args.api_timeout_sec,
                    )
                elif api_mode == "responses":
                    out = _call_azure_responses_url(
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model_name,
                        api_version=api_version,
                        messages=payload_messages,
                        temperature=args.temperature,
                        max_completion_tokens=args.max_completion_tokens,
                        timeout_sec=args.api_timeout_sec,
                    )
                else:
                    raise RuntimeError(f"Unsupported api mode: {api_mode}")
                return str(out.get("answer") or ""), time.time() - start
            except Exception as e:
                if attempt >= args.api_retries or not _is_retryable_error(e):
                    raise
                backoff = min(args.api_retry_max, args.api_retry_base * (2 ** attempt))
                retry_after = getattr(e, "retry_after_sec", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    backoff = max(backoff, min(args.api_retry_max, float(retry_after)))
                backoff *= 1.0 + random.uniform(0, 0.25)
                time.sleep(backoff)
                attempt += 1

    start_ts = time.time()
    log = {
        "question": args.question,
        "answer": "",
        "prompt_config": args.prompt_config,
        "prompt_version": args.prompt_version,
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

        for attempt in range(args.format_retries + 1):
            agent_text, agent_time = send_to_model(sent_content)
            messages.append({"role": "user", "content": sent_content})
            messages.append({"role": "assistant", "content": agent_text})
            try:
                action_obj = parse_response(agent_text, relaxed=not args.strict_parse)
                validate_action(action_obj)
                break
            except Exception as e:
                format_errors.append({
                    "attempt": attempt + 1,
                    "agent_response": agent_text,
                    "error": str(e),
                })
                if attempt >= args.format_retries:
                    correction = (
                        "FORMAT ERROR. You MUST output ONLY:\n"
                        "<think>...</think> followed by either:\n"
                        "<tool>{\"name\":\"terminal\",\"arguments\":{\"command\":\"...\"}}</tool>\n"
                        "OR\n"
                        "<answer>...</answer>\n"
                        "No extra text."
                    )
                    log["steps"][step_key] = {
                        "agent_response": agent_text,
                        "agent_time_sec": agent_time,
                        "user_input": sent_msg,
                        "error": f"parse_error: {e}",
                        "format_errors": format_errors,
                    }
                    pending_user_text = correction
                    pending_user_content = correction
                    action_obj = None
                    break
                sent_msg = (
                    "FORMAT ERROR. Output ONLY strict tags. "
                    "Example tool:\n<think>...</think><tool>{\"name\":\"terminal\",\"arguments\":{\"command\":\"list_files\"}}</tool>"
                )
                sent_content = sent_msg

        if action_obj is None:
            continue

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

        output_raw, exit_code, docker_time = run_docker_command(args.container, cmd, user=args.docker_user)
        is_grep_no_match = _is_grep_no_match(cmd, output_raw, exit_code)
        if is_grep_no_match and not (output_raw or "").strip():
            output_raw = "(no matches)"

        output_for_log = output_raw
        if len(output_for_log) > args.max_output_chars:
            output_for_log = output_for_log[:args.max_output_chars] + "\n...[truncated]"

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
            max_media_items=max(1, args.max_mm_items),
            max_b64_chars=max(10000, args.max_mm_b64_chars),
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
            if mm_content and isinstance(mm_content[0], dict):
                mm_content[0]["text"] = f"{mm_content[0].get('text', '')}{notice}"

        if mm_content:
            pending_user_content = mm_content
        else:
            pending_user_content = pending_user_text

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
