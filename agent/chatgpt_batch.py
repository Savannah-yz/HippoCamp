#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for chatgpt.py (Azure OpenAI ChatGPT multimodal agent).

Supports input as:
- JSON list of objects (expects a "question" field by default)
- JSONL (one object per line or raw question text per line)
- Plain text (one question per line)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from prompt_modules.config import AVAILABLE_CONFIGS
from prompt_modules.prompt_body import AVAILABLE_PROMPT_VERSIONS


QUESTION_FALLBACK_KEYS = ("question", "query", "prompt", "text")
FILE_EXTENSIONS = (
    ".txt",
    ".py",
    ".ipynb",
    ".log",
    ".json",
    ".bin",
    ".npy",
    ".pkl",
    ".pt",
    ".pth",
    ".csv",
    ".docx",
    ".md",
    ".eml",
    ".ics",
    ".pdf",
    ".xlsx",
    ".pptx",
    ".sqlite",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp3",
    ".mp4",
    ".mkv",
)


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_")
    if not text:
        return "q"
    return text[:max_len]


def load_questions(path: Path, question_key: str = "question"):
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    items = []

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if question_key in data and isinstance(data[question_key], list):
                items = data[question_key]
            else:
                for key in ("questions", "data", "items", "records", "examples"):
                    if key in data and isinstance(data[key], list):
                        items = data[key]
                        break
                if not items and all(isinstance(v, str) for v in data.values()):
                    items = [{"id": k, question_key: v} for k, v in data.items()]
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        # JSONL or TXT
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    items.append(json.loads(line))
                    continue
                except Exception:
                    pass
            items.append(line)

    questions = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            q = item.strip()
            if not q:
                continue
            qid = str(idx)
            questions.append({"id": qid, "question": q, "raw": item})
            continue
        if isinstance(item, dict):
            q = item.get(question_key)
            if not q:
                for k in QUESTION_FALLBACK_KEYS:
                    q = item.get(k)
                    if q:
                        break
            if not q:
                continue
            qid = (
                str(item.get("id"))
                if item.get("id") is not None
                else str(item.get("qid"))
                if item.get("qid") is not None
                else str(idx)
            )
            questions.append({"id": qid, "question": str(q), "raw": item})
            continue
    return questions


def build_result_payload(
    raw_item,
    log_data,
    duration_sec: float | None = None,
    query_id: str | None = None,
):
    def dedupe(items):
        seen = set()
        out = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def looks_like_file(token: str) -> bool:
        if not token:
            return False
        if token.startswith("-"):
            return False
        if token in ("|", ">", ">>", "<", "2>", "2>>"):
            return False
        if token.startswith("http://") or token.startswith("https://"):
            return False
        if "/" in token:
            return True
        lower = token.lower()
        return any(lower.endswith(ext) for ext in FILE_EXTENSIONS)

    def extract_files_from_steps(steps):
        if not isinstance(steps, dict):
            return []
        files = []
        for step in steps.values():
            if not isinstance(step, dict):
                continue
            cmd = step.get("command") or ""
            if not cmd:
                continue
            try:
                parts = shlex.split(cmd)
            except Exception:
                continue
            if not parts:
                continue
            head = parts[0]
            if head in ("return_txt", "return_img", "return_ori", "return_metadata"):
                if len(parts) >= 2:
                    files.append(parts[1])
                continue
            for token in parts[1:]:
                if looks_like_file(token):
                    files.append(token)
        return files

    ground_files = None
    if isinstance(raw_item, dict):
        ground_files = raw_item.get("file_path")
        if ground_files is None:
            ground_files = raw_item.get("ground_file_list")
    if ground_files is None:
        ground_files = []
    if isinstance(ground_files, str):
        ground_files = [ground_files]
    if not isinstance(ground_files, list):
        ground_files = list(ground_files) if ground_files else []

    payload = {
        "agent": "ChatGPT",
        "query_id": str(query_id) if query_id is not None else "",
        "question": "",
        "answer": "",
        "steps": {},
        "ground_file_list": ground_files,
        "agent_file_list": [],
        "ground_truth": None,
        "evidence": None,
        "rationale": None,
        "data_source": None,
        "profiling_type": None,
        "agent_cap": None,
        "QA_type": None,
        "time_ms": None,
    }

    if isinstance(raw_item, dict):
        if not payload["query_id"]:
            payload["query_id"] = (
                str(raw_item.get("id"))
                if raw_item.get("id") is not None
                else str(raw_item.get("qid"))
                if raw_item.get("qid") is not None
                else ""
            )
        payload["question"] = raw_item.get("question") or raw_item.get("query") or raw_item.get("prompt") or raw_item.get("text") or ""
        payload["ground_truth"] = raw_item.get("answer")
        payload["evidence"] = raw_item.get("evidence")
        payload["rationale"] = raw_item.get("rationale")
        payload["data_source"] = raw_item.get("data_source")
        payload["profiling_type"] = raw_item.get("profiling_type")
        payload["agent_cap"] = raw_item.get("agent_cap")
        payload["QA_type"] = raw_item.get("QA_type")
    else:
        payload["question"] = str(raw_item) if raw_item is not None else ""

    if isinstance(log_data, dict):
        payload["answer"] = log_data.get("answer") or log_data.get("final_answer") or ""
        payload["steps"] = log_data.get("steps") or {}
        if not payload["question"]:
            payload["question"] = log_data.get("question") or ""
        files_touched = log_data.get("files_touched") or []
        files_from_steps = extract_files_from_steps(payload["steps"])
        if isinstance(files_touched, list):
            agent_files = files_touched + files_from_steps
        else:
            agent_files = files_from_steps
        payload["agent_file_list"] = dedupe([str(f) for f in agent_files if f])

    if duration_sec is not None:
        payload["time_ms"] = int(round(duration_sec * 1000))

    return payload


def run_one(args, question, qid, out_dir: Path, index: int, summary_path: Path, result_dir: Path, raw_item):
    slug = slugify(question)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(qid)).strip("_") or str(index)
    log_name = f"{index:04d}_{safe_id}.json"
    stdout_path = out_dir / f"{index:04d}_{safe_id}.stdout.log"
    stderr_path = out_dir / f"{index:04d}_{safe_id}.stderr.log"
    result_path = result_dir / log_name
    keep_log_json = bool(args.keep_log_json)
    log_path = None
    if keep_log_json:
        log_json_dir = out_dir / "log_json"
        log_json_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_json_dir / log_name
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=f"{index:04d}_{safe_id}_")
        log_path = Path(tmp.name)
        tmp.close()

    if args.skip_existing and result_path.exists():
        return {
            "id": qid,
            "question": question,
            "log_json": str(log_path),
            "result_json": str(result_path),
            "skipped": True,
        }

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "chatgpt.py"),
        "--container",
        args.container,
        "--question",
        question,
        "--log-json",
        str(log_path),
        "--model",
        args.model,
        "--max-steps",
        str(args.max_steps),
        "--max-output-chars",
        str(args.max_output_chars),
        "--format-retries",
        str(args.format_retries),
    ]
    if args.endpoint:
        cmd += ["--endpoint", args.endpoint]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.api_version:
        cmd += ["--api-version", args.api_version]
    if args.api_mode:
        cmd += ["--api-mode", args.api_mode]
    if args.api_timeout_sec:
        cmd += ["--api-timeout-sec", str(args.api_timeout_sec)]
    cmd += ["--temperature", str(args.temperature)]
    cmd += ["--max-completion-tokens", str(args.max_completion_tokens)]
    cmd += ["--api-retries", str(args.api_retries)]
    cmd += ["--api-retry-base", str(args.api_retry_base)]
    cmd += ["--api-retry-max", str(args.api_retry_max)]
    cmd += ["--max-mm-items", str(args.max_mm_items)]
    cmd += ["--max-mm-b64-chars", str(args.max_mm_b64_chars)]
    if args.ensure_webui:
        cmd.append("--ensure-webui")
    if args.docker_user:
        cmd += ["--docker-user", args.docker_user]
    if args.strict_parse:
        cmd.append("--strict-parse")
    if args.webui_base:
        cmd += ["--webui-base", args.webui_base]
    if args.prompt_config:
        cmd += ["--prompt-config", args.prompt_config]
    if args.prompt_version:
        cmd += ["--prompt-version", args.prompt_version]

    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.time() - start

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    summary = {
        "id": qid,
        "question": question,
        "log_json": str(log_path) if keep_log_json else "",
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log_data = {}
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log_data = {}
    if not keep_log_json:
        try:
            log_path.unlink()
        except Exception:
            pass

    result_payload = build_result_payload(raw_item, log_data, duration_sec=duration, query_id=str(qid))
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["result_json"] = str(result_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True, help="Docker container name or ID")
    parser.add_argument("--questions-file", required=True, help="Path to JSON/JSONL/TXT questions file")
    parser.add_argument("--question-key", default="question", help="Key name for question in JSON items")
    parser.add_argument("--out-dir", default="", help="(Deprecated) Output directory for logs (use --log-dir)")
    parser.add_argument("--log-dir", default="", help="Output directory for stdout/stderr logs (default: ./log)")
    parser.add_argument("--result-dir", default="", help="Directory for per-question result JSON files")
    parser.add_argument("--keep-log-json", action="store_true", help="Keep per-question log JSON files under log-dir/log_json")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--endpoint", default="", help="Azure endpoint/base URL")
    parser.add_argument("--api-key", default="", help="Azure API key")
    parser.add_argument("--api-version", default="", help="Azure API version for URL-based modes")
    parser.add_argument("--api-mode", default="auto", choices=("auto", "chat", "azure_chat", "responses"))
    parser.add_argument("--api-timeout-sec", type=int, default=120, help="API timeout seconds")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--api-retry-base", type=float, default=1.5)
    parser.add_argument("--api-retry-max", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--format-retries", type=int, default=2)
    parser.add_argument("--max-mm-items", type=int, default=2, help="Max media items per follow-up for return_img/return_ori")
    parser.add_argument("--max-mm-b64-chars", type=int, default=150000000, help="Skip media if base64 exceeds this size")
    parser.add_argument("--ensure-webui", action="store_true")
    parser.add_argument("--docker-user", default="")
    parser.add_argument("--strict-parse", action="store_true")
    parser.add_argument("--webui-base", default="")
    parser.add_argument(
        "--prompt-config",
        default="config0",
        choices=AVAILABLE_CONFIGS,
        help="Auxiliary-function prompt config to use.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=AVAILABLE_PROMPT_VERSIONS,
        help="Main prompt version to use.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max number of questions to run (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between runs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if result JSON already exists")
    parser.add_argument("--aggregate-json", default="", help="Write aggregated results JSON (default: <result-dir>/aggregate.json)")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions_file), question_key=args.question_key)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    questions_path = Path(args.questions_file)

    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = questions_path.with_name(f"{questions_path.stem}_chatgpt_result")

    if args.log_dir:
        out_dir = Path(args.log_dir)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("log")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.jsonl"

    for idx, item in enumerate(questions, start=1):
        qid = item.get("id", str(idx))
        question = item.get("question", "").strip()
        if not question:
            continue
        run_one(args, question, qid, out_dir, idx, summary_path, result_dir, item.get("raw"))
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Aggregate per-question results into one JSON file.
    aggregate_path = Path(args.aggregate_json) if args.aggregate_json else (result_dir / "aggregate.json")
    results_by_id = {}
    duplicate_count = 0
    for p in sorted(result_dir.glob("[0-9][0-9][0-9][0-9]_*.json")):
        try:
            item = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or item.get("query_id") or p.stem)
            if key in results_by_id:
                duplicate_count += 1
            results_by_id[key] = item
        except Exception:
            continue
    results = list(results_by_id.values())
    if duplicate_count > 0:
        print(f"[aggregate] deduplicated {duplicate_count} duplicate items by id/query_id")
    aggregate_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
