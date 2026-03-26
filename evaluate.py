#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    # -- Simple (original) -------------------------------------------------
    "simple": {
        "system": (
            "You are an evaluation assistant. Compare the model output with the ground truth and judge meaningful match.\n"
            "Rules:\n"
            "- Focus on semantic equivalence, allow paraphrase and synonyms.\n"
            "- If model adds extra but non-conflicting details, that's OK; penalize if it misses key points.\n"
            "- Score is an INTEGER 0..5 (5=excellent match, 0=totally incorrect).\n"
            "- Output STRICTLY a JSON object with keys 'pred' ('yes'/'no') and 'score' (INTEGER).\n"
            'Example outputs: {"pred":"yes","score":4} or {"pred":"no","score":1}\n'
        ),
        "user": (
            "Please evaluate the following QA pair:\n\n"
            "Question:\n{question}\n\n"
            "Ground Truth:\n{ground_truth}\n\n"
            "Model Answer:\n{model_answer}\n\n"
            "Return ONLY a JSON object with keys 'pred' and 'score' as specified."
        ),
    },
    # -- Detailed (Chain-of-Thought with rubric) ---------------------------
    "detailed": {
        "system": (
            "You are an expert RAG evaluation assistant. Your task is to rigorously compare a model-generated answer "
            "against a ground truth reference and produce a structured judgment.\n\n"
            "# EVALUATION DIMENSIONS\n"
            "You MUST evaluate along these four dimensions:\n"
            "1. **Factual Accuracy**: Are the facts in the model answer correct and consistent with the ground truth?\n"
            "2. **Completeness**: Does the model answer cover ALL key points present in the ground truth?\n"
            "3. **Relevance**: Is the answer focused on the question without introducing irrelevant or misleading information?\n"
            "4. **Specificity**: Does the answer preserve specific details (names, numbers, dates, quantities) from the ground truth?\n\n"
            "# SCORING RUBRIC (0-5 INTEGER scale)\n"
            "- 5 (Excellent): Semantically equivalent to ground truth. All key facts, entities, and details are present. Minor paraphrasing is fine.\n"
            "- 4 (Good): Captures all major points with at most minor omissions of secondary details. No factual errors.\n"
            "- 3 (Acceptable): Covers the main idea correctly but misses some important details or includes minor inaccuracies.\n"
            "- 2 (Partial): Gets the general topic right but misses multiple key points or contains notable factual errors.\n"
            "- 1 (Poor): Only tangentially related to the ground truth or contains significant factual errors.\n"
            "- 0 (Wrong): Completely incorrect, irrelevant, or contradicts the ground truth.\n\n"
            "# CRITICAL REQUIREMENTS\n"
            "- NEVER ignore specific names, numbers, dates, or quantities - compare them explicitly.\n"
            "- Allow paraphrasing and synonyms, but NOT factual substitutions (e.g., different names or numbers).\n"
            "- Extra non-conflicting details are acceptable; MISSING key information must be penalized.\n"
            "- You MUST follow the structured Chain-of-Thought process below before scoring.\n\n"
            "# OUTPUT FORMAT\n"
            "You MUST output a single JSON object with these keys:\n"
            '- "pred": "yes" if score >= 3, "no" otherwise\n'
            '- "score": INTEGER 0..5\n'
            '- "rationale": A concise string (2-4 sentences) summarizing your judgment\n\n'
            "Example:\n"
            '{"pred":"yes","score":4,"rationale":"The answer correctly identifies X and Y. It omits the specific date (Jan 2024) mentioned in ground truth, but all other facts are accurate."}\n'
        ),
        "user": (
            "Evaluate the following QA pair using the structured process below.\n\n"
            "---\n"
            "Question:\n{question}\n\n"
            "Ground Truth:\n{ground_truth}\n\n"
            "Model Answer:\n{model_answer}\n"
            "---\n\n"
            "## STEP 1: KEY FACTS EXTRACTION\n"
            "List the key facts, entities, numbers, and details from the Ground Truth.\n\n"
            "## STEP 2: FACT-BY-FACT COMPARISON\n"
            "For each key fact from Step 1, check whether the Model Answer includes it correctly.\n"
            "Mark each as: MATCH / MISSING / INCORRECT / PARAPHRASED.\n\n"
            "## STEP 3: EXTRA INFORMATION CHECK\n"
            "Note any extra information in the Model Answer not present in Ground Truth.\n"
            "Is it non-conflicting (acceptable) or contradictory (penalize)?\n\n"
            "## STEP 4: DIMENSION SCORING\n"
            "Rate each dimension (1-5):\n"
            "- Factual Accuracy:\n"
            "- Completeness:\n"
            "- Relevance:\n"
            "- Specificity:\n\n"
            "## STEP 5: FINAL JUDGMENT\n"
            "Based on the above analysis, output ONLY a JSON object:\n"
            '{{"pred":"yes/no","score":<0-5>,"rationale":"<2-4 sentence summary>"}}'
        ),
    },
}


def get_prompt_template(name: str) -> Dict[str, str]:
    key = (name or "").strip().lower()
    return PROMPT_TEMPLATES.get(key, PROMPT_TEMPLATES["detailed"])


def _safe_parse_pred_score(text: str) -> Tuple[str, int]:
    text = (text or "").strip()
    json_obj_match = re.search(r"\{.*\}", text, flags=re.S)
    if json_obj_match:
        snippet = json_obj_match.group(0)
        try:
            obj = json.loads(snippet)
            pred = str(obj.get("pred", "no")).lower()
            score = int(obj.get("score", 0))
            if pred not in ("yes", "no"):
                pred = "no"
            score = max(0, min(5, score))
            return pred, score
        except Exception:
            pass

    pred_match = re.search(r'"?pred"?\s*:\s*"?\b(yes|no)\b"?', text, flags=re.I)
    score_match = re.search(r'"?score"?\s*:\s*(\d+)', text, flags=re.I)
    pred = pred_match.group(1).lower() if pred_match else "no"
    score = int(score_match.group(1)) if score_match else 0
    score = max(0, min(5, score))
    return pred, score


@dataclass
class LLMJudgeClient:
    api_url: Optional[str]
    api_key: Optional[str]
    timeout: int = 60
    max_retries: int = 3
    prompt_template: str = "detailed"

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    async def judge(
        self,
        question: str,
        model_answer: str,
        ground_truth: str,
    ) -> Tuple[str, int, int, float]:
        if not self.is_configured():
            raise RuntimeError("LLM Judge not configured")

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        prompt = get_prompt_template(self.prompt_template)
        payload = {
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"].format(
                    question=question or "",
                    ground_truth=ground_truth or "",
                    model_answer=model_answer or "",
                )},
            ],
        }

        attempt = 0
        backoff = 1.0
        start_time = time.time()
        while True:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    ),
                )
                status = response.status_code
                if 200 <= status < 300:
                    try:
                        data = response.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        ) or ""
                        return content, status, attempt, time.time() - start_time
                    except Exception:
                        return response.text, status, attempt, time.time() - start_time

                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue

                try:
                    return response.text or response.json(), status, attempt, time.time() - start_time
                except Exception:
                    return response.text, status, attempt, time.time() - start_time

            except Exception:
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                raise


def _dedupe_text_list(values: Any) -> Optional[List[str]]:
    if values is None:
        return None
    if not isinstance(values, list):
        return None
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def calculate_file_list_metrics(
    ground_file_list: Optional[List[str]],
    agent_file_list: Optional[List[str]],
) -> Dict[str, float]:
    if not ground_file_list:
        return {"f1_score": 0.0, "recall": 0.0, "precision": 0.0}
    if not agent_file_list:
        return {"f1_score": 0.0, "recall": 0.0, "precision": 0.0}

    gt_set = set(ground_file_list)
    pred_set = set(agent_file_list)
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    return {"f1_score": f1_score, "recall": recall, "precision": precision}


def _load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                return data["results"]
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
        raise ValueError("Unsupported JSON structure")
    except json.JSONDecodeError:
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def _evaluate_item(
    item: Dict[str, Any],
    judge_client: LLMJudgeClient,
    semaphore: asyncio.Semaphore,
    fallback_query_id: str = "",
) -> Dict[str, Any]:
    query = item.get("query") or item.get("question", "")
    answer = item.get("answer", "")
    ground_truth = item.get("ground_truth", "")
    query_id = str(item.get("query_id") or item.get("id") or fallback_query_id)
    raw_time_ms = item.get("time_ms")
    try:
        time_ms = float(raw_time_ms) if raw_time_ms is not None else None
    except (TypeError, ValueError):
        time_ms = None

    if not ground_truth:
        judge = {
            "llm_as_a_judge_score": 0,
            "pred": "no",
            "score_0_5": 0,
            "score_normalized": 0.0,
            "rationale": "Prediction: no, Score: 0/5",
            "api_status": "error_no_ground_truth",
            "eval_api_latency_seconds": 0.0,
            "raw_response": "",
        }
    elif not answer or not answer.strip():
        judge = {
            "llm_as_a_judge_score": 0,
            "pred": "no",
            "score_0_5": 0,
            "score_normalized": 0.0,
            "rationale": "Prediction: no, Score: 0/5",
            "api_status": "error_no_answer",
            "eval_api_latency_seconds": 0.0,
            "raw_response": "",
        }
    elif not judge_client.is_configured():
        judge = {
            "llm_as_a_judge_score": 0,
            "pred": "no",
            "score_0_5": 0,
            "score_normalized": 0.0,
            "rationale": "Prediction: no, Score: 0/5",
            "api_status": "error_not_configured",
            "eval_api_latency_seconds": 0.0,
            "raw_response": "",
        }
    else:
        async with semaphore:
            raw_response, status, _, latency = await judge_client.judge(
                query, answer, ground_truth
            )
        pred, score = _safe_parse_pred_score(raw_response)
        judge = {
            "llm_as_a_judge_score": score,
            "pred": pred,
            "score_0_5": score,
            "score_normalized": score / 5.0,
            "rationale": f"Prediction: {pred}, Score: {score}/5",
            "api_status": "success" if 200 <= status < 300 else f"error_{status}",
            "eval_api_latency_seconds": round(latency, 3),
            "raw_response": raw_response,
        }

    ground_file_list = item.get("ground_file_list")
    if ground_file_list is None:
        # Backward compatibility with old dataset schema.
        ground_file_list = item.get("file_list")
    if isinstance(ground_file_list, list):
        ground_file_list = _dedupe_text_list(ground_file_list)

    agent_file_list = item.get("agent_file_list")
    if agent_file_list is None:
        # Backward compatibility with old dataset schema.
        agent_file_list = item.get("retrieved_file_list")
    if isinstance(agent_file_list, list):
        agent_file_list = _dedupe_text_list(agent_file_list)

    if ground_file_list is not None:
        file_list_metrics = calculate_file_list_metrics(ground_file_list, agent_file_list)
    else:
        file_list_metrics = None

    result = {
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "ground_truth": ground_truth,
        "time_ms": time_ms,
        "judge": judge,
        "timestamp": _timestamp_now(),
        "ground_file_list": ground_file_list,
        "agent_file_list": agent_file_list,
        "file_list_metrics": file_list_metrics,
    }
    return result


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["judge"]["score_0_5"] for r in results]
    count = len(scores)
    mean = sum(scores) / count if count else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    yes_count = sum(1 for r in results if r["judge"]["pred"] == "yes")
    no_count = sum(1 for r in results if r["judge"]["pred"] == "no")
    pass_rate = yes_count / count if count else 0.0
    latencies_ms: List[float] = []
    for r in results:
        value = r.get("time_ms")
        try:
            latency_ms = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            latency_ms = 0.0
        if latency_ms > 0:
            latencies_ms.append(latency_ms)
    avg_latency_ms = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

    f1s: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    for r in results:
        metrics = r.get("file_list_metrics")
        if metrics is None:
            continue
        f1s.append(metrics.get("f1_score", 0.0))
        recalls.append(metrics.get("recall", 0.0))
        precisions.append(metrics.get("precision", 0.0))

    if f1s:
        avg_f1 = sum(f1s) / len(f1s)
        avg_recall = sum(recalls) / len(recalls)
        avg_precision = sum(precisions) / len(precisions)
        file_hit_rate = avg_recall
    else:
        avg_f1 = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        file_hit_rate = 0.0

    return {
        "total_queries": len(results),
        "metrics": {
            "llm_judge": {
                "mean": mean,
                "min": min_score,
                "max": max_score,
                "count": count,
                "yes_count": yes_count,
                "no_count": no_count,
                "pass_rate": pass_rate,
                "avg_latency_ms": avg_latency_ms,
            }
        },
        "file_list_metrics": {
            "total_evaluated": len(f1s),
            "average_f1_score": avg_f1,
            "average_recall": avg_recall,
            "average_precision": avg_precision,
            "file_hit_rate": file_hit_rate,
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    items = _load_items(args.input_path)
    judge_client = LLMJudgeClient(
        api_url=args.judge_api_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("GPT4O_MINI_API"),
        api_key=args.judge_api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("GPT4O_MINI_KEY"),
        timeout=args.request_timeout_sec,
        max_retries=args.judge_max_retries,
        prompt_template=args.prompt_template,
    )
    semaphore = asyncio.Semaphore(args.judge_concurrency)
    tasks = [
        _evaluate_item(item, judge_client, semaphore, fallback_query_id=str(i))
        for i, item in enumerate(items, start=1)
    ]
    results = await asyncio.gather(*tasks)
    aggregate = _aggregate(results)

    if args.per_query_output_path:
        with open(args.per_query_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    if args.aggregate_metrics_output_path:
        with open(args.aggregate_metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)

    if args.print_results:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    if load_dotenv:
        load_dotenv()
    parser = argparse.ArgumentParser(description="LLM-as-a-judge + file-list metrics evaluator.")
    parser.add_argument(
        "--input-dataset",
        "--input",
        dest="input_path",
        required=True,
        help="Input evaluation dataset (JSON/JSONL).",
    )
    parser.add_argument(
        "--per-query-results-json",
        "--output",
        dest="per_query_output_path",
        help="Output JSON path for per-query judge results.",
    )
    parser.add_argument(
        "--aggregate-metrics-json",
        "--aggregate",
        dest="aggregate_metrics_output_path",
        help="Output JSON path for aggregate judge metrics.",
    )
    parser.add_argument(
        "--print-results",
        "--print",
        dest="print_results",
        action="store_true",
        help="Print per-query results and aggregate metrics to stdout.",
    )
    parser.add_argument(
        "--judge-api-url",
        "--api-url",
        dest="judge_api_url",
        help="Azure OpenAI chat completions endpoint for judge model.",
    )
    parser.add_argument(
        "--judge-api-key",
        "--api-key",
        dest="judge_api_key",
        help="Azure OpenAI API key for judge model.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        "--timeout",
        dest="request_timeout_sec",
        type=int,
        default=60,
        help="Judge API request timeout in seconds.",
    )
    parser.add_argument(
        "--judge-max-retries",
        "--max-retries",
        dest="judge_max_retries",
        type=int,
        default=3,
        help="Max retries for judge API 429/5xx responses.",
    )
    parser.add_argument(
        "--judge-concurrency",
        "--concurrency",
        dest="judge_concurrency",
        type=int,
        default=4,
        help="Max concurrent judge API calls.",
    )
    parser.add_argument(
        "--prompt-template",
        choices=("simple", "detailed"),
        default="detailed",
        help="Prompt template for LLM judge.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
