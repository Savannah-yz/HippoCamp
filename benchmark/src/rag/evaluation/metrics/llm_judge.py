"""
LLM as a Judge metric for RAG evaluation using Azure OpenAI.

Supports two prompt templates:
- "simple": Lightweight, fast evaluation (default). Outputs {"pred", "score"}.
- "detailed": Chain-of-Thought evaluation with structured rubric. Outputs {"pred", "score", "rationale"}.

Prompts are loaded from configs/evaluation.yaml (preferred) with built-in fallbacks.
"""

import os
import logging
import json
import re
import asyncio
import time
from typing import Optional, List, Dict, Any, Tuple

import requests

from .base import MetricCalculator, EvaluationMetric, RetrievedChunk

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Built-in Fallback Prompts (used when YAML config is not available)
# ────────────────────────────────────────────────────────────────────────────

_BUILTIN_PROMPTS: Dict[str, Dict[str, Any]] = {
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
        "max_tokens": 64,
    },
    "detailed": {
        "system": "You are an expert RAG evaluation assistant.",
        "user": "Evaluate: Question={question}, Ground Truth={ground_truth}, Answer={model_answer}",
        "max_tokens": 512,
    },
}


class LLMJudgeMetric(MetricCalculator):
    """LLM as a Judge metric using Azure OpenAI for evaluation.

    Prompt resolution priority:
        1. ``prompts`` param (dict from YAML config, passed via metric_kwargs)
        2. ``_BUILTIN_PROMPTS`` fallback (hardcoded in code)

    Args:
        prompt_template: Which prompt style to use — "simple" or "detailed".
        prompts: Full prompts dict from YAML, keyed by template name.
            Example: {"simple": {"system": "...", "user": "...", "max_tokens": 64}, ...}
    """

    def __init__(
        self,
        name: str = "llm_judge",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        prompt_template: str = "simple",
        prompts: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__(name)

        # Resolve prompts: YAML config > built-in fallback
        available_prompts = prompts if prompts else _BUILTIN_PROMPTS

        if prompt_template not in available_prompts:
            raise ValueError(
                f"Unknown prompt_template '{prompt_template}'. "
                f"Available: {list(available_prompts.keys())}"
            )
        self.prompt_template = prompt_template
        self._prompts = available_prompts[prompt_template]
        self._max_tokens = self._prompts.get("max_tokens", 64)

        source = "yaml config" if prompts else "built-in fallback"
        logger.info(f"LLM Judge using prompt template: {prompt_template} ({source})")

        # Get API configuration from parameters or environment
        self.api_url = (
            api_url or
            os.getenv('AZURE_OPENAI_ENDPOINT') or
            os.getenv('GPT4O_MINI_API')
        )
        self.api_key = (
            api_key or
            os.getenv('AZURE_OPENAI_API_KEY') or
            os.getenv('GPT4O_MINI_KEY')
        )

        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_url or not self.api_key:
            logger.warning(
                "LLM Judge not fully configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                "environment variables to enable LLM judge evaluation."
            )

    def is_configured(self) -> bool:
        """Check if LLM judge is properly configured."""
        return bool(self.api_url and self.api_key)

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate LLM judge score."""
        if not self.is_configured():
            return self._create_metric(0.0, {
                "error": "LLM Judge not configured",
                "pred": "no",
                "score_0_5": 0
            })

        try:
            if not ground_truth:
                logger.warning("No ground truth provided for LLM judge evaluation")
                return self._create_metric(0.0, {"error": "No ground truth provided", "pred": "no"})

            if not generated_answer or not generated_answer.strip():
                logger.warning("No generated answer provided for LLM judge evaluation")
                return self._create_metric(0.0, {"error": "No generated answer provided", "pred": "no"})

            # Call Azure OpenAI for judgment with timing
            start_time = time.time()
            raw_response, status_code = await self._call_azure_openai(
                query, generated_answer, ground_truth
            )
            eval_latency = time.time() - start_time

            # Parse the response
            pred, score = self._safe_parse_pred_score(raw_response)

            # Convert score from 0-5 scale to 0-1 scale for consistency
            normalized_score = score / 5.0

            # Determine API status string
            api_status = "success" if 200 <= status_code < 300 else f"error_{status_code}"

            # Extract rationale from detailed mode, fallback to generic
            rationale = self._extract_rationale(raw_response)
            if not rationale:
                rationale = f"Prediction: {pred}, Score: {score}/5"

            details = {
                "llm_as_a_judge_score": score,
                "pred": pred,
                "score_0_5": score,
                "score_normalized": normalized_score,
                "rationale": rationale,
                "prompt_template": self.prompt_template,
                "api_status": api_status,
                "eval_api_latency_seconds": round(eval_latency, 3),
                "raw_response": raw_response,
            }

            return self._create_metric(score, details)

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return self._create_metric(0.0, {
                "error": str(e),
                "pred": "no",
                "score_0_5": 0,
                "score_normalized": 0.0
            })

    async def _call_azure_openai(
        self,
        question: str,
        model_answer: str,
        ground_truth: str
    ) -> Tuple[str, int]:
        """Call Azure OpenAI API."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._prompts["system"]},
                {"role": "user", "content": self._prompts["user"].format(
                    question=question or "",
                    ground_truth=ground_truth or "",
                    model_answer=model_answer or ""
                )}
            ]
        }

        attempt = 0
        backoff = 1.0

        while True:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                )

                status = response.status_code

                # Success
                if 200 <= status < 300:
                    try:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                        return content, status
                    except Exception:
                        return response.text, status

                # Retryable errors: 429/5xx
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue

                # Other errors or max retries exceeded
                try:
                    return response.text or response.json(), status
                except Exception:
                    return response.text, status

            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                else:
                    raise e

    def _safe_parse_pred_score(self, text: str) -> Tuple[str, int]:
        """Parse {"pred":"yes","score":4} from model output robustly."""
        text = text.strip()

        # Extract first {...} JSON object
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

        # Regex fallback
        pred_match = re.search(r'"?pred"?\s*:\s*"?\b(yes|no)\b"?', text, flags=re.I)
        score_match = re.search(r'"?score"?\s*:\s*(\d+)', text, flags=re.I)
        pred = pred_match.group(1).lower() if pred_match else "no"
        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(5, score))
        return pred, score

    def _extract_rationale(self, text: str) -> Optional[str]:
        """Extract rationale field from JSON response (used by detailed template)."""
        text = text.strip()
        json_obj_match = re.search(r"\{.*\}", text, flags=re.S)
        if json_obj_match:
            try:
                obj = json.loads(json_obj_match.group(0))
                rationale = obj.get("rationale")
                if rationale and isinstance(rationale, str):
                    return rationale.strip()
            except Exception:
                pass
        return None

    async def health_check(self) -> bool:
        """Check if Azure OpenAI service is available."""
        if not self.is_configured():
            return False
        try:
            test_response, status = await self._call_azure_openai(
                "Test question", "Test answer", "Test ground truth"
            )
            return 200 <= status < 300
        except Exception as e:
            logger.error(f"Azure OpenAI health check failed: {e}")
            return False
