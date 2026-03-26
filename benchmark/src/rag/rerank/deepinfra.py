"""
DeepInfra reranker using HTTP inference API.

API format:
POST https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B
Headers: Authorization: bearer $DEEPINFRA_TOKEN
Payload: {"queries": [query], "documents": [doc1, doc2, ...]}
Response: {"scores": [0.1, 0.2, ...], ...}
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseReranker

logger = logging.getLogger(__name__)


class DeepInfraReranker(BaseReranker):
    """
    DeepInfra reranker via HTTP API.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        api_key: Optional[str] = None,
        api_key_env: str = "DEEPINFRA_RERANK_API_KEY",
        base_url: str = "https://api.deepinfra.com/v1/inference",
        timeout: float = 30.0,
        max_batch_size: int = 32,
        max_concurrent: int = 8,
        retry_max_attempts: int = 3,
    ):
        import os

        if api_key is None:
            api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"DeepInfra rerank API key not found. Set {api_key_env} or pass api_key."
            )

        self.model = model
        self.api_key_env = api_key_env
        self.endpoint = f"{base_url.rstrip('/')}/{model}"
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        self.retry_max_attempts = retry_max_attempts
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._headers = {
            "Authorization": f"bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "Initialized DeepInfraReranker: model=%s endpoint=%s max_batch_size=%s max_concurrent=%s retry_max_attempts=%s",
            model,
            self.endpoint,
            max_batch_size,
            max_concurrent,
            retry_max_attempts,
        )

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        documents = [c.get("chunk", "") for c in candidates]
        batches = [
            documents[i : i + self.max_batch_size]
            for i in range(0, len(documents), self.max_batch_size)
        ]

        batch_scores: List[Optional[List[float]]] = [None] * len(batches)

        async def process_batch(idx: int, batch_docs: List[str]) -> None:
            async with self._semaphore:
                batch_scores[idx] = await self._rerank_batch(query, batch_docs)

        await asyncio.gather(
            *[process_batch(i, batch) for i, batch in enumerate(batches)]
        )

        scores: List[float] = []
        for batch in batch_scores:
            if batch:
                scores.extend(batch)

        if len(scores) != len(candidates):
            raise ValueError(
                f"DeepInfra rerank returned {len(scores)} scores for {len(candidates)} candidates."
            )

        results: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = score
            if "score" in candidate:
                candidate["retrieval_score"] = candidate["score"]
            candidate["score"] = score
            results.append(candidate)

        if threshold is not None:
            results = [r for r in results if r.get("score", 0) >= threshold]

        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        if top_k is not None:
            results = results[:top_k]

        logger.info(
            "DeepInfra reranked %s candidates → %s results (top_k=%s, threshold=%s)",
            len(candidates),
            len(results),
            top_k,
            threshold,
        )
        return results

    async def _rerank_batch(self, query: str, documents: List[str]) -> List[float]:
        payload = {"queries": [query], "documents": documents}

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.retry_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "DeepInfra rerank retry %s/%s for batch_size=%s",
                        attempt.retry_state.attempt_number,
                        self.retry_max_attempts,
                        len(documents),
                    )
                resp = await self._client.post(
                    self.endpoint, headers=self._headers, json=payload
                )
                resp.raise_for_status()
                data = resp.json()

                scores = data.get("scores")
                if not isinstance(scores, list):
                    raise ValueError(
                        f"DeepInfra rerank response missing 'scores'. keys={list(data.keys())}"
                    )
                if len(scores) != len(documents):
                    raise ValueError(
                        f"DeepInfra rerank returned {len(scores)} scores for batch size {len(documents)}."
                    )
                return [float(s) for s in scores]

        return []

    def get_model_name(self) -> str:
        return self.model

    async def aclose(self):
        if self._client:
            await self._client.aclose()
