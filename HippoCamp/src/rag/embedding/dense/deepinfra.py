"""
DeepInfra OpenAI-compatible Embedder.

This embedder uses DeepInfra's API which follows the OpenAI embeddings format.
It's ideal for:
- Cloud-based embedding without local model management
- Using models like Qwen/Qwen3-Embedding-0.6B via API
- Quick prototyping and testing

Example DeepInfra API usage:
    from openai import OpenAI

    openai = OpenAI(
        api_key="$DEEPINFRA_TOKEN",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    embeddings = openai.embeddings.create(
        model="Qwen/Qwen3-Embedding-0.6B",
        input="The food was delicious...",
        encoding_format="float"
    )
"""

import asyncio
import logging
import re
from typing import List, Optional, Tuple
import numpy as np
from openai import OpenAI, AsyncOpenAI
from tenacity import AsyncRetrying, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.rag.embedding.base import BaseDenseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    from openai import (
        APIError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
    RETRYABLE_EXCEPTIONS = (APIError, APIConnectionError, APITimeoutError, RateLimitError)
except Exception:
    RETRYABLE_EXCEPTIONS = (Exception,)


class DeepInfraOpenAIEmbedder(BaseDenseEmbedder):
    """
    DeepInfra-based dense embedder using OpenAI-compatible API.

    This embedder calls DeepInfra's embedding API which follows the OpenAI
    embeddings format. It supports various embedding models available on
    DeepInfra.

    Features:
    - OpenAI-compatible API format
    - Cloud-based embeddings
    - Batch processing support
    - Async support for better performance
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        dimension: int = 1024,
        api_key: Optional[str] = None,
        api_key_env: str = "DEEPINFRA_API_KEY",
        max_batch_size: int = 32,
        max_concurrent: int = 10,
        encoding_format: str = "float",
        timeout: float = 60.0,
        retry_max_attempts: int = 3,
    ):
        """
        Initialize DeepInfra OpenAI-style embedder.

        Args:
            model: Model identifier for DeepInfra API
                   (e.g., "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-large-en-v1.5")
            dimension: Expected embedding dimension
            api_key: DeepInfra API key (optional, will read from env if not provided)
            api_key_env: Environment variable name for API key
            max_batch_size: Maximum number of texts per API call
            max_concurrent: Maximum concurrent API requests
            encoding_format: Output format ("float" only supported currently)
            timeout: HTTP request timeout in seconds
        """
        self.model_name = model
        self.dimension = dimension
        self.api_key_env = api_key_env
        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        self.encoding_format = encoding_format
        self.timeout = timeout
        self.retry_max_attempts = retry_max_attempts
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stats_lock = asyncio.Lock()
        self._stats = {"batch_halving": 0, "text_splits": 0}

        # Get API key from parameter or environment
        if api_key is None:
            import os
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"DeepInfra API key not found. Set {api_key_env} environment variable "
                    "or pass api_key directly."
                )

        # Create synchronous client for compatibility
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            timeout=timeout,
        )

        # Create async client for async operations
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            timeout=timeout,
        )

        logger.info(
            f"Initialized DeepInfraOpenAIEmbedder: "
            f"model={model}, dimension={dimension}, max_batch_size={max_batch_size}, "
            f"max_concurrent={max_concurrent}, encoding_format={encoding_format}"
        )

    async def embed_dense(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embeddings via DeepInfra API.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape [N, dimension]
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            # Process in batches with concurrency limit
            batches = [
                texts[i : i + self.max_batch_size]
                for i in range(0, len(texts), self.max_batch_size)
            ]
            results: List[Optional[np.ndarray]] = [None] * len(batches)

            async def process_batch(index: int, batch: List[str]) -> None:
                async with self._semaphore:
                    results[index] = await self._embed_batch_with_fallback(batch)

            await asyncio.gather(
                *[process_batch(i, batch) for i, batch in enumerate(batches)]
            )

            all_embeddings = [emb for emb in results if emb is not None]
            if not all_embeddings:
                return np.array([]).reshape(0, self.dimension)

            merged = np.vstack(all_embeddings)

            # Validate dimension
            if merged.shape[1] != self.dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {merged.shape[1]}. Updating dimension."
                )
                self.dimension = merged.shape[1]

            logger.info(f"Generated embeddings via DeepInfra: shape={merged.shape}")
            await self._log_stats_if_any()
            return merged.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embeddings via DeepInfra: {e}", exc_info=True)
            raise

    async def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Embed a single batch of texts.

        Args:
            batch: List of text strings

        Returns:
            numpy array of embeddings
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.retry_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "DeepInfra embedding retry %s/%s for batch_size=%s",
                        attempt.retry_state.attempt_number,
                        self.retry_max_attempts,
                        len(batch),
                    )
                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format=self.encoding_format,
                )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    async def _embed_batch_with_fallback(self, batch: List[str]) -> np.ndarray:
        try:
            return await self._embed_batch(batch)
        except Exception as exc:
            if not self._is_batch_error(exc):
                raise

            if len(batch) > 1:
                next_size = max(1, len(batch) // 2)
                logger.warning(
                    "DeepInfra batch size=%s failed; retrying with smaller batches=%s",
                    len(batch),
                    next_size,
                )
                await self._inc_stat("batch_halving")
                return await self._embed_batched(batch, next_size)

            # Single item failed -> split text
            text = batch[0]
            logger.warning("DeepInfra single chunk failed; attempting split embedding.")
            return await self._embed_single_with_split(text)

    async def _embed_batched(self, texts: List[str], batch_size: int) -> np.ndarray:
        batches = [
            texts[i : i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        embeddings_list: List[np.ndarray] = []
        for batch in batches:
            embeddings_list.append(await self._embed_batch_with_fallback(batch))
        return np.vstack(embeddings_list) if embeddings_list else np.array([]).reshape(0, self.dimension)

    async def _embed_single_with_split(self, text: str) -> np.ndarray:
        segments = [text]
        collected: List[np.ndarray] = []
        weights: List[float] = []

        while segments:
            segment = segments.pop(0)
            if not segment or not segment.strip():
                continue
            try:
                emb = await self._embed_batch([segment])
                if emb.size == 0:
                    continue
                collected.append(emb[0])
                weights.append(float(len(segment)))
            except Exception as exc:
                if not self._is_batch_error(exc):
                    raise
                if len(segment) <= 1:
                    raise
                await self._inc_stat("text_splits")
                left, right = self._split_text(segment)
                if left:
                    segments.append(left)
                if right:
                    segments.append(right)

        if not collected:
            return np.array([]).reshape(0, self.dimension)

        weight_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        stacked = np.vstack(collected)
        averaged = (stacked * weight_arr).sum(axis=0) / weight_arr.sum()
        return averaged.reshape(1, -1)

    async def _inc_stat(self, key: str, amount: int = 1) -> None:
        async with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    async def _log_stats_if_any(self) -> None:
        async with self._stats_lock:
            batch_halving = self._stats.get("batch_halving", 0)
            text_splits = self._stats.get("text_splits", 0)
            self._stats = {"batch_halving": 0, "text_splits": 0}
        if batch_halving or text_splits:
            logger.info(
                "DeepInfra fallback stats: batch_halving=%s text_splits=%s",
                batch_halving,
                text_splits,
            )

    def _split_text(self, text: str) -> Tuple[str, str]:
        mid = len(text) // 2
        if mid <= 0 or mid >= len(text):
            return text, ""

        # Try to split on whitespace near the midpoint
        window = text[max(0, mid - 200) : min(len(text), mid + 200)]
        match = re.search(r"\s", window)
        if match:
            split_idx = max(0, mid - 200) + match.start()
            return text[:split_idx].strip(), text[split_idx:].strip()

        return text[:mid].strip(), text[mid:].strip()

    def _is_batch_error(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
        if status in (400, 413, 422, 429):
            return True
        message = str(exc)
        patterns = (
            r"too\s+large",
            r"context\s+length",
            r"maximum\s+context",
            r"request\s+too\s+large",
            r"payload\s+too\s+large",
        )
        return any(re.search(p, message, flags=re.IGNORECASE) for p in patterns)

    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Embed texts and return EmbeddingResult.

        Args:
            texts: List of text strings to embed
            return_sparse: Ignored (DeepInfra API doesn't provide sparse embeddings)

        Returns:
            EmbeddingResult with dense_embeddings only
        """
        if return_sparse:
            logger.warning(
                "return_sparse=True but DeepInfraOpenAIEmbedder doesn't support sparse embeddings. "
                "Returning dense only."
            )

        dense_emb = await self.embed_dense(texts)
        return EmbeddingResult(dense_embeddings=dense_emb, sparse_embeddings=None)

    def embed_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous embedding method for compatibility.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape [N, dimension]
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            all_embeddings: List[np.ndarray] = []
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i : i + self.max_batch_size]
                embeddings = self._embed_batch_sync_with_fallback(batch)
                all_embeddings.append(embeddings)

            if not all_embeddings:
                return np.array([]).reshape(0, self.dimension)

            merged = np.vstack(all_embeddings)

            # Validate dimension
            if merged.shape[1] != self.dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {merged.shape[1]}. Updating dimension."
                )
                self.dimension = merged.shape[1]

            logger.info(f"Generated embeddings via DeepInfra (sync): shape={merged.shape}")
            self._log_sync_stats_if_any()
            return merged.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embeddings via DeepInfra (sync): {e}", exc_info=True)
            raise

    def _embed_batch_sync(self, batch: List[str]) -> np.ndarray:
        """
        Embed a single batch of texts synchronously.

        Args:
            batch: List of text strings

        Returns:
            numpy array of embeddings
        """
        for attempt in Retrying(
            stop=stop_after_attempt(self.retry_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "DeepInfra embedding retry %s/%s (sync) for batch_size=%s",
                        attempt.retry_state.attempt_number,
                        self.retry_max_attempts,
                        len(batch),
                    )
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format=self.encoding_format,
                )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def _embed_batch_sync_with_fallback(self, batch: List[str]) -> np.ndarray:
        try:
            return self._embed_batch_sync(batch)
        except Exception as exc:
            if not self._is_batch_error(exc):
                raise
            if len(batch) > 1:
                next_size = max(1, len(batch) // 2)
                logger.warning(
                    "DeepInfra batch size=%s failed (sync); retrying with smaller batches=%s",
                    len(batch),
                    next_size,
                )
                self._inc_sync_stat("batch_halving")
                return self._embed_batched_sync(batch, next_size)
            text = batch[0]
            logger.warning("DeepInfra single chunk failed (sync); attempting split embedding.")
            return self._embed_single_with_split_sync(text)

    def _embed_batched_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        batches = [
            texts[i : i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        embeddings_list: List[np.ndarray] = []
        for batch in batches:
            embeddings_list.append(self._embed_batch_sync_with_fallback(batch))
        return np.vstack(embeddings_list) if embeddings_list else np.array([]).reshape(0, self.dimension)

    def _embed_single_with_split_sync(self, text: str) -> np.ndarray:
        segments = [text]
        collected: List[np.ndarray] = []
        weights: List[float] = []

        while segments:
            segment = segments.pop(0)
            if not segment or not segment.strip():
                continue
            try:
                emb = self._embed_batch_sync([segment])
                if emb.size == 0:
                    continue
                collected.append(emb[0])
                weights.append(float(len(segment)))
            except Exception as exc:
                if not self._is_batch_error(exc):
                    raise
                if len(segment) <= 1:
                    raise
                self._inc_sync_stat("text_splits")
                left, right = self._split_text(segment)
                if left:
                    segments.append(left)
                if right:
                    segments.append(right)

        if not collected:
            return np.array([]).reshape(0, self.dimension)

        weight_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        stacked = np.vstack(collected)
        averaged = (stacked * weight_arr).sum(axis=0) / weight_arr.sum()
        return averaged.reshape(1, -1)

    def _inc_sync_stat(self, key: str, amount: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + amount

    def _log_sync_stats_if_any(self) -> None:
        batch_halving = self._stats.get("batch_halving", 0)
        text_splits = self._stats.get("text_splits", 0)
        self._stats = {"batch_halving": 0, "text_splits": 0}
        if batch_halving or text_splits:
            logger.info(
                "DeepInfra fallback stats (sync): batch_halving=%s text_splits=%s",
                batch_halving,
                text_splits,
            )

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model_name

    async def aclose(self):
        """
        Cleanup resources for async client.
        """
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()

    def close(self):
        """
        Cleanup resources for sync client.
        """
        pass  # OpenAI client handles cleanup automatically
