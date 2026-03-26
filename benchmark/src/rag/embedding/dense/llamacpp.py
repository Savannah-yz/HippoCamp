"""
LlamaCpp Dense Embedder implementation.

This embedder connects to a llama.cpp embedding service via HTTP.
It's ideal for:
- Using GGUF quantized models (Q4_K_M, Q8_0, etc.)
- Running embedding models as separate services
- Easy model swapping without reloading Python
- Metal/CUDA acceleration via llama.cpp

Example llama.cpp command to start embedding service:
    llama-server -m Qwen3-Embedding-0.6B-Q4_K_M.gguf \
        --embedding --pooling cls \
        --host 127.0.0.1 --port 8005 \
        -c 2048 -ngl 999
"""

import logging
import re
import asyncio
from typing import List, Optional
import numpy as np

from src.clients.embedding import EmbeddingServiceClient
from src.rag.embedding.base import BaseDenseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class LlamaCppDenseEmbedder(BaseDenseEmbedder):
    """
    LlamaCpp-based dense embedder via HTTP service.

    This embedder calls a llama.cpp embedding server running separately.
    The server handles model loading, quantization, and GPU acceleration.

    Features:
    - No model loading in Python (saves memory)
    - Easy to swap models (restart service with different GGUF)
    - Supports batch processing
    - Compatible with llama.cpp's /v1/embeddings endpoint
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8005",
        model: Optional[str] = None,
        dimension: int = 1024,
        max_batch_size: int = 32,
        split_on_failure: bool = True,
        timeout: float = 60.0,
    ):
        """
        Initialize LlamaCpp dense embedder.

        Args:
            endpoint: Base URL of llama.cpp embedding service
            model: Model identifier (for tracking/logging, optional)
            dimension: Expected embedding dimension
            timeout: HTTP request timeout in seconds
        """
        self.endpoint = endpoint
        self.model_name = model or "llamacpp"
        self.dimension = dimension
        self.max_batch_size = max_batch_size
        self.split_on_failure = split_on_failure
        self.timeout = timeout
        self._stats_lock = asyncio.Lock()
        self._stats = {"batch_halving": 0, "text_splits": 0}

        # Create HTTP client
        self.client = EmbeddingServiceClient(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
        )

        logger.info(
            f"Initialized LlamaCppDenseEmbedder: "
            f"endpoint={endpoint}, model={model}, dimension={dimension}, max_batch_size={max_batch_size}, "
            f"split_on_failure={split_on_failure}"
        )

    async def embed_dense(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embeddings via llama.cpp service.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape [N, dimension]
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            batches = [
                texts[i : i + self.max_batch_size]
                for i in range(0, len(texts), self.max_batch_size)
            ]
            all_embeddings: List[np.ndarray] = []
            for batch_idx, batch in enumerate(batches):
                embeddings = await self._embed_batch_with_fallback(batch)

                # Validate dimension
                if len(embeddings) > 0:
                    actual_dim = embeddings.shape[1]
                    if actual_dim != self.dimension:
                        logger.warning(
                            f"Embedding dimension mismatch: expected {self.dimension}, "
                            f"got {actual_dim}. Updating dimension."
                        )
                        self.dimension = actual_dim
                all_embeddings.append(embeddings)

            if not all_embeddings:
                return np.array([]).reshape(0, self.dimension)

            merged = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings via llama.cpp: shape={merged.shape}")
            await self._log_stats_if_any()
            return merged.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embeddings via llama.cpp: {e}", exc_info=True)
            raise

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
                "llama.cpp fallback stats: batch_halving=%s text_splits=%s",
                batch_halving,
                text_splits,
            )

    def _is_physical_batch_error(self, exc: Exception) -> bool:
        message = str(exc)
        # Prefer precise llama.cpp-style errors to avoid false positives.
        patterns = (
            r"input\s*\(\d+\s*tokens\)\s*is\s*too\s*large\s*to\s*process",
            r"increase\s+the\s+physical\s+batch\s+size",
            r"current\s+batch\s+size:\s*\d+",
            r"\bn_ubatch\b",
            r"\bn_batch\b",
            r"\btimeout\b",
            r"\breadtimeout\b",
            r"\bconnecttimeout\b",
            r"timed\s+out",
        )
        return any(re.search(p, message, flags=re.IGNORECASE) for p in patterns)

    async def _embed_batch_with_fallback(self, batch: List[str]) -> np.ndarray:
        try:
            return await self.client.encode(batch)
        except Exception as exc:
            if not self._is_physical_batch_error(exc):
                raise

            logger.warning(
                "Embedding batch size=%s failed due to physical batch limits. "
                "Trying halving fallback.",
                len(batch),
            )

            # Halve batch size until 1
            size = max(1, len(batch) // 2)
            while size >= 1:
                if size >= len(batch):
                    size = max(1, size // 2)
                    continue
                try:
                    await self._inc_stat("batch_halving")
                    return await self._embed_batched(batch, size)
                except Exception as fallback_exc:
                    if self._is_physical_batch_error(fallback_exc):
                        if size == 1:
                            break
                        size = max(1, size // 2)
                        continue
                    raise

            if self.split_on_failure:
                logger.warning("Falling back to per-text split embedding.")
                return await self._embed_texts_with_split(batch)

            raise

    async def _embed_batched(self, texts: List[str], batch_size: int) -> np.ndarray:
        batches = [
            texts[i : i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        embeddings_list: List[np.ndarray] = []
        for batch in batches:
            embeddings_list.append(await self.client.encode(batch))
        return np.vstack(embeddings_list) if embeddings_list else np.array([]).reshape(0, self.dimension)

    async def _embed_texts_with_split(self, texts: List[str]) -> np.ndarray:
        embeddings_list: List[np.ndarray] = []
        for text in texts:
            if not text or not text.strip():
                continue
            try:
                emb = await self.client.encode([text])
                if emb.size == 0:
                    continue
                embeddings_list.append(emb[0])
            except Exception as exc:
                if not self._is_physical_batch_error(exc):
                    raise
                logger.warning("Splitting oversized chunk during per-text fallback.")
                emb = await self._embed_single_with_split(text)
                if emb.size > 0:
                    embeddings_list.append(emb[0])
        if not embeddings_list:
            return np.array([]).reshape(0, self.dimension)
        return np.vstack(embeddings_list)

    async def _embed_single_with_split(self, text: str) -> np.ndarray:
        segments = [text]
        collected: List[np.ndarray] = []
        weights: List[float] = []

        while segments:
            segment = segments.pop(0)
            if not segment or not segment.strip():
                continue
            try:
                embedding = await self.client.encode([segment])
                if embedding.size == 0:
                    continue
                collected.append(embedding[0])
                weights.append(float(len(segment)))
            except Exception as exc:
                if not self._is_physical_batch_error(exc):
                    raise
                if len(segment) <= 1:
                    raise
                await self._inc_stat("text_splits")
                left, right = self._split_text(segment)
                segments = [left, right] + segments

        if not collected:
            return np.array([]).reshape(0, self.dimension)

        emb_matrix = np.vstack(collected)
        weight_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        averaged = (emb_matrix * weight_arr).sum(axis=0) / weight_arr.sum()
        return averaged.reshape(1, -1)

    def _split_text(self, text: str) -> tuple[str, str]:
        if len(text) <= 1:
            return text, ""

        mid = len(text) // 2
        split_candidates = ["\n\n", "\n", ". ", " "]
        for sep in split_candidates:
            idx = text.rfind(sep, 0, mid)
            if idx != -1:
                split_point = idx + len(sep)
                left = text[:split_point].strip()
                right = text[split_point:].strip()
                if left and right:
                    return left, right

        return text[:mid].strip(), text[mid:].strip()

    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Embed texts and return EmbeddingResult.

        Args:
            texts: List of text strings to embed
            return_sparse: Ignored (llama.cpp doesn't provide sparse embeddings)

        Returns:
            EmbeddingResult with dense_embeddings only
        """
        if return_sparse:
            logger.warning(
                "return_sparse=True but LlamaCppDenseEmbedder doesn't support sparse embeddings. "
                "Returning dense only."
            )

        dense_emb = await self.embed_dense(texts)
        return EmbeddingResult(dense_embeddings=dense_emb, sparse_embeddings=None)

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model_name

    async def aclose(self):
        """
        Cleanup resources (no-op for HTTP client).

        The HTTP client is stateless, so no cleanup needed.
        """
        pass
