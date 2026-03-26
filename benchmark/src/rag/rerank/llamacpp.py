"""
LlamaCpp-based reranker using HTTP service.

This reranker connects to a running llama.cpp server with reranking support.
It uses the /v1/rerank endpoint which is compatible with llama.cpp's implementation.
"""

import logging
from typing import Any, Dict, List, Optional

from src.clients.rerank import RerankServiceClient
from .base import BaseReranker

logger = logging.getLogger(__name__)


class LlamaCppReranker(BaseReranker):
    """
    Reranker that uses a llama.cpp HTTP service.

    Requires a running llama.cpp server with a reranker model loaded.
    Start with:
        ./services/start_rerank_service.sh /path/to/reranker.gguf
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8006",
        model: str = "default",
        timeout: float = 30.0,
    ):
        """
        Initialize LlamaCpp reranker.

        Args:
            endpoint: URL of the llama.cpp rerank service
            model: Model identifier (for logging, not used by service)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.client = RerankServiceClient(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
        )
        logger.info(f"Initialized LlamaCppReranker with endpoint: {endpoint}")

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks based on query relevance.

        Args:
            query: Query text string
            candidates: List of candidate dictionaries with 'chunk' field
            top_k: Optional limit on number of results
            threshold: Optional minimum relevance score threshold

        Returns:
            List of reranked candidates with updated scores
        """
        if not candidates:
            return []

        # Extract document texts from candidates
        documents = [c.get("chunk", "") for c in candidates]

        # Determine top_k (default to all if not specified)
        k = top_k if top_k is not None else len(candidates)

        # Call rerank service
        ranked = await self.client.rerank(
            query=query,
            documents=documents,
            top_k=k,
        )

        # Map results back to candidates with scores
        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            if 0 <= idx < len(candidates):
                candidate = candidates[idx].copy()
                candidate["rerank_score"] = score
                # Preserve original retrieval score if present
                if "score" in candidate:
                    candidate["retrieval_score"] = candidate["score"]
                candidate["score"] = score
                results.append(candidate)

        # Filter by threshold if specified
        if threshold is not None:
            results = [r for r in results if r.get("score", 0) >= threshold]

        # Sort by score descending (should already be sorted, but ensure)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(
            f"Reranked {len(candidates)} candidates → {len(results)} results "
            f"(top_k={k}, threshold={threshold})"
        )

        return results

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

    async def aclose(self):
        """Close the reranker (no-op for HTTP client)."""
        pass
