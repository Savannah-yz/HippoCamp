"""
Standard RAG Provider - Retrieve, Rerank, Return.

This provider implements the standard RAG pipeline from HippoCamp:
1. Retrieve documents using vector similarity
2. Optionally rerank results
3. Return top-k relevant chunks

Pipeline: Retrieve → Rerank (optional) → Return

This is equivalent to HippoCamp's standard_rag.py implementation.

Usage:
    from src.providers import StandardRAGProvider, ProviderConfig

    config = ProviderConfig(
        name="standard_rag",
        params={
            "top_k": 20,
            "use_reranker": True,
            "rerank_top_k": 5,
        }
    )

    provider = StandardRAGProvider(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
    )

    await provider.setup()
    result = await provider.retrieve("What is machine learning?")
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..utils import format_query_for_embedding
from ..base import (
    RetrievalProvider,
    ProviderConfig,
    RetrievalResult,
    ChunkResult,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class StandardRAGProvider(RetrievalProvider):
    """
    Standard RAG retrieval provider.

    Implements the classic RAG pipeline:
    - Retrieve: Vector similarity search
    - Rerank: Optional reordering by relevance
    - Return: Top-k results

    This matches the HippoCamp StandardRAG implementation for compatibility.

    Attributes:
        embedder: Embedding service for query vectorization
        vector_store: Vector database for similarity search
        reranker: Optional reranker for result refinement
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        reranker=None,
    ):
        """
        Initialize the Standard RAG provider.

        Args:
            config: Provider configuration with params:
                - top_k (int): Number of documents to retrieve (default: 20)
                - use_reranker (bool): Whether to use reranker (default: True)
                - rerank_top_k (int): Number of documents after reranking (default: 5)
            embedder: Embedding service instance
            vector_store: Vector store instance
            reranker: Optional reranker instance
        """
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

        # Extract config params with defaults
        params = config.params
        self.default_top_k = params.get("top_k", 20)
        self.use_reranker = params.get("use_reranker", True)
        self.rerank_top_k = params.get("rerank_top_k", 5)

    async def setup(self):
        """Initialize provider components."""
        if self._initialized:
            return

        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())

        logger.info(
            f"StandardRAGProvider initialized "
            f"(top_k={self.default_top_k}, reranker={self.reranker is not None})"
        )
        self._initialized = True

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[ChunkResult],
        top_k: int,
    ) -> List[ChunkResult]:
        """
        Rerank chunks by relevance.

        Args:
            query: Search query
            chunks: Chunks to rerank
            top_k: Number of results after reranking

        Returns:
            Reranked ChunkResult list
        """
        if not self.reranker or not chunks:
            return chunks[:top_k]

        # Convert to reranker format
        candidates = [
            {
                "id": c.id,
                "chunk": c.content,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        # Rerank
        reranked = await self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        # Convert back to ChunkResult
        return [
            ChunkResult(
                id=r.get("id", ""),
                content=r.get("chunk", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in reranked
        ]

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        skip_rerank: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using Standard RAG pipeline.

        Pipeline: Retrieve → Rerank (optional) → Return

        Args:
            query: Search query
            top_k: Number of results to retrieve (overrides config)
            filter: Optional metadata filter
            skip_rerank: Skip reranking even if enabled
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with retrieved chunks
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        logger.debug(f"StandardRAG: Processing query: {query[:50]}...")

        # Initialize latency and step tracking
        latency_breakdown = {}
        thinking_steps = []
        step_counts = {
            "search_count": 0,
            "rerank_count": 0,
        }
        total_start = time.time()
        step_counter = 0

        try:
            # Step 1: Embed query
            step_counter += 1
            embed_start = time.time()
            thinking_steps.append({
                "step_id": f"embed_{step_counter}",
                "step_type": "embed",
                "title": "Embedding Query",
                "status": "running",
                "summary": "Converting query to vector representation...",
                "timestamp_ms": 0,
            })

            query_for_embed = format_query_for_embedding(query)
            query_embedding = await self.embedder.embed([query_for_embed])
            embed_latency = (time.time() - embed_start) * 1000
            latency_breakdown["embed_query_ms"] = embed_latency

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Query embedded in {embed_latency:.0f}ms"

            dense_embedding = (
                query_embedding.dense_embeddings[0]
                if query_embedding.dense_embeddings is not None
                and len(query_embedding.dense_embeddings) > 0
                else None
            )
            sparse_embedding = (
                query_embedding.sparse_embeddings[0]
                if query_embedding.sparse_embeddings is not None
                and len(query_embedding.sparse_embeddings) > 0
                else None
            )

            # Step 2: Vector search
            step_counter += 1
            search_start = time.time()
            thinking_steps.append({
                "step_id": f"retrieve_{step_counter}",
                "step_type": "retrieve",
                "title": "Searching Documents",
                "status": "running",
                "summary": f"Searching for top-{top_k} documents...",
                "timestamp_ms": (search_start - total_start) * 1000,
            })

            logger.debug(f"StandardRAG: Retrieving top-{top_k} documents")
            search_results = await self.vector_store.query(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=top_k,
                filter=filter,
            )

            retrieved_chunks = [
                ChunkResult(
                    id=r.get("id", ""),
                    content=r.get("chunk", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in search_results
            ]

            search_latency = (time.time() - search_start) * 1000
            latency_breakdown["retrieve_ms"] = search_latency
            step_counts["search_count"] = 1

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Found {len(retrieved_chunks)} documents"
            thinking_steps[-1]["metadata"] = {"results_count": len(retrieved_chunks)}

            logger.debug(f"StandardRAG: Retrieved {len(retrieved_chunks)} documents")

            if not retrieved_chunks:
                logger.warning("StandardRAG: No documents retrieved")
                latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                step_counts["total_iterations"] = 1
                return RetrievalResult(
                    chunks=[],
                    query=query,
                    metadata={"provider": self.get_name(), "error": "no_documents"},
                    latency_breakdown=latency_breakdown,
                    thinking_steps=thinking_steps,
                    step_counts=step_counts,
                )

            # Step 3: Rerank (optional)
            final_chunks = retrieved_chunks
            reranked = False

            if self.use_reranker and self.reranker and not skip_rerank:
                step_counter += 1
                rerank_start = time.time()
                thinking_steps.append({
                    "step_id": f"rerank_{step_counter}",
                    "step_type": "rerank",
                    "title": "Reranking Results",
                    "status": "running",
                    "summary": f"Reranking {len(retrieved_chunks)} documents to top-{self.rerank_top_k}...",
                    "timestamp_ms": (rerank_start - total_start) * 1000,
                })

                logger.debug(f"StandardRAG: Reranking to top-{self.rerank_top_k}")
                final_chunks = await self._rerank_chunks(query, retrieved_chunks, self.rerank_top_k)
                reranked = True

                rerank_latency = (time.time() - rerank_start) * 1000
                latency_breakdown["rerank_ms"] = rerank_latency
                step_counts["rerank_count"] = 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Reranked to {len(final_chunks)} documents"
                thinking_steps[-1]["metadata"] = {"final_count": len(final_chunks)}

                logger.debug(f"StandardRAG: Reranked to {len(final_chunks)} documents")
            else:
                # Just take top rerank_top_k without reranking
                final_chunks = retrieved_chunks[: self.rerank_top_k]

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
            step_counts["total_iterations"] = 1

            return RetrievalResult(
                chunks=final_chunks,
                query=query,
                iterations=1,
                metadata={
                    "provider": self.get_name(),
                    "retrieved_count": len(retrieved_chunks),
                    "final_count": len(final_chunks),
                    "reranked": reranked,
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"StandardRAG retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        """Get provider name."""
        return "standard_rag"

    async def aclose(self):
        """Cleanup resources."""
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
