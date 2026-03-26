"""
Vector Search Provider - Simple vector similarity retrieval.

This provider performs basic vector similarity search using embeddings
and optionally applies reranking.

Pipeline: Embed Query → Vector Search → (Optional Rerank) → Return

Usage:
    from src.providers import VectorSearchProvider, ProviderConfig

    config = ProviderConfig(
        name="vector_search",
        params={
            "top_k": 20,
            "use_reranker": True,
            "rerank_top_k": 5,
        }
    )

    provider = VectorSearchProvider(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,  # Optional
    )

    await provider.setup()
    result = await provider.retrieve("What is machine learning?", top_k=10)
"""

import logging
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


class VectorSearchProvider(RetrievalProvider):
    """
    Simple vector similarity search provider.

    This is the most basic retrieval method:
    1. Embed the query using the configured embedder
    2. Search the vector store for similar chunks
    3. Optionally rerank results
    4. Return top-k chunks

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
        Initialize the Vector Search provider.

        Args:
            config: Provider configuration with params:
                - top_k (int): Default number of results to retrieve (default: 20)
                - use_reranker (bool): Whether to use reranker (default: True)
                - rerank_top_k (int): Number of results after reranking (default: 5)
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

        logger.info(f"VectorSearchProvider initialized (reranker: {self.reranker is not None})")
        self._initialized = True

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
        Retrieve relevant chunks using vector similarity search.

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
        logger.debug(f"VectorSearch: Retrieving top-{top_k} for query: {query[:50]}...")

        try:
            # Step 1: Embed query (with instruction prefix for retrieval)
            query_for_embed = format_query_for_embedding(query)
            query_embedding = await self.embedder.embed([query_for_embed])
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

            # Step 2: Search vector store
            search_results = await self.vector_store.query(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=top_k,
                filter=filter,
            )

            # Convert to ChunkResult objects
            chunks = []
            for r in search_results:
                chunks.append(
                    ChunkResult(
                        id=r.get("id", ""),
                        content=r.get("chunk", ""),
                        score=r.get("score", 0.0),
                        metadata=r.get("metadata", {}),
                    )
                )

            logger.debug(f"VectorSearch: Retrieved {len(chunks)} chunks")

            # Step 3: Optional reranking
            if self.use_reranker and self.reranker and not skip_rerank and chunks:
                logger.debug(f"VectorSearch: Reranking to top-{self.rerank_top_k}")

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

                reranked = await self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=self.rerank_top_k,
                )

                # Convert back to ChunkResult
                chunks = [
                    ChunkResult(
                        id=r.get("id", ""),
                        content=r.get("chunk", ""),
                        score=r.get("score", 0.0),
                        metadata=r.get("metadata", {}),
                    )
                    for r in reranked
                ]

                logger.debug(f"VectorSearch: Reranked to {len(chunks)} chunks")

            return RetrievalResult(
                chunks=chunks,
                query=query,
                iterations=1,
                metadata={
                    "provider": self.get_name(),
                    "top_k": top_k,
                    "reranked": self.use_reranker and self.reranker is not None and not skip_rerank,
                },
            )

        except Exception as e:
            logger.error(f"VectorSearch retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        """Get provider name."""
        return "vector_search"

    async def aclose(self):
        """Cleanup resources."""
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
