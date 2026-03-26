"""
Hybrid Embedder that combines dense and sparse embeddings.
Uses composition pattern to combine multiple embedders.
"""

import logging
from typing import List, Optional

from src.rag.embedding.base import BaseDenseEmbedder, BaseSparseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """
    Hybrid embedder that combines dense and sparse embeddings.
    
    Uses composition pattern (not inheritance) to combine:
    - A dense embedder (required)
    - A sparse embedder (optional)
    
    Both embedders run in parallel for efficiency.
    """
    
    def __init__(
        self,
        dense_embedder: BaseDenseEmbedder,
        sparse_embedder: Optional[BaseSparseEmbedder] = None,
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            dense_embedder: Dense embedder instance (required)
            sparse_embedder: Sparse embedder instance (optional)
        """
        if dense_embedder is None:
            raise ValueError("dense_embedder is required")
        
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        
        logger.info(
            f"Initialized HybridEmbedder: "
            f"dense={dense_embedder.get_model_name()}, "
            f"sparse={sparse_embedder.get_model_name() if sparse_embedder else None}"
        )
    
    async def embed(
        self,
        texts: List[str],
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """
        Generate both dense and sparse embeddings (if available).
        
        Args:
            texts: List of text strings to embed
            return_sparse: Whether to generate sparse embeddings (if sparse_embedder is available)
            
        Returns:
            EmbeddingResult containing:
            - dense_embeddings: Always present
            - sparse_embeddings: Present if sparse_embedder is set and return_sparse=True
        """
        import asyncio
        
        # Always generate dense embeddings
        dense_task = self.dense_embedder.embed_dense(texts)
        
        # Generate sparse embeddings if requested and available
        sparse_task = None
        if return_sparse and self.sparse_embedder is not None:
            sparse_task = self.sparse_embedder.embed_sparse(texts)
        
        # Run both in parallel
        if sparse_task is not None:
            dense_emb, sparse_emb = await asyncio.gather(dense_task, sparse_task)
        else:
            dense_emb = await dense_task
            sparse_emb = None
        
        return EmbeddingResult(
            dense_embeddings=dense_emb,
            sparse_embeddings=sparse_emb,
        )
    
    def get_dimension(self) -> int:
        """Return dense embedding dimension."""
        return self.dense_embedder.get_dimension()
    
    def get_model_name(self) -> str:
        """Return model identifiers."""
        dense_name = self.dense_embedder.get_model_name()
        sparse_name = self.sparse_embedder.get_model_name() if self.sparse_embedder else None
        if sparse_name:
            return f"hybrid({dense_name}+{sparse_name})"
        return f"dense_only({dense_name})"
    
    async def aclose(self):
        """Close the embedder connections."""
        # Close dense embedder if it has aclose method
        if hasattr(self.dense_embedder, 'aclose'):
            await self.dense_embedder.aclose()
        
        # Close sparse embedder if it has aclose method
        if self.sparse_embedder and hasattr(self.sparse_embedder, 'aclose'):
            await self.sparse_embedder.aclose()

