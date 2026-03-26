"""
BGE Sparse Embedder implementation.
Uses FlagEmbedding's BGEM3FlagModel for generating sparse embeddings.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from threading import Lock

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

from src.rag.embedding.base import BaseSparseEmbedder

logger = logging.getLogger(__name__)


class BGESparseEmbedder(BaseSparseEmbedder):
    """
    BGE sparse embedder using BGEM3FlagModel.
    
    Note: BGE model is synchronous, so we use asyncio.to_thread to run it
    in a thread pool to avoid blocking the event loop.
    """
    
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
    ):
        """
        Initialize BGE sparse embedder.
        
        Args:
            model: BGE model name (default: "BAAI/bge-m3")
        """
        if BGEM3FlagModel is None:
            raise ImportError(
                "Please install 'FlagEmbedding' to use BGESparseEmbedder: "
                "pip install FlagEmbedding"
            )
        
        self.model = model
        self.client = BGEM3FlagModel(model)
        self._lock = Lock()  # Thread-safe lock for BGE model
        
        logger.info(f"Initialized BGESparseEmbedder: model={model}")
    
    def _safe_sparse_encode(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Synchronous sparse encoding with thread safety.
        BGE model is not thread-safe, so we use a lock.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dictionaries, each mapping token -> weight
        """
        with self._lock:
            result = self.client.encode(
                texts,
                return_dense=False,
                return_sparse=True,
            )
        return result["lexical_weights"]
    
    async def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Generate sparse embeddings for a list of texts.
        Uses asyncio.to_thread to run synchronous BGE model in thread pool.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dictionaries, each mapping token -> weight
            Example: [{"neural": 0.8, "network": 0.6}, ...]
        """
        if not texts:
            return []
        
        # Run synchronous BGE model in thread pool
        sparse_embeddings = await asyncio.to_thread(self._safe_sparse_encode, texts)
        
        logger.debug(f"Generated {len(sparse_embeddings)} sparse embeddings")
        return sparse_embeddings
    
    async def embed(self, texts: List[str], return_sparse: bool = True):
        """
        Embed texts and return EmbeddingResult.
        Note: BGE only provides sparse embeddings, dense will be empty.
        
        Args:
            texts: List of text strings to embed
            return_sparse: Whether to return sparse embeddings (always True for BGE)
            
        Returns:
            EmbeddingResult with sparse_embeddings only
        """
        from src.rag.embedding.base import EmbeddingResult
        import numpy as np
        
        sparse_emb = await self.embed_sparse(texts)
        # BGE doesn't provide dense embeddings, so we return empty array
        return EmbeddingResult(
            dense_embeddings=np.array([]).reshape(0, 0),  # Empty array
            sparse_embeddings=sparse_emb,
        )
    
    def get_dimension(self) -> Optional[int]:
        """Sparse embeddings are variable-length, so dimension is None."""
        return None
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

