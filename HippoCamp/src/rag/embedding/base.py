"""
Core abstractions for the embedding module.
Defines base classes for all embedders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class EmbeddingResult:
    """
    Standardized result format for embedding operations.
    """
    dense_embeddings: np.ndarray  # Shape: [N, dimension]
    sparse_embeddings: Optional[List[Dict[str, float]]] = None  # List of token->weight dicts


class BaseEmbedder(ABC):
    """
    Top-level abstract base class for all embedders.
    Provides the common interface for embedding operations.
    """
    
    @abstractmethod
    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Main embedding interface.
        
        Args:
            texts: List of text strings to embed
            return_sparse: Whether to return sparse embeddings (if supported)
            
        Returns:
            EmbeddingResult containing dense_embeddings and optionally sparse_embeddings
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the embedding dimension.
        
        Returns:
            Dimension of the dense embeddings (e.g., 1536 for Cohere embed-v4.0)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model identifier.
        
        Returns:
            Model name/identifier (e.g., "embed-v4.0")
        """
        pass


class BaseDenseEmbedder(BaseEmbedder):
    """
    Abstract base class for dense embedding providers.
    Dense embeddings are fixed-dimensional vectors (e.g., 1536-dim for Cohere).
    """
    
    @abstractmethod
    async def embed_dense(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape [N, dimension] where N = len(texts)
        """
        pass
    
    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Default implementation: only returns dense embeddings.
        Subclasses can override if they support sparse embeddings.
        """
        dense_emb = await self.embed_dense(texts)
        return EmbeddingResult(dense_embeddings=dense_emb, sparse_embeddings=None)


class BaseSparseEmbedder(BaseEmbedder):
    """
    Abstract base class for sparse embedding providers.
    Sparse embeddings are variable-length dictionaries mapping tokens to weights.
    """
    
    @abstractmethod
    async def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Generate sparse embeddings (lexical/BM25-style).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dictionaries, each mapping token -> weight
            Example: [{"neural": 0.8, "network": 0.6}, ...]
        """
        pass
    
    def get_dimension(self) -> Optional[int]:
        """
        Sparse embeddings are variable-length, so dimension may be None.
        """
        return None

