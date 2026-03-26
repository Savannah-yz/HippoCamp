"""
Core abstractions for the vector store module.
Defines base classes for all vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    Provides a standardized interface for storing and retrieving vectors.
    Supports both dense and sparse embeddings for hybrid search.
    """
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        dense_embeddings: np.ndarray,
        sparse_embeddings: Optional[List[Dict[str, float]]] = None,
        chunks: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Store vectors with metadata.
        
        Args:
            ids: List of unique identifiers for each chunk
            dense_embeddings: numpy array of shape [N, dimension] for dense embeddings
            sparse_embeddings: Optional list of sparse embeddings (token -> weight dicts)
            chunks: Optional list of original chunk texts
            metadatas: Optional list of metadata dictionaries for each chunk
            
        Raises:
            ValueError: If input lengths don't match
        """
        pass
    
    @abstractmethod
    async def query(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        hybrid_search: bool = False,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            dense_embedding: Query dense embedding vector [dimension]
            sparse_embedding: Optional query sparse embedding (token -> weight dict)
            top_k: Number of results to return
            hybrid_search: Whether to use hybrid search (requires sparse_embedding)
            filter: Optional metadata filter expression (e.g., "file_id == 'xxx'")
            
        Returns:
            List of result dictionaries, each containing:
            - id: str
            - chunk: str (original text)
            - metadata: Dict[str, Any]
            - score: float (similarity score)
        """
        pass
    
    @abstractmethod
    async def filter_existing(
        self,
        ids: List[str],
    ) -> List[str]:
        """
        Check which IDs already exist in the store.
        
        Args:
            ids: List of IDs to check
            
        Returns:
            List of IDs that already exist in the store
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[str] = None,
    ) -> int:
        """
        Delete vectors from the store.
        
        Args:
            ids: Optional list of IDs to delete
            filter: Optional metadata filter expression
            
        Returns:
            Number of deleted vectors
            
        Note:
            Either ids or filter must be provided
        """
        pass

