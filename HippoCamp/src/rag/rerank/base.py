"""
Core abstractions for the reranking module.
Defines base classes for all rerankers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    Provides a standardized interface for reranking candidate chunks
    based on query relevance.
    """
    
    @abstractmethod
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
            candidates: List of candidate dictionaries, each containing:
                - id: str
                - chunk: str (text content)
                - metadata: Dict[str, Any]
                - score: float (initial retrieval score, optional)
            top_k: Optional limit on number of results (None = return all)
            threshold: Optional minimum relevance score threshold
            
        Returns:
            List of reranked candidates (same format as input, with updated scores)
            Sorted by relevance score (highest first)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return model identifier.
        
        Returns:
            Model name/identifier (e.g., "rerank-english-v3.0")
        """
        pass
    
    @abstractmethod
    async def aclose(self):
        """
        Close the reranker connections (if needed).
        
        This method should be called when the reranker is no longer needed
        to properly clean up resources (e.g., close API clients, release GPU memory).
        """
        pass

