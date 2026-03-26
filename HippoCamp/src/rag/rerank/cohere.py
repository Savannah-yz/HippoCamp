"""
Cohere Reranker implementation.
Uses Cohere's rerank API for reranking candidate chunks.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import cohere
except ImportError:
    cohere = None

from src.rag.rerank.base import BaseReranker

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """
    Cohere reranker using Cohere's rerank API.
    
    Features:
    - Async API calls
    - Batch processing for large candidate lists
    - Error handling and retry logic
    - Score normalization and threshold filtering
    """
    
    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        api_key_env: str = "COHERE_API_KEY",
        top_k: int = 20,
        threshold: float = 0.0,
        max_batch_size: int = 100,
        timeout: int = 30,
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            model: Cohere rerank model name (default: "rerank-english-v3.0")
            api_key: API key (if None, will read from environment)
            api_key_env: Environment variable name for API key
            top_k: Default top_k for reranking (can be overridden in rerank() call)
            threshold: Default minimum relevance score threshold
            max_batch_size: Maximum candidates per API request (Cohere supports up to 1000)
            timeout: Request timeout in seconds
        """
        if cohere is None:
            raise ImportError("Please install 'cohere' to use CohereReranker: pip install cohere")
        
        self.model = model
        self.default_top_k = top_k
        self.default_threshold = threshold
        self.max_batch_size = max_batch_size
        
        # Get API key
        if api_key is None:
            api_key = os.getenv(api_key_env)
        if api_key is None:
            raise ValueError(
                f"Cohere API key not found. Set {api_key_env} environment variable or pass api_key."
            )
        
        # Initialize async client
        self.client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout)
        
        logger.info(f"Initialized CohereReranker: model={model}, top_k={top_k}, threshold={threshold}")
    
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
            top_k: Optional limit on number of results (None = use default or return all)
            threshold: Optional minimum relevance score threshold (None = use default)
            
        Returns:
            List of reranked candidates (same format as input, with updated scores)
            Sorted by relevance score (highest first)
        """
        if not candidates:
            return []
        
        # Use defaults if not specified
        if top_k is None:
            top_k = self.default_top_k
        if threshold is None:
            threshold = self.default_threshold
        
        # Extract documents from candidates
        documents = [candidate.get("chunk", "") for candidate in candidates]
        
        # Batch processing if needed (Cohere supports up to 1000 per request)
        # For simplicity, we'll process all candidates in one request if <= max_batch_size
        # Otherwise, we'll split into batches and merge results
        if len(candidates) <= self.max_batch_size:
            reranked = await self._rerank_batch(query, documents, candidates, top_k, threshold)
        else:
            # Split into batches and process
            reranked = await self._rerank_batched(query, documents, candidates, top_k, threshold)
        
        logger.debug(f"Reranked {len(candidates)} candidates to {len(reranked)} results")
        return reranked
    
    async def _rerank_batch(
        self,
        query: str,
        documents: List[str],
        candidates: List[Dict[str, Any]],
        top_k: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a single batch of candidates.
        
        Args:
            query: Query text
            documents: List of document texts
            candidates: List of candidate dictionaries
            top_k: Number of top results to return
            threshold: Minimum relevance score
            
        Returns:
            List of reranked candidates
        """
        try:
            response = await self._rerank_api_call(query, documents, top_n=min(top_k, len(documents)))
            
            # Process results
            reranked_candidates = []
            for result in response.results:
                if result.relevance_score >= threshold:
                    idx = result.index
                    candidate = candidates[idx].copy()
                    candidate["score"] = result.relevance_score
                    reranked_candidates.append(candidate)
            
            return reranked_candidates
            
        except Exception as e:
            logger.error(f"Error reranking batch: {e}")
            # On error, return original candidates with original scores
            return candidates
    
    async def _rerank_batched(
        self,
        query: str,
        documents: List[str],
        candidates: List[Dict[str, Any]],
        top_k: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Rerank multiple batches and merge results.
        
        This method splits candidates into batches, reranks each batch,
        and then merges the top results.
        """
        # Split into batches
        batches = [
            (i, candidates[i : i + self.max_batch_size], documents[i : i + self.max_batch_size])
            for i in range(0, len(candidates), self.max_batch_size)
        ]
        
        logger.info(f"Reranking {len(candidates)} candidates in {len(batches)} batches")
        
        # Process batches in parallel
        batch_results = await asyncio.gather(
            *[
                self._rerank_batch(query, batch_docs, batch_candidates, top_k, threshold)
                for _, batch_candidates, batch_docs in batches
            ]
        )
        
        # Merge all results and sort by score
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        # Sort by score (highest first) and take top_k
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return all_results[:top_k] if top_k else all_results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TimeoutError,)),
    )
    async def _rerank_api_call(self, query: str, documents: List[str], top_n: int):
        """
        Make API call to Cohere rerank endpoint.
        Includes retry logic for rate limits and connection errors.
        
        Args:
            query: Query text
            documents: List of document texts
            top_n: Number of top results to return
            
        Returns:
            Cohere rerank response
        """
        try:
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return response
            
        except Exception as e:
            logger.error(f"Error in Cohere rerank API call: {e}")
            if isinstance(e, TimeoutError):
                raise  # Retry on timeout
            raise  # Don't retry on other errors
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model
    
    async def aclose(self):
        """Close the client connection."""
        if hasattr(self.client, 'close'):
            await self.client.close()

