"""
BGE Reranker implementation.
Uses BGE reranker models (local) for reranking candidate chunks.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

try:
    from sentence_transformers import CrossEncoder
    import torch
except ImportError:
    CrossEncoder = None
    torch = None

from src.rag.rerank.base import BaseReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """
    BGE reranker using local sentence-transformers models.
    
    Features:
    - Local model (no API calls, faster for large batches)
    - GPU acceleration support
    - Batch processing for efficiency
    - Lower latency (no network overhead)
    """
    
    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize BGE reranker.
        
        Args:
            model: HuggingFace model name (default: "BAAI/bge-reranker-v2-m3")
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            batch_size: Batch size for processing candidates
        """
        if CrossEncoder is None:
            raise ImportError(
                "Please install 'sentence-transformers' to use BGEReranker: "
                "pip install sentence-transformers"
            )
        
        self.model_name = model
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        
        # Load model
        logger.info(f"Loading BGE reranker model: {model} on {device}")
        self.model = CrossEncoder(model, device=device)
        logger.info(f"Initialized BGEReranker: model={model}, device={device}, batch_size={batch_size}")
    
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
        if not candidates:
            return []
        
        # Extract documents from candidates
        documents = [candidate.get("chunk", "") for candidate in candidates]
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Run reranking in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _predict():
            return self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        scores = await loop.run_in_executor(None, _predict)
        
        # Update candidates with scores
        reranked_candidates = []
        for candidate, score in zip(candidates, scores):
            # Normalize score (CrossEncoder scores can be negative, normalize to 0-1)
            # Different models may have different score ranges, so we'll use raw scores
            # and let threshold handle filtering
            normalized_score = float(score)
            
            if threshold is None or normalized_score >= threshold:
                candidate_copy = candidate.copy()
                candidate_copy["score"] = normalized_score
                reranked_candidates.append(candidate_copy)
        
        # Sort by score (highest first)
        reranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            reranked_candidates = reranked_candidates[:top_k]
        
        logger.debug(f"Reranked {len(candidates)} candidates to {len(reranked_candidates)} results")
        return reranked_candidates
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model_name
    
    async def aclose(self):
        """
        Close the reranker and release resources.
        
        For local models, this mainly releases GPU memory if applicable.
        """
        # Clear model from memory
        if hasattr(self, 'model'):
            del self.model
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.debug("BGE reranker closed and resources released")

