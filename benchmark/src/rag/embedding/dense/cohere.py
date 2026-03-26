"""
Cohere Dense Embedder implementation.
Uses Cohere's embed-v4.0 model for generating dense embeddings.
"""

import os
import asyncio
import logging
from typing import List, Optional
import numpy as np
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import cohere
    # Cohere 5.x doesn't have CohereError, but has specific error classes
    # We'll catch cohere errors by checking if exception is from cohere module
    # For retry logic, we'll catch common cohere exceptions
    CohereError = Exception  # Will be used to check if error is from cohere
except ImportError:
    cohere = None
    CohereError = Exception

from src.rag.embedding.base import BaseDenseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class CohereDenseEmbedder(BaseDenseEmbedder):
    """
    Cohere dense embedder using embed-v4.0 model.
    
    Features:
    - Async API calls
    - Automatic batching
    - Error handling and retry logic
    - Rate limiting via semaphore
    """
    
    def __init__(
        self,
        model: str = "embed-v4.0",
        dimension: int = 1536,
        api_key: Optional[str] = None,
        api_key_env: str = "COHERE_API_KEY",
        max_batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
    ):
        """
        Initialize Cohere dense embedder.
        
        Args:
            model: Cohere model name (default: "embed-v4.0")
            dimension: Embedding dimension (default: 1536 for embed-v4.0)
            api_key: API key (if None, will read from environment)
            api_key_env: Environment variable name for API key
            max_batch_size: Maximum texts per API request
            max_concurrent: Maximum concurrent API requests
        """
        if cohere is None:
            raise ImportError("Please install 'cohere' to use CohereDenseEmbedder: pip install cohere")
        
        self.model = model
        self.dimension = dimension
        if max_batch_size is None or max_concurrent is None:
            raise ValueError("CohereDenseEmbedder requires max_batch_size and max_concurrent.")

        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        
        # Get API key
        if api_key is None:
            api_key = os.getenv(api_key_env)
        if api_key is None:
            raise ValueError(f"Cohere API key not found. Set {api_key_env} environment variable or pass api_key.")
        
        # Initialize async client with longer timeout
        self.client = cohere.AsyncClientV2(api_key=api_key, timeout=120)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Initialized CohereDenseEmbedder: model={model}, dimension={dimension}")
    
    async def embed_dense(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape [N, dimension] where N = len(texts)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        # Batch texts
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        
        logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches")
        
        # Process batches with concurrency control
        async def process_batch(batch):
            async with self._semaphore:
                return await self._embed_batch(batch)
        
        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Concatenate all results
        all_embeddings = np.vstack(results)
        
        logger.debug(f"Generated embeddings shape: {all_embeddings.shape}")
        return all_embeddings.astype(np.float32)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout)),
    )
    async def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a single batch of texts.
        Includes retry logic for rate limits and connection errors.
        
        Args:
            texts: List of texts in this batch
            
        Returns:
            numpy array of shape [len(texts), dimension]
        """
        try:
            response = await self.client.embed(
                model=self.model,
                texts=texts,
                input_type="classification",
                embedding_types=["float"],
            )
            
            # Extract embeddings from response
            # Cohere returns response.embeddings.float_ which is a list of lists
            # Each element is a list of floats representing one embedding vector
            embeddings_list = response.embeddings.float_
            
            # Convert to numpy array
            # Shape should be [batch_size, dimension]
            embeddings = np.array(embeddings_list, dtype=np.float32)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            # Re-raise to trigger retry if it's a retryable error
            if isinstance(e, (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout)):
                raise
            # For other errors (including cohere API errors), raise without retry
            raise
    
    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Embed texts and return EmbeddingResult.
        Note: CohereDenseEmbedder doesn't support sparse embeddings.
        
        Args:
            texts: List of text strings to embed
            return_sparse: Ignored (Cohere doesn't provide sparse embeddings)
            
        Returns:
            EmbeddingResult with dense_embeddings only
        """
        dense_emb = await self.embed_dense(texts)
        return EmbeddingResult(dense_embeddings=dense_emb, sparse_embeddings=None)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model
    
    async def aclose(self):
        """Close the client connection."""
        if hasattr(self.client, 'close'):
            await self.client.close()
