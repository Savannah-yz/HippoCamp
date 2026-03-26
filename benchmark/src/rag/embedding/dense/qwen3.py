"""
Qwen3 Dense Embedder implementation.
Uses Qwen3-embedding-0.6b model for generating dense embeddings.
"""

import os
import asyncio
import logging
from typing import List, Optional
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
except ImportError:
    AutoModel = None
    AutoTokenizer = None
    torch = None

from src.rag.embedding.base import BaseDenseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class Qwen3DenseEmbedder(BaseDenseEmbedder):
    """
    Qwen3 dense embedder using qwen3-embedding-0.6b model.
    
    Features:
    - Local model (no API needed)
    - Batch processing
    - GPU support if available
    """
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-embedding-0.6b",
        device: Optional[str] = None,
        max_batch_size: Optional[int] = None,
    ):
        """
        Initialize Qwen3 dense embedder.
        
        Args:
            model: Model name/path (default: "Qwen/Qwen3-embedding-0.6b")
            device: Device to use ("cuda", "cpu", or None for auto)
            max_batch_size: Maximum texts per batch
        """
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                "Please install 'transformers' and 'torch' to use Qwen3DenseEmbedder: "
                "pip install transformers torch"
            )
        
        if max_batch_size is None:
            raise ValueError("Qwen3DenseEmbedder requires max_batch_size.")

        self.model_name = model
        self.max_batch_size = max_batch_size
        
        # Determine device
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading Qwen3 model: {model} on {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model)
            self.model.to(device)
            self.model.eval()
            logger.info(f"✅ Qwen3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            raise
        
        # Get dimension from model config
        try:
            self.dimension = self.model.config.hidden_size
        except AttributeError:
            # Fallback dimension for qwen3-embedding-0.6b
            self.dimension = 768
        
        logger.info(f"Initialized Qwen3DenseEmbedder: model={model}, dimension={self.dimension}, device={device}")
    
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
        
        # Process in batches
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        
        logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches")
        
        # Process batches
        all_embeddings = []
        for batch in batches:
            embeddings = await asyncio.to_thread(self._embed_batch, batch)
            all_embeddings.append(embeddings)
        
        # Concatenate all results
        result = np.vstack(all_embeddings)
        logger.debug(f"Generated embeddings shape: {result.shape}")
        return result.astype(np.float32)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a single batch of texts (runs in thread pool).
        
        Args:
            texts: List of texts in this batch
            
        Returns:
            numpy array of shape [len(texts), dimension]
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling or CLS token
                # For Qwen3, typically use last_hidden_state and pool
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                else:
                    # Fallback: use first token
                    embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings (common practice)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    async def embed(self, texts: List[str], return_sparse: bool = False) -> EmbeddingResult:
        """
        Embed texts and return EmbeddingResult.
        
        Args:
            texts: List of text strings to embed
            return_sparse: Ignored (Qwen3 doesn't provide sparse embeddings)
            
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
        return self.model_name
