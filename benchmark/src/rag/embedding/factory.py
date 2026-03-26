"""
Factory for creating embedders from configuration.
Supports config-driven creation for easy ablation studies.
"""

import os
import logging
from typing import Dict, Any, Optional

from src.rag.embedding.base import BaseDenseEmbedder, BaseSparseEmbedder
from src.rag.embedding.dense.cohere import CohereDenseEmbedder
from src.rag.embedding.dense.deepinfra import DeepInfraOpenAIEmbedder
from src.rag.embedding.dense.llamacpp import LlamaCppDenseEmbedder
from src.rag.embedding.hybrid import HybridEmbedder

# Lazy imports to avoid numpy compatibility issues
Qwen3DenseEmbedder = None
BGESparseEmbedder = None

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """
    Factory for creating embedders from configuration.
    
    Configuration format:
    {
        "type": "dense_only" | "hybrid",
        "dense": {
            "provider": "cohere" | "qwen3" | "llamacpp",
            "model": "embed-v4.0",
            "dimension": 1536,
            "api_key_env": "COHERE_API_KEY",  # For cohere only
            "max_batch_size": 32,
            "endpoint": "http://127.0.0.1:8005",  # For llamacpp only
            "timeout": 60.0  # For llamacpp only
        },
        "sparse": {
            "provider": "bge",  # Optional
            "model": "BAAI/bge-m3"
        }
    }

    Provider-specific configs:
    - cohere: Requires api_key_env, model, dimension
    - qwen3: Loads model via transformers (local), requires model, device
    - llamacpp: Calls HTTP service, requires endpoint, model, dimension
    """
    
    @staticmethod
    def create(config: Dict[str, Any]):
        """
        Create an embedder from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Embedder instance (HybridEmbedder or BaseDenseEmbedder)
        """
        embedder_type = config.get("type", "dense_only")
        
        # Create dense embedder
        dense_config = config.get("dense", {})
        if not dense_config:
            raise ValueError("'dense' configuration is required")
        
        dense_embedder = EmbedderFactory._create_dense_embedder(dense_config)
        
        # Create sparse embedder (optional)
        sparse_embedder = None
        if embedder_type == "hybrid":
            sparse_config = config.get("sparse")
            if sparse_config:
                sparse_embedder = EmbedderFactory._create_sparse_embedder(sparse_config)
            else:
                logger.warning("'hybrid' type specified but no sparse config provided. Using dense-only.")
        
        # Return appropriate embedder
        if embedder_type == "hybrid" and sparse_embedder is not None:
            return HybridEmbedder(
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
            )
        else:
            # Return dense-only (either explicitly or because sparse is None)
            return dense_embedder
    
    @staticmethod
    def _create_dense_embedder(config: Dict[str, Any]) -> BaseDenseEmbedder:
        """
        Create a dense embedder from configuration.
        
        Args:
            config: Dense embedder configuration
            
        Returns:
            BaseDenseEmbedder instance
        """
        provider = config.get("provider", "cohere")
        
        if provider == "cohere":
            # Get API key from env (not from config to avoid leakage)
            api_key_env = config.get("api_key_env", "COHERE_API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found. Set {api_key_env} environment variable.")
            max_batch_size = config.get("max_batch_size")
            max_concurrent = config.get("max_concurrent")
            if max_batch_size is None or max_concurrent is None:
                raise ValueError("Cohere embedder requires max_batch_size and max_concurrent in config.")

            return CohereDenseEmbedder(
                model=config.get("model", "embed-v4.0"),
                dimension=config.get("dimension", 1536),
                api_key=api_key,
                api_key_env=api_key_env,
                max_batch_size=max_batch_size,
                max_concurrent=max_concurrent,
            )
        elif provider == "qwen3":
            # Lazy import to avoid numpy compatibility issues
            from src.rag.embedding.dense.qwen3 import Qwen3DenseEmbedder
            max_batch_size = config.get("max_batch_size")
            if max_batch_size is None:
                raise ValueError("Qwen3 embedder requires max_batch_size in config.")
            return Qwen3DenseEmbedder(
                model=config.get("model", "Qwen/Qwen3-embedding-0.6b"),
                device=config.get("device"),  # None = auto
                max_batch_size=max_batch_size,
            )
        elif provider == "llamacpp":
            # LlamaCpp service-based embedder
            endpoint = config.get("endpoint")
            if not endpoint:
                raise ValueError(
                    "llamacpp provider requires 'endpoint' in config "
                    "(e.g., 'http://127.0.0.1:8005')"
                )

            return LlamaCppDenseEmbedder(
                endpoint=endpoint,
                model=config.get("model"),  # Optional, for tracking
                dimension=config.get("dimension", 1024),
                max_batch_size=config.get("max_batch_size", 32),
                split_on_failure=config.get("split_on_failure", True),
                timeout=config.get("timeout", 60.0),
            )
        elif provider == "deepinfra":
            # DeepInfra OpenAI-compatible API embedder
            api_key_env = config.get("api_key_env", "DEEPINFRA_API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"DeepInfra API key not found. Set {api_key_env} environment variable."
                )
            max_batch_size = config.get("max_batch_size", 32)
            max_concurrent = config.get("max_concurrent", 10)

            return DeepInfraOpenAIEmbedder(
                model=config.get("model", "Qwen/Qwen3-Embedding-0.6B"),
                dimension=config.get("dimension", 1024),
                api_key=api_key,
                api_key_env=api_key_env,
                max_batch_size=max_batch_size,
                max_concurrent=max_concurrent,
                encoding_format=config.get("encoding_format", "float"),
                timeout=config.get("timeout", 60.0),
                retry_max_attempts=config.get("retry_max_attempts", 3),
            )
        else:
            raise NotImplementedError(
                f"Dense embedder provider '{provider}' is not implemented yet. "
                f"Available: 'cohere', 'qwen3', 'llamacpp', 'deepinfra'"
            )
    
    @staticmethod
    def _create_sparse_embedder(config: Dict[str, Any]) -> Optional[BaseSparseEmbedder]:
        """
        Create a sparse embedder from configuration.
        
        Args:
            config: Sparse embedder configuration
            
        Returns:
            BaseSparseEmbedder instance or None
        """
        provider = config.get("provider")
        
        if provider is None:
            return None
        
        if provider == "bge":
            # Lazy import to avoid numpy compatibility issues
            from src.rag.embedding.sparse.bge import BGESparseEmbedder
            return BGESparseEmbedder(
                model=config.get("model", "BAAI/bge-m3"),
            )
        else:
            raise NotImplementedError(
                f"Sparse embedder provider '{provider}' is not implemented yet. "
                f"Available: 'bge' (future)"
            )
