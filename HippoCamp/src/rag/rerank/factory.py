"""
Factory for creating rerankers from configuration.
Supports config-driven creation for easy ablation studies.
"""

import os
import logging
from typing import Dict, Any, Optional

from src.rag.rerank.base import BaseReranker
from src.rag.rerank.cohere import CohereReranker
from src.rag.rerank.bge import BGEReranker
from src.rag.rerank.llamacpp import LlamaCppReranker
from src.rag.rerank.deepinfra import DeepInfraReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """
    Factory for creating rerankers from configuration.
    
    Configuration format:
    {
        "type": "cohere" | "bge" | "llamacpp" | "deepinfra",
        "model": "rerank-english-v3.0",  # Model name
        "api_key": "xxx",  # Optional, can use api_key_env instead
        "api_key_env": "COHERE_API_KEY",  # Optional, environment variable name
        "top_k": 20,  # Optional, default top_k
        "threshold": 0.4,  # Optional, default threshold
        "max_batch_size": 100,  # Optional, for Cohere
        "device": "cuda",  # Optional, for BGE (auto-detect if not specified)
        "batch_size": 32,  # Optional, for BGE
        "endpoint": "http://127.0.0.1:8006",  # Optional, for llamacpp
        "timeout": 30,  # Optional, for llamacpp
        # DeepInfra only:
        # "base_url": "https://api.deepinfra.com/v1/inference",
        # "max_batch_size": 32,
        # "max_concurrent": 8,
        # "retry_max_attempts": 3,
    }
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseReranker:
        """
        Create a reranker from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            BaseReranker instance
            
        Raises:
            ValueError: If configuration is invalid
            NotImplementedError: If reranker type is not supported
        """
        rerank_type = config.get("type")
        
        if rerank_type is None:
            raise ValueError("'type' is required in reranker configuration")
        
        if rerank_type == "cohere":
            return RerankerFactory._create_cohere_reranker(config)
        elif rerank_type == "bge":
            return RerankerFactory._create_bge_reranker(config)
        elif rerank_type == "llamacpp":
            return RerankerFactory._create_llamacpp_reranker(config)
        elif rerank_type == "deepinfra":
            return RerankerFactory._create_deepinfra_reranker(config)
        else:
            raise NotImplementedError(
                f"Reranker type '{rerank_type}' is not implemented yet. "
                f"Available: 'cohere', 'bge', 'llamacpp', 'deepinfra'"
            )
    
    @staticmethod
    def _create_cohere_reranker(config: Dict[str, Any]) -> CohereReranker:
        """
        Create a Cohere reranker from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            CohereReranker instance
        """
        # Get API key (prefer direct api_key, fallback to env var)
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "COHERE_API_KEY")
        
        # If api_key not provided, try to get from environment
        if api_key is None:
            api_key = os.getenv(api_key_env)
        
        return CohereReranker(
            model=config.get("model", "rerank-english-v3.0"),
            api_key=api_key,
            api_key_env=api_key_env,
            top_k=config.get("top_k", 20),
            threshold=config.get("threshold", 0.0),
            max_batch_size=config.get("max_batch_size", 100),
            timeout=config.get("timeout", 30),
        )
    
    @staticmethod
    def _create_bge_reranker(config: Dict[str, Any]) -> BGEReranker:
        """
        Create a BGE reranker from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            BGEReranker instance
        """
        return BGEReranker(
            model=config.get("model", "BAAI/bge-reranker-v2-m3"),
            device=config.get("device"),  # None = auto-detect
            batch_size=config.get("batch_size", 32),
        )

    @staticmethod
    def _create_llamacpp_reranker(config: Dict[str, Any]) -> LlamaCppReranker:
        """
        Create a LlamaCpp reranker from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            LlamaCppReranker instance
        """
        return LlamaCppReranker(
            endpoint=config.get("endpoint", "http://127.0.0.1:8006"),
            model=config.get("model", "default"),
            timeout=config.get("timeout", 30.0),
        )

    @staticmethod
    def _create_deepinfra_reranker(config: Dict[str, Any]) -> DeepInfraReranker:
        """
        Create a DeepInfra reranker from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            DeepInfraReranker instance
        """
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "DEEPINFRA_RERANK_API_KEY")
        if api_key is None:
            api_key = os.getenv(api_key_env)

        return DeepInfraReranker(
            model=config.get("model", "Qwen/Qwen3-Reranker-0.6B"),
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=config.get("base_url", "https://api.deepinfra.com/v1/inference"),
            timeout=config.get("timeout", 30.0),
            max_batch_size=config.get("max_batch_size", 32),
            max_concurrent=config.get("max_concurrent", 8),
            retry_max_attempts=config.get("retry_max_attempts", 3),
        )
