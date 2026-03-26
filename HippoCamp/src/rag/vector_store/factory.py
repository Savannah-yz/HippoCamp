"""
Factory for creating vector stores from configuration.
Supports config-driven creation for easy swapping of vector stores.
"""

import os
import logging
from typing import Dict, Any, Optional

from src.rag.vector_store.base import BaseVectorStore
from src.rag.vector_store.milvus import MilvusVectorStore
from src.rag.vector_store.milvus_local import MilvusLocalVectorStore
from src.rag.vector_store.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory for creating vector stores from configuration.
    
    Configuration format:
    {
        "type": "qdrant",
        "path": "./data/vector_store/qdrant",  # Required (embedded/local mode)
        "collection_name": "research_rag_collection",
        "dimension": 1536,
        "top_k": 10,
        "batch_size": 128,
        "similarity_threshold": 0.0,
        "metric_type": "COSINE",  # "COSINE", "IP", "L2"
        "prefer_grpc": False,
        "enable_sparse": True
    }
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseVectorStore:
        """
        Create a vector store from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            VectorStore instance
        """
        store_type = config.get("type", "qdrant")
        
        if store_type == "milvus":
            return VectorStoreFactory._create_milvus_store(config)
        elif store_type == "milvus_local":
            return VectorStoreFactory._create_milvus_local_store(config)
        elif store_type == "qdrant":
            return VectorStoreFactory._create_qdrant_store(config)
        else:
            raise NotImplementedError(
                f"Vector store type '{store_type}' is not implemented yet. "
                f"Available: 'milvus', 'milvus_local', 'qdrant'"
            )

    @staticmethod
    def _create_milvus_local_store(config: Dict[str, Any]) -> MilvusLocalVectorStore:
        """Create local file-based Milvus store with HNSW index."""
        return MilvusLocalVectorStore(
            uri=config.get("uri", "./data/vector_store/rag.milvus.db"),
            collection_name=config.get("collection_name", "research_rag"),
            dimension=config.get("dimension", 1024),
            metric_type=config.get("metric_type", "COSINE"),
            index_type=config.get("index_type", "HNSW"),
            index_params=config.get("index_params", {"M": 16, "efConstruction": 200}),
        )
    
    @staticmethod
    def _create_milvus_store(config: Dict[str, Any]) -> MilvusVectorStore:
        """
        Create a Milvus vector store from configuration.
        
        Args:
            config: Milvus configuration dictionary
            
        Returns:
            MilvusVectorStore instance
        """
        # Extract configuration with defaults
        # Support both direct values and env var references
        endpoint = config.get("endpoint")
        if not endpoint:
            endpoint_env = config.get("endpoint_env", "VECTOR_DB_ENDPOINT")
            endpoint = os.getenv(endpoint_env)

        api_key = config.get("api_key")
        if not api_key:
            api_key_env = config.get("api_key_env", "VECTOR_DB_API_KEY")
            api_key = os.getenv(api_key_env)
        
        batch_size = config.get("batch_size")
        if batch_size is None:
            raise ValueError("Milvus vector store requires batch_size in config.")

        return MilvusVectorStore(
            endpoint=endpoint,
            api_key=api_key,
            collection_name=config.get("collection_name", "research_rag_collection"),
            dimension=config.get("dimension", 1536),
            top_k=config.get("top_k", 10),
            top_k_dense=config.get("top_k_dense"),
            top_k_sparse=config.get("top_k_sparse"),
            batch_size=batch_size,
            similarity_threshold=config.get("similarity_threshold", 0.0),
            metric_type=config.get("metric_type", "COSINE"),
            partition_name=config.get("partition_name"),
            index_params=config.get("index_params"),
            search_params=config.get("search_params"),
        )

    @staticmethod
    def _create_qdrant_store(config: Dict[str, Any]) -> QdrantVectorStore:
        """Create a Qdrant vector store."""
        path = config.get("path") or os.getenv(config.get("path_env", "QDRANT_PATH"))
        url = config.get("url") or os.getenv(config.get("url_env", "QDRANT_URL"))
        api_key = config.get("api_key") or os.getenv(config.get("api_key_env", "QDRANT_API_KEY"))

        return QdrantVectorStore(
            path=path,
            url=url,
            api_key=api_key,
            collection_name=config.get("collection_name", "research_rag"),
            dimension=config.get("dimension", 1536),
            top_k=config.get("top_k", 10),
            top_k_dense=config.get("top_k_dense"),
            top_k_sparse=config.get("top_k_sparse"),
            batch_size=config.get("batch_size", 128),
            similarity_threshold=config.get("similarity_threshold", 0.0),
            metric_type=config.get("metric_type", "COSINE"),
            prefer_grpc=config.get("prefer_grpc", False),
            dense_vector_name=config.get("dense_vector_name", "dense"),
            sparse_vector_name=config.get("sparse_vector_name", "sparse"),
            enable_sparse=config.get("enable_sparse", True),
            sparse_vectors_config=config.get("sparse_vectors_config"),
            partition_name=config.get("partition_name"),
            index_params=config.get("index_params"),
            search_params=config.get("search_params"),
        )
