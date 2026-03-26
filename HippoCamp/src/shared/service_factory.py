"""
Shared Service Factory for Research Core and User Memory.

This factory creates services from a unified configuration with profile support,
allowing easy switching between local (llama.cpp) and cloud (Cohere/Azure) modes.

Usage:
    factory = SharedServiceFactory.from_yaml("configs/services.yaml")

    # Services use the active profile (local or cloud)
    embedder = factory.create_embedder()
    vectordb = factory.create_vectordb()
    llm = factory.create_llm()
    reranker = factory.create_reranker()

    # For RAG with experiment isolation
    vectordb = factory.create_vectordb(collection_suffix="exp_001")

    # For User Memory
    vectordb = factory.create_vectordb(collection_type="user_profile")
"""

import os
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SharedServiceFactory:
    """
    Unified service factory with profile support.

    Reads the active profile (local/cloud) from config and creates
    appropriate service instances for all pipelines.
    """

    config: Dict[str, Any]
    _active_profile: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Load the active profile after initialization."""
        profile_name = self.config.get("profile", "cloud")
        profiles = self.config.get("profiles", {})
        self._active_profile = profiles.get(profile_name, {})
        logger.info(f"ServiceFactory initialized with profile: {profile_name}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SharedServiceFactory":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SharedServiceFactory":
        """Create factory from configuration dictionary."""
        return cls(config=config)

    @property
    def profile_name(self) -> str:
        """Get the active profile name."""
        return self.config.get("profile", "cloud")

    @property
    def collections(self) -> Dict[str, str]:
        """Get collection names configuration."""
        return self.config.get("collections", {})

    # =========================================================================
    # Unified Service Creators
    # =========================================================================

    def create_embedder(self):
        """
        Create embedder based on active profile.

        Returns:
            Embedder instance (RAG-compatible)
        """
        from src.rag.embedding.factory import EmbedderFactory

        embed_config = self._active_profile.get("embedding", {})
        if not embed_config:
            raise ValueError("Embedding config missing in services.yaml")

        provider = embed_config.get("provider", "cohere")
        dimension = embed_config.get("dimension", 1536)
        max_batch_size = embed_config.get("max_batch_size")
        if max_batch_size is None:
            raise ValueError("Embedding max_batch_size must be set in services.yaml")

        rag_config = {
            "type": "dense_only",
            "dense": {
                "provider": provider,
                "model": embed_config.get("model", "embed-v4.0"),
                "dimension": dimension,
                "max_batch_size": max_batch_size,
            }
        }

        if provider == "cohere":
            rag_config["dense"]["max_concurrent"] = embed_config.get("max_concurrent")
        elif provider == "llamacpp":
            rag_config["dense"]["endpoint"] = embed_config.get("endpoint")
            rag_config["dense"]["timeout"] = embed_config.get("timeout", 60)
            if "split_on_failure" in embed_config:
                rag_config["dense"]["split_on_failure"] = embed_config.get("split_on_failure")
        elif provider == "deepinfra":
            rag_config["dense"]["api_key_env"] = embed_config.get("api_key_env", "DEEPINFRA_API_KEY")
            rag_config["dense"]["encoding_format"] = embed_config.get("encoding_format", "float")
            rag_config["dense"]["timeout"] = embed_config.get("timeout", 60)

        return EmbedderFactory.create(rag_config)

    def create_vectordb(
        self,
        collection_suffix: str = None,
        collection_type: str = "rag",
    ):
        """
        Create vector store based on active profile.

        Args:
            collection_suffix: Suffix to add to collection name (e.g., experiment_id)
            collection_type: Type of collection - "rag", "user_profile", "user_event", "user_atomic_facts"

        Returns:
            VectorStore instance (RAG-compatible)
        """
        from src.rag.vector_store.factory import VectorStoreFactory

        vdb_config = self._active_profile.get("vectordb", {})
        db_type = vdb_config.get("type", "qdrant")
        dimension = vdb_config.get("dimension", 1536)
        batch_size = vdb_config.get("batch_size")
        if batch_size is None:
            raise ValueError("VectorDB batch_size must be set in services.yaml")

        # Get base collection name
        collections = self.collections
        if collection_type == "rag":
            base_name = collections.get("rag", "research_rag")
        elif collection_type == "user_profile":
            base_name = collections.get("user_profile", "synvo_user_profile")
        elif collection_type == "user_event":
            base_name = collections.get("user_event", "synvo_user_event")
        elif collection_type == "user_atomic_facts":
            base_name = collections.get("user_atomic_facts", "synvo_user_atomic_facts")
        else:
            base_name = collection_type

        # Add suffix if provided
        collection_name = f"{base_name}_{collection_suffix}" if collection_suffix else base_name

        rag_config = {
            "type": db_type,
            "collection_name": collection_name,
            "dimension": dimension,
            "metric_type": vdb_config.get("metric_type", "COSINE"),
            "batch_size": batch_size,
        }

        # Add type-specific settings
        if db_type == "milvus":
            rag_config["endpoint_env"] = "VECTOR_DB_ENDPOINT"
            rag_config["api_key_env"] = "VECTOR_DB_API_KEY"
        elif db_type == "milvus_local":
            rag_config["uri"] = vdb_config.get("uri", "./data/vector_store/milvus.db")
        elif db_type == "qdrant":
            rag_config["path"] = vdb_config.get("path", "./data/vector_store/qdrant")
            rag_config["url"] = vdb_config.get("url")
            rag_config["api_key"] = vdb_config.get("api_key")
            rag_config["prefer_grpc"] = vdb_config.get("prefer_grpc", False)
            rag_config["enable_sparse"] = vdb_config.get("enable_sparse", True)
            rag_config["top_k"] = vdb_config.get("top_k", 10)
            rag_config["top_k_dense"] = vdb_config.get("top_k_dense")
            rag_config["top_k_sparse"] = vdb_config.get("top_k_sparse")
            rag_config["batch_size"] = vdb_config.get("batch_size", 128)
            rag_config["similarity_threshold"] = vdb_config.get("similarity_threshold", 0.0)
            rag_config["dense_vector_name"] = vdb_config.get("dense_vector_name", "dense")
            rag_config["sparse_vector_name"] = vdb_config.get("sparse_vector_name", "sparse")
            rag_config["partition_name"] = vdb_config.get("partition_name")
            rag_config["index_params"] = vdb_config.get("index_params")
            rag_config["search_params"] = vdb_config.get("search_params")
            rag_config["sparse_vectors_config"] = vdb_config.get("sparse_vectors_config")

        return VectorStoreFactory.create(rag_config)

    def create_llm(self):
        """
        Create LLM/Generator based on active profile.

        Returns:
            Generator instance (RAG-compatible)
        """
        from src.rag.generator.factory import GeneratorFactory

        llm_config = self._active_profile.get("llm", {})
        provider = llm_config.get("provider", "azure")

        rag_config = {
            "type": provider,
            "model": llm_config.get("model", "gpt-4o"),
            "timeout": llm_config.get("timeout", 30),
            "max_tokens": llm_config.get("max_tokens", 512),
        }

        # Add provider-specific settings
        if provider == "llamacpp":
            rag_config["endpoint"] = llm_config.get("endpoint")

        return GeneratorFactory.create(rag_config)

    def create_reranker(self):
        """
        Create reranker based on active profile.

        Returns:
            Reranker instance or None if not configured
        """
        from src.rag.rerank.factory import RerankerFactory

        rerank_config = self._active_profile.get("rerank", {})
        if not rerank_config:
            return None

        provider = rerank_config.get("provider", "cohere")

        rag_config = {
            "type": provider,
            "model": rerank_config.get("model"),
            "top_k": rerank_config.get("top_k", 5),
            "timeout": rerank_config.get("timeout", 30),
        }

        if provider == "llamacpp":
            rag_config["endpoint"] = rerank_config.get("endpoint")
        elif provider == "bge":
            rag_config["device"] = rerank_config.get("device")
            rag_config["batch_size"] = rerank_config.get("batch_size", 32)
        elif provider == "deepinfra":
            rag_config["api_key_env"] = rerank_config.get("api_key_env", "DEEPINFRA_RERANK_API_KEY")
            rag_config["base_url"] = rerank_config.get("base_url", "https://api.deepinfra.com/v1/inference")
            rag_config["max_batch_size"] = rerank_config.get("max_batch_size", 32)
            rag_config["max_concurrent"] = rerank_config.get("max_concurrent", 8)
            rag_config["retry_max_attempts"] = rerank_config.get("retry_max_attempts", 3)

        return RerankerFactory.create(rag_config)

    # =========================================================================
    # User Memory / Profiling Service Creators
    # =========================================================================

    def create_profiling_embedding_service(self):
        """Create embedding service for User Memory pipeline."""
        from src.profiling.apis.embedding import EmbeddingService

        embed_config = self._active_profile.get("embedding", {})

        return EmbeddingService(
            type=embed_config.get("provider", "cohere"),
            dense_config={
                "model": embed_config.get("model", "embed-v4.0"),
                "embedding_dim": embed_config.get("dimension", 1536),
                "max_elements_per_request": embed_config.get("max_batch_size", 32),
                "endpoint": embed_config.get("endpoint"),  # For llamacpp
            }
        )

    def create_profiling_vectordb_service(self):
        """Create VectorDB service for User Memory pipeline."""
        from src.profiling.apis.vectordb import VectorDB

        vdb_config = self._active_profile.get("vectordb", {})
        collections = self.collections

        return VectorDB(
            type=vdb_config.get("type", "qdrant"),
            config={
                "embedding_dim": vdb_config.get("dimension", 1536),
                "user_profile_collection_name": collections.get("user_profile", "synvo_user_profile"),
                "user_event_collection_name": collections.get("user_event", "synvo_user_event"),
                "user_atomic_facts_collection_name": collections.get("user_atomic_facts", "synvo_user_atomic_facts"),
                "top_k": 100,
                "uri": vdb_config.get("uri"),  # For milvus_local
                "path": vdb_config.get("path", "./data/vector_store/qdrant"),
                "prefer_grpc": vdb_config.get("prefer_grpc", False),
            }
        )

    def create_profiling_llm_service(self):
        """Create LLM service for User Memory pipeline."""
        from src.profiling.apis.llm import LLM

        llm_config = self._active_profile.get("llm", {})

        return LLM(
            type=llm_config.get("provider", "azure"),
            config={
                "model": llm_config.get("model", "gpt-4o"),
                "system_prompt": "You are a helpful assistant.",
                "endpoint": llm_config.get("endpoint"),  # For llamacpp
            }
        )

    # =========================================================================
    # Legacy Aliases (for backward compatibility)
    # =========================================================================

    def create_rag_embedder(self):
        """Alias for create_embedder()."""
        return self.create_embedder()

    def create_rag_vectordb(self, experiment_id: str = "default"):
        """Alias for create_vectordb() with experiment suffix."""
        return self.create_vectordb(collection_suffix=experiment_id)

    def create_rag_generator(self):
        """Alias for create_llm()."""
        return self.create_llm()

    # =========================================================================
    # Configuration Getters
    # =========================================================================

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for current profile."""
        return self._active_profile.get("embedding", {}).get("dimension", 1536)

    def get_active_profile_config(self) -> Dict[str, Any]:
        """Get the full active profile configuration."""
        return self._active_profile.copy()


# =========================================================================
# Convenience Functions
# =========================================================================

def load_services(yaml_path: str = "configs/services.yaml") -> SharedServiceFactory:
    """
    Load services configuration from YAML file.

    Args:
        yaml_path: Path to services.yaml

    Returns:
        SharedServiceFactory instance
    """
    return SharedServiceFactory.from_yaml(yaml_path)
