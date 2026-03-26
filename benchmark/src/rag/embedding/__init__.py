"""
Embedding module for RAG pipeline.
Provides modular, extensible embedding capabilities.
"""

from src.rag.embedding.base import (
    BaseEmbedder,
    BaseDenseEmbedder,
    BaseSparseEmbedder,
    EmbeddingResult,
)
from src.rag.embedding.dense.cohere import CohereDenseEmbedder
from src.rag.embedding.sparse.bge import BGESparseEmbedder
from src.rag.embedding.hybrid import HybridEmbedder
from src.rag.embedding.factory import EmbedderFactory

__all__ = [
    "BaseEmbedder",
    "BaseDenseEmbedder",
    "BaseSparseEmbedder",
    "EmbeddingResult",
    "CohereDenseEmbedder",
    "BGESparseEmbedder",
    "HybridEmbedder",
    "EmbedderFactory",
]

