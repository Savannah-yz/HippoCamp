"""
Sparse embedders module.
"""

from src.rag.embedding.sparse.base import BaseSparseEmbedder
from src.rag.embedding.sparse.bge import BGESparseEmbedder

__all__ = ["BaseSparseEmbedder", "BGESparseEmbedder"]

