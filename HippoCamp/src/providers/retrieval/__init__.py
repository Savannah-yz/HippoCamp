"""
Retrieval Providers for Research Core.

Available providers:
    - VectorSearchProvider: Simple vector similarity search
    - StandardRAGProvider: Retrieve -> Rerank -> Return
    - SelfRAGProvider: Retrieve -> Grade -> Filter -> (Iterate)
    - GradedRAGProvider: (Route) -> Retrieve -> (Grade) -> (Rewrite) -> Return
    - HybridRAGProvider: Dense + BM25 via SQLite FTS5 -> RRF Fusion -> Rerank

Post-processors:
    - ParentChildExpander: Expand chunks to page-level context
"""

from .vector_search import VectorSearchProvider
from .standard_rag import StandardRAGProvider
from .self_rag import SelfRAGProvider
from .graded_rag import GradedRAGProvider
from .hybrid_rag import HybridRAGProvider
from .parent_child import ParentChildExpander

__all__ = [
    "VectorSearchProvider",
    "StandardRAGProvider",
    "SelfRAGProvider",
    "GradedRAGProvider",
    "HybridRAGProvider",
    "ParentChildExpander",
]
