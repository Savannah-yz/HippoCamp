"""
Retrieval Providers for Research Core.

Available providers:
    - VectorSearchProvider: Simple vector similarity search
    - StandardRAGProvider: Retrieve -> Rerank -> Return
    - SelfRAGProvider: Retrieve -> Grade -> Filter -> (Iterate)
    - GradedRAGProvider: (Route) -> Retrieve -> (Grade) -> (Rewrite) -> Return
    - HybridRAGProvider: Dense + BM25 via SQLite FTS5 -> RRF Fusion -> Rerank
    - CorrectiveRAGProvider: Retrieve -> Evaluate (3-class) -> Refine/Re-retrieve (CRAG)
    - AdaptiveRAGProvider: Classify complexity -> Route to Simple/Moderate/Complex
    - HyDERAGProvider: Generate hypothetical docs -> Average embeddings -> Retrieve (HyDE)
    - IRCoTRAGProvider: Interleave retrieval with chain-of-thought (IRCoT)
    - DecompositionRAGProvider: Decompose query -> Sequential solving (Least-to-Most)

Post-processors:
    - ParentChildExpander: Expand chunks to page-level context
"""

from .vector_search import VectorSearchProvider
from .standard_rag import StandardRAGProvider
from .self_rag import SelfRAGProvider
from .graded_rag import GradedRAGProvider
from .hybrid_rag import HybridRAGProvider
from .corrective_rag import CorrectiveRAGProvider
from .adaptive_rag import AdaptiveRAGProvider
from .hyde_rag import HyDERAGProvider
from .ircot_rag import IRCoTRAGProvider
from .decomposition_rag import DecompositionRAGProvider
from .parent_child import ParentChildExpander

__all__ = [
    "VectorSearchProvider",
    "StandardRAGProvider",
    "SelfRAGProvider",
    "GradedRAGProvider",
    "HybridRAGProvider",
    "CorrectiveRAGProvider",
    "AdaptiveRAGProvider",
    "HyDERAGProvider",
    "IRCoTRAGProvider",
    "DecompositionRAGProvider",
    "ParentChildExpander",
]
