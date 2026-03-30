"""
Providers Module for Research Core.

This module implements a flexible provider system for RAG (Retrieval-Augmented Generation)
pipelines, supporting multiple retrieval and generation strategies.

Architecture:
    - RetrievalProvider: Base class for chunk retrieval (VectorSearch, StandardRAG, SelfRAG, GradedRAG)
    - GeneratorProvider: Base class for answer generation (Gemini, SearchR1)

The provider system allows easy switching between different retrieval and generation
strategies through configuration, enabling ablation studies and method comparisons.

Usage:
    from src.providers import ProviderFactory

    # Create providers from config
    factory = ProviderFactory.from_yaml("configs/providers.yaml")
    retrieval_provider = factory.create_retrieval_provider()
    generator_provider = factory.create_generator_provider()

    # Run query
    chunks = await retrieval_provider.retrieve(query, top_k=10)
    answer = await generator_provider.generate(query, chunks)

Available Retrieval Providers:
    - vector_search: Simple vector similarity search
    - standard_rag: Retrieve -> Rerank -> Return
    - self_rag: Retrieve -> Grade -> Filter -> (Iterate)
    - graded_rag: (Route) -> Retrieve -> (Grade) -> (Rewrite) -> Return
    - corrective_rag: Retrieve -> Evaluate (3-class) -> Refine/Re-retrieve (CRAG)
    - adaptive_rag: Classify complexity -> Route to Simple/Moderate/Complex
    - hyde_rag: Generate hypothetical docs -> Average embeddings -> Retrieve (HyDE)
    - ircot_rag: Interleave retrieval with chain-of-thought (IRCoT)
    - decomposition_rag: Decompose query -> Sequential solving (Least-to-Most)

Available Generator Providers:
    - gemini: Google Gemini API for generation
    - search_r1: End-to-end reasoning with search (no separate retrieval needed)

Note: When using search_r1 generator, the retrieval provider should be set to None
as Search-R1 performs its own retrieval internally.
"""

from .base import (
    RetrievalProvider,
    GeneratorProvider,
    ProviderConfig,
    RetrievalResult,
    GenerationResult,
    ProviderError,
)
from .factory import ProviderFactory

# Retrieval providers
from .retrieval.vector_search import VectorSearchProvider
from .retrieval.standard_rag import StandardRAGProvider
from .retrieval.self_rag import SelfRAGProvider
from .retrieval.graded_rag import GradedRAGProvider
from .retrieval.hybrid_rag import HybridRAGProvider
from .retrieval.corrective_rag import CorrectiveRAGProvider
from .retrieval.adaptive_rag import AdaptiveRAGProvider
from .retrieval.hyde_rag import HyDERAGProvider
from .retrieval.ircot_rag import IRCoTRAGProvider
from .retrieval.decomposition_rag import DecompositionRAGProvider

# Generator providers
from .generator.gemini import GeminiGeneratorProvider
from .generator.gemini_react import GeminiReActProvider
from .generator.search_r1 import SearchR1Provider
from .generator.qwen_react import QwenReActProvider

__all__ = [
    # Base classes
    "RetrievalProvider",
    "GeneratorProvider",
    "ProviderConfig",
    "RetrievalResult",
    "GenerationResult",
    "ProviderError",
    # Factory
    "ProviderFactory",
    # Retrieval providers
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
    # Generator providers
    "GeminiGeneratorProvider",
    "GeminiReActProvider",
    "SearchR1Provider",
    "QwenReActProvider",
]
