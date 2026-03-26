"""
Base classes and models for the Provider system.

This module defines the abstract base classes and data models for all providers,
following a similar pattern to ContextEval's provider architecture.

Usage:
    # Extend RetrievalProvider for new retrieval methods
    class MyRetrievalProvider(RetrievalProvider):
        async def retrieve(self, query, top_k=10) -> RetrievalResult:
            ...

    # Extend GeneratorProvider for new generation methods
    class MyGeneratorProvider(GeneratorProvider):
        async def generate(self, query, context) -> GenerationResult:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RetrievalProviderType(Enum):
    """Available retrieval provider types."""
    VECTOR_SEARCH = "vector_search"
    STANDARD_RAG = "standard_rag"
    SELF_RAG = "self_rag"
    GRADED_RAG = "graded_rag"
    HYBRID_RAG = "hybrid_rag"
    NONE = "none"  # For end-to-end methods like Search-R1


class GeneratorProviderType(Enum):
    """Available generator provider types."""
    GEMINI = "gemini"
    GEMINI_REACT = "gemini_react"
    AZURE = "azure"
    OPENAI = "openai"
    LLAMACPP = "llamacpp"
    SEARCH_R1 = "search_r1"
    QWEN_REACT = "qwen_react"


# =============================================================================
# Exceptions
# =============================================================================

class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: str = None, details: Dict = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class RetrievalError(ProviderError):
    """Error during retrieval."""
    pass


class GenerationError(ProviderError):
    """Error during generation."""
    pass


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class ProviderConfig:
    """
    Configuration for a provider.

    Attributes:
        name: Provider name/type
        enabled: Whether the provider is enabled
        params: Provider-specific parameters
    """
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create config from dictionary."""
        return cls(
            name=data.get("name", data.get("type", "")),
            enabled=data.get("enabled", True),
            params=data.get("params", {k: v for k, v in data.items() if k not in ("name", "type", "enabled")}),
        )


# =============================================================================
# Result Models
# =============================================================================

@dataclass
class ChunkResult:
    """
    A single retrieved chunk.

    Attributes:
        id: Unique chunk identifier (hash-based)
        content: The chunk text content
        score: Retrieval/relevance score
        metadata: Chunk metadata in unified format (see below)
        is_relevant: Whether marked as relevant (for grading methods)
        relevance_reason: Explanation of relevance decision

    Unified Metadata Format:
        {
            "file_info": {
                "file_id": "doc_456",
                "file_path": "reports/doc.pdf",
                "file_type": "pdf",
                "file_name": "doc.pdf"
            },
            "segment_info": {
                "segment_indices": [0, 1],
                "page_numbers": [1, 2],       // PDF only
                "time_ranges": [{"start": 0.0, "end": 30.0}]  // MP3 only
            },
            "chunk_meta": {
                "type": "content",
                "chunk_index": 0,
                "char_count": 512,
                "token_count": 128
            }
        }
    """
    id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_relevant: Optional[bool] = None
    relevance_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "is_relevant": self.is_relevant,
            "relevance_reason": self.relevance_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkResult":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", data.get("chunk", data.get("text", ""))),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            is_relevant=data.get("is_relevant"),
            relevance_reason=data.get("relevance_reason"),
        )


@dataclass
class RetrievalResult:
    """
    Result from a retrieval operation.

    Attributes:
        chunks: List of retrieved chunks
        query: Original query
        rewritten_query: Rewritten query (if applicable)
        iterations: Number of retrieval iterations (for iterative methods)
        metadata: Additional provider-specific metadata
        latency_breakdown: Step-by-step latency in milliseconds
        thinking_steps: Detailed reasoning trace for each step
        step_counts: Counts of various operations (search_count, rewrite_count, etc.)
    """
    chunks: List[ChunkResult]
    query: str
    rewritten_query: Optional[str] = None
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    thinking_steps: List[Dict[str, Any]] = field(default_factory=list)
    step_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def relevant_chunks(self) -> List[ChunkResult]:
        """Get only chunks marked as relevant."""
        return [c for c in self.chunks if c.is_relevant is True or c.is_relevant is None]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "chunks": [c.to_dict() for c in self.chunks],
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }
        if self.latency_breakdown:
            result["latency_breakdown"] = self.latency_breakdown
        if self.thinking_steps:
            result["thinking_steps"] = self.thinking_steps
        if self.step_counts:
            result["step_counts"] = self.step_counts
        return result


@dataclass
class GenerationResult:
    """
    Result from a generation operation.

    Attributes:
        answer: Generated answer text
        query: Original query
        sources: List of source chunks used
        reasoning_trace: Reasoning trace (for Search-R1 style methods)
        search_queries: Search queries made (for Search-R1 style methods)
        metadata: Additional provider-specific metadata
    """
    answer: str
    query: str
    sources: List[ChunkResult] = field(default_factory=list)
    reasoning_trace: Optional[str] = None
    search_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "reasoning_trace": self.reasoning_trace,
            "search_queries": self.search_queries,
            "metadata": self.metadata,
        }


# =============================================================================
# Abstract Base Classes
# =============================================================================

class RetrievalProvider(ABC):
    """
    Abstract base class for retrieval providers.

    Retrieval providers are responsible for finding relevant chunks
    from the vector store given a query.

    Subclasses must implement:
        - retrieve(): Main retrieval method
        - get_name(): Provider name

    Optional overrides:
        - setup(): Async initialization
        - aclose(): Cleanup resources

    Usage:
        class MyProvider(RetrievalProvider):
            async def retrieve(self, query, top_k=10) -> RetrievalResult:
                # Implement retrieval logic
                chunks = await self._search(query, top_k)
                return RetrievalResult(chunks=chunks, query=query)

            def get_name(self) -> str:
                return "my_provider"
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize the retrieval provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._initialized = False

    async def setup(self):
        """
        Async initialization. Override for providers needing async setup.

        Called once before first use. Safe to call multiple times.
        """
        self._initialized = True

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: Optional[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            filter: Optional metadata filter expression
            **kwargs: Provider-specific parameters

        Returns:
            RetrievalResult with retrieved chunks

        Raises:
            RetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the provider name/type."""
        pass

    async def aclose(self):
        """Close and cleanup resources. Override if needed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.get_name()!r})"


class GeneratorProvider(ABC):
    """
    Abstract base class for generator providers.

    Generator providers are responsible for producing answers
    given a query and optional context chunks.

    Subclasses must implement:
        - generate(): Main generation method
        - get_name(): Provider name

    Optional overrides:
        - setup(): Async initialization
        - aclose(): Cleanup resources
        - rewrite_query(): Query rewriting for better retrieval

    Usage:
        class MyGenerator(GeneratorProvider):
            async def generate(self, query, context) -> GenerationResult:
                # Implement generation logic
                answer = await self._generate(query, context)
                return GenerationResult(answer=answer, query=query)

            def get_name(self) -> str:
                return "my_generator"

    Special Case - End-to-End Providers (e.g., Search-R1):
        For providers that do their own retrieval internally,
        set `requires_retrieval = False` and implement the full
        search+generate logic in generate().
    """

    # Set to False for end-to-end providers like Search-R1
    requires_retrieval: bool = True

    def __init__(self, config: ProviderConfig):
        """
        Initialize the generator provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._initialized = False

    async def setup(self):
        """
        Async initialization. Override for providers needing async setup.

        Called once before first use. Safe to call multiple times.
        """
        self._initialized = True

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: Optional[List[ChunkResult]] = None,
        *,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate an answer for the query.

        Args:
            query: The user query
            context: Optional list of context chunks (not used by end-to-end providers)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            GenerationResult with the generated answer

        Raises:
            GenerationError: If generation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the provider name/type."""
        pass

    async def rewrite_query(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Rewrite a query for better retrieval.

        Default implementation returns the original query.
        Override for providers with query rewriting capability.

        Args:
            query: Original query
            context: Optional conversation context

        Returns:
            Rewritten query (or original if not supported)
        """
        return query

    async def aclose(self):
        """Close and cleanup resources. Override if needed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.get_name()!r})"
