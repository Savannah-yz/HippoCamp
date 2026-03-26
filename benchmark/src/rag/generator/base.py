"""
Base class for RAG answer generators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseGenerator(ABC):
    """
    Abstract base class for answer generators.

    Generators take a query and retrieved context, then produce an answer.
    """

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate an answer based on query and retrieved context.

        Args:
            query: User query
            context: List of retrieved chunks (each with 'chunk' text and optional 'metadata')
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer text
        """
        pass

    @abstractmethod
    async def rewrite_query(
        self,
        query: str,
        *,
        context: Optional[str] = None,
    ) -> str:
        """
        Rewrite/expand a query for better retrieval.

        Args:
            query: Original user query
            context: Optional conversation context

        Returns:
            Rewritten query for retrieval
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        pass

    @abstractmethod
    async def aclose(self):
        """Close generator resources."""
        pass

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10,
    ) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of chunk dictionaries
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("chunk", "")
            if text:
                context_parts.append(f"[{i}] {text}")

        return "\n\n".join(context_parts)
