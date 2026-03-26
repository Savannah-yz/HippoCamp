"""
Gemini Generator Provider - Google Gemini API for answer generation.

This provider wraps the existing GeminiGenerator from research_core,
adapting it to the provider interface.

Usage:
    from src.providers import GeminiGeneratorProvider, ProviderConfig

    config = ProviderConfig(
        name="gemini",
        params={
            "model": "gemini-2.5-flash",
            "max_tokens": 512,
            "temperature": 0.7,
        }
    )

    provider = GeminiGeneratorProvider(config=config)
    await provider.setup()

    result = await provider.generate(
        query="What is machine learning?",
        context=[chunk1, chunk2, chunk3],
    )
    print(result.answer)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from ..base import (
    GeneratorProvider,
    ProviderConfig,
    GenerationResult,
    ChunkResult,
    GenerationError,
)

logger = logging.getLogger(__name__)


# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain relevant information, say so.
Be concise and accurate."""

DEFAULT_RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer based on the context above:"""

DEFAULT_REWRITE_PROMPT = """Rewrite the following query to be more specific and suitable for semantic search.
Keep it concise but add relevant keywords.

Original query: {query}

Rewritten query:"""


class GeminiGeneratorProvider(GeneratorProvider):
    """
    Generator provider using Google Gemini API.

    This provider uses the Google Generative AI SDK (google-genai)
    to generate answers from queries and context.

    Attributes:
        model_name: Gemini model to use
        client: Google GenAI client
        system_prompt: System prompt for generation
        rag_prompt_template: Template for RAG prompts
    """

    requires_retrieval = True  # Needs context from retrieval provider

    def __init__(
        self,
        config: ProviderConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Gemini generator provider.

        Args:
            config: Provider configuration with params:
                - model (str): Model name (default: "gemini-2.5-flash")
                - max_tokens (int): Default max tokens (default: 512)
                - temperature (float): Default temperature (default: 0.7)
                - system_prompt (str): Custom system prompt
                - rag_prompt_template (str): Custom RAG prompt template
                - api_key_env (str): Env var for API key (default: "GEMINI_API_KEY")
            api_key: Optional API key (overrides env var)
        """
        super().__init__(config)

        params = config.params
        self.model_name = params.get("model", "gemini-2.5-flash")
        self.default_max_tokens = params.get("max_tokens", 512)
        self.default_temperature = params.get("temperature", 0.7)
        self.system_prompt = params.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.rag_prompt_template = params.get("rag_prompt_template", DEFAULT_RAG_PROMPT_TEMPLATE)
        self.rewrite_prompt_template = params.get("rewrite_prompt_template", DEFAULT_REWRITE_PROMPT)

        # API key handling
        api_key_env = params.get("api_key_env", "GEMINI_API_KEY")
        self._api_key = api_key or os.getenv(api_key_env)

        self.client = None

    async def setup(self):
        """Initialize the Gemini client."""
        if self._initialized:
            return

        if not self._api_key:
            raise GenerationError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable.",
                provider=self.get_name(),
            )

        try:
            from google import genai

            self.client = genai.Client(api_key=self._api_key)
            logger.info(f"GeminiGeneratorProvider initialized with model: {self.model_name}")
            self._initialized = True

        except ImportError:
            raise GenerationError(
                "google-genai not installed. Run: pip install google-genai",
                provider=self.get_name(),
            )

    def _format_context(
        self,
        chunks: List[ChunkResult],
        max_chunks: int = 10,
    ) -> str:
        """
        Format chunks into a context string.

        Args:
            chunks: List of chunks to format
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.content
            if text:
                # Include source info if available
                source = chunk.metadata.get("file_path", "")
                if source:
                    context_parts.append(f"[{i}] (Source: {source})\n{text}")
                else:
                    context_parts.append(f"[{i}] {text}")

        return "\n\n".join(context_parts)

    async def generate(
        self,
        query: str,
        context: Optional[List[ChunkResult]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate an answer using Gemini.

        Args:
            query: User query
            context: List of context chunks
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            system_prompt: Override system prompt
            **kwargs: Additional parameters

        Returns:
            GenerationResult with the generated answer
        """
        if not self._initialized:
            await self.setup()

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        try:
            from google.genai import types

            # Format context if provided
            if context:
                context_str = self._format_context(context)
                prompt = self.rag_prompt_template.format(
                    context=context_str,
                    query=query,
                )
            else:
                # No context - just answer the query
                prompt = query

            # Build full prompt with system instruction
            system = system_prompt or self.system_prompt
            full_prompt = f"{system}\n\n{prompt}"

            # Generate
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            answer = response.text if response.text else ""
            logger.debug(f"Gemini generated {len(answer)} chars for query: {query[:50]}...")

            return GenerationResult(
                answer=answer,
                query=query,
                sources=context or [],
                metadata={
                    "provider": self.get_name(),
                    "model": self.model_name,
                    "context_chunks": len(context) if context else 0,
                },
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise GenerationError(
                f"Generation failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    async def rewrite_query(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Rewrite a query for better retrieval.

        Args:
            query: Original query
            context: Optional conversation context

        Returns:
            Rewritten query
        """
        if not self._initialized:
            await self.setup()

        prompt = self.rewrite_prompt_template.format(query=query)

        if context:
            prompt = f"Previous context:\n{context}\n\n{prompt}"

        try:
            from google.genai import types

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=100,
                    temperature=0.3,
                ),
            )

            rewritten = response.text.strip() if response.text else query

            # Clean up common prefixes
            for prefix in ["Rewritten query:", "Query:", "Rewritten:"]:
                if rewritten.lower().startswith(prefix.lower()):
                    rewritten = rewritten[len(prefix):].strip()

            logger.debug(f"Gemini rewrote: '{query}' → '{rewritten}'")
            return rewritten or query

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original")
            return query

    def get_name(self) -> str:
        """Get provider name."""
        return "gemini"

    async def aclose(self):
        """Cleanup resources."""
        # Gemini client doesn't need explicit cleanup
        pass
