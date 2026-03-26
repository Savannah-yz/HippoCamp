"""
Gemini-based generator using Google AI API.

Uses Gemini models for RAG answer generation.
"""

import os
import logging
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from .base import BaseGenerator

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


class GeminiGenerator(BaseGenerator):
    """
    Generator using Google Gemini API.

    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None,
        rewrite_prompt_template: Optional[str] = None,
    ):
        """
        Initialize Gemini generator.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash", "gemini-2.0-flash")
            api_key: API key (if None, will read from environment)
            api_key_env: Environment variable name for API key
            max_tokens: Default max tokens for generation
            temperature: Default temperature for generation
            system_prompt: Custom system prompt
            rag_prompt_template: Custom RAG prompt template (use {context} and {query})
            rewrite_prompt_template: Custom rewrite prompt template (use {query})
        """
        if genai is None:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )

        self.model_name = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.rag_prompt_template = rag_prompt_template or DEFAULT_RAG_PROMPT_TEMPLATE
        self.rewrite_prompt_template = rewrite_prompt_template or DEFAULT_REWRITE_PROMPT

        # Get API key
        if api_key is None:
            api_key = os.getenv(api_key_env)
        if api_key is None:
            raise ValueError(
                f"Gemini API key not found. Set {api_key_env} environment variable or pass api_key."
            )

        # Initialize client
        self.client = genai.Client(api_key=api_key)

        logger.info(f"Initialized GeminiGenerator with model: {model}")

    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate an answer based on query and retrieved context.

        Args:
            query: User query
            context: List of retrieved chunks
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer text
        """
        # Format context from chunks
        context_str = self.format_context(context)

        # Build prompt
        prompt = self.rag_prompt_template.format(
            context=context_str,
            query=query,
        )

        # Build messages
        system = system_prompt or self.system_prompt
        full_prompt = f"{system}\n\n{prompt}"

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens or self.default_max_tokens,
                    temperature=temperature or self.default_temperature,
                ),
            )

            answer = response.text if response.text else ""
            logger.info(f"Generated {len(answer)} chars for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

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
            Rewritten query
        """
        prompt = self.rewrite_prompt_template.format(query=query)

        if context:
            prompt = f"Previous context:\n{context}\n\n{prompt}"

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=100,
                    temperature=0.3,
                ),
            )

            rewritten = response.text.strip() if response.text else query
            # Remove any leading "Rewritten query:" if model included it
            if rewritten.lower().startswith("rewritten query:"):
                rewritten = rewritten[16:].strip()
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten or query

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original")
            return query

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model_name

    async def aclose(self):
        """Close generator (no-op for Gemini client)."""
        pass
