"""
LlamaCpp-based generator using HTTP service.

Connects to a running llama.cpp server for text generation.
"""

import logging
from typing import Any, Dict, List, Optional

from src.clients.llm import LLMServiceClient
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


class LlamaCppGenerator(BaseGenerator):
    """
    Generator that uses a llama.cpp HTTP service.

    Requires a running llama.cpp server with an LLM model loaded.
    Start with:
        ./services/start_llm_service.sh /path/to/model.gguf
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8007",
        model: str = "default",
        timeout: float = 120.0,
        max_prompt_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None,
        rewrite_prompt_template: Optional[str] = None,
    ):
        """
        Initialize LlamaCpp generator.

        Args:
            endpoint: URL of the llama.cpp LLM service
            model: Model identifier (for logging)
            timeout: Request timeout in seconds
            max_prompt_tokens: Max tokens for prompt truncation
            system_prompt: Custom system prompt
            rag_prompt_template: Custom RAG prompt template (use {context} and {query})
            rewrite_prompt_template: Custom rewrite prompt template (use {query})
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.rag_prompt_template = rag_prompt_template or DEFAULT_RAG_PROMPT_TEMPLATE
        self.rewrite_prompt_template = rewrite_prompt_template or DEFAULT_REWRITE_PROMPT

        self.client = LLMServiceClient(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            max_prompt_tokens=max_prompt_tokens,
        )

        logger.info(f"Initialized LlamaCppGenerator with endpoint: {endpoint}")

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

        # Use chat completion for better results with instruction-tuned models
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info(f"Generated {len(response)} chars for query: {query[:50]}...")
            return response
        except Exception as e:
            logger.warning(f"Chat completion failed, falling back to completion: {e}")
            # Fallback to simple completion
            full_prompt = f"{system_prompt or self.system_prompt}\n\n{prompt}"
            return await self.client.complete(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

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
            response = await self.client.complete(
                prompt=prompt,
                max_tokens=100,  # Short response for query rewrite
                temperature=0.3,  # Lower temp for consistency
            )
            # Clean up response
            rewritten = response.strip()
            # Remove any leading "Rewritten query:" if model included it
            if rewritten.lower().startswith("rewritten query:"):
                rewritten = rewritten[16:].strip()
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten or query  # Fallback to original if empty
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original")
            return query

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

    async def aclose(self):
        """Close generator (no-op for HTTP client)."""
        pass
