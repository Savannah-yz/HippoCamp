"""
Azure OpenAI Generator for RAG pipeline.

This generator uses Azure OpenAI API for text generation,
providing cloud-based LLM capabilities for the RAG system.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI

from src.rag.generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class AzureOpenAIGenerator(BaseGenerator):
    """
    Azure OpenAI-based text generator for RAG.

    Features:
    - Uses Azure OpenAI API for generation
    - Supports streaming (optional)
    - Compatible with shared service factory
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_key_env: str = "TOOLS_LLM_API_KEY",
        endpoint: Optional[str] = None,
        endpoint_env: str = "TOOLS_LLM_ENDPOINT",
        api_version: Optional[str] = None,
        api_version_env: str = "TOOLS_LLM_API_VERSION",
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 30.0,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None,
        rewrite_prompt_template: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI generator.

        Args:
            model: Model deployment name (e.g., "gpt-4o")
            api_key: Azure OpenAI API key (or use env var)
            api_key_env: Environment variable for API key
            endpoint: Azure OpenAI endpoint URL (or use env var)
            endpoint_env: Environment variable for endpoint
            api_version: Azure API version (or use env var)
            api_version_env: Environment variable for API version
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt
            rag_prompt_template: Optional RAG prompt template
            rewrite_prompt_template: Optional query rewrite template
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Get credentials from env if not provided
        api_key = api_key or os.getenv(api_key_env)
        endpoint = endpoint or os.getenv(endpoint_env)
        api_version = api_version or os.getenv(api_version_env, "2024-02-15-preview")

        if not api_key:
            raise ValueError(f"Azure API key not found. Set {api_key_env} environment variable.")
        if not endpoint:
            raise ValueError(f"Azure endpoint not found. Set {endpoint_env} environment variable.")

        # Initialize async client
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )

        # Prompt templates
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.rag_prompt_template = rag_prompt_template or self._default_rag_template()
        self.rewrite_prompt_template = rewrite_prompt_template or self._default_rewrite_template()

        logger.info(
            f"Initialized AzureOpenAIGenerator: model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    def _default_system_prompt(self) -> str:
        """Default system prompt for RAG."""
        return (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context doesn't contain enough information to answer the question, "
            "say so honestly. Be concise and accurate."
        )

    def _default_rag_template(self) -> str:
        """Default RAG prompt template."""
        return (
            "Context information:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Based on the context above, answer the following question:\n"
            "{query}\n\n"
            "Answer:"
        )

    def _default_rewrite_template(self) -> str:
        """Default query rewrite template."""
        return (
            "Rewrite the following query to improve retrieval results. "
            "Make it more specific and searchable while preserving the original intent.\n\n"
            "Original query: {query}\n\n"
            "Rewritten query:"
        )

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
        Generate answer using Azure OpenAI.

        Args:
            query: User query
            context: List of retrieved chunks (each with 'chunk' text and optional 'metadata')
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer string
        """
        # Format context from chunks
        context_str = self.format_context(context)

        # Format prompt
        prompt = self.rag_prompt_template.format(
            context=context_str,
            query=query,
        )

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer.strip()

        except Exception as e:
            logger.error(f"Azure OpenAI generation failed: {e}")
            raise

    async def rewrite_query(
        self,
        query: str,
        *,
        context: Optional[str] = None,
    ) -> str:
        """
        Rewrite query using Azure OpenAI.

        Args:
            query: Original query
            context: Optional conversation context

        Returns:
            Rewritten query string
        """
        prompt = self.rewrite_prompt_template.format(query=query)
        if context:
            prompt = f"Previous context:\n{context}\n\n{prompt}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites queries."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,
                temperature=0.3,
            )

            rewritten = response.choices[0].message.content
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten.strip()

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return query  # Return original on failure

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

    async def aclose(self):
        """Close the async client."""
        if hasattr(self.client, "close"):
            await self.client.close()
