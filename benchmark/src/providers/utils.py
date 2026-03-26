"""
Utility functions for providers.
"""

import logging

logger = logging.getLogger(__name__)

# Qwen3-Embedding official default instruction for retrieval.
# See: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
# Documents are embedded raw (no prefix). Only queries get this instruction.
_DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_query_for_embedding(
    query: str,
    instruction: str = _DEFAULT_QUERY_INSTRUCTION,
) -> str:
    """Format a query with instruction prefix for Qwen3-Embedding.

    This follows the official Qwen3-Embedding format:
        Instruct: {task_description}
        Query: {query}

    Only applied to queries at embedding time. Reranker and generator
    still receive the raw query.

    Args:
        query: Raw user query.
        instruction: Task instruction. Defaults to the Qwen3-Embedding
            official retrieval instruction.

    Returns:
        Formatted string for the embedding API.
    """
    formatted = f"Instruct: {instruction}\nQuery: {query}"
    logger.debug("Query formatted with instruction prefix for embedding")
    return formatted
