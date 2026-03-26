"""
Generator module for RAG answer generation.

Provides LLM-based generators for producing answers from retrieved context.
"""

from .base import BaseGenerator
from .factory import GeneratorFactory

__all__ = [
    "BaseGenerator",
    "GeneratorFactory",
]
