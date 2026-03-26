"""
Generator Providers for Research Core.

Available providers:
    - GeminiGeneratorProvider: Google Gemini API for answer generation
    - GeminiReActProvider: Gemini-driven ReAct with search (end-to-end)
    - SearchR1Provider: End-to-end reasoning with search (Search-R1)
    - QwenReActProvider: Qwen-based ReAct with search
"""

from .gemini import GeminiGeneratorProvider
from .gemini_react import GeminiReActProvider
from .search_r1 import SearchR1Provider
from .qwen_react import QwenReActProvider

__all__ = [
    "GeminiGeneratorProvider",
    "GeminiReActProvider",
    "SearchR1Provider",
    "QwenReActProvider",
]
