"""
Factory for creating generators from configuration.
"""

import logging
from typing import Any, Dict

from .base import BaseGenerator
from .llamacpp import LlamaCppGenerator

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """
    Factory for creating generators from configuration.

    Configuration format (llamacpp):
    {
        "type": "llamacpp",
        "endpoint": "http://127.0.0.1:8007",
        "model": "Qwen3-4B",
        "timeout": 120,
        "max_prompt_tokens": 4096,
        "system_prompt": "...",  # Optional
        "rag_prompt_template": "...",  # Optional
    }

    Configuration format (gemini):
    {
        "type": "gemini",
        "model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "max_tokens": 512,
        "temperature": 0.7,
        "system_prompt": "...",  # Optional
    }

    Configuration format (azure):
    {
        "type": "azure",
        "model": "gpt-4o",
        "api_key_env": "TOOLS_LLM_API_KEY",
        "endpoint_env": "TOOLS_LLM_ENDPOINT",
        "max_tokens": 512,
        "temperature": 0.7,
        "timeout": 30,
        "system_prompt": "...",  # Optional
    }
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseGenerator:
        """
        Create a generator from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            BaseGenerator instance

        Raises:
            ValueError: If configuration is invalid
            NotImplementedError: If generator type is not supported
        """
        generator_type = config.get("type")

        if generator_type is None:
            raise ValueError("'type' is required in generator configuration")

        if generator_type == "llamacpp":
            return GeneratorFactory._create_llamacpp_generator(config)
        elif generator_type == "gemini":
            return GeneratorFactory._create_gemini_generator(config)
        elif generator_type == "azure":
            return GeneratorFactory._create_azure_generator(config)
        else:
            raise NotImplementedError(
                f"Generator type '{generator_type}' is not implemented yet. "
                f"Available: 'llamacpp', 'gemini', 'azure'"
            )

    @staticmethod
    def _create_llamacpp_generator(config: Dict[str, Any]) -> LlamaCppGenerator:
        """
        Create a LlamaCpp generator from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            LlamaCppGenerator instance
        """
        return LlamaCppGenerator(
            endpoint=config.get("endpoint", "http://127.0.0.1:8007"),
            model=config.get("model", "default"),
            timeout=config.get("timeout", 120.0),
            max_prompt_tokens=config.get("max_prompt_tokens", 4096),
            system_prompt=config.get("system_prompt"),
            rag_prompt_template=config.get("rag_prompt_template"),
            rewrite_prompt_template=config.get("rewrite_prompt_template"),
        )

    @staticmethod
    def _create_gemini_generator(config: Dict[str, Any]):
        """
        Create a Gemini generator from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            GeminiGenerator instance
        """
        from .gemini import GeminiGenerator

        return GeminiGenerator(
            model=config.get("model", "gemini-2.5-flash"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env", "GEMINI_API_KEY"),
            max_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.7),
            system_prompt=config.get("system_prompt"),
            rag_prompt_template=config.get("rag_prompt_template"),
            rewrite_prompt_template=config.get("rewrite_prompt_template"),
        )

    @staticmethod
    def _create_azure_generator(config: Dict[str, Any]):
        """
        Create an Azure OpenAI generator from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            AzureOpenAIGenerator instance
        """
        from .azure import AzureOpenAIGenerator

        return AzureOpenAIGenerator(
            model=config.get("model", "gpt-4o"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env", "TOOLS_LLM_API_KEY"),
            endpoint=config.get("endpoint"),
            endpoint_env=config.get("endpoint_env", "TOOLS_LLM_ENDPOINT"),
            api_version=config.get("api_version"),
            api_version_env=config.get("api_version_env", "TOOLS_LLM_API_VERSION"),
            max_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 30.0),
            system_prompt=config.get("system_prompt"),
            rag_prompt_template=config.get("rag_prompt_template"),
            rewrite_prompt_template=config.get("rewrite_prompt_template"),
        )
