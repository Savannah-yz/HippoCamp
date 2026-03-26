"""
Provider Factory for Research Core.

This factory creates retrieval and generator providers from configuration,
supporting easy switching between different methods for ablation studies.

The factory pattern allows configuration-driven provider creation without
modifying code, similar to ContextEval's approach.

Usage:
    from src.providers import ProviderFactory

    # Create from YAML config
    factory = ProviderFactory.from_yaml("configs/providers.yaml")

    # Or from service factory + provider config
    factory = ProviderFactory.from_service_factory(
        service_factory=SharedServiceFactory.from_yaml("configs/services.yaml"),
        provider_config={"retrieval": {"type": "self_rag"}, "generator": {"type": "gemini"}}
    )

    # Create providers
    retrieval_provider = await factory.create_retrieval_provider()
    generator_provider = await factory.create_generator_provider()

    # Run query
    chunks = await retrieval_provider.retrieve(query)
    result = await generator_provider.generate(query, chunks.chunks)

Configuration Example (providers.yaml):
    retrieval:
      type: "self_rag"  # vector_search, standard_rag, self_rag, graded_rag, none
      params:
        top_k: 20
        relevance_threshold: 0.5
        max_iterations: 2

    generator:
      type: "gemini"  # gemini, search_r1
      params:
        model: "gemini-2.5-flash"
        max_tokens: 512
        temperature: 0.7
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml

from .base import (
    RetrievalProvider,
    GeneratorProvider,
    ProviderConfig,
    ProviderError,
    RetrievalProviderType,
    GeneratorProviderType,
)

# Import all providers
from .retrieval import (
    VectorSearchProvider,
    StandardRAGProvider,
    SelfRAGProvider,
    GradedRAGProvider,
    HybridRAGProvider,
)
from .generator import (
    GeminiGeneratorProvider,
    GeminiReActProvider,
    SearchR1Provider,
    QwenReActProvider,
)

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating retrieval and generator providers.

    This factory supports:
    - Configuration-driven provider creation
    - Integration with SharedServiceFactory for service dependencies
    - Multiple retrieval methods (vector_search, standard_rag, self_rag, graded_rag)
    - Multiple generator methods (gemini, search_r1)
    - Easy switching between methods via config

    Attributes:
        config: Provider configuration dictionary
        service_factory: Optional SharedServiceFactory for service dependencies
        experiment_id: Experiment ID for vector store collection suffix
    """

    # Provider type to class mapping
    RETRIEVAL_PROVIDERS = {
        RetrievalProviderType.VECTOR_SEARCH.value: VectorSearchProvider,
        RetrievalProviderType.STANDARD_RAG.value: StandardRAGProvider,
        RetrievalProviderType.SELF_RAG.value: SelfRAGProvider,
        RetrievalProviderType.GRADED_RAG.value: GradedRAGProvider,
        RetrievalProviderType.HYBRID_RAG.value: HybridRAGProvider,
        RetrievalProviderType.NONE.value: None,
    }

    GENERATOR_PROVIDERS = {
        GeneratorProviderType.GEMINI.value: GeminiGeneratorProvider,
        GeneratorProviderType.GEMINI_REACT.value: GeminiReActProvider,
        GeneratorProviderType.SEARCH_R1.value: SearchR1Provider,
        GeneratorProviderType.QWEN_REACT.value: QwenReActProvider,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        service_factory=None,
        experiment_id: str = None,
    ):
        """
        Initialize the provider factory.

        Args:
            config: Provider configuration dictionary with 'retrieval' and 'generator' sections
            service_factory: Optional SharedServiceFactory for creating service dependencies
            experiment_id: Experiment ID for vector store collection suffix
        """
        self.config = config
        self.service_factory = service_factory
        self.experiment_id = experiment_id or config.get("experiment_id", "default")

        # Cache created services
        self._embedder = None
        self._vector_store = None
        self._reranker = None
        self._fts_store = None
        self._base_generator = None

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        service_factory=None,
        experiment_id: str = None,
    ) -> "ProviderFactory":
        """
        Create factory from YAML configuration file.

        Args:
            yaml_path: Path to providers.yaml
            service_factory: Optional SharedServiceFactory
            experiment_id: Experiment ID

        Returns:
            ProviderFactory instance
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(config=config, service_factory=service_factory, experiment_id=experiment_id)

    @classmethod
    def from_service_factory(
        cls,
        service_factory,
        provider_config: Dict[str, Any],
        experiment_id: str = None,
    ) -> "ProviderFactory":
        """
        Create factory from a SharedServiceFactory and provider config.

        Args:
            service_factory: SharedServiceFactory instance
            provider_config: Provider configuration dictionary
            experiment_id: Experiment ID

        Returns:
            ProviderFactory instance
        """
        return cls(
            config=provider_config,
            service_factory=service_factory,
            experiment_id=experiment_id,
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        service_factory=None,
        experiment_id: str = None,
    ) -> "ProviderFactory":
        """
        Create factory from configuration dictionary.

        Args:
            config: Configuration dictionary
            service_factory: Optional SharedServiceFactory
            experiment_id: Experiment ID

        Returns:
            ProviderFactory instance
        """
        return cls(config=config, service_factory=service_factory, experiment_id=experiment_id)

    # =========================================================================
    # Service Access (lazy initialization)
    # =========================================================================

    async def _get_embedder(self):
        """Get or create embedder service."""
        if self._embedder is None and self.service_factory:
            self._embedder = self.service_factory.create_embedder()
        return self._embedder

    async def _get_vector_store(self):
        """Get or create vector store service."""
        if self._vector_store is None and self.service_factory:
            self._vector_store = self.service_factory.create_vectordb(
                collection_suffix=self.experiment_id
            )
        return self._vector_store

    async def _get_reranker(self):
        """Get or create reranker service."""
        if self._reranker is None and self.service_factory:
            self._reranker = self.service_factory.create_reranker()
        return self._reranker

    async def _get_fts_store(self, params: dict):
        """Get or create SQLite FTS store from provider params."""
        if self._fts_store is None:
            from src.rag.fts.sqlite_fts import SqliteFtsStore

            db_path = params.get("sqlite_db_path")
            if not db_path:
                raise ProviderError(
                    "hybrid_rag requires 'sqlite_db_path' in retrieval params"
                )
            self._fts_store = SqliteFtsStore(db_path=db_path)
            await self._fts_store.setup()
        return self._fts_store

    def _get_keyword_llm(self):
        """Create a Gemini client for BM25 keyword extraction (lightweight)."""
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning(
                    "GEMINI_API_KEY not set; BM25 will use raw query "
                    "(no keyword extraction)"
                )
                return None
            client = genai.Client(api_key=api_key)
            logger.info(
                "Keyword LLM (Gemini) initialized for BM25 keyword extraction "
                "(api_key=...%s, client=%s)",
                api_key[-4:] if len(api_key) >= 4 else "****",
                type(client).__name__,
            )
            return client
        except Exception as e:
            logger.warning(
                "Failed to create keyword LLM (%s): %s; BM25 will use raw query",
                type(e).__name__,
                e,
            )
            return None

    async def _get_base_generator(self):
        """Get or create base LLM generator (for grading, rewriting)."""
        if self._base_generator is None and self.service_factory:
            self._base_generator = self.service_factory.create_llm()
        return self._base_generator

    # =========================================================================
    # Provider Creation
    # =========================================================================

    async def create_retrieval_provider(self) -> Optional[RetrievalProvider]:
        """
        Create a retrieval provider based on configuration.

        The retrieval provider is responsible for finding relevant chunks.

        Config structure:
            retrieval:
              type: "self_rag"
              params:
                top_k: 20
                ...

        Returns:
            RetrievalProvider instance, or None if type is "none"

        Raises:
            ProviderError: If provider type is unknown or creation fails
        """
        retrieval_config = self.config.get("retrieval", {})
        provider_type = retrieval_config.get("type", "vector_search")
        params = retrieval_config.get("params", {})

        logger.info(f"Creating retrieval provider: {provider_type}")

        # Handle "none" type (for end-to-end methods like Search-R1)
        if provider_type == "none" or provider_type is None:
            logger.info("No retrieval provider (using end-to-end generator)")
            return None

        # Get provider class
        provider_class = self.RETRIEVAL_PROVIDERS.get(provider_type)
        if provider_class is None:
            raise ProviderError(
                f"Unknown retrieval provider type: {provider_type}. "
                f"Available: {list(self.RETRIEVAL_PROVIDERS.keys())}"
            )

        # Create provider config
        config = ProviderConfig(name=provider_type, params=params)

        # Get required services
        embedder = await self._get_embedder()
        vector_store = await self._get_vector_store()

        # Create provider based on type
        if provider_type == "vector_search":
            reranker = await self._get_reranker() if params.get("use_reranker", True) else None
            provider = VectorSearchProvider(
                config=config,
                embedder=embedder,
                vector_store=vector_store,
                reranker=reranker,
            )

        elif provider_type == "standard_rag":
            reranker = await self._get_reranker() if params.get("use_reranker", True) else None
            provider = StandardRAGProvider(
                config=config,
                embedder=embedder,
                vector_store=vector_store,
                reranker=reranker,
            )

        elif provider_type == "self_rag":
            # Self-RAG needs a generator for grading and rewriting
            base_gen = await self._get_base_generator()
            # Wrap base generator to match expected interface
            gen_wrapper = await self._create_generator_wrapper(base_gen)
            provider = SelfRAGProvider(
                config=config,
                embedder=embedder,
                vector_store=vector_store,
                generator=gen_wrapper,
            )

        elif provider_type == "graded_rag":
            # Graded-RAG needs a generator for routing, grading, rewriting
            base_gen = await self._get_base_generator()
            gen_wrapper = await self._create_generator_wrapper(base_gen)
            provider = GradedRAGProvider(
                config=config,
                embedder=embedder,
                vector_store=vector_store,
                generator=gen_wrapper,
            )

        elif provider_type == "hybrid_rag":
            reranker = await self._get_reranker() if params.get("use_reranker", True) else None
            fts_store = await self._get_fts_store(params)
            keyword_llm = self._get_keyword_llm()
            # Default keyword_model from gemini generator config if not explicitly set
            if "keyword_model" not in params:
                gemini_cfg = self.config.get("generators", {}).get("gemini", {})
                gemini_model = gemini_cfg.get("params", {}).get("model")
                if gemini_model:
                    params["keyword_model"] = gemini_model
            provider = HybridRAGProvider(
                config=config,
                embedder=embedder,
                vector_store=vector_store,
                reranker=reranker,
                fts_store=fts_store,
                keyword_llm=keyword_llm,
            )

        else:
            raise ProviderError(f"Unknown retrieval provider type: {provider_type}")

        # Initialize provider
        await provider.setup()

        logger.info(f"Retrieval provider '{provider_type}' created successfully")
        return provider

    async def _create_generator_wrapper(self, base_generator):
        """
        Create a simple wrapper to adapt the base generator for use in retrieval providers.

        This wrapper provides a simple interface for grading and rewriting
        that retrieval providers can use.
        """

        class GeneratorWrapper:
            def __init__(self, generator):
                self.generator = generator

            async def generate(
                self,
                query: str,
                context=None,
                max_tokens: int = 512,
                temperature: float = 0.7,
            ):
                """Generate text using the base generator."""
                result = await self.generator.generate(
                    query=query,
                    context=[],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Return a simple object with answer attribute
                return type("Response", (), {"answer": result})()

            async def aclose(self):
                await self.generator.aclose()

        return GeneratorWrapper(base_generator)

    def _resolve_generator_config(self) -> Dict[str, Any]:
        """Resolve which generator config to use.

        Supports two config formats:

        **New format** (preferred)::

            active_generator: "gemini_react"
            generators:
              gemini: {type: "gemini", params: {...}}
              gemini_react: {type: "gemini_react", params: {...}}

        **Legacy format** (still supported)::

            generator:
              type: "gemini"
              params: {...}

        Returns:
            Dict with ``type`` and ``params`` keys for the selected generator.
        """
        # New format: generators map + active_generator selector
        generators_map = self.config.get("generators")
        if generators_map:
            active = self.config.get("active_generator")
            if not active:
                raise ProviderError(
                    "Config has 'generators' map but no 'active_generator' selector. "
                    "Set active_generator to one of: " + ", ".join(generators_map.keys())
                )
            gen_config = generators_map.get(active)
            if gen_config is None:
                raise ProviderError(
                    f"active_generator '{active}' not found in generators map. "
                    f"Available: {list(generators_map.keys())}"
                )
            return gen_config

        # Legacy format: single generator block
        return self.config.get("generator", {})

    async def create_generator_provider(self) -> GeneratorProvider:
        """
        Create a generator provider based on configuration.

        The generator provider is responsible for producing answers.

        Supports both the new multi-generator config format
        (``generators`` map + ``active_generator`` selector) and the
        legacy single ``generator`` block.

        Returns:
            GeneratorProvider instance

        Raises:
            ProviderError: If provider type is unknown or creation fails
        """
        generator_config = self._resolve_generator_config()
        provider_type = generator_config.get("type", "gemini")
        params = generator_config.get("params", {})

        logger.info(f"Creating generator provider: {provider_type}")

        # Get provider class
        provider_class = self.GENERATOR_PROVIDERS.get(provider_type)
        if provider_class is None:
            raise ProviderError(
                f"Unknown generator provider type: {provider_type}. "
                f"Available: {list(self.GENERATOR_PROVIDERS.keys())}"
            )

        # Create provider config
        config = ProviderConfig(name=provider_type, params=params)

        # Create provider from registry
        provider = provider_class(config=config)

        # Initialize provider
        await provider.setup()

        logger.info(f"Generator provider '{provider_type}' created successfully")
        return provider

    async def create_providers(self) -> tuple:
        """
        Create both retrieval and generator providers.

        Returns:
            Tuple of (retrieval_provider, generator_provider)
            Note: retrieval_provider may be None for end-to-end generators
        """
        retrieval_provider = await self.create_retrieval_provider()
        generator_provider = await self.create_generator_provider()

        # Validate compatibility
        if not generator_provider.requires_retrieval and retrieval_provider is not None:
            logger.warning(
                f"Generator '{generator_provider.get_name()}' doesn't require retrieval, "
                f"but retrieval provider '{retrieval_provider.get_name()}' was created. "
                "The retrieval provider will be ignored."
            )

        return retrieval_provider, generator_provider

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def get_retrieval_type(self) -> str:
        """Get configured retrieval provider type."""
        return self.config.get("retrieval", {}).get("type", "vector_search")

    def get_generator_type(self) -> str:
        """Get configured generator provider type."""
        return self._resolve_generator_config().get("type", "gemini")

    def is_end_to_end(self) -> bool:
        """Check if using an end-to-end generator (no separate retrieval)."""
        gen_type = self.get_generator_type()
        gen_class = self.GENERATOR_PROVIDERS.get(gen_type)
        if gen_class:
            return not gen_class.requires_retrieval
        return False

    @staticmethod
    def get_available_retrieval_types() -> list:
        """Get list of available retrieval provider types."""
        return list(ProviderFactory.RETRIEVAL_PROVIDERS.keys())

    @staticmethod
    def get_available_generator_types() -> list:
        """Get list of available generator provider types."""
        return list(ProviderFactory.GENERATOR_PROVIDERS.keys())

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self):
        """Cleanup all cached services."""
        if self._embedder:
            await self._embedder.aclose()
            self._embedder = None

        if self._reranker:
            await self._reranker.aclose()
            self._reranker = None

        if self._fts_store:
            await self._fts_store.aclose()
            self._fts_store = None

        if self._base_generator:
            await self._base_generator.aclose()
            self._base_generator = None

        logger.info("ProviderFactory cleanup complete")
