"""
Search-R1 Generator Provider - End-to-end reasoning with search.

This provider implements the Search-R1 approach, which performs end-to-end
question answering with integrated search capabilities. Unlike traditional
RAG, Search-R1 doesn't require a separate retrieval provider - it handles
both search and generation internally.

Key Features:
- Multi-turn reasoning with <think> tags
- Dynamic search invocation with <search> tags
- Final answer extraction from <answer> tags
- Support for both local and remote search backends

Reference: https://github.com/PeterJinGo/Search-R1

Usage:
    from src.providers import SearchR1Provider, ProviderConfig

    config = ProviderConfig(
        name="search_r1",
        params={
            "model_path": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
            "search_url": "http://127.0.0.1:8000/retrieve",
            "max_turns": 5,
            "topk": 3,
        }
    )

    provider = SearchR1Provider(config=config)
    await provider.setup()

    # No retrieval needed - Search-R1 handles it internally
    result = await provider.generate(
        query="What is the capital of France?",
        context=None,  # Not used
    )
    print(result.answer)
    print(result.search_queries)  # Searches performed
    print(result.reasoning_trace)  # Full reasoning trace
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from ..base import (
    GeneratorProvider,
    ProviderConfig,
    GenerationResult,
    ChunkResult,
    GenerationError,
)

logger = logging.getLogger(__name__)


# ── Prompt presets ─────────────────────────────────────────────────────────
# "hotpotqa" — Official Search-R1 prompt (short, factoid answers)
HOTPOTQA_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {question}\n"
)

# "custom" — Detailed answer prompt (our own benchmark)
CUSTOM_PROMPT = (
    "Answer the given question with a comprehensive, detailed response. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "When you have gathered sufficient information, provide a complete and thorough answer "
    "inside <answer> and </answer>. The answer should be detailed and well-explained, "
    "citing specific information from the search results. Question: {question}\n"
)

PROMPT_PRESETS = {
    "hotpotqa": HOTPOTQA_PROMPT,
    "custom": CUSTOM_PROMPT,
}


class SearchR1Provider(GeneratorProvider):
    """
    End-to-end Search-R1 generator provider.

    This provider implements the Search-R1 approach where the model
    decides when to search and integrates search results into its
    reasoning process.

    **Important**: This is an end-to-end provider that does NOT require
    a separate retrieval provider. Set `requires_retrieval = False`.

    The model uses special tags:
    - <think>...</think>: Internal reasoning
    - <search>query</search>: Trigger a search
    - <information>...</information>: Search results (injected by system)
    - <answer>...</answer>: Final answer

    Attributes:
        model: Loaded transformers model
        tokenizer: Model tokenizer
        search_url: URL for the retrieval server
        max_turns: Maximum search iterations
    """

    requires_retrieval = False  # Search-R1 handles retrieval internally

    def __init__(
        self,
        config: ProviderConfig,
    ):
        """
        Initialize the Search-R1 provider.

        Args:
            config: Provider configuration with params:
                - mode (str): Prompt preset — "hotpotqa" or "custom" (default: "custom")
                - model_path (str): HuggingFace model path or local path
                - search_url (str): URL for retrieval server
                - max_turns (int): Maximum search turns (default: 5)
                - topk (int): Number of search results per query (default: 3)
                - temperature (float): Sampling temperature (default: 0.7)
                - max_new_tokens (int): Max tokens per generation (default: 1024)
                - device (str): Device to use (default: "cuda" if available)
                - prompt_template (str): Custom prompt template (overrides mode)
                - initial_search_original (bool): Use original query for first search (default: False)
        """
        super().__init__(config)

        params = config.params
        self.model_path = params.get(
            "model_path",
            "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
        )
        self.search_url = params.get("search_url", "http://127.0.0.1:18000/retrieve")
        self.max_turns = params.get("max_turns", 5)
        self.topk = params.get("topk", 3)
        self.temperature = params.get("temperature", 0.7)
        self.max_new_tokens = params.get("max_new_tokens", 1024)
        self.device_str = params.get("device", None)
        self.initial_search_original = params.get("initial_search_original", False)

        # Prompt selection: explicit prompt_template > mode preset > default
        self.mode = params.get("mode", "custom")
        if "prompt_template" in params:
            self.prompt_template = params["prompt_template"]
        else:
            self.prompt_template = PROMPT_PRESETS.get(self.mode, CUSTOM_PROMPT)

        self.model = None
        self.tokenizer = None
        self.device = None
        self.stopping_criteria = None

        # EOS token IDs for Qwen2.5 series
        self.eos_token_ids = params.get("eos_token_ids", [151645, 151643])

        logger.info(f"Search-R1 mode: {self.mode}, model: {self.model_path}, initial_search_original: {self.initial_search_original}")

    async def setup(self):
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            import transformers

            logger.info(f"Loading Search-R1 model: {self.model_path}")

            # Determine device
            if self.device_str:
                self.device = torch.device(self.device_str)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

            # Load model
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            # Setup stopping criteria for </search> tag
            self.stopping_criteria = self._create_stopping_criteria()

            logger.info("Search-R1 provider initialized successfully")
            self._initialized = True

        except ImportError as e:
            raise GenerationError(
                f"Required packages not installed: {e}. "
                "Run: pip install torch transformers",
                provider=self.get_name(),
            )
        except Exception as e:
            raise GenerationError(
                f"Failed to load Search-R1 model: {e}",
                provider=self.get_name(),
            )

    def _create_stopping_criteria(self):
        """Create stopping criteria for </search> tag."""
        import transformers

        class StopOnSequence(transformers.StoppingCriteria):
            def __init__(self, target_sequences, tokenizer):
                self.target_ids = [
                    tokenizer.encode(seq, add_special_tokens=False)
                    for seq in target_sequences
                ]
                self.target_lengths = [len(ids) for ids in self.target_ids]

            def __call__(self, input_ids, scores, **kwargs):
                import torch

                for i, target in enumerate(self.target_ids):
                    target_tensor = torch.as_tensor(target, device=input_ids.device)
                    if input_ids.shape[1] >= self.target_lengths[i]:
                        if torch.equal(
                            input_ids[0, -self.target_lengths[i]:],
                            target_tensor
                        ):
                            return True
                return False

        target_sequences = [
            "</search>", " </search>",
            "</search>\n", " </search>\n",
            "</search>\n\n", " </search>\n\n"
        ]

        return transformers.StoppingCriteriaList([
            StopOnSequence(target_sequences, self.tokenizer)
        ])

    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract the last search query from text."""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from text."""
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _search(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Perform a search using the retrieval server.

        Args:
            query: Search query

        Returns:
            Tuple of (formatted search results string, raw retrieved chunks)
        """
        try:
            payload = {
                "query": query,
                "top_k": self.topk,
            }

            response = requests.post(self.search_url, json=payload, timeout=30)
            response.raise_for_status()

            response_data = response.json()
            results = response_data.get("results", [])

            # Format results for model context
            formatted_parts = []
            retrieved_chunks = []

            for idx, doc_item in enumerate(results):
                content = doc_item.get("text", "")
                if content:
                    # Extract title (first line) and text (rest)
                    lines = content.split("\n")
                    title = lines[0] if lines else "Untitled"
                    text = "\n".join(lines[1:]) if len(lines) > 1 else content
                    formatted_parts.append(f"Doc {idx + 1}(Title: {title}) {text}")

                    # Store raw chunk data for result tracking
                    chunk_data = {
                        "id": doc_item.get("id", ""),
                        "chunk": content,
                        "score": doc_item.get("score", 0.0),
                        "metadata": doc_item.get("metadata", {}),
                    }
                    retrieved_chunks.append(chunk_data)

                    # Print first 3 chunks for debugging
                    if idx < 3:
                        logger.info(f"\n{'='*80}\nChunk {idx + 1} received by Search-R1:\n{'='*80}")
                        logger.info(f"Source: Retrieval server at {self.search_url}")
                        logger.info(f"Query: {query}")
                        logger.info(f"ID: {chunk_data['id']}")
                        logger.info(f"Score: {chunk_data['score']:.4f}")
                        logger.info(f"Content preview: {content[:200]}...")
                        logger.info(f"Metadata: {chunk_data['metadata']}")
                        logger.info(f"{'='*80}\n")

            return "\n".join(formatted_parts), retrieved_chunks

        except requests.exceptions.RequestException as e:
            logger.warning(f"Search request failed: {e}")
            return f"[Search failed: {e}]", []
        except Exception as e:
            logger.warning(f"Search error: {e}")
            return f"[Search error: {e}]", []

    async def generate(
        self,
        query: str,
        context: Optional[List[ChunkResult]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate an answer using Search-R1's multi-turn reasoning.

        Note: The `context` parameter is ignored as Search-R1 performs
        its own retrieval. This method runs an iterative process where
        the model can invoke searches as needed.

        Args:
            query: User question
            context: Ignored (Search-R1 handles retrieval internally)
            max_tokens: Max tokens per turn (overrides config)
            temperature: Sampling temperature (overrides config)
            **kwargs: Additional parameters

        Returns:
            GenerationResult with answer, reasoning trace, and search queries
        """
        if not self._initialized:
            await self.setup()

        import torch

        max_new_tokens = max_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        # Prepare query
        question = query.strip()
        if question and question[-1] not in ".?!":
            question += "?"

        # Build initial prompt
        prompt = self.prompt_template.format(question=question)

        # Apply chat template if available
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        search_queries = []
        all_retrieved_chunks = []  # Track all chunks from all searches
        full_trace = prompt
        turn = 0

        logger.debug(f"Search-R1: Starting generation for query: {query[:50]}...")

        try:
            # Optional: pre-search with original query before entering generation loop
            if self.initial_search_original:
                logger.info(f"Search-R1: Initial search with original query")
                search_results, retrieved_chunks = self._search(question)
                search_queries.append(question)
                all_retrieved_chunks.extend(retrieved_chunks)

                # Inject initial results into prompt as if model had searched
                initial_context = f"<search>{question}</search><information>{search_results}</information>\n\n"
                prompt += initial_context
                full_trace += initial_context

            while turn < self.max_turns:
                turn += 1

                # Tokenize
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # Generate
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temp,
                )

                # Check if reached EOS
                if outputs[0][-1].item() in self.eos_token_ids:
                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    full_trace += output_text
                    logger.debug(f"Search-R1: Completed at turn {turn} (EOS)")
                    break

                # Decode generated text
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Check for search query
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                search_query = self._extract_search_query(full_output)

                if search_query:
                    logger.debug(f"Search-R1: Searching for '{search_query}'")
                    search_queries.append(search_query)

                    # Perform search and capture retrieved chunks
                    search_results, retrieved_chunks = self._search(search_query)
                    all_retrieved_chunks.extend(retrieved_chunks)

                    # Update prompt with search results
                    search_text = f"\n\n{output_text}<information>{search_results}</information>\n\n"
                    prompt += search_text
                    full_trace += search_text
                else:
                    # No search, continue
                    full_trace += output_text
                    break

            # Extract final answer
            answer = self._extract_answer(full_trace) or ""

            logger.info(
                f"Search-R1: Generated answer in {turn} turn(s) "
                f"with {len(search_queries)} search(es), "
                f"retrieved {len(all_retrieved_chunks)} total chunks"
            )

            return GenerationResult(
                answer=answer,
                query=query,
                sources=[],  # Search-R1 doesn't return structured sources
                reasoning_trace=full_trace,
                search_queries=search_queries,
                metadata={
                    "provider": self.get_name(),
                    "model": self.model_path,
                    "turns": turn,
                    "searches_performed": len(search_queries),
                    "retrieved_chunks": all_retrieved_chunks,  # Include all retrieved chunks
                },
            )

        except Exception as e:
            logger.error(f"Search-R1 generation failed: {e}")
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
        Search-R1 doesn't support separate query rewriting.

        Returns the original query unchanged.
        """
        return query

    def get_name(self) -> str:
        """Get provider name."""
        return "search_r1"

    async def aclose(self):
        """Cleanup resources."""
        if self.model is not None:
            import torch

            # Move model to CPU and delete
            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Search-R1 model unloaded")
