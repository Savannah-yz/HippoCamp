"""
Qwen ReAct Generator Provider - Classic ReAct reasoning with search.

This provider implements the ReAct (Reason + Act) paradigm using Qwen3 models.
The model generates Thought/Action/Observation traces, deciding when to search
for information and when to produce a final answer.

ReAct format:
    Thought: <reasoning about what to do>
    Action: search[query] or finish[answer]
    Observation: <search results injected by system>

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"

Usage:
    from src.providers import QwenReActProvider, ProviderConfig

    config = ProviderConfig(
        name="qwen_react",
        params={
            "model_path": "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "search_url": "http://127.0.0.1:8000/retrieve",
            "max_turns": 5,
            "topk": 3,
        }
    )

    provider = QwenReActProvider(config=config)
    await provider.setup()

    result = await provider.generate(
        query="What is the capital of France?",
        context=None,
    )
    print(result.answer)
    print(result.search_queries)
    print(result.reasoning_trace)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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

REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by searching for information.

You have access to the following tool:
- search[query]: Search for information about a topic. Returns relevant documents.

Use the following format strictly:

Thought: reason about what information you need
Action: search[your search query]

After receiving search results as Observation, continue with:

Thought: reason about the new information
Action: search[another query] OR finish[your final answer]

Rules:
- Always start with a Thought before taking an Action.
- You can search as many times as needed.
- When you have enough information, use Action: finish[your answer].
- Each Action must be on its own line."""

HOTPOTQA_REACT_PROMPT = """\
Question: {question}

Answer the question with a short, factoid answer. \
When you use finish[], provide only the answer phrase without explanation. \
For example: finish[Beijing]"""

CUSTOM_REACT_PROMPT = """\
Question: {question}

Provide a comprehensive and detailed answer. \
When you use finish[], include a thorough explanation citing information from the search results."""

PROMPT_PRESETS = {
    "hotpotqa": HOTPOTQA_REACT_PROMPT,
    "custom": CUSTOM_REACT_PROMPT,
}


class QwenReActProvider(GeneratorProvider):
    """
    Qwen ReAct generator provider using classic Thought/Action/Observation format.

    The model reasons step-by-step and decides when to search for information
    or produce a final answer using Action: search[query] / finish[answer].

    **Important**: This is an end-to-end provider that does NOT require
    a separate retrieval provider. Set `requires_retrieval = False`.

    Attributes:
        model: Loaded transformers model
        tokenizer: Model tokenizer
        search_url: URL for the retrieval server
        max_turns: Maximum search iterations
    """

    requires_retrieval = False

    def __init__(self, config: ProviderConfig):
        """
        Initialize the Qwen ReAct provider.

        Args:
            config: Provider configuration with params:
                - mode (str): Prompt preset — "hotpotqa" or "custom" (default: "custom")
                - model_path (str): HuggingFace model path or local path
                - search_url (str): URL for retrieval server
                - max_turns (int): Maximum search turns (default: 5)
                - topk (int): Number of search results per query (default: 3)
                - temperature (float): Sampling temperature (default: 0.7)
                - max_new_tokens (int): Max tokens per generation step (default: 2048)
                - device (str): Device to use (default: "cuda" if available)
                - prompt_template (str): Custom user prompt template (overrides mode)
                - system_prompt (str): Custom system prompt (overrides default)
                - initial_search_original (bool): Use original query for first search (default: False)
        """
        super().__init__(config)

        params = config.params
        self.model_path = params.get(
            "model_path",
            "Qwen/Qwen3-30B-A3B-Thinking-2507"
        )
        self.search_url = params.get("search_url", "http://127.0.0.1:18000/retrieve")
        self.max_turns = params.get("max_turns", 10)
        self.topk = params.get("topk", 20)
        self.temperature = params.get("temperature", 0.7)
        self.max_new_tokens = params.get("max_new_tokens", 16384)
        self.device_str = params.get("device", None)
        self.initial_search_original = params.get("initial_search_original", False)

        # Prompt selection
        self.mode = params.get("mode", "custom")
        if "prompt_template" in params:
            self.user_prompt_template = params["prompt_template"]
        else:
            self.user_prompt_template = PROMPT_PRESETS.get(self.mode, CUSTOM_REACT_PROMPT)

        self.system_prompt = params.get("system_prompt", REACT_SYSTEM_PROMPT)

        self.model = None
        self.tokenizer = None
        self.device = None
        self.stopping_criteria = None

        logger.info(f"QwenReAct mode: {self.mode}, model: {self.model_path}, initial_search_original: {self.initial_search_original}")

    async def setup(self):
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            import transformers

            logger.info(f"Loading QwenReAct model: {self.model_path}")

            if self.device_str:
                self.device = torch.device(self.device_str)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Using device: {self.device}")

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self.stopping_criteria = self._create_stopping_criteria()

            logger.info("QwenReAct provider initialized successfully")
            self._initialized = True

        except ImportError as e:
            raise GenerationError(
                f"Required packages not installed: {e}. "
                "Run: pip install torch transformers",
                provider=self.get_name(),
            )
        except Exception as e:
            raise GenerationError(
                f"Failed to load QwenReAct model: {e}",
                provider=self.get_name(),
            )

    def _create_stopping_criteria(self):
        """Create stopping criteria to stop generation at 'Action:' lines."""
        import transformers

        class StopOnAction(transformers.StoppingCriteria):
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

        # Stop when model produces an Action line ending with "]"
        # We stop on the closing bracket so we capture the full action
        target_sequences = ["]\n", "]\n\n"]

        return transformers.StoppingCriteriaList([
            StopOnAction(target_sequences, self.tokenizer)
        ])

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the last Action line from generated text.

        Returns:
            Tuple of (action_type, argument) where action_type is "search" or "finish",
            or (None, None) if no valid action found.
        """
        pattern = re.compile(r"Action:\s*(search|finish)\[(.+?)\]", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            action_type, argument = matches[-1]
            return action_type.strip(), argument.strip()
        return None, None

    def _search(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform a search using the retrieval server.

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

            formatted_parts = []
            retrieved_chunks = []

            for idx, doc_item in enumerate(results):
                content = doc_item.get("text", "")
                if content:
                    lines = content.split("\n")
                    title = lines[0] if lines else "Untitled"
                    text = "\n".join(lines[1:]) if len(lines) > 1 else content
                    formatted_parts.append(f"[{idx + 1}] {title}: {text}")

                    chunk_data = {
                        "id": doc_item.get("id", ""),
                        "chunk": content,
                        "score": doc_item.get("score", 0.0),
                        "metadata": doc_item.get("metadata", {}),
                    }
                    retrieved_chunks.append(chunk_data)

            return "\n".join(formatted_parts), retrieved_chunks

        except requests.exceptions.RequestException as e:
            logger.warning(f"Search request failed: {e}")
            return f"[Search failed: {e}]", []
        except Exception as e:
            logger.warning(f"Search error: {e}")
            return f"[Search error: {e}]", []

    def _build_messages(self, question: str) -> List[Dict[str, str]]:
        """Build initial chat messages for the ReAct loop."""
        user_content = self.user_prompt_template.format(question=question)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

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
        Generate an answer using the ReAct loop.

        The model alternates between Thought and Action steps. When it outputs
        Action: search[query], we call the retriever and inject the results as
        Observation. When it outputs Action: finish[answer], we return.

        Args:
            query: User question
            context: Ignored (handles retrieval internally)
            max_tokens: Max tokens per generation step (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            GenerationResult with answer, reasoning trace, and search queries
        """
        if not self._initialized:
            await self.setup()

        import torch

        max_new_tokens = max_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        question = query.strip()

        # Build chat messages
        messages = self._build_messages(question)

        search_queries = []
        all_retrieved_chunks = []
        # Accumulates the assistant's full reasoning trace (text only, no chat framing)
        full_trace = ""
        turn = 0

        logger.debug(f"QwenReAct: Starting generation for query: {query[:50]}...")

        try:
            # Optional: pre-search with original query before entering ReAct loop
            if self.initial_search_original:
                logger.info(f"QwenReAct: Initial search with original query")
                search_results, retrieved_chunks = self._search(question)
                search_queries.append(question)
                all_retrieved_chunks.extend(retrieved_chunks)

                observation = f"\nObservation: {search_results}\n"
                full_trace += f"[Initial search with original query]\n"
                full_trace += observation

                # Inject as pre-existing context so the model sees results from turn 1
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: Let me first search with the original question to find relevant information.\nAction: search[{question}]",
                })
                messages.append({"role": "user", "content": observation})

            while turn < self.max_turns:
                turn += 1

                # Encode with chat template
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
                attention_mask = torch.ones_like(input_ids)

                eos_token_id = self.tokenizer.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_ids = eos_token_id
                else:
                    eos_ids = [eos_token_id] if eos_token_id is not None else []

                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id or eos_token_id,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.8,
                    top_k=20,
                )

                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_trace += output_text

                # Check if EOS was reached (model finished without action)
                if eos_ids and outputs[0][-1].item() in eos_ids:
                    logger.debug(f"QwenReAct: EOS at turn {turn}")
                    break

                # Parse action
                action_type, argument = self._parse_action(output_text)

                if action_type == "finish":
                    logger.debug(f"QwenReAct: finish at turn {turn}")
                    break

                elif action_type == "search":
                    logger.debug(f"QwenReAct: search[{argument}] at turn {turn}")
                    search_queries.append(argument)

                    search_results, retrieved_chunks = self._search(argument)
                    all_retrieved_chunks.extend(retrieved_chunks)

                    observation = f"\nObservation: {search_results}\n"
                    full_trace += observation

                    # Append assistant output + observation to messages for next turn
                    messages.append({"role": "assistant", "content": output_text})
                    messages.append({"role": "user", "content": observation})

                else:
                    # No valid action parsed — model may have hit max tokens or produced malformed output
                    logger.warning(f"QwenReAct: No valid action parsed at turn {turn}")
                    break

            # Extract answer
            action_type, argument = self._parse_action(full_trace)
            if action_type == "finish" and argument:
                answer = argument
            else:
                # Fallback: use last generated text as answer
                answer = output_text.strip()

            logger.info(
                f"QwenReAct: Generated answer in {turn} turn(s) "
                f"with {len(search_queries)} search(es), "
                f"retrieved {len(all_retrieved_chunks)} total chunks"
            )

            return GenerationResult(
                answer=answer,
                query=query,
                sources=[],
                reasoning_trace=full_trace,
                search_queries=search_queries,
                metadata={
                    "provider": self.get_name(),
                    "model": self.model_path,
                    "mode": self.mode,
                    "turns": turn,
                    "searches_performed": len(search_queries),
                    "retrieved_chunks": all_retrieved_chunks,
                },
            )

        except Exception as e:
            logger.error(f"QwenReAct generation failed: {e}")
            raise GenerationError(
                f"Generation failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    async def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """QwenReAct doesn't support separate query rewriting."""
        return query

    def get_name(self) -> str:
        return "qwen_react"

    async def aclose(self):
        """Cleanup resources."""
        if self.model is not None:
            import torch

            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("QwenReAct model unloaded")
