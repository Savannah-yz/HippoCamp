"""
Gemini ReAct Generator Provider - Gemini-driven ReAct reasoning with search.

This provider implements the ReAct (Reason + Act) paradigm using Google Gemini API.
Gemini generates Thought/Action/Observation traces, deciding when to search
for information and when to produce a final answer.

ReAct format:
    Thought: <reasoning about what to do>
    Action: search[query] or finish[answer]
    Observation: <search results injected by system>

This is similar to QwenReActProvider but uses Gemini API instead of local Qwen model.

Usage:
    from src.providers import GeminiReActProvider, ProviderConfig

    config = ProviderConfig(
        name="gemini_react",
        params={
            "model": "gemini-2.5-flash",
            "search_url": "http://127.0.0.1:18000/retrieve",
            "max_turns": 5,
            "topk": 5,
        }
    )

    provider = GeminiReActProvider(config=config)
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
import os
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
- You can search as many times as needed (up to {max_turns} times).
- When you have enough information, use Action: finish[your answer].
- Each Action must be on its own line.
- Be concise in your search queries - use specific keywords."""

HOTPOTQA_REACT_PROMPT = """\
Question: {question}

Answer the question with a short, factoid answer. \
When you use finish[], provide only the answer phrase without explanation. \
For example: finish[Beijing]"""

CUSTOM_REACT_PROMPT = """\
Question: {question}

Provide a comprehensive and detailed answer based on the search results. \
When you use finish[], include a thorough explanation citing information from the search results."""

PROMPT_PRESETS = {
    "hotpotqa": HOTPOTQA_REACT_PROMPT,
    "custom": CUSTOM_REACT_PROMPT,
}


class GeminiReActProvider(GeneratorProvider):
    """
    Gemini ReAct generator provider using classic Thought/Action/Observation format.

    The model reasons step-by-step and decides when to search for information
    or produce a final answer using Action: search[query] / finish[answer].

    **Important**: This is an end-to-end provider that does NOT require
    a separate retrieval provider. Set `requires_retrieval = False`.

    Attributes:
        model: Gemini model name
        client: Google GenAI client
        search_url: URL for the retrieval server
        max_turns: Maximum search iterations
    """

    requires_retrieval = False

    def __init__(self, config: ProviderConfig):
        """
        Initialize the Gemini ReAct provider.

        Args:
            config: Provider configuration with params:
                - mode (str): Prompt preset — "hotpotqa" or "custom" (default: "custom")
                - model (str): Gemini model name (default: "gemini-2.5-flash")
                - search_url (str): URL for retrieval server
                - max_turns (int): Maximum search turns (default: 5)
                - topk (int): Number of search results per query (default: 5)
                - temperature (float): Sampling temperature (default: 0.7)
                - max_tokens (int): Max tokens per generation (default: 2048)
                - api_key_env (str): Environment variable name for API key (default: "GEMINI_API_KEY")
                - prompt_template (str): Custom user prompt template (overrides mode)
                - system_prompt (str): Custom system prompt (overrides default)
                - initial_search_original (bool): Use original query for first search (default: False)
        """
        super().__init__(config)

        params = config.params
        self.model_name = params.get("model", "gemini-2.5-flash")
        self.search_url = params.get("search_url", "http://127.0.0.1:18000/retrieve")
        self.max_turns = params.get("max_turns", 5)
        self.topk = params.get("topk", 20)
        self.temperature = params.get("temperature", 0.7)
        self.max_tokens = params.get("max_tokens", 2048)
        self.api_key_env = params.get("api_key_env", "GEMINI_API_KEY")
        self.initial_search_original = params.get("initial_search_original", False)

        # Prompt selection
        self.mode = params.get("mode", "custom")
        if "prompt_template" in params:
            self.user_prompt_template = params["prompt_template"]
        else:
            self.user_prompt_template = PROMPT_PRESETS.get(self.mode, CUSTOM_REACT_PROMPT)

        base_system_prompt = params.get("system_prompt", REACT_SYSTEM_PROMPT)
        self.system_prompt = base_system_prompt.format(max_turns=self.max_turns)

        self.client = None
        self.genai_version = None

        logger.info(f"GeminiReAct mode: {self.mode}, model: {self.model_name}, initial_search_original: {self.initial_search_original}")

    async def setup(self):
        """Initialize the Gemini client."""
        if self._initialized:
            return

        api_key = os.environ.get(self.api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise GenerationError(
                f"API key not found. Set {self.api_key_env} or GOOGLE_API_KEY environment variable.",
                provider=self.get_name(),
            )

        try:
            # Try new google-genai SDK first
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                self.genai_version = "new"
                logger.info(f"GeminiReAct using new google-genai SDK")
            except ImportError:
                # Fall back to old SDK
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.genai_version = "old"
                logger.info(f"GeminiReAct using old google-generativeai SDK")

            logger.info(f"GeminiReAct provider initialized: {self.model_name}")
            self._initialized = True

        except ImportError as e:
            raise GenerationError(
                f"Required packages not installed: {e}. "
                "Run: pip install google-genai or pip install google-generativeai",
                provider=self.get_name(),
            )
        except Exception as e:
            raise GenerationError(
                f"Failed to initialize Gemini client: {e}",
                provider=self.get_name(),
            )

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the last Action line from generated text.

        Returns:
            Tuple of (action_type, argument) where action_type is "search" or "finish",
            or (None, None) if no valid action found.
        """
        pattern = re.compile(r"Action:\s*(search|finish)\[(.+?)\]", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            action_type, argument = matches[-1]
            return action_type.strip().lower(), argument.strip()
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
                content = doc_item.get("text", doc_item.get("chunk", ""))
                metadata = doc_item.get("metadata", {})

                # Extract file info for display
                file_info = metadata.get("file_info", {})
                file_name = file_info.get("file_name", "")

                if content:
                    # Truncate long content for observation
                    display_content = content[:500] + "..." if len(content) > 500 else content
                    if file_name:
                        formatted_parts.append(f"[{idx + 1}] ({file_name}) {display_content}")
                    else:
                        formatted_parts.append(f"[{idx + 1}] {display_content}")

                    chunk_data = {
                        "id": doc_item.get("id", f"chunk_{idx}"),
                        "chunk": content,
                        "content": content,
                        "score": doc_item.get("score", 0.0),
                        "metadata": metadata,
                    }
                    retrieved_chunks.append(chunk_data)

            if not formatted_parts:
                return "No relevant documents found.", []

            return "\n\n".join(formatted_parts), retrieved_chunks

        except requests.exceptions.RequestException as e:
            logger.warning(f"Search request failed: {e}")
            return f"[Search failed: {e}]", []
        except Exception as e:
            logger.warning(f"Search error: {e}")
            return f"[Search error: {e}]", []

    async def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Call Gemini API with messages."""
        import asyncio

        try:
            if self.genai_version == "new":
                # New SDK format
                contents = []
                for msg in messages:
                    role = msg["role"]
                    # Map roles: system -> user (prepended), user -> user, assistant -> model
                    if role == "system":
                        contents.append({
                            "role": "user",
                            "parts": [{"text": f"[System Instructions]\n{msg['content']}"}]
                        })
                        contents.append({
                            "role": "model",
                            "parts": [{"text": "I understand. I'll follow the ReAct format with Thought/Action/Observation."}]
                        })
                    elif role == "user":
                        contents.append({
                            "role": "user",
                            "parts": [{"text": msg["content"]}]
                        })
                    elif role == "assistant":
                        contents.append({
                            "role": "model",
                            "parts": [{"text": msg["content"]}]
                        })

                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
                return response.text

            else:
                # Old SDK format
                chat = self.client.start_chat(history=[])

                # Build conversation history
                for msg in messages[:-1]:
                    if msg["role"] == "system":
                        chat.send_message(f"[System Instructions]\n{msg['content']}")
                    elif msg["role"] == "user":
                        chat.send_message(msg["content"])
                    # Note: assistant messages are automatically in history

                # Send last message
                last_msg = messages[-1]
                response = await asyncio.to_thread(
                    chat.send_message,
                    last_msg["content"],
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
                return response.text

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise GenerationError(
                f"Gemini API call failed: {e}",
                provider=self.get_name(),
            )

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
        Generate an answer using the ReAct loop with Gemini.

        The model alternates between Thought and Action steps. When it outputs
        Action: search[query], we call the retriever and inject the results as
        Observation. When it outputs Action: finish[answer], we return.

        Args:
            query: User question
            context: Ignored (handles retrieval internally)
            max_tokens: Max tokens per generation (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            GenerationResult with answer, reasoning trace, and search queries
        """
        if not self._initialized:
            await self.setup()

        if max_tokens:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        question = query.strip()

        # Build initial messages
        user_content = self.user_prompt_template.format(question=question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        search_queries = []
        all_retrieved_chunks = []
        full_trace = ""
        turn = 0

        logger.info(f"GeminiReAct: Starting generation for query: {query[:50]}...")

        try:
            # Optional: pre-search with original query before entering ReAct loop
            if self.initial_search_original:
                logger.info(f"GeminiReAct: Initial search with original query")
                search_results, retrieved_chunks = self._search(question)
                search_queries.append(question)
                all_retrieved_chunks.extend(retrieved_chunks)

                observation = f"\nObservation: {search_results}\n"
                full_trace += f"[Initial search with original query]\n"
                full_trace += observation

                # Inject as pre-existing context so Gemini sees results from turn 1
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: Let me first search with the original question to find relevant information.\nAction: search[{question}]",
                })
                messages.append({"role": "user", "content": observation})

            while turn < self.max_turns:
                turn += 1
                logger.debug(f"GeminiReAct: Turn {turn}/{self.max_turns}")

                # Call Gemini
                output_text = await self._call_gemini(messages)
                if output_text is None:
                    output_text = ""
                    logger.warning(f"GeminiReAct: Got None response at turn {turn}")
                full_trace += output_text

                if output_text:
                    logger.debug(f"GeminiReAct response: {output_text[:200]}...")

                # Parse action
                action_type, argument = self._parse_action(output_text)

                if action_type == "finish":
                    logger.info(f"GeminiReAct: finish at turn {turn}")
                    break

                elif action_type == "search":
                    logger.info(f"GeminiReAct: search[{argument[:50]}...] at turn {turn}")
                    search_queries.append(argument)

                    search_results, retrieved_chunks = self._search(argument)
                    all_retrieved_chunks.extend(retrieved_chunks)

                    observation = f"\nObservation: {search_results}\n"
                    full_trace += observation

                    # Append to conversation
                    messages.append({"role": "assistant", "content": output_text})
                    messages.append({"role": "user", "content": observation})

                else:
                    # No valid action parsed — model may have produced malformed output
                    logger.warning(f"GeminiReAct: No valid action parsed at turn {turn}, retrying...")

                    # Nudge the model to use correct format
                    nudge = "\n\nPlease continue with the correct format:\nThought: <your reasoning>\nAction: search[query] or finish[answer]"
                    messages.append({"role": "assistant", "content": output_text})
                    messages.append({"role": "user", "content": nudge})
                    full_trace += nudge

            # Extract final answer
            action_type, argument = self._parse_action(full_trace)
            if action_type == "finish" and argument:
                answer = argument
            else:
                # Fallback: use last generated text as answer
                # Try to extract any meaningful content
                answer = output_text.strip()
                if not answer or len(answer) < 10:
                    answer = f"[ReAct loop completed after {turn} turns without explicit finish. Last output: {output_text[:500]}]"

            logger.info(
                f"GeminiReAct: Generated answer in {turn} turn(s) "
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
                    "model": self.model_name,
                    "mode": self.mode,
                    "turns": turn,
                    "searches_performed": len(search_queries),
                    "retrieved_chunks": all_retrieved_chunks,
                },
            )

        except Exception as e:
            logger.error(f"GeminiReAct generation failed: {e}")
            raise GenerationError(
                f"Generation failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    async def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """GeminiReAct doesn't support separate query rewriting."""
        return query

    def get_name(self) -> str:
        return "gemini_react"

    async def aclose(self):
        """Cleanup resources."""
        self.client = None
        logger.info("GeminiReAct provider closed")
