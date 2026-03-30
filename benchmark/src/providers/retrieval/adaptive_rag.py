"""
Adaptive RAG Provider.

Faithful implementation of Adaptive-RAG from:
    Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large
    Language Models through Question Complexity" (NAACL 2024)
    Paper: https://arxiv.org/abs/2403.14403
    Official repo: https://github.com/starsuzi/Adaptive-RAG

Core algorithm from the paper:
    1. Classify query complexity -> {A, B, C}
       A = simple (no retrieval needed)
       B = moderate (single-step retrieval, like standard RAG)
       C = complex (multi-step iterative retrieval, like IRCoT)
    2. Route to the simplest sufficient strategy

Adaptations from the original paper:
    - Classifier: LLM-as-judge instead of fine-tuned T5-large
      (paper trains T5-large on silver labels from strategy outcomes)
    - Strategy A: skipped (we always retrieve since this is a retrieval provider)
    - Strategy B: single-step retrieve + rerank (equivalent to paper's oner_qa)
    - Strategy C: multi-step iterative retrieval with CoT (equivalent to paper's ircot_qa)

Usage:
    provider = AdaptiveRAGProvider(
        config=ProviderConfig(name="adaptive_rag", params={
            "top_k": 20,
            "rerank_top_k": 5,
            "max_iterations_complex": 3,
        }),
        embedder=embedder, vector_store=vector_store,
        reranker=reranker, generator=generator,
    )
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..utils import format_query_for_embedding
from ..base import (
    RetrievalProvider,
    ProviderConfig,
    RetrievalResult,
    ChunkResult,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class AdaptiveRAGProvider(RetrievalProvider):
    """
    Adaptive RAG provider that routes queries to different retrieval
    strategies based on estimated complexity.

    Routes:
    - A (simple): single-step retrieval (minimal LLM overhead)
    - B (moderate): single-step retrieval + rerank
    - C (complex): multi-step iterative retrieval with CoT reasoning
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        reranker=None,
        generator=None,
    ):
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.generator = generator

        params = config.params
        self.default_top_k = params.get("top_k", 20)
        self.rerank_top_k = params.get("rerank_top_k", 5)
        self.use_reranker = params.get("use_reranker", True)
        # Paper's ircot_qa uses max ~6 docs per step, ~15 total
        self.max_iterations_complex = params.get("max_iterations_complex", 3)
        self.max_para_complex = params.get("max_paragraphs_complex", 15)

    async def setup(self):
        if self._initialized:
            return
        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for complexity classification",
                provider=self.get_name(),
            )
        logger.info(
            f"AdaptiveRAGProvider initialized "
            f"(max_iter_complex={self.max_iterations_complex})"
        )
        self._initialized = True

    # =========================================================================
    # Helper: embed + search
    # =========================================================================

    async def _embed_and_search(
        self, query: str, top_k: int, filter=None
    ) -> List[ChunkResult]:
        query_for_embed = format_query_for_embedding(query)
        query_embedding = await self.embedder.embed([query_for_embed])
        dense = (
            query_embedding.dense_embeddings[0]
            if query_embedding.dense_embeddings is not None
            and len(query_embedding.dense_embeddings) > 0
            else None
        )
        sparse = (
            query_embedding.sparse_embeddings[0]
            if query_embedding.sparse_embeddings is not None
            and len(query_embedding.sparse_embeddings) > 0
            else None
        )
        results = await self.vector_store.query(
            dense_embedding=dense, sparse_embedding=sparse,
            top_k=top_k, filter=filter,
        )
        return [
            ChunkResult(
                id=r.get("id", ""),
                content=r.get("chunk", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    async def _rerank_chunks(
        self, query: str, chunks: List[ChunkResult], top_k: int
    ) -> List[ChunkResult]:
        if not self.reranker or not chunks:
            return chunks[:top_k]
        candidates = [
            {"id": c.id, "chunk": c.content, "score": c.score, "metadata": c.metadata}
            for c in chunks
        ]
        reranked = await self.reranker.rerank(
            query=query, candidates=candidates, top_k=top_k
        )
        return [
            ChunkResult(
                id=r.get("id", ""),
                content=r.get("chunk", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in reranked
        ]

    # =========================================================================
    # Step 1: Complexity Classifier
    # Paper: T5-large outputs A/B/C token, softmax over those 3.
    # Here: LLM classifies into A/B/C with definitions.
    # =========================================================================

    async def _classify_complexity(self, query: str) -> str:
        """
        Classify query complexity into A (simple), B (moderate), C (complex).

        Paper definitions:
        - A: Can be answered from general knowledge without retrieval
        - B: Requires single-step retrieval from one source
        - C: Requires multi-step reasoning across multiple sources
        """
        classify_prompt = f"""Classify this question's complexity for a personal file retrieval system.

Question: {query}

A = Simple: single fact, likely in one document, no reasoning needed
    (e.g., "What is my phone number?" "When was this file created?")
B = Moderate: needs context from one document or straightforward lookup
    (e.g., "What were the key findings in my research paper?")
C = Complex: needs information from multiple documents or multi-step reasoning
    (e.g., "Compare the budgets from Q1 and Q3 reports" "How has my project evolved?")

Output ONLY the letter (A, B, or C):"""

        try:
            response = await self.generator.generate(
                query=classify_prompt, context=None, max_tokens=10, temperature=0.1,
            )
            text = response.answer.strip().upper()
            # Extract first A, B, or C
            match = re.search(r"[ABC]", text)
            if match:
                return match.group(0)
            return "B"  # Default to moderate
        except Exception as e:
            logger.warning(f"Complexity classification failed: {e}, defaulting to B")
            return "B"

    # =========================================================================
    # Strategy A: Simple retrieval (paper: nor_qa, no retrieval)
    # Since we're a retrieval provider, we do minimal single-step retrieval.
    # =========================================================================

    async def _strategy_simple(
        self, query: str, top_k: int, filter, thinking_steps, step_counts, total_start
    ) -> List[ChunkResult]:
        step_id = len(thinking_steps) + 1
        thinking_steps.append({
            "step_id": f"simple_retrieve_{step_id}",
            "step_type": "retrieve",
            "title": "Simple Retrieval (Strategy A)",
            "status": "running",
            "summary": f"Single-step retrieval: {query[:50]}...",
            "timestamp_ms": (time.time() - total_start) * 1000,
        })

        chunks = await self._embed_and_search(query, top_k, filter)
        step_counts["search_count"] += 1

        # Just take top results, no reranking needed for simple queries
        chunks = chunks[:self.rerank_top_k]

        thinking_steps[-1]["status"] = "complete"
        thinking_steps[-1]["summary"] = f"Found {len(chunks)} documents"
        return chunks

    # =========================================================================
    # Strategy B: Single-step retrieval + rerank (paper: oner_qa)
    # Paper: BM25 retrieval with 15 docs, then generate.
    # =========================================================================

    async def _strategy_moderate(
        self, query: str, top_k: int, filter, thinking_steps, step_counts, total_start
    ) -> List[ChunkResult]:
        step_id = len(thinking_steps) + 1
        thinking_steps.append({
            "step_id": f"moderate_retrieve_{step_id}",
            "step_type": "retrieve",
            "title": "Single-step Retrieval (Strategy B)",
            "status": "running",
            "summary": f"Retrieving top-{top_k}: {query[:50]}...",
            "timestamp_ms": (time.time() - total_start) * 1000,
        })

        chunks = await self._embed_and_search(query, top_k, filter)
        step_counts["search_count"] += 1

        thinking_steps[-1]["status"] = "complete"
        thinking_steps[-1]["summary"] = f"Found {len(chunks)} documents"

        if self.use_reranker and self.reranker and chunks:
            step_id = len(thinking_steps) + 1
            thinking_steps.append({
                "step_id": f"moderate_rerank_{step_id}",
                "step_type": "rerank",
                "title": "Reranking (Strategy B)",
                "status": "running",
                "summary": f"Reranking {len(chunks)} to top-{self.rerank_top_k}...",
                "timestamp_ms": (time.time() - total_start) * 1000,
            })
            chunks = await self._rerank_chunks(query, chunks, self.rerank_top_k)
            step_counts["rerank_count"] = step_counts.get("rerank_count", 0) + 1
            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Reranked to {len(chunks)} documents"

        return chunks

    # =========================================================================
    # Strategy C: Multi-step iterative retrieval (paper: ircot_qa)
    # Paper: IRCoT-style interleaving. We implement a simplified version:
    # retrieve -> generate CoT sentence -> use it as next query -> repeat.
    # =========================================================================

    def _is_reasoning_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence is a reasoning sentence (should not be used as query).
        From IRCoT paper: sentences starting with conjunctions or containing arithmetic.
        """
        reasoning_starters = ["thus", "so", "therefore", "hence", "consequently",
                              "as a result", "in conclusion", "this means"]
        lower = sentence.lower().strip()
        for starter in reasoning_starters:
            if lower.startswith(starter):
                return True
        # Contains arithmetic
        if re.search(r"\d+\s*[+\-*/]\s*\d+", sentence):
            return True
        return False

    async def _strategy_complex(
        self, query: str, top_k: int, filter, thinking_steps, step_counts, total_start
    ) -> List[ChunkResult]:
        """
        Multi-step IRCoT-style retrieval (paper's ircot_qa strategy).

        Loop: retrieve -> generate one CoT sentence -> use non-reasoning
              sentence as next query -> accumulate paragraphs.
        """
        all_chunks = {}  # id -> ChunkResult
        generated_sentences = []
        current_query = query
        per_step_top_k = min(6, top_k)  # Paper uses 6 docs per step

        for step in range(self.max_iterations_complex):
            # Retrieve
            step_id = len(thinking_steps) + 1
            thinking_steps.append({
                "step_id": f"complex_retrieve_{step_id}",
                "step_type": "retrieve",
                "title": f"Iterative Retrieval Step {step + 1} (Strategy C)",
                "status": "running",
                "summary": f"Searching: {current_query[:50]}...",
                "timestamp_ms": (time.time() - total_start) * 1000,
                "metadata": {"query": current_query, "step": step + 1},
            })

            new_chunks = await self._embed_and_search(current_query, per_step_top_k, filter)
            step_counts["search_count"] += 1

            for c in new_chunks:
                if c.id not in all_chunks:
                    all_chunks[c.id] = c

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Found {len(new_chunks)} docs (total unique: {len(all_chunks)})"
            )

            # Check paragraph limit (paper: max 15)
            if len(all_chunks) >= self.max_para_complex:
                break

            # Generate one CoT reasoning sentence
            context_text = ""
            for i, c in enumerate(list(all_chunks.values())[-6:], 1):
                truncated = c.content[:300] + "..." if len(c.content) > 300 else c.content
                context_text += f"\n[Paragraph {i}] {truncated}\n"

            cot_prompt = f"""Based on these paragraphs, generate the next step of reasoning to answer the question.
Output exactly ONE sentence of reasoning.

Question: {query}

Paragraphs:{context_text}

Previous reasoning: {' '.join(generated_sentences) if generated_sentences else '(none)'}

Next reasoning sentence:"""

            step_id = len(thinking_steps) + 1
            thinking_steps.append({
                "step_id": f"cot_{step_id}",
                "step_type": "reason",
                "title": f"CoT Step {step + 1}",
                "status": "running",
                "summary": "Generating reasoning sentence...",
                "timestamp_ms": (time.time() - total_start) * 1000,
            })

            try:
                response = await self.generator.generate(
                    query=cot_prompt, context=None, max_tokens=100, temperature=0.3,
                )
                sentence = response.answer.strip()
                # Extract first sentence only
                first_sent_match = re.match(r"([^.!?]+[.!?])", sentence)
                if first_sent_match:
                    sentence = first_sent_match.group(1)

                generated_sentences.append(sentence)
                step_counts["reasoning_count"] = step_counts.get("reasoning_count", 0) + 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"CoT: {sentence[:80]}..."

                # Check if answer found (paper: regex "the answer is:")
                if re.search(r"the answer is:?", sentence, re.IGNORECASE):
                    break

                # Use non-reasoning sentence as next query
                if not self._is_reasoning_sentence(sentence):
                    current_query = sentence
                # else keep current_query unchanged

            except Exception as e:
                logger.warning(f"CoT generation failed at step {step}: {e}")
                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"CoT failed: {e}"
                break

        # Final: rerank all accumulated chunks against original query
        final = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
        if self.use_reranker and self.reranker and final:
            final = await self._rerank_chunks(query, final, self.rerank_top_k)
            step_counts["rerank_count"] = step_counts.get("rerank_count", 0) + 1
        else:
            final = final[:self.rerank_top_k]

        return final

    # =========================================================================
    # Main retrieve
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve using Adaptive RAG pipeline.

        1. Classify complexity -> A/B/C
        2. Route to simplest sufficient strategy
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        logger.debug(f"AdaptiveRAG: Processing query: {query[:50]}...")

        latency_breakdown = {}
        thinking_steps = []
        step_counts = {"search_count": 0}
        total_start = time.time()

        try:
            # Step 1: Classify
            t0 = time.time()
            thinking_steps.append({
                "step_id": "classify_1",
                "step_type": "classify",
                "title": "Classifying Query Complexity",
                "status": "running",
                "summary": "Determining A (simple) / B (moderate) / C (complex)...",
                "timestamp_ms": 0,
            })

            complexity = await self._classify_complexity(query)
            latency_breakdown["classify_ms"] = (time.time() - t0) * 1000

            strategy_names = {"A": "simple", "B": "moderate (single-step)", "C": "complex (multi-step)"}
            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Complexity: {complexity} - {strategy_names.get(complexity, 'unknown')}"
            thinking_steps[-1]["metadata"] = {"complexity": complexity}

            logger.debug(f"AdaptiveRAG: Complexity={complexity}")

            # Step 2: Route
            if complexity == "A":
                chunks = await self._strategy_simple(
                    query, top_k, filter, thinking_steps, step_counts, total_start
                )
            elif complexity == "C":
                chunks = await self._strategy_complex(
                    query, top_k, filter, thinking_steps, step_counts, total_start
                )
            else:  # B (default)
                chunks = await self._strategy_moderate(
                    query, top_k, filter, thinking_steps, step_counts, total_start
                )

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000

            return RetrievalResult(
                chunks=chunks,
                query=query,
                iterations=step_counts["search_count"],
                metadata={
                    "provider": self.get_name(),
                    "complexity": complexity,
                    "strategy": strategy_names.get(complexity, "unknown"),
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"AdaptiveRAG retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "adaptive_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.generator:
            await self.generator.aclose()
