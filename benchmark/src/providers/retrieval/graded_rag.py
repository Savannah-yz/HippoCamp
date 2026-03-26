"""
Graded RAG Provider - Retrieval with optional routing and LLM grading.

Pipeline: (Route) → Retrieve → (Grade) → (Rewrite) → Return

Features (all optional, disabled by default):
- Routing: LLM decides if retrieval is needed (filters greetings etc.)
- Grading: LLM judges document relevance
- Rewriting: Improve query if no relevant docs found

Usage:
    from src.providers import GradedRAGProvider, ProviderConfig

    config = ProviderConfig(
        name="graded_rag",
        params={
            "top_k": 20,
            "enable_routing": True,
            "enable_grading": False,
        }
    )

    provider = GradedRAGProvider(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,  # For routing, grading, rewriting
    )

    await provider.setup()
    result = await provider.retrieve("What is machine learning?")
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..utils import format_query_for_embedding
from ..base import (
    RetrievalProvider,
    ProviderConfig,
    RetrievalResult,
    ChunkResult,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class AgentDecision(Enum):
    """Agent decision types."""
    RETRIEVE = "retrieve"
    ANSWER_DIRECTLY = "answer_directly"
    REWRITE_QUERY = "rewrite_query"


class GradedRAGProvider(RetrievalProvider):
    """
    Graded RAG retrieval provider with optional routing and grading.

    Implements a RAG pipeline with optional LLM-based steps:
    - Routing: Decide if retrieval is needed (filters greetings)
    - Grading: LLM judges document relevance
    - Rewriting: Improve query if no relevant docs

    Attributes:
        embedder: Embedding service for query vectorization
        vector_store: Vector database for similarity search
        generator: Generator for routing, grading, and rewriting
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        generator=None,
    ):
        """
        Initialize the Graded RAG provider.

        Args:
            config: Provider configuration with params:
                - top_k (int): Number of documents to retrieve (default: 20)
                - relevance_threshold (float): Threshold for relevance (default: 0.5)
                - max_rewrite_attempts (int): Maximum query rewrites (default: 2)
                - enable_routing (bool): Enable routing decision (default: True)
            embedder: Embedding service instance
            vector_store: Vector store instance
            generator: Generator instance for routing/grading/rewriting
        """
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator

        # Extract config params with defaults
        params = config.params
        self.default_top_k = params.get("top_k", 20)
        self.relevance_threshold = params.get("relevance_threshold", 0.5)
        self.max_rewrite_attempts = params.get("max_rewrite_attempts", 2)
        self.enable_routing = params.get("enable_routing", True)
        self.enable_grading = params.get("enable_grading", False)

    async def setup(self):
        """Initialize provider components."""
        if self._initialized:
            return

        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for routing, grading, and rewriting",
                provider=self.get_name(),
            )

        logger.info(
            f"GradedRAGProvider initialized "
            f"(threshold={self.relevance_threshold}, max_rewrites={self.max_rewrite_attempts})"
        )
        self._initialized = True

    async def _route_query(self, query: str) -> AgentDecision:
        """
        Decide whether to retrieve or answer directly.

        Args:
            query: User query

        Returns:
            AgentDecision indicating next action
        """
        # Simple keyword-based routing for common patterns
        greetings = ["hello", "hi", "hey", "greetings"]
        thanks = ["thank", "thanks"]

        query_lower = query.lower()

        # Direct answer for greetings
        for greeting in greetings:
            if greeting in query_lower and len(query.split()) <= 3:
                logger.debug("GradedRAG: Routing decision - DIRECT (greeting)")
                return AgentDecision.ANSWER_DIRECTLY

        # Direct answer for thanks
        for thank in thanks:
            if thank in query_lower and len(query.split()) <= 5:
                logger.debug("GradedRAG: Routing decision - DIRECT (thanks)")
                return AgentDecision.ANSWER_DIRECTLY

        # Use LLM for complex routing
        routing_prompt = f"""Determine if this query requires retrieving information from documents.

Query: {query}

If the query is a greeting, thank you, or casual conversation → respond "DIRECT"
If the query asks for specific information or facts → respond "RETRIEVE"

Respond with only one word: DIRECT or RETRIEVE"""

        try:
            response = await self.generator.generate(
                query=routing_prompt,
                context=None,
                max_tokens=16384,
                temperature=0.1,
            )

            decision_text = response.answer.strip().upper()

            if "DIRECT" in decision_text:
                logger.debug("GradedRAG: Routing decision - DIRECT (LLM)")
                return AgentDecision.ANSWER_DIRECTLY
            else:
                logger.debug("GradedRAG: Routing decision - RETRIEVE (LLM)")
                return AgentDecision.RETRIEVE

        except Exception as e:
            logger.warning(f"Routing failed: {e}, defaulting to RETRIEVE")
            return AgentDecision.RETRIEVE

    async def _grade_documents(
        self,
        query: str,
        chunks: List[ChunkResult],
    ) -> Tuple[List[ChunkResult], bool]:
        """
        Grade relevance of documents and determine if rewrite is needed.

        Args:
            query: The search query
            chunks: Chunks to grade

        Returns:
            Tuple of (graded chunks, needs_rewrite)
        """
        if not chunks:
            return [], True

        doc_texts = [c.content for c in chunks]

        grading_prompt = f"""Grade document relevance to the query.
For each document, provide: score (0.0-1.0), reason.

Query: {query}

Documents:
"""
        for i, text in enumerate(doc_texts, 1):
            truncated = text[:400] + "..." if len(text) > 400 else text
            grading_prompt += f"\n[Doc {i}]\n{truncated}\n"

        grading_prompt += """
Format: Doc N: score=X.X, reason=<reason>"""

        try:
            response = await self.generator.generate(
                query=grading_prompt,
                context=None,
                max_tokens=16384,
                temperature=0.1,
            )

            grade_text = response.answer
            graded_chunks = []
            relevant_count = 0

            import re

            for i, chunk in enumerate(chunks, 1):
                score = 0.5
                reason = "Grade not parsed"

                pattern = rf"Doc\s*{i}\s*:\s*score\s*=\s*([\d.]+)"
                match = re.search(pattern, grade_text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass

                reason_pattern = rf"Doc\s*{i}\s*:.*?reason\s*=\s*(.+?)(?=Doc\s*\d|$)"
                reason_match = re.search(reason_pattern, grade_text, re.IGNORECASE | re.DOTALL)
                if reason_match:
                    reason = reason_match.group(1).strip()[:80]

                is_relevant = score >= self.relevance_threshold
                if is_relevant:
                    relevant_count += 1

                graded_chunks.append(
                    ChunkResult(
                        id=chunk.id,
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata,
                        is_relevant=is_relevant,
                        relevance_reason=reason,
                    )
                )

            graded_chunks.sort(key=lambda x: x.score, reverse=True)
            needs_rewrite = relevant_count == 0

            return graded_chunks, needs_rewrite

        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            return [
                ChunkResult(
                    id=c.id,
                    content=c.content,
                    score=c.score,
                    metadata=c.metadata,
                    is_relevant=True,
                )
                for c in chunks
            ], False

    async def _rewrite_query(self, original_query: str, feedback: str = "") -> str:
        """Rewrite query for better retrieval."""
        rewrite_prompt = f"""Rewrite this query to improve document retrieval.
Make it more specific and add relevant keywords.

Original: {original_query}
{f'Issue: {feedback}' if feedback else ''}

Rewritten query:"""

        try:
            response = await self.generator.generate(
                query=rewrite_prompt,
                context=None,
                max_tokens=16384,
                temperature=0.3,
            )

            rewritten = response.answer.strip()
            for prefix in ["Rewritten query:", "Query:", "Rewritten:"]:
                if rewritten.lower().startswith(prefix.lower()):
                    rewritten = rewritten[len(prefix):].strip()

            return rewritten or original_query

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return original_query

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        skip_routing: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using Graded RAG pipeline.

        Pipeline: Route → Retrieve → Grade → (Rewrite) → Return

        Args:
            query: Search query
            top_k: Number of results to retrieve
            filter: Optional metadata filter
            skip_routing: Skip routing decision, always retrieve
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with intelligently retrieved chunks
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        decisions = []
        thinking_steps = []
        latency_breakdown = {}
        step_counts = {
            "search_count": 0,
            "rewrite_count": 0,
            "grade_count": 0,
            "route_count": 0,
        }
        step_counter = 0
        total_start = time.time()

        logger.debug(f"GradedRAG: Processing query: {query[:50]}...")

        try:
            # Step 1: Route (optional)
            if self.enable_routing and not skip_routing:
                step_counter += 1
                route_start = time.time()
                thinking_steps.append({
                    "step_id": f"route_{step_counter}",
                    "step_type": "route",
                    "title": "Analyzing Query",
                    "status": "running",
                    "summary": "Deciding whether to retrieve documents or answer directly...",
                    "timestamp_ms": (route_start - total_start) * 1000,
                })

                routing_decision = await self._route_query(query)
                route_latency = (time.time() - route_start) * 1000
                latency_breakdown["route_ms"] = route_latency
                step_counts["route_count"] = 1

                decisions.append({"step": "route", "decision": routing_decision.value})
                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Decision: {routing_decision.value}"
                thinking_steps[-1]["metadata"] = {"decision": routing_decision.value}

                if routing_decision == AgentDecision.ANSWER_DIRECTLY:
                    logger.info("GradedRAG: Routing to direct answer (no retrieval)")
                    latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                    step_counts["total_iterations"] = 0
                    return RetrievalResult(
                        chunks=[],
                        query=query,
                        metadata={
                            "provider": self.get_name(),
                            "decisions": decisions,
                            "skip_retrieval": True,
                            "reason": "Direct answer recommended",
                        },
                        latency_breakdown=latency_breakdown,
                        thinking_steps=thinking_steps,
                        step_counts=step_counts,
                    )

            # Step 2-4: Retrieve → Grade → (Rewrite) loop
            current_query = query
            attempt = 0
            graded_chunks = []
            total_embed_ms = 0.0
            total_retrieve_ms = 0.0
            total_grade_ms = 0.0
            total_rewrite_ms = 0.0

            while attempt <= self.max_rewrite_attempts:
                attempt += 1

                # Retrieve (includes embedding)
                step_counter += 1
                retrieve_start = time.time()
                thinking_steps.append({
                    "step_id": f"retrieve_{step_counter}",
                    "step_type": "retrieve",
                    "title": f"Search Attempt {attempt}",
                    "status": "running",
                    "summary": f"Searching for: {current_query[:50]}...",
                    "timestamp_ms": (retrieve_start - total_start) * 1000,
                    "metadata": {"query": current_query, "attempt": attempt},
                })

                logger.debug(f"GradedRAG: Attempt {attempt} - Retrieving")
                # Track embedding time separately
                embed_start = time.time()
                current_query_for_embed = format_query_for_embedding(current_query)
                query_embedding = await self.embedder.embed([current_query_for_embed])
                embed_latency = (time.time() - embed_start) * 1000
                total_embed_ms += embed_latency

                dense_embedding = (
                    query_embedding.dense_embeddings[0]
                    if query_embedding.dense_embeddings is not None
                    and len(query_embedding.dense_embeddings) > 0
                    else None
                )
                sparse_embedding = (
                    query_embedding.sparse_embeddings[0]
                    if query_embedding.sparse_embeddings is not None
                    and len(query_embedding.sparse_embeddings) > 0
                    else None
                )

                search_start = time.time()
                search_results = await self.vector_store.query(
                    dense_embedding=dense_embedding,
                    sparse_embedding=sparse_embedding,
                    top_k=top_k,
                    filter=filter,
                )
                search_latency = (time.time() - search_start) * 1000

                retrieved_chunks = [
                    ChunkResult(
                        id=r.get("id", ""),
                        content=r.get("chunk", ""),
                        score=r.get("score", 0.0),
                        metadata=r.get("metadata", {}),
                    )
                    for r in search_results
                ]

                retrieve_latency = (time.time() - retrieve_start) * 1000
                total_retrieve_ms += retrieve_latency
                step_counts["search_count"] += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Found {len(retrieved_chunks)} documents"
                thinking_steps[-1]["metadata"]["results_count"] = len(retrieved_chunks)
                thinking_steps[-1]["metadata"]["embed_ms"] = embed_latency
                thinking_steps[-1]["metadata"]["search_ms"] = search_latency

                decisions.append({
                    "step": "retrieve",
                    "attempt": attempt,
                    "query": current_query,
                    "count": len(retrieved_chunks),
                })

                if not retrieved_chunks:
                    if attempt <= self.max_rewrite_attempts:
                        step_counter += 1
                        rewrite_start = time.time()
                        thinking_steps.append({
                            "step_id": f"rewrite_{step_counter}",
                            "step_type": "rewrite",
                            "title": "Query Rewrite",
                            "status": "running",
                            "summary": "No documents found, rewriting query...",
                            "timestamp_ms": (rewrite_start - total_start) * 1000,
                        })

                        current_query = await self._rewrite_query(
                            query, "No documents found."
                        )
                        rewrite_latency = (time.time() - rewrite_start) * 1000
                        total_rewrite_ms += rewrite_latency
                        step_counts["rewrite_count"] += 1

                        thinking_steps[-1]["status"] = "complete"
                        thinking_steps[-1]["summary"] = f"New query: {current_query[:50]}..."
                        thinking_steps[-1]["metadata"] = {
                            "new_query": current_query,
                            "reason": "no_documents",
                        }

                        decisions.append({
                            "step": "rewrite",
                            "reason": "no_documents",
                            "new_query": current_query,
                        })
                        continue
                    break

                # Grading disabled: return retrieved chunks directly
                if not self.enable_grading:
                    final_chunks = retrieved_chunks
                    latency_breakdown["embed_query_ms"] = total_embed_ms
                    latency_breakdown["retrieve_ms"] = total_retrieve_ms
                    latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                    step_counts["total_iterations"] = attempt

                    return RetrievalResult(
                        chunks=final_chunks,
                        query=query,
                        iterations=attempt,
                        metadata={
                            "provider": self.get_name(),
                            "decisions": decisions,
                            "final_query": current_query,
                        },
                        latency_breakdown=latency_breakdown,
                        thinking_steps=thinking_steps,
                        step_counts=step_counts,
                    )

                # Grade
                step_counter += 1
                grade_start = time.time()
                thinking_steps.append({
                    "step_id": f"grade_{step_counter}",
                    "step_type": "grade",
                    "title": "Grading Relevance",
                    "status": "running",
                    "summary": f"Evaluating relevance of {len(retrieved_chunks)} documents...",
                    "timestamp_ms": (grade_start - total_start) * 1000,
                })

                logger.debug(f"GradedRAG: Grading {len(retrieved_chunks)} documents")
                graded_chunks, needs_rewrite = await self._grade_documents(
                    current_query, retrieved_chunks
                )
                grade_latency = (time.time() - grade_start) * 1000
                total_grade_ms += grade_latency
                step_counts["grade_count"] += 1

                relevant_chunks = [c for c in graded_chunks if c.is_relevant]

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"{len(relevant_chunks)}/{len(graded_chunks)} relevant"
                thinking_steps[-1]["metadata"] = {
                    "graded": len(graded_chunks),
                    "relevant": len(relevant_chunks),
                    "needs_rewrite": needs_rewrite,
                }

                decisions.append({
                    "step": "grade",
                    "graded": len(graded_chunks),
                    "relevant": len(relevant_chunks),
                    "needs_rewrite": needs_rewrite,
                })

                # Rewrite if needed
                if needs_rewrite and attempt <= self.max_rewrite_attempts:
                    step_counter += 1
                    rewrite_start = time.time()
                    thinking_steps.append({
                        "step_id": f"rewrite_{step_counter}",
                        "step_type": "rewrite",
                        "title": "Query Rewrite",
                        "status": "running",
                        "summary": "Documents not relevant, rewriting query...",
                        "timestamp_ms": (rewrite_start - total_start) * 1000,
                    })

                    logger.debug("GradedRAG: No relevant docs, rewriting")
                    current_query = await self._rewrite_query(
                        query, "Documents not relevant."
                    )
                    rewrite_latency = (time.time() - rewrite_start) * 1000
                    total_rewrite_ms += rewrite_latency
                    step_counts["rewrite_count"] += 1

                    thinking_steps[-1]["status"] = "complete"
                    thinking_steps[-1]["summary"] = f"New query: {current_query[:50]}..."
                    thinking_steps[-1]["metadata"] = {
                        "new_query": current_query,
                        "reason": "no_relevant",
                    }

                    decisions.append({
                        "step": "rewrite",
                        "reason": "no_relevant",
                        "new_query": current_query,
                    })
                    continue

                # Success
                final_chunks = relevant_chunks if relevant_chunks else graded_chunks[:5]

                # Finalize latency breakdown
                latency_breakdown["embed_query_ms"] = total_embed_ms
                latency_breakdown["retrieve_ms"] = total_retrieve_ms
                latency_breakdown["grade_ms"] = total_grade_ms
                latency_breakdown["rewrite_ms"] = total_rewrite_ms
                latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                step_counts["total_iterations"] = attempt

                return RetrievalResult(
                    chunks=final_chunks,
                    query=query,
                    rewritten_query=current_query if current_query != query else None,
                    iterations=attempt,
                    metadata={
                        "provider": self.get_name(),
                        "decisions": decisions,
                        "final_query": current_query,
                    },
                    latency_breakdown=latency_breakdown,
                    thinking_steps=thinking_steps,
                    step_counts=step_counts,
                )

            # Max attempts reached
            latency_breakdown["embed_query_ms"] = total_embed_ms
            latency_breakdown["retrieve_ms"] = total_retrieve_ms
            latency_breakdown["grade_ms"] = total_grade_ms
            latency_breakdown["rewrite_ms"] = total_rewrite_ms
            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
            step_counts["total_iterations"] = attempt

            return RetrievalResult(
                chunks=graded_chunks[:5] if graded_chunks else [],
                query=query,
                rewritten_query=current_query if current_query != query else None,
                iterations=attempt,
                metadata={
                    "provider": self.get_name(),
                    "decisions": decisions,
                    "error": "max_attempts_reached",
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"GradedRAG retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        """Get provider name."""
        return "graded_rag"

    async def aclose(self):
        """Cleanup resources."""
        if self.embedder:
            await self.embedder.aclose()
        if self.generator:
            await self.generator.aclose()
