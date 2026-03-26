"""
Self-RAG Provider - Retrieval with reflection and iterative refinement.

This provider implements Self-RAG from HippoCamp with the following pipeline:
1. Retrieve documents
2. Grade relevance of each document
3. Filter to keep only relevant documents
4. Optionally iterate with query rewriting if no relevant docs found

Pipeline: Retrieve → Grade → Filter → (Iterate with Rewrite)

This is equivalent to HippoCamp's self_rag.py implementation.

Usage:
    from src.providers import SelfRAGProvider, ProviderConfig

    config = ProviderConfig(
        name="self_rag",
        params={
            "top_k": 20,
            "relevance_threshold": 0.5,
            "max_iterations": 2,
        }
    )

    provider = SelfRAGProvider(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,  # For grading and rewriting
    )

    await provider.setup()
    result = await provider.retrieve("What is machine learning?")
"""

import logging
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


class SelfRAGProvider(RetrievalProvider):
    """
    Self-RAG retrieval provider with reflection.

    Implements an advanced RAG pipeline with:
    - Relevance grading: LLM judges document relevance
    - Filtering: Only keep relevant documents
    - Iteration: Rewrite query and retry if no relevant docs

    This matches the HippoCamp SelfRAG implementation.

    Attributes:
        embedder: Embedding service for query vectorization
        vector_store: Vector database for similarity search
        generator: Generator for relevance grading and query rewriting
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        generator=None,
    ):
        """
        Initialize the Self-RAG provider.

        Args:
            config: Provider configuration with params:
                - top_k (int): Number of documents to retrieve (default: 20)
                - relevance_threshold (float): Threshold for relevance (default: 0.5)
                - max_iterations (int): Maximum refinement iterations (default: 2)
                - return_all_graded (bool): Return all graded docs (default: False)
            embedder: Embedding service instance
            vector_store: Vector store instance
            generator: Generator instance for grading/rewriting
        """
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator

        # Extract config params with defaults
        params = config.params
        self.default_top_k = params.get("top_k", 20)
        self.relevance_threshold = params.get("relevance_threshold", 0.5)
        self.max_iterations = params.get("max_iterations", 2)
        self.return_all_graded = params.get("return_all_graded", False)

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
                "Generator is required for grading and rewriting",
                provider=self.get_name(),
            )

        logger.info(
            f"SelfRAGProvider initialized "
            f"(threshold={self.relevance_threshold}, max_iter={self.max_iterations})"
        )
        self._initialized = True

    async def _grade_documents(
        self,
        query: str,
        chunks: List[ChunkResult],
    ) -> List[ChunkResult]:
        """
        Grade relevance of documents using LLM.

        Uses batch grading for efficiency (single API call).

        Args:
            query: The search query
            chunks: Chunks to grade

        Returns:
            Chunks with relevance scores and is_relevant flags
        """
        if not chunks:
            return []

        # Build grading prompt
        doc_texts = [c.content for c in chunks]

        grading_prompt = f"""Grade the relevance of each document to the query.
For each document, provide a relevance score from 0.0 to 1.0 and a brief reason.

Query: {query}

Documents:
"""
        for i, text in enumerate(doc_texts, 1):
            # Truncate long documents
            truncated = text[:500] + "..." if len(text) > 500 else text
            grading_prompt += f"\n[Document {i}]\n{truncated}\n"

        grading_prompt += f"""
For each document, respond in this exact format:
Document 1: score=X.X, reason=<brief reason>
Document 2: score=X.X, reason=<brief reason>
...

Only include the document number, score, and reason. Be concise."""

        try:
            # Generate grades
            from ..base import ChunkResult as CR

            response = await self.generator.generate(
                query=grading_prompt,
                context=None,
                max_tokens=200 + len(chunks) * 50,
                temperature=0.1,
            )

            grade_text = response.answer

            # Parse grades (simple parsing)
            graded_chunks = []
            for i, chunk in enumerate(chunks, 1):
                # Try to find grade for this document
                score = 0.5  # Default
                reason = "Could not parse grade"

                # Look for patterns like "Document 1: score=0.8"
                import re

                pattern = rf"Document\s*{i}\s*:\s*score\s*=\s*([\d.]+)"
                match = re.search(pattern, grade_text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    except ValueError:
                        pass

                # Extract reason if present
                reason_pattern = rf"Document\s*{i}\s*:.*?reason\s*=\s*(.+?)(?=Document\s*\d|$)"
                reason_match = re.search(reason_pattern, grade_text, re.IGNORECASE | re.DOTALL)
                if reason_match:
                    reason = reason_match.group(1).strip()[:100]

                is_relevant = score >= self.relevance_threshold

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

            # Sort by score descending
            graded_chunks.sort(key=lambda x: x.score, reverse=True)

            return graded_chunks

        except Exception as e:
            logger.warning(f"Grading failed: {e}, returning ungraded chunks")
            # Return chunks with default grades
            return [
                ChunkResult(
                    id=c.id,
                    content=c.content,
                    score=c.score,
                    metadata=c.metadata,
                    is_relevant=True,  # Assume relevant if grading fails
                    relevance_reason="Grading failed, assuming relevant",
                )
                for c in chunks
            ]

    async def _rewrite_query(self, query: str, feedback: str = "") -> str:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query
            feedback: Why rewrite is needed

        Returns:
            Rewritten query
        """
        rewrite_prompt = f"""Rewrite this query to be more specific and suitable for document retrieval.
Keep it concise but add relevant keywords.

Original query: {query}
{f'Feedback: {feedback}' if feedback else ''}

Rewritten query:"""

        try:
            response = await self.generator.generate(
                query=rewrite_prompt,
                context=None,
                max_tokens=100,
                temperature=0.3,
            )

            rewritten = response.answer.strip()
            # Clean up common prefixes
            for prefix in ["Rewritten query:", "Query:", "New query:"]:
                if rewritten.lower().startswith(prefix.lower()):
                    rewritten = rewritten[len(prefix):].strip()

            return rewritten or query

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        skip_grading: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks using Self-RAG pipeline.

        Pipeline: Retrieve → Grade → Filter → (Iterate with Rewrite)

        Args:
            query: Search query
            top_k: Number of results to retrieve (overrides config)
            filter: Optional metadata filter
            skip_grading: Skip relevance grading
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with graded and filtered chunks
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        logger.debug(f"SelfRAG: Processing query: {query[:50]}...")

        iteration = 0
        current_query = query
        all_iterations = []

        # Initialize latency and step tracking
        latency_breakdown = {}
        thinking_steps = []
        step_counts = {
            "search_count": 0,
            "grade_count": 0,
            "rewrite_count": 0,
        }
        total_start = time.time()
        step_counter = 0
        total_embed_ms = 0.0
        total_retrieve_ms = 0.0
        total_grade_ms = 0.0
        total_rewrite_ms = 0.0

        try:
            while iteration < self.max_iterations:
                iteration += 1
                iter_data = {"iteration": iteration, "query": current_query}

                # Step 1: Retrieve (includes embedding)
                step_counter += 1
                retrieve_start = time.time()
                thinking_steps.append({
                    "step_id": f"retrieve_{step_counter}",
                    "step_type": "retrieve",
                    "title": f"Search Iteration {iteration}",
                    "status": "running",
                    "summary": f"Searching for: {current_query[:50]}...",
                    "timestamp_ms": (retrieve_start - total_start) * 1000,
                    "metadata": {"query": current_query, "iteration": iteration},
                })

                logger.debug(f"SelfRAG: Iteration {iteration} - Retrieving")

                # Track embedding time
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

                iter_data["retrieved_count"] = len(retrieved_chunks)

                if not retrieved_chunks:
                    logger.warning(f"SelfRAG: No documents in iteration {iteration}")
                    iter_data["error"] = "no_documents"
                    all_iterations.append(iter_data)

                    if iteration < self.max_iterations:
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
                            query, "No documents found. Try more specific terms."
                        )
                        rewrite_latency = (time.time() - rewrite_start) * 1000
                        total_rewrite_ms += rewrite_latency
                        step_counts["rewrite_count"] += 1

                        thinking_steps[-1]["status"] = "complete"
                        thinking_steps[-1]["summary"] = f"New query: {current_query[:50]}..."
                        thinking_steps[-1]["metadata"] = {"new_query": current_query}

                        iter_data["rewritten_query"] = current_query
                        continue
                    break

                # Step 2: Grade relevance
                if not skip_grading:
                    step_counter += 1
                    grade_start = time.time()
                    thinking_steps.append({
                        "step_id": f"grade_{step_counter}",
                        "step_type": "grade",
                        "title": "Grading Relevance",
                        "status": "running",
                        "summary": f"Evaluating {len(retrieved_chunks)} documents...",
                        "timestamp_ms": (grade_start - total_start) * 1000,
                    })

                    logger.debug(f"SelfRAG: Grading {len(retrieved_chunks)} documents")
                    graded_chunks = await self._grade_documents(current_query, retrieved_chunks)

                    grade_latency = (time.time() - grade_start) * 1000
                    total_grade_ms += grade_latency
                    step_counts["grade_count"] += 1

                    relevant_chunks = [c for c in graded_chunks if c.is_relevant]

                    thinking_steps[-1]["status"] = "complete"
                    thinking_steps[-1]["summary"] = f"{len(relevant_chunks)}/{len(graded_chunks)} relevant"
                    thinking_steps[-1]["metadata"] = {
                        "graded": len(graded_chunks),
                        "relevant": len(relevant_chunks),
                    }
                else:
                    graded_chunks = retrieved_chunks
                    for c in graded_chunks:
                        c.is_relevant = True
                    relevant_chunks = graded_chunks

                iter_data["graded_count"] = len(graded_chunks)

                # Step 3: Filter relevant
                iter_data["relevant_count"] = len(relevant_chunks)

                logger.debug(
                    f"SelfRAG: {len(relevant_chunks)}/{len(graded_chunks)} relevant"
                )

                # Check if we have relevant documents
                if not relevant_chunks and iteration < self.max_iterations:
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

                    logger.debug("SelfRAG: No relevant docs, rewriting query")
                    current_query = await self._rewrite_query(
                        query, "Retrieved documents were not relevant."
                    )
                    rewrite_latency = (time.time() - rewrite_start) * 1000
                    total_rewrite_ms += rewrite_latency
                    step_counts["rewrite_count"] += 1

                    thinking_steps[-1]["status"] = "complete"
                    thinking_steps[-1]["summary"] = f"New query: {current_query[:50]}..."
                    thinking_steps[-1]["metadata"] = {"new_query": current_query}

                    iter_data["rewritten_query"] = current_query
                    all_iterations.append(iter_data)
                    continue

                # Success - return results
                all_iterations.append(iter_data)

                # Decide what to return
                final_chunks = relevant_chunks if relevant_chunks else graded_chunks[:5]

                # Finalize latency breakdown
                latency_breakdown["embed_query_ms"] = total_embed_ms
                latency_breakdown["retrieve_ms"] = total_retrieve_ms
                latency_breakdown["grade_ms"] = total_grade_ms
                latency_breakdown["rewrite_ms"] = total_rewrite_ms
                latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                step_counts["total_iterations"] = iteration

                return RetrievalResult(
                    chunks=final_chunks,
                    query=query,
                    rewritten_query=current_query if current_query != query else None,
                    iterations=iteration,
                    metadata={
                        "provider": self.get_name(),
                        "iteration_details": all_iterations,
                        "total_retrieved": sum(i.get("retrieved_count", 0) for i in all_iterations),
                        "total_relevant": len(relevant_chunks),
                    },
                    latency_breakdown=latency_breakdown,
                    thinking_steps=thinking_steps,
                    step_counts=step_counts,
                )

            # Max iterations reached without good results
            latency_breakdown["embed_query_ms"] = total_embed_ms
            latency_breakdown["retrieve_ms"] = total_retrieve_ms
            latency_breakdown["grade_ms"] = total_grade_ms
            latency_breakdown["rewrite_ms"] = total_rewrite_ms
            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
            step_counts["total_iterations"] = iteration

            return RetrievalResult(
                chunks=[],
                query=query,
                rewritten_query=current_query if current_query != query else None,
                iterations=iteration,
                metadata={
                    "provider": self.get_name(),
                    "iteration_details": all_iterations,
                    "error": "max_iterations_reached",
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"SelfRAG retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        """Get provider name."""
        return "self_rag"

    async def aclose(self):
        """Cleanup resources."""
        if self.embedder:
            await self.embedder.aclose()
        if self.generator:
            await self.generator.aclose()
