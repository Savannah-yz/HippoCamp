"""
Query Decomposition RAG Provider - Least-to-Most Prompting style.

Faithful implementation inspired by:
    Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning in
    Large Language Models" (ICLR 2023)
    Paper: https://arxiv.org/abs/2205.10625

    Also informed by:
    - LlamaIndex SubQuestionQueryEngine
      (llama_index/core/query_engine/sub_question_query_engine.py)
    - LangChain RAG from Scratch #7 (decomposition)
      (https://github.com/langchain-ai/rag-from-scratch)

Core algorithm (Least-to-Most, 2 stages):
    Stage 1 -- DECOMPOSITION:
        Prompt: "To solve [question], I need to first solve: ..."
        LLM generates ordered list of sub-problems (simplest first)

    Stage 2 -- SEQUENTIAL SOLVING:
        context = ""
        for each sub_question in order (simplest first):
            docs = retrieve(sub_question, context_from_previous_answers)
            answer = generate(sub_question, docs, previous_answers)
            context += f"Q: {sub_question}\\nA: {answer}\\n"

    KEY: execution is SEQUENTIAL and DEPENDENT. Each sub-question's answer
    feeds into the next sub-question's prompt. This is the critical
    difference from parallel multi-query approaches.

Usage:
    provider = DecompositionRAGProvider(
        config=ProviderConfig(name="decomposition_rag", params={
            "top_k_per_sub": 10,
            "max_sub_queries": 4,
            "rerank_top_k": 5,
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


class DecompositionRAGProvider(RetrievalProvider):
    """
    Query Decomposition RAG provider using Least-to-Most style.

    Decomposes complex questions into ordered sub-questions (simplest first),
    solves each sequentially with retrieval, and accumulates context from
    previous answers to inform subsequent retrievals and answers.
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
        self.top_k_per_sub = params.get("top_k_per_sub", 10)
        self.max_sub_queries = params.get("max_sub_queries", 4)
        self.rerank_top_k = params.get("rerank_top_k", 5)
        self.use_reranker = params.get("use_reranker", True)

    async def setup(self):
        if self._initialized:
            return
        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for decomposition and answering",
                provider=self.get_name(),
            )
        logger.info(
            f"DecompositionRAGProvider initialized "
            f"(max_sub={self.max_sub_queries}, top_k_per_sub={self.top_k_per_sub})"
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
    # Stage 1: Decomposition
    # Paper: "To solve [question], I need to first solve: ..."
    # =========================================================================

    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex question into ordered sub-questions.

        Follows Least-to-Most: simplest/most foundational questions first,
        building up to the full answer.
        """
        decompose_prompt = f"""To answer the following question, we need to first answer these simpler sub-questions.
List them in order from simplest to most complex.
If the question is already simple enough to answer directly, just list it as-is.

Question: {query}

Sub-questions (from simplest to most complex):
1."""

        try:
            response = await self.generator.generate(
                query=decompose_prompt, context=None,
                max_tokens=300, temperature=0.3,
            )
            # Parse numbered sub-questions
            text = "1." + response.answer
            sub_queries = re.findall(r"\d+\.\s*(.+)", text)
            sub_queries = [q.strip() for q in sub_queries if q.strip()]

            if not sub_queries:
                return [query]

            return sub_queries[:self.max_sub_queries]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]

    # =========================================================================
    # Stage 2: Sequential solving with accumulated context
    # Paper: each sub-question's answer feeds into next prompt
    # =========================================================================

    async def _answer_sub_question(
        self,
        sub_question: str,
        retrieved_chunks: List[ChunkResult],
        previous_qa_pairs: List[Dict[str, str]],
    ) -> str:
        """
        Generate intermediate answer for a sub-question.

        Follows Least-to-Most Stage 2: includes previous Q&A pairs in prompt.
        """
        # Build context from retrieved docs
        doc_context = ""
        for i, c in enumerate(retrieved_chunks[:5], 1):
            truncated = c.content[:300] + "..." if len(c.content) > 300 else c.content
            doc_context += f"\n[Document {i}] {truncated}\n"

        # Build accumulated context from previous answers (Least-to-Most key feature)
        prev_context = ""
        if previous_qa_pairs:
            prev_context = "\nPreviously answered:\n"
            for qa in previous_qa_pairs:
                prev_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

        answer_prompt = f"""{prev_context}
Based on the above context and these documents, answer the following question concisely.

Documents:{doc_context}

Q: {sub_question}
A:"""

        try:
            response = await self.generator.generate(
                query=answer_prompt, context=None,
                max_tokens=150, temperature=0.3,
            )
            return response.answer.strip()
        except Exception as e:
            logger.warning(f"Sub-question answering failed: {e}")
            return ""

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
        Retrieve using Least-to-Most Query Decomposition pipeline.

        Stage 1: Decompose into ordered sub-questions
        Stage 2: For each sub-question (sequentially):
                 - Retrieve documents
                 - Generate answer using previous Q&A context
                 - Accumulate for next sub-question
        Final: Merge and rerank all retrieved chunks
        """
        if not self._initialized:
            await self.setup()

        logger.debug(f"Decomposition: Processing query: {query[:50]}...")

        latency_breakdown = {}
        thinking_steps = []
        step_counts = {"search_count": 0, "decompose_count": 0, "answer_count": 0}
        total_start = time.time()
        step_counter = 0

        try:
            # ── Stage 1: Decompose ─────────────────────────────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"decompose_{step_counter}",
                "step_type": "decompose",
                "title": "Stage 1: Query Decomposition",
                "status": "running",
                "summary": "Decomposing into sub-questions (Least-to-Most)...",
                "timestamp_ms": 0,
            })

            sub_queries = await self._decompose_query(query)
            step_counts["decompose_count"] = 1
            latency_breakdown["decompose_ms"] = (time.time() - t0) * 1000

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Decomposed into {len(sub_queries)} sub-questions"
            thinking_steps[-1]["metadata"] = {"sub_queries": sub_queries}

            logger.debug(f"Decomposition: {len(sub_queries)} sub-queries: {sub_queries}")

            # ── Stage 2: Sequential solving ────────────────────────────────
            all_chunks = {}  # id -> ChunkResult (accumulated, deduped)
            previous_qa_pairs = []

            for i, sub_q in enumerate(sub_queries):
                # Retrieve for this sub-question
                # Least-to-Most: use previous answer context to enhance query
                retrieval_query = sub_q
                if previous_qa_pairs:
                    # Enhance with last answer (provides context)
                    last_answer = previous_qa_pairs[-1]["answer"]
                    if last_answer:
                        retrieval_query = f"{sub_q} {last_answer[:80]}"

                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"sub_retrieve_{step_counter}",
                    "step_type": "retrieve",
                    "title": f"Sub-question {i + 1}/{len(sub_queries)}",
                    "status": "running",
                    "summary": f"Retrieving for: {sub_q[:50]}...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                    "metadata": {
                        "sub_query": sub_q,
                        "retrieval_query": retrieval_query[:100],
                        "has_prior_context": bool(previous_qa_pairs),
                    },
                })

                sub_chunks = await self._embed_and_search(
                    retrieval_query, self.top_k_per_sub, filter
                )
                step_counts["search_count"] += 1

                # Accumulate (dedup by id)
                new_added = 0
                for c in sub_chunks:
                    if c.id not in all_chunks:
                        all_chunks[c.id] = c
                        new_added += 1
                    elif c.score > all_chunks[c.id].score:
                        all_chunks[c.id] = c

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = (
                    f"Found {len(sub_chunks)} docs ({new_added} new)"
                )

                # Generate intermediate answer (Stage 2 core: sequential solving)
                # The last sub-question doesn't need intermediate answer
                if i < len(sub_queries) - 1:
                    step_counter += 1
                    thinking_steps.append({
                        "step_id": f"answer_{step_counter}",
                        "step_type": "answer",
                        "title": f"Intermediate Answer {i + 1}",
                        "status": "running",
                        "summary": f"Answering: {sub_q[:50]}...",
                        "timestamp_ms": (time.time() - total_start) * 1000,
                    })

                    answer = await self._answer_sub_question(
                        sub_q, sub_chunks, previous_qa_pairs
                    )
                    previous_qa_pairs.append({
                        "question": sub_q,
                        "answer": answer,
                    })
                    step_counts["answer_count"] += 1

                    thinking_steps[-1]["status"] = "complete"
                    thinking_steps[-1]["summary"] = f"A: {answer[:80]}..." if answer else "No answer"

            # ── Final: merge and rerank against original query ─────────────
            merged = sorted(
                all_chunks.values(), key=lambda x: x.score, reverse=True
            )

            if self.use_reranker and self.reranker and merged:
                step_counter += 1
                thinking_steps.append({
                    "step_id": f"final_rerank_{step_counter}",
                    "step_type": "rerank",
                    "title": "Final Reranking",
                    "status": "running",
                    "summary": f"Reranking {len(merged)} accumulated chunks against original query...",
                    "timestamp_ms": (time.time() - total_start) * 1000,
                })

                merged = await self._rerank_chunks(query, merged, self.rerank_top_k)
                step_counts["rerank_count"] = 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Final {len(merged)} chunks"
            else:
                merged = merged[:self.rerank_top_k]

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000

            return RetrievalResult(
                chunks=merged,
                query=query,
                iterations=step_counts["search_count"],
                metadata={
                    "provider": self.get_name(),
                    "sub_queries": sub_queries,
                    "num_sub_queries": len(sub_queries),
                    "intermediate_qa_pairs": previous_qa_pairs,
                    "total_unique_chunks": len(all_chunks),
                    "execution_style": "sequential_least_to_most",
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"Decomposition retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "decomposition_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.generator:
            await self.generator.aclose()
