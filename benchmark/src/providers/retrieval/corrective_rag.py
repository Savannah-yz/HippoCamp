"""
Corrective RAG (CRAG) Provider.

Faithful implementation of CRAG from:
    Yan et al., "Corrective Retrieval Augmented Generation" (ICLR 2024)
    Paper: https://arxiv.org/abs/2401.15884
    Official repo: https://github.com/HuskyInSalt/CRAG

Algorithm 1 from the paper:
    1. Score each retrieved document d_i with evaluator E(x, d_i)
    2. Confidence = Judge(scores) -> {CORRECT, INCORRECT, AMBIGUOUS}
    3. CORRECT:   k = Knowledge_Refine(x, D)      -- decompose-then-recompose
    4. INCORRECT: k = Rewrite(x) -> Re-retrieve    -- (web search in paper, re-retrieval here)
    5. AMBIGUOUS: k = Knowledge_Refine(x, D) + Re-retrieve with rewritten query

Adaptations from the original paper:
    - Retrieval evaluator: LLM-as-judge instead of fine-tuned T5-large
    - Web search: replaced with query rewrite + re-retrieval (no external web search)
    - Knowledge refinement: decompose-then-recompose preserved faithfully
      (split into strips -> score each strip -> filter irrelevant -> recompose)

Usage:
    provider = CorrectiveRAGProvider(
        config=ProviderConfig(name="corrective_rag", params={
            "top_k": 20,
            "confidence_threshold_upper": 0.5,
            "confidence_threshold_lower": -0.5,
            "strip_relevance_threshold": -0.5,
            "max_strips": 5,
            "rerank_top_k": 5,
        }),
        embedder=embedder, vector_store=vector_store,
        reranker=reranker, generator=generator,
    )
"""

import logging
import re
import time
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


class CorrectiveRAGProvider(RetrievalProvider):
    """
    Corrective RAG retrieval provider.

    Implements the CRAG pipeline with 3-way retrieval evaluation and
    decompose-then-recompose knowledge refinement.
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
        # Paper uses per-dataset thresholds; these are reasonable defaults.
        # CORRECT if any doc >= upper; INCORRECT if all docs < lower; else AMBIGUOUS.
        self.confidence_upper = params.get("confidence_threshold_upper", 0.5)
        self.confidence_lower = params.get("confidence_threshold_lower", -0.5)
        # Threshold for filtering knowledge strips in decompose-then-recompose
        self.strip_threshold = params.get("strip_relevance_threshold", -0.5)
        self.max_strips = params.get("max_strips", 5)

    async def setup(self):
        if self._initialized:
            return
        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for CRAG evaluation and refinement",
                provider=self.get_name(),
            )
        logger.info(
            f"CorrectiveRAGProvider initialized "
            f"(upper={self.confidence_upper}, lower={self.confidence_lower})"
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
            dense_embedding=dense,
            sparse_embedding=sparse,
            top_k=top_k,
            filter=filter,
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
    # Step 1: Retrieval Evaluator  (paper: fine-tuned T5-large; here: LLM)
    # Scores each (query, document) pair and returns a score in [-1, 1].
    # =========================================================================

    async def _evaluate_documents(
        self, query: str, chunks: List[ChunkResult]
    ) -> List[float]:
        """Score each document's relevance to the query. Returns list of scores in [-1, 1]."""
        if not chunks:
            return []

        doc_texts = []
        for i, c in enumerate(chunks, 1):
            truncated = c.content[:500] + "..." if len(c.content) > 500 else c.content
            doc_texts.append(f"[Document {i}]\n{truncated}")

        eval_prompt = f"""Score the relevance of each retrieved document to the query.
For each document, output a relevance score from -1.0 (completely irrelevant) to 1.0 (highly relevant).

Query: {query}

{chr(10).join(doc_texts)}

Respond in this exact format (one line per document):
Document 1: score=X.X
Document 2: score=X.X
..."""

        try:
            response = await self.generator.generate(
                query=eval_prompt, context=None,
                max_tokens=50 + len(chunks) * 30, temperature=0.1,
            )
            text = response.answer
            scores = []
            for i in range(1, len(chunks) + 1):
                match = re.search(
                    rf"Document\s*{i}\s*:\s*score\s*=\s*(-?[\d.]+)", text, re.IGNORECASE
                )
                if match:
                    s = max(-1.0, min(1.0, float(match.group(1))))
                    scores.append(s)
                else:
                    scores.append(0.0)
            return scores
        except Exception as e:
            logger.warning(f"CRAG evaluation failed: {e}, returning neutral scores")
            return [0.0] * len(chunks)

    # =========================================================================
    # Step 2: Confidence Judge
    # Paper: CORRECT if any score >= upper; INCORRECT if all < lower; else AMBIGUOUS.
    # =========================================================================

    def _judge_confidence(self, scores: List[float]) -> str:
        if not scores:
            return "incorrect"
        if any(s >= self.confidence_upper for s in scores):
            return "correct"
        if all(s < self.confidence_lower for s in scores):
            return "incorrect"
        return "ambiguous"

    # =========================================================================
    # Step 3: Knowledge Refinement (Decompose-then-Recompose)
    # Paper: split document into strips (fixed_num / excerption / selection),
    # score each strip, filter irrelevant, concatenate remaining.
    # =========================================================================

    def _decompose_into_strips(self, text: str) -> List[str]:
        """
        Decompose document into knowledge strips.

        Follows the paper's 'excerption' strategy: split on sentence boundaries,
        filter very short strips, then group every ~2-3 sentences.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

        if not sentences:
            return [text] if text.strip() else []

        # Group into strips of ~2-3 sentences (similar to paper's excerption)
        strips = []
        for i in range(0, len(sentences), 2):
            strip = " ".join(sentences[i:i + 2])
            strips.append(strip)

        return strips

    async def _score_strip(self, query: str, strip: str) -> float:
        """Score a single knowledge strip against the query."""
        score_prompt = f"""Score how relevant this text strip is to the query.
Output a single number from -1.0 (irrelevant) to 1.0 (highly relevant).

Query: {query}
Strip: {strip[:300]}

Score:"""
        try:
            response = await self.generator.generate(
                query=score_prompt, context=None, max_tokens=20, temperature=0.1,
            )
            match = re.search(r"(-?[\d.]+)", response.answer)
            if match:
                return max(-1.0, min(1.0, float(match.group(1))))
            return 0.0
        except Exception:
            return 0.0

    async def _knowledge_refine(
        self, query: str, chunks: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Decompose-then-recompose: split each document into strips, score each,
        filter by threshold, recompose from remaining strips.
        """
        refined_chunks = []

        for chunk in chunks:
            strips = self._decompose_into_strips(chunk.content)
            if not strips:
                continue

            # Score strips in batch
            strip_prompt_parts = []
            for i, strip in enumerate(strips, 1):
                truncated = strip[:200] + "..." if len(strip) > 200 else strip
                strip_prompt_parts.append(f"Strip {i}: {truncated}")

            batch_prompt = f"""Score relevance of each strip to the query (-1.0 to 1.0).

Query: {query}

{chr(10).join(strip_prompt_parts)}

Respond:
Strip 1: score=X.X
Strip 2: score=X.X
..."""

            try:
                response = await self.generator.generate(
                    query=batch_prompt, context=None,
                    max_tokens=30 + len(strips) * 20, temperature=0.1,
                )
                text = response.answer

                # Parse scores and filter
                kept_strips = []
                for i, strip in enumerate(strips, 1):
                    match = re.search(
                        rf"Strip\s*{i}\s*:\s*score\s*=\s*(-?[\d.]+)",
                        text, re.IGNORECASE,
                    )
                    score = 0.0
                    if match:
                        score = max(-1.0, min(1.0, float(match.group(1))))
                    if score >= self.strip_threshold:
                        kept_strips.append(strip)

                if kept_strips:
                    refined_content = " ".join(kept_strips[:self.max_strips])
                    refined_chunks.append(
                        ChunkResult(
                            id=chunk.id,
                            content=refined_content,
                            score=chunk.score,
                            metadata=chunk.metadata,
                            is_relevant=True,
                            relevance_reason="CRAG: knowledge refined (decompose-then-recompose)",
                        )
                    )
            except Exception as e:
                logger.warning(f"Strip scoring failed for chunk {chunk.id}: {e}")
                # Keep original chunk as fallback
                refined_chunks.append(
                    ChunkResult(
                        id=chunk.id, content=chunk.content, score=chunk.score,
                        metadata=chunk.metadata, is_relevant=True,
                        relevance_reason="CRAG: refinement failed, kept original",
                    )
                )

        return refined_chunks

    # =========================================================================
    # Step 4: Query Rewrite (paper uses ChatGPT -> keyword-based search query)
    # =========================================================================

    async def _rewrite_query(self, query: str) -> str:
        """Rewrite query into keyword-based search query (paper: max 3 keywords)."""
        rewrite_prompt = f"""Rewrite this question as a short keyword-based search query (max 5 keywords).
Focus on the core information need.

Question: {query}
Search query:"""
        try:
            response = await self.generator.generate(
                query=rewrite_prompt, context=None, max_tokens=50, temperature=0.3,
            )
            rewritten = response.answer.strip()
            for prefix in ["Search query:", "Query:", "Keywords:"]:
                if rewritten.lower().startswith(prefix.lower()):
                    rewritten = rewritten[len(prefix):].strip()
            return rewritten or query
        except Exception as e:
            logger.warning(f"CRAG query rewrite failed: {e}")
            return query

    # =========================================================================
    # Main retrieve pipeline
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
        Retrieve using CRAG pipeline (Algorithm 1 from paper).

        1. Retrieve -> 2. Evaluate each doc -> 3. Judge confidence ->
        4a. CORRECT: Knowledge_Refine(D)
        4b. INCORRECT: Rewrite -> Re-retrieve
        4c. AMBIGUOUS: Knowledge_Refine(D) + Rewrite -> Re-retrieve
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        logger.debug(f"CRAG: Processing query: {query[:50]}...")

        latency_breakdown = {}
        thinking_steps = []
        step_counts = {
            "search_count": 0, "evaluate_count": 0,
            "refine_count": 0, "rewrite_count": 0,
        }
        total_start = time.time()
        step_counter = 0

        try:
            # ── Step 1: Initial retrieval ──────────────────────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"retrieve_{step_counter}",
                "step_type": "retrieve",
                "title": "Initial Retrieval",
                "status": "running",
                "summary": f"Retrieving for: {query[:50]}...",
                "timestamp_ms": (t0 - total_start) * 1000,
                "metadata": {"query": query},
            })

            retrieved = await self._embed_and_search(query, top_k, filter)
            step_counts["search_count"] += 1
            latency_breakdown["retrieve_ms"] = (time.time() - t0) * 1000

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Found {len(retrieved)} documents"

            if not retrieved:
                latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                return RetrievalResult(
                    chunks=[], query=query,
                    metadata={"provider": self.get_name(), "verdict": "no_documents"},
                    latency_breakdown=latency_breakdown,
                    thinking_steps=thinking_steps, step_counts=step_counts,
                )

            # Optional rerank before evaluation
            if self.use_reranker and self.reranker:
                retrieved = await self._rerank_chunks(query, retrieved, self.rerank_top_k)

            # ── Step 2: Evaluate each document ─────────────────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"evaluate_{step_counter}",
                "step_type": "evaluate",
                "title": "Evaluating Documents",
                "status": "running",
                "summary": f"Scoring {len(retrieved)} documents...",
                "timestamp_ms": (t0 - total_start) * 1000,
            })

            scores = await self._evaluate_documents(query, retrieved)
            step_counts["evaluate_count"] += 1
            latency_breakdown["evaluate_ms"] = (time.time() - t0) * 1000

            # ── Step 3: Judge confidence ───────────────────────────────────
            verdict = self._judge_confidence(scores)

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Verdict: {verdict.upper()} "
                f"(scores: {[f'{s:.2f}' for s in scores]})"
            )
            thinking_steps[-1]["metadata"] = {
                "scores": scores, "verdict": verdict,
            }

            logger.debug(f"CRAG: Verdict={verdict}, scores={scores}")

            # ── Step 4: Act on verdict ─────────────────────────────────────
            final_chunks = []

            if verdict == "correct":
                # Knowledge refinement: decompose-then-recompose
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"refine_{step_counter}",
                    "step_type": "refine",
                    "title": "Knowledge Refinement",
                    "status": "running",
                    "summary": "Decompose-then-recompose on relevant docs...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                })

                final_chunks = await self._knowledge_refine(query, retrieved)
                step_counts["refine_count"] += 1
                latency_breakdown["refine_ms"] = (time.time() - t0) * 1000

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Refined to {len(final_chunks)} chunks"

            elif verdict == "incorrect":
                # Rewrite query and re-retrieve
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"rewrite_{step_counter}",
                    "step_type": "rewrite",
                    "title": "Query Rewrite (Corrective)",
                    "status": "running",
                    "summary": "Documents irrelevant, rewriting query...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                })

                rewritten = await self._rewrite_query(query)
                step_counts["rewrite_count"] += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Rewritten: {rewritten[:50]}..."

                # Re-retrieve with rewritten query
                step_counter += 1
                thinking_steps.append({
                    "step_id": f"re_retrieve_{step_counter}",
                    "step_type": "retrieve",
                    "title": "Corrective Re-retrieval",
                    "status": "running",
                    "summary": f"Re-retrieving with: {rewritten[:50]}...",
                    "timestamp_ms": (time.time() - total_start) * 1000,
                })

                final_chunks = await self._embed_and_search(rewritten, top_k, filter)
                step_counts["search_count"] += 1

                if self.use_reranker and self.reranker and final_chunks:
                    final_chunks = await self._rerank_chunks(
                        rewritten, final_chunks, self.rerank_top_k
                    )

                latency_breakdown["corrective_ms"] = (time.time() - t0) * 1000

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Re-retrieved {len(final_chunks)} documents"

            else:  # ambiguous
                # Combine: Knowledge_Refine(D) + Re-retrieve with rewritten query
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"refine_{step_counter}",
                    "step_type": "refine",
                    "title": "Knowledge Refinement (Ambiguous)",
                    "status": "running",
                    "summary": "Refining partially relevant documents...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                })

                internal_chunks = await self._knowledge_refine(query, retrieved)
                step_counts["refine_count"] += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Internal: {len(internal_chunks)} refined chunks"

                # Also rewrite and re-retrieve
                step_counter += 1
                thinking_steps.append({
                    "step_id": f"rewrite_{step_counter}",
                    "step_type": "rewrite",
                    "title": "Supplementary Query Rewrite",
                    "status": "running",
                    "summary": "Generating supplementary retrieval...",
                    "timestamp_ms": (time.time() - total_start) * 1000,
                })

                rewritten = await self._rewrite_query(query)
                step_counts["rewrite_count"] += 1
                external_chunks = await self._embed_and_search(rewritten, top_k, filter)
                step_counts["search_count"] += 1

                if self.use_reranker and self.reranker and external_chunks:
                    external_chunks = await self._rerank_chunks(
                        rewritten, external_chunks, self.rerank_top_k
                    )

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"External: {len(external_chunks)} supplementary chunks"

                # Combine: internal (refined) + external (re-retrieved)
                # Paper: "Knowledge1: ... [sep] Knowledge2: ..."
                seen_ids = {c.id for c in internal_chunks}
                for c in external_chunks:
                    if c.id not in seen_ids:
                        c.relevance_reason = "CRAG: supplementary retrieval (ambiguous)"
                        internal_chunks.append(c)
                        seen_ids.add(c.id)

                latency_breakdown["ambiguous_ms"] = (time.time() - t0) * 1000
                final_chunks = internal_chunks

            # ── Return result ──────────────────────────────────────────────
            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
            step_counts["total_iterations"] = 1

            return RetrievalResult(
                chunks=final_chunks,
                query=query,
                rewritten_query=None,
                iterations=step_counts["search_count"],
                metadata={
                    "provider": self.get_name(),
                    "verdict": verdict,
                    "doc_scores": scores,
                    "action": verdict,
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"CRAG retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "corrective_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.generator:
            await self.generator.aclose()
