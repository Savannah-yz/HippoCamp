"""
HyDE RAG Provider - Hypothetical Document Embeddings.

Faithful implementation of HyDE from:
    Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023)
    Paper: https://arxiv.org/abs/2212.10496
    Official repo: https://github.com/texttron/hyde

Core algorithm from the paper and official code (hyde.py::encode):
    1. Generate N hypothetical documents using task-specific prompts
       (default: N=8, temperature=0.7, max_tokens=512)
    2. Embed each hypothetical document AND the original query
    3. AVERAGE all N+1 embeddings into a single vector:
       hyde_vector = mean([embed(query)] + [embed(hyp_doc_i) for i in range(N)])
    4. Search the vector store with this averaged embedding

Key implementation details from official code:
    - Hypothetical documents use task-specific prompt templates (web/scifact/etc.)
    - Original query IS included in the averaging (code: [query] + hypothesis_documents)
    - Encoder: Contriever (facebook/contriever) with mean pooling
    - Generator defaults: n=8, temperature=0.7, stop='\\n\\n\\n'

Adaptations:
    - Prompt: adapted for personal file system retrieval (not web search)
    - Encoder: uses project's existing embedder instead of Contriever
    - N defaults to 4 (fewer than paper's 8 for cost, configurable)
    - Added optional reranking (not in original paper)

Usage:
    provider = HyDERAGProvider(
        config=ProviderConfig(name="hyde_rag", params={
            "top_k": 20,
            "num_hypothetical": 4,
            "rerank_top_k": 5,
        }),
        embedder=embedder, vector_store=vector_store,
        reranker=reranker, generator=generator,
    )
"""

import logging
import time
import numpy as np
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


class HyDERAGProvider(RetrievalProvider):
    """
    HyDE RAG provider using hypothetical document embeddings.

    Generates hypothetical answers, embeds them together with the original
    query, averages all embeddings, and uses the averaged vector for retrieval.
    Faithfully follows the official texttron/hyde implementation.
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
        # Paper default: 8. We default to 4 for cost efficiency.
        self.num_hypothetical = params.get("num_hypothetical", 4)
        self.use_reranker = params.get("use_reranker", True)
        # Paper: temperature=0.7 for diverse hypothetical docs
        self.generation_temperature = params.get("generation_temperature", 0.7)

    async def setup(self):
        if self._initialized:
            return
        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for hypothetical document generation",
                provider=self.get_name(),
            )
        logger.info(
            f"HyDERAGProvider initialized "
            f"(num_hypothetical={self.num_hypothetical}, "
            f"temperature={self.generation_temperature})"
        )
        self._initialized = True

    # =========================================================================
    # Step 1: Generate hypothetical documents
    # Paper prompt: "Please write a passage to answer the question.\n
    #               Question: {query}\nPassage:"
    # =========================================================================

    async def _generate_hypothetical_documents(self, query: str) -> List[str]:
        """
        Generate N hypothetical documents that would answer the query.

        Uses a prompt adapted from the paper's web_search template,
        modified for personal file retrieval context.
        """
        # Paper's original prompt for web search:
        #   "Please write a passage to answer the question."
        # Adapted for personal file system context:
        prompt = f"""Please write a short passage to answer the question.
The passage should read like an excerpt from a real personal document
(report, email, note, spreadsheet description, etc.).

Question: {query}
Passage:"""

        hypothetical_docs = []
        for i in range(self.num_hypothetical):
            try:
                response = await self.generator.generate(
                    query=prompt, context=None,
                    max_tokens=512,  # Paper default
                    temperature=self.generation_temperature,
                )
                doc = response.answer.strip()
                if doc:
                    hypothetical_docs.append(doc)
            except Exception as e:
                logger.warning(f"Hypothetical doc {i+1} generation failed: {e}")

        if not hypothetical_docs:
            logger.warning("All hypothetical doc generations failed, using query as fallback")
            hypothetical_docs = [query]

        return hypothetical_docs

    # =========================================================================
    # Step 2-3: Embed and average (faithful to hyde.py::encode)
    # Official code: all_emb_c = [embed(query)] + [embed(hyp_doc_i) for i]
    #                avg_emb_c = np.mean(all_emb_c, axis=0)
    # =========================================================================

    async def _compute_hyde_embedding(
        self, query: str, hypothetical_docs: List[str]
    ):
        """
        Compute the HyDE averaged embedding.

        Following official code exactly:
        1. Embed the original query
        2. Embed each hypothetical document
        3. Average ALL embeddings (query + all hypothetical docs)

        Returns:
            Tuple of (averaged_dense_embedding, averaged_sparse_embedding_or_None)
        """
        # Collect all texts to embed: [query] + hypothetical_docs
        # Paper: query is included in the averaging
        all_texts = [query] + hypothetical_docs

        # Embed all texts
        # Note: documents in the paper are embedded WITHOUT instruction prefix.
        # The query gets the instruction prefix per our embedding convention.
        all_dense_embeddings = []
        all_sparse_embeddings = []

        for idx, text in enumerate(all_texts):
            # First text is the query (with instruction prefix),
            # rest are hypothetical docs (no prefix, like stored documents)
            if idx == 0:
                text_for_embed = format_query_for_embedding(text)
            else:
                text_for_embed = text

            emb_result = await self.embedder.embed([text_for_embed])

            if (emb_result.dense_embeddings is not None
                    and len(emb_result.dense_embeddings) > 0):
                all_dense_embeddings.append(
                    np.array(emb_result.dense_embeddings[0])
                )

            if (emb_result.sparse_embeddings is not None
                    and len(emb_result.sparse_embeddings) > 0):
                all_sparse_embeddings.append(emb_result.sparse_embeddings[0])

        # Average dense embeddings: np.mean(all_emb_c, axis=0) from paper
        avg_dense = None
        if all_dense_embeddings:
            avg_dense = np.mean(all_dense_embeddings, axis=0).tolist()

        # For sparse embeddings, we just use the query's sparse embedding
        # (averaging sparse vectors is not standard and not in the paper)
        avg_sparse = all_sparse_embeddings[0] if all_sparse_embeddings else None

        return avg_dense, avg_sparse

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
        Retrieve using HyDE pipeline.

        1. Generate N hypothetical documents
        2. Embed query + all hypothetical docs
        3. Average all embeddings into one vector
        4. Search with averaged vector
        5. Optional rerank against original query
        """
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k
        logger.debug(f"HyDE: Processing query: {query[:50]}...")

        latency_breakdown = {}
        thinking_steps = []
        step_counts = {"search_count": 0, "hypothetical_count": 0}
        total_start = time.time()
        step_counter = 0

        try:
            # ── Step 1: Generate hypothetical documents ────────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"generate_{step_counter}",
                "step_type": "hypothetical",
                "title": f"Generating {self.num_hypothetical} Hypothetical Documents",
                "status": "running",
                "summary": "Creating hypothetical answer passages...",
                "timestamp_ms": 0,
            })

            hyp_docs = await self._generate_hypothetical_documents(query)
            step_counts["hypothetical_count"] = len(hyp_docs)
            latency_breakdown["hypothetical_gen_ms"] = (time.time() - t0) * 1000

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Generated {len(hyp_docs)} hypothetical documents"
            )
            thinking_steps[-1]["metadata"] = {
                "hypothetical_docs": [d[:150] + "..." for d in hyp_docs],
            }

            # ── Step 2-3: Compute averaged HyDE embedding ─────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"embed_{step_counter}",
                "step_type": "embed",
                "title": "Computing HyDE Averaged Embedding",
                "status": "running",
                "summary": f"Averaging query + {len(hyp_docs)} hypothetical doc embeddings...",
                "timestamp_ms": (t0 - total_start) * 1000,
            })

            avg_dense, avg_sparse = await self._compute_hyde_embedding(query, hyp_docs)
            latency_breakdown["embed_avg_ms"] = (time.time() - t0) * 1000

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Averaged {1 + len(hyp_docs)} embeddings into single HyDE vector"
            )

            # ── Step 4: Search with averaged embedding ─────────────────────
            step_counter += 1
            t0 = time.time()
            thinking_steps.append({
                "step_id": f"search_{step_counter}",
                "step_type": "retrieve",
                "title": "HyDE Vector Search",
                "status": "running",
                "summary": f"Searching with HyDE embedding for top-{top_k}...",
                "timestamp_ms": (t0 - total_start) * 1000,
            })

            search_results = await self.vector_store.query(
                dense_embedding=avg_dense,
                sparse_embedding=avg_sparse,
                top_k=top_k,
                filter=filter,
            )
            step_counts["search_count"] = 1
            latency_breakdown["search_ms"] = (time.time() - t0) * 1000

            chunks = [
                ChunkResult(
                    id=r.get("id", ""),
                    content=r.get("chunk", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in search_results
            ]

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Found {len(chunks)} documents"

            # ── Step 5: Optional rerank against ORIGINAL query ─────────────
            if self.use_reranker and self.reranker and chunks:
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"rerank_{step_counter}",
                    "step_type": "rerank",
                    "title": "Reranking Against Original Query",
                    "status": "running",
                    "summary": f"Reranking {len(chunks)} to top-{self.rerank_top_k}...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                })

                chunks = await self._rerank_chunks(query, chunks, self.rerank_top_k)
                step_counts["rerank_count"] = 1
                latency_breakdown["rerank_ms"] = (time.time() - t0) * 1000

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Reranked to {len(chunks)} documents"
            else:
                chunks = chunks[:self.rerank_top_k]

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000

            return RetrievalResult(
                chunks=chunks,
                query=query,
                iterations=1,
                metadata={
                    "provider": self.get_name(),
                    "num_hypothetical": len(hyp_docs),
                    "hypothetical_docs": [d[:200] for d in hyp_docs],
                    "embedding_method": "averaged (query + N hypothetical docs)",
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"HyDE retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "hyde_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.generator:
            await self.generator.aclose()
