"""
Hybrid RAG Provider — Dense vector search + BM25 lexical search with RRF fusion.

Pipeline:
  1. Extract keywords (LLM) ─┐
  2. Embed query (dense)      │  step 1 & 2 concurrent
  3. Dense search (Qdrant) ─┐ │
  4. BM25 search (SQLite)  ─┘ ┘  step 3 & 4 concurrent (wait on 1/2)
  5. RRF Fusion
  6. Rerank (optional)
  7. Return top-k results
"""

import asyncio
import json
import logging
import os
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
from src.rag.fusion import rrf_fuse
from src.rag.metadata.extractor import (
    build_company_map,
    extract_metadata_rules,
    extract_file_llm,
)
from src.rag.metadata.file_filter import (
    get_file_ids_by_metadata,
    get_file_ids_by_names,
    get_all_file_names,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Keyword extraction prompt — kept minimal for speed (~30 tokens out)
# ──────────────────────────────────────────────────────────────────────────────

_KEYWORD_SYSTEM = (
    "Extract search keywords from the user's question. "
    "Return ONLY a JSON array of 3-8 keywords/phrases that are most useful "
    "for finding relevant documents. Focus on entities, financial terms, "
    "company names, years, and specific metrics. "
    "Do NOT include generic words like 'what', 'how', 'please', etc.\n"
    'Example: ["3M", "capital expenditure", "FY2018", "cash flow statement"]'
)


class HybridRAGProvider(RetrievalProvider):
    """Hybrid RAG retrieval provider.

    Combines dense vector search (Qdrant) with BM25 lexical search
    (SQLite FTS5) using Reciprocal Rank Fusion, then optionally reranks.

    When a keyword_llm is provided, uses LLM to extract search keywords
    from the query before BM25 search (instead of sending the raw question).
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        reranker=None,
        fts_store=None,
        keyword_llm=None,
    ):
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.fts_store = fts_store
        self.keyword_llm = keyword_llm  # google.genai.Client or None

        params = config.params
        self.default_top_k = params.get("top_k", 20)
        self.top_k_dense = params.get("top_k_dense", 100)
        self.top_k_bm25 = params.get("top_k_bm25", 100)
        self.rrf_k = params.get("rrf_k", 60)
        self.use_reranker = params.get("use_reranker", True)
        self.rerank_top_k = params.get("rerank_top_k", 5)
        self.keyword_model = params.get("keyword_model", "gemini-2.5-flash")
        self.store_prerank = params.get("store_prerank", False)

        # Metadata-based file filtering
        # Values: "none" | "rule" | "rule_norerank" | "llm" | "llm_norerank"
        self.metadata_filter = params.get("metadata_filter", "none")
        self.sqlite_db_path = params.get("sqlite_db_path", "")

        # Lazy-init: company map and file names (built on first use)
        self._company_map: Optional[Dict[str, str]] = None
        self._all_file_names: Optional[List[str]] = None

    async def setup(self):
        if self._initialized:
            return

        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.fts_store is None:
            raise RetrievalError("FTS store (SQLite) is required", provider=self.get_name())

        logger.info(
            "HybridRAGProvider initialized "
            "(top_k_dense=%d, top_k_bm25=%d, rrf_k=%d, reranker=%s, keyword_llm=%s)",
            self.top_k_dense,
            self.top_k_bm25,
            self.rrf_k,
            self.reranker is not None,
            self.keyword_llm is not None,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Metadata extraction for file filtering
    # ------------------------------------------------------------------

    def _ensure_metadata_structures(self) -> None:
        """Lazy-build company map and file name list on first use."""
        if self._company_map is None and self.sqlite_db_path:
            self._company_map = build_company_map(self.sqlite_db_path)
        if self._all_file_names is None and self.sqlite_db_path:
            self._all_file_names = get_all_file_names(self.sqlite_db_path)

    async def _extract_file_filter(
        self,
        query: str,
    ) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        """Extract metadata and resolve to file_ids for filtering.

        Returns:
            (file_ids or None, metadata_info dict for diagnostics)
        """
        if self.metadata_filter == "none":
            return None, {}

        self._ensure_metadata_structures()
        meta_info: Dict[str, Any] = {"filter_mode": self.metadata_filter}

        if self.metadata_filter.startswith("rule"):
            meta = extract_metadata_rules(query, self._company_map or {})
            meta_info["extracted"] = meta
            file_ids = get_file_ids_by_metadata(
                self.sqlite_db_path,
                meta.get("company_code"),
                meta.get("years"),
                meta.get("quarter"),
            )
        elif self.metadata_filter.startswith("llm"):
            selected = await extract_file_llm(
                query,
                self._all_file_names or [],
                self.keyword_llm,
                self.keyword_model,
            )
            meta_info["llm_selected_files"] = selected
            if selected:
                file_ids = get_file_ids_by_names(self.sqlite_db_path, selected)
            else:
                file_ids = []
        else:
            return None, {}

        if not file_ids:
            logger.warning(
                "Metadata filter extracted no matching files — falling back to unfiltered"
            )
            meta_info["fallback"] = True
            return None, meta_info

        meta_info["filtered_file_ids"] = file_ids
        meta_info["filtered_file_count"] = len(file_ids)
        return file_ids, meta_info

    # ------------------------------------------------------------------
    # Keyword extraction via LLM
    # ------------------------------------------------------------------

    async def _extract_keywords(self, query: str) -> str:
        """Use LLM to extract search keywords from a natural-language query.

        Returns a space-separated keyword string suitable for FTS5 search.
        Falls back to the raw query if LLM call fails.
        """
        if not self.keyword_llm:
            logger.warning(
                "keyword_llm is None — skipping keyword extraction, using raw query"
            )
            return query

        try:
            from google.genai import types

            logger.info(
                "Calling keyword LLM (model=%s) for query: %.80s...",
                self.keyword_model,
                query,
            )

            response = await self.keyword_llm.aio.models.generate_content(
                model=self.keyword_model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=_KEYWORD_SYSTEM,
                    max_output_tokens=256,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text
            logger.info("Keyword LLM raw response: %r", raw_text)

            text = (raw_text or "").strip()
            # Parse JSON array
            keywords = json.loads(text)
            if isinstance(keywords, list) and keywords:
                kw_str = " ".join(str(k) for k in keywords)
                logger.info("Keywords extracted: %s", kw_str)
                return kw_str

            logger.warning("Keyword extraction returned non-list: %r", text)
            return query

        except Exception as e:
            logger.warning(
                "Keyword extraction failed (type=%s): %s — using raw query",
                type(e).__name__,
                e,
            )
            return query

    # ------------------------------------------------------------------
    # Rerank helper (same pattern as StandardRAGProvider)
    # ------------------------------------------------------------------

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[ChunkResult],
        top_k: int,
    ) -> List[ChunkResult]:
        if not self.reranker or not chunks:
            return chunks[:top_k]

        candidates = [
            {
                "id": c.id,
                "chunk": c.content,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        reranked = await self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
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

    # ------------------------------------------------------------------
    # Main retrieve
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        skip_rerank: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        if not self._initialized:
            await self.setup()

        top_k = top_k or self.default_top_k

        latency_breakdown: Dict[str, float] = {}
        thinking_steps: List[Dict[str, Any]] = []
        step_counts = {"search_count": 0, "rerank_count": 0}
        total_start = time.time()
        step_counter = 0

        try:
            # ── Step 1 & 2: Keyword extraction + Embed (concurrent) ───
            step_counter += 1
            prep_start = time.time()

            thinking_steps.append({
                "step_id": f"prep_{step_counter}",
                "step_type": "prepare",
                "title": "Preparing Query (Embed + Keywords)",
                "status": "running",
                "summary": "Embedding query and extracting BM25 keywords...",
                "timestamp_ms": 0,
            })

            query_for_embed = format_query_for_embedding(query)
            embed_task = self.embedder.embed([query_for_embed])
            keyword_task = self._extract_keywords(query)
            filter_task = self._extract_file_filter(query)

            query_embedding, bm25_query, (file_ids, filter_meta) = await asyncio.gather(
                embed_task, keyword_task, filter_task,
            )

            prep_latency = (time.time() - prep_start) * 1000
            latency_breakdown["embed_query_ms"] = prep_latency
            latency_breakdown["keyword_extract_ms"] = prep_latency  # combined wall-clock

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Prepared in {prep_latency:.0f}ms "
                f"(keywords: {bm25_query[:80]}{'...' if len(bm25_query) > 80 else ''})"
            )

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

            # ── Step 3 & 4: Dense + BM25 search (concurrent) ────────
            step_counter += 1
            search_start = time.time()
            thinking_steps.append({
                "step_id": f"hybrid_search_{step_counter}",
                "step_type": "retrieve",
                "title": "Hybrid Search (Dense + BM25)",
                "status": "running",
                "summary": (
                    f"Dense top-{self.top_k_dense} from Qdrant "
                    f"+ BM25 top-{self.top_k_bm25} from SQLite FTS5..."
                ),
                "timestamp_ms": (search_start - total_start) * 1000,
            })

            # Build Qdrant filter: combine user filter with metadata file filter
            qdrant_filter = filter
            if file_ids:
                file_filter_expr = f"file_id in {file_ids}"
                if qdrant_filter:
                    qdrant_filter = f"{qdrant_filter} AND {file_filter_expr}"
                else:
                    qdrant_filter = file_filter_expr

            dense_task = self.vector_store.query(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=self.top_k_dense,
                filter=qdrant_filter,
            )
            bm25_task = self.fts_store.search(
                query=bm25_query,
                top_k=self.top_k_bm25,
                file_ids=file_ids,
            )

            dense_results, bm25_results = await asyncio.gather(
                dense_task, bm25_task,
            )

            search_latency = (time.time() - search_start) * 1000
            latency_breakdown["dense_search_ms"] = search_latency
            latency_breakdown["bm25_search_ms"] = search_latency  # combined wall-clock
            step_counts["search_count"] = 2

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = (
                f"Dense: {len(dense_results)} results, "
                f"BM25: {len(bm25_results)} results"
            )
            thinking_steps[-1]["metadata"] = {
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "bm25_query": bm25_query,
            }

            logger.debug(
                "HybridRAG: dense=%d  bm25=%d",
                len(dense_results),
                len(bm25_results),
            )

            # ── Step 5: RRF Fusion ───────────────────────────────────
            step_counter += 1
            fusion_start = time.time()
            thinking_steps.append({
                "step_id": f"fusion_{step_counter}",
                "step_type": "fusion",
                "title": "RRF Fusion",
                "status": "running",
                "summary": "Merging dense and BM25 results with Reciprocal Rank Fusion...",
                "timestamp_ms": (fusion_start - total_start) * 1000,
            })

            fused = rrf_fuse(
                dense_results,
                bm25_results,
                k=self.rrf_k,
                limit=top_k,
            )

            fusion_latency = (time.time() - fusion_start) * 1000
            latency_breakdown["rrf_fusion_ms"] = fusion_latency

            thinking_steps[-1]["status"] = "complete"
            thinking_steps[-1]["summary"] = f"Fused to {len(fused)} results"

            retrieved_chunks = [
                ChunkResult(
                    id=r.get("id", ""),
                    content=r.get("chunk", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in fused
            ]

            if not retrieved_chunks:
                logger.warning("HybridRAG: No documents after fusion")
                latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
                step_counts["total_iterations"] = 1
                return RetrievalResult(
                    chunks=[],
                    query=query,
                    metadata={"provider": self.get_name(), "error": "no_documents"},
                    latency_breakdown=latency_breakdown,
                    thinking_steps=thinking_steps,
                    step_counts=step_counts,
                )

            # ── Step 6: Rerank (optional) ────────────────────────────
            final_chunks = retrieved_chunks
            reranked = False

            skip_rerank_meta = self.metadata_filter.endswith("_norerank")
            if self.use_reranker and self.reranker and not skip_rerank and not skip_rerank_meta:
                step_counter += 1
                rerank_start = time.time()
                thinking_steps.append({
                    "step_id": f"rerank_{step_counter}",
                    "step_type": "rerank",
                    "title": "Reranking Results",
                    "status": "running",
                    "summary": (
                        f"Reranking {len(retrieved_chunks)} documents "
                        f"to top-{self.rerank_top_k}..."
                    ),
                    "timestamp_ms": (rerank_start - total_start) * 1000,
                })

                final_chunks = await self._rerank_chunks(
                    query, retrieved_chunks, self.rerank_top_k,
                )
                reranked = True

                rerank_latency = (time.time() - rerank_start) * 1000
                latency_breakdown["rerank_ms"] = rerank_latency
                step_counts["rerank_count"] = 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Reranked to {len(final_chunks)} documents"
                thinking_steps[-1]["metadata"] = {"final_count": len(final_chunks)}
            else:
                final_chunks = retrieved_chunks[: self.rerank_top_k]

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000
            step_counts["total_iterations"] = 1

            result_metadata = {
                "provider": self.get_name(),
                "bm25_query": bm25_query,
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "fused_count": len(fused),
                "final_count": len(final_chunks),
                "reranked": reranked,
            }
            if filter_meta:
                result_metadata["metadata_filter"] = filter_meta

            # Optionally store pre-rerank candidates for diagnostic analysis
            if self.store_prerank:
                result_metadata["prerank_chunks"] = [
                    {
                        "id": c.id,
                        "content": c.content,
                        "score": c.score,
                        "metadata": c.metadata,
                    }
                    for c in retrieved_chunks
                ]

            return RetrievalResult(
                chunks=final_chunks,
                query=query,
                iterations=1,
                metadata=result_metadata,
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error("HybridRAG retrieval failed: %s", e)
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "hybrid_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.fts_store:
            await self.fts_store.aclose()
