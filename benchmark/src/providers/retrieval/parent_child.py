"""
Parent-Child Chunk Expansion — post-processing step that expands
individual chunks to full-page context.

After retrieval + reranking produces top-N chunks, this module:
1. Extracts unique (file_id, page_number) pairs from the results
2. Queries Qdrant for ALL chunks on each of those pages
3. Merges chunks per page, ordered by chunk_index
4. Returns the top-M pages ranked by the best chunk score on that page

Usage:
    expander = ParentChildExpander(
        vector_store=qdrant_vector_store,
        max_pages=10,
    )
    page_chunks = await expander.expand(reranked_chunks)
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import models

from ..base import ChunkResult

logger = logging.getLogger(__name__)


class ParentChildExpander:
    """
    Expands chunk-level retrieval results to page-level "parent" chunks.

    After reranking produces top-k small chunks, this expander:
    - Groups chunks by (file_id, page_number)
    - Fetches all sibling chunks on the same page from Qdrant
    - Merges them into full-page text
    - Returns top max_pages results ranked by best chunk score

    Args:
        vector_store: QdrantVectorStore instance
        max_pages: Maximum number of pages to return (default: 10)
    """

    def __init__(
        self,
        vector_store,
        max_pages: int = 10,
    ):
        self.vector_store = vector_store
        self.max_pages = max_pages

    def _extract_page_key(self, chunk: ChunkResult) -> Optional[Tuple[str, int]]:
        """Extract (file_id, page_number) from a ChunkResult's metadata.

        Handles the normalized metadata format (always nested after
        _extract_chunk_and_metadata):
            metadata.file_info.file_id
            metadata.segment_info.page_numbers[0]
        """
        meta = chunk.metadata
        file_info = meta.get("file_info", {})
        segment_info = meta.get("segment_info", {})

        file_id = file_info.get("file_id")
        page_numbers = segment_info.get("page_numbers", [])

        if file_id is None or not page_numbers:
            return None

        # Use the first page number (most chunks map to a single page)
        return (str(file_id), int(page_numbers[0]))

    async def _fetch_page_chunks(
        self,
        file_id: str,
        page_number: int,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for all chunks on a specific (file_id, page) pair.

        Uses scroll (filter-only, no vector search) for efficiency.
        Handles both flat (synvo-local) and nested payload formats via
        the vector store's scroll_by_filter method.
        """
        # Build filter — use flat keys (synvo-local format: file_id, page_num)
        # and also try nested keys for compatibility
        # The qdrant MatchValue on a non-existent key simply won't match,
        # so we use MatchAny-style with two separate filter attempts
        # For simplicity, try flat format first (financebench uses synvo-local)
        must_conditions = [
            models.FieldCondition(
                key="file_id",
                match=models.MatchValue(value=file_id),
            ),
            models.FieldCondition(
                key="page_num",
                match=models.MatchValue(value=page_number),
            ),
        ]

        # Add partition filter if configured
        if self.vector_store.partition_name:
            must_conditions.append(
                models.FieldCondition(
                    key="partition",
                    match=models.MatchValue(value=self.vector_store.partition_name),
                )
            )

        scroll_filter = models.Filter(must=must_conditions)

        results = await self.vector_store.scroll_by_filter(
            scroll_filter=scroll_filter,
            limit=100,  # Safety limit — a single page rarely has >50 chunks
        )

        # If flat format returned nothing, try nested format
        if not results:
            must_conditions_nested = [
                models.FieldCondition(
                    key="metadata.file_info.file_id",
                    match=models.MatchValue(value=file_id),
                ),
                models.FieldCondition(
                    key="metadata.segment_info.page_numbers",
                    match=models.MatchValue(value=page_number),
                ),
            ]
            if self.vector_store.partition_name:
                must_conditions_nested.append(
                    models.FieldCondition(
                        key="partition",
                        match=models.MatchValue(value=self.vector_store.partition_name),
                    )
                )
            scroll_filter_nested = models.Filter(must=must_conditions_nested)
            results = await self.vector_store.scroll_by_filter(
                scroll_filter=scroll_filter_nested,
                limit=100,
            )

        # Sort by chunk_index to maintain document order
        results.sort(key=lambda x: x.get("chunk_index", 0))
        return results

    async def expand(
        self,
        chunks: List[ChunkResult],
    ) -> List[ChunkResult]:
        """
        Expand chunk-level results to page-level results.

        Args:
            chunks: Reranked chunk results from retrieval provider

        Returns:
            List of ChunkResult where each result is a full page of content,
            limited to self.max_pages, ordered by best chunk score descending.
        """
        if not chunks:
            return []

        start_time = time.time()

        # Step 1: Collect unique (file_id, page_number) pairs and track best scores
        page_best_score: Dict[Tuple[str, int], float] = {}
        page_source_ids: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        seen_pages: set = set()
        ordered_pages: List[Tuple[str, int]] = []

        for chunk in chunks:
            key = self._extract_page_key(chunk)
            if key is None:
                continue

            page_source_ids[key].append(chunk.id)
            if key not in page_best_score or chunk.score > page_best_score[key]:
                page_best_score[key] = chunk.score
            if key not in seen_pages:
                seen_pages.add(key)
                ordered_pages.append(key)

        if not ordered_pages:
            logger.warning("ParentChild: No page keys found, returning original chunks")
            return chunks

        # Step 2: Rank pages by best chunk score, take top max_pages
        ranked_pages = sorted(
            ordered_pages,
            key=lambda k: page_best_score[k],
            reverse=True,
        )[: self.max_pages]

        logger.info(
            "ParentChild: Expanding %d chunks -> %d unique pages -> top %d pages",
            len(chunks),
            len(ordered_pages),
            len(ranked_pages),
        )

        # Step 3: Fetch sibling chunks for each page concurrently
        fetch_tasks = [
            self._fetch_page_chunks(file_id, page_num)
            for file_id, page_num in ranked_pages
        ]
        page_chunks_list = await asyncio.gather(*fetch_tasks)

        # Step 4: Merge chunks per page into full-page content
        expanded_results: List[ChunkResult] = []

        for (file_id, page_num), page_chunks in zip(ranked_pages, page_chunks_list):
            if not page_chunks:
                logger.warning(
                    "ParentChild: No chunks found for file_id=%s page=%d, skipping",
                    file_id,
                    page_num,
                )
                continue

            # Merge chunk texts (already sorted by chunk_index)
            merged_text = "\n".join(c["chunk"] for c in page_chunks if c.get("chunk"))

            # Use metadata from first chunk as representative
            representative_meta = page_chunks[0].get("metadata", {})

            page_result = ChunkResult(
                id=f"page_{file_id}_{page_num}",
                content=merged_text,
                score=page_best_score[(file_id, page_num)],
                metadata={
                    "file_info": representative_meta.get("file_info", {}),
                    "segment_info": {
                        "page_numbers": [page_num],
                        "segment_indices": [],
                        "time_ranges": [],
                    },
                    "chunk_meta": {
                        "type": "parent_page",
                        "chunk_index": 0,
                        "char_count": len(merged_text),
                        "chunks_merged": len(page_chunks),
                    },
                    "parent_child": {
                        "source_chunk_ids": page_source_ids[(file_id, page_num)],
                        "chunks_on_page": len(page_chunks),
                    },
                },
            )
            expanded_results.append(page_result)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "ParentChild: Expanded to %d pages in %.0fms",
            len(expanded_results),
            elapsed_ms,
        )

        return expanded_results
