"""
Local Milvus Vector Store - file-based, no cloud/API needed.
Simplified version for research and testing.
"""

import logging
from typing import List, Dict, Any, Optional, Sequence
import numpy as np

try:
    from pymilvus import MilvusClient
except ImportError:
    raise ImportError("pymilvus not installed. Run: pip install pymilvus")

from src.rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusLocalVectorStore(BaseVectorStore):
    """
    Local file-based Milvus vector store.
    No cloud endpoint needed - stores data in a local file.

    Supports:
    - Dense embeddings (required)
    - Sparse embeddings (optional, for hybrid search)
    - Metadata storage
    """

    def __init__(
        self,
        uri: str = "./data/vector_store/rag.milvus.db",
        collection_name: str = "research_rag",
        dimension: int = 1024,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        index_params: Dict[str, Any] = None,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type
        # HNSW params: M=16 (connections), efConstruction=200 (build quality)
        self.index_params = index_params or {"M": 16, "efConstruction": 200}

        # Initialize client
        self.client = MilvusClient(uri=uri)
        self._ensure_collection()

        logger.info(
            f"MilvusLocalVectorStore: uri={uri}, collection={collection_name}, "
            f"dim={dimension}, index={index_type}"
        )

    def _ensure_collection(self):
        """Create collection with HNSW index if not exists."""
        if not self.client.has_collection(self.collection_name):
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type=self.metric_type,
                primary_field_name="id",
                vector_field_name="vector",
                id_type="str",
                max_length=256,
                auto_id=False,
                enable_dynamic_field=True,
                consistency_level="Strong",
            )
            logger.info(f"Created collection: {self.collection_name}")

            # Create HNSW index for better search performance
            try:
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type=self.index_type,
                    metric_type=self.metric_type,
                    params=self.index_params,
                )
                self.client.create_index(
                    collection_name=self.collection_name,
                    index_params=index_params,
                )
                logger.info(f"Created {self.index_type} index with params: {self.index_params}")
            except Exception as e:
                logger.warning(f"Could not create HNSW index (using default): {e}")

        self.client.load_collection(self.collection_name)

    async def upsert(
        self,
        ids: List[str],
        dense_embeddings: np.ndarray,
        sparse_embeddings: Optional[List[Dict[str, float]]] = None,
        chunks: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert vectors to collection."""
        n = len(ids)
        chunks = chunks or [""] * n
        metadatas = metadatas or [{}] * n

        data = []
        for i in range(n):
            record = {
                "id": ids[i],
                "vector": dense_embeddings[i].tolist(),
                "chunk": chunks[i],
                "metadata": metadatas[i],
            }
            # Store sparse as part of metadata if provided
            if sparse_embeddings and sparse_embeddings[i]:
                record["metadata"]["sparse"] = sparse_embeddings[i]
            data.append(record)

        self.client.upsert(collection_name=self.collection_name, data=data)
        logger.info(f"Upserted {n} vectors to {self.collection_name}")

    async def query(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        filter: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[dense_embedding.tolist()],
            limit=top_k,
            filter=filter,
            output_fields=["chunk", "metadata"],
        )

        hits = []
        for result in results[0]:
            score = self._normalize_score(result.get("distance", 0.0))
            hits.append({
                "id": result["id"],
                "chunk": result.get("chunk", ""),
                "metadata": result.get("metadata", {}),
                "score": score,
            })
        return hits

    def _normalize_score(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self.metric_type == "COSINE":
            return 1.0 - distance
        elif self.metric_type == "L2":
            return 1.0 / (1.0 + distance)
        return distance

    async def filter_existing(self, ids: List[str]) -> List[str]:
        """Check which IDs already exist in the store."""
        if not ids:
            return []
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"id in {ids}",
                output_fields=["id"],
            )
            return [r["id"] for r in results if "id" in r]
        except Exception as e:
            logger.warning(f"Error checking existing IDs: {e}")
            return []

    async def delete(self, ids: List[str] = None, filter: str = None) -> int:
        """Delete vectors by IDs or filter."""
        if ids:
            self.client.delete(collection_name=self.collection_name, ids=ids)
            return len(ids)
        return 0

    def drop_collection(self):
        """Drop the collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")

    async def close(self):
        """Close client."""
        # MilvusClient doesn't require explicit closing for local file-based storage
        pass
    
    async def aclose(self):
        """Alias for close() to match async context manager pattern."""
        await self.close()
