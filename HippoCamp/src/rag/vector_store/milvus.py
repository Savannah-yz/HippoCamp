"""
Milvus Vector Store implementation.
Provides async operations for storing and retrieving vectors using Milvus.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from pymilvus import (
        AnnSearchRequest,
        AsyncMilvusClient,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
        RRFRanker,
    )
except ImportError:
    raise ImportError(
        "pymilvus is not installed. Please install it using 'pip install pymilvus'."
    )

from src.rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus vector store implementation.
    
    Features:
    - Async operations using AsyncMilvusClient
    - Hybrid search support (dense + sparse with RRF ranker)
    - Batch upsert with retry logic
    - Collection auto-creation if not exists
    - Filtering support (by metadata fields)
    - Score normalization and thresholding
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "research_rag_collection",
        dimension: int = 1536,
        top_k: int = 10,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        batch_size: Optional[int] = None,
        similarity_threshold: float = 0.0,
        metric_type: str = "COSINE",
        partition_name: Optional[str] = None,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            endpoint: Milvus endpoint URL (defaults to VECTOR_DB_ENDPOINT env var)
            api_key: Milvus API key (defaults to VECTOR_DB_API_KEY env var)
            collection_name: Name of the collection to use/create
            dimension: Dimension of dense embeddings (default: 1536 for Cohere embed-v4.0)
            top_k: Default number of results for queries (final results after merge)
            top_k_dense: Number of dense results to retrieve (defaults to top_k * 10 for rerank prep)
            top_k_sparse: Number of sparse results to retrieve (defaults to top_k * 10 for rerank prep)
            batch_size: Batch size for upsert operations
            similarity_threshold: Minimum similarity score (0.0 = no threshold)
            metric_type: Distance metric ("COSINE", "IP", "L2")
            partition_name: Optional partition name for experiment isolation
            index_params: Optional custom index parameters (M, efConstruction for HNSW)
            search_params: Optional custom search parameters (ef for HNSW search)
        """
        # Get credentials from environment if not provided
        endpoint = endpoint or os.getenv("VECTOR_DB_ENDPOINT")
        api_key = api_key or os.getenv("VECTOR_DB_API_KEY")
        
        if endpoint is None:
            raise ValueError("Milvus endpoint not provided. Set VECTOR_DB_ENDPOINT env var or pass endpoint.")
        if api_key is None:
            raise ValueError("Milvus API key not provided. Set VECTOR_DB_API_KEY env var or pass api_key.")
        
        self.endpoint = endpoint
        self.api_key = api_key
        self.collection_name = collection_name
        self.dimension = dimension
        self.top_k = top_k
        # For rerank preparation: retrieve more candidates from each source
        self.top_k_dense = top_k_dense or (top_k * 10 if top_k < 100 else 100)
        self.top_k_sparse = top_k_sparse or (top_k * 10 if top_k < 100 else 100)
        if batch_size is None:
            raise ValueError("MilvusVectorStore requires batch_size.")
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.metric_type = metric_type
        self.partition_name = partition_name
        
        # Optimized index parameters for HNSW
        # M=16: Good balance between accuracy and memory (8-64 range)
        # efConstruction=200: Higher quality index (100-500 range, higher = better quality but slower build)
        self.index_params = index_params or {
            "metric_type": metric_type,
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,  # Higher quality index
            },
        }

        # Search parameters for HNSW
        # ef: search-time width (higher = better recall, slower search)
        # If not provided, will be calculated dynamically based on top_k
        self.search_params = search_params or {}

        # Initialize clients
        self.client = MilvusClient(uri=endpoint, token=api_key)
        self.async_client = AsyncMilvusClient(uri=endpoint, token=api_key)
        
        # Collection will be created on first use (lazy initialization)
        self._collection_initialized = False
        self._has_partition_field = None  # Will be set when collection is checked/created
        
        logger.info(
            f"Initialized MilvusVectorStore: collection={collection_name}, "
            f"dimension={dimension}, metric={metric_type}, "
            f"partition={partition_name or 'default'}, "
            f"top_k_dense={self.top_k_dense}, top_k_sparse={self.top_k_sparse}"
        )
    
    async def _ensure_collection(self):
        """Ensure collection exists, create if not."""
        if self._collection_initialized:
            return
        
        try:
            # Check if collection exists
            if self.collection_name in self.client.list_collections():
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # Check if collection has partition field
                # If collection exists but doesn't have partition field, we can't add it
                # So we'll skip partition in upsert if collection doesn't support it
                try:
                    collection_info = self.client.describe_collection(self.collection_name)
                    # Check if partition field exists in schema
                    # describe_collection returns fields in different formats depending on Milvus version
                    fields = collection_info.get("fields", [])
                    
                    # Try to find partition field - handle different field formats
                    has_partition = False
                    for field in fields:
                        # Field can be dict with "name" key, or have "field_name" key, or be a string
                        field_name = None
                        if isinstance(field, dict):
                            field_name = field.get("name") or field.get("field_name")
                        elif isinstance(field, str):
                            field_name = field
                        
                        if field_name == "partition":
                            has_partition = True
                            break
                    
                    if self.partition_name and not has_partition:
                        logger.warning(
                            f"Collection '{self.collection_name}' exists but doesn't have partition field. "
                            f"Partition '{self.partition_name}' will be ignored. "
                            f"Consider creating a new collection with partition support."
                        )
                        self._has_partition_field = False
                    else:
                        self._has_partition_field = has_partition
                        if self.partition_name:
                            logger.info(
                                f"Collection '{self.collection_name}' {'has' if has_partition else 'does not have'} "
                                f"partition field. Partition filtering: {has_partition}"
                            )
                except Exception as e:
                    logger.warning(f"Could not check collection schema: {e}. Assuming no partition field.")
                    logger.debug(f"Collection info structure: {collection_info if 'collection_info' in locals() else 'N/A'}")
                    self._has_partition_field = False
                
                # Load collection to ensure it's ready
                await self.async_client.load_collection(collection_name=self.collection_name)
                self._collection_initialized = True
                return
            
            # Collection doesn't exist, mark that we'll create it with partition if needed
            self._has_partition_field = self.partition_name is not None
            
            # Create collection schema
            schema = self._create_schema()
            
            # Create collection using async client
            await self.async_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
            )
            
            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            
            # Add index for dense vector (HNSW with COSINE for normalized embeddings)
            index_params.add_index(
                field_name="vector",
                index_type=self.index_params.get("index_type", "HNSW"),
                metric_type=self.index_params.get("metric_type", self.metric_type),
                params=self.index_params.get("params", {}),
            )
            
            # Add index for sparse vector (SPARSE_INVERTED_INDEX with IP)
            # IP (Inner Product) is standard for sparse embeddings
            try:
                index_params.add_index(
                    field_name="sparse_vector",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",  # IP is optimal for sparse embeddings
                )
            except Exception as e:
                logger.warning(f"Could not add sparse vector index: {e}")
            
            # Create indexes
            await self.async_client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            
            # Load collection
            await self.async_client.load_collection(collection_name=self.collection_name)
            
            logger.info(f"Created collection '{self.collection_name}' with indexes")
            # Set partition field flag based on whether we created it with partition
            self._has_partition_field = self.partition_name is not None
            self._collection_initialized = True
            
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def _create_schema(self) -> CollectionSchema:
        """Create collection schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        # Add partition key if partition_name is provided
        if self.partition_name:
            fields.insert(0, FieldSchema(
                name="partition",
                dtype=DataType.VARCHAR,
                max_length=256,
                is_partition_key=True,
            ))
        
        return CollectionSchema(fields=fields, description="Research RAG collection")
    
    def _normalize_sparse(self, sparse: Dict[str, float]) -> List[Tuple[int, float]]:
        """
        Normalize sparse embedding from dict to list of tuples.
        
        Args:
            sparse: Dictionary mapping token (str) -> weight (float)
            
        Returns:
            List of (token_id, weight) tuples
        """
        if not sparse:
            return []
        # Convert string tokens to integers (hash-based)
        # Note: This is a simplified approach. In production, you might want
        # a proper tokenizer/vocabulary mapping.
        return [(hash(k) % (2**31), float(v)) for k, v in sparse.items()]
    
    async def upsert(
        self,
        ids: List[str],
        dense_embeddings: np.ndarray,
        sparse_embeddings: Optional[List[Dict[str, float]]] = None,
        chunks: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Store vectors with metadata.
        
        Args:
            ids: List of unique identifiers for each chunk
            dense_embeddings: numpy array of shape [N, dimension]
            sparse_embeddings: Optional list of sparse embeddings
            chunks: Optional list of original chunk texts
            metadatas: Optional list of metadata dictionaries
        """
        # Ensure collection exists
        await self._ensure_collection()
        
        # Validate inputs
        n = len(ids)
        if dense_embeddings.shape[0] != n:
            raise ValueError(
                f"Length mismatch: ids={n}, dense_embeddings={dense_embeddings.shape[0]}"
            )
        
        # Default values
        if chunks is None:
            chunks = [""] * n
        if metadatas is None:
            metadatas = [{}] * n
        if sparse_embeddings is None:
            sparse_embeddings = [{}] * n
        
        # Ensure all lists have same length
        assert len(chunks) == n, f"chunks length mismatch: {len(chunks)} != {n}"
        assert len(metadatas) == n, f"metadatas length mismatch: {len(metadatas)} != {n}"
        assert len(sparse_embeddings) == n, f"sparse_embeddings length mismatch: {len(sparse_embeddings)} != {n}"
        
        # Prepare data for Milvus
        data = []
        for i in range(n):
                # Normalize sparse embedding
                sparse_vec = self._normalize_sparse(sparse_embeddings[i])
                
                record = {
                    "id": ids[i],
                    "vector": dense_embeddings[i].tolist(),
                    "sparse_vector": sparse_vec,
                    "chunk": chunks[i] if chunks[i] else "",
                    "metadata": metadatas[i],
                }
                
                # Add partition if specified AND collection supports it
                if self.partition_name and self._has_partition_field:
                    record["partition"] = self.partition_name
                
                data.append(record)
        
        # Batch upsert with retry
        await self._batch_upsert(data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _batch_upsert(self, data: List[Dict[str, Any]]) -> None:
        """
        Batch upsert data to Milvus with retry logic.
        
        Args:
            data: List of data dictionaries to upsert
        """
        if not data:
            logger.info("No data to upsert")
            return
        
        # Process in batches
        batches = [
            data[i : i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]
        
        logger.info(f"Upserting {len(data)} vectors in {len(batches)} batches")
        
        with tqdm(total=len(data), desc="Upserting to Milvus") as pbar:
            for batch in batches:
                try:
                    await self.async_client.upsert(
                        collection_name=self.collection_name,
                        data=batch,
                    )
                    pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
                    raise
    
    async def query(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        hybrid_search: bool = False,
        filter: Optional[str] = None,
        prepare_for_rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            dense_embedding: Query dense embedding vector [dimension]
            sparse_embedding: Optional query sparse embedding
            top_k: Number of results to return (defaults to self.top_k)
            hybrid_search: Whether to use hybrid search
            filter: Optional metadata filter expression
            prepare_for_rerank: If True, retrieve top_k_dense + top_k_sparse and merge
            
        Returns:
            List of result dictionaries with id, chunk, metadata, score
        """
        # Ensure collection exists
        await self._ensure_collection()
        
        if top_k is None:
            top_k = self.top_k
        
        # Build filter with partition if specified AND collection supports it
        partition_filter = None
        if self.partition_name and self._has_partition_field:
            partition_filter = f'partition == "{self.partition_name}"'
            if filter:
                filter = f"{partition_filter} AND {filter}"
            else:
                filter = partition_filter
        
        if hybrid_search and sparse_embedding is None:
            logger.warning("hybrid_search=True but sparse_embedding is None. Using dense-only search.")
            hybrid_search = False
        
        try:
            # Get ef from search_params if configured, otherwise calculate dynamically
            # ef: typically top_k * 2 to top_k * 10, higher = better recall but slower search
            if "ef" in self.search_params:
                ef = self.search_params["ef"]
            else:
                ef = max(100, top_k * 5) if not prepare_for_rerank else max(100, max(self.top_k_dense, self.top_k_sparse) * 5)
            
            if prepare_for_rerank and sparse_embedding:
                # For rerank preparation: retrieve top_k_dense from dense and top_k_sparse from sparse
                # Then merge and deduplicate results (return ALL merged results, not limited to top_k)
                dense_results = await self._dense_search(
                    dense_embedding, self.top_k_dense, filter, ef
                )
                sparse_results = await self._sparse_search(
                    sparse_embedding, self.top_k_sparse, filter
                )
                
                # Merge and deduplicate results - return ALL candidates for rerank
                # top_k is only used as a backup/fallback, not a hard limit
                merged_results = self._merge_and_deduplicate(
                    dense_results, sparse_results, limit_to_top_k=False
                )
                logger.info(
                    f"Rerank preparation: retrieved {len(merged_results)} candidates "
                    f"(from {self.top_k_dense} dense + {self.top_k_sparse} sparse)"
                )
                return merged_results
                
            elif hybrid_search and sparse_embedding:
                # Hybrid search with RRF ranker
                dense_req = AnnSearchRequest(
                    data=[dense_embedding.tolist()],
                    anns_field="vector",
                    param={
                        "metric_type": self.metric_type,  # COSINE for normalized embeddings
                        "params": {"ef": ef},
                    },
                    limit=top_k,
                    expr=filter,
                )
                
                sparse_req = AnnSearchRequest(
                    data=[self._normalize_sparse(sparse_embedding)],
                    anns_field="sparse_vector",
                    param={
                        "metric_type": "IP",  # IP is optimal for sparse embeddings
                        "params": {},
                    },
                    limit=top_k,
                    expr=filter,
                )
                
                ranker = RRFRanker()
                
                results = await self.async_client.hybrid_search(
                    collection_name=self.collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=ranker,
                    limit=top_k,
                    output_fields=["id", "chunk", "metadata"],
                    timeout=30,
                )
            else:
                # Dense-only search
                results = await self._dense_search(dense_embedding, top_k, filter, ef)
            
            # Process results
            return self._process_search_results(results, top_k)
            
        except Exception as e:
            logger.error(f"Error in Milvus search: {e}")
            raise
    
    async def _dense_search(
        self,
        dense_embedding: np.ndarray,
        limit: int,
        filter: Optional[str],
        ef: int,
    ) -> List:
        """Perform dense-only search."""
        search_params = {
            "metric_type": self.metric_type,  # COSINE for normalized embeddings
            "params": {"ef": ef},
        }
        
        return await self.async_client.search(
            collection_name=self.collection_name,
            data=[dense_embedding.tolist()],
            limit=limit,
            output_fields=["id", "chunk", "metadata"],
            filter=filter,
            search_params=search_params,
            anns_field="vector",
            timeout=30,
        )
    
    async def _sparse_search(
        self,
        sparse_embedding: Dict[str, float],
        limit: int,
        filter: Optional[str],
    ) -> List:
        """
        Perform sparse-only search.
        
        Note: Milvus requires hybrid_search to have both dense and sparse requests.
        We use a dummy dense request with zero vector and rely on sparse results.
        """
        sparse_req = AnnSearchRequest(
            data=[self._normalize_sparse(sparse_embedding)],
            anns_field="sparse_vector",
            param={
                "metric_type": "IP",  # IP is optimal for sparse embeddings
                "params": {},
            },
            limit=limit,
            expr=filter,
        )
        
        # Milvus hybrid_search requires both dense and sparse requests
        # Create a dummy dense request that will be ignored (we only care about sparse results)
        dummy_dense_req = AnnSearchRequest(
            data=[[0.0] * self.dimension],
            anns_field="vector",
            param={"metric_type": self.metric_type, "params": {"ef": 1}},
            limit=1,  # Minimal limit since we don't use dense results
            expr=filter,
        )
        
        # Use RRFRanker but sparse results will dominate
        ranker = RRFRanker()
        results = await self.async_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dummy_dense_req, sparse_req],
            ranker=ranker,
            limit=limit,
            output_fields=["id", "chunk", "metadata"],
            timeout=30,
        )
        
        return results
    
    def _merge_and_deduplicate(
        self,
        dense_results: List,
        sparse_results: List,
        top_k: Optional[int] = None,
        limit_to_top_k: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse results, deduplicate by ID.
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of final results to return (only used if limit_to_top_k=True)
            limit_to_top_k: If True, limit results to top_k; if False, return all merged results
            
        Returns:
            Merged and deduplicated results (all results if limit_to_top_k=False, top_k if True)
        """
        # Extract hits from both result sets
        dense_hits = dense_results[0] if dense_results else []
        sparse_hits = sparse_results[0] if sparse_results else []
        
        # Create a dictionary to store best score for each ID
        id_to_result = {}
        
        # Process dense results
        for i, hit in enumerate(dense_hits):
            hit_id = hit.get("id") or hit.get("entity", {}).get("id", "")
            if not hit_id:
                continue

            distance = hit.get("distance", 0.0)
            score = self._calculate_score(distance)

            if hit_id not in id_to_result or score > id_to_result[hit_id]["score"]:
                chunk = hit.get("chunk") or hit.get("entity", {}).get("chunk", "")
                metadata = hit.get("metadata") or hit.get("entity", {}).get("metadata", {})

                # Debug: Print first 3 chunks from Milvus
                if i < 3:
                    logger.info(f"\n{'='*80}\nChunk {i + 1} from Milvus (dense search):\n{'='*80}")
                    logger.info(f"Collection: {self.collection_name}")
                    logger.info(f"Hit ID: {hit_id}")
                    logger.info(f"Distance: {distance:.4f}, Score: {score:.4f}")
                    logger.info(f"Raw hit structure: {hit}")
                    logger.info(f"Extracted metadata: {metadata}")
                    logger.info(f"Chunk preview: {chunk[:200]}...")
                    logger.info(f"{'='*80}\n")

                id_to_result[hit_id] = {
                    "id": hit_id,
                    "chunk": chunk,
                    "metadata": metadata,
                    "score": score,
                    "source": "dense",
                }
        
        # Process sparse results
        for hit in sparse_hits:
            hit_id = hit.get("id") or hit.get("entity", {}).get("id", "")
            if not hit_id:
                continue
            
            distance = hit.get("distance", 0.0)
            # For sparse IP, higher is better, so use distance directly as score
            score = float(distance)
            
            if hit_id not in id_to_result or score > id_to_result[hit_id]["score"]:
                chunk = hit.get("chunk") or hit.get("entity", {}).get("chunk", "")
                metadata = hit.get("metadata") or hit.get("entity", {}).get("metadata", {})
                id_to_result[hit_id] = {
                    "id": hit_id,
                    "chunk": chunk,
                    "metadata": metadata,
                    "score": score,
                    "source": "sparse",
                }
            elif hit_id in id_to_result:
                # Mark as hybrid if appears in both
                id_to_result[hit_id]["source"] = "hybrid"
        
        # Sort by score
        sorted_results = sorted(id_to_result.values(), key=lambda x: x["score"], reverse=True)
        
        # Apply threshold
        filtered_results = [
            r for r in sorted_results if r["score"] >= self.similarity_threshold
        ]
        
        # Return all results if limit_to_top_k=False (for rerank preparation)
        # Otherwise limit to top_k
        if limit_to_top_k and top_k is not None:
            return filtered_results[:top_k]
        else:
            return filtered_results
    
    def _calculate_score(self, distance: float) -> float:
        """Calculate similarity score from distance based on metric type."""
        if self.metric_type == "COSINE":
            # Cosine distance: 1 - cosine_similarity
            # So similarity = 1 - distance
            return 1.0 - distance
        elif self.metric_type == "IP":
            # Inner Product: higher is better, but not normalized
            return float(distance)
        elif self.metric_type == "L2":
            # L2 distance: lower is better
            # Convert to similarity: 1 / (1 + distance)
            return 1.0 / (1.0 + distance)
        else:
            return float(distance)
    
    def _process_search_results(
        self,
        results: List,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Process raw Milvus search results into standardized format.
        
        Args:
            results: Raw results from Milvus search
            top_k: Maximum number of results
            
        Returns:
            List of standardized result dictionaries
        """
        processed = []
        
        # Results is a list of hits (one per query vector)
        # Since we only query with one vector, results[0] contains our hits
        hits = results[0] if results else []
        
        for i, hit in enumerate(hits[:top_k]):
            # Calculate similarity score from distance
            distance = hit.get("distance", 0.0)
            score = self._calculate_score(distance)

            # Apply threshold
            if score < self.similarity_threshold:
                continue

            # Extract fields - Milvus returns fields directly in the hit
            # Try both formats: direct fields and entity wrapper
            chunk = hit.get("chunk") or hit.get("entity", {}).get("chunk", "")
            metadata = hit.get("metadata") or hit.get("entity", {}).get("metadata", {})
            hit_id = hit.get("id") or hit.get("entity", {}).get("id", "")

            processed.append({
                "id": hit_id,
                "chunk": chunk,
                "metadata": metadata,
                "score": score,
            })
        
        return processed
    
    async def filter_existing(
        self,
        ids: List[str],
    ) -> List[str]:
        """
        Check which IDs already exist in the store.
        
        Args:
            ids: List of IDs to check
            
        Returns:
            List of IDs that already exist
        """
        if not ids:
            return []
        
        # Ensure collection exists
        await self._ensure_collection()
        
        try:
            # Build filter with partition if specified AND collection supports it
            filter_parts = [f"id in {ids}"]
            if self.partition_name and self._has_partition_field:
                filter_parts.insert(0, f'partition == "{self.partition_name}"')
            filter_expr = " AND ".join(filter_parts)
            
            results = await self.async_client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["id"],
                timeout=30,
            )
            
            existing_ids = [hit.get("id") for hit in results if "id" in hit]
            return existing_ids
            
        except Exception as e:
            logger.error(f"Error filtering existing IDs: {e}")
            # If query fails, return empty list (assume none exist)
            return []
    
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[str] = None,
    ) -> int:
        """
        Delete vectors from the store.
        
        Args:
            ids: Optional list of IDs to delete
            filter: Optional metadata filter expression
            
        Returns:
            Number of deleted vectors
        """
        if ids is None and filter is None:
            raise ValueError("Either 'ids' or 'filter' must be provided")
        
        # Ensure collection exists
        await self._ensure_collection()
        
        try:
            # Build filter with partition if specified AND collection supports it
            filter_parts = []
            if self.partition_name and self._has_partition_field:
                filter_parts.append(f'partition == "{self.partition_name}"')
            
            if ids is not None:
                filter_parts.append(f"id in {ids}")
            elif filter:
                filter_parts.append(filter)
            
            if not filter_parts:
                raise ValueError("Either 'ids' or 'filter' must be provided")
            
            filter_expr = " AND ".join(filter_parts)
            
            logger.debug(f"Deleting with filter: {filter_expr}")
            result = await self.async_client.delete(
                collection_name=self.collection_name,
                filter=filter_expr,
                timeout=30,
            )
            
            deleted_count = result.get("delete_count", 0)
            logger.info(
                f"Deleted {deleted_count} vectors from collection '{self.collection_name}' "
                f"(filter: {filter_expr})"
            )
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    async def close(self):
        """Close the client connections."""
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()
