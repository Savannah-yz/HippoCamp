"""
Qdrant Vector Store implementation.
Provides async operations for storing and retrieving vectors using Qdrant.
"""

import ast
import enum
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

import numpy as np

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError:
    raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

from src.rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


def _model_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model to dict, handling both v1 and v2 APIs."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {}


def _coerce_to_model(model_cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string values to enums based on a Pydantic model's field annotations.

    This handles cases where YAML config provides strings like "idf" that need
    to be converted to enum values like models.Modifier.IDF.
    """
    if not hasattr(model_cls, "model_fields"):
        return params

    result = dict(params)
    for field_name, field_info in model_cls.model_fields.items():
        if field_name not in result:
            continue

        value = result[field_name]
        if not isinstance(value, str):
            continue

        # Get the field's type annotation
        annotation = field_info.annotation
        if annotation is None:
            continue

        # Unwrap Optional[T] -> T
        origin = get_origin(annotation)
        if origin is type(None):
            continue
        if origin is not None:
            # Handle Union types (including Optional which is Union[T, None])
            args = get_args(annotation)
            # Find non-None type args that are enums
            enum_types = [arg for arg in args if isinstance(arg, type) and issubclass(arg, enum.Enum)]
            if enum_types:
                annotation = enum_types[0]
            else:
                continue

        # Check if annotation is an enum class
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            # Try multiple strategies to convert string to enum
            converted = False
            for attempt_value in (value, value.upper(), value.lower()):
                try:
                    result[field_name] = annotation(attempt_value)
                    converted = True
                    break
                except ValueError:
                    continue
            if not converted:
                # Try by name (e.g., Modifier.IDF accessed via Modifier["IDF"])
                try:
                    result[field_name] = annotation[value.upper()]
                    converted = True
                except KeyError:
                    pass
            if not converted:
                logger.warning(
                    "Cannot convert '%s' to %s for field '%s'; removing field.",
                    value, annotation.__name__, field_name
                )
                del result[field_name]

    return result


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store implementation.

    Supports:
    - Dense embeddings (required)
    - Sparse embeddings (optional, for hybrid search)
    - Metadata storage and filtering
    """

    _FILTER_EQ_RE = re.compile(r"^\s*([A-Za-z0-9_\.]+)\s*==\s*['\"](.+?)['\"]\s*$")
    _FILTER_IN_RE = re.compile(r"^\s*([A-Za-z0-9_\.]+)\s+in\s+(.+)$")
    _RRF_K = 60

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "research_rag_collection",
        dimension: int = 1536,
        top_k: int = 10,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        batch_size: Optional[int] = None,
        similarity_threshold: float = 0.0,
        metric_type: str = "COSINE",
        prefer_grpc: bool = False,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        enable_sparse: bool = True,
        sparse_vectors_config: Optional[Dict[str, Any]] = None,
        partition_name: Optional[str] = None,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        self.path = path
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.dimension = dimension
        self.top_k = top_k
        self.top_k_dense = top_k_dense or (top_k * 10 if top_k < 100 else 100)
        self.top_k_sparse = top_k_sparse or (top_k * 10 if top_k < 100 else 100)
        if batch_size is None:
            raise ValueError("QdrantVectorStore requires batch_size.")
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.metric_type = metric_type.upper()
        self.prefer_grpc = prefer_grpc
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.enable_sparse = enable_sparse
        self.sparse_vectors_config = sparse_vectors_config
        self.partition_name = partition_name
        self.index_params = index_params or {}
        self.search_params = search_params or {}

        if not self.path and not self.url:
            raise ValueError("QdrantVectorStore requires 'path' (embedded) or 'url' (server).")

        if self.url:
            self.client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc,
            )
        else:
            self.client = AsyncQdrantClient(path=self.path, prefer_grpc=self.prefer_grpc)
        self._collection_ready = False

        logger.info(
            "QdrantVectorStore: path=%s, collection=%s, dim=%s, metric=%s, "
            "partition=%s, top_k_dense=%s, top_k_sparse=%s",
            self.path,
            self.collection_name,
            self.dimension,
            self.metric_type,
            self.partition_name or "default",
            self.top_k_dense,
            self.top_k_sparse,
        )

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if self._collection_ready:
            return

        try:
            collection_info = await self.client.get_collection(self.collection_name)
            self._sync_vector_names_from_collection(collection_info)
            self._collection_ready = True
            return
        except Exception:
            pass

        distance = self._map_distance(self.metric_type)
        vectors_config = {
            self.dense_vector_name: models.VectorParams(size=self.dimension, distance=distance)
        }

        sparse_vectors_config = None
        if self.sparse_vectors_config is not None:
            sparse_vectors_config = {}
            # Get valid field names from SparseIndexParams and SparseVectorParams
            sparse_index_fields = set(models.SparseIndexParams.model_fields.keys())
            sparse_vector_fields = set(models.SparseVectorParams.model_fields.keys())

            # Separate vector configs (dict values) from top-level params (non-dict values)
            # This allows: { sparse: {on_disk: true}, modifier: idf } where modifier applies to sparse
            vector_configs: Dict[str, Dict[str, Any]] = {}
            top_level_params: Dict[str, Any] = {}

            for key, value in self.sparse_vectors_config.items():
                if isinstance(value, dict):
                    vector_configs[key] = dict(value)  # copy to avoid mutating original
                elif key in sparse_vector_fields or key in sparse_index_fields:
                    # Top-level param that should be merged into vector configs
                    top_level_params[key] = value
                else:
                    logger.warning(
                        "Unknown sparse_vectors_config key '%s' with non-dict value; ignoring.", key
                    )

            # If no vector configs defined but we have top-level params, create default
            if not vector_configs and top_level_params:
                vector_configs[self.sparse_vector_name] = {}

            # Process each vector config
            for name, params in vector_configs.items():
                # Merge top-level params (vector config values take precedence)
                merged = {**top_level_params, **params}

                # Support nested "index" dict or flat index params
                if "index" in merged and isinstance(merged["index"], dict):
                    # Nested structure: { index: { on_disk: true, ... }, modifier: ... }
                    index_dict = _coerce_to_model(models.SparseIndexParams, merged["index"])
                    merged["index"] = models.SparseIndexParams(**index_dict)
                else:
                    # Flat structure: auto-extract SparseIndexParams fields
                    index_params = {k: merged.pop(k) for k in list(merged.keys()) if k in sparse_index_fields}
                    if index_params:
                        index_params = _coerce_to_model(models.SparseIndexParams, index_params)
                        merged["index"] = models.SparseIndexParams(**index_params)

                # Convert modifier string to enum if specified (default: none)
                if "modifier" in merged and isinstance(merged["modifier"], str):
                    modifier_str = merged["modifier"].lower()
                    if modifier_str == "idf":
                        merged["modifier"] = models.Modifier.IDF
                    elif modifier_str == "none":
                        merged.pop("modifier")  # Use Qdrant default (no modifier)
                    else:
                        logger.warning("Unknown modifier '%s'; ignoring", merged["modifier"])
                        merged.pop("modifier")
                sparse_vectors_config[name] = models.SparseVectorParams(**merged)
        elif self.enable_sparse:
            # Hardcode IDF modifier for sparse vectors
            sparse_vectors_config = {
                self.sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
            }

        create_kwargs: Dict[str, Any] = dict(self.index_params)

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            **create_kwargs,
        )
        self._collection_ready = True
        # Log actual collection config as stored by Qdrant.
        try:
            created_info = await self.client.get_collection(self.collection_name)
            created_info = _model_to_dict(created_info)
            full_config = created_info.get("config", {}) if isinstance(created_info, dict) else {}
            # Extract relevant config sections for logging
            log_config = {
                "hnsw_config": full_config.get("hnsw_config"),
                "optimizers_config": full_config.get("optimizer_config"),
                "wal_config": full_config.get("wal_config"),
                "vectors": full_config.get("params", {}).get("vectors", {}),
                "sparse_vectors": full_config.get("params", {}).get("sparse_vectors", {}),
            }
        except Exception as exc:
            log_config = {"error": f"failed to read collection config: {exc}"}

        logger.info(
            "Created Qdrant collection: %s with config: %s",
            self.collection_name,
            log_config,
        )

    def _sync_vector_names_from_collection(self, collection_info: Any) -> None:
        info: Dict[str, Any] = _model_to_dict(collection_info)

        vectors = (
            info.get("config", {})
            .get("params", {})
            .get("vectors", {})
        )
        sparse_vectors = (
            info.get("config", {})
            .get("params", {})
            .get("sparse_vectors", {})
        )

        if isinstance(vectors, dict) and vectors:
            # Check if this is an unnamed (flat) vector config — has "size"/"distance"
            # keys directly instead of named sub-dicts
            if "size" in vectors or "distance" in vectors:
                # Unnamed vector — use empty string for query_points `using` param
                if self.dense_vector_name != "":
                    logger.info(
                        "Collection uses unnamed vectors; overriding dense_vector_name "
                        "'%s' -> '' (unnamed).",
                        self.dense_vector_name,
                    )
                    self.dense_vector_name = ""
            elif self.dense_vector_name not in vectors:
                if len(vectors) == 1:
                    detected = next(iter(vectors.keys()))
                    logger.warning(
                        "Dense vector name '%s' not found in existing collection; using '%s' instead.",
                        self.dense_vector_name,
                        detected,
                    )
                    self.dense_vector_name = detected
                else:
                    raise ValueError(
                        f"Dense vector name '{self.dense_vector_name}' not found in existing collection. "
                        f"Available: {list(vectors.keys())}"
                    )

        if isinstance(sparse_vectors, dict) and sparse_vectors and self.enable_sparse:
            if self.sparse_vector_name not in sparse_vectors:
                if len(sparse_vectors) == 1:
                    detected = next(iter(sparse_vectors.keys()))
                    logger.warning(
                        "Sparse vector name '%s' not found in existing collection; using '%s' instead.",
                        self.sparse_vector_name,
                        detected,
                    )
                    self.sparse_vector_name = detected
                else:
                    raise ValueError(
                        f"Sparse vector name '{self.sparse_vector_name}' not found in existing collection. "
                        f"Available: {list(sparse_vectors.keys())}"
                    )

    def _map_distance(self, metric_type: str) -> models.Distance:
        metric_type = metric_type.upper()
        if metric_type == "COSINE":
            return models.Distance.COSINE
        if metric_type in {"IP", "DOT"}:
            return models.Distance.DOT
        if metric_type in {"L2", "EUCLID"}:
            return models.Distance.EUCLID
        return models.Distance.COSINE

    def _normalize_sparse(self, sparse_embedding: Dict[str, float]) -> models.SparseVector:
        items: List[Tuple[int, float]] = []
        for key, value in sparse_embedding.items():
            if value == 0:
                continue
            try:
                idx = int(key)
            except (TypeError, ValueError):
                logger.warning("Skipping sparse key that is not int-convertible: %s", key)
                continue
            items.append((idx, float(value)))

        if not items:
            return models.SparseVector(indices=[], values=[])

        items.sort(key=lambda x: x[0])
        indices = [i for i, _ in items]
        values = [v for _, v in items]
        return models.SparseVector(indices=indices, values=values)

    def _build_filter(self, filter_expr: Optional[str]) -> Optional[models.Filter]:
        if not filter_expr and not self.partition_name:
            return None

        parts = [p.strip() for p in filter_expr.split("AND")] if filter_expr else []
        must_conditions: List[Any] = []

        if self.partition_name:
            must_conditions.append(
                models.FieldCondition(
                    key="partition",
                    match=models.MatchValue(value=self.partition_name),
                )
            )

        for part in parts:
            if not part:
                continue

            match = self._FILTER_EQ_RE.match(part)
            if match:
                key = match.group(1)
                value = match.group(2)
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
                continue

            match = self._FILTER_IN_RE.match(part)
            if match:
                key = match.group(1)
                raw_values = match.group(2).strip()
                try:
                    values = ast.literal_eval(raw_values)
                except Exception:
                    logger.warning("Unsupported filter list format: %s", part)
                    continue

                if key == "id":
                    if isinstance(values, list):
                        must_conditions.append(models.HasIdCondition(has_id=values))
                    continue

                # General field IN filter (e.g., file_id in [...])
                if isinstance(values, list) and values:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=values),
                        )
                    )
                    continue

                logger.warning("Unsupported IN filter values for key '%s': %s", key, part)
                continue

            logger.warning("Unsupported filter expression: %s", part)

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def _normalize_score(self, score: float) -> float:
        if self.metric_type in {"L2", "EUCLID"}:
            return 1.0 / (1.0 + score)
        return float(score)

    @staticmethod
    def _extract_chunk_and_metadata(payload: Dict[str, Any]) -> tuple:
        """Extract chunk text and metadata from a Qdrant payload.

        Supports two payload layouts:
          1. Nested (standard): {"chunk": "...", "metadata": {...}}
          2. Flat (synvo-local): {"text": "...", "file_name": "...", "path": "...", ...}

        Returns:
            (chunk_text, metadata_dict)
        """
        # Chunk text: try "chunk" first, fall back to "text"
        chunk = payload.get("chunk") or payload.get("text") or ""

        # Metadata: try nested "metadata" dict first
        metadata = payload.get("metadata")
        if metadata and isinstance(metadata, dict):
            return chunk, metadata

        # Flat payload — reconstruct metadata in standard format
        file_ext = (payload.get("extension") or "").lstrip(".")
        metadata = {
            "file_info": {
                "file_id": payload.get("file_id", "unknown"),
                "file_path": payload.get("path") or payload.get("full_path") or "",
                "file_name": payload.get("file_name") or "",
                "file_type": file_ext or payload.get("kind") or "text",
            },
            "segment_info": {
                "segment_indices": [],
                "page_numbers": [payload["page_num"]] if payload.get("page_num") is not None else [],
                "time_ranges": [],
            },
            "chunk_meta": {
                "type": payload.get("kind") or "content",
                "chunk_index": payload.get("_rc_chunk_index", 0),
                "char_count": payload.get("char_count", 0),
                "token_count": payload.get("token_count", 0),
            },
        }
        return chunk, metadata

    def _rrf_fuse(
        self,
        dense_points: List[Any],
        sparse_points: List[Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict[str, Any]] = {}

        for rank, point in enumerate(dense_points, start=1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (self._RRF_K + rank)
            if point_id not in payloads:
                payloads[point_id] = point.payload or {}

        for rank, point in enumerate(sparse_points, start=1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (self._RRF_K + rank)
            if point_id not in payloads:
                payloads[point_id] = point.payload or {}

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for point_id, score in ranked[:limit]:
            if score < self.similarity_threshold:
                continue
            payload = payloads.get(point_id, {})
            chunk, metadata = self._extract_chunk_and_metadata(payload)
            results.append({
                "id": point_id,
                "chunk": chunk,
                "metadata": metadata,
                "score": score,
            })
        return results

    def _build_search_params(self) -> Optional[models.SearchParams]:
        if not self.search_params:
            return None
        if isinstance(self.search_params, models.SearchParams):
            return self.search_params
        if isinstance(self.search_params, dict):
            try:
                return models.SearchParams(**self.search_params)
            except Exception as exc:
                logger.warning("Invalid search_params; ignoring. Error: %s", exc)
                return None
        logger.warning("search_params must be a dict or models.SearchParams; ignoring.")
        return None

    async def upsert(
        self,
        ids: List[str],
        dense_embeddings: np.ndarray,
        sparse_embeddings: Optional[List[Dict[str, float]]] = None,
        chunks: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        await self._ensure_collection()

        n = len(ids)
        chunks = chunks or [""] * n
        metadatas = metadatas or [{}] * n

        points: List[models.PointStruct] = []
        for i in range(n):
            vector: Dict[str, Any] = {
                self.dense_vector_name: dense_embeddings[i].tolist(),
            }
            if sparse_embeddings and sparse_embeddings[i]:
                if not self.enable_sparse:
                    logger.warning("Sparse embeddings provided but enable_sparse=False; skipping sparse vectors.")
                else:
                    vector[self.sparse_vector_name] = self._normalize_sparse(sparse_embeddings[i])

            payload = {
                "chunk": chunks[i],
                "metadata": metadatas[i],
            }
            if self.partition_name:
                if "partition" in payload and payload["partition"] != self.partition_name:
                    logger.warning(
                        "Overriding payload partition '%s' with configured partition '%s'",
                        payload["partition"],
                        self.partition_name,
                    )
                payload["partition"] = self.partition_name

            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=vector,
                    payload=payload,
                )
            )

        batches = [
            points[i : i + self.batch_size]
            for i in range(0, len(points), self.batch_size)
        ]

        for batch in batches:
            await self.client.upsert(collection_name=self.collection_name, points=batch)

        logger.info("Upserted %s vectors to %s", len(points), self.collection_name)

    async def query(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        hybrid_search: bool = False,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        await self._ensure_collection()

        query_filter = self._build_filter(filter)
        limit = top_k or self.top_k
        search_params = self._build_search_params()

        if hybrid_search and not sparse_embedding:
            logger.info("hybrid_search=True but sparse_embedding is None. Using dense-only search.")
            hybrid_search = False
        if hybrid_search and not self.enable_sparse:
            logger.info("hybrid_search=True but enable_sparse=False. Using dense-only search.")
            hybrid_search = False

        # For unnamed vectors, don't pass `using` (Qdrant expects it omitted)
        dense_using = self.dense_vector_name or None

        if hybrid_search and sparse_embedding:
            dense_limit = max(limit, self.top_k_dense)
            sparse_limit = max(limit, self.top_k_sparse)
            dense_response = await self.client.query_points(
                collection_name=self.collection_name,
                query=dense_embedding if isinstance(dense_embedding, list) else dense_embedding.tolist(),
                using=dense_using,
                query_filter=query_filter,
                limit=dense_limit,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,
            )
            sparse_response = await self.client.query_points(
                collection_name=self.collection_name,
                query=self._normalize_sparse(sparse_embedding),
                using=self.sparse_vector_name,
                query_filter=query_filter,
                limit=sparse_limit,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,
            )
            return self._rrf_fuse(dense_response.points, sparse_response.points, limit)
        else:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=dense_embedding if isinstance(dense_embedding, list) else dense_embedding.tolist(),
                using=dense_using,
                query_filter=query_filter,
                limit=limit,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points

        results: List[Dict[str, Any]] = []
        for point in points:
            score = self._normalize_score(point.score)
            if score < self.similarity_threshold:
                continue
            payload = point.payload or {}
            chunk, metadata = self._extract_chunk_and_metadata(payload)
            results.append({
                "id": str(point.id),
                "chunk": chunk,
                "metadata": metadata,
                "score": score,
            })

        return results

    async def scroll_by_filter(
        self,
        scroll_filter: models.Filter,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Scroll through points matching a filter (no vector search).

        Args:
            scroll_filter: Qdrant Filter object with field conditions
            limit: Maximum number of points to return

        Returns:
            List of dicts with keys: id, chunk, metadata, chunk_index
        """
        await self._ensure_collection()

        points, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: List[Dict[str, Any]] = []
        for point in points:
            payload = point.payload or {}
            chunk_text, metadata = self._extract_chunk_and_metadata(payload)
            chunk_index = metadata.get("chunk_meta", {}).get("chunk_index", 0)
            results.append({
                "id": str(point.id),
                "chunk": chunk_text,
                "metadata": metadata,
                "chunk_index": chunk_index,
            })

        return results

    async def filter_existing(self, ids: List[str]) -> List[str]:
        if not ids:
            return []

        await self._ensure_collection()
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        return [str(p.id) for p in points]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[str] = None,
    ) -> int:
        if ids is None and filter is None:
            raise ValueError("Either 'ids' or 'filter' must be provided")

        await self._ensure_collection()

        if ids is not None:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
            )
            return len(ids)

        qdrant_filter = self._build_filter(filter)
        if qdrant_filter is None:
            return 0

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter),
        )
        return 0

    async def aclose(self) -> None:
        if hasattr(self, "client") and self.client:
            await self.client.close()

    async def close(self) -> None:
        """Alias for aclose() to match other vector stores."""
        await self.aclose()
