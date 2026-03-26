"""
Query Module: User input -> Retrieve -> Rank -> Return results
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

from src.rag.embedding.factory import EmbedderFactory
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.rerank.factory import RerankerFactory
from src.providers.utils import format_query_for_embedding

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path_obj, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


async def query_single(
    query_text: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    rerank_config_path: Optional[str] = None,
    top_k: int = 10,
    use_rerank: bool = True,
    retrieve_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector database and return ranked results.
    
    Args:
        query_text: User query string
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        rerank_config_path: Path to rerank config YAML (optional)
        top_k: Number of final results to return
        use_rerank: Whether to use reranking
        retrieve_top_k: Number of candidates to retrieve for reranking (default: top_k * 10)
        
    Returns:
        List of retrieved and reranked results
        
    Example:
        >>> results = await query(
        ...     query_text="What is this video about?",
        ...     embedding_config_path="configs/embedding.yaml",
        ...     vector_store_config_path="configs/vector_store.yaml",
        ...     rerank_config_path="configs/rerank.yaml",
        ...     top_k=10
        ... )
        >>> for result in results:
        ...     print(f"Score: {result['score']}, Chunk: {result['chunk'][:100]}")
    """
    logger.info(f"[Query] Query: {query_text}")
    
    # Load configurations
    try:
        embedding_config = load_config(embedding_config_path)
        vector_store_config = load_config(vector_store_config_path)
        rerank_config = None
        if use_rerank and rerank_config_path:
            rerank_config = load_config(rerank_config_path)
    except FileNotFoundError as e:
        logger.error(f"[Query] Configuration error: {e}")
        return []
    
    # Step 1: Initialize embedder
    logger.info("[Query] Initializing embedder...")
    embedder_config = embedding_config.get("embedding", {})
    embedder = EmbedderFactory.create(embedder_config)
    logger.info(f"[Query] ✅ Initialized embedder: {embedder.get_model_name()}")
    
    # Step 2: Initialize vector store
    logger.info("[Query] Initializing vector store...")
    vs_config = vector_store_config.get("vector_store", {})
    vector_store = VectorStoreFactory.create(vs_config)
    logger.info(f"[Query] ✅ Initialized vector store: {vs_config.get('type', 'unknown')}")
    
    # Step 3: Initialize reranker (if enabled)
    reranker = None
    if use_rerank and rerank_config:
        logger.info("[Query] Initializing reranker...")
        rerank_cfg = rerank_config.get("rerank", {})
        reranker = RerankerFactory.create(rerank_cfg)
        logger.info(f"[Query] ✅ Initialized reranker: {reranker.get_model_name()}")
    
    try:
        # Step 4: Embed query (with instruction prefix for retrieval)
        logger.info("[Query] Embedding query...")
        query_for_embed = format_query_for_embedding(query_text)
        query_result = await embedder.embed([query_for_embed], return_sparse=True)
        query_dense = query_result.dense_embeddings[0]
        query_sparse = query_result.sparse_embeddings[0] if query_result.sparse_embeddings else None
        logger.info("[Query] ✅ Query embedded")
        
        # Step 5: Retrieve candidates
        if use_rerank and reranker:
            # For reranking, retrieve more candidates
            if retrieve_top_k is None:
                retrieve_top_k = max(top_k * 10, 100)  # Get more candidates for reranking
            
            logger.info(f"[Query] Retrieving {retrieve_top_k} candidates for reranking...")
            candidates = await vector_store.query(
                dense_embedding=query_dense,
                sparse_embedding=query_sparse,
                top_k=retrieve_top_k,
                hybrid_search=(query_sparse is not None),
            )
            logger.info(f"[Query] ✅ Retrieved {len(candidates)} candidates")
            
            # Step 6: Rerank candidates
            logger.info("[Query] Reranking candidates...")
            reranked = await reranker.rerank(
                query=query_text,
                candidates=candidates,
                top_k=top_k,
                threshold=rerank_cfg.get("threshold", 0.0),
            )
            logger.info(f"[Query] ✅ Reranked to {len(reranked)} results")
            results = reranked
        else:
            # Direct retrieval without reranking
            logger.info(f"[Query] Retrieving {top_k} results...")
            results = await vector_store.query(
                dense_embedding=query_dense,
                sparse_embedding=query_sparse,
                top_k=top_k,
                hybrid_search=(query_sparse is not None),
            )
            logger.info(f"[Query] ✅ Retrieved {len(results)} results")
        
        # Step 7: Display results
        logger.info("\n[Query] === Retrieval Results ===")
        for i, result in enumerate(results, 1):
            score = result.get("score", 0.0)
            chunk = result.get("chunk", "")[:200]  # Preview first 200 chars
            metadata = result.get("metadata", {})
            file_id = metadata.get("file_id", "unknown")
            
            logger.info(f"\n[Query] [{i}] Score: {score:.4f} | File: {file_id}")
            logger.info(f"[Query]     Chunk: {chunk}...")
        
        return results
    
    finally:
        # Cleanup
        await embedder.aclose()
        await vector_store.aclose()
        if reranker:
            await reranker.aclose()


async def query(
    query_input: Union[str, List[Dict[str, Any]]],
    embedding_config_path: str,
    vector_store_config_path: str,
    rerank_config_path: Optional[str] = None,
    top_k: int = 10,
    use_rerank: bool = True,
    retrieve_top_k: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query with support for single query or batch queries from JSON.
    
    Args:
        query_input: Can be:
            - str: Single query string or path to JSON file
            - List[Dict]: List of query dicts with 'question' key
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        rerank_config_path: Path to rerank config YAML (optional)
        top_k: Number of final results to return per query
        use_rerank: Whether to use reranking
        retrieve_top_k: Number of candidates to retrieve for reranking
        
    Returns:
        Dictionary mapping query text -> results list
        
    Example:
        >>> # Single query
        >>> results = await query("What is this about?", ...)
        
        >>> # JSON file with queries
        >>> results = await query("/path/to/queries.json", ...)
    """
    from src.rag.pipeline.utils import load_queries
    
    # Load queries
    query_dicts = load_queries(query_input)
    
    if len(query_dicts) == 1:
        # Single query
        query_text = query_dicts[0]["question"]
        results_list = await query_single(
            query_text=query_text,
            embedding_config_path=embedding_config_path,
            vector_store_config_path=vector_store_config_path,
            rerank_config_path=rerank_config_path,
            top_k=top_k,
            use_rerank=use_rerank,
            retrieve_top_k=retrieve_top_k,
        )
        return {query_text: results_list}
    else:
        # Batch queries
        results = {}
        for query_dict in query_dicts:
            query_text = query_dict["question"]
            logger.info(f"\n[Query] Processing query: {query_text}")
            query_results = await query_single(
                query_text=query_text,
                embedding_config_path=embedding_config_path,
                vector_store_config_path=vector_store_config_path,
                rerank_config_path=rerank_config_path,
                top_k=top_k,
                use_rerank=use_rerank,
                retrieve_top_k=retrieve_top_k,
            )
            results[query_text] = query_results
        
        return results

