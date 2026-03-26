"""
Indexing Module: Chunk -> Embed (hybrid) -> Store in vector DB
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import asyncio

from src.rag.chunking.recursive import ChonkieRecursiveChunker
from src.rag.embedding.factory import EmbedderFactory
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.chunking.base import Chunk

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path_obj, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


async def index_metadata(
    metadata: Dict[str, Any],
    file_id: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> bool:
    """
    Index metadata: Chunk -> Embed -> Store in vector DB.
    
    Args:
        metadata: Metadata dictionary from ingestion
        file_id: File ID
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> metadata = load_metadata("trial")
        >>> success = await index_metadata(
        ...     metadata=metadata,
        ...     file_id="trial",
        ...     embedding_config_path="configs/embedding.yaml",
        ...     vector_store_config_path="configs/vector_store.yaml"
        ... )
    """
    logger.info(f"[Indexing] Indexing file: {file_id}")
    
    # Load configurations
    try:
        embedding_config = load_config(embedding_config_path)
        vector_store_config = load_config(vector_store_config_path)
    except FileNotFoundError as e:
        logger.error(f"[Indexing] Configuration error: {e}")
        return False
    
    # Step 1: Create chunks
    logger.info("[Indexing] Creating chunks from metadata...")
    chunker = ChonkieRecursiveChunker(chunk_size=chunk_size, min_chars=10, overlap=50)
    chunks = chunker.create_chunks_from_metadata(metadata)
    logger.info(f"[Indexing] ✅ Created {len(chunks)} chunks")
    
    if not chunks:
        logger.warning("[Indexing] No chunks created. Skipping indexing.")
        return False
    
    # Step 2: Initialize embedder
    logger.info("[Indexing] Initializing embedder...")
    embedder_config = embedding_config.get("embedding", {})
    embedder = EmbedderFactory.create(embedder_config)
    logger.info(f"[Indexing] ✅ Initialized embedder: {embedder.get_model_name()}")
    
    # Step 3: Initialize vector store
    logger.info("[Indexing] Initializing vector store...")
    vs_config = vector_store_config.get("vector_store", {})
    vector_store = VectorStoreFactory.create(vs_config)
    logger.info(f"[Indexing] ✅ Initialized vector store: {vs_config.get('type', 'unknown')}")
    
    # Step 4: Filter existing chunks (optional, to avoid re-indexing)
    if not force_reindex:
        logger.info("[Indexing] Checking for existing chunks...")
        chunk_ids = [chunk.id for chunk in chunks]
        existing_ids = await vector_store.filter_existing(chunk_ids)
        if existing_ids:
            logger.info(f"[Indexing] Found {len(existing_ids)} existing chunks. Skipping them.")
            chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
            if not chunks:
                logger.info("[Indexing] All chunks already indexed. Use force_reindex=True to re-index.")
                await embedder.aclose()
                await vector_store.aclose()
                return True
    
    if not chunks:
        logger.info("[Indexing] No new chunks to index.")
        await embedder.aclose()
        await vector_store.aclose()
        return True
    
    # Step 5: Generate embeddings
    logger.info(f"[Indexing] Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.content for chunk in chunks]
    embedding_result = await embedder.embed(texts, return_sparse=True)
    logger.info(f"[Indexing] ✅ Generated embeddings: dense shape={embedding_result.dense_embeddings.shape}")
    
    # Step 6: Store in vector DB
    logger.info("[Indexing] Storing chunks in vector database...")
    await vector_store.upsert(
        ids=[chunk.id for chunk in chunks],
        dense_embeddings=embedding_result.dense_embeddings,
        sparse_embeddings=embedding_result.sparse_embeddings,
        chunks=[chunk.content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    logger.info(f"[Indexing] ✅ Successfully indexed {len(chunks)} chunks")
    
    # Cleanup
    await embedder.aclose()
    await vector_store.aclose()
    
    return True


async def index_file_from_path(
    file_path: str,
    metadata_dir: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> bool:
    """
    Index a file by loading its metadata from file path.
    
    This is for Mode 2: given file path, find metadata and index.
    Supports flexible metadata file naming.
    
    Args:
        file_path: File path (will be used to find metadata)
        metadata_dir: Directory where metadata is stored
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        True if successful, False otherwise
    """
    from src.rag.pipeline.utils import find_metadata_file
    
    # Try to find metadata file (supports flexible naming)
    metadata_path = find_metadata_file(file_path, metadata_dir)
    file_id = Path(file_path).stem
    
    if not metadata_path:
        logger.error(f"[Indexing] Metadata not found for {file_path}")
        logger.error(f"[Indexing] Searched in: {metadata_dir}")
        logger.error(f"[Indexing] Please ensure metadata file exists or check metadata_dir")
        return False
    
    # Load metadata
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"[Indexing] Loaded metadata from {metadata_path.name}")
    
    # Index metadata
    return await index_metadata(
        metadata=metadata,
        file_id=file_id,
        embedding_config_path=embedding_config_path,
        vector_store_config_path=vector_store_config_path,
        chunk_size=chunk_size,
        force_reindex=force_reindex,
    )


async def index_from_metadata_folder(
    metadata_folder: str,
    file_paths: Optional[List[str]] = None,
    embedding_config_path: str = None,
    vector_store_config_path: str = None,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> Dict[str, bool]:
    """
    Index from a folder containing metadata JSON files.
    
    This is for importing metadata from external sources.
    Metadata files can have any naming convention.
    
    Args:
        metadata_folder: Folder containing metadata JSON files
        file_paths: Optional list of file paths to match against metadata files
                    If None, indexes all metadata files in the folder
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        Dictionary mapping identifier -> success (True/False)
        
    Example:
        >>> # Index all metadata files in folder
        >>> results = await index_from_metadata_folder(
        ...     metadata_folder="/path/to/metadata_folder",
        ...     embedding_config_path="configs/embedding.yaml",
        ...     vector_store_config_path="configs/vector_store.yaml"
        ... )
        
        >>> # Index with file path matching
        >>> results = await index_from_metadata_folder(
        ...     metadata_folder="/path/to/metadata_folder",
        ...     file_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"],
        ...     embedding_config_path="configs/embedding.yaml",
        ...     vector_store_config_path="configs/vector_store.yaml"
        ... )
    """
    from src.rag.pipeline.utils import load_metadata_from_folder
    
    logger.info(f"[Indexing] Indexing from metadata folder: {metadata_folder}")
    
    # Load metadata from folder
    metadata_map = load_metadata_from_folder(metadata_folder, file_paths)
    
    if not metadata_map:
        logger.warning("[Indexing] No metadata files found in folder")
        return {}
    
    results = {}
    
    # Index each metadata
    for identifier, metadata in metadata_map.items():
        # Use identifier as file_id (could be file_path or json filename)
        if Path(identifier).exists():
            # If identifier is a file path, use its stem
            file_id = Path(identifier).stem
        else:
            # Otherwise use identifier as-is (json filename without extension)
            file_id = identifier
        
        logger.info(f"[Indexing] Indexing metadata: {identifier} (file_id: {file_id})")
        
        success = await index_metadata(
            metadata=metadata,
            file_id=file_id,
            embedding_config_path=embedding_config_path,
            vector_store_config_path=vector_store_config_path,
            chunk_size=chunk_size,
            force_reindex=force_reindex,
        )
        
        results[identifier] = success
    
    return results


async def index_file(
    file_id: str,
    metadata_dir: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> bool:
    """
    Index a file by loading its metadata and indexing it.
    
    Args:
        file_id: File ID to index
        metadata_dir: Directory where metadata is stored
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        True if successful, False otherwise
    """
    from src.rag.pipeline.ingestion import load_metadata
    
    # Load metadata
    metadata = load_metadata(file_id, metadata_dir=metadata_dir)
    if not metadata:
        logger.error(f"[Indexing] Metadata not found for {file_id}. Please run ingestion first.")
        return False
    
    # Index metadata
    return await index_metadata(
        metadata=metadata,
        file_id=file_id,
        embedding_config_path=embedding_config_path,
        vector_store_config_path=vector_store_config_path,
        chunk_size=chunk_size,
        force_reindex=force_reindex,
    )


async def index_files(
    file_ids: List[str],
    metadata_dir: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> Dict[str, bool]:
    """
    Index multiple files by file ID.
    
    Args:
        file_ids: List of file IDs to index
        metadata_dir: Directory where metadata is stored
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        Dictionary mapping file_id -> success (True/False)
    """
    results = {}
    
    for file_id in file_ids:
        success = await index_file(
            file_id=file_id,
            metadata_dir=metadata_dir,
            embedding_config_path=embedding_config_path,
            vector_store_config_path=vector_store_config_path,
            chunk_size=chunk_size,
            force_reindex=force_reindex,
        )
        results[file_id] = success
    
    return results


async def index_files_from_paths(
    file_paths: List[str],
    metadata_dir: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> Dict[str, bool]:
    """
    Index multiple files from file paths (Mode 2).
    
    Given file paths, find corresponding metadata files and index them.
    
    Args:
        file_paths: List of file paths
        metadata_dir: Directory where metadata is stored
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        Dictionary mapping file_path -> success (True/False)
    """
    results = {}
    
    for file_path in file_paths:
        success = await index_file_from_path(
            file_path=file_path,
            metadata_dir=metadata_dir,
            embedding_config_path=embedding_config_path,
            vector_store_config_path=vector_store_config_path,
            chunk_size=chunk_size,
            force_reindex=force_reindex,
        )
        results[file_path] = success
    
    return results


async def index_direct(
    file_path: str,
    embedding_config_path: str,
    vector_store_config_path: str,
    chunk_size: int = 400,
    force_reindex: bool = False,
) -> bool:
    """
    Direct indexing: Process file -> Chunk -> Embed -> Store (skip metadata storage).
    
    This is useful for Mode 1: File -> VectorDB -> Query (skip metadata storage).
    
    Args:
        file_path: Path to the file to process
        embedding_config_path: Path to embedding config YAML
        vector_store_config_path: Path to vector store config YAML
        chunk_size: Chunk size for chunking
        force_reindex: Whether to force re-indexing even if chunks exist
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> success = await index_direct(
        ...     file_path="/path/to/video.mp4",
        ...     embedding_config_path="configs/embedding.yaml",
        ...     vector_store_config_path="configs/vector_store.yaml"
        ... )
    """
    from storage.metadata_storage import MetadataStore
    from memory.models.factory import BackendFactory
    
    logger.info(f"[Indexing] Direct indexing: {file_path}")
    
    file_path_obj = Path(file_path)
    file_id = file_path_obj.stem
    
    if not file_path_obj.exists():
        logger.error(f"[Indexing] File not found: {file_path_obj}")
        return False
    
    # Step 1: Process file directly (generate metadata in memory)
    try:
        processor = BackendFactory.get_processor(file_path_obj)
        logger.info(f"[Indexing] Selected processor: {processor.__class__.__name__}")
    except Exception as e:
        logger.error(f"[Indexing] Failed to get processor: {e}")
        return False
    
    try:
        logger.info("[Indexing] Generating metadata... (This may take a while)")
        metadata = processor.generate(
            file_path=file_path_obj,
            num_processes=5,
            file_id=file_id
        )
        logger.info("[Indexing] ✅ Metadata generated (not saved)")
    except Exception as e:
        logger.error(f"[Indexing] Processing failed: {e}", exc_info=True)
        return False
    
    if not metadata:
        logger.error("[Indexing] Failed to generate metadata")
        return False
    
    # Step 2: Index metadata (chunk + embed + store)
    return await index_metadata(
        metadata=metadata,
        file_id=file_id,
        embedding_config_path=embedding_config_path,
        vector_store_config_path=vector_store_config_path,
        chunk_size=chunk_size,
        force_reindex=force_reindex,
    )

