"""
RAG Pipeline Modules
====================

Three independent modules for flexible pipeline composition:

1. ingestion: File upload -> Backend processing -> Store metadata
2. indexing: Chunk -> Embed -> Store in vector DB
3. query: User input -> Retrieve -> Rank -> Return results
"""

from src.rag.pipeline.ingestion import (
    ingest_file,
    ingest_files,
    load_metadata,
)

from src.rag.pipeline.indexing import (
    index_metadata,
    index_file,
    index_files,
    index_file_from_path,
    index_files_from_paths,
    index_from_metadata_folder,
    index_direct,
)

from src.rag.pipeline.query import (
    query,
    query_single,
)

from src.rag.pipeline.utils import (
    load_file_paths,
    load_queries,
    find_metadata_file,
    load_metadata_from_folder,
)

__all__ = [
    # Ingestion
    "ingest_file",
    "ingest_files",
    "load_metadata",
    # Indexing
    "index_metadata",
    "index_file",
    "index_files",
    "index_file_from_path",
    "index_files_from_paths",
    "index_from_metadata_folder",
    "index_direct",
    # Query
    "query",
    "query_single",
    # Utils
    "load_file_paths",
    "load_queries",
    "find_metadata_file",
    "load_metadata_from_folder",
]

