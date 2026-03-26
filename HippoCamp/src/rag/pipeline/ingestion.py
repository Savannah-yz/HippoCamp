"""
Ingestion Module: File upload -> Backend processing -> Store metadata and captions
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from storage.metadata_storage import MetadataStore
from memory.models.factory import BackendFactory

logger = logging.getLogger(__name__)


def stable_file_id(path: Path) -> str:
    """Generate a stable, collision-resistant file_id from the full path."""
    resolved = path.resolve()
    h = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}__{h}"


def _get_file_type(file_path: Path) -> str:
    """Determine file type from extension."""
    ext = file_path.suffix.lower()
    ext_map = {
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "docx",
        ".ppt": "ppt",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ics": "ics",
        ".eml": "eml",
        ".ipynb": "ipynb",
        ".md": "md",
        ".txt": "txt",
        ".log": "log",
        ".py": "py",
        ".sh": "sh",
        ".sqlite": "sqlite",
        ".db": "sqlite",
        ".mp4": "mp4",
        ".mov": "mov",
        ".avi": "avi",
        ".mkv": "mkv",
        ".webm": "webm",
        ".mp3": "mp3",
        ".wav": "wav",
        ".flac": "flac",
        ".aac": "aac",
        ".ogg": "ogg",
        ".m4a": "m4a",
        ".wma": "wma",
        ".jpg": "jpg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".bmp": "bmp",
        ".webp": "webp",
        ".tiff": "tiff",
        ".tif": "tif",
        ".bin": "bin",
        ".pkl": "pkl",
        ".npy": "npy",
        ".pt": "pt",
        ".pth": "pth",
    }
    if ext in ext_map:
        return ext_map[ext]
    return "unknown"


def ingest_file(
    file_path: str,
    metadata_dir: Optional[str] = None,
    force_refresh: bool = False,
    num_processes: int = 5,
    config_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a file and generate metadata.
    
    Args:
        file_path: Path to the file to process
        metadata_dir: Directory to store metadata (default: ./data/metadata)
        force_refresh: Whether to force reprocessing even if metadata exists
        num_processes: Number of parallel processes for processing
        config_path: Path to ingestion config YAML file (default: configs/pipelines/ingestion.yaml)
        
    Returns:
        Metadata dictionary if successful, None otherwise
        
    Example:
        >>> metadata = ingest_file("/path/to/video.mp4")
        >>> if metadata:
        ...     print(f"Processed: {metadata['file_info']['file_id']}")
    """
    file_path_obj = Path(file_path)
    
    # Initialize storage
    if metadata_dir is None:
        metadata_dir = "./data/metadata"
    store = MetadataStore(base_dir=metadata_dir)
    file_id = stable_file_id(file_path_obj)
    
    logger.info(f"[Ingestion] Processing file: {file_path_obj.name} (ID: {file_id})")
    
    # Check cache
    if store.exists(file_id) and not force_refresh:
        logger.info(f"[Ingestion] Metadata already exists for {file_id}. Loading from cache.")
        metadata = store.load(file_id)
        
        # Check if metadata has file_info (for backward compatibility)
        # If missing, regenerate to add file_info
        if metadata and "file_info" not in metadata:
            logger.info(f"[Ingestion] Metadata missing file_info, will regenerate to add it...")
            # Continue to regeneration below
        else:
            logger.info(f"[Ingestion] ✅ Loaded existing metadata for {file_id}")
            return metadata
    
    # If we reach here, either:
    # 1. File doesn't exist, OR
    # 2. force_refresh=True, OR  
    # 3. Metadata exists but missing file_info (backward compatibility)
    
    if not file_path_obj.exists():
        logger.error(f"[Ingestion] File not found: {file_path_obj}")
        return None
    
    # Get processor
    try:
        processor = BackendFactory.get_processor(file_path_obj, config_path=config_path)
        logger.info(f"[Ingestion] Selected processor: {processor.__class__.__name__}")
    except Exception as e:
        logger.error(f"[Ingestion] Failed to get processor: {e}")
        return None
    
    # Process (Generate Metadata)
    try:
        logger.info("[Ingestion] Generating metadata... (This may take a while)")
        metadata = processor.generate(
            file_path=file_path_obj,
            num_processes=num_processes,
            file_id=file_id
        )
        
        # Add file-level metadata before saving
        file_info = {
            "file_id": file_id,
            "file_path": str(file_path_obj.resolve()),
            "file_type": _get_file_type(file_path_obj),
            "file_name": file_path_obj.name,
        }
        
        # Wrap metadata with file_info at the top
        final_metadata = {
            "file_info": file_info,
            **metadata  # summary, hash_tags, segments, etc.
        }
        
        # Save
        store.save(file_id, final_metadata)
        logger.info(f"[Ingestion] ✅ Success! Metadata saved to {store.base_dir}/{file_id}.json")
        return final_metadata
    
    except Exception as e:
        logger.error(f"[Ingestion] Processing failed: {e}", exc_info=True)
        return None


def ingest_files(
    file_paths: list[str],
    metadata_dir: Optional[str] = None,
    force_refresh: bool = False,
    num_processes: int = 5,
    config_path: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Process multiple files and generate metadata.
    
    Args:
        file_paths: List of file paths to process
        metadata_dir: Directory to store metadata (default: ./data/metadata)
        force_refresh: Whether to force reprocessing even if metadata exists
        num_processes: Number of parallel processes for processing
        config_path: Path to ingestion config YAML file (default: configs/pipelines/ingestion.yaml)
        
    Returns:
        Dictionary mapping file_id -> metadata (or None if failed)
        
    Example:
        >>> files = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
        >>> results = ingest_files(files)
        >>> for file_id, metadata in results.items():
        ...     if metadata:
        ...         print(f"✅ {file_id}")
    """
    results = {}
    
    for file_path in file_paths:
        file_id = stable_file_id(Path(file_path))
        metadata = ingest_file(
            file_path=file_path,
            metadata_dir=metadata_dir,
            force_refresh=force_refresh,
            num_processes=num_processes,
            config_path=config_path,
        )
        results[file_id] = metadata
    
    return results


def load_metadata(
    file_id: str,
    metadata_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load existing metadata for a file.
    
    Args:
        file_id: File ID to load metadata for
        metadata_dir: Directory where metadata is stored (default: ./data/metadata)
        
    Returns:
        Metadata dictionary if found, None otherwise
    """
    if metadata_dir is None:
        metadata_dir = "./data/metadata"
    
    store = MetadataStore(base_dir=metadata_dir)
    
    if not store.exists(file_id):
        logger.warning(f"[Ingestion] Metadata not found for {file_id}")
        return None
    
    return store.load(file_id)
