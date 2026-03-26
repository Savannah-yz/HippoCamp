"""
Utility functions for pipeline operations.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


def load_file_paths(input_path: Union[str, List[str]]) -> List[str]:
    """
    Load file paths from various input formats.
    
    Supports:
    - Single file path (string)
    - List of file paths
    - Text file with one path per line
    - JSON file with list of paths or dict with 'files' key
    
    Args:
        input_path: Input path(s) - can be:
            - str: Single file path or path to txt/json file
            - List[str]: List of file paths
            
    Returns:
        List of file paths
        
    Example:
        >>> # Single file
        >>> paths = load_file_paths("/path/to/video.mp4")
        
        >>> # List of files
        >>> paths = load_file_paths(["/path/to/v1.mp4", "/path/to/v2.mp4"])
        
        >>> # Text file (one path per line)
        >>> paths = load_file_paths("/path/to/files.txt")
        
        >>> # JSON file
        >>> # {"files": ["/path/to/v1.mp4", "/path/to/v2.mp4"]}
        >>> paths = load_file_paths("/path/to/files.json")
    """
    # If it's already a list, return as is
    if isinstance(input_path, list):
        return input_path
    
    # If it's a string, check if it's a file that contains paths
    input_path_obj = Path(input_path)
    
    # If the path doesn't exist, assume it's a direct file path
    if not input_path_obj.exists():
        logger.warning(f"Path does not exist, treating as direct file path: {input_path}")
        return [str(input_path)]
    
    # Check file extension
    if input_path_obj.suffix == '.txt':
        # Text file: one path per line
        logger.info(f"Loading file paths from text file: {input_path}")
        with open(input_path_obj, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(paths)} file paths from text file")
        return paths
    
    elif input_path_obj.suffix == '.json':
        # JSON file: can be list or dict with 'files' key
        logger.info(f"Loading file paths from JSON file: {input_path}")
        with open(input_path_obj, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            paths = data
        elif isinstance(data, dict):
            # Try common keys
            if 'files' in data:
                paths = data['files']
            elif 'file_paths' in data:
                paths = data['file_paths']
            elif 'paths' in data:
                paths = data['paths']
            else:
                raise ValueError(
                    f"JSON file must contain 'files', 'file_paths', or 'paths' key, "
                    f"or be a list. Found keys: {list(data.keys())}"
                )
        else:
            raise ValueError(f"JSON file must be a list or dict, got {type(data)}")
        
        logger.info(f"Loaded {len(paths)} file paths from JSON file")
        return paths
    
    else:
        # Assume it's a direct file path
        logger.info(f"Treating as direct file path: {input_path}")
        return [str(input_path)]


def load_queries(input_path: Union[str, List[str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Load queries from various input formats.

    Supports:
    - Single query string
    - List of query strings
    - JSON file with list of queries or dict with 'questions' key
    - Dict with query data
    - HippoCamp/Victoria benchmark format with 'question', 'answer', 'evidence' fields

    Args:
        input_path: Input query(s) - can be:
            - str: Single query string or path to JSON file
            - List[str]: List of query strings
            - Dict: Query dict with 'question' key

    Returns:
        List of query dictionaries, each with at least 'question' key
        May also include 'id', 'answer' (ground truth), 'evidence', 'file_path' fields

    Example:
        >>> # Single query string
        >>> queries = load_queries("What is this about?")

        >>> # List of queries
        >>> queries = load_queries(["Q1", "Q2"])

        >>> # JSON file with list
        >>> # [{"question": "Q1"}, {"question": "Q2"}]
        >>> queries = load_queries("/path/to/queries.json")

        >>> # JSON file with dict
        >>> # {"questions": [{"question": "Q1"}, {"question": "Q2"}]}
        >>> queries = load_queries("/path/to/queries.json")

        >>> # HippoCamp/Victoria benchmark format
        >>> # [{"id": "1", "question": "...", "answer": "...", "evidence": [...]}]
        >>> queries = load_queries("/path/to/benchmark.json")
    """
    # If it's already a list of dicts, return as is
    if isinstance(input_path, list):
        if all(isinstance(q, dict) for q in input_path):
            return input_path
        elif all(isinstance(q, str) for q in input_path):
            # Convert list of strings to list of dicts
            return [{"question": q} for q in input_path]
        else:
            raise ValueError("List must contain only strings or dicts")

    # If it's a dict, wrap in list
    if isinstance(input_path, dict):
        if 'question' in input_path:
            return [input_path]
        else:
            raise ValueError("Dict must contain 'question' key")

    # If it's a string, check if it's a file path or direct query
    input_path_obj = Path(input_path)

    # If the path doesn't exist, assume it's a direct query string
    if not input_path_obj.exists():
        logger.info(f"Path does not exist, treating as direct query: {input_path}")
        return [{"question": str(input_path)}]

    # Check file extension
    if input_path_obj.suffix == '.json':
        logger.info(f"Loading queries from JSON file: {input_path}")
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # List of query dicts or strings
            queries = []
            for item in data:
                if isinstance(item, dict):
                    # Support 'question' key directly or fallback to other formats
                    if 'question' in item:
                        queries.append(item)
                    elif 'query' in item:
                        # Map 'query' to 'question' for compatibility
                        item_copy = dict(item)
                        item_copy['question'] = item_copy.pop('query')
                        queries.append(item_copy)
                    else:
                        raise ValueError(f"Query dict must contain 'question' or 'query' key: {list(item.keys())}")
                elif isinstance(item, str):
                    queries.append({"question": item})
                else:
                    raise ValueError(f"Query item must be dict or string, got {type(item)}")
        elif isinstance(data, dict):
            # Try common keys
            if 'questions' in data:
                queries = data['questions']
            elif 'queries' in data:
                queries = data['queries']
            elif 'question' in data:
                queries = [data]
            else:
                raise ValueError(
                    f"JSON file must contain 'questions', 'queries', or 'question' key, "
                    f"or be a list. Found keys: {list(data.keys())}"
                )
        else:
            raise ValueError(f"JSON file must be a list or dict, got {type(data)}")

        logger.info(f"Loaded {len(queries)} queries from JSON file")
        return queries

    else:
        # Assume it's a direct query string
        logger.info(f"Treating as direct query: {input_path}")
        return [{"question": str(input_path)}]


def extract_expected_chunks_from_evidence(evidence: List[Dict[str, Any]]) -> List[str]:
    """
    Extract expected chunk identifiers from evidence list.

    This is useful for retrieval evaluation - comparing retrieved chunks
    against expected chunks from benchmark data.

    Args:
        evidence: List of evidence dicts from benchmark format

    Returns:
        List of file_path values from evidence items

    Example:
        >>> evidence = [
        ...     {"file_path": "Documents/file1.txt", "evidence_text": "..."},
        ...     {"file_path": "Documents/file2.txt", "evidence_text": "..."}
        ... ]
        >>> extract_expected_chunks_from_evidence(evidence)
        ['Documents/file1.txt', 'Documents/file2.txt']
    """
    if not evidence:
        return []
    return [e.get("file_path", "") for e in evidence if e.get("file_path")]


def get_metadata_path(file_path: str, metadata_dir: str) -> Path:
    """
    Get metadata file path from file path.
    
    Args:
        file_path: Original file path
        metadata_dir: Directory where metadata is stored
        
    Returns:
        Path to metadata JSON file
    """
    file_id = Path(file_path).stem
    return Path(metadata_dir) / f"{file_id}.json"


def find_metadata_file(file_path: str, metadata_dir: str) -> Optional[Path]:
    """
    Find metadata file for a given file path.
    
    Tries multiple strategies:
    1. Standard: {file_id}.json
    2. By filename: {filename}.json (with extension)
    3. By full filename: {full_filename}.json
    
    Args:
        file_path: Original file path
        metadata_dir: Directory where metadata is stored
        
    Returns:
        Path to metadata JSON file if found, None otherwise
    """
    metadata_dir_path = Path(metadata_dir)
    file_path_obj = Path(file_path)
    
    # Strategy 1: Standard {file_id}.json (stem)
    file_id = file_path_obj.stem
    metadata_path = metadata_dir_path / f"{file_id}.json"
    if metadata_path.exists():
        return metadata_path
    
    # Strategy 2: By filename with extension {filename}.json
    filename = file_path_obj.name
    metadata_path = metadata_dir_path / f"{filename}.json"
    if metadata_path.exists():
        return metadata_path
    
    # Strategy 3: Search for any JSON file that might match
    # Try to find by partial match
    for json_file in metadata_dir_path.glob("*.json"):
        json_stem = json_file.stem
        # Check if json filename matches file_id or filename
        if json_stem == file_id or json_stem == filename or json_stem == file_path_obj.name:
            return json_file
    
    return None


def load_metadata_from_folder(
    metadata_folder: str,
    file_paths: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata files from a folder.
    
    Args:
        metadata_folder: Folder containing metadata JSON files
        file_paths: Optional list of file paths to match against metadata files
                    If None, loads all JSON files in the folder
        
    Returns:
        Dictionary mapping file_path -> metadata dict
        
    Example:
        >>> # Load all metadata files in folder
        >>> metadata_map = load_metadata_from_folder("/path/to/metadata_folder")
        
        >>> # Load and match with file paths
        >>> metadata_map = load_metadata_from_folder(
        ...     "/path/to/metadata_folder",
        ...     file_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"]
        ... )
    """
    import json
    
    metadata_folder_path = Path(metadata_folder)
    if not metadata_folder_path.exists():
        raise FileNotFoundError(f"Metadata folder not found: {metadata_folder}")
    
    if not metadata_folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {metadata_folder}")
    
    # Load all JSON files
    json_files = list(metadata_folder_path.glob("*.json"))
    logger.info(f"[Utils] Found {len(json_files)} JSON files in {metadata_folder}")
    
    metadata_map = {}
    
    if file_paths is None:
        # Load all JSON files, use filename as key
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                # Use JSON filename (without extension) as key
                key = json_file.stem
                metadata_map[key] = metadata
                logger.debug(f"[Utils] Loaded metadata: {json_file.name}")
            except Exception as e:
                logger.warning(f"[Utils] Failed to load {json_file}: {e}")
    else:
        # Match JSON files with file paths
        for file_path in file_paths:
            file_path_obj = Path(file_path)
            
            # Try to find matching metadata file
            metadata_path = find_metadata_file(file_path, str(metadata_folder_path))
            
            if metadata_path:
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata_map[file_path] = metadata
                    logger.info(f"[Utils] Matched {file_path} -> {metadata_path.name}")
                except Exception as e:
                    logger.warning(f"[Utils] Failed to load {metadata_path}: {e}")
            else:
                logger.warning(f"[Utils] No metadata file found for {file_path}")
    
    logger.info(f"[Utils] Loaded {len(metadata_map)} metadata files")
    return metadata_map

