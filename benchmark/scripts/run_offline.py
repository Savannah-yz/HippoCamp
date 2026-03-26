#!/usr/bin/env python3
"""
Full Offline Pipeline: File -> [Ingest] -> Chunk -> Embed -> VectorDB [-> Profiling]

Features:
- Text-like files (txt, md, csv, json, etc.) bypass captioning, go directly to chunking
- No intermediate I/O by default (data flows in memory)
- Optional save_intermediate for debugging
- Process multiple files from JSON or text file list
- Optional user profiling extraction (--user, --with-profiling)
- Configurable chunk overlap for better context preservation
- File-type chunking router with per-type chunking methods

Usage:
    # Process a single file (no I/O, in-memory)
    python scripts/run_offline.py /path/to/video.mp4

    # Process text file (bypasses captioning)
    python scripts/run_offline.py /path/to/document.txt

    # Process with experiment isolation
    python scripts/run_offline.py /path/to/file --experiment exp_001

    # Process from file list (JSON or text)
    python scripts/run_offline.py files.json --file-list -e exp_001

    # Process with user profiling
    python scripts/run_offline.py /path/to/file --user alice --with-profiling

    # Save intermediate results for debugging
    python scripts/run_offline.py /path/to/file --save-intermediate

    # Process all files in folder
    python scripts/run_offline.py /path/to/folder/ --all

    # Process pre-labeled metadata JSON (skip ingestion)
    python scripts/run_offline.py metadata.json --metadata-json -e exp_001

    # Process with chunk overlap (100 characters overlap between chunks)
    python scripts/run_offline.py /path/to/file --overlap 100

    # Process without overlap (disable overlap)
    python scripts/run_offline.py /path/to/file --no-overlap

    # Override default chunking settings
    python scripts/run_offline.py file.txt --chunk-size 800 --chunk-min-chars 50

    # Override chunking by file type (method/size/overlap)
    python scripts/run_offline.py file.txt \\
      --chunk-method-by-type "csv=line,json=json" \\
      --chunk-size-by-type "csv=120,json=50" \\
      --chunk-overlap-by-type "pdf=100"

Chunking Strategy by File Type:
===============================
Default configs are in configs/pipelines/indexing.yaml (chunking.by_type section).

| File Type      | Method    | chunk_size | overlap | Notes                              |
|----------------|-----------|------------|---------|-------------------------------------|
| pdf            | recursive | 800 tokens | 100     | Page/section -> paragraph recursive |
| docx/doc       | recursive | 700 tokens | 80      | Heading/paragraph recursive         |
| pptx/ppt       | recursive | 600 tokens | 50      | Slide-level, then recursive         |
| md/txt         | recursive | 600 tokens | 80      | Paragraph recursive                 |
| csv            | segment   | max 100kch | -       | Whole file as single chunk          |
| xlsx           | segment   | max 100kch | -       | Whole file as single chunk          |
| sqlite         | line      | 120 lines  | 0       | Row blocks with schema header       |
| json           | segment   | max 100kch | -       | Whole file as single chunk          |
| jsonl          | line      | 100 lines  | 0       | Line-based chunking                 |
| ics            | block     | 1 block    | 0       | Each VEVENT/VTODO as separate chunk |
| eml            | segment   | max 10000ch| -       | Whole email as single chunk         |
| log            | line      | 200 lines  | 0       | Time-window line blocks             |
| ipynb          | json      | 20 items   | 0       | Cell-based (code/markdown)          |
| py/sh          | code      | 10 blocks  | 1       | Function/class blocks               |
| mp3/mp4/mkv    | segment   | max 2000ch | -       | Transcription segments              |
| png/jpg/gif    | segment   | max 1000ch | -       | OCR/caption as single chunk         |
| bin/pkl/npy/pt | segment   | max 2000ch | -       | Metadata/shape as single chunk      |

Chunking Methods:
- recursive: Token-based recursive splitting (chonkie), overlap in characters
- line: Lines per chunk, overlap in lines, supports include_header
- json: Items per chunk for arrays/dicts, overlap in items
- block: Blocks per chunk with delimiter, overlap in blocks
- code: Function/class blocks per chunk, overlap in blocks
- email: Header + body separation, body chunked by lines
- segment: Keep segment intact, split with recursive if exceeds max_chars
"""

import sys
import os
import argparse
import json
import logging
import asyncio
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import yaml
from src.shared.env import load_release_env
load_release_env(project_root)

from src.rag.chunking.router import FileTypeChunkRouter
from src.rag.chunking.base import Chunk
from src.shared.service_factory import SharedServiceFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Text File Extensions (bypass captioning)
# ============================================================================
TEXT_EXTENSIONS = {
    '.txt', '.xml', '.md', '.json', '.jsonl', '.csv', '.log', '.ics', '.eml',
    '.ipynb', '.py', '.sh'
}

# Media extensions (require captioning via Gemini/VLM)
MEDIA_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.webm',  # Video
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',  # Image
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma',  # Audio
    '.pdf', '.doc', '.docx',  # Documents
}


def load_config(config_path: str) -> dict:
    """Load pipeline config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def stable_file_id(path: Path) -> str:
    """Generate a stable, collision-resistant file_id from the full path."""
    resolved = path.resolve()
    h = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}__{h}"


def parse_kv_map(raw: str, value_cast=str) -> Dict[str, Any]:
    """
    Parse "key=value,key2=value2" or JSON dict string into a dict.
    """
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object for key-value overrides")
        return {str(k).lower(): value_cast(v) for k, v in data.items()}
    parts = [p for p in re.split(r"[;,]", text) if p.strip()]
    result: Dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid override entry: {part}")
        key, value = part.split("=", 1)
        result[key.strip().lower()] = value_cast(value.strip())
    return result


def resolve_chunking_config(indexing_config: Dict[str, Any]) -> Dict[str, Any]:
    chunking_config = indexing_config.get("chunking", {}) if indexing_config else {}
    mode = chunking_config.get("mode")
    modes = chunking_config.get("modes", {})
    if mode and isinstance(modes, dict) and mode in modes:
        resolved = dict(modes.get(mode, {}) or {})
        resolved["mode"] = mode
        return resolved
    return dict(chunking_config)


def get_collection_name(base_name: str, experiment_id: str) -> str:
    """Get collection name with experiment suffix."""
    if experiment_id and experiment_id != "default":
        return f"{base_name}_{experiment_id}"
    return base_name


def is_text_file(file_path: Path, config: dict = None) -> bool:
    """Check if file is a text file that bypasses captioning."""
    ext = file_path.suffix.lower()

    # Check config for custom text extensions
    if config:
        text_config = config.get("text", {})
        custom_extensions = text_config.get("extensions", [])
        if custom_extensions:
            return ext in [e.lower() for e in custom_extensions]

    return ext in TEXT_EXTENSIONS


def load_text_file(file_path: Path, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Load text file and convert to metadata format compatible with chunking.

    Returns metadata dict with structure:
    {
        "file_info": {...},
        "summary": "",
        "segments": [{"content": "...", "start_time": 0, "end_time": 0}]
    }

    For JSON files that contain structured metadata (with file_info, segments, etc.),
    this function will extract and preserve the original file_info including file_type.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()

    # For JSON files, try to parse and extract structured metadata
    if file_path.suffix.lower() == '.json':
        try:
            data = json.loads(content)
            # Check if this is a structured metadata JSON (has file_info or segments)
            if isinstance(data, dict) and ("file_info" in data or "segments" in data):
                # Preserve complete original file_info (all fields)
                original_file_info = data.get("file_info", {})
                file_info = dict(original_file_info)  # Copy all original fields

                # Ensure required fields have values (use original or fallback)
                if "file_id" not in file_info:
                    file_info["file_id"] = file_info.get("id") or file_path.stem
                if "file_path" not in file_info:
                    file_info["file_path"] = str(file_path.resolve())
                if "file_type" not in file_info:
                    file_info["file_type"] = _get_file_type(file_path)
                if "file_name" not in file_info:
                    file_info["file_name"] = file_path.name

                # Use segments from JSON if available
                segments = data.get("segments", [])
                if not segments:
                    # Fall back to content field or full JSON as text
                    text_content = data.get("content", content)
                    segments = [{"content": text_content, "start_time": 0, "end_time": 0}]

                return {
                    "file_info": file_info,
                    "summary": data.get("summary", ""),
                    "hash_tags": data.get("hash_tags", []),
                    "segments": segments,
                }
        except json.JSONDecodeError:
            # Not valid JSON, treat as plain text
            pass

    return {
        "file_info": {
            "file_id": file_path.stem,
            "file_path": str(file_path.resolve()),
            "file_type": _get_file_type(file_path),
            "file_name": file_path.name,
        },
        "summary": "",
        "hash_tags": [],
        "segments": [
            {
                "content": content,
                "start_time": 0,
                "end_time": 0,
            }
        ]
    }


def ingest_media_file(
    file_path: Path,
    config: dict,
    save_intermediate: bool = False,
    metadata_dir: str = None,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Ingest media file using Gemini/VLM backend.

    Returns metadata dict in memory (no I/O unless save_intermediate=True).
    """
    from src.rag.pipeline.ingestion import ingest_file

    if save_intermediate:
        # Use ingest_file which saves to disk
        return ingest_file(
            file_path=str(file_path),
            metadata_dir=metadata_dir,
            force_refresh=force,
            num_processes=5,
            config_path=None,
        )
    else:
        # Process in memory without saving
        from src.storage.metadata_storage import MetadataStore
        from src.memory.models.factory import BackendFactory

        file_id = stable_file_id(file_path)

        # Get processor
        try:
            processor = BackendFactory.get_processor(file_path)
            logger.info(f"  Using processor: {processor.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to get processor: {e}")
            return None

        # Process (in memory, no save)
        try:
            metadata = processor.generate(
                file_path=file_path,
                num_processes=5,
                file_id=file_id
            )

            # Add file_info
            file_info = {
                "file_id": file_id,
                "file_path": str(file_path.resolve()),
                "file_type": _get_file_type(file_path),
                "file_name": file_path.name,
            }

            return {
                "file_info": file_info,
                **metadata
            }
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None


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
    if ext in TEXT_EXTENSIONS:
        return ext.lstrip(".")
    return "unknown"


def validate_metadata_dict(data: Dict[str, Any], source: str = "metadata", default_file_id: str = None) -> Dict[str, Any]:
    """
    Validate and normalize a metadata dictionary.

    Supports two formats:
    1. Direct content: {"content": "full text content..."}
    2. Segments array: {"segments": [{"content": "..."}]}

    Args:
        data: Raw metadata dict
        source: Description of source for error messages
        default_file_id: Default file_id if not present in data

    Returns:
        Validated and normalized metadata dict (always has segments array)
    """
    # Support direct content field (convert to segments format)
    if "content" in data and "segments" not in data:
        content = data.pop("content")
        if not content or not isinstance(content, str):
            raise ValueError(f"Metadata 'content' must be a non-empty string: {source}")
        data["segments"] = [{"content": content, "start_time": 0, "end_time": 0}]

    # Validate required fields
    if "segments" not in data or not data["segments"]:
        raise ValueError(f"Metadata must contain 'content' or non-empty 'segments' list: {source}")

    for i, seg in enumerate(data["segments"]):
        if "content" not in seg:
            raise ValueError(f"Segment {i} missing 'content' field in: {source}")

    # Ensure file_info exists with at least file_id
    if "file_info" not in data:
        data["file_info"] = {
            "file_id": default_file_id or "unknown",
            "file_path": "",
            "file_type": "text",
            "file_name": "",
        }
    else:
        # Ensure file_id is set: prioritize "id", then "file_id", then default
        file_info = data["file_info"]
        if "file_id" not in file_info:
            # Use "id" field if present, otherwise use default
            file_info["file_id"] = file_info.get("id") or default_file_id or "unknown"
        # Keep "id" field if it exists (for compatibility)
        # This allows both file_info.id and file_info.file_id to coexist

    # Set defaults for optional fields
    data.setdefault("summary", "")
    data.setdefault("hash_tags", [])

    # Ensure segments have start_time and end_time
    for seg in data["segments"]:
        seg.setdefault("start_time", 0)
        seg.setdefault("end_time", 0)

    return data


def load_metadata_json(json_path: Path) -> Dict[str, Any]:
    """
    Load pre-labeled metadata from JSON file (skip ingestion step).

    Supports multiple formats:

    Format 1: With file_info.id (golden metadata format):
    {
        "file_info": {
            "id": 369,  # Unique file identifier (number or string)
            "user": "Victoria",
            "file_path": "path/to/file.mp3",
            "file_type": "mp3",
            "file_name": "file.mp3"
        },
        "segments": [{"content": "...", "start": "00:00", "end": "00:01"}]
    }

    Format 2: Simple format with file_id:
    {
        "file_info": {"file_id": "doc1", "file_type": "txt"},
        "content": "Your full text content here..."
    }

    Format 3: Segments format:
    {
        "file_info": {"file_id": "doc1"},
        "segments": [{"content": "segment 1"}, {"content": "segment 2"}]
    }

    Format 4: Minimal format:
    {"content": "Your text content..."}

    File ID Resolution (in priority order):
    1. file_info.id (if present)
    2. file_info.file_id (if present)
    3. Auto-generated from JSON filename

    Optional fields:
    - file_info.file_type: use extension-like values (e.g., pdf, csv, json, mp4, md)
    - summary: Summary of the content
    - hash_tags: Array of tags

    Returns:
        Validated metadata dict ready for chunking pipeline
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use the path stem as default file_id
    json_path = Path(json_path)
    data = validate_metadata_dict(data, source=str(json_path), default_file_id=json_path.stem)

    # Fill in file_info from path if missing
    if not data["file_info"].get("file_path"):
        data["file_info"]["file_path"] = str(json_path.resolve())
    if not data["file_info"].get("file_name"):
        data["file_info"]["file_name"] = json_path.name

    logger.info(f"  Loaded metadata JSON: {len(data['segments'])} segments")
    return data


async def process_file(
    file_path: str,
    ingestion_config: dict,
    indexing_config: dict,
    service_factory: SharedServiceFactory,
    experiment_id: str = None,
    force: bool = False,
    save_intermediate: bool = False,
    user: str = None,
    with_profiling: bool = False,
    preloaded_metadata: Dict[str, Any] = None,
    overlap: int = None,
    embed_metadata: bool = False,
    dump_embed_texts: bool = False,
    embed_texts_path: str = None,
) -> bool:
    """
    Process a single file: [Ingest] -> Chunk -> Embed -> VectorDB

    Text files bypass ingestion step.
    No I/O by default unless save_intermediate=True.
    Services (embedder, vectordb) come from SharedServiceFactory.

    Args:
        preloaded_metadata: If provided, skip ingestion and use this metadata directly.
                           Used with --metadata-json option.
    """
    os.chdir(project_root)
    file_path = Path(file_path)

    # When using preloaded metadata, file doesn't need to exist
    if preloaded_metadata is None and not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    # Get file_id from preloaded metadata or file path
    if preloaded_metadata:
        # Try to get file_id from file_info.id first, then file_info.file_id, then fallback to path-hash
        file_info = preloaded_metadata.get("file_info", {})
        file_id = file_info.get("id") or file_info.get("file_id") or stable_file_id(file_path)
    else:
        file_id = stable_file_id(file_path)
    exp_id = experiment_id or indexing_config.get("experiment", {}).get("id", "default")

    # Get save_intermediate from config if not overridden
    if not save_intermediate:
        save_intermediate = indexing_config.get("save_intermediate", {}).get("enabled", False)

    # Calculate total stages based on profiling
    total_stages = 4 if with_profiling else 3

    logger.info(f"{'='*60}")
    logger.info(f"Processing: {file_path.name}")
    logger.info(f"Experiment: {exp_id}")
    logger.info(f"Save intermediate: {save_intermediate}")
    if with_profiling:
        logger.info(f"User profiling: {user}")
    logger.info(f"{'='*60}")

    # ========== Stage 1: Get Metadata ==========
    if preloaded_metadata:
        # Use preloaded metadata (skip ingestion entirely)
        logger.info(f"[Stage 1/{total_stages}] Using preloaded metadata JSON - skipping ingestion")
        metadata = preloaded_metadata
        total_chars = sum(len(s.get("content", "")) for s in metadata.get("segments", []))
        logger.info(f"  Loaded {len(metadata.get('segments', []))} segments ({total_chars} chars)")
    elif is_text_file(file_path, ingestion_config):
        # Text file: bypass captioning, load directly
        logger.info(f"[Stage 1/{total_stages}] Text file detected - bypassing captioning")
        encoding = ingestion_config.get("text", {}).get("encoding", "utf-8")
        metadata = load_text_file(file_path, encoding)
        segments = metadata.get("segments") if isinstance(metadata, dict) else None
        first_content = None
        if isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, dict) and isinstance(seg.get("content"), str) and seg.get("content"):
                    first_content = seg.get("content")
                    break
        if not isinstance(first_content, str):
            logger.error("Text ingestion produced empty content for %s", file_path.name)
            return False
        total_chars = sum(
            len(seg.get("content"))
            for seg in segments
            if isinstance(seg, dict) and isinstance(seg.get("content"), str)
        )
        if segments and isinstance(segments[0], dict) and segments[0].get("content") is None:
            logger.warning("First segment content is None for %s; using first non-empty segment.", file_path.name)
        logger.info(f"  Loaded text file: {total_chars} chars")
    else:
        # Media file: use Gemini/VLM for captioning
        logger.info(f"[Stage 1/{total_stages}] Ingestion: File -> Metadata")
        metadata_dir = ingestion_config.get("save_intermediate", {}).get("metadata_dir", "./data/metadata")

        metadata = ingest_media_file(
            file_path=file_path,
            config=ingestion_config,
            save_intermediate=save_intermediate,
            metadata_dir=metadata_dir,
            force=force,
        )

        if not metadata:
            logger.error(f"Ingestion failed for {file_path.name}")
            return False

        logger.info(f"  Ingestion complete: {len(metadata.get('segments', []))} segments")

    # Ensure file_id is present for downstream chunking and storage
    if isinstance(metadata, dict):
        info = metadata.setdefault("file_info", {})
        info.setdefault("file_id", file_id)

    # Optionally save metadata
    if save_intermediate and not preloaded_metadata and not is_text_file(file_path, ingestion_config):
        save_config = indexing_config.get("save_intermediate", {})
        metadata_dir = save_config.get("metadata_dir", "./data/metadata")
        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        metadata_path = Path(metadata_dir) / f"{file_id}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved metadata: {metadata_path}")

    # ========== Stage 2: Chunking ==========
    logger.info(f"[Stage 2/{total_stages}] Chunking: Metadata -> Chunks")

    chunk_config = resolve_chunking_config(indexing_config)
    if overlap is not None:
        chunk_config["overlap"] = overlap

    router = FileTypeChunkRouter(
        default_config=chunk_config,
        type_configs=chunk_config.get("by_type", {}),
    )
    file_type, effective = router.get_effective_config(metadata.get("file_info", {}))
    logger.info(
        "  File type=%s | method=%s | chunk_size=%s | min_chars=%s | overlap=%s",
        file_type,
        effective.get("method"),
        effective.get("chunk_size"),
        effective.get("min_chars"),
        effective.get("overlap"),
    )
    chunks = router.create_chunks_from_metadata(metadata)

    # Normalize chunks to avoid None content crashes (treat None as empty string)
    def _coerce_chunk(chunk: Optional[Chunk], idx: int) -> Chunk:
        if chunk is None:
            logger.warning("Chunk %d is None for %s; using empty content.", idx, file_path.name)
            empty_id = hashlib.md5(f"{file_id}:empty:{idx}".encode("utf-8")).hexdigest()
            return Chunk(
                content="",
                id=empty_id,
                file_info=dict(metadata.get("file_info", {})),
            )
        if getattr(chunk, "content", None) is None:
            logger.warning("Chunk %d content is None for %s; using empty string.", idx, file_path.name)
            chunk.content = ""
            if chunk.chunk_meta:
                chunk.chunk_meta.char_count = 0
        return chunk

    chunks = [_coerce_chunk(c, i) for i, c in enumerate(chunks)]

    if not chunks:
        logger.warning("No chunks created. Skipping.")
        return True

    logger.info(f"  Created {len(chunks)} chunks")

    # Optionally save chunks (using unified format)
    if save_intermediate:
        save_config = indexing_config.get("save_intermediate", {})
        chunks_dir = save_config.get("chunks_dir", "./data/chunks")
        Path(chunks_dir).mkdir(parents=True, exist_ok=True)
        chunks_path = Path(chunks_dir) / f"{file_id}_chunks.json"
        chunks_data = {
            "file_id": file_id,
            "file_info": metadata.get("file_info", {}),
            "count": len(chunks),
            "chunks": [c.to_dict() for c in chunks]  # Use unified format
        }
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved chunks: {chunks_path}")

    # ========== Stage 3: Embedding + VectorDB ==========
    logger.info(f"[Stage 3/{total_stages}] Embedding + VectorDB")
    logger.info(f"  Profile: {service_factory.profile_name}")

    embedder = service_factory.create_embedder()
    model_name = embedder.get_model_name()

    logger.info(f"  Embedding model: {model_name}")

    if embed_metadata:
        texts = []
        for chunk in chunks:
            parts = []
            file_info = chunk.file_info or {}
            if file_info:
                parts.append(json.dumps({"file_info": file_info}, ensure_ascii=False, sort_keys=True))
            segment_info = chunk.segment_info.to_dict() if chunk.segment_info else {}
            if segment_info:
                parts.append(json.dumps({"segment_info": segment_info}, ensure_ascii=False, sort_keys=True))
            parts.append(chunk.content)
            texts.append("\n\n".join(parts))
    else:
        texts = [chunk.content for chunk in chunks]

    if dump_embed_texts:
        import json as _json
        if embed_texts_path:
            out_path = Path(embed_texts_path)
        else:
            save_dir = indexing_config.get("save_intermediate", {}).get("embeddings_dir", "./data/embeddings")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(save_dir) / f"{file_id}_embed_texts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(_json.dumps({"index": i, "text": text}, ensure_ascii=False))
                f.write("\n")
        logger.info(f"  Saved embed texts: {out_path}")

    try:
        result = await embedder.embed(texts, return_sparse=False)
        dense_embeddings = result.dense_embeddings
        sparse_embeddings = result.sparse_embeddings
        logger.info(f"  Generated {len(dense_embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        await embedder.aclose()
        return False

    # Optionally save embeddings
    if save_intermediate:
        import numpy as np
        save_config = indexing_config.get("save_intermediate", {})
        embeddings_dir = save_config.get("embeddings_dir", "./data/embeddings")
        Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
        embeddings_path = Path(embeddings_dir) / f"{file_id}_embeddings.npz"
        np.savez(embeddings_path,
                 chunk_ids=[c.id for c in chunks],
                 dense_embeddings=dense_embeddings)
        logger.info(f"  Saved embeddings: {embeddings_path}")

    # Vector Store
    store = service_factory.create_vectordb(collection_suffix=exp_id)
    collection_name = f"{service_factory.collections.get('rag', 'research_rag')}_{exp_id}"

    logger.info(f"  Collection: {collection_name}")

    # Check existing if not force
    skip_upsert = False
    if not force:
        chunk_ids = [chunk.id for chunk in chunks]
        existing_ids = await store.filter_existing(chunk_ids)
        if existing_ids:
            logger.info(f"  Skipping {len(existing_ids)} existing chunks")
            new_chunks = [c for c in chunks if c.id not in existing_ids]
            if not new_chunks:
                logger.info("  All chunks already indexed")
                skip_upsert = True
            else:
                # Filter embeddings
                import numpy as np
                keep_indices = np.array([i for i, c in enumerate(chunks) if c.id not in existing_ids])
                dense_embeddings = dense_embeddings[keep_indices]
                if sparse_embeddings:
                    sparse_embeddings = [sparse_embeddings[i] for i in keep_indices]
                chunks = new_chunks

    # Upsert (skip if all chunks already exist)
    if not skip_upsert:
        # Convert chunks to unified metadata format (file_info, segment_info, chunk_meta)
        def chunk_to_metadata(chunk):
            """Convert Chunk object to unified metadata dict for Milvus storage."""
            return {
                "file_info": chunk.file_info,
                "segment_info": chunk.segment_info.to_dict() if chunk.segment_info else {},
                "chunk_meta": chunk.chunk_meta.to_dict() if chunk.chunk_meta else {},
            }

        await store.upsert(
            ids=[chunk.id for chunk in chunks],
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            chunks=[chunk.content for chunk in chunks],
            metadatas=[chunk_to_metadata(chunk) for chunk in chunks],
        )
        logger.info(f"  Indexed {len(chunks)} chunks")

    # Cleanup RAG components
    await embedder.aclose()
    await store.close()

    logger.info(f"[RAG Complete] {file_path.name} -> {collection_name}")

    # ========== Stage 4 (Optional): User Profiling ==========
    if with_profiling and user:
        logger.info(f"[Stage 4/{total_stages}] User Profiling...")

        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.warning("  MONGODB_URI not set, skipping profiling")
        else:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
                from pymongo import MongoClient
                from src.profiling.core.profiling_manager import ProfilingManager
                from src.profiling.types import Metadata

                mongo_client = MongoClient(mongodb_uri)
                async_mongo_client = AsyncIOMotorClient(mongodb_uri)

                # Get text content for profiling
                if preloaded_metadata:
                    # Use content from preloaded metadata
                    segments = metadata.get("segments", [])
                    content = "\n\n".join([s.get("content", "") for s in segments])
                elif is_text_file(file_path, ingestion_config):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                else:
                    # Use segments from metadata
                    segments = metadata.get("segments", [])
                    content = "\n\n".join([s.get("content", "") for s in segments])

                # Create metadata and profiling manager
                # Use file_id as reference when using preloaded metadata
                file_ref = file_id if preloaded_metadata else str(file_path)
                profile_metadata = Metadata.from_text(content, file_path=file_ref)

                profiling_manager = ProfilingManager(
                    userfs=None,
                    mongodb_client=mongo_client,
                    async_mongodb_client=async_mongo_client,
                )
                await profiling_manager.create()

                # Process file for profiling
                await profiling_manager.invoke_metadata(
                    user=user,
                    file_path=str(file_path),
                    metadata=profile_metadata,
                    exp=exp_id,
                )

                await profiling_manager.aclose()
                logger.info(f"  User profiling complete for user: {user}")

            except Exception as e:
                logger.error(f"  Profiling failed: {e}")

    logger.info(f"[Complete] {file_path.name}")
    return True


async def process_files(
    file_paths: List[str],
    ingestion_config: dict,
    indexing_config: dict,
    service_factory: SharedServiceFactory,
    experiment_id: str = None,
    force: bool = False,
    save_intermediate: bool = False,
    user: str = None,
    with_profiling: bool = False,
    preloaded_metadatas: List[Dict[str, Any]] = None,
    overlap: int = None,
    embed_metadata: bool = False,
    dump_embed_texts: bool = False,
    embed_texts_path: str = None,
) -> Dict[str, bool]:
    """Process multiple files.

    Args:
        preloaded_metadatas: If provided, parallel list of metadata dicts for each file_path.
                            Used with --metadata-json option.
        overlap: Overlap size for chunking. If None, uses config value.
    """
    results = {}
    total = len(file_paths)

    # Use tqdm progress bar for multiple items
    use_progress_bar = total > 1
    iterator = enumerate(file_paths)

    if use_progress_bar:
        iterator = tqdm(
            list(iterator),
            desc="Processing",
            unit="item",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

    for i, fp in iterator:

        # Get display name (short) and result key (stable, unique)
        if preloaded_metadatas and preloaded_metadatas[i]:
            name = preloaded_metadatas[i].get("file_info", {}).get("file_id", Path(fp).name)
        else:
            name = Path(fp).name
        # Use full path as key to avoid collisions from duplicate basenames
        result_key = str(fp)

        # Update progress bar description
        if use_progress_bar:
            iterator.set_description(f"Processing {name[:20]}")

        preloaded = preloaded_metadatas[i] if preloaded_metadatas else None
        success = await process_file(
            fp, ingestion_config, indexing_config, service_factory,
            experiment_id, force, save_intermediate,
            user, with_profiling, preloaded, overlap, embed_metadata,
            dump_embed_texts, embed_texts_path
        )

        results[result_key] = success

    return results


def get_supported_extensions() -> set:
    """Get all supported file extensions."""
    return TEXT_EXTENSIONS | MEDIA_EXTENSIONS


def find_files_in_folder(folder: Path) -> List[Path]:
    """Find all supported files in a folder."""
    extensions = get_supported_extensions()
    files = []

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)

    return sorted(files)


def load_file_list(file_list_path: str) -> List[str]:
    """
    Load file paths from a file list.

    Supports:
    - Text file: one path per line (lines starting with # are ignored)
    - JSON file: {"files": ["path1", "path2"]} or ["path1", "path2"]

    Examples:
        # files.txt
        /path/to/video1.mp4
        /path/to/video2.mp4
        # this line is ignored

        # files.json
        {"files": ["/path/to/video1.mp4", "/path/to/video2.mp4"]}

        # or simple array
        ["/path/to/video1.mp4", "/path/to/video2.mp4"]
    """
    file_list_path = Path(file_list_path)

    # JSON format
    if file_list_path.suffix.lower() == '.json':
        with open(file_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support {"files": [...]} or just [...]
        if isinstance(data, dict):
            paths = data.get("files", data.get("paths", []))
        elif isinstance(data, list):
            paths = data
        else:
            raise ValueError(f"Invalid JSON format in {file_list_path}")

        return [str(p) for p in paths if p]

    # Text format (one path per line)
    else:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return paths


def main():
    parser = argparse.ArgumentParser(
        description="Full offline pipeline: File -> VectorDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python scripts/run_offline.py video.mp4 --experiment exp_001

  # Process text file (bypasses captioning)
  python scripts/run_offline.py document.txt --experiment exp_001

  # Process from file list (JSON or text)
  python scripts/run_offline.py files.json --file-list --experiment exp_001
  python scripts/run_offline.py files.txt --file-list --experiment exp_001

  # Process all files in folder
  python scripts/run_offline.py /path/to/folder/ --all --experiment exp_001

  # Save intermediate results for debugging
  python scripts/run_offline.py file.mp4 -e exp_001 --save-intermediate

  # Process pre-labeled metadata JSON (skip ingestion)
  python scripts/run_offline.py metadata.json --metadata-json -e exp_001

  # Process with chunk overlap (100 characters overlap between chunks)
  python scripts/run_offline.py file.txt -e exp_001 --overlap 100

  # Process without overlap (explicitly disable)
  python scripts/run_offline.py file.txt -e exp_001 --no-overlap

  # Override default chunking settings
  python scripts/run_offline.py file.txt -e exp_001 --chunk-size 800 --chunk-min-chars 50

  # Override per-type chunking (method/size/overlap)
  python scripts/run_offline.py file.txt -e exp_001 \\
    --chunk-method-by-type "csv=line,json=json" \\
    --chunk-size-by-type "csv=120,json=50" \\
    --chunk-overlap-by-type "pdf=100"

File list formats:
  # files.txt (one path per line)
  /path/to/video1.mp4
  /path/to/video2.mp4

  # files.json
  {"files": ["/path/to/video1.mp4", "/path/to/video2.mp4"]}

Metadata JSON format (for --metadata-json):
  # Simple format (recommended) - direct content field
  {"content": "Your full text content here..."}

  # With file_info
  {
    "file_info": {"file_id": "doc1", "file_type": "txt"},
    "content": "Your full text content here..."
  }

  # Array of metadata objects
  [
    {"file_info": {"file_id": "doc1"}, "content": "Content 1"},
    {"file_info": {"file_id": "doc2"}, "content": "Content 2"}
  ]
"""
    )
    parser.add_argument("input", help="File path, folder path, or file list")
    parser.add_argument(
        "--ingestion-config",
        default="configs/pipelines/ingestion.yaml",
        help="Ingestion config file"
    )
    parser.add_argument(
        "--indexing-config",
        default="configs/pipelines/indexing.yaml",
        help="Indexing config file"
    )
    parser.add_argument(
        "--experiment", "-e",
        help="Experiment ID for collection isolation"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all supported files in folder"
    )
    parser.add_argument(
        "--file-list", action="store_true",
        help="Input is a file list (JSON or text, see examples)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force reprocessing"
    )
    parser.add_argument(
        "--save-intermediate", "-s", action="store_true",
        help="Save intermediate results (metadata, chunks, embeddings)"
    )
    parser.add_argument(
        "--user", "-u",
        help="User identifier for profiling (requires --with-profiling)"
    )
    parser.add_argument(
        "--with-profiling", "-p", action="store_true",
        help="Enable user profiling extraction (requires --user and MONGODB_URI)"
    )
    parser.add_argument(
        "--services-config",
        default="configs/services.yaml",
        help="Services config file (default: configs/services.yaml)"
    )
    parser.add_argument(
        "--metadata-json", "-m", action="store_true",
        help="Input is a pre-labeled metadata JSON file (skip ingestion, go directly to chunking)"
    )
    parser.add_argument(
        "--overlap", type=int, default=None,
        help="Overlap size for chunking (in characters). Overrides config value. Use 0 to disable overlap."
    )
    parser.add_argument(
        "--no-overlap", action="store_true",
        help="Disable overlap for chunking (equivalent to --overlap 0)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Default chunk_size override (applies to all types unless overridden per-type)."
    )
    parser.add_argument(
        "--chunk-min-chars", type=int, default=None,
        help="Default min_chars override (applies to all types unless overridden per-type)."
    )
    parser.add_argument(
        "--chunk-method", type=str, default=None,
        help="Default chunking method override (e.g. recursive, line, json, block, code, email, segment)."
    )
    parser.add_argument(
        "--chunk-size-by-type", type=str, default=None,
        help="Per-type chunk_size overrides. Format: \"csv=120,json=50\" or JSON dict."
    )
    parser.add_argument(
        "--chunk-overlap-by-type", type=str, default=None,
        help="Per-type overlap overrides. Format: \"pdf=100,md=80\" or JSON dict."
    )
    parser.add_argument(
        "--chunk-min-chars-by-type", type=str, default=None,
        help="Per-type min_chars overrides. Format: \"pdf=150,txt=50\" or JSON dict."
    )
    parser.add_argument(
        "--chunk-method-by-type", type=str, default=None,
        help="Per-type chunking method overrides. Format: \"csv=line,json=json\" or JSON dict."
    )
    parser.add_argument(
        "--embed-metadata", action="store_true",
        help="Prefix each chunk with its file_info when generating embeddings."
    )
    parser.add_argument(
        "--dump-embed-texts", action="store_true",
        help="Dump the exact texts sent to the embedder as JSONL."
    )
    parser.add_argument(
        "--embed-texts-path", type=str, default=None,
        help="Output path for dumped embed texts (JSONL). Defaults to embeddings_dir/{file_id}_embed_texts.jsonl."
    )

    args = parser.parse_args()

    # Handle --no-overlap flag
    if args.no_overlap:
        args.overlap = 0

    # Validate profiling arguments
    if args.with_profiling and not args.user:
        parser.error("--with-profiling requires --user to be specified")

    # Load configs
    ingestion_config = load_config(args.ingestion_config)
    indexing_config = load_config(args.indexing_config)

    # Apply chunking overrides from CLI
    try:
        chunking_root = indexing_config.setdefault("chunking", {})
        mode = chunking_root.get("mode")
        modes = chunking_root.get("modes", {})
        if mode and isinstance(modes, dict) and mode in modes:
            chunk_config = modes.setdefault(mode, {})
        else:
            chunk_config = chunking_root

        if args.chunk_size is not None:
            chunk_config["chunk_size"] = args.chunk_size
        if args.chunk_min_chars is not None:
            chunk_config["min_chars"] = args.chunk_min_chars
        if args.chunk_method:
            chunk_config["type"] = args.chunk_method
        if args.overlap is not None:
            chunk_config["overlap"] = args.overlap

        by_type = chunk_config.setdefault("by_type", {})
        for ftype, value in parse_kv_map(args.chunk_size_by_type, int).items():
            by_type.setdefault(ftype, {})["chunk_size"] = value
        for ftype, value in parse_kv_map(args.chunk_overlap_by_type, int).items():
            by_type.setdefault(ftype, {})["overlap"] = value
        for ftype, value in parse_kv_map(args.chunk_min_chars_by_type, int).items():
            by_type.setdefault(ftype, {})["min_chars"] = value
        for ftype, value in parse_kv_map(args.chunk_method_by_type, str).items():
            by_type.setdefault(ftype, {})["method"] = value
    except ValueError as e:
        parser.error(str(e))

    # Load services (with profile support)
    service_factory = SharedServiceFactory.from_yaml(args.services_config)
    logger.info(f"Using service profile: {service_factory.profile_name}")

    input_path = Path(args.input)

    # Determine files to process
    preloaded_metadatas = None

    if args.metadata_json:
        # Load pre-labeled metadata JSON file(s)
        if input_path.suffix.lower() == '.json':
            # Single JSON file or array of metadatas in one file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's a list of metadata objects or a single one
            if isinstance(data, list):
                # Array of metadata objects in one file
                preloaded_metadatas = []
                file_paths = []
                for i, meta in enumerate(data):
                    try:
                        validated = validate_metadata_dict(
                            meta,
                            source=f"{input_path}[{i}]",
                            default_file_id=f"item_{i}"
                        )
                        preloaded_metadatas.append(validated)
                        # Use file_id as placeholder path
                        file_paths.append(validated.get("file_info", {}).get("file_id", f"item_{i}"))
                    except ValueError as e:
                        logger.error(f"Invalid metadata at index {i}: {e}")
                        sys.exit(1)
                logger.info(f"Loaded {len(preloaded_metadatas)} metadata objects from JSON array")
            else:
                # Single metadata object
                metadata = load_metadata_json(input_path)
                preloaded_metadatas = [metadata]
                file_paths = [str(input_path)]
                logger.info("Loaded 1 metadata JSON file")
        else:
            logger.error("--metadata-json requires a .json file as input")
            sys.exit(1)
    elif args.file_list:
        file_paths = load_file_list(args.input)
        logger.info(f"Loaded {len(file_paths)} files from list")
    elif input_path.is_dir() or args.all:
        files = find_files_in_folder(input_path)
        file_paths = [str(f) for f in files]
        logger.info(f"Found {len(file_paths)} supported files in {input_path}")
    else:
        file_paths = [str(input_path)]

    if not file_paths:
        logger.error("No files to process")
        sys.exit(1)

    # Process
    results = asyncio.run(process_files(
        file_paths,
        ingestion_config,
        indexing_config,
        service_factory,
        args.experiment,
        args.force,
        args.save_intermediate,
        args.user,
        args.with_profiling,
        preloaded_metadatas,
        args.overlap,
        args.embed_metadata,
        args.dump_embed_texts,
        args.embed_texts_path,
    ))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success_count = sum(results.values())
    total = len(results)

    print(f"Profile: {service_factory.profile_name}")
    if args.experiment:
        print(f"Experiment: {args.experiment}")
    print(f"Processed: {success_count}/{total} files")
    print(f"Save intermediate: {args.save_intermediate}")
    print(f"Embed metadata: {args.embed_metadata}")
    print(f"Dump embed texts: {args.dump_embed_texts}")
    if args.with_profiling:
        print(f"User profiling: {args.user}")
    print()

    for path_str, success in results.items():
        status = "OK" if success else "FAILED"
        name = Path(path_str).name
        print(f"  [{status}] {name} | {path_str}")

    print(f"{'='*60}")

    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
