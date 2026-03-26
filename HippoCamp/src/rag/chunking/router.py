from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import BaseChunker, Chunk, SegmentInfo, ChunkMeta
from .recursive import ChonkieRecursiveChunker
from .semantic import SemanticChunker


def _hash_id(content: str, file_id: str = "") -> str:
    combined = f"{file_id}:{content}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _normalize_file_type(value: str) -> str:
    return value.lower().strip().lstrip(".")


def _parse_time_value(value: Any) -> float | None:
    """
    Parse time values that may be numeric or string timestamps like HH:MM:SS or MM:SS.
    Returns seconds as float, or None if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try numeric string
        try:
            return float(text)
        except ValueError:
            pass
        # Try timestamp formats HH:MM:SS or MM:SS
        parts = text.split(":")
        if len(parts) in (2, 3) and all(p.strip().isdigit() for p in parts):
            nums = [int(p) for p in parts]
            if len(nums) == 2:
                minutes, seconds = nums
                return float(minutes * 60 + seconds)
            hours, minutes, seconds = nums
            return float(hours * 3600 + minutes * 60 + seconds)
    return None


class FileTypeChunkRouter(BaseChunker):
    """
    Route chunking by file_type with method-specific behavior.

    All methods now output unified Chunk format with:
    - file_info: complete file metadata
    - segment_info: segment indices, page numbers, time ranges
    - chunk_meta: type, chunk_index, char_count
    """

    def __init__(self, default_config: Dict[str, Any], type_configs: Dict[str, Any] | None = None):
        self.default_config = default_config or {}
        self.type_configs = {str(k).lower(): v for k, v in (type_configs or {}).items()}

    def get_effective_config(self, file_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        file_type = self._resolve_file_type(file_info)
        base = dict(self.default_config)
        base.pop("by_type", None)
        method = base.get("method") or base.get("type") or "recursive"
        effective = {**base}

        if file_type in self.type_configs:
            effective.update(self.type_configs[file_type] or {})

        if "method" not in effective and "type" in effective:
            effective["method"] = effective["type"]
        effective.setdefault("method", method)
        return file_type, effective

    def create_chunks_from_metadata(self, file_data: Dict[str, Any]) -> List[Chunk]:
        """
        Create chunks from metadata with unified format.

        All chunks will have:
        - file_info: complete file metadata from input
        - segment_info: segment indices, page numbers (PDF), time ranges (MP3)
        - chunk_meta: type, chunk_index, char_count
        """
        file_info = file_data.get("file_info", {})
        file_info_copy = dict(file_info)

        # Get file_id for hash generation (use "id" field, don't add redundant "file_id")
        file_id = str(file_info.get("id", file_info.get("file_id", "unknown")))
        file_type = file_info_copy.get("file_type", "text")

        _, effective = self.get_effective_config(file_info)
        method = str(effective.get("method", "recursive")).lower()

        # For recursive method, delegate to ChonkieRecursiveChunker
        if method == "recursive":
            chunker = ChonkieRecursiveChunker(
                chunk_size=int(effective.get("chunk_size")),
                min_chars=int(effective.get("min_chars")),
                overlap=int(effective.get("overlap")),
            )
            return chunker.create_chunks_from_metadata(file_data)

        # For other methods, handle with unified format
        target_metadata = self._get_target_metadata(file_data)
        chunks: List[Chunk] = []
        chunk_global_index = 0

        # Process summary
        summary = target_metadata.get("summary", "")
        if summary and summary.strip():
            summary_text = summary.strip()
            chunk = Chunk(
                id=_hash_id(summary_text, file_id),
                content=summary_text,
                file_info=dict(file_info_copy),
                segment_info=SegmentInfo(segment_indices=[-1]),
                chunk_meta=ChunkMeta(
                    type="summary",
                    chunk_index=chunk_global_index,
                    char_count=len(summary_text)
                )
            )
            chunks.append(chunk)
            chunk_global_index += 1

        # Process segments
        segments = target_metadata.get("segments", [])

        if method == "file":
            combined_parts: List[str] = []
            segment_indices: List[int] = []
            page_numbers: List[Any] = []
            time_ranges: List[Dict[str, float]] = []

            for seg_idx, segment in enumerate(segments):
                text = segment.get("content", "")
                if not text or not text.strip():
                    continue
                segment_indices.append(seg_idx)
                info = self._extract_segment_info(segment, seg_idx, file_type)
                if info.get("page_numbers"):
                    page_numbers.extend(info.get("page_numbers", []))
                if info.get("time_ranges"):
                    time_ranges.extend(info.get("time_ranges", []))
                combined_parts.append(text.strip())

            combined_text = "\n\n".join(combined_parts).strip()
            if combined_text:
                max_chars = int(effective.get("max_chars") or 0)
                min_chars = int(effective.get("min_chars") or 1)
                overlap = int(effective.get("overlap") or 0)
                token_limit = effective.get("token_limit")
                token_char_ratio = float(effective.get("token_char_ratio") or 2.7)
                token_encoding = effective.get("token_encoding")

                should_split = max_chars > 0 and len(combined_text) > max_chars
                if should_split:
                    extra_kwargs: Dict[str, Any] = {
                        "token_limit": token_limit,
                        "token_char_ratio": token_char_ratio,
                    }
                    if token_encoding:
                        extra_kwargs["token_encoding"] = str(token_encoding)

                    splitter = SemanticChunker(
                        max_chars=max_chars,
                        min_chars=min_chars,
                        overlap_chars=overlap,
                        section_split_file_types=effective.get("section_split_file_types"),
                        section_split_all=bool(effective.get("section_split_all", False)),
                        **extra_kwargs,
                    )
                    parts = splitter.chunk_text(combined_text, file_type=file_type)
                else:
                    parts = [combined_text]

                for part in parts:
                    part_text = part.strip()
                    if not part_text:
                        continue
                    chunk = Chunk(
                        id=_hash_id(part_text, file_id),
                        content=part_text,
                        file_info=dict(file_info_copy),
                        segment_info=SegmentInfo(
                            segment_indices=segment_indices,
                            page_numbers=page_numbers,
                            time_ranges=time_ranges,
                        ),
                        chunk_meta=ChunkMeta(
                            type="content",
                            chunk_index=chunk_global_index,
                            char_count=len(part_text),
                        ),
                    )
                    chunks.append(chunk)
                    chunk_global_index += 1
            return chunks

        for seg_idx, segment in enumerate(segments):
            text = segment.get("content", "")
            if not text or not text.strip():
                continue

            # Extract segment-specific info
            segment_info = self._extract_segment_info(segment, seg_idx, file_type)

            # Chunk the text using the specified method
            chunk_entries = self._chunk_text(text, method, effective, file_info)

            for local_idx, entry in enumerate(chunk_entries):
                content = entry.get("content", "")
                chunk_content = content.strip()
                min_chars = effective.get("min_chars")
                if min_chars is not None and len(chunk_content) < int(min_chars):
                    continue

                chunk = Chunk(
                    id=_hash_id(chunk_content, file_id),
                    content=chunk_content,
                    file_info=dict(file_info_copy),
                    segment_info=SegmentInfo(
                        segment_indices=segment_info["segment_indices"],
                        page_numbers=segment_info.get("page_numbers", []),
                        time_ranges=segment_info.get("time_ranges", [])
                    ),
                    chunk_meta=ChunkMeta(
                        type="content",
                        chunk_index=chunk_global_index,
                        char_count=len(chunk_content)
                    )
                )
                chunks.append(chunk)
                chunk_global_index += 1

        return chunks

    def _extract_segment_info(
        self, segment: Dict[str, Any], seg_idx: int, file_type: str
    ) -> Dict[str, Any]:
        """Extract segment-specific info based on file type."""
        info: Dict[str, Any] = {"segment_indices": [seg_idx]}

        # MP3/audio: extract start/end time
        if file_type == "mp3" or "start" in segment or "end" in segment:
            start = segment.get("start")
            end = segment.get("end")
            if start is not None or end is not None:
                time_range: Dict[str, float] = {}
                start_val = _parse_time_value(start)
                end_val = _parse_time_value(end)
                if start_val is not None:
                    time_range["start"] = start_val
                if end_val is not None:
                    time_range["end"] = end_val
                info["time_ranges"] = [time_range]

        # PDF/DOCX: extract page_number
        if file_type in ("pdf", "docx") or "page_number" in segment or "page" in segment:
            page_num = segment.get("page_number") or segment.get("page")
            if page_num is not None:
                info["page_numbers"] = [page_num]

        return info

    def _get_target_metadata(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get the metadata containing segments."""
        if "metadata" in file_data and isinstance(file_data["metadata"], dict) and file_data["metadata"]:
            return file_data["metadata"]
        return file_data

    def _resolve_file_type(self, file_info: Dict[str, Any]) -> str:
        candidates: List[str] = []
        file_type = _normalize_file_type(file_info.get("file_type", ""))
        if file_type:
            candidates.append(file_type)

        file_name = file_info.get("file_name", "") or ""
        file_path = file_info.get("file_path", "") or ""
        ext = Path(file_name or file_path).suffix.lower().lstrip(".")
        if ext:
            candidates.append(ext)

        for candidate in candidates:
            if candidate in self.type_configs:
                return candidate

        return file_type or ext or "text"

    def _chunk_text(
        self,
        text: str,
        method: str,
        config: Dict[str, Any],
        file_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if method == "line":
            return [{"content": c} for c in self._chunk_by_lines(text, config)]
        if method == "json":
            return [{"content": c} for c in self._chunk_by_json(text, config)]
        if method == "block":
            return [{"content": c} for c in self._chunk_by_blocks(text, config)]
        if method == "code":
            language = _normalize_file_type(file_info.get("file_type", ""))
            return [{"content": c} for c in self._chunk_code(text, config, language)]
        if method == "email":
            return self._chunk_email(text, config)
        if method == "semantic":
            return [{"content": c} for c in self._chunk_semantic(text, config, file_info)]
        if method == "ics":
            return [{"content": c} for c in self._chunk_ics(text)]
        if method == "segment":
            max_chars = config.get("max_chars")
            if max_chars and int(max_chars) > 0 and len(text) > int(max_chars):
                return [{"content": c} for c in self._split_with_recursive(text, config)]
            return [{"content": text}]
        if method == "file":
            return [{"content": text}]
        return [{"content": text}]

    def _chunk_by_lines(self, text: str, config: Dict[str, Any]) -> List[str]:
        lines = text.splitlines()
        if not lines:
            return []
        lines_per_chunk = max(1, int(config.get("chunk_size")))
        overlap = max(0, int(config.get("overlap")))
        step = max(1, lines_per_chunk - overlap)
        include_header = bool(config.get("include_header", False))
        header_lines = max(0, int(config.get("header_lines", 1)))
        header = lines[:header_lines] if include_header and header_lines > 0 else []

        chunks: List[str] = []
        for start in range(0, len(lines), step):
            end = min(len(lines), start + lines_per_chunk)
            chunk_lines = lines[start:end]
            if include_header and start > 0 and header:
                chunk_lines = header + chunk_lines
            if chunk_lines:
                chunks.append("\n".join(chunk_lines))
        return chunks

    def _chunk_by_json(self, text: str, config: Dict[str, Any]) -> List[str]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._chunk_by_lines(text, config)

        items_per_chunk = max(1, int(config.get("chunk_size")))
        overlap = max(0, int(config.get("overlap")))
        step = max(1, items_per_chunk - overlap)
        chunks: List[str] = []

        if isinstance(data, list):
            for start in range(0, len(data), step):
                end = min(len(data), start + items_per_chunk)
                chunk_obj = data[start:end]
                chunks.append(json.dumps(chunk_obj, ensure_ascii=False, indent=2))
            return chunks

        if isinstance(data, dict):
            items = list(data.items())
            for start in range(0, len(items), step):
                end = min(len(items), start + items_per_chunk)
                chunk_obj = {k: v for k, v in items[start:end]}
                chunks.append(json.dumps(chunk_obj, ensure_ascii=False, indent=2))
            return chunks

        return [str(data)]

    def _chunk_by_blocks(self, text: str, config: Dict[str, Any]) -> List[str]:
        delimiter = config.get("delimiter")
        if not delimiter:
            return self._chunk_by_lines(text, config)

        lines = text.splitlines()
        blocks: List[str] = []
        current: List[str] = []
        for line in lines:
            if line.strip().startswith(str(delimiter)) and current:
                blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        blocks_per_chunk = max(1, int(config.get("chunk_size")))
        overlap = max(0, int(config.get("overlap")))
        step = max(1, blocks_per_chunk - overlap)

        chunks: List[str] = []
        for start in range(0, len(blocks), step):
            end = min(len(blocks), start + blocks_per_chunk)
            chunk_blocks = blocks[start:end]
            chunks.append("\n".join(chunk_blocks))
        return chunks

    def _chunk_code(self, text: str, config: Dict[str, Any], language: str) -> List[str]:
        lines = text.splitlines()
        if not lines:
            return []

        if language == "py":
            pattern = re.compile(r"^\s*(def|class)\s+\w+")
        else:
            pattern = re.compile(r"^\s*(function\s+\w+|\w+\s*\(\)\s*\{)")

        blocks: List[str] = []
        current: List[str] = []
        for line in lines:
            if pattern.match(line) and current:
                blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        if len(blocks) <= 1:
            return self._chunk_by_lines(text, config)

        blocks_per_chunk = max(1, int(config.get("chunk_size")))
        overlap = max(0, int(config.get("overlap")))
        step = max(1, blocks_per_chunk - overlap)
        chunks: List[str] = []
        for start in range(0, len(blocks), step):
            end = min(len(blocks), start + blocks_per_chunk)
            chunks.append("\n".join(blocks[start:end]))
        return chunks

    def _chunk_email(self, text: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        parts = text.split("\n\n", 1)
        header = parts[0].strip() if parts else ""
        body = parts[1].strip() if len(parts) > 1 else ""

        entries: List[Dict[str, Any]] = []
        if header:
            entries.append({"content": header})
        if body:
            body_config = dict(config)
            body_chunks = self._chunk_by_lines(body, body_config)
            for chunk in body_chunks:
                entries.append({"content": chunk})
        return entries

    def _split_with_recursive(self, text: str, config: Dict[str, Any]) -> List[str]:
        chunker = ChonkieRecursiveChunker(
            chunk_size=int(config.get("chunk_size")),
            min_chars=int(config.get("min_chars")),
            overlap=int(config.get("overlap")),
        )
        chonkie_chunks = chunker.chunker.chunk(text)
        if chunker.overlap_refinery and chonkie_chunks:
            chonkie_chunks = chunker.overlap_refinery(chonkie_chunks)
        return [c.text for c in chonkie_chunks] if chonkie_chunks else [text]

    def _chunk_semantic(self, text: str, config: Dict[str, Any], file_info: Dict[str, Any]) -> List[str]:
        max_chars = config.get("max_chars")
        min_chars = config.get("min_chars")
        overlap_chars = config.get("overlap")
        if max_chars is None or min_chars is None or overlap_chars is None:
            raise ValueError("Semantic chunking requires max_chars, min_chars, and overlap in config.")

        section_types = config.get("section_split_file_types", [])
        section_split_all = bool(config.get("section_split_all", False))
        file_type = _normalize_file_type(file_info.get("file_type", ""))
        token_limit = config.get("token_limit")
        token_char_ratio = config.get("token_char_ratio")
        token_encoding = config.get("token_encoding")
        extra_kwargs: Dict[str, Any] = {}
        if token_limit is not None:
            extra_kwargs["token_limit"] = int(token_limit)
        if token_char_ratio is not None:
            extra_kwargs["token_char_ratio"] = float(token_char_ratio)
        if token_encoding:
            extra_kwargs["token_encoding"] = str(token_encoding)
        chunker = SemanticChunker(
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            overlap_chars=int(overlap_chars),
            section_split_file_types=section_types,
            section_split_all=section_split_all,
            **extra_kwargs,
        )
        return chunker.chunk_text(text, file_type=file_type)

    def _chunk_ics(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        events: List[str] = []
        current: List[str] = []
        in_event = False

        for line in text.splitlines():
            if line.strip().startswith("BEGIN:VEVENT"):
                in_event = True
                current = [line]
                continue
            if in_event:
                current.append(line)
                if line.strip().startswith("END:VEVENT"):
                    event_text = "\n".join(current).strip()
                    events.append(event_text)
                    current = []
                    in_event = False

        if not events:
            return [text]

        chunks: List[str] = []
        for event in events:
            summary_match = re.search(r"^SUMMARY:(.*)$", event, flags=re.MULTILINE)
            summary = summary_match.group(1).strip() if summary_match else ""
            title_line = f"SUMMARY: {summary}".strip()
            if title_line and not event.lstrip().startswith("SUMMARY:"):
                chunk_text = f"{title_line}\n{event}"
            else:
                chunk_text = event
            chunks.append(chunk_text)

        return chunks
