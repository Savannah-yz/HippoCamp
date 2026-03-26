from __future__ import annotations

import logging
import re
from typing import List, Tuple, Optional, Dict, Any

import tiktoken

from .types import SemanticBlock

logger = logging.getLogger(__name__)

_DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


def _get_tiktoken_encoder(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        try:
            return tiktoken.encoding_for_model(name)
        except Exception as exc:
            raise ValueError(f"Unknown tiktoken encoding/model: {name}") from exc

# Patterns for structure detection
SECTION_PATTERN = re.compile(r"^(#+)\s+(.*)$", flags=re.MULTILINE)
LIST_PATTERN = re.compile(r"^\s*([\-*+]|\d+\.)\s+", flags=re.MULTILINE)
SENTENCE_END_PATTERN = re.compile(r"([.!?。！？；;]+)\s+")
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$", flags=re.MULTILINE)
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
LIST_ITEM_PATTERN = re.compile(r"^(\s*)([-*+]|\d+\.)\s+", flags=re.MULTILINE)


class SemanticSplitter:
    """
    Handles the core logic of splitting text into semantic blocks and merging/splitting them
    into chunks based on size constraints.
    """

    def __init__(self, char_ratio: int = 4):
        self.char_ratio = char_ratio

    def semantic_chunk(
        self,
        text: str,
        max_chars: int,
        min_chars: int,
        overlap_chars: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Perform semantic-aware chunking on text.

        Strategy:
        1. Parse text into semantic blocks (paragraphs, lists, code blocks, tables)
        2. Merge small adjacent blocks
        3. Split large blocks at sentence boundaries
        4. Add overlapping context at chunk boundaries
        """
        if not text:
            return []

        # Parse into semantic blocks
        blocks = self._parse_semantic_blocks(text)

        if not blocks:
            return [(text, 0, len(text))]

        # Merge and split blocks to meet size constraints
        chunks: List[Tuple[str, int, int]] = []
        current_chunk = ""
        current_start = 0

        for block in blocks:
            block_text = block.text
            block_len = len(block_text)

            # If block itself exceeds max size, split it
            if block_len > max_chars:
                # First, add current accumulated chunk if exists
                if current_chunk.strip():
                    chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_chunk = ""

                # Split the large block
                sub_chunks = self._split_large_block(block, max_chars, min_chars)
                for sub_text, sub_rel_start, sub_rel_end in sub_chunks:
                    chunks.append((sub_text, block.start_offset + sub_rel_start, block.start_offset + sub_rel_end))

                # Reset current_start for next chunk
                if chunks:
                    current_start = chunks[-1][2]
                continue

            # Check if adding this block would exceed max size
            if current_chunk and len(current_chunk) + len(block_text) + 2 > max_chars:
                # Save current chunk if it meets minimum size
                if len(current_chunk.strip()) >= min_chars:
                    chunks.append((current_chunk, current_start, current_start + len(current_chunk)))

                    # Add overlap: take last part of current chunk as context for next
                    overlap_text = self._get_overlap_text(current_chunk, overlap_chars)
                    current_chunk = overlap_text + "\n\n" + block_text if overlap_text else block_text
                    current_start = block.start_offset - len(overlap_text) if overlap_text else block.start_offset
                else:
                    # Current chunk too small: fill with a prefix of this block without exceeding max
                    remaining = max_chars - len(current_chunk) - 2
                    if remaining <= 0:
                        chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                        current_chunk = block_text
                        current_start = block.start_offset
                    else:
                        prefix, suffix, prefix_len = self._split_prefix(block_text, remaining)
                        if prefix.strip():
                            combined = current_chunk + "\n\n" + prefix
                            chunks.append((combined, current_start, current_start + len(combined)))
                            if suffix.strip():
                                current_chunk = suffix
                                current_start = block.start_offset + prefix_len
                                if current_chunk.startswith("\n"):
                                    current_chunk = current_chunk[1:]
                                    current_start += 1
                            else:
                                current_chunk = ""
                        else:
                            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                            current_chunk = block_text
                            current_start = block.start_offset
            else:
                # Add block to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + block_text
                else:
                    current_chunk = block_text
                    current_start = block.start_offset

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))

        return chunks if chunks else [(text, 0, len(text))]

    def _parse_semantic_blocks(self, text: str) -> List[SemanticBlock]:
        """
        Parse text into semantic blocks: paragraphs, lists, code blocks, tables.
        """
        blocks: List[SemanticBlock] = []

        # Find code blocks first (they should not be split)
        code_blocks = list(CODE_BLOCK_PATTERN.finditer(text))
        code_ranges = [(m.start(), m.end()) for m in code_blocks]

        # Find table regions
        table_ranges = self._find_table_ranges(text)

        # Combine protected ranges (code + tables)
        protected_ranges = sorted(code_ranges + table_ranges, key=lambda x: x[0])

        # Process text, respecting protected ranges
        current_pos = 0

        while current_pos < len(text):
            # Check if we're in a protected range
            in_protected = False
            for start, end in protected_ranges:
                if start <= current_pos < end:
                    # Add the protected block
                    block_text = text[start:end]
                    block_type = "code" if (start, end) in code_ranges else "table"
                    blocks.append(
                        SemanticBlock(
                            text=block_text,
                            block_type=block_type,
                            start_offset=start,
                            end_offset=end,
                        )
                    )
                    current_pos = end
                    in_protected = True
                    break
                if current_pos < start:
                    # Process text up to the protected range
                    segment = text[current_pos:start]
                    blocks.extend(self._parse_paragraphs_and_lists(segment, current_pos))
                    current_pos = start
                    in_protected = True
                    break

            if not in_protected:
                # Find next protected range or end of text
                next_protected = len(text)
                for start, end in protected_ranges:
                    if start > current_pos:
                        next_protected = start
                        break

                # Process remaining text
                segment = text[current_pos:next_protected]
                blocks.extend(self._parse_paragraphs_and_lists(segment, current_pos))
                current_pos = next_protected

        return blocks

    def _parse_paragraphs_and_lists(self, text: str, base_offset: int) -> List[SemanticBlock]:
        """
        Parse text into paragraphs and lists.
        Lists are kept together as single blocks.
        """
        blocks: List[SemanticBlock] = []

        if not text.strip():
            return blocks

        # Split by paragraph breaks
        parts = PARAGRAPH_SPLIT.split(text)

        current_pos = 0
        i = 0
        while i < len(parts):
            part = parts[i]
            if not part.strip():
                current_pos += len(part) + 2  # +2 for the \n\n
                i += 1
                continue

            # Find actual position in original text
            part_start = text.find(part, current_pos)
            if part_start == -1:
                part_start = current_pos

            # Check if this is a list item
            if LIST_ITEM_PATTERN.match(part):
                # Collect consecutive list items
                list_parts = [part]
                j = i + 1
                while j < len(parts) and LIST_ITEM_PATTERN.match(parts[j].strip() if parts[j] else ""):
                    list_parts.append(parts[j])
                    j += 1

                list_text = "\n\n".join(list_parts)
                list_end = part_start + len(list_text)

                blocks.append(
                    SemanticBlock(
                        text=list_text,
                        block_type="list",
                        start_offset=base_offset + part_start,
                        end_offset=base_offset + list_end,
                    )
                )

                i = j
                current_pos = list_end
            else:
                # Regular paragraph
                blocks.append(
                    SemanticBlock(
                        text=part,
                        block_type="paragraph",
                        start_offset=base_offset + part_start,
                        end_offset=base_offset + part_start + len(part),
                    )
                )
                i += 1
                current_pos = part_start + len(part)

        return blocks

    def _find_table_ranges(self, text: str) -> List[Tuple[int, int]]:
        """
        Find markdown table ranges in text.
        Tables start with a row containing | and continue until non-table lines.
        """
        ranges: List[Tuple[int, int]] = []
        lines = text.split("\n")

        in_table = False
        table_start = 0
        current_pos = 0

        for line in lines:
            line_stripped = line.strip()
            is_table_open = line_stripped == "[TABLE]"
            is_table_close = line_stripped == "[/TABLE]"
            is_table_row = line_stripped.startswith("|") and "|" in line_stripped[1:]
            is_separator = "---" in line_stripped and "|" in line_stripped

            if is_table_open:
                if not in_table:
                    table_start = current_pos
                    in_table = True
            elif is_table_close:
                if in_table:
                    ranges.append((table_start, current_pos + len(line) + 1))
                    in_table = False
            elif is_table_row or (in_table and is_separator):
                if not in_table:
                    table_start = current_pos
                    in_table = True
            else:
                if in_table:
                    # End of table
                    ranges.append((table_start, current_pos))
                    in_table = False

            current_pos += len(line) + 1  # +1 for newline

        # Handle table at end of text
        if in_table:
            ranges.append((table_start, len(text)))

        return ranges

    def _split_large_block(
        self,
        block: SemanticBlock,
        max_chars: int,
        min_chars: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Split a large semantic block into smaller chunks.
        For code/tables: split at natural boundaries (blank lines, etc.)
        For paragraphs/lists: split at sentence boundaries
        """
        text = block.text

        if block.block_type in ("code", "table"):
            # Split code/tables at blank lines within them
            chunks = self._split_at_blank_lines(text, max_chars, min_chars)
            if any(len(chunk[0]) > max_chars for chunk in chunks):
                return self._split_by_length(text, max_chars, min_chars)
            return chunks

        # For paragraphs and lists, split at sentence boundaries
        chunks = self._split_at_sentences(text, max_chars, min_chars)
        if any(len(chunk[0]) > max_chars for chunk in chunks):
            return self._split_by_length(text, max_chars, min_chars)
        return chunks

    def _split_at_sentences(
        self,
        text: str,
        max_chars: int,
        min_chars: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Split text at sentence boundaries.
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        chunks: List[Tuple[str, int, int]] = []
        current_chunk = ""
        current_start = 0

        for sentence, sent_start, sent_end in sentences:
            if not sentence.strip():
                continue

            # If adding this sentence exceeds max, save current and start new
            if current_chunk and len(current_chunk) + len(sentence) > max_chars:
                if len(current_chunk.strip()) >= min_chars:
                    chunks.append((current_chunk.strip(), current_start, current_start + len(current_chunk.rstrip())))
                    current_chunk = sentence
                    current_start = sent_start
                else:
                    # Too small, keep adding
                    current_chunk += sentence
            else:
                if not current_chunk:
                    current_start = sent_start
                current_chunk += sentence

        # Last chunk
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), current_start, current_start + len(current_chunk.rstrip())))

        return chunks if chunks else [(text, 0, len(text))]

    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences while preserving positions.
        Handles Chinese and English sentence endings.
        """
        sentences: List[Tuple[str, int, int]] = []

        # Pattern for sentence endings (including Chinese punctuation)
        pattern = re.compile(r"([.!?。！？；]+[\s\n]*)")

        last_end = 0
        for match in pattern.finditer(text):
            sentence_end = match.end()
            sentence = text[last_end:sentence_end]
            if sentence.strip():
                sentences.append((sentence, last_end, sentence_end))
            last_end = sentence_end

        # Remaining text
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                sentences.append((remaining, last_end, len(text)))

        # If no sentence breaks found, return whole text as one sentence
        if not sentences:
            sentences.append((text, 0, len(text)))

        return sentences

    def _split_at_blank_lines(
        self,
        text: str,
        max_chars: int,
        min_chars: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Split at blank lines (for code blocks and tables).
        """
        parts = re.split(r"(\n\s*\n)", text)

        chunks: List[Tuple[str, int, int]] = []
        current_chunk = ""
        current_start = 0
        current_pos = 0

        for part in parts:
            if current_chunk and len(current_chunk) + len(part) > max_chars:
                if len(current_chunk.strip()) >= min_chars:
                    chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_chunk = part
                    current_start = current_pos
                else:
                    current_chunk += part
            else:
                if not current_chunk:
                    current_start = current_pos
                current_chunk += part
            current_pos += len(part)

        if current_chunk.strip():
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))

        return chunks if chunks else [(text, 0, len(text))]

    def _split_by_length(
        self,
        text: str,
        max_chars: int,
        min_chars: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks of at most max_chars.
        Prefer breaking at newlines, then pipes, then spaces.
        """
        if not text:
            return [(text, 0, 0)]

        chunks: List[Tuple[str, int, int]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(text_len, start + max_chars)
            if end < text_len:
                cut = text.rfind("\n", start, end)
                if cut <= start:
                    cut = text.rfind("|", start, end)
                if cut <= start:
                    cut = text.rfind(" ", start, end)
                if cut <= start:
                    cut = end
                if (cut - start) < min_chars:
                    cut = end
                end = cut

            if end <= start:
                end = min(text_len, start + max_chars)

            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk, start, end))

            start = end
            if start < text_len and text[start] == "\n":
                start += 1

        if not chunks:
            return [(text, 0, len(text))]

        # Merge small trailing chunks if it doesn't exceed max_chars
        merged: List[Tuple[str, int, int]] = []
        for chunk_text, chunk_start, chunk_end in chunks:
            if not merged:
                merged.append((chunk_text, chunk_start, chunk_end))
                continue
            if len(chunk_text.strip()) < min_chars:
                prev_text, prev_start, _ = merged[-1]
                if len(prev_text) + len(chunk_text) <= max_chars:
                    merged[-1] = (prev_text + chunk_text, prev_start, chunk_end)
                else:
                    merged.append((chunk_text, chunk_start, chunk_end))
            else:
                merged.append((chunk_text, chunk_start, chunk_end))

        return merged

    def _split_prefix(self, text: str, max_len: int) -> tuple[str, str, int]:
        """
        Split text into a prefix of at most max_len and the remaining suffix.
        Prefer cutting at newline, then pipe, then space.
        Returns (prefix, suffix, prefix_len).
        """
        if max_len <= 0:
            return "", text, 0
        if len(text) <= max_len:
            return text, "", len(text)

        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = text.rfind("|", 0, max_len)
        if cut <= 0:
            cut = text.rfind(" ", 0, max_len)
        if cut <= 0:
            cut = max_len

        prefix = text[:cut]
        suffix = text[cut:]
        return prefix, suffix, cut

    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """
        Get overlap text from the end of a chunk.
        Tries to end at a sentence boundary for better context.
        """
        if overlap_chars <= 0 or len(text) <= overlap_chars:
            return ""

        # Get the last overlap_chars
        overlap_region = text[-overlap_chars:]

        # Try to start at a sentence boundary
        sentences = self._split_into_sentences(overlap_region)
        if len(sentences) > 1:
            # Skip the first (potentially partial) sentence
            return "".join(s[0] for s in sentences[1:]).strip()

        # If no sentence boundary, try to start at a word boundary
        space_pos = overlap_region.find(" ")
        if 0 < space_pos < len(overlap_region) // 2:
            return overlap_region[space_pos + 1 :].strip()

        return overlap_region.strip()

    def split_sections(self, text: str) -> List[tuple[Optional[str], str, int]]:
        """Split text by markdown headers, merging small sections to avoid fragmentation."""
        if not text:
            return [(None, "", 0)]
        matches = list(SECTION_PATTERN.finditer(text))
        if not matches:
            return [(None, text, 0)]

        raw_sections: List[tuple[Optional[str], str, int]] = []

        # Handle preamble (text before first header)
        first_match_start = matches[0].start()
        preamble = ""
        preamble_start = 0

        if first_match_start > 0:
            preamble = text[:first_match_start]
            # If preamble is substantial, treat it as its own section.
            # Otherwise (e.g. just a page marker), we'll merge it into the first section.
            if preamble.strip() and len(preamble) > 300:
                raw_sections.append((None, preamble, 0))
                preamble = ""  # Consumed
            else:
                preamble_start = 0

        for index, match in enumerate(matches):
            # Include the header in the section body so it gets chunked and highlighted
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            section_marker = f"{heading_level}:{heading_text}"

            # If this is the first section and we have a short preamble to merge
            if index == 0 and preamble:
                body = preamble + text[start:end]
                raw_sections.append((section_marker, body, preamble_start))
            else:
                body = text[start:end]
                raw_sections.append((section_marker, body, start))

        return raw_sections

    @staticmethod
    def list_density(text: str) -> float:
        if not text:
            return 0.0
        list_matches = LIST_PATTERN.findall(text)
        if not list_matches:
            return 0.0
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        return min(len(list_matches) / len(lines), 1.0)


class SemanticChunker:
    """Semantic chunking with optional section splitting and small-chunk merge."""

    def __init__(
        self,
        max_chars: int,
        min_chars: int,
        overlap_chars: int,
        section_split_file_types: Optional[List[str]] = None,
        section_split_all: bool = False,
        char_ratio: int = 4,
        token_limit: Optional[int] = 512,
        token_char_ratio: float = 2.7,
        token_encoding: str = _DEFAULT_TIKTOKEN_ENCODING,
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars
        self.section_split_all = section_split_all
        self.section_split_file_types = {
            str(t).lower().strip().lstrip(".")
            for t in (section_split_file_types or [])
        }
        self.splitter = SemanticSplitter(char_ratio=char_ratio)
        self.token_limit = token_limit
        self.token_char_ratio = token_char_ratio
        self.token_encoding = token_encoding
        self._tokenizer = _get_tiktoken_encoder(token_encoding)

    def chunk_text(self, text: str, file_type: str = "") -> List[str]:
        file_type_norm = str(file_type or "").lower().strip().lstrip(".")
        if not text:
            return []

        if self.section_split_all or file_type_norm in self.section_split_file_types:
            sections = self.splitter.split_sections(text)
            chunks: List[str] = []
            for _, section_text, _ in sections:
                section_chunks = self._semantic_chunks(section_text)
                chunks.extend(self._enforce_token_limit(self._merge_small_chunks(section_chunks)))
            return chunks

        raw_chunks = self._semantic_chunks(text)
        return self._enforce_token_limit(self._merge_small_chunks(raw_chunks))

    def _semantic_chunks(self, text: str) -> List[str]:
        entries = self.splitter.semantic_chunk(
            text=text,
            max_chars=self.max_chars,
            min_chars=self.min_chars,
            overlap_chars=self.overlap_chars,
        )
        return [c[0] for c in entries if c and c[0].strip()]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        merged: List[str] = []
        pending = chunks[0]
        for payload in chunks[1:]:
            combined = len(pending) + len(payload) + 2
            should_merge = combined <= self.max_chars
            if should_merge:
                pending = f"{pending}\n\n{payload}"
            else:
                merged.append(pending)
                pending = payload

        if pending:
            merged.append(pending)

        return merged

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._tokenizer.encode(text, disallowed_special=()))
        except Exception as exc:
            logger.warning("tiktoken failed to encode text; falling back to char estimate: %s", exc)
            cjk = sum(
                1
                for ch in text
                if ("\u4e00" <= ch <= "\u9fff")
                or ("\u3400" <= ch <= "\u4dbf")
                or ("\u3040" <= ch <= "\u30ff")
            )
            non_cjk_len = max(0, len(text) - cjk)
            words = len(re.findall(r"\S+", text))
            est_by_chars = int(non_cjk_len / self.token_char_ratio) + cjk
            est_by_words = int(words * 1.2)
            return max(est_by_chars, est_by_words)

    def _split_to_token_limit(self, text: str, max_chars: int) -> List[str]:
        if self.token_limit is None or self.token_limit <= 0:
            return [text]
        if self._estimate_tokens(text) <= self.token_limit:
            return [text]
        return self._split_by_token_limit_blocks(text)

    def _split_by_token_limit_blocks(self, text: str) -> List[str]:
        """
        Split text by semantic blocks using exact token counts.

        Strategy:
        - Parse semantic blocks.
        - Accumulate blocks until token_limit is reached.
        - If a single block exceeds token_limit, split it precisely via binary search.
        """
        if not text or not text.strip():
            return []

        blocks = self.splitter._parse_semantic_blocks(text)
        if not blocks:
            return self._split_large_block_by_tokens(text)

        separator = "\n\n"
        separator_tokens = self._estimate_tokens(separator)

        results: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for block in blocks:
            block_text = block.text
            if not block_text or not block_text.strip():
                continue

            block_tokens = self._estimate_tokens(block_text)
            if block_tokens > self.token_limit:
                if current_parts:
                    results.append(separator.join(current_parts))
                    current_parts = []
                    current_tokens = 0

                results.extend(self._split_large_block_by_tokens(block_text))
                continue

            additional_tokens = block_tokens + (separator_tokens if current_parts else 0)
            if current_tokens + additional_tokens <= self.token_limit:
                current_parts.append(block_text)
                current_tokens += additional_tokens
            else:
                if current_parts:
                    results.append(separator.join(current_parts))
                current_parts = [block_text]
                current_tokens = block_tokens

        if current_parts:
            results.append(separator.join(current_parts))

        return results

    def _split_large_block_by_tokens(self, text: str) -> List[str]:
        """Split a single large block into token-limited pieces using binary search."""
        results: List[str] = []
        remaining = text

        while remaining and remaining.strip():
            if self._estimate_tokens(remaining) <= self.token_limit:
                results.append(remaining)
                remaining = ""
                break

            prefix, rest = self._truncate_to_token_limit(remaining, self.token_limit)
            if prefix.strip():
                results.append(prefix)
            remaining = rest

        return results

    def _truncate_to_token_limit(self, text: str, limit: int) -> Tuple[str, str]:
        """Find the longest prefix within token limit using binary search."""
        if limit <= 0 or not text:
            return "", text

        low = 1
        high = len(text)
        best = 0

        while low <= high:
            mid = (low + high) // 2
            if self._estimate_tokens(text[:mid]) <= limit:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best <= 0:
            # Fallback to ensure progress
            return text[:1], text[1:]

        return text[:best], text[best:]

    def _enforce_token_limit(self, chunks: List[str]) -> List[str]:
        if self.token_limit is None or self.token_limit <= 0:
            return chunks

        adjusted: List[str] = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) > self.token_limit:
                adjusted.extend(self._split_to_token_limit(chunk, self.max_chars))
            else:
                adjusted.append(chunk)
        return adjusted
