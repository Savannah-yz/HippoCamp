from typing import List, Dict, Any, Optional
import hashlib
from .base import BaseChunker, Chunk, SegmentInfo, ChunkMeta

try:
    from chonkie import RecursiveChunker
    from chonkie import OverlapRefinery
except ImportError:
    RecursiveChunker = None
    OverlapRefinery = None


class ChonkieRecursiveChunker(BaseChunker):
    """
    基于 Chonkie 库的递归分块器。

    支持所有10种file_type的统一处理：
    - mp3: 音频文件，segment包含start/end时间
    - pdf: PDF文件，segment包含page_number
    - docx/txt/csv/json/eml/md/ics/png: 其他文件类型

    Overlap 通过 OverlapRefinery 实现，在 chunking 后添加上下文重叠。
    """

    def __init__(self, chunk_size: int = 512, min_chars: int = 10, overlap: int = 0):
        """
        初始化递归分块器。

        Args:
            chunk_size: 每个 chunk 的最大 token 数
            min_chars: 每个 chunk 的最小字符数
            overlap: 重叠大小。如果 > 0，会使用 OverlapRefinery 添加上下文重叠。
        """
        if RecursiveChunker is None:
            raise ImportError("Please install 'chonkie' to use ChonkieRecursiveChunker.")

        self.chunk_size = chunk_size
        self.min_chars = min_chars
        self.overlap = overlap

        # 初始化 Chonkie 的 RecursiveChunker
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chars,
        )

        # 初始化 OverlapRefinery（如果需要 overlap）
        self.overlap_refinery = None
        if overlap > 0:
            if OverlapRefinery is None:
                raise ImportError("Please install 'chonkie' with overlap support to use overlap feature.")
            self.overlap_refinery = OverlapRefinery(
                tokenizer="character",
                context_size=overlap,
                method="suffix",
                merge=True,
                inplace=False,
            )

    def _generate_id(self, content: str, file_id: str = "") -> str:
        """生成唯一的 hash ID"""
        combined = f"{file_id}:{content}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def create_chunks_from_metadata(self, file_data: Dict[str, Any]) -> List[Chunk]:
        """
        从 metadata JSON 中提取文本并进行分块。

        Args:
            file_data: 包含 file_info 和 segments 的字典

        Returns:
            List[Chunk]: 统一格式的 Chunk 对象列表

        输出的每个Chunk包含：
        - id: 唯一hash ID
        - content: chunk文本内容
        - file_info: 完整的原始file_info信息
        - segment_info: 来源segment的信息（包含segment_indices, page_numbers, time_ranges）
        - chunk_meta: chunk元信息（type, chunk_index, char_count）
        """
        chunks: List[Chunk] = []

        # 1. 获取 file_info (完整保留所有原始字段)
        file_info = file_data.get("file_info", {})
        file_info_copy = dict(file_info)

        # 获取 file_id 用于生成 hash（使用 "id" 字段，不添加冗余的 "file_id"）
        file_id = str(file_info.get("id", file_info.get("file_id", "unknown")))
        file_type = file_info_copy.get("file_type", "text")

        # 兼容两种格式
        if "metadata" in file_data and isinstance(file_data["metadata"], dict) and file_data["metadata"]:
            target_metadata = file_data["metadata"]
        else:
            target_metadata = file_data

        chunk_global_index = 0

        # 2. 处理全局摘要 (Global Summary)
        summary = target_metadata.get("summary", "")
        if summary and summary.strip():
            summary_text = summary.strip()
            chunk = Chunk(
                id=self._generate_id(summary_text, file_id),
                content=summary_text,
                file_info=dict(file_info_copy),
                segment_info=SegmentInfo(segment_indices=[-1]),  # -1 表示来自summary
                chunk_meta=ChunkMeta(
                    type="summary",
                    chunk_index=chunk_global_index,
                    char_count=len(summary_text),
                                    )
            )
            chunks.append(chunk)
            chunk_global_index += 1

        # 3. 处理具体内容 (Segments)
        segments = target_metadata.get("segments", [])

        for seg_idx, segment in enumerate(segments):
            text = segment.get("content", "")
            if not text or not text.strip():
                continue

            # 根据file_type提取segment特有信息
            segment_info = self._extract_segment_info(segment, seg_idx, file_type)

            # 调用 Chonkie 进行分块
            chonkie_chunks = self.chunker.chunk(text)

            # 应用 overlap refinery（如果启用）
            if chonkie_chunks and self.overlap_refinery:
                chonkie_chunks = self.overlap_refinery(chonkie_chunks)

            if chonkie_chunks:
                for local_idx, c_chunk in enumerate(chonkie_chunks):
                    final_content = c_chunk.text.strip()
                    if not final_content:
                        continue

                    chunk = Chunk(
                        id=self._generate_id(final_content, file_id),
                        content=final_content,
                        file_info=dict(file_info_copy),
                        segment_info=SegmentInfo(
                            segment_indices=segment_info["segment_indices"],
                            page_numbers=segment_info.get("page_numbers", []),
                            time_ranges=segment_info.get("time_ranges", [])
                        ),
                        chunk_meta=ChunkMeta(
                            type="content",
                            chunk_index=chunk_global_index,
                            char_count=len(final_content)
                        )
                    )
                    chunks.append(chunk)
                    chunk_global_index += 1
            else:
                # Fallback: 如果Chonkie返回空，创建单个chunk
                fallback_text = text.strip()
                if len(fallback_text) >= self.min_chars:
                    chunk = Chunk(
                        id=self._generate_id(fallback_text, file_id),
                        content=fallback_text,
                        file_info=dict(file_info_copy),
                        segment_info=SegmentInfo(
                            segment_indices=segment_info["segment_indices"],
                            page_numbers=segment_info.get("page_numbers", []),
                            time_ranges=segment_info.get("time_ranges", [])
                        ),
                        chunk_meta=ChunkMeta(
                            type="content",
                            chunk_index=chunk_global_index,
                            char_count=len(fallback_text)
                        )
                    )
                    chunks.append(chunk)
                    chunk_global_index += 1

        return chunks

    def _extract_segment_info(
        self, segment: Dict[str, Any], seg_idx: int, file_type: str
    ) -> Dict[str, Any]:
        """
        根据file_type从segment中提取特有信息。

        Args:
            segment: segment字典
            seg_idx: segment的索引
            file_type: 文件类型

        Returns:
            包含segment信息的字典
        """
        info = {"segment_indices": [seg_idx]}

        # MP3/音频文件: 提取start/end时间
        if file_type == "mp3" or "start" in segment or "end" in segment:
            start = segment.get("start")
            end = segment.get("end")
            if start is not None or end is not None:
                time_range = {}
                if start is not None:
                    time_range["start"] = float(start) if not isinstance(start, (int, float)) else start
                if end is not None:
                    time_range["end"] = float(end) if not isinstance(end, (int, float)) else end
                info["time_ranges"] = [time_range]

        # PDF/DOCX文件: 提取page_number
        if file_type in ("pdf", "docx") or "page_number" in segment:
            page_num = segment.get("page_number")
            if page_num is not None:
                info["page_numbers"] = [page_num]

        return info

    def create_chunks_cross_segments(
        self, file_data: Dict[str, Any], max_chars: int = 2000
    ) -> List[Chunk]:
        """
        跨segment进行分块（当一个chunk需要横跨多个segment时使用）。

        适用于segment内容较短需要合并的情况。

        Args:
            file_data: 包含 file_info 和 segments 的字典
            max_chars: 单个chunk的最大字符数

        Returns:
            List[Chunk]: 统一格式的 Chunk 对象列表
        """
        chunks: List[Chunk] = []

        # 获取 file_info
        file_info = file_data.get("file_info", {})
        file_info_copy = dict(file_info)

        # 获取 file_id 用于生成 hash（使用 "id" 字段，不添加冗余的 "file_id"）
        file_id = str(file_info.get("id", file_info.get("file_id", "unknown")))
        file_type = file_info_copy.get("file_type", "text")

        # 兼容两种格式
        if "metadata" in file_data and isinstance(file_data["metadata"], dict) and file_data["metadata"]:
            target_metadata = file_data["metadata"]
        else:
            target_metadata = file_data

        segments = target_metadata.get("segments", [])
        if not segments:
            return chunks

        chunk_global_index = 0

        # 处理summary
        summary = target_metadata.get("summary", "")
        if summary and summary.strip():
            summary_text = summary.strip()
            chunk = Chunk(
                id=self._generate_id(summary_text, file_id),
                content=summary_text,
                file_info=dict(file_info_copy),
                segment_info=SegmentInfo(segment_indices=[-1]),
                chunk_meta=ChunkMeta(
                    type="summary",
                    chunk_index=chunk_global_index,
                    char_count=len(summary_text),
                                    )
            )
            chunks.append(chunk)
            chunk_global_index += 1

        # 合并短segment并创建跨segment的chunk
        current_content = ""
        current_segment_indices = []
        current_page_numbers = []
        current_time_ranges = []

        for seg_idx, segment in enumerate(segments):
            text = segment.get("content", "")
            if not text or not text.strip():
                continue

            seg_info = self._extract_segment_info(segment, seg_idx, file_type)

            # 如果加入当前segment后超过max_chars，先保存当前chunk
            if current_content and len(current_content) + len(text) > max_chars:
                chunk_text = current_content.strip()
                if len(chunk_text) >= self.min_chars:
                    chunk = Chunk(
                        id=self._generate_id(chunk_text, file_id),
                        content=chunk_text,
                        file_info=dict(file_info_copy),
                        segment_info=SegmentInfo(
                            segment_indices=current_segment_indices,
                            page_numbers=current_page_numbers,
                            time_ranges=current_time_ranges
                        ),
                        chunk_meta=ChunkMeta(
                            type="content",
                            chunk_index=chunk_global_index,
                            char_count=len(chunk_text)
                        )
                    )
                    chunks.append(chunk)
                    chunk_global_index += 1

                # 重置
                current_content = ""
                current_segment_indices = []
                current_page_numbers = []
                current_time_ranges = []

            # 累积内容
            current_content += ("\n\n" if current_content else "") + text.strip()
            current_segment_indices.append(seg_idx)

            # 累积page_numbers
            if seg_info.get("page_numbers"):
                for pn in seg_info["page_numbers"]:
                    if pn not in current_page_numbers:
                        current_page_numbers.append(pn)

            # 累积time_ranges
            if seg_info.get("time_ranges"):
                current_time_ranges.extend(seg_info["time_ranges"])

        # 处理最后一个chunk
        if current_content:
            final_text = current_content.strip()
            if len(final_text) >= self.min_chars:
                chunk = Chunk(
                    id=self._generate_id(final_text, file_id),
                    content=final_text,
                    file_info=dict(file_info_copy),
                    segment_info=SegmentInfo(
                        segment_indices=current_segment_indices,
                        page_numbers=current_page_numbers,
                        time_ranges=current_time_ranges
                    ),
                    chunk_meta=ChunkMeta(
                        type="content",
                        chunk_index=chunk_global_index,
                        char_count=len(final_text)
                    )
                )
                chunks.append(chunk)

        return chunks
