from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod


@dataclass
class SegmentInfo:
    """
    记录chunk来源的segment信息。
    支持不同file_type的特殊字段：
    - MP3/音频: time_ranges (start/end)
    - PDF: page_numbers
    - 其他: 仅segment_indices
    """
    segment_indices: List[int] = field(default_factory=list)  # 来自哪些segment
    page_numbers: List[int] = field(default_factory=list)     # PDF页码列表
    time_ranges: List[Dict[str, float]] = field(default_factory=list)  # 音频时间范围

    def to_dict(self) -> Dict[str, Any]:
        result = {"segment_indices": self.segment_indices}
        if self.page_numbers:
            result["page_numbers"] = self.page_numbers
        if self.time_ranges:
            result["time_ranges"] = self.time_ranges
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentInfo":
        return cls(
            segment_indices=data.get("segment_indices", []),
            page_numbers=data.get("page_numbers", []),
            time_ranges=data.get("time_ranges", [])
        )


@dataclass
class ChunkMeta:
    """
    Chunk的元信息。
    """
    type: str = "content"  # "content" 或 "summary"
    chunk_index: int = 0   # 在文件中的顺序
    char_count: int = 0    # 字符数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMeta":
        return cls(
            type=data.get("type", "content"),
            chunk_index=data.get("chunk_index", 0),
            char_count=data.get("char_count", 0)
        )


@dataclass
class Chunk:
    """
    统一的 Chunk 定义。

    包含：
    - id: chunk的唯一hash ID
    - content: chunk的实际内容文本
    - file_info: 完整保留原始file_info的所有字段
    - segment_info: 记录chunk来源的segment信息
    - chunk_meta: chunk的元信息

    兼容旧的 metadata 字典格式。
    """
    content: str
    id: Optional[str] = None
    file_info: Dict[str, Any] = field(default_factory=dict)
    segment_info: Optional[SegmentInfo] = None
    chunk_meta: Optional[ChunkMeta] = None
    # 保留旧的metadata字段以兼容现有代码
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.segment_info is None:
            self.segment_info = SegmentInfo()
        if self.chunk_meta is None:
            self.chunk_meta = ChunkMeta(char_count=len(self.content))

    def to_dict(self) -> Dict[str, Any]:
        """转换为统一的字典格式"""
        return {
            "id": self.id,
            "content": self.content,
            "file_info": self.file_info,
            "segment_info": self.segment_info.to_dict() if self.segment_info else {},
            "chunk_meta": self.chunk_meta.to_dict() if self.chunk_meta else {}
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """转换为旧格式（兼容现有代码）"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建Chunk对象"""
        segment_info = None
        chunk_meta = None

        if "segment_info" in data:
            segment_info = SegmentInfo.from_dict(data["segment_info"])
        if "chunk_meta" in data:
            chunk_meta = ChunkMeta.from_dict(data["chunk_meta"])

        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            file_info=data.get("file_info", {}),
            segment_info=segment_info,
            chunk_meta=chunk_meta,
            metadata=data.get("metadata", {})
        )

class BaseChunker(ABC):
    """
    所有分块器的基类。
    继承此类并实现 chunk 方法来进行消融实验。
    """
    
    @abstractmethod
    def create_chunks_from_metadata(self, file_data: Dict[str, Any]) -> List[Chunk]:
        """
        Args:
            file_data: 从 metadata json 加载的字典。
                       通常包含 {"metadata": {"summary": ..., "segments": ...}}
        
        Returns:
            List[Chunk]: 分好块的 Chunk 对象列表
        """
        pass

