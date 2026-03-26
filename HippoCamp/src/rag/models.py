"""
Data models for RAG query results and evaluation.

These models are designed to be compatible with the rag_eval metrics system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    """A single retrieved chunk from the vector store."""
    rank: int
    content: str
    score: float
    id: str  # chunk_id renamed to id for consistency
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepLatency:
    """Latency information for a single pipeline step."""
    step_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepLatency":
        return cls(
            step_name=data.get("step_name", ""),
            latency_ms=data.get("latency_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ThinkingStep:
    """A single thinking/reasoning step in the RAG pipeline."""
    step_id: str
    step_type: str  # e.g., "route", "retrieve", "grade", "rewrite", "generate"
    title: str
    status: str  # "running", "complete", "failed"
    summary: Optional[str] = None
    timestamp_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "timestamp_ms": self.timestamp_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingStep":
        return cls(
            step_id=data.get("step_id", ""),
            step_type=data.get("step_type", ""),
            title=data.get("title", ""),
            status=data.get("status", ""),
            summary=data.get("summary"),
            timestamp_ms=data.get("timestamp_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryResultRecord:
    """
    Record for a single query result.

    This format is designed to be compatible with rag_eval metrics evaluation.

    Example JSON output:
    {
        "timestamp": "20250115_143022",
        "provider": "research_core",
        "bench": "test_bench",
        "query_id": "q1",
        "query": "What is machine learning?",
        "ground_truth": "Machine learning is...",
        "answer": "Based on the context, machine learning is...",
        "retrieved_chunks": [
            {
                "rank": 1,
                "content": "Machine learning is a subset of AI...",
                "score": 0.89,
                "id": "d71d2dbf54592bfbbb09ce98c7052b7a",
                "metadata": {
                    "file_info": {
                        "file_id": "doc_456",
                        "file_path": "reports/doc.pdf",
                        "file_type": "pdf",
                        "file_name": "doc.pdf"
                    },
                    "segment_info": {
                        "segment_indices": [0, 1],
                        "page_numbers": [1, 2],
                        "time_ranges": []
                    },
                    "chunk_meta": {
                        "type": "content",
                        "chunk_index": 0,
                        "char_count": 512,
                        "token_count": 128
                    }
                }
            }
        ],
        "expected_chunks": [
            {
                "evidence_id": "1",
                "content": "Evidence text...",
                "metadata": {"file_name": "xxx.eml", "file_path": "...", "file_type": "document"}
            }
        ],
        "file_list": ["path/to/file1.pdf", "path/to/file2.docx"],
        "retrieved_file_list": ["path/to/file1.pdf", "path/to/file3.txt"],
        "stages": ["retrieve", "rerank", "generate"],
        "execution_time_ms": 1234,
        "latency_breakdown": {
            "embed_query_ms": 50,
            "retrieve_ms": 200,
            "rerank_ms": 150,
            "grade_ms": 300,
            "rewrite_ms": 100,
            "generate_ms": 400,
            "total_ms": 1200
        },
        "thinking_steps": [
            {"step_id": "route_1", "step_type": "route", "title": "Routing Query", "status": "complete"},
            {"step_id": "retrieve_1", "step_type": "retrieve", "title": "Retrieving Documents", "status": "complete"}
        ],
        "step_counts": {
            "search_count": 2,
            "rewrite_count": 1,
            "grade_count": 1,
            "total_iterations": 2
        }
    }
    """
    timestamp: str
    provider: str
    bench: str
    query_id: str
    query: str
    answer: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    ground_truth: Optional[str] = None
    expected_chunks: Optional[List[Dict[str, Any]]] = None  # Changed from List[str] to List[Dict]
    file_list: Optional[List[str]] = None  # Required files from benchmark (from file_path field)
    retrieved_file_list: Optional[List[str]] = None  # Unique file paths from retrieved chunks
    stages: List[str] = field(default_factory=list)
    execution_time_ms: Optional[int] = None
    rewritten_query: Optional[str] = None
    user_profile_included: bool = False
    # New fields for detailed metrics
    latency_breakdown: Optional[Dict[str, float]] = None  # Step-by-step latency in ms
    thinking_steps: Optional[List[Dict[str, Any]]] = None  # Detailed thinking/reasoning trace
    step_counts: Optional[Dict[str, int]] = None  # Counts of various operations
    # ReAct / end-to-end reasoning fields
    reasoning_trace: Optional[str] = None  # Full ReAct Thought/Action/Observation trace
    search_queries: Optional[List[str]] = None  # Search queries issued during ReAct loop
    retrieval_metadata: Optional[Dict[str, Any]] = None  # Provider-specific metadata (e.g. prerank_chunks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "bench": self.bench,
            "query_id": self.query_id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "retrieved_chunks": self.retrieved_chunks,
            "expected_chunks": self.expected_chunks,
            "file_list": self.file_list,
            "retrieved_file_list": self.retrieved_file_list,
            "stages": self.stages,
            "execution_time_ms": self.execution_time_ms,
            "rewritten_query": self.rewritten_query,
            "user_profile_included": self.user_profile_included,
        }
        # Include new metrics fields if present
        if self.latency_breakdown:
            result["latency_breakdown"] = self.latency_breakdown
        if self.thinking_steps:
            result["thinking_steps"] = self.thinking_steps
        if self.step_counts:
            result["step_counts"] = self.step_counts
        if self.reasoning_trace:
            result["reasoning_trace"] = self.reasoning_trace
        if self.search_queries:
            result["search_queries"] = self.search_queries
        if self.retrieval_metadata:
            result["retrieval_metadata"] = self.retrieval_metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResultRecord":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            provider=data.get("provider", "research_core"),
            bench=data.get("bench", ""),
            query_id=data.get("query_id", ""),
            query=data.get("query", ""),
            ground_truth=data.get("ground_truth"),
            answer=data.get("answer"),
            retrieved_chunks=data.get("retrieved_chunks", []),
            expected_chunks=data.get("expected_chunks"),
            file_list=data.get("file_list"),
            retrieved_file_list=data.get("retrieved_file_list"),
            stages=data.get("stages", []),
            execution_time_ms=data.get("execution_time_ms"),
            rewritten_query=data.get("rewritten_query"),
            user_profile_included=data.get("user_profile_included", False),
            latency_breakdown=data.get("latency_breakdown"),
            thinking_steps=data.get("thinking_steps"),
            step_counts=data.get("step_counts"),
            reasoning_trace=data.get("reasoning_trace"),
            search_queries=data.get("search_queries"),
            retrieval_metadata=data.get("retrieval_metadata"),
        )


@dataclass
class BenchmarkTestCase:
    """
    A test case from a benchmark JSON file.

    Supports the HippoCamp/Victoria format with evidence fields.
    """
    id: str
    question: str
    ground_truth: Optional[str] = None
    file_path: Optional[List[str]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTestCase":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            ground_truth=data.get("answer"),  # 'answer' field is ground truth
            file_path=data.get("file_path"),
            evidence=data.get("evidence"),
            metadata={
                k: v for k, v in data.items()
                if k not in ["id", "question", "answer", "file_path", "evidence"]
            }
        )

    def get_expected_chunk_ids(self) -> List[str]:
        """
        Extract expected chunk identifiers from evidence.

        Returns list of file_path values from evidence items.
        """
        if not self.evidence:
            return []
        return [e.get("file_path", "") for e in self.evidence if e.get("file_path")]

    def get_expected_chunks(self) -> List[Dict[str, Any]]:
        """
        Convert evidence to expected_chunks format (similar to retrieved_chunks).

        Each evidence item is converted to:
        {
            "evidence_id": "1",
            "content": "evidence_text content...",
            "metadata": {
                "file_name": "xxx.eml",
                "file_path": "Documents/Outlook/Inbox/xxx.eml",
                "file_type": "document"  # modality_type
            }
        }
        """
        if not self.evidence:
            return []

        expected_chunks = []
        for e in self.evidence:
            file_path = e.get("file_path", "")
            # Extract file_name from file_path
            file_name = file_path.split("/")[-1] if file_path else ""

            # Extract page numbers from evidence_locator if available
            page_numbers = []
            for locator in e.get("evidence_locator", []):
                if locator.get("unit") == "page":
                    page_num = locator.get("position", {}).get("system_page")
                    if page_num is not None:
                        page_numbers.append(page_num)

            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_type": e.get("modality_type", ""),
            }
            if page_numbers:
                metadata["page_numbers"] = page_numbers

            expected_chunks.append({
                "evidence_id": e.get("evidence_id", ""),
                "content": e.get("evidence_text", ""),
                "metadata": metadata,
            })

        return expected_chunks

    def get_file_list(self) -> List[str]:
        """
        Get the list of required file paths for this test case.

        This is used for file list evaluation metrics.
        Returns the file_path field which contains all files needed to answer the question.
        """
        return self.file_path or []


def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
