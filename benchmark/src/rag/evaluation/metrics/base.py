"""
Base classes for evaluation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationMetric:
    """Result of a metric calculation."""
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
        }


@dataclass
class RetrievedChunk:
    """A retrieved chunk for evaluation."""
    content: str
    chunk_id: str = ""
    score: float = 0.0
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        return cls(
            content=data.get("content", ""),
            chunk_id=data.get("chunk_id", ""),
            score=data.get("score", 0.0),
            source_id=data.get("source_id", ""),
            metadata=data.get("metadata", {}),
        )


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate the metric score."""
        pass

    def _create_metric(
        self,
        score: float,
        details: Optional[Dict[str, Any]] = None
    ) -> EvaluationMetric:
        """Helper to create an EvaluationMetric instance."""
        return EvaluationMetric(
            name=self.name,
            score=score,
            details=details or {}
        )
