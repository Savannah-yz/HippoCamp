"""
Retrieval-specific metrics for RAG evaluation.
"""

from typing import Any, Dict, List, Optional, Set
from .base import MetricCalculator, EvaluationMetric, RetrievedChunk


class RetrievalPrecisionMetric(MetricCalculator):
    """Precision for retrieved chunks based on source_id matching."""

    def __init__(self):
        super().__init__("retrieval_precision")

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate precision of retrieved chunks."""
        if not retrieved_chunks or not expected_chunks:
            return self._create_metric(0.0, {"error": "Missing retrieved_chunks or expected_chunks"})

        try:
            # Get source_ids from retrieved chunks
            retrieved_sources = set()
            for chunk in retrieved_chunks:
                if chunk.source_id:
                    retrieved_sources.add(chunk.source_id)
                # Also check metadata for file_path
                if chunk.metadata.get("file_path"):
                    retrieved_sources.add(chunk.metadata["file_path"])
                if chunk.metadata.get("file_id"):
                    retrieved_sources.add(chunk.metadata["file_id"])

            # Normalize expected chunks (file paths)
            expected_set = {chunk.strip() for chunk in expected_chunks if chunk.strip()}

            if not retrieved_sources:
                return self._create_metric(0.0, {"error": "No source IDs in retrieved chunks"})

            # Calculate intersection
            relevant_retrieved = retrieved_sources.intersection(expected_set)
            precision = len(relevant_retrieved) / len(retrieved_sources)

            details = {
                "retrieved_count": len(retrieved_sources),
                "expected_count": len(expected_set),
                "relevant_retrieved_count": len(relevant_retrieved),
                "retrieved_sources": list(retrieved_sources),
                "expected_sources": list(expected_set),
                "matched_sources": list(relevant_retrieved)
            }

            return self._create_metric(precision, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class RetrievalRecallMetric(MetricCalculator):
    """Recall for retrieved chunks based on source_id matching."""

    def __init__(self):
        super().__init__("retrieval_recall")

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate recall of retrieved chunks."""
        if not retrieved_chunks or not expected_chunks:
            return self._create_metric(0.0, {"error": "Missing retrieved_chunks or expected_chunks"})

        try:
            # Get source_ids from retrieved chunks
            retrieved_sources = set()
            for chunk in retrieved_chunks:
                if chunk.source_id:
                    retrieved_sources.add(chunk.source_id)
                if chunk.metadata.get("file_path"):
                    retrieved_sources.add(chunk.metadata["file_path"])
                if chunk.metadata.get("file_id"):
                    retrieved_sources.add(chunk.metadata["file_id"])

            # Normalize expected chunks
            expected_set = {chunk.strip() for chunk in expected_chunks if chunk.strip()}

            if not expected_set:
                return self._create_metric(0.0, {"error": "No expected chunks"})

            # Calculate intersection
            relevant_retrieved = retrieved_sources.intersection(expected_set)
            recall = len(relevant_retrieved) / len(expected_set)

            details = {
                "retrieved_count": len(retrieved_sources),
                "expected_count": len(expected_set),
                "relevant_retrieved_count": len(relevant_retrieved),
                "retrieved_sources": list(retrieved_sources),
                "expected_sources": list(expected_set),
                "matched_sources": list(relevant_retrieved)
            }

            return self._create_metric(recall, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class RetrievalF1Metric(MetricCalculator):
    """F1 score for retrieved chunks."""

    def __init__(self):
        super().__init__("retrieval_f1")
        self.precision_metric = RetrievalPrecisionMetric()
        self.recall_metric = RetrievalRecallMetric()

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate F1 score of retrieved chunks."""
        if not retrieved_chunks or not expected_chunks:
            return self._create_metric(0.0, {"error": "Missing retrieved_chunks or expected_chunks"})

        try:
            # Calculate precision and recall
            precision_result = await self.precision_metric.calculate(
                query, generated_answer, ground_truth, retrieved_chunks, expected_chunks, **kwargs
            )
            recall_result = await self.recall_metric.calculate(
                query, generated_answer, ground_truth, retrieved_chunks, expected_chunks, **kwargs
            )

            precision = precision_result.score
            recall = recall_result.score

            # Calculate F1 score
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            details = {
                "precision": precision,
                "recall": recall,
                "precision_details": precision_result.details,
                "recall_details": recall_result.details
            }

            return self._create_metric(f1_score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class RetrievalMRRMetric(MetricCalculator):
    """Mean Reciprocal Rank for retrieved chunks."""

    def __init__(self):
        super().__init__("retrieval_mrr")

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate Mean Reciprocal Rank of retrieved chunks."""
        if not retrieved_chunks or not expected_chunks:
            return self._create_metric(0.0, {"error": "Missing retrieved_chunks or expected_chunks"})

        try:
            expected_set = {chunk.strip() for chunk in expected_chunks if chunk.strip()}

            # Find the rank of the first relevant chunk
            first_relevant_rank = None
            for i, chunk in enumerate(retrieved_chunks):
                source_ids = set()
                if chunk.source_id:
                    source_ids.add(chunk.source_id)
                if chunk.metadata.get("file_path"):
                    source_ids.add(chunk.metadata["file_path"])
                if chunk.metadata.get("file_id"):
                    source_ids.add(chunk.metadata["file_id"])

                if source_ids.intersection(expected_set):
                    first_relevant_rank = i + 1  # 1-indexed
                    break

            # Calculate MRR
            mrr_score = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

            details = {
                "first_relevant_rank": first_relevant_rank,
                "total_retrieved": len(retrieved_chunks),
                "total_expected": len(expected_chunks)
            }

            return self._create_metric(mrr_score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})
