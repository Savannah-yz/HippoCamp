"""
Evaluation runner for computing metrics on query results.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .metrics import (
    MetricCalculator,
    EvaluationMetric,
    RetrievedChunk,
    ROUGEMetric,
    BLEUMetric,
    ExactMatchMetric,
    CoveredExactMatchMetric,
    SemanticSimilarityMetric,
    BERTScoreMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
    RetrievalF1Metric,
    LLMJudgeMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query_id: str
    query: str
    answer: Optional[str]
    ground_truth: Optional[str]
    metrics: List[EvaluationMetric]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    file_list_metrics: Optional[Dict[str, float]] = None  # Added for file list metrics

    def to_dict(self) -> Dict[str, Any]:
        """Legacy format: returns metrics as a list."""
        result = {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "metrics": [m.to_dict() for m in self.metrics],
            "timestamp": self.timestamp,
        }
        if self.file_list_metrics:
            result["file_list_metrics"] = self.file_list_metrics
        return result

    def to_structured_dict(self) -> Dict[str, Any]:
        """
        Returns evaluation result in structured format with judge and simple_metrics separated.

        Output format:
        {
            "query_id": "...",
            "query": "...",
            "answer": "...",
            "ground_truth": "...",
            "judge": { ... },  # LLM judge results
            "simple_metrics": { ... },  # All other metrics
            "file_list_metrics": { ... },  # File list retrieval metrics
            "timestamp": "..."
        }
        """
        result = {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "judge": None,
            "simple_metrics": {},
            "timestamp": self.timestamp,
        }

        for metric in self.metrics:
            if metric.name == "llm_judge":
                # LLM Judge goes in its own section
                result["judge"] = {
                    "llm_as_a_judge_score": metric.details.get("llm_as_a_judge_score", metric.score),
                    "pred": metric.details.get("pred", "no"),
                    "score_0_5": metric.details.get("score_0_5", metric.score),
                    "score_normalized": metric.details.get("score_normalized", metric.score / 5.0),
                    "rationale": metric.details.get("rationale", ""),
                    "api_status": metric.details.get("api_status", "unknown"),
                    "eval_api_latency_seconds": metric.details.get("eval_api_latency_seconds", 0),
                }
            else:
                # All other metrics go in simple_metrics
                result["simple_metrics"][metric.name] = {
                    "score": metric.score,
                    "details": metric.details,
                }

        # Add file list metrics if available
        if self.file_list_metrics:
            result["file_list_metrics"] = self.file_list_metrics

        return result


# Available metric types
AVAILABLE_METRICS: Dict[str, Type[MetricCalculator]] = {
    "rouge": ROUGEMetric,
    "bleu": BLEUMetric,
    "exact_match": ExactMatchMetric,
    "covered_exact_match": CoveredExactMatchMetric,
    "semantic_similarity": SemanticSimilarityMetric,
    "bert_score": BERTScoreMetric,
    "retrieval_precision": RetrievalPrecisionMetric,
    "retrieval_recall": RetrievalRecallMetric,
    "retrieval_f1": RetrievalF1Metric,
    "llm_judge": LLMJudgeMetric,
}


class EvaluationRunner:
    """
    Runner for evaluating RAG query results with multiple metrics.

    Usage:
        runner = EvaluationRunner(metrics=["rouge", "bleu", "semantic_similarity"])
        results = await runner.evaluate_results(query_results)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        metric_instances: Optional[List[MetricCalculator]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the evaluation runner.

        Args:
            metrics: List of metric names to use (e.g., ["rouge", "bleu"])
            metric_instances: Pre-configured metric instances (overrides metrics param)
            metric_kwargs: Per-metric keyword arguments, keyed by metric name.
                Example: {"llm_judge": {"prompt_template": "detailed"}}
        """
        self._metric_kwargs = metric_kwargs or {}

        if metric_instances:
            self.metrics = metric_instances
        elif metrics:
            self.metrics = []
            for name in metrics:
                if name not in AVAILABLE_METRICS:
                    logger.warning(f"Unknown metric: {name}, skipping")
                    continue
                try:
                    kwargs = self._metric_kwargs.get(name, {})
                    metric = AVAILABLE_METRICS[name](**kwargs)
                    self.metrics.append(metric)
                except ImportError as e:
                    logger.warning(f"Could not initialize {name}: {e}")
        else:
            # Default metrics
            self.metrics = []
            for name in ["rouge", "bleu"]:
                try:
                    self.metrics.append(AVAILABLE_METRICS[name]())
                except ImportError as e:
                    logger.warning(f"Could not initialize {name}: {e}")

        logger.info(f"Initialized EvaluationRunner with metrics: {[m.name for m in self.metrics]}")

    async def evaluate_single(
        self,
        query_id: str,
        query: str,
        answer: Optional[str],
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        expected_chunks: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single query result.

        Args:
            query_id: Unique identifier for the query
            query: The original query
            answer: Generated answer
            ground_truth: Expected answer (for answer quality metrics)
            retrieved_chunks: List of retrieved chunk dicts
            expected_chunks: List of expected chunk IDs (for retrieval metrics)

        Returns:
            EvaluationResult with all computed metrics
        """
        # Convert chunk dicts to RetrievedChunk objects
        chunks = None
        if retrieved_chunks:
            chunks = [RetrievedChunk.from_dict(c) for c in retrieved_chunks]

        # Compute all metrics
        metric_results = []
        for metric in self.metrics:
            try:
                result = await metric.calculate(
                    query=query,
                    generated_answer=answer or "",
                    ground_truth=ground_truth,
                    retrieved_chunks=chunks,
                    expected_chunks=expected_chunks,
                )
                metric_results.append(result)
                logger.debug(f"[{query_id}] {metric.name}: {result.score:.4f}")
            except Exception as e:
                logger.error(f"[{query_id}] {metric.name} failed: {e}")
                metric_results.append(EvaluationMetric(
                    name=metric.name,
                    score=0.0,
                    details={"error": str(e)}
                ))

        return EvaluationResult(
            query_id=query_id,
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            metrics=metric_results,
        )

    async def evaluate_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of query results.

        Args:
            results: List of QueryResultRecord dicts

        Returns:
            List of EvaluationResult objects
        """
        evaluations = []
        total = len(results)

        for i, result in enumerate(results, 1):
            logger.info(f"[{i}/{total}] Evaluating query: {result.get('query_id', 'unknown')}")

            eval_result = await self.evaluate_single(
                query_id=result.get("query_id", f"q{i}"),
                query=result.get("query", ""),
                answer=result.get("answer"),
                ground_truth=result.get("ground_truth"),
                retrieved_chunks=result.get("retrieved_chunks"),
                expected_chunks=result.get("expected_chunks"),
            )
            evaluations.append(eval_result)

        return evaluations

    async def evaluate_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        structured_format: bool = True,
    ) -> List[EvaluationResult]:
        """
        Load results from JSON file, evaluate, and optionally save.

        Args:
            input_file: Path to JSON file with query results
            output_file: Optional path to save evaluation results
            structured_format: If True, use structured format with judge/simple_metrics separation

        Returns:
            List of EvaluationResult objects
        """
        # Load results
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single result and list of results
        if isinstance(data, dict):
            results = [data]
        else:
            results = data

        # Evaluate
        evaluations = await self.evaluate_results(results)

        # Save if requested
        if output_file:
            if structured_format:
                output_data = [e.to_structured_dict() for e in evaluations]
            else:
                output_data = [e.to_dict() for e in evaluations]
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to: {output_file}")

        return evaluations

    def get_summary(
        self,
        evaluations: List[EvaluationResult],
        input_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for evaluation results.

        Args:
            evaluations: List of EvaluationResult objects
            input_data: Optional list of original query result dicts (for latency/step_counts)

        Returns:
            Dictionary with summary statistics
        """
        if not evaluations:
            return {"error": "No evaluations to summarize"}

        # Aggregate scores by metric
        metric_scores: Dict[str, List[float]] = {}
        # Track yes/no predictions for llm_judge
        llm_judge_preds: List[str] = []

        for eval_result in evaluations:
            for metric in eval_result.metrics:
                if metric.name not in metric_scores:
                    metric_scores[metric.name] = []
                metric_scores[metric.name].append(metric.score)

                # Track yes/no predictions for llm_judge
                if metric.name == "llm_judge" and metric.details:
                    pred = metric.details.get("pred", "").lower()
                    if pred in ("yes", "no"):
                        llm_judge_preds.append(pred)

        # Calculate statistics
        summary: Dict[str, Any] = {
            "total_queries": len(evaluations),
            "metrics": {}
        }

        for name, scores in metric_scores.items():
            if name == "llm_judge":
                # Simplified llm_judge: only mean, yes_count, accuracy
                yes_count = sum(1 for p in llm_judge_preds if p == "yes")
                total_preds = len(llm_judge_preds)
                accuracy = yes_count / total_preds if total_preds > 0 else 0

                summary["metrics"][name] = {
                    "mean": sum(scores) / len(scores) if scores else 0,
                    "yes_count": yes_count,
                    "accuracy": accuracy,
                }
            else:
                summary["metrics"][name] = {
                    "mean": sum(scores) / len(scores) if scores else 0,
                }

        # Aggregate latency_breakdown and step_counts from input data
        if input_data:
            latency_keys: Dict[str, List[float]] = {}
            step_keys: Dict[str, List[float]] = {}

            for item in input_data:
                lb = item.get("latency_breakdown")
                if lb and isinstance(lb, dict):
                    for k, v in lb.items():
                        if isinstance(v, (int, float)):
                            if k not in latency_keys:
                                latency_keys[k] = []
                            latency_keys[k].append(float(v))

                sc = item.get("step_counts")
                if sc and isinstance(sc, dict):
                    for k, v in sc.items():
                        if isinstance(v, (int, float)):
                            if k not in step_keys:
                                step_keys[k] = []
                            step_keys[k].append(float(v))

            if latency_keys:
                summary["latency_breakdown"] = {
                    k: {
                        "mean": sum(v) / len(v),
                    }
                    for k, v in latency_keys.items()
                }

            if step_keys:
                summary["step_counts"] = {
                    k: {
                        "mean": sum(v) / len(v),
                    }
                    for k, v in step_keys.items()
                }

        return summary
