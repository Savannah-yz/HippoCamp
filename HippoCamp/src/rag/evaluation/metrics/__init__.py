"""
Evaluation metrics for RAG systems.

Metrics include:
- Lexical: ROUGE, BLEU
- Semantic: Semantic Similarity, BERTScore
- Retrieval: Precision, Recall, F1, MRR
- LLM Judge: Uses LLM to evaluate answer quality
"""

from .base import MetricCalculator, EvaluationMetric, RetrievedChunk
from .lexical import ROUGEMetric, BLEUMetric, ExactMatchMetric, CoveredExactMatchMetric
from .semantic import SemanticSimilarityMetric, BERTScoreMetric
from .retrieval import (
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
    RetrievalF1Metric,
    RetrievalMRRMetric,
)
from .llm_judge import LLMJudgeMetric

__all__ = [
    'MetricCalculator',
    'EvaluationMetric',
    'RetrievedChunk',
    'ROUGEMetric',
    'BLEUMetric',
    'ExactMatchMetric',
    'CoveredExactMatchMetric',
    'SemanticSimilarityMetric',
    'BERTScoreMetric',
    'RetrievalPrecisionMetric',
    'RetrievalRecallMetric',
    'RetrievalF1Metric',
    'RetrievalMRRMetric',
    'LLMJudgeMetric',
]
