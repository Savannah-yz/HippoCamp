"""
Semantic similarity metrics for RAG evaluation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .base import MetricCalculator, EvaluationMetric, RetrievedChunk

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError:
    SEMANTIC_DEPS_AVAILABLE = False

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


class SemanticSimilarityMetric(MetricCalculator):
    """Semantic similarity using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__("semantic_similarity")
        self.model_name = model_name
        self._model = None

        if not SEMANTIC_DEPS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and scikit-learn required for semantic similarity. "
                "Install with: pip install sentence-transformers scikit-learn"
            )

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate semantic similarity between generated answer and ground truth."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            model = self._get_model()

            # Encode texts
            embeddings = model.encode([generated_answer, ground_truth])

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity_score = float(similarity_matrix[0][0])

            details = {
                "model_used": self.model_name,
                "generated_length": len(generated_answer),
                "ground_truth_length": len(ground_truth)
            }

            return self._create_metric(similarity_score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class BERTScoreMetric(MetricCalculator):
    """BERTScore metric for evaluating generated text."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        super().__init__("bert_score")
        self.model_type = model_type

        if not BERT_SCORE_AVAILABLE:
            raise ImportError(
                "bert-score required for BERTScore metric. "
                "Install with: pip install bert-score"
            )

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate BERTScore between generated answer and ground truth."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            # Calculate BERTScore
            P, R, F1 = bert_score.score(
                [generated_answer],
                [ground_truth],
                model_type=self.model_type,
                verbose=False
            )

            # Use F1 score as the main metric
            f1_score = float(F1.mean())

            details = {
                "precision": float(P.mean()),
                "recall": float(R.mean()),
                "f1": f1_score,
                "model_type": self.model_type
            }

            return self._create_metric(f1_score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})
