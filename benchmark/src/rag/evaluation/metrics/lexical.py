"""
Lexical similarity metrics for RAG evaluation.
"""

from typing import Any, Dict, List, Optional
from .base import MetricCalculator, EvaluationMetric, RetrievedChunk

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class ROUGEMetric(MetricCalculator):
    """ROUGE metrics for evaluating generated text."""

    def __init__(self, rouge_types: List[str] = None):
        super().__init__("rouge")
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']

        if not ROUGE_AVAILABLE:
            raise ImportError(
                "rouge-score required for ROUGE metric. "
                "Install with: pip install rouge-score"
            )

        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate ROUGE scores between generated answer and ground truth."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            scores = self.scorer.score(ground_truth, generated_answer)

            # Extract F1 scores for each ROUGE type
            rouge_scores = {}
            total_f1 = 0.0

            for rouge_type in self.rouge_types:
                if rouge_type in scores:
                    f1_score = scores[rouge_type].fmeasure
                    rouge_scores[f"{rouge_type}_f1"] = f1_score
                    rouge_scores[f"{rouge_type}_precision"] = scores[rouge_type].precision
                    rouge_scores[f"{rouge_type}_recall"] = scores[rouge_type].recall
                    total_f1 += f1_score

            # Average F1 score across all ROUGE types
            avg_f1 = total_f1 / len(self.rouge_types) if self.rouge_types else 0.0

            details = {
                "rouge_types": self.rouge_types,
                **rouge_scores
            }

            return self._create_metric(avg_f1, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class BLEUMetric(MetricCalculator):
    """BLEU score for evaluating generated text."""

    def __init__(self, max_n: int = 4):
        super().__init__("bleu")
        self.max_n = max_n

        if not NLTK_AVAILABLE:
            raise ImportError(
                "nltk required for BLEU metric. "
                "Install with: pip install nltk"
            )

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate BLEU score between generated answer and ground truth."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            # Tokenize sentences
            reference_tokens = nltk.word_tokenize(ground_truth.lower())
            candidate_tokens = nltk.word_tokenize(generated_answer.lower())

            # Calculate BLEU score with smoothing
            smoothing_function = SmoothingFunction().method1

            # Calculate individual n-gram scores
            weights = [1.0/self.max_n] * self.max_n
            bleu_score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )

            # Calculate individual BLEU-n scores
            individual_scores = {}
            for n in range(1, self.max_n + 1):
                n_weights = [0.0] * self.max_n
                n_weights[n-1] = 1.0
                n_score = sentence_bleu(
                    [reference_tokens],
                    candidate_tokens,
                    weights=n_weights,
                    smoothing_function=smoothing_function
                )
                individual_scores[f"bleu_{n}"] = n_score

            details = {
                "max_n": self.max_n,
                "reference_length": len(reference_tokens),
                "candidate_length": len(candidate_tokens),
                **individual_scores
            }

            return self._create_metric(bleu_score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact match comparison.

    Following standard QA evaluation practices (SQuAD, Natural Questions, etc.):
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    import re
    import string

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()


class ExactMatchMetric(MetricCalculator):
    """
    Exact Match (EM) metric for QA evaluation.

    Returns 1.0 if the normalized generated answer exactly matches
    the normalized ground truth, 0.0 otherwise.

    This is a standard metric used in SQuAD, Natural Questions, TriviaQA, etc.
    """

    def __init__(self):
        super().__init__("exact_match")

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate Exact Match score."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            # Normalize both answers
            norm_generated = normalize_answer(generated_answer)
            norm_ground_truth = normalize_answer(ground_truth)

            # Exact match check
            is_match = norm_generated == norm_ground_truth
            score = 1.0 if is_match else 0.0

            details = {
                "normalized_generated": norm_generated,
                "normalized_ground_truth": norm_ground_truth,
                "is_exact_match": is_match,
            }

            return self._create_metric(score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})


class CoveredExactMatchMetric(MetricCalculator):
    """
    Covered Exact Match metric for QA evaluation.

    Returns 1.0 if the normalized ground truth is contained within
    the normalized generated answer, 0.0 otherwise.

    This is useful when the generated answer may contain additional
    context but still includes the correct answer.

    Also known as "Answer Containment" or "Substring Match" in some papers.
    """

    def __init__(self):
        super().__init__("covered_exact_match")

    async def calculate(
        self,
        query: str,
        generated_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        expected_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationMetric:
        """Calculate Covered Exact Match score."""
        if not ground_truth:
            return self._create_metric(0.0, {"error": "No ground truth provided"})

        try:
            # Normalize both answers
            norm_generated = normalize_answer(generated_answer)
            norm_ground_truth = normalize_answer(ground_truth)

            # Check if ground truth is contained in generated answer
            is_covered = norm_ground_truth in norm_generated
            score = 1.0 if is_covered else 0.0

            details = {
                "normalized_generated": norm_generated,
                "normalized_ground_truth": norm_ground_truth,
                "is_covered": is_covered,
            }

            return self._create_metric(score, details)

        except Exception as e:
            return self._create_metric(0.0, {"error": str(e)})
