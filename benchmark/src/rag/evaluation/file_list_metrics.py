"""
File List Metrics for RAG Evaluation.

Evaluates the quality of retrieved files by comparing retrieved_file_list
against the expected file_list from the benchmark.

Metrics Calculated:
- F1 Score: Harmonic mean of precision and recall
- Recall: TP / (TP + FN) - how many ground truth files were found
- Precision: TP / (TP + FP) - how many predicted files were correct

Usage:
    from src.rag.evaluation.file_list_metrics import calculate_file_list_metrics

    metrics = calculate_file_list_metrics(
        file_list=["path/to/file1.pdf", "path/to/file2.docx"],
        retrieved_file_list=["path/to/file1.pdf", "path/to/file3.txt"]
    )
    # Returns: {"f1_score": 0.5, "recall": 0.5, "precision": 0.5}
"""

from typing import Any, Dict, List, Optional


def calculate_file_list_metrics(
    file_list: Optional[List[str]],
    retrieved_file_list: Optional[List[str]],
) -> Dict[str, float]:
    """
    Calculate file list matching metrics.

    Args:
        file_list: List of expected/required file paths (ground truth)
        retrieved_file_list: List of file paths from retrieved chunks (predictions)

    Returns:
        Dictionary with metrics:
        - f1_score: Harmonic mean of precision and recall (0-1)
        - recall: TP / (TP + FN) - proportion of GT files found (0-1)
        - precision: TP / (TP + FP) - proportion of correct predictions (0-1)
    """
    # Handle empty or None cases
    if not file_list:
        # No ground truth files - can't evaluate
        return {
            "f1_score": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }

    if not retrieved_file_list:
        # No retrieved files - all misses
        return {
            "f1_score": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }

    # Convert to sets for efficient comparison
    gt_set = set(file_list)
    pred_set = set(retrieved_file_list)

    # Calculate true positives (intersection)
    tp = len(gt_set & pred_set)

    # Calculate precision and recall
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0

    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "f1_score": f1_score,
        "recall": recall,
        "precision": precision,
    }


def calculate_batch_file_list_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate aggregated file list metrics for a batch of results.

    Args:
        results: List of result dictionaries, each containing:
            - file_list: List of expected file paths
            - retrieved_file_list: List of retrieved file paths

    Returns:
        Dictionary with aggregated metrics:
        - total_evaluated: Number of queries with file_list data
        - average_f1_score: Mean F1 score across all queries
        - average_recall: Mean recall across all queries
        - average_precision: Mean precision across all queries
    """
    f1_scores = []
    recalls = []
    precisions = []

    for result in results:
        file_list = result.get("file_list")
        retrieved_file_list = result.get("retrieved_file_list")

        # Skip if no file_list in ground truth
        if not file_list:
            continue

        metrics = calculate_file_list_metrics(file_list, retrieved_file_list)
        f1_scores.append(metrics["f1_score"])
        recalls.append(metrics["recall"])
        precisions.append(metrics["precision"])

    if not f1_scores:
        return {
            "total_evaluated": 0,
            "average_f1_score": 0.0,
            "average_recall": 0.0,
            "average_precision": 0.0,
        }

    return {
        "total_evaluated": len(f1_scores),
        "average_f1_score": sum(f1_scores) / len(f1_scores),
        "average_recall": sum(recalls) / len(recalls),
        "average_precision": sum(precisions) / len(precisions),
    }


def add_file_list_metrics_to_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add file_list_metrics to a single result dictionary.

    Args:
        result: Result dictionary with file_list and retrieved_file_list

    Returns:
        The same result dictionary with file_list_metrics added
    """
    file_list = result.get("file_list")
    retrieved_file_list = result.get("retrieved_file_list")

    if file_list:
        result["file_list_metrics"] = calculate_file_list_metrics(
            file_list, retrieved_file_list
        )
    else:
        result["file_list_metrics"] = None

    return result
