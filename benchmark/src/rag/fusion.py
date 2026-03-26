"""
Reciprocal Rank Fusion (RRF) utility for merging ranked result lists.

Generalised from QdrantVectorStore._rrf_fuse() to work with any result
dictionaries that have an "id" key.
"""

from typing import Any, Dict, List

DEFAULT_RRF_K = 60


def rrf_fuse(
    *result_lists: List[Dict[str, Any]],
    k: int = DEFAULT_RRF_K,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each result dict must contain at least ``"id"`` (str).  The first
    occurrence of a given id supplies the payload (``chunk``, ``metadata``,
    etc.) that is propagated to the output.

    Args:
        *result_lists: One or more ranked result lists (highest-relevance
            first).
        k: RRF smoothing constant (default 60).
        limit: Maximum results to return.

    Returns:
        Fused results sorted by descending RRF score, each dict containing
        the original payload fields plus an updated ``"score"`` (the RRF
        score).
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            result_id = str(result["id"])
            scores[result_id] = scores.get(result_id, 0.0) + 1.0 / (k + rank)
            if result_id not in payloads:
                payloads[result_id] = result

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused: List[Dict[str, Any]] = []
    for result_id, score in ranked[:limit]:
        payload = payloads[result_id]
        fused.append({
            "id": result_id,
            "chunk": payload.get("chunk", ""),
            "metadata": payload.get("metadata", {}),
            "score": score,
        })
    return fused
