"""Evaluation metrics for information retrieval and search ranking."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int) -> float:
    """Compute Recall@k (Hit Rate): evaluates whether any relevant document appears in top-k.

    Parameters
    ----------
    retrieved_ids : Sequence[str]
        Ranked list of document/chunk IDs returned by search engine.
    target_ids : Sequence[str]
        Ground-truth relevant document/chunk IDs.
    k : int
        Cutoff rank threshold.

    Returns
    -------
    float
        1.0 if at least one relevant document is found in top-k, otherwise 0.0.
    """
    if k <= 0 or not target_ids or not retrieved_ids:
        return 0.0

    top_k_set = set(retrieved_ids[:k])
    target_set = set(target_ids)

    return 1.0 if bool(top_k_set & target_set) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], target_ids: Sequence[str]) -> float:
    """Compute Reciprocal Rank (RR) of the first relevant document.

    RR = 1 / rank, where rank is the 1-based position of the first matching target document.
    Returns 0.0 if no target document is found in retrieved results.

    Parameters
    ----------
    retrieved_ids : Sequence[str]
        Ranked list of document/chunk IDs returned by search engine.
    target_ids : Sequence[str]
        Ground-truth relevant document/chunk IDs.

    Returns
    -------
    float
        Reciprocal rank value in [0.0, 1.0].
    """
    if not target_ids or not retrieved_ids:
        return 0.0

    target_set = set(target_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in target_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[str]], all_targets: Sequence[Sequence[str]]
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across a collection of queries.

    Parameters
    ----------
    all_retrieved : Sequence[Sequence[str]]
        List of retrieved ID sequences for each query.
    all_targets : Sequence[Sequence[str]]
        List of target ID sequences for each query.

    Returns
    -------
    float
        Average reciprocal rank across all evaluated queries.
    """
    if not all_retrieved or not all_targets or len(all_retrieved) != len(all_targets):
        return 0.0

    total_rr = sum(
        reciprocal_rank(retrieved, targets)
        for retrieved, targets in zip(all_retrieved, all_targets, strict=True)
    )
    return float(total_rr / len(all_retrieved))
