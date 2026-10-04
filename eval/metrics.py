"""Retrieval metrics. Pure functions over gold ranks; no model, no LLM judge.

A "gold rank" is the 1-based position of the best-ranked gold chunk in the
retrieved list, or None if no gold chunk was retrieved.
"""

import numpy as np


def gold_rank(retrieved_ids: list[str], gold_ids: list[str]) -> int | None:
    """Rank of the first retrieved chunk that is a gold chunk, else None.

    Some questions have several gold chunks (e.g. the answer sits in the
    overlap between two neighbouring chunks); finding any one of them counts.
    """
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose gold chunk is in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], cutoff: int = 10) -> float:
    """Mean Reciprocal Rank@cutoff: average of 1/rank, counting 0 when the
    gold chunk is missing or ranked below the cutoff."""
    return sum(1 / r for r in ranks if r is not None and r <= cutoff) / len(ranks)


def latency_summary(latencies_ms: list[float]) -> tuple[float, float]:
    """(mean, p95) in milliseconds."""
    return float(np.mean(latencies_ms)), float(np.percentile(latencies_ms, 95))
