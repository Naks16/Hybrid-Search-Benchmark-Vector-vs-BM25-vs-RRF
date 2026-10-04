"""Reciprocal Rank Fusion (RRF), written by hand."""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked lists of ids into one ranked list.

    Formula: for every id d,
        RRF(d) = sum over each list L that contains d of  1 / (k + rank_L(d))
    where rank starts at 1 for the best item. An id missing from a list gets
    nothing from that list (so it is not penalised, it just earns less).

    Why rank and not the raw scores? The scores are on different scales:
    cosine similarity is roughly 0..1, BM25 is unbounded (e.g. 0..30) and
    depends on query length. Adding them would let BM25 dominate, and
    normalising them is fragile. Ranks are always comparable: "1st" means
    the same thing in every list.

    Why k? It controls how quickly credit drops with rank. With k=60, rank 1
    gives 1/61 and rank 10 gives 1/70, so a document ranked well in BOTH
    lists beats one ranked 1st in only one list.

    Returns [(id, rrf_score)] sorted best first. Ties keep the order in which
    ids were first seen (Python's sort is stable).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
