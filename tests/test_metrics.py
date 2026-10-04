import pytest

from eval.analysis import find_win_loss
from eval.metrics import gold_rank, hit_rate, mrr


def test_gold_rank_uses_best_ranked_gold_chunk():
    retrieved = ["x", "g2", "y", "g1"]
    assert gold_rank(retrieved, ["g1", "g2"]) == 2
    assert gold_rank(retrieved, ["missing"]) is None


def test_hit_rate_and_mrr():
    ranks = [1, 3, None, 12]  # 12 is outside the @10 cutoff
    assert hit_rate(ranks, 1) == pytest.approx(1 / 4)
    assert hit_rate(ranks, 3) == pytest.approx(2 / 4)
    assert mrr(ranks, cutoff=10) == pytest.approx((1 + 1 / 3) / 4)


def test_win_loss_categories():
    ranks = {
        "a": {"vector": 1, "bm25": 5, "hybrid": 1},        # vector beats bm25 by 4
        "b": {"vector": None, "bm25": 2, "hybrid": 3},     # bm25 beats vector; hybrid worse than bm25
        "c": {"vector": 4, "bm25": 3, "hybrid": 1},        # hybrid beats both
    }
    found = find_win_loss(ranks)
    assert found["vector_beats_bm25"] == [("a", 4)]
    assert found["bm25_beats_vector"] == [("b", 9)]  # missing counts as rank 11
    assert found["hybrid_beats_both"] == [("c", 2)]
    assert found["hybrid_worse_than_best"] == [("b", 1)]
