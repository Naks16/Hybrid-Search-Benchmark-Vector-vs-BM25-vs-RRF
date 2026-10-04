import pytest

from search.fusion import reciprocal_rank_fusion


def test_rrf_order_with_item_in_only_one_list():
    vector_ranking = ["a", "b", "c"]
    bm25_ranking = ["b", "d", "a"]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=60)

    # b: 1/62 + 1/61 = 0.03252   (2nd and 1st)
    # a: 1/61 + 1/63 = 0.03227   (1st and 3rd)
    # d: 1/62        = 0.01613   (only in BM25, rank 2)
    # c: 1/63        = 0.01587   (only in vector, rank 3)
    assert [item_id for item_id, _ in fused] == ["b", "a", "d", "c"]

    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)  # one list only: no penalty, just less credit
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_single_list_keeps_its_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]])
    assert [item_id for item_id, _ in fused] == ["x", "y", "z"]
