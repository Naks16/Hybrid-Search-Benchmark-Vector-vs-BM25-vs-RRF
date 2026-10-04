import pytest

from search.chunking import chunk_documents
from search.index import embed_texts
from search.loader import load_documents
from search.retrieve import MODES, Retriever
from tests.conftest import FIXTURE_DOCS, TEST_CHUNK_OVERLAP, TEST_CHUNK_SIZE

TOP_K = 3


@pytest.fixture(scope="module")
def retriever(model):
    documents = load_documents(FIXTURE_DOCS)
    chunks = chunk_documents(documents, model.tokenizer, TEST_CHUNK_SIZE, TEST_CHUNK_OVERLAP)
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunks])
    return Retriever(chunks, embeddings, model)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_top_k(retriever, mode):
    results = retriever.retrieve("how many trees does n_estimators set", mode, top_k=TOP_K)

    assert len(results) == TOP_K
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert set(results[0]) == {"chunk_id", "source", "score", "rank", "text"}
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("trees", "magic")
