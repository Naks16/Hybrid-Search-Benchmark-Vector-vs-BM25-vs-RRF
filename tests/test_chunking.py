from search.chunking import chunk_documents
from search.loader import join_hyphenated_line_breaks, load_documents
from search.tokenizer import tokenize
from tests.conftest import FIXTURE_DOCS, TEST_CHUNK_OVERLAP, TEST_CHUNK_SIZE


def build_chunks(model):
    documents = load_documents(FIXTURE_DOCS)
    return chunk_documents(documents, model.tokenizer, TEST_CHUNK_SIZE, TEST_CHUNK_OVERLAP)


def test_chunk_ids_are_stable_across_runs(model):
    first_run = build_chunks(model)
    second_run = build_chunks(model)

    assert len(first_run) > len(load_documents(FIXTURE_DOCS))  # some files really were split
    assert [c["chunk_id"] for c in first_run] == [c["chunk_id"] for c in second_run]
    assert first_run == second_run  # same text and source for every id too


def test_chunk_id_format_and_source(model):
    for chunk in build_chunks(model):
        filename, index = chunk["chunk_id"].split("::")
        assert filename == chunk["source"]
        assert index.isdigit()


def test_bm25_tokenizer_keeps_identifiers():
    tokens = tokenize("Raises ValueError if n_estimators < 1.")
    assert "valueerror" in tokens and "value" in tokens and "error" in tokens
    assert "n_estimators" in tokens and "estimators" in tokens
    assert "<" not in tokens and "." not in tokens


def test_pdf_line_break_hyphens_are_joined():
    text = "pricing American op-\ntions on a cross-chain bridge.\nEvery cross-\nchain hop"
    fixed = join_hyphenated_line_breaks(text)
    assert "options" in fixed  # split word joined
    assert "cross-chain hop" in fixed  # real compound (seen elsewhere) keeps its hyphen
