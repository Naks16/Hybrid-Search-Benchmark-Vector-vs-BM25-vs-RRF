"""Shared test fixtures: the embedding model and a tiny corpus."""

from pathlib import Path

import pytest

from search.index import load_model

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"

# Small chunk settings so the tiny fixture files still produce several chunks.
TEST_CHUNK_SIZE = 30
TEST_CHUNK_OVERLAP = 5


@pytest.fixture(scope="session")
def model():
    """Load the model once for the whole test run (it is slow to load)."""
    return load_model()
