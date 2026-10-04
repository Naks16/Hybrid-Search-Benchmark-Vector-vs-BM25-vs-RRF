"""Retrieve chunks for a query with mode "vector", "bm25" or "hybrid"."""

import numpy as np
from rank_bm25 import BM25Okapi

import config
from search.fusion import reciprocal_rank_fusion
from search.tokenizer import tokenize

MODES = ("vector", "bm25", "hybrid")


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix.

    cos(a, b) = (a . b) / (|a| * |b|). Our embeddings are already unit length,
    so this equals the dot product, but we divide anyway so the function is
    correct for any input.
    """
    dot_products = matrix @ query_vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return dot_products / norms


class Retriever:
    """Holds both indexes over the same list of chunks.

    Searches return (chunk_index, score) pairs, best first; chunk_index is
    the position in self.chunks (and the row in self.embeddings).
    """

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, model):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
        self.index_of_id = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}

    def vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vector = self.model.encode_query(query, normalize_embeddings=True)
        scores = cosine_similarity(query_vector, self.embeddings)
        # kind="stable": equal scores keep chunk order, so results are repeatable.
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(i), float(scores[i])) for i in best]

    def bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(-scores, kind="stable")[:top_k]
        # Score 0 means no query word appears in the chunk: not a real match.
        # Dropping these means BM25 can return fewer than top_k results, but
        # it stops random chunks from earning RRF credit in hybrid mode.
        return [(int(i), float(scores[i])) for i in best if scores[i] > 0]

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        candidates: int = config.HYBRID_CANDIDATES,
        rrf_k: int = config.RRF_K,
    ) -> list[tuple[int, float]]:
        """Top `candidates` from each retriever, fused with RRF.

        candidates and rrf_k default to config; the eval sweep overrides them.
        """
        vector_hits = self.vector_search(query, candidates)
        bm25_hits = self.bm25_search(query, candidates)
        vector_ids = [self.chunks[i]["chunk_id"] for i, _ in vector_hits]
        bm25_ids = [self.chunks[i]["chunk_id"] for i, _ in bm25_hits]
        fused = reciprocal_rank_fusion([vector_ids, bm25_ids], k=rrf_k)
        return [(self.index_of_id[chunk_id], score) for chunk_id, score in fused[:top_k]]

    def retrieve(self, query: str, mode: str, top_k: int = 5) -> list[dict]:
        """Returns [{chunk_id, source, score, rank, text}], rank 1 = best.

        score meaning depends on mode: cosine similarity (vector),
        BM25 score (bm25) or RRF score (hybrid).
        """
        if mode == "vector":
            hits = self.vector_search(query, top_k)
        elif mode == "bm25":
            hits = self.bm25_search(query, top_k)
        elif mode == "hybrid":
            hits = self.hybrid_search(query, top_k)
        else:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        results = []
        for rank, (i, score) in enumerate(hits, start=1):
            chunk = self.chunks[i]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": score,
                "rank": rank,
                "text": chunk["text"],
            })
        return results


_default_retriever = None


def retrieve(query: str, mode: str, top_k: int = 5) -> list[dict]:
    """retrieve() over the saved index in config.INDEX_DIR.

    The model and index are loaded once, on the first call, then reused.
    """
    global _default_retriever
    if _default_retriever is None:
        from search.index import load_index, load_model  # heavy import, only when needed

        chunks, embeddings = load_index()
        _default_retriever = Retriever(chunks, embeddings, load_model())
    return _default_retriever.retrieve(query, mode, top_k)
