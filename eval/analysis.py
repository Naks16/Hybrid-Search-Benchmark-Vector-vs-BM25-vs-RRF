"""Win/loss analysis between retrieval modes, written to analysis.md.

Every explanation is built from real retrieval data for that question:
the BM25 tokens the query shares with the gold chunk (and in how many chunks
each token appears), the gold chunk's full rank in each mode, cosine
similarities, each mode's top hit, and for hybrid the ranks that RRF fused.
Nothing is guessed by a model.
"""

from collections import Counter

import numpy as np

import config
from search.retrieve import Retriever, cosine_similarity
from search.tokenizer import tokenize

MISSING = 11  # a gold chunk not in the top 10 is treated as rank 11 when comparing
TOP_EXAMPLES = 5
PREVIEW_CHARS = 300
# A token is called "rare" if it appears in at most this share of chunks.
RARE_SHARE = 0.10

CATEGORIES = {
    "bm25_beats_vector": "BM25 ranked the gold chunk at least 3 places better than vector",
    "vector_beats_bm25": "Vector ranked the gold chunk at least 3 places better than BM25",
    "hybrid_beats_both": "Hybrid ranked the gold chunk better than both vector and BM25",
    "hybrid_worse_than_best": "Hybrid ranked the gold chunk worse than the best single mode",
}


def as_number(rank: int | None) -> int:
    return MISSING if rank is None else rank


def show(rank: int | None) -> str:
    return "not in top 10" if rank is None else str(rank)


def find_win_loss(ranks: dict[str, dict[str, int | None]]) -> dict[str, list[tuple[str, int]]]:
    """ranks = {question_id: {"vector": r, "bm25": r, "hybrid": r}}.

    Returns {category: [(question_id, margin), ...]} sorted by biggest margin
    first (ties by question id), so the top entries are the clearest cases.
    """
    found = {name: [] for name in CATEGORIES}
    for qid, r in ranks.items():
        v, b, h = as_number(r["vector"]), as_number(r["bm25"]), as_number(r["hybrid"])
        best_single = min(v, b)
        if v - b >= 3:
            found["bm25_beats_vector"].append((qid, v - b))
        if b - v >= 3:
            found["vector_beats_bm25"].append((qid, b - v))
        if h < best_single:
            found["hybrid_beats_both"].append((qid, best_single - h))
        if h > best_single:
            found["hybrid_worse_than_best"].append((qid, h - best_single))
    for name in found:
        found[name].sort(key=lambda pair: (-pair[1], pair[0]))
    return found


class Explainer:
    """Collects the facts behind one question's ranks and turns them into text."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.num_chunks = len(retriever.chunks)
        # In how many chunks each token appears (BM25's document frequency).
        self.doc_freq = Counter(t for doc in retriever.bm25.doc_freqs for t in doc)

    def shared_rare_terms(self, query_tokens: set[str], chunk_index: int) -> list[str]:
        """Rare query tokens that also occur in the chunk, rarest first."""
        chunk_tokens = set(tokenize(self.retriever.chunks[chunk_index]["text"]))
        rare = [t for t in query_tokens & chunk_tokens if self.doc_freq[t] <= RARE_SHARE * self.num_chunks]
        return sorted(rare, key=lambda t: (self.doc_freq[t], t))

    def terms_text(self, terms: list[str], limit: int = 4) -> str:
        if not terms:
            return "no rare words"
        return ", ".join(f"'{t}' (in {self.doc_freq[t]} of {self.num_chunks} chunks)" for t in terms[:limit])

    @staticmethod
    def full_rank(scores: np.ndarray, index: int) -> int:
        """Rank of one chunk among ALL chunks (not just the top 10)."""
        return int((scores > scores[index]).sum()) + 1

    def explain(self, category: str, question: dict, ranks: dict, retrieved: dict) -> str:
        r = self.retriever
        query = question["question"]
        query_tokens = set(tokenize(query))
        gold_index = self.gold_index(question, retrieved)
        gold_id = r.chunks[gold_index]["chunk_id"]
        gold_source = r.chunks[gold_index]["source"]

        cosines = cosine_similarity(r.model.encode_query(query, normalize_embeddings=True), r.embeddings)
        bm25_scores = r.bm25.get_scores(tokenize(query))
        gold_rare = self.shared_rare_terms(query_tokens, gold_index)
        vector_full = self.full_rank(cosines, gold_index)
        bm25_full = self.full_rank(bm25_scores, gold_index)

        def where(chunk_id):
            source = r.chunks[r.index_of_id[chunk_id]]["source"]
            return "same paper" if source == gold_source else f"from {source}"

        if category == "bm25_beats_vector":
            top_vec = retrieved["vector"][0]
            return (
                f"Rare words shared by question and gold chunk: {self.terms_text(gold_rare)}; BM25 gives "
                f"the gold chunk score {bm25_scores[gold_index]:.1f}. The embedding puts it at rank "
                f"{vector_full} of {self.num_chunks} (cosine {cosines[gold_index]:.3f}) behind its top hit "
                f"{top_vec} ({where(top_vec)}, cosine {cosines[r.index_of_id[top_vec]]:.3f})."
            )

        if category == "vector_beats_bm25":
            text = (f"Question and gold chunk share {self.terms_text(gold_rare)}, so BM25 ranks the gold "
                    f"chunk {bm25_full} of {self.num_chunks} (score {bm25_scores[gold_index]:.1f})")
            if retrieved["bm25"]:
                top_id = retrieved["bm25"][0]
                top_i = r.index_of_id[top_id]
                text += (f" while its top hit {top_id} ({where(top_id)}, score {bm25_scores[top_i]:.1f}) "
                         f"matches {self.terms_text(self.shared_rare_terms(query_tokens, top_i), 3)}")
            return text + (f". Vector does not depend on shared words and ranks the gold chunk "
                           f"{vector_full} (cosine {cosines[gold_index]:.3f}).")

        # Hybrid categories: look at the candidate lists that RRF actually fused.
        n = config.HYBRID_CANDIDATES
        vector_list = [r.chunks[i]["chunk_id"] for i, _ in r.vector_search(query, n)]
        bm25_list = [r.chunks[i]["chunk_id"] for i, _ in r.bm25_search(query, n)]

        def list_rank(chunk_id, ranked):
            return ranked.index(chunk_id) + 1 if chunk_id in ranked else None

        def rrf(chunk_id):
            return sum(1 / (config.RRF_K + rk) for rk in
                       (list_rank(chunk_id, vector_list), list_rank(chunk_id, bm25_list)) if rk)

        def describe(chunk_id):
            v, b = list_rank(chunk_id, vector_list), list_rank(chunk_id, bm25_list)
            return (f"{chunk_id} (vector {v or f'not in top {n}'}, BM25 {b or f'not in top {n}'}, "
                    f"RRF {rrf(chunk_id):.4f})")

        if category == "hybrid_beats_both":
            leaders = [ids[0] for ids in (retrieved["vector"], retrieved["bm25"]) if ids and ids[0] != gold_id]
            text = (f"The gold chunk was in both candidate lists: {describe(gold_id)}. "
                    f"Being found by both retrievers gave it more RRF credit than chunks that only one "
                    f"retriever ranked highly")
            if leaders:
                text += ", e.g. " + "; ".join(describe(c) for c in dict.fromkeys(leaders))
            return text + "."

        # hybrid_worse_than_best
        best_mode = "vector" if as_number(ranks["vector"]) <= as_number(ranks["bm25"]) else "bm25"
        other_mode = "bm25" if best_mode == "vector" else "vector"
        hybrid_ids = retrieved["hybrid"]
        gold_pos = hybrid_ids.index(gold_id) if gold_id in hybrid_ids else len(hybrid_ids)
        ahead = hybrid_ids[:gold_pos][:2]
        text = (f"{best_mode} had the gold chunk at rank {show(ranks[best_mode])} but {other_mode} "
                f"had it at {show(ranks[other_mode])}, so it got credit from only one list or a "
                f"weak second rank: {describe(gold_id)}. ")
        if ahead:
            text += "Chunks fused above it: " + "; ".join(describe(c) for c in ahead) + "."
        return text

    def gold_index(self, question: dict, retrieved: dict) -> int:
        """The gold chunk to talk about: the one ranked best by any mode, else the first listed."""
        gold = question["gold_chunk_ids"]
        best = None
        for ids in retrieved.values():
            for rank, chunk_id in enumerate(ids, start=1):
                if chunk_id in gold and (best is None or rank < best[0]):
                    best = (rank, chunk_id)
        return self.retriever.index_of_id[best[1] if best else gold[0]]


def preview_around(text: str, evidence: str) -> str:
    """About PREVIEW_CHARS of the chunk, starting a little before the evidence
    (the part that answers the question), or the chunk start if not found."""
    start = text.find(evidence) if evidence else -1
    start = max(0, start - 60) if start >= 0 else 0
    end = start + PREVIEW_CHARS
    return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")


def write_analysis(path, retriever: Retriever, questions: list[dict], ranks: dict, retrieved: dict) -> None:
    """ranks[qid][mode] = gold rank; retrieved[qid][mode] = top-10 chunk ids."""
    by_id = {q["id"]: q for q in questions}
    explainer = Explainer(retriever)
    found = find_win_loss(ranks)

    lines = ["# Win/loss analysis", "",
             "Generated by `python -m eval.run_eval`. Ranks are the gold chunk's position in the top 10 "
             "(\"not in top 10\" = missed). Explanations are built from this run's actual BM25 tokens, "
             "cosine scores and RRF inputs. A word is called rare if it appears in at most "
             f"{RARE_SHARE:.0%} of chunks; BM25 gives rare words much more weight. "
             "\"Rank X of N\" is the gold chunk's position among all chunks, beyond the top 10.", ""]
    for name, title in CATEGORIES.items():
        cases = found[name]
        lines += [f"## {title}", "", f"{len(cases)} question(s) in this category; showing up to {TOP_EXAMPLES}.", ""]
        for qid, _ in cases[:TOP_EXAMPLES]:
            q, r = by_id[qid], ranks[qid]
            gold_text = retriever.chunks[explainer.gold_index(q, retrieved[qid])]["text"]
            preview = preview_around(gold_text, q.get("evidence", ""))
            lines += [
                f"### {qid} ({q['type']}): {q['question']}",
                "",
                f"- Ranks: vector **{show(r['vector'])}**, BM25 **{show(r['bm25'])}**, hybrid **{show(r['hybrid'])}**",
                f"- Gold chunk ({', '.join(q['gold_chunk_ids'])}): \"{preview}\"",
                f"- Why: {explainer.explain(name, q, r, retrieved[qid])}",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")
