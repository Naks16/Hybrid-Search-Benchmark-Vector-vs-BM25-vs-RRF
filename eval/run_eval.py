"""Evaluate vector vs BM25 vs hybrid retrieval on eval/questions.json.

Run: python -m eval.run_eval

Writes to eval/:
  results.jsonl  one line per (question, mode): gold rank in top 10, latency, retrieved ids
  results.md     main table, per-type table, RRF sweep, chunk-size sweep
  analysis.md    win/loss examples between modes, with data-based explanations

All numbers come from gold_chunk_ids (or, in the chunk-size sweep, from the
evidence text). There is no LLM judge.
"""

import argparse
import json
import time
from pathlib import Path

import config
from eval.analysis import write_analysis
from eval.metrics import gold_rank, hit_rate, latency_summary, mrr
from search.index import build_index, load_index, load_model
from search.retrieve import MODES, Retriever

EVAL_DIR = Path(__file__).resolve().parent
TOP_K = 10
TYPES = ("exact_term", "paraphrase", "mixed")
RRF_K_VALUES = (10, 60, 100)
CANDIDATE_VALUES = (10, 20, 50)
CHUNK_SIZES = (200, 400, 800)


def load_questions(path: Path, chunks: list[dict]) -> list[dict]:
    """Load questions and check every gold id exists in the current index."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    unknown = [(q["id"], g) for q in questions for g in q["gold_chunk_ids"] if g not in known_ids]
    if unknown:
        raise ValueError(f"Gold ids not in the index (re-index or fix questions): {unknown}")
    return questions


# --- Main run --------------------------------------------------------------

def run_queries(retriever: Retriever, questions: list[dict]) -> list[dict]:
    """Retrieve top 10 for every question in every mode; record gold rank and latency."""
    for mode in MODES:  # warm-up so one-time costs (first model call) don't count as latency
        retriever.retrieve("warm up", mode, top_k=TOP_K)

    records = []
    for q in questions:
        for mode in MODES:
            start = time.perf_counter()
            results = retriever.retrieve(q["question"], mode, top_k=TOP_K)
            latency_ms = (time.perf_counter() - start) * 1000
            retrieved_ids = [r["chunk_id"] for r in results]
            records.append({
                "id": q["id"],
                "type": q["type"],
                "mode": mode,
                "gold_rank": gold_rank(retrieved_ids, q["gold_chunk_ids"]),
                "latency_ms": round(latency_ms, 2),
                "retrieved_ids": retrieved_ids,
            })
    return records


def score_row(ranks: list[int | None]) -> dict:
    return {
        "n": len(ranks),
        "hit@1": hit_rate(ranks, 1),
        "hit@3": hit_rate(ranks, 3),
        "hit@5": hit_rate(ranks, 5),
        "mrr@10": mrr(ranks, TOP_K),
        "misses": sum(1 for r in ranks if r is None),
    }


def summarize(records: list[dict]) -> tuple[dict, dict]:
    """Returns (overall[mode], by_type[type][mode]) metric rows."""
    overall, by_type = {}, {t: {} for t in TYPES}
    for mode in MODES:
        mode_records = [r for r in records if r["mode"] == mode]
        overall[mode] = score_row([r["gold_rank"] for r in mode_records])
        overall[mode]["latency_mean"], overall[mode]["latency_p95"] = latency_summary(
            [r["latency_ms"] for r in mode_records])
        for t in TYPES:
            typed = [r["gold_rank"] for r in mode_records if r["type"] == t]
            if typed:
                by_type[t][mode] = score_row(typed)
    return overall, by_type


# --- Sweeps ----------------------------------------------------------------

def sweep_rrf(retriever: Retriever, questions: list[dict]) -> dict[tuple[int, int], float]:
    """Hybrid MRR@10 for every (rrf_k, candidates-per-retriever) pair."""
    table = {}
    for rrf_k in RRF_K_VALUES:
        for candidates in CANDIDATE_VALUES:
            ranks = []
            for q in questions:
                hits = retriever.hybrid_search(q["question"], TOP_K, candidates=candidates, rrf_k=rrf_k)
                ids = [retriever.chunks[i]["chunk_id"] for i, _ in hits]
                ranks.append(gold_rank(ids, q["gold_chunk_ids"]))
            table[(rrf_k, candidates)] = mrr(ranks, TOP_K)
    return table


def sweep_chunk_size(model, questions: list[dict]) -> list[dict]:
    """Re-chunk and re-index at each size; MRR@10 per mode.

    Gold chunks are re-found by searching for each question's evidence text:
    every new chunk that contains the evidence counts as gold. If the
    evidence is split across two chunks at some size, no chunk contains it
    and the question is left out at that size (counted in "unmapped").
    """
    rows = []
    for size in CHUNK_SIZES:
        if size == config.CHUNK_SIZE:
            index_dir = config.INDEX_DIR  # the main index, already built
        else:
            index_dir = config.INDEX_DIR / f"sweep_chunk_{size}"
        chunks, embeddings = build_index(
            model, index_dir=index_dir, chunk_size=size, chunk_overlap=config.CHUNK_OVERLAP)
        retriever = Retriever(chunks, embeddings, model)

        mapped = []
        for q in questions:
            gold = [c["chunk_id"] for c in chunks if q["evidence"] in c["text"]]
            if gold:
                mapped.append((q, gold))

        row = {"chunk_size": size, "chunks": len(chunks), "mapped": len(mapped),
               "unmapped": len(questions) - len(mapped)}
        for mode in MODES:
            ranks = []
            for q, gold in mapped:
                ids = [r["chunk_id"] for r in retriever.retrieve(q["question"], mode, top_k=TOP_K)]
                ranks.append(gold_rank(ids, gold))
            row[mode] = mrr(ranks, TOP_K)
        rows.append(row)
    return rows


# --- Report ----------------------------------------------------------------

def fmt(x: float) -> str:
    return f"{x:.3f}"


def write_results_md(path: Path, questions, overall, by_type, rrf_table, chunk_rows, max_seq_length) -> None:
    n = len(questions)
    type_counts = ", ".join(f"{t}: {sum(1 for q in questions if q['type'] == t)}" for t in TYPES)
    best_mode = max(MODES, key=lambda m: overall[m]["mrr@10"])
    best_mrr = overall[best_mode]["mrr@10"]
    tied = [m for m in MODES if overall[m]["mrr@10"] == best_mrr]

    lines = [
        "# Retrieval evaluation results", "",
        "Generated by `python -m eval.run_eval`. Do not edit by hand; re-run instead.", "",
        f"- Questions: {n} ({type_counts})",
        f"- Embedding model: {config.EMBEDDING_MODEL}; chunks: {config.CHUNK_SIZE} tokens, overlap {config.CHUNK_OVERLAP}",
        f"- Hybrid: top {config.HYBRID_CANDIDATES} per retriever, RRF k={config.RRF_K}; all modes retrieve top {TOP_K}",
        "- A hit means any of the question's gold_chunk_ids is retrieved. MRR@10 counts 0 for a miss.",
        f"- With {n} questions, one question changes a hit rate by {1 / n:.3f}. Small differences are noise.",
        "",
        "## Main results", "",
        "| mode | hit@1 | hit@3 | hit@5 | MRR@10 | missed (not in top 10) | mean latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MODES:
        o = overall[m]
        lines.append(f"| {m} | {fmt(o['hit@1'])} | {fmt(o['hit@3'])} | {fmt(o['hit@5'])} | {fmt(o['mrr@10'])} "
                     f"| {o['misses']}/{o['n']} | {o['latency_mean']:.1f} | {o['latency_p95']:.1f} |")
    lines.append("")
    if len(tied) > 1:
        lines.append(f"**Best overall MRR@10: tie between {', '.join(tied)} ({fmt(best_mrr)}).**")
    else:
        lines.append(f"**Best overall MRR@10: {best_mode} ({fmt(best_mrr)}).**")
    if "hybrid" not in tied:
        lines.append(f"Hybrid does **not** win overall: hybrid MRR@10 is {fmt(overall['hybrid']['mrr@10'])} "
                     f"vs {fmt(best_mrr)} for {best_mode}.")
    lines += ["",
              "Latency is wall-clock time of one `retrieve()` call on this machine (CPU), after a warm-up call. "
              "Vector and hybrid include embedding the query; hybrid runs both retrievers and the fusion.", ""]

    lines += ["## By question type", "",
              "| type | n | mode | hit@1 | hit@3 | hit@5 | MRR@10 | missed |", "|---|---|---|---|---|---|---|---|"]
    for t in TYPES:
        for m in MODES:
            if m in by_type[t]:
                o = by_type[t][m]
                lines.append(f"| {t} | {o['n']} | {m} | {fmt(o['hit@1'])} | {fmt(o['hit@3'])} | "
                             f"{fmt(o['hit@5'])} | {fmt(o['mrr@10'])} | {o['misses']} |")
    lines.append("")

    lines += ["## Sweep: RRF k x candidates per retriever (hybrid MRR@10)", "",
              "| RRF k \\ top-n per retriever | " + " | ".join(str(c) for c in CANDIDATE_VALUES) + " |",
              "|---" * (len(CANDIDATE_VALUES) + 1) + "|"]
    for k in RRF_K_VALUES:
        cells = []
        for c in CANDIDATE_VALUES:
            cell = fmt(rrf_table[(k, c)])
            if (k, c) == (config.RRF_K, config.HYBRID_CANDIDATES):
                cell += " (default)"
            cells.append(cell)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "Tuned and scored on the same questions (no held-out set), so the best cell is an optimistic estimate.", ""]

    lines += ["## Sweep: chunk size (MRR@10 per mode)", "",
              "| chunk size (tokens) | chunks | questions mapped | " + " | ".join(MODES) + " |",
              "|---" * (3 + len(MODES)) + "|"]
    for row in chunk_rows:
        lines.append(f"| {row['chunk_size']} | {row['chunks']} | {row['mapped']}/{row['mapped'] + row['unmapped']} | "
                     + " | ".join(fmt(row[m]) for m in MODES) + " |")
    lines += [
        "",
        f"- Overlap fixed at {config.CHUNK_OVERLAP} tokens for every size.",
        "- Gold chunks are re-found by the question's evidence text, so this table can differ slightly from the "
        "main table even at the default size (the main table uses gold_chunk_ids as reviewed).",
        "- Questions whose evidence is split across a chunk boundary at that size are left out (see \"questions mapped\").",
        f"- The embedding model reads at most {max_seq_length} tokens. Chunks longer than that are truncated "
        "for vector search (BM25 still sees the whole chunk), so at the largest size the vector mode only embeds "
        "the start of each chunk.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    chunks, embeddings = load_index()
    questions = load_questions(args.questions, chunks)
    retriever = Retriever(chunks, embeddings, model)
    print(f"Evaluating {len(questions)} questions x {len(MODES)} modes...")

    records = run_queries(retriever, questions)
    with open(args.out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    overall, by_type = summarize(records)

    ranks = {q["id"]: {} for q in questions}
    retrieved = {q["id"]: {} for q in questions}
    for r in records:
        ranks[r["id"]][r["mode"]] = r["gold_rank"]
        retrieved[r["id"]][r["mode"]] = r["retrieved_ids"]
    write_analysis(args.out_dir / "analysis.md", retriever, questions, ranks, retrieved)

    print("Sweeping RRF k x candidates...")
    rrf_table = sweep_rrf(retriever, questions)
    print("Sweeping chunk size (re-indexes; first run embeds each size)...")
    chunk_rows = sweep_chunk_size(model, questions)

    write_results_md(args.out_dir / "results.md", questions, overall, by_type, rrf_table, chunk_rows,
                     model.max_seq_length)
    print(f"Wrote results.jsonl, results.md, analysis.md to {args.out_dir}")


if __name__ == "__main__":
    main()
