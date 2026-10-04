# Hybrid Search Comparison (vector vs BM25 vs hybrid RRF)

## Rules
- I'm a fresher and must explain every file in an interview. Clear code over
  clever abstractions.
- Retrieval only. No LLM generation, no UI, no vector database server.
- Don't add features I didn't ask for. Work in stages, stop after each one.
- Check installed library docs before writing code; don't rely on memory.
- Never invent metrics. All numbers in docs must come from eval outputs.

## Stack
Python 3.11+, sentence-transformers (BAAI/bge-small-en-v1.5), rank_bm25,
numpy, pandas, pytest. Anthropic SDK only for drafting eval questions
(MODEL_NAME and ANTHROPIC_API_KEY from env vars, never hardcoded).

## Pipeline
load docs -> chunk (stable chunk_id) -> embed + BM25 index -> retrieve with
mode: vector | bm25 | hybrid.

## Commands
- Index: python -m search.index
- Query: python -m search.query --mode hybrid --q "..."
- Eval:  python -m eval.run_eval
- Test:  pytest