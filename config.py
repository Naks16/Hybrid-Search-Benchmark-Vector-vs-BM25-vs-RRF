"""Central settings for the retrieval system. Change values here, not in code."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Embedding model (named EMBEDDING_MODEL so it is not confused with the
# MODEL_NAME env var used for the Anthropic SDK).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Chunking -------------------------------------------------------------
# Sizes are counted in tokens of the embedding model's own tokenizer, so a
# chunk is exactly what the model sees.
#
# CHUNK_SIZE = 400 trade-offs:
#   - bge-small-en-v1.5 accepts at most 512 tokens. Anything longer is silently
#     truncated, so the end of the chunk would never be embedded. 400 (+2
#     special tokens) stays safely under that limit.
#   - Smaller chunks (~100-200) give more precise matches, but a single
#     paragraph/argument gets split up and each piece loses context.
#   - Bigger chunks keep context, but one vector has to summarise more topics,
#     so the embedding gets "blurry", and BM25 favours long chunks that happen
#     to contain many query words.
#   400 is a middle ground: roughly 2-3 paragraphs of a paper.
#
# CHUNK_OVERLAP = 50 trade-offs:
#   - Without overlap, a sentence cut at a chunk boundary is split in two and
#     neither half may match the query.
#   - 50 tokens (~2 sentences, 12.5% of the chunk) repeats enough text to
#     cover a boundary sentence, while only adding ~14% more chunks to embed.
#   - Much larger overlap wastes compute and makes neighbouring chunks
#     near-duplicates that crowd the top-k results.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- Retrieval ------------------------------------------------------------
# Hybrid mode takes this many candidates from each retriever before fusing.
HYBRID_CANDIDATES = 20
# RRF constant k (60 is the value used in the original RRF paper).
RRF_K = 60
