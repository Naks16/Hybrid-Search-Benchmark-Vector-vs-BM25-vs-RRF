"""Build the vector index: chunks -> embeddings, saved to disk.

Run: python -m search.index

Files written to config.INDEX_DIR:
  chunks.json     {"model": ..., "chunks": [{"chunk_id", "source", "text"}, ...]}
  embeddings.npy  float32 array, shape (num_chunks, 384); row i belongs to chunks[i]

Caching: on every run we re-read and re-chunk the documents (cheap), then
compare with chunks.json. If the model and every chunk (id + text) are the
same, the saved embeddings are reused and nothing is re-embedded. Any change
(new file, edited file, new chunk settings, new model) re-embeds everything,
which keeps the logic simple.

The BM25 index is not saved: building it from the chunk texts takes well
under a second, and rank_bm25 has no save/load function.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from search.chunking import chunk_documents
from search.loader import load_documents

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"


def load_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed chunk texts. normalize_embeddings=True gives unit-length vectors."""
    embeddings = model.encode_document(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 32
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_index(index_dir: Path = config.INDEX_DIR) -> tuple[list[dict], np.ndarray]:
    """Load chunks and embeddings saved by build_index()."""
    chunks_path = Path(index_dir) / CHUNKS_FILE
    embeddings_path = Path(index_dir) / EMBEDDINGS_FILE
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(f"No index in {index_dir}. Run: python -m search.index")
    saved = json.loads(chunks_path.read_text(encoding="utf-8"))
    return saved["chunks"], np.load(embeddings_path)


def build_index(
    model: SentenceTransformer,
    docs_dir: Path = config.DOCS_DIR,
    index_dir: Path = config.INDEX_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> tuple[list[dict], np.ndarray]:
    """Chunk all documents and embed them (or reuse the cache). Returns (chunks, embeddings).

    chunk_size/chunk_overlap default to config; the eval chunk-size sweep
    overrides them and passes its own index_dir so the main index is untouched.
    """
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, model.tokenizer, chunk_size, chunk_overlap)

    chunks_path = Path(index_dir) / CHUNKS_FILE
    embeddings_path = Path(index_dir) / EMBEDDINGS_FILE
    if chunks_path.exists() and embeddings_path.exists():
        saved = json.loads(chunks_path.read_text(encoding="utf-8"))
        if saved["model"] == config.EMBEDDING_MODEL and saved["chunks"] == chunks:
            print(f"Cache hit: {len(chunks)} chunks unchanged, skipping embedding.")
            return chunks, np.load(embeddings_path)

    print(f"Embedding {len(chunks)} chunks from {len(documents)} documents...")
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunks])

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    chunks_path.write_text(
        json.dumps({"model": config.EMBEDDING_MODEL, "chunks": chunks}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return chunks, embeddings


def main() -> None:
    chunks, embeddings = build_index(load_model())
    sources = sorted({chunk["source"] for chunk in chunks})
    print(f"Index ready: {len(chunks)} chunks from {len(sources)} files, embeddings {embeddings.shape}")
    for source in sources:
        count = sum(1 for chunk in chunks if chunk["source"] == source)
        print(f"  {source}: {count} chunks")


if __name__ == "__main__":
    main()
