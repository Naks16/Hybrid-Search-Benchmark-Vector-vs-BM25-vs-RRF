"""Split documents into overlapping, token-sized chunks with stable ids."""


def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    """Split text into windows of `chunk_size` tokens that overlap by `overlap`.

    We count tokens with the embedding model's tokenizer, but we cut the
    ORIGINAL string using each token's character offsets. Decoding tokens back
    to text would lowercase it (bge's tokenizer is uncased) and break words
    like "ValueError" into "value ##er ##ror", which would hurt BM25.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # verbose=False: the full document is longer than the model's 512 limit;
    # that is fine here because we only want token boundaries, not embeddings.
    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    offsets = encoding["offset_mapping"]  # one (start_char, end_char) per token

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(offsets), step):
        end = min(start + chunk_size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(text[char_start:char_end])
        if end == len(offsets):  # last window reached the end of the text
            break
    return chunks


def chunk_documents(documents: list[dict], tokenizer, chunk_size: int, overlap: int) -> list[dict]:
    """Chunk every document. Returns [{"chunk_id", "source", "text"}].

    chunk_id = "<filename>::<index>". It depends only on the file name, the
    file content and the chunk settings, so it is identical across runs.
    """
    chunks = []
    for doc in documents:
        # Collapse runs of whitespace/newlines (PDF text has many line breaks).
        clean_text = " ".join(doc["text"].split())
        for index, piece in enumerate(chunk_text(clean_text, tokenizer, chunk_size, overlap)):
            chunks.append({
                "chunk_id": f"{doc['source']}::{index}",
                "source": doc["source"],
                "text": piece,
            })
    return chunks
