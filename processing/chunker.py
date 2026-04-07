from __future__ import annotations

from typing import Dict, List


DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100


def _word_tokenize(text: str) -> List[str]:
    return text.split()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    tokens = _word_tokenize(text)
    if not tokens:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
    return chunks


def build_pdf_chunks(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    results: List[Dict] = []

    for page in pages:
        text_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for local_idx, content in enumerate(text_chunks):
            chunk_id = f"pdf::{page['file_name']}::p{page['page_number']}::c{local_idx}"
            results.append(
                {
                    "text": content,
                    "metadata": {
                        "source_type": "pdf",
                        "file_name": page["file_name"],
                        "page_number": page["page_number"],
                        "chunk_id": chunk_id,
                    },
                }
            )

    return results
