"""Fixed-size token chunker for RAG.

Splits documents into exact token chunks using tiktoken (cl100k_base).
Supports sliding-window chunking with token overlap.
Preserves document metadata and chunk indexing.
"""

from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from phase_3_rag.corpus import Document


class Chunk(BaseModel):
    """Represents a text chunk with metadata for vector search and prompt stuffing."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. doc_01_c0")
    doc_id: str = Field(description="Parent document identifier, e.g. doc_01")
    source_title: str = Field(description="Human-readable title of parent document")
    category: str = Field(description="Document category / subsystem")
    chunk_index: int = Field(description="Zero-indexed position within the document")
    text: str = Field(description="Decoded text content of the chunk")
    token_count: int = Field(description="Exact token count of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict)


def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Return cached tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def chunk_document(
    doc: Document,
    *,
    chunk_size: int = 500,
    overlap: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split a Document into token chunks with optional sliding-window overlap.

    Args:
        doc: The Document instance to split.
        chunk_size: Target token size per chunk (default 500).
        overlap: Overlapping tokens between consecutive chunks (default 0).
        encoding_name: Tiktoken encoding identifier.

    Returns:
        list[Chunk]: Ordered list of Chunk objects.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    enc = get_tokenizer(encoding_name)
    tokens = enc.encode(doc.text)

    if not tokens:
        return []

    chunks: list[Chunk] = []
    total_tokens = len(tokens)
    step = max(1, chunk_size - overlap)

    chunk_idx = 0
    start_idx = 0
    while start_idx < total_tokens:
        end_idx = min(start_idx + chunk_size, total_tokens)
        slice_tokens = tokens[start_idx:end_idx]
        chunk_text = enc.decode(slice_tokens)

        chunk_id = f"{doc.id}_c{chunk_idx}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.id,
                source_title=doc.title,
                category=doc.category,
                chunk_index=chunk_idx,
                text=chunk_text,
                token_count=len(slice_tokens),
                metadata={
                    "start_token": start_idx,
                    "end_token": end_idx,
                    "filename": doc.filename,
                },
            )
        )
        chunk_idx += 1
        if end_idx >= total_tokens:
            break
        start_idx += step

    return chunks


def chunk_corpus(
    documents: list[Document],
    *,
    chunk_size: int = 500,
    overlap: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk all documents in a corpus into fixed-size chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(
                doc,
                chunk_size=chunk_size,
                overlap=overlap,
                encoding_name=encoding_name,
            )
        )
    return all_chunks


def chunk_corpus_fine(
    documents: list[Document],
    *,
    chunk_size: int = 120,
    overlap: int = 20,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Generate fine-grained overlapping chunks (65+ chunks) for reranking."""
    return chunk_corpus(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        encoding_name=encoding_name,
    )
