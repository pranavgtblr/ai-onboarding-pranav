"""Fixed-size token chunker for RAG.

Splits documents into exact fixed 500-token chunks using tiktoken (cl100k_base).
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
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split a Document into fixed-size token chunks.

    Args:
        doc: The Document instance to split.
        chunk_size: Target token size per chunk (default 500).
        encoding_name: Tiktoken encoding identifier.

    Returns:
        list[Chunk]: Ordered list of Chunk objects.
    """
    enc = get_tokenizer(encoding_name)
    tokens = enc.encode(doc.text)

    if not tokens:
        return []

    chunks: list[Chunk] = []
    total_tokens = len(tokens)

    chunk_idx = 0
    for start_idx in range(0, total_tokens, chunk_size):
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

    return chunks


def chunk_corpus(
    documents: list[Document],
    *,
    chunk_size: int = 500,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk all documents in a corpus into fixed-size chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, encoding_name=encoding_name)
        )
    return all_chunks
