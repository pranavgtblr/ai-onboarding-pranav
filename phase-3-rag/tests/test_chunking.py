"""Unit tests for fixed-size 500-token chunking."""

from phase_3_rag.chunking import chunk_corpus, chunk_document, get_tokenizer
from phase_3_rag.corpus import Document


def test_chunk_document_fixed_token_size() -> None:
    """Verify document is split into exact 500-token windows."""
    enc = get_tokenizer("cl100k_base")
    # Generate long text with ~1250 tokens
    words = "The quick brown fox jumps over the lazy dog. " * 150
    doc = Document(
        id="doc_test",
        title="Test Document",
        filename="test.md",
        category="Test",
        text=words,
    )

    chunks = chunk_document(doc, chunk_size=500)

    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert chunk.token_count == 500
        assert len(enc.encode(chunk.text)) == 500

    # Last chunk may have <= 500 tokens
    assert chunks[-1].token_count <= 500
    assert chunks[0].chunk_id == "doc_test_c0"
    assert chunks[1].chunk_id == "doc_test_c1"


def test_chunk_document_empty() -> None:
    """Verify empty document produces 0 chunks."""
    doc = Document(
        id="empty",
        title="Empty",
        filename="empty.md",
        category="Empty",
        text="",
    )
    chunks = chunk_document(doc, chunk_size=500)
    assert len(chunks) == 0


def test_chunk_corpus_multiple_documents() -> None:
    """Verify chunk_corpus processes multiple documents with correct indexing."""
    doc1 = Document(
        id="d1", title="Doc 1", filename="d1.md", category="C1", text="Hello world"
    )
    doc2 = Document(
        id="d2", title="Doc 2", filename="d2.md", category="C2", text="Alpha Beta Gamma"
    )
    corpus_chunks = chunk_corpus([doc1, doc2], chunk_size=500)

    assert len(corpus_chunks) == 2
    assert corpus_chunks[0].chunk_id == "d1_c0"
    assert corpus_chunks[1].chunk_id == "d2_c0"
