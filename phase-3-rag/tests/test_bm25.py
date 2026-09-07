"""Unit tests for BM25 keyword search index and tokenizer."""

from phase_3_rag.bm25 import BM25Index, tokenize
from phase_3_rag.chunking import Chunk


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    """Helper to create minimal Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_c")[0],
        source_title=f"Title {chunk_id}",
        category="Testing",
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
    )


def test_tokenize_alphanumeric_and_acronyms() -> None:
    """Verify tokenizer lowercases and preserves technical acronyms."""
    text = "Odyssey Base: ERV LCH4 fuel tank #14 at -161.6°C."
    tokens = tokenize(text)
    assert "odyssey" in tokens
    assert "base" in tokens
    assert "erv" in tokens
    assert "lch4" in tokens
    assert "fuel" in tokens
    assert "tank" in tokens
    assert "14" in tokens


def test_bm25_exact_keyword_matching() -> None:
    """Verify BM25 ranks document with exact matching terms above unrelated docs."""
    c1 = _make_chunk(
        "doc_01_c0",
        "Liquid oxygen LOX and liquid methane LCH4 are stored for the ERV.",
    )
    c2 = _make_chunk(
        "doc_02_c0",
        "Hydroponic potato and dwarf wheat growth cycles in greenhouse alpha.",
    )
    c3 = _make_chunk(
        "doc_03_c0",
        "Rover mobility chassis suspension and wheel wear on basalt regolith.",
    )

    index = BM25Index([c1, c2, c3])
    results = index.search("LCH4 ERV", top_k=3)

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "doc_01_c0"
    assert results[0].score > 0.0
    # c2 and c3 do not contain LCH4 or ERV
    assert results[1].score == 0.0
    assert results[2].score == 0.0


def test_bm25_length_normalization() -> None:
    """Verify length normalization (b parameter) favors concise relevant docs."""
    # Short chunk with target keyword
    concise = _make_chunk(
        "concise_c0",
        "Emergency medical defibrillator protocol in airlock.",
    )
    # Very verbose chunk with the same keyword once amidst lots of filler
    filler = " " + " ".join(["system status nominal monitor reading"] * 25)
    verbose = _make_chunk(
        "verbose_c0",
        f"Emergency medical defibrillator protocol in airlock.{filler}",
    )

    index = BM25Index([concise, verbose])
    results = index.search("defibrillator protocol", top_k=2)

    assert results[0].chunk.chunk_id == "concise_c0"
    assert results[0].score > results[1].score


def test_bm25_empty_query_and_empty_index() -> None:
    """Verify graceful handling of empty queries and empty indices."""
    empty_index = BM25Index([])
    assert empty_index.search("test") == []

    populated = BM25Index([_make_chunk("c1", "Sample text here")])
    assert populated.search("") == []
    assert populated.search("    ") == []


def test_bm25_unseen_term() -> None:
    """Verify query with words not in corpus yields zero score."""
    chunk = _make_chunk("c1", "Water filtration using reverse osmosis membrane")
    index = BM25Index([chunk])
    results = index.search("superconductor quantum")
    assert len(results) == 1
    assert results[0].score == 0.0
