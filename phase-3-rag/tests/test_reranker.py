"""Unit tests for Cross-Encoder reranking models and overlap chunking."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_3_rag.chunking import Chunk, chunk_document
from phase_3_rag.corpus import Document
from phase_3_rag.reranker import (
    GeminiCrossEncoder,
    HeuristicCrossEncoder,
)


def _make_test_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_c")[0],
        source_title=f"Manual {chunk_id}",
        category="Ops",
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
    )


def test_chunking_with_overlap() -> None:
    """Verify sliding-window chunking creates overlapping text segments."""
    doc = Document(
        id="doc_test",
        title="Test Document",
        category="Test",
        filename="test.md",
        text="word " * 250,
    )

    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) >= 3

    # Check start and end token metadata
    assert chunks[0].metadata["start_token"] == 0
    assert chunks[0].metadata["end_token"] == 100

    # Next chunk starts at 100 - 20 = 80
    assert chunks[1].metadata["start_token"] == 80
    assert chunks[1].metadata["end_token"] == 180


def test_chunking_overlap_validation() -> None:
    """Verify ValueError when overlap is greater than or equal to chunk_size."""
    doc = Document(
        id="doc_err",
        title="Error Doc",
        category="Test",
        filename="err.md",
        text="sample text",
    )
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=50, overlap=50)


def test_heuristic_cross_encoder_promotes_relevant_chunk() -> None:
    """Verify cross-encoder scores exact contextual match significantly higher."""
    c1 = _make_test_chunk("c1", "Routine rover tire check and suspension pressure.")
    c2 = _make_test_chunk(
        "c2",
        "Emergency medical defibrillator protocol and cardiac arrest steps in airlock.",
    )
    c3 = _make_test_chunk("c3", "Hydroponics fertilizer nitrogen balance in module B.")

    reranker = HeuristicCrossEncoder()
    results = reranker.rerank(
        "emergency medical defibrillator protocol", [c1, c2, c3], top_k=3
    )

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].reranked_rank == 1
    assert results[0].original_rank == 2  # Came from position 2
    assert results[0].relevance_score > results[1].relevance_score


def test_reranker_top_k_truncation() -> None:
    """Verify reranker truncates candidate list down to requested top_k (e.g. 8)."""
    chunks = [
        _make_test_chunk(f"chunk_{i}", f"Text snippet number {i}") for i in range(50)
    ]
    reranker = HeuristicCrossEncoder()

    # Pass 50 candidates, keep top 8
    results = reranker.rerank("snippet number", chunks, top_k=8)
    assert len(results) == 8
    assert [r.reranked_rank for r in results] == list(range(1, 9))


def test_gemini_cross_encoder_mocked() -> None:
    """Verify GeminiCrossEncoder correctly parses structured JSON batch scores."""
    c1 = _make_test_chunk("c1", "Nuclear fission reactor telemetry and coolant flow.")
    c2 = _make_test_chunk("c2", "Solar dust mitigation and brush maintenance.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '[{"chunk_id": "c1", "relevance_score": 0.95}, '
                                '{"chunk_id": "c2", "relevance_score": 0.10}]'
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_client.post.return_value = mock_response

    gemini_reranker = GeminiCrossEncoder(client=mock_client, api_key="dummy_key")
    results = gemini_reranker.rerank("nuclear reactor coolant", [c2, c1], top_k=2)

    assert len(results) == 2
    # c1 was in original position 2, should be reranked to position 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].relevance_score == 0.95
    assert results[0].reranked_rank == 1
    assert results[0].original_rank == 2

    assert results[1].chunk.chunk_id == "c2"
    assert results[1].relevance_score == 0.10
