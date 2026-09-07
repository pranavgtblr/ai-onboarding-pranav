"""Unit and integration tests for Task 3.1 Naive RAG Baseline."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_3_rag.chunking import Chunk
from phase_3_rag.config import get_settings
from phase_3_rag.naive_rag import (
    execute_naive_rag,
    format_rag_prompt,
)
from phase_3_rag.vector_store import SearchResult, VectorIndex


def test_format_rag_prompt() -> None:
    """Verify augmented prompt formats context chunks and instructions."""
    c1 = Chunk(
        chunk_id="doc_01_c0",
        doc_id="doc_01",
        source_title="Oxygen Gen",
        category="Life Support",
        chunk_index=0,
        text="Oxygen is generated at 2.4 kg/hr.",
        token_count=10,
    )
    retrieved = [SearchResult(chunk=c1, score=0.85, rank=1)]

    prompt = format_rag_prompt("How much oxygen?", retrieved)

    assert "Oxygen Gen" in prompt
    assert "doc_01_c0" in prompt
    assert "0.8500" in prompt
    assert "Question: How much oxygen?" in prompt
    assert "=== RETRIEVED CONTEXT (TOP 1) ===" in prompt


def test_execute_naive_rag_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end Naive RAG pipeline execution with mocked LLM and embeddings."""
    mock_index = VectorIndex()
    for i in range(5):
        chunk = Chunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            source_title=f"Title {i}",
            category="Test",
            chunk_index=0,
            text=f"Sample text content for chunk {i}",
            token_count=8,
        )
        mock_index.add(chunk, [1.0 - (i * 0.1), 0.0, 0.0])

    # Mock embed_single_text to return query vector [1, 0, 0]
    monkeypatch.setattr(
        "phase_3_rag.naive_rag.embed_single_text",
        lambda query, **kwargs: [1.0, 0.0, 0.0],
    )

    # Mock Gemini REST response
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Grounded answer based on chunk c0."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 250,
            "candidatesTokenCount": 15,
            "totalTokenCount": 265,
        },
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    result = execute_naive_rag(
        query="What does chunk 0 contain?",
        top_k=3,
        index=mock_index,
        client=mock_client,
        api_key="mock-key",
    )

    assert result.answer == "Grounded answer based on chunk c0."
    assert len(result.retrieved_chunks) == 3
    assert result.retrieved_chunks[0].chunk.chunk_id == "c0"
    assert result.prompt_tokens == 250
    assert result.answer_tokens == 15
    assert result.total_chunks_in_index == 5


def test_live_naive_rag_integration() -> None:
    """Live API test verifying end-to-end Naive RAG against Gemini."""
    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not configured; skipping live test")

    result = execute_naive_rag(
        query="What is the daily oxygen consumption of an adult crew member?",
        top_k=5,
        use_cache=True,
    )

    assert len(result.retrieved_chunks) == 5
    # Rank 1 chunk must be the Oxygen Generation document (doc_01)
    top_chunk = result.retrieved_chunks[0].chunk
    assert top_chunk.doc_id == "doc_01"
    # Grounded answer must contain 0.84 kg
    assert "0.84" in result.answer
    assert result.prompt_tokens > 0
    assert result.answer_tokens > 0
