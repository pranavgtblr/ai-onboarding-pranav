"""Integration tests for two-stage RAG with Cross-Encoder Reranker."""

from unittest.mock import MagicMock, patch

import httpx

from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import Chunk, chunk_corpus_fine
from phase_3_rag.corpus import load_corpus
from phase_3_rag.reranked_rag import (
    execute_reranked_rag,
    format_reranked_prompt,
)
from phase_3_rag.reranker import HeuristicCrossEncoder, RerankResult


def _make_dummy_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_c")[0],
        source_title=f"Manual {chunk_id}",
        category="Ops",
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
    )


def test_format_reranked_prompt() -> None:
    """Verify reranked prompt includes cross-encoder scores and rank shifts."""
    chunk = _make_dummy_chunk(
        "doc_06_c0",
        "Battery backup lasts 72 hours under stage 1 load shedding.",
    )
    result = RerankResult(
        chunk=chunk,
        relevance_score=0.9234,
        reranked_rank=1,
        original_rank=14,
        reason="Direct coverage",
    )

    prompt = format_reranked_prompt("What is the battery runtime?", [result])
    assert "Cross-Score: 0.9234" in prompt
    assert "Rank #1 (was #14)" in prompt
    assert "Battery backup lasts 72 hours" in prompt
    assert "Question: What is the battery runtime?" in prompt


def test_execute_reranked_rag_mocked() -> None:
    """Verify two-stage retrieve-then-rerank pipeline logic with mocked LLM."""
    chunks = [
        _make_dummy_chunk(f"c_{i}", f"General outpost maintenance note {i}")
        for i in range(50)
    ]
    # Insert target chunk at position 35
    chunks[35] = _make_dummy_chunk(
        "c_target",
        "Emergency backup battery matrix provides 72 hours of life support power.",
    )

    bm25_index = BM25Index(chunks)
    reranker = HeuristicCrossEncoder()
    mock_client = MagicMock(spec=httpx.Client)

    with patch(
        "phase_3_rag.reranked_rag.generate_llm_response",
        return_value=("Answer: 72 hours of life support.", 200, 20),
    ):
        (
            answer,
            stage1_candidates,
            reranked_results,
            p_toks,
            c_toks,
        ) = execute_reranked_rag(
            "emergency backup battery matrix 72 hours",
            chunks,
            bm25_index,
            reranker,
            client=mock_client,
            candidate_k=50,
            top_k=8,
        )

        assert len(stage1_candidates) == 50
        assert len(reranked_results) == 8
        # Target chunk must be promoted to #1 in Stage 2
        assert reranked_results[0].chunk.chunk_id == "c_target"
        assert reranked_results[0].reranked_rank == 1
        assert "72 hours" in answer


def test_live_two_stage_reranked_retrieval() -> None:
    """Test full two-stage retrieval on real 77-chunk corpus (retrieve 50, keep 8)."""
    docs = load_corpus()
    chunks = chunk_corpus_fine(docs, chunk_size=120, overlap=20)
    assert len(chunks) >= 50

    bm25_index = BM25Index(chunks)
    reranker = HeuristicCrossEncoder()

    query = "What is the emergency backup battery runtime?"

    # Stage 1: Retrieve 50 candidates
    stage1_matches = bm25_index.search(query, top_k=50)
    candidates = [m.chunk for m in stage1_matches]
    assert len(candidates) == 50

    # Stage 2: Cross-Encoder rerank and keep 8
    reranked = reranker.rerank(query, candidates, top_k=8)
    assert len(reranked) == 8

    # The top reranked chunk should belong to doc_06 (battery storage)
    top_doc_ids = [r.chunk.doc_id for r in reranked[:3]]
    assert "doc_06" in top_doc_ids
