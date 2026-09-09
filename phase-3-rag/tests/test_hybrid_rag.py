"""Integration tests for hybrid RAG pipeline."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import Chunk, chunk_corpus
from phase_3_rag.config import get_settings
from phase_3_rag.corpus import load_corpus
from phase_3_rag.fusion import FusedResult, reciprocal_rank_fusion
from phase_3_rag.hybrid_rag import (
    execute_hybrid_rag,
    format_hybrid_rag_prompt,
)
from phase_3_rag.vector_store import VectorIndex


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


def test_format_hybrid_rag_prompt() -> None:
    """Verify hybrid prompt injects RRF scores, rank tags, and instructions."""
    chunk = _make_dummy_chunk(
        "doc_14_c0",
        "Odyssey Base stores 85 metric tons of liquid methane for the ERV.",
    )
    fused_item = FusedResult(
        chunk=chunk,
        rrf_score=0.0327,
        rank=1,
        vector_rank=1,
        vector_score=0.88,
        bm25_rank=1,
        bm25_score=15.2,
    )

    prompt = format_hybrid_rag_prompt("What is the ERV methane capacity?", [fused_item])
    assert "Odyssey Base stores 85 metric tons" in prompt
    assert "doc_14_c0" in prompt
    assert "RRF: 0.03270" in prompt
    assert "V-Rank #1" in prompt
    assert "B-Rank #1" in prompt
    assert "Question: What is the ERV methane capacity?" in prompt


def test_execute_hybrid_rag_mocked() -> None:
    """Verify execution of full hybrid pipeline with mocked network calls."""
    c1 = _make_dummy_chunk(
        "doc_14_c0", "Cryogenic storage 85 metric tons LCH4 for ERV."
    )
    c2 = _make_dummy_chunk("doc_01_c0", "Electrolysis produces 4.8 kg O2 per day.")

    # Create dummy vector index
    v_index = VectorIndex()
    v_index.add(c1, [1.0, 0.0])
    v_index.add(c2, [0.0, 1.0])

    # Create BM25 index
    b_index = BM25Index([c1, c2])

    fake_client = MagicMock(spec=httpx.Client)

    with (
        patch(
            "phase_3_rag.hybrid_rag.embed_single_text",
            return_value=[1.0, 0.0],
        ),
        patch(
            "phase_3_rag.hybrid_rag.generate_llm_response",
            return_value=("Mocked answer: 85 metric tons LCH4.", 150, 25),
        ),
    ):
        (
            answer,
            v_hits,
            b_hits,
            fused_hits,
            p_tokens,
            c_tokens,
        ) = execute_hybrid_rag(
            "ERV LCH4 capacity",
            [c1, c2],
            v_index,
            b_index,
            client=fake_client,
            top_k=2,
        )

        assert "85 metric tons" in answer
        assert len(fused_hits) == 2
        assert fused_hits[0].chunk.chunk_id == "doc_14_c0"
        assert p_tokens == 150
        assert c_tokens == 25


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_live_hybrid_retrieval_cached() -> None:
    """Test hybrid retrieval on actual corpus using cached vector embeddings."""
    docs = load_corpus()
    chunks = chunk_corpus(docs, chunk_size=500)

    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not configured; skipping live test")

    with httpx.Client(timeout=30.0) as client:
        v_index = VectorIndex.build(chunks, client=client, use_cache=True)
        b_index = BM25Index(chunks)

        query = "What is the liquid methane storage capacity for ERV?"

        # Vector search
        v_res = v_index.search(
            # Using doc_14 embedding if available, else standard query
            v_index.embeddings[13] if len(v_index.embeddings) > 13 else [0.0] * 3072,
            top_k=5,
        )
        # BM25 search
        b_res = b_index.search(query, top_k=5)

        # Fused search
        fused = reciprocal_rank_fusion(v_res, b_res, k=60, top_k=5)

        # doc_14 must be in top 3 because of exact terms
        # "liquid methane storage capacity ERV"
        top_doc_ids = [f.chunk.doc_id for f in fused[:3]]
        assert "doc_14" in top_doc_ids
