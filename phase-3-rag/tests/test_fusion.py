"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from phase_3_rag.bm25 import BM25SearchResult
from phase_3_rag.chunking import Chunk
from phase_3_rag.fusion import reciprocal_rank_fusion
from phase_3_rag.vector_store import SearchResult


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_c")[0],
        source_title=f"Title {chunk_id}",
        category="Test",
        chunk_index=0,
        text="Sample content",
        token_count=10,
    )


def test_rrf_mathematical_scoring() -> None:
    """Verify exact mathematical computation of RRF score: sum(1 / (k + rank))."""
    c1 = _make_chunk("chunk_1")
    c2 = _make_chunk("chunk_2")

    # Vector results: chunk_1 is rank 1, chunk_2 is rank 2
    vec_results = [
        SearchResult(chunk=c1, score=0.95, rank=1),
        SearchResult(chunk=c2, score=0.85, rank=2),
    ]

    # BM25 results: chunk_1 is rank 1, chunk_2 is absent
    bm25_results = [
        BM25SearchResult(chunk=c1, score=12.5, rank=1),
    ]

    k = 60
    fused = reciprocal_rank_fusion(vec_results, bm25_results, k=k, top_k=5)

    assert len(fused) == 2
    assert fused[0].chunk.chunk_id == "chunk_1"
    # Expected chunk_1 score: 1/(60+1) [vector] + 1/(60+1) [bm25] = 2/61
    expected_c1 = (1.0 / 61.0) + (1.0 / 61.0)
    assert abs(fused[0].rrf_score - expected_c1) < 1e-6
    assert fused[0].vector_rank == 1
    assert fused[0].bm25_rank == 1

    assert fused[1].chunk.chunk_id == "chunk_2"
    # Expected chunk_2 score: 1/(60+2) [vector only] = 1/62
    expected_c2 = 1.0 / 62.0
    assert abs(fused[1].rrf_score - expected_c2) < 1e-6
    assert fused[1].vector_rank == 2
    assert fused[1].bm25_rank is None


def test_rrf_agreement_boosts_rank() -> None:
    """Verify a chunk supported by BOTH rankers beats a chunk supported by only ONE."""
    strong_single = _make_chunk("doc_single")
    agreed_both = _make_chunk("doc_agreed")

    # doc_single is #1 in Vector, but completely absent from BM25
    vec_results = [
        SearchResult(chunk=strong_single, score=0.99, rank=1),
        SearchResult(chunk=agreed_both, score=0.90, rank=2),
    ]

    # doc_agreed is #2 in BM25
    bm25_results = [
        BM25SearchResult(chunk=agreed_both, score=10.0, rank=2),
    ]

    # At k=60:
    # doc_single: 1/(60+1) = 1/61 ≈ 0.01639
    # doc_agreed: 1/(60+2) + 1/(60+2) = 2/62 = 1/31 ≈ 0.03226
    fused = reciprocal_rank_fusion(vec_results, bm25_results, k=60, top_k=2)

    assert fused[0].chunk.chunk_id == "doc_agreed"
    assert fused[1].chunk.chunk_id == "doc_single"


def test_rrf_top_k_truncation() -> None:
    """Verify top_k parameter properly caps output list length."""
    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    vec_results = [
        SearchResult(chunk=c, score=1.0 - i * 0.1, rank=i + 1)
        for i, c in enumerate(chunks)
    ]
    bm25_results = [
        BM25SearchResult(chunk=c, score=10.0 - i, rank=i + 1)
        for i, c in enumerate(chunks)
    ]

    fused = reciprocal_rank_fusion(vec_results, bm25_results, k=60, top_k=3)
    assert len(fused) == 3
    assert [f.rank for f in fused] == [1, 2, 3]
