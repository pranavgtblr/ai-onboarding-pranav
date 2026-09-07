"""Reciprocal Rank Fusion (RRF) for hybrid search retrieval.

Merges ranked search results from multiple disparate retrieval systems
(e.g., semantic vector search and BM25 lexical keyword search) into a single,
re-ranked list without requiring score normalization.

Reference:
    Cormack, Clarke, and Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Run Systems." SIGIR '09.
"""

from pydantic import BaseModel, Field

from phase_3_rag.bm25 import BM25SearchResult
from phase_3_rag.chunking import Chunk
from phase_3_rag.vector_store import SearchResult

# Standard smoothing constant from Information Retrieval literature
DEFAULT_RRF_K = 60


class FusedResult(BaseModel):
    """Represents a chunk ranked by Reciprocal Rank Fusion."""

    chunk: Chunk = Field(description="The underlying text chunk")
    rrf_score: float = Field(description="Combined RRF score")
    rank: int = Field(description="1-based fused rank position")
    vector_rank: int | None = Field(
        default=None, description="Original rank in vector search, if present"
    )
    vector_score: float | None = Field(
        default=None, description="Original cosine similarity score, if present"
    )
    bm25_rank: int | None = Field(
        default=None, description="Original rank in BM25 search, if present"
    )
    bm25_score: float | None = Field(
        default=None, description="Original BM25 score, if present"
    )


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[BM25SearchResult],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> list[FusedResult]:
    """Fuse vector and BM25 search results using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{ranker} (1 / (k + rank(ranker, d)))

    Args:
        vector_results: Ranked list of results from semantic vector search.
        bm25_results: Ranked list of results from BM25 keyword search.
        k: Smoothing constant (default 60). Prevents top ranks from dominating.
        top_k: Maximum number of fused results to return.

    Returns:
        List of FusedResult items sorted in descending order of RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    vector_info: dict[str, tuple[int, float]] = {}
    bm25_info: dict[str, tuple[int, float]] = {}

    # Accumulate RRF points from vector results
    for item in vector_results:
        cid = item.chunk.chunk_id
        chunk_map[cid] = item.chunk
        vector_info[cid] = (item.rank, item.score)
        contribution = 1.0 / (k + item.rank)
        scores[cid] = scores.get(cid, 0.0) + contribution

    # Accumulate RRF points from BM25 results
    for item in bm25_results:
        cid = item.chunk.chunk_id
        chunk_map[cid] = item.chunk
        bm25_info[cid] = (item.rank, item.score)
        contribution = 1.0 / (k + item.rank)
        scores[cid] = scores.get(cid, 0.0) + contribution

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: list[FusedResult] = []
    for rank, (cid, score) in enumerate(sorted_candidates[:top_k], start=1):
        v_rank, v_score = vector_info.get(cid, (None, None))
        b_rank, b_score = bm25_info.get(cid, (None, None))

        fused_results.append(
            FusedResult(
                chunk=chunk_map[cid],
                rrf_score=score,
                rank=rank,
                vector_rank=v_rank,
                vector_score=v_score,
                bm25_rank=b_rank,
                bm25_score=b_score,
            )
        )

    return fused_results
