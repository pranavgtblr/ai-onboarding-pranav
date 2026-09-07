"""Unit tests for vector index, embeddings, and cosine similarity."""

from pathlib import Path

from phase_3_rag.chunking import Chunk
from phase_3_rag.vector_store import (
    VectorIndex,
    cosine_similarity,
    dot_product,
    vector_norm,
)


def test_cosine_similarity_identical() -> None:
    """Verify identical vectors yield cosine similarity of 1.0."""
    vec = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec, vec) == 1.0


def test_cosine_similarity_orthogonal() -> None:
    """Verify orthogonal vectors yield cosine similarity of 0.0."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_opposite() -> None:
    """Verify opposite vectors yield cosine similarity of -1.0."""
    vec_a = [1.0, 2.0]
    vec_b = [-1.0, -2.0]
    assert cosine_similarity(vec_a, vec_b) == -1.0


def test_vector_norm_and_dot_product() -> None:
    """Verify vector norm and dot product computations."""
    assert vector_norm([3.0, 4.0]) == 5.0
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0


def test_vector_index_search_ranking() -> None:
    """Verify top-k search ranks results by highest cosine similarity."""
    index = VectorIndex()

    c1 = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_title="Title 1",
        category="Cat",
        chunk_index=0,
        text="Oxygen generation",
        token_count=10,
    )
    c2 = Chunk(
        chunk_id="c2",
        doc_id="d2",
        source_title="Title 2",
        category="Cat",
        chunk_index=0,
        text="Water recycling",
        token_count=10,
    )
    c3 = Chunk(
        chunk_id="c3",
        doc_id="d3",
        source_title="Title 3",
        category="Cat",
        chunk_index=0,
        text="Solar arrays",
        token_count=10,
    )

    # Add vectors: c1 is closest to [1, 0, 0]
    index.add(c1, [0.9, 0.1, 0.0])
    index.add(c2, [0.5, 0.5, 0.0])
    index.add(c3, [0.0, 0.9, 0.1])

    query_vec = [1.0, 0.0, 0.0]
    results = index.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].rank == 1
    assert results[1].chunk.chunk_id == "c2"
    assert results[1].rank == 2
    assert results[0].score > results[1].score


def test_vector_index_save_and_load_cache(tmp_path: Path) -> None:
    """Verify vector index saves to disk cache and loads correctly."""
    cache_path = tmp_path / "test_cache.json"
    index = VectorIndex()

    c1 = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_title="T1",
        category="Cat",
        chunk_index=0,
        text="Sample text",
        token_count=5,
    )
    index.add(c1, [0.1, 0.2, 0.3])
    index.save_cache(cache_path)

    loaded = VectorIndex()
    success = loaded.load_cache(cache_path)

    assert success is True
    assert loaded.size() == 1
    assert loaded.chunks[0].chunk_id == "c1"
    assert loaded.embeddings[0] == [0.1, 0.2, 0.3]
