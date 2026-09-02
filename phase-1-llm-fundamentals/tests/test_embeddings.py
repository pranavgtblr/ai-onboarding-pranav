"""Tests for text embeddings generation and cosine similarity matrix."""

import math
from unittest.mock import MagicMock, patch

import pytest

from phase_1_llm_fundamentals.embeddings import (
    SentenceItem,
    SimilarityMatrixResult,
    compute_similarity_matrix,
    cosine_similarity,
    dot_product,
    embed_single_text,
    vector_magnitude,
)


def test_dot_product():
    """Verify dot product calculation."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [4.0, 5.0, 6.0]
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert dot_product(vec_a, vec_b) == 32.0


def test_dot_product_dimension_mismatch():
    """Verify ValueError on unequal vector dimensions."""
    with pytest.raises(ValueError, match="must have same dimension"):
        dot_product([1.0, 2.0], [1.0, 2.0, 3.0])


def test_vector_magnitude():
    """Verify Euclidean norm (L2 magnitude)."""
    vec = [3.0, 4.0]
    # sqrt(9 + 16) = 5.0
    assert vector_magnitude(vec) == 5.0


def test_cosine_similarity():
    """Verify cosine similarity properties."""
    # Identical vectors -> 1.0
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    # Orthogonal vectors -> 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    # Opposite vectors -> -1.0
    assert cosine_similarity([2.0, 0.0], [-2.0, 0.0]) == -1.0

    # Zero vector -> error
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


def test_compute_similarity_matrix():
    """Verify matrix calculation and symmetry."""
    sentences = [
        SentenceItem(
            index=0,
            label="S1",
            text="The infant is asleep.",
            category="Sleep",
        ),
        SentenceItem(
            index=1,
            label="S2",
            text="A newborn baby rests quietly.",
            category="Sleep",
        ),
        SentenceItem(
            index=2,
            label="S3",
            text="Refund SKU-12345",
            category="Support",
        ),
        SentenceItem(
            index=3,
            label="S4",
            text="Blueprint SKU-12345",
            category="Engineering",
        ),
    ]
    # Synthetic 3D embeddings
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.8, 0.2],
    ]

    res = compute_similarity_matrix(
        sentences=sentences,
        embeddings=embeddings,
        model_name="test-embedding-model",
    )

    assert isinstance(res, SimilarityMatrixResult)
    assert len(res.matrix) == 4
    assert len(res.matrix[0]) == 4

    # Diagonal should be 1.0
    for i in range(4):
        assert math.isclose(res.matrix[i][i], 1.0, rel_tol=1e-3)

    # Matrix symmetry M[i][j] == M[j][i]
    for i in range(4):
        for j in range(4):
            assert math.isclose(res.matrix[i][j], res.matrix[j][i], rel_tol=1e-3)

    assert res.pair_a_analysis["shared_words_count"] == 0
    assert res.pair_b_analysis["shared_words_count"] == 1


@patch("google.genai.Client")
def test_embed_single_text_mock(mock_client_class: MagicMock):
    """Verify embed_single_text unpacks response values."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.values = [0.12, -0.45, 0.88]
    mock_response.embeddings = [mock_item]
    mock_client.models.embed_content.return_value = mock_response

    vec = embed_single_text("Hello test", client=mock_client)
    assert vec == [0.12, -0.45, 0.88]
    mock_client.models.embed_content.assert_called_once()
