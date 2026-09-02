"""Text embeddings generation and vector cosine similarity analysis."""

from __future__ import annotations

import math
from typing import Any

from google import genai
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Configuration settings for embedding models."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SentenceItem(BaseModel):
    """Metadata for an embedded sentence."""

    index: int
    label: str
    text: str
    category: str


class SimilarityMatrixResult(BaseModel):
    """Complete results of sentence embeddings and pairwise similarity matrix."""

    model: str
    dimensions: int
    sentences: list[SentenceItem]
    matrix: list[list[float]] = Field(description="10x10 cosine similarity matrix.")
    pair_a_analysis: dict[str, Any] = Field(
        description="Analysis of zero-word semantic equivalence pair."
    )
    pair_b_analysis: dict[str, Any] = Field(
        description="Analysis of shared product code pair."
    )


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the dot product (scalar product) of two vectors."""
    if len(vec_a) != len(vec_b):
        msg = f"Vectors must have same dimension: {len(vec_a)} != {len(vec_b)}"
        raise ValueError(msg)
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_magnitude(vec: list[float]) -> float:
    """Calculate Euclidean norm (L2 norm / magnitude) of a vector."""
    squared_sum = sum(x * x for x in vec)
    return math.sqrt(squared_sum)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two non-zero vectors in range [-1.0, 1.0]."""
    mag_a = vector_magnitude(vec_a)
    mag_b = vector_magnitude(vec_b)
    if mag_a == 0.0 or mag_b == 0.0:
        msg = "Cannot compute cosine similarity for a zero vector."
        raise ValueError(msg)
    raw_cos = dot_product(vec_a, vec_b) / (mag_a * mag_b)
    # Clamp to [-1.0, 1.0] to account for floating point inaccuracies
    clamped = max(-1.0, min(1.0, raw_cos))
    return round(clamped, 4)


def embed_single_text(
    text: str,
    client: genai.Client,
    model: str = "gemini-embedding-001",
) -> list[float]:
    """Generate embedding vector for a single text string."""
    response = client.models.embed_content(
        model=model,
        contents=text,
    )
    if response.embeddings and len(response.embeddings) > 0:
        vals = response.embeddings[0].values
        if vals is not None:
            return list(vals)
    msg = f"Unexpected response format from embedding API: {response}"
    raise RuntimeError(msg)


def embed_corpus(
    texts: list[str],
    settings: EmbeddingSettings | None = None,
) -> list[list[float]]:
    """Embed a list of text strings sequentially using Gemini API."""
    if settings is None:
        settings = EmbeddingSettings()
    if not settings.gemini_api_key:
        msg = "GEMINI_API_KEY is not configured in .env or environment."
        raise ValueError(msg)

    client = genai.Client(api_key=settings.gemini_api_key)
    embeddings: list[list[float]] = []

    for text in texts:
        vec = embed_single_text(
            text=text,
            client=client,
            model=settings.gemini_embedding_model,
        )
        embeddings.append(vec)

    return embeddings


def compute_similarity_matrix(
    sentences: list[SentenceItem],
    embeddings: list[list[float]],
    model_name: str = "gemini-embedding-001",
) -> SimilarityMatrixResult:
    """Compute 10x10 pairwise cosine similarity matrix and return structured results."""
    n = len(sentences)
    if len(embeddings) != n:
        msg = (
            f"Count of sentences ({n}) must match count of "
            f"embeddings ({len(embeddings)})"
        )
        raise ValueError(msg)

    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0000)
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                row.append(sim)
        matrix.append(row)

    dim = len(embeddings[0]) if embeddings else 0

    pair_a_sim = matrix[0][1]
    pair_b_sim = matrix[2][3]

    pair_a_words_0 = set(sentences[0].text.lower().strip().split())
    pair_a_words_1 = set(sentences[1].text.lower().strip().split())
    shared_words_a = list(pair_a_words_0.intersection(pair_a_words_1))

    pair_b_words_2 = set(sentences[2].text.lower().strip().split())
    pair_b_words_3 = set(sentences[3].text.lower().strip().split())
    shared_words_b = list(pair_b_words_2.intersection(pair_b_words_3))

    pair_a_analysis = {
        "sentence_1": sentences[0].text,
        "sentence_2": sentences[1].text,
        "shared_words_count": len(shared_words_a),
        "shared_words": shared_words_a,
        "cosine_similarity": pair_a_sim,
        "phenomenon": (
            "Dense semantic capture: High similarity score despite zero "
            "lexical overlap, demonstrating that embeddings map conceptual "
            "meaning rather than keyword tokens."
        ),
    }

    pair_b_analysis = {
        "sentence_1": sentences[2].text,
        "sentence_2": sentences[3].text,
        "shared_words_count": len(shared_words_b),
        "shared_words": shared_words_b,
        "cosine_similarity": pair_b_sim,
        "phenomenon": (
            "Lexical collision with semantic divergence: Shows how a unique "
            "product ID token interacts with contrasting context "
            "(customer refund vs engineering spec)."
        ),
    }

    return SimilarityMatrixResult(
        model=model_name,
        dimensions=dim,
        sentences=sentences,
        matrix=matrix,
        pair_a_analysis=pair_a_analysis,
        pair_b_analysis=pair_b_analysis,
    )
