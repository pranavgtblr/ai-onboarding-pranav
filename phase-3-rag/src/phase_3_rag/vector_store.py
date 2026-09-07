"""Vector store and embedding engine for Phase 3 RAG.

Provides:
1. Batch embedding generation using Gemini API (gemini-embedding-001).
2. Pure math cosine similarity calculation.
3. In-memory vector store with disk caching to avoid redundant API calls.
4. Top-K nearest neighbor search.
"""

import json
import math
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from phase_3_rag.chunking import Chunk
from phase_3_rag.config import get_settings

CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "embeddings_cache.json"


class SearchResult(BaseModel):
    """Result of a vector search query."""

    chunk: Chunk
    score: float = Field(description="Cosine similarity score in [-1.0, 1.0]")
    rank: int = Field(description="Rank in search results (1-indexed)")


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute scalar dot product between two float vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_norm(vec: list[float]) -> float:
    """Compute Euclidean L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two non-zero vectors in range [-1.0, 1.0]."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions do not match: {len(vec_a)} != {len(vec_b)}"
        )

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw = dot_product(vec_a, vec_b) / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to guard against floating point inaccuracies
    return max(-1.0, min(1.0, round(raw, 5)))


def embed_single_text(
    text: str,
    *,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[float]:
    """Generate embedding for a single text string via Gemini REST API."""
    settings = get_settings()
    effective_key = api_key or settings.gemini_api_key
    effective_model = model or settings.gemini_embedding_model

    if not effective_key:
        raise ValueError("Gemini API key is required for embeddings.")

    url = f"{settings.gemini_base_url}/models/{effective_model}:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_key,
    }
    payload = {"content": {"parts": [{"text": text}]}}

    should_close = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close = True

    try:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]
    finally:
        if should_close:
            client.close()


def embed_texts_batch(
    texts: list[str],
    *,
    batch_size: int = 10,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts using batchEmbedContents endpoint."""
    if not texts:
        return []

    settings = get_settings()
    effective_key = api_key or settings.gemini_api_key
    effective_model = model or settings.gemini_embedding_model

    if not effective_key:
        raise ValueError("Gemini API key is required for embeddings.")

    url = f"{settings.gemini_base_url}/models/{effective_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_key,
    }

    should_close = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close = True

    all_embeddings: list[list[float]] = []

    try:
        for i in range(0, len(texts), batch_size):
            slice_texts = texts[i : i + batch_size]
            requests = [
                {
                    "model": f"models/{effective_model}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in slice_texts
            ]

            resp = client.post(url, headers=headers, json={"requests": requests})
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("embeddings", []):
                all_embeddings.append(item["values"])
    finally:
        if should_close:
            client.close()

    return all_embeddings


class VectorIndex:
    """In-memory vector search index for text chunks."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        """Add a single chunk and its embedding vector to the index."""
        self.chunks.append(chunk)
        self.embeddings.append(vector)

    def size(self) -> int:
        """Return the count of indexed chunks."""
        return len(self.chunks)

    def search(
        self, query_vector: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        """Search index for top-k similar chunks using cosine similarity."""
        if not self.chunks:
            return []

        scored: list[tuple[float, int]] = []
        for idx, vec in enumerate(self.embeddings):
            sim = cosine_similarity(query_vector, vec)
            scored.append((sim, idx))

        # Sort descending by cosine similarity score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                )
            )

        return results

    def save_cache(self, cache_path: Path = CACHE_FILE) -> None:
        """Persist index chunks and vectors to JSON on disk."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "embeddings": self.embeddings,
        }
        cache_path.write_text(json.dumps(data), encoding="utf-8")

    def load_cache(self, cache_path: Path = CACHE_FILE) -> bool:
        """Load index from JSON cache if file exists. Returns True on success."""
        if not cache_path.exists():
            return False

        try:
            content = json.loads(cache_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk.model_validate(c) for c in content["chunks"]]
            self.embeddings = content["embeddings"]
            return True
        except Exception:
            return False

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        *,
        use_cache: bool = True,
        cache_path: Path = CACHE_FILE,
        client: httpx.Client | None = None,
    ) -> "VectorIndex":
        """Build or load a VectorIndex from a list of Chunks."""
        index = cls()

        if use_cache and index.load_cache(cache_path):
            # If chunk count matches, cache is fresh
            if len(index.chunks) == len(chunks):
                return index

        # Compute embeddings via batch endpoint
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts_batch(texts, client=client)

        for chunk, vec in zip(chunks, embeddings):
            index.add(chunk, vec)

        if use_cache:
            index.save_cache(cache_path)

        return index
