"""Okapi BM25 keyword search index.

Implements standard BM25 (Best Matching 25) probabilistic relevance model:
- Tokenization and lowercasing
- Term frequency saturation (k1 parameter)
- Document length normalization (b parameter)
- Inverse Document Frequency (IDF) with Lucene/Robertson-Jones smoothing
"""

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from phase_3_rag.chunking import Chunk

# Default BM25 hyperparameters recommended in IR literature
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words.

    Matches alphanumeric terms including technical acronyms, codes, and numbers.
    """
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class BM25SearchResult(BaseModel):
    """Represents a single search match from the BM25 index."""

    chunk: Chunk = Field(description="Matched text chunk")
    score: float = Field(description="BM25 relevance score (higher is better)")
    rank: int = Field(description="1-based rank position in search results")


class BM25Index:
    """In-memory Okapi BM25 inverted index for fast keyword retrieval."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Initialize and optionally index chunks."""
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        self.chunks = list(chunks)
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_freqs = {}

        if not self.chunks:
            self.avgdl = 0.0
            self.idf = {}
            return

        total_length = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Record document frequency (count distinct terms per chunk)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        n_docs = len(self.chunks)
        self.avgdl = total_length / n_docs if n_docs > 0 else 0.0

        # Precompute Robertson-Spärck Jones IDF for every term in corpus
        # IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            numerator = n_docs - freq + 0.5
            denominator = freq + 0.5
            self.idf[term] = math.log((numerator / denominator) + 1.0)

    def search(self, query: str, *, top_k: int = 5) -> list[BM25SearchResult]:
        """Score and rank indexed chunks against query terms."""
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self.chunks)

        for i, tokens in enumerate(self.doc_tokens):
            doc_len = self.doc_lengths[i]
            # Term frequencies within this document
            tf_counter = Counter(tokens)

            doc_score = 0.0
            for q_term in query_tokens:
                if q_term not in tf_counter:
                    continue

                tf = tf_counter[q_term]
                idf = self.idf.get(q_term, 0.0)

                # Okapi BM25 term score formula:
                # Score = IDF * (tf * (k1 + 1)) / (tf + k1 * len_norm)
                len_norm = (
                    1.0 - self.b + self.b * (doc_len / self.avgdl)
                    if self.avgdl > 0
                    else 1.0
                )
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm
                doc_score += idf * (numerator / denominator)

            scores[i] = doc_score

        # Pair scores with chunk indices
        scored_pairs = [(scores[idx], idx) for idx in range(len(self.chunks))]

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        results: list[BM25SearchResult] = []
        for rank, (score, idx) in enumerate(scored_pairs[:top_k], start=1):
            results.append(
                BM25SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                )
            )

        return results
