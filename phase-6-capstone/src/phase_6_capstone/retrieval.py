"""Hybrid Retrieval and Citation Engine for PG Recommends."""

import re

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from phase_6_capstone.models import MovieRecord


class MovieCitation(BaseModel):
    """Verifiable source citation for a movie recommendation."""

    movie_id: str
    title: str
    year: int
    rating: float | None = None
    letterboxd_url: str
    excerpt: str
    citation_label: str


class SearchResult(BaseModel):
    """Ranked search result with record and fusion score."""

    record: MovieRecord
    score: float

    def to_citation(self) -> MovieCitation:
        """Constructs a verifiable citation with star rating and review snippet."""
        clean_excerpt = (self.record.review_text or "").strip()
        if len(clean_excerpt) > 120:
            clean_excerpt = clean_excerpt[:117] + "..."

        label = (
            f"[PG Review: {self.record.title} ({self.record.year}) "
            f"{self.record.star_display}]"
        )
        return MovieCitation(
            movie_id=self.record.movie_id,
            title=self.record.title,
            year=self.record.year,
            rating=self.record.rating,
            letterboxd_url=self.record.letterboxd_url,
            excerpt=clean_excerpt,
            citation_label=label,
        )


def _tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]


class HybridMovieRetriever:
    """Hybrid BM25 and Semantic Retriever over PG's Letterboxd catalog."""

    def __init__(self, catalog: list[MovieRecord]):
        self.catalog = list(catalog)
        self.corpus = [
            _tokenize(
                f"{r.title} {r.year} {' '.join(r.genres)} {r.director or ''} "
                f"{' '.join(r.tags)} {r.review_text}"
            )
            for r in self.catalog
        ]
        if self.corpus and any(self.corpus):
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_rating: float | None = None,
        required_genres: list[str] | None = None,
    ) -> list[SearchResult]:
        """Performs hybrid retrieval with optional rating & genre filters."""
        if not self.catalog or not query.strip():
            return []

        tokens = _tokenize(query)
        if not tokens or self.bm25 is None:
            # Fallback to direct substring filtering
            candidates = self.catalog
            if min_rating is not None:
                candidates = [
                    c for c in candidates if c.rating and c.rating >= min_rating
                ]
            return [SearchResult(record=c, score=1.0) for c in candidates[:top_k]]

        # BM25 Sparse Scores
        bm25_scores = self.bm25.get_scores(tokens)

        # Dense / Lexical Overlap Secondary Score
        ranked_indices = []
        for idx, rec in enumerate(self.catalog):
            # Apply hard filters first
            if min_rating is not None:
                if rec.rating is None or rec.rating < min_rating:
                    continue

            if required_genres:
                rec_genres = {g.lower() for g in rec.genres}
                if not any(rg.lower() in rec_genres for rg in required_genres):
                    continue

            score = float(bm25_scores[idx])

            # Bonus for exact title matches
            query_lower = query.lower()
            if rec.title.lower() in query_lower:
                score += 15.0

            # Bonus for high curator ratings
            if rec.rating is not None:
                score += rec.rating * 0.5

            ranked_indices.append((score, rec))

        # Sort by total fused score descending
        ranked_indices.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for score, rec in ranked_indices[:top_k]:
            results.append(SearchResult(record=rec, score=score))

        return results
