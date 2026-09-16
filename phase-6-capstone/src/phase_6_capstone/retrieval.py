"""Hybrid Retrieval and Citation Engine for PG Recommends."""

import html
import re
from typing import Any

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
    source_type: str = "letterboxd"  # "letterboxd" or "acclaimed_cinema"
    source_portal: str = "Letterboxd"


class SearchResult(BaseModel):
    """Ranked search result with record and fusion score."""

    record: MovieRecord
    score: float

    def to_citation(self) -> MovieCitation:
        """Constructs a verifiable citation with star rating and review snippet."""
        raw_excerpt = (self.record.review_text or "").strip()
        clean_excerpt = (
            html.unescape(html.unescape(raw_excerpt))
            .replace("&#039;", "'")
            .replace("&#39;", "'")
            .replace("&quot;", '"')
            .replace("&amp;", "&")
        )
        clean_excerpt = " ".join(clean_excerpt.split())
        if len(clean_excerpt) > 180:
            clean_excerpt = clean_excerpt[:177] + "..."

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
            source_type="letterboxd",
            source_portal="Letterboxd",
        )


CINEMA_STOPWORDS = {
    "suggest",
    "suggestions",
    "suggestion",
    "movie",
    "movies",
    "film",
    "films",
    "recommend",
    "recommendation",
    "recommendations",
    "show",
    "give",
    "please",
    "some",
    "good",
    "best",
    "great",
    "any",
    "top",
    "what",
    "which",
    "want",
    "like",
    "watch",
    "about",
    "with",
    "me",
    "an",
    "a",
    "the",
    "for",
    "in",
    "of",
    "and",
    "or",
    "to",
    "can",
    "you",
    "explore",
    "exploring",
    "look",
    "looking",
    "into",
    "need",
    "feel",
    "feeling",
    "craving",
    "something",
    "is",
    "isn't",
    "was",
    "are",
    "tell",
    "it",
    "this",
    "that",
    "i",
    "my",
}


def _tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric tokens with query expansion."""
    raw = [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]
    tokens: list[str] = list(raw)
    for t in raw:
        if t == "romcom":
            tokens.extend(["romance", "comedy"])
        elif t == "scifi":
            tokens.extend(["sci", "fi", "science"])
    return tokens


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
        taste_profile: Any = None,
        excluded_titles: set[str] | None = None,
    ) -> list[SearchResult]:
        """Performs hybrid retrieval with stopword filtering and pure BM25 scoring."""
        if not self.catalog or not query.strip():
            return []

        tokens = _tokenize(query)
        if not tokens or self.bm25 is None:
            candidates = self.catalog
            if min_rating is not None:
                candidates = [
                    c for c in candidates if c.rating and c.rating >= min_rating
                ]
            if excluded_titles:
                candidates = [
                    c
                    for c in candidates
                    if not any(
                        ex in c.title.lower() or c.title.lower() in ex
                        for ex in excluded_titles
                    )
                ]
            return [SearchResult(record=c, score=1.0) for c in candidates[:top_k]]

        # Separate content tokens from stopwords
        content_tokens = [t for t in tokens if t not in CINEMA_STOPWORDS]
        active_tokens = content_tokens if content_tokens else tokens

        # BM25 Sparse Scores on active tokens
        bm25_scores = self.bm25.get_scores(active_tokens)

        ranked_indices = []
        query_lower = query.lower()

        for idx, rec in enumerate(self.catalog):
            # Exclude rejected/corrected titles
            if excluded_titles and any(
                ex in rec.title.lower() or rec.title.lower() in ex
                for ex in excluded_titles
            ):
                continue

            # Apply hard rating filter if requested
            if min_rating is not None:
                if rec.rating is None or rec.rating < min_rating:
                    continue

            # Check explicit required_genres filter
            if required_genres:
                rec_genres = {g.lower() for g in (rec.genres + rec.tags)}
                if not any(rg.lower() in rec_genres for rg in required_genres):
                    continue

            raw_bm25 = float(bm25_scores[idx])
            exact_title_match = rec.title.lower() in query_lower

            # If content tokens were specified, require BM25 overlap or
            # exact title match
            if content_tokens:
                if raw_bm25 <= 0.0 and not exact_title_match:
                    continue

            score = raw_bm25

            # Bonus for exact title matches
            if exact_title_match:
                score += 15.0

            # Weighting for curator rating
            if rec.rating is not None:
                score += rec.rating * 0.4

            # Taste profile personalization
            if taste_profile is not None:
                liked_directors = getattr(taste_profile, "liked_directors", [])
                if rec.director and any(
                    ld.lower() in rec.director.lower() for ld in liked_directors
                ):
                    score += 4.0

                liked_genres = getattr(taste_profile, "liked_genres", [])
                rec_genres = {g.lower() for g in (rec.genres + rec.tags)}
                if any(lg.lower() in rec_genres for lg in liked_genres):
                    score += 2.0

                disliked_elements = getattr(taste_profile, "disliked_elements", [])
                review_lower = (rec.review_text or "").lower()
                if any(de.lower() in review_lower for de in disliked_elements):
                    score -= 15.0

            if score > 0:
                ranked_indices.append((score, rec))

        # Sort by total fused score descending
        ranked_indices.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for score, rec in ranked_indices[:top_k]:
            results.append(SearchResult(record=rec, score=score))

        return results
