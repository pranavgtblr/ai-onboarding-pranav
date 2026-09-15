"""Data models for PG Recommends."""

from pydantic import BaseModel, Field


class MovieRecord(BaseModel):
    """Structured record representing a movie reviewed by PG."""

    movie_id: str
    title: str
    year: int
    letterboxd_url: str
    rating: float | None = None
    review_text: str = ""
    watched_date: str | None = None
    rewatch: bool = False
    tags: list[str] = Field(default_factory=list)
    poster_url: str | None = None
    guid: str | None = None
    genres: list[str] = Field(default_factory=list)
    director: str | None = None

    @property
    def star_display(self) -> str:
        """Returns a Letterboxd-style star rating display (e.g. ★★★★½)."""
        if self.rating is None:
            return "Unrated"
        full_stars = int(self.rating)
        half_star = "½" if (self.rating - full_stars) >= 0.5 else ""
        return ("★" * full_stars) + half_star
