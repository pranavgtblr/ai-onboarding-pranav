"""Acclaimed Wider Cinema Registry & Retrieval for PG Recommends.

Enables recommendations to expand beyond PG's personal Letterboxd diary into
acclaimed masterpieces, world cinema, and critic-celebrated gems, tailored
to the user's taste profile.
"""

from typing import Any

from pydantic import BaseModel

from phase_6_capstone.retrieval import (
    CINEMA_STOPWORDS,
    GENRE_SYNONYMS,
    MovieCitation,
    _tokenize,
)


class AcclaimedFilm(BaseModel):
    """Acclaimed film from wider cinema outside PG's personal diary."""

    title: str
    year: int
    director: str
    genres: list[str]
    consensus: str
    portal_name: str
    review_url: str
    rating_or_score: str

    def to_citation(self) -> MovieCitation:
        """Converts into a verifiable MovieCitation marked as acclaimed_cinema."""
        movie_id = f"acclaimed_{abs(hash((self.title, self.year))) % 100000000}"
        label = (
            f"[Acclaimed Cinema: {self.title} ({self.year}) - {self.rating_or_score}]"
        )
        return MovieCitation(
            movie_id=movie_id,
            title=self.title,
            year=self.year,
            rating=4.5,
            letterboxd_url=self.review_url,
            excerpt=self.consensus,
            citation_label=label,
            source_type="acclaimed_cinema",
            source_portal=self.portal_name,
        )


# Curated catalog of acclaimed world cinema spanning key genres and directors
ACCLAIMED_WIDER_CINEMA: list[AcclaimedFilm] = [
    # Action / Martial Arts / Thriller
    AcclaimedFilm(
        title="The Raid",
        year=2011,
        director="Gareth Evans",
        genres=["Action", "Crime", "Thriller"],
        consensus=(
            "No frills and all thrills, The Raid is an inventive, relentless "
            "action film expertly paced and edited for maximum kinetic impact."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/the_raid_redemption",
        rating_or_score="87% RT / Certified Fresh",
    ),
    AcclaimedFilm(
        title="John Wick: Chapter 4",
        year=2023,
        director="Chad Stahelski",
        genres=["Action", "Crime", "Thriller"],
        consensus=(
            "Piles on more of everything—and suggests that when it comes to "
            "bare-knuckle, beautifully choreographed action, there can never "
            "be too much."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/john_wick_chapter_4",
        rating_or_score="94% RT / RogerEbert ★ 4.0",
    ),
    AcclaimedFilm(
        title="Heat",
        year=1995,
        director="Michael Mann",
        genres=["Action", "Crime", "Drama"],
        consensus=(
            "Though Al Pacino and Robert De Niro share but a handful of screen "
            "minutes, Heat is an absorbing crime drama that draws compelling "
            "performances from its stars."
        ),
        portal_name="RogerEbert.com",
        review_url="https://www.rogerebert.com/reviews/heat-1995",
        rating_or_score="★ 3.5 / 4.0 RogerEbert",
    ),
    AcclaimedFilm(
        title="Hard Boiled",
        year=1992,
        director="John Woo",
        genres=["Action", "Crime", "Thriller"],
        consensus=(
            "Boasting exceeding bravura gunplay and kinetic choreography, "
            "John Woo's Hong Kong action classic remains a high-water mark of "
            "the genre."
        ),
        portal_name="The Guardian",
        review_url="https://www.theguardian.com/film/hard-boiled",
        rating_or_score="Acclaimed Classic",
    ),
    # Sci-Fi / Cyberpunk / Mind-Benders
    AcclaimedFilm(
        title="Blade Runner 2049",
        year=2017,
        director="Denis Villeneuve",
        genres=["Sci-Fi", "Mystery", "Drama", "Action"],
        consensus=(
            "Visually stunning and narratively satisfying, Blade Runner 2049 deepens "
            "and expands its predecessor's story while standing as an impressive "
            "filmmaking achievement in its own right."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/blade_runner_2049",
        rating_or_score="88% RT / RogerEbert ★ 3.5",
    ),
    AcclaimedFilm(
        title="Children of Men",
        year=2006,
        director="Alfonso Cuarón",
        genres=["Sci-Fi", "Action", "Drama", "Thriller"],
        consensus=(
            "Children of Men works on every level: as a violent, chase-filled "
            "thriller, a harrowing dystopian vision, and an indictment of our "
            "complacent present."
        ),
        portal_name="RogerEbert.com",
        review_url="https://www.rogerebert.com/reviews/children-of-men-2006",
        rating_or_score="★ 4.0 / 4.0 RogerEbert",
    ),
    AcclaimedFilm(
        title="Arrival",
        year=2016,
        director="Denis Villeneuve",
        genres=["Sci-Fi", "Drama", "Mystery"],
        consensus=(
            "Arrival delivers a must-see experience for fans of thinking person's "
            "sci-fi that anchors its heady concepts with genuinely affecting emotion."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/arrival_2016",
        rating_or_score="94% RT / Certified Fresh",
    ),
    # Horror / Dread / Atmospheric
    AcclaimedFilm(
        title="The Thing",
        year=1982,
        director="John Carpenter",
        genres=["Horror", "Sci-Fi", "Mystery"],
        consensus=(
            "Grimmer and more terrifying than its 1950s predecessor, John Carpenter's "
            "The Thing is a tense, paranoia-laced sci-fi masterpiece with peerless "
            "practical effects."
        ),
        portal_name="The Guardian",
        review_url="https://www.theguardian.com/film/the-thing-review",
        rating_or_score="Masterpiece / ★ 5.0",
    ),
    AcclaimedFilm(
        title="Hereditary",
        year=2018,
        director="Ari Aster",
        genres=["Horror", "Mystery", "Drama"],
        consensus=(
            "Hereditary uses the classic horror setup as the framework for a "
            "deeply unsettling, richly atmospheric family tragedy with astonishing "
            "performances."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/hereditary",
        rating_or_score="90% RT / Certified Fresh",
    ),
    # Crime / Neo-Noir / Psychological
    AcclaimedFilm(
        title="Memories of Murder",
        year=2003,
        director="Bong Joon-ho",
        genres=["Crime", "Drama", "Mystery", "Thriller"],
        consensus=(
            "Bong Joon-ho blends familiar procedural tropes with biting satire "
            "and genuine sociopolitical dread in this gripping South Korean "
            "masterpiece."
        ),
        portal_name="RogerEbert.com",
        review_url="https://www.rogerebert.com/reviews/memories-of-murder-2003",
        rating_or_score="★ 4.0 / 4.0 RogerEbert",
    ),
    AcclaimedFilm(
        title="No Country for Old Men",
        year=2007,
        director="Joel Coen",
        genres=["Crime", "Drama", "Thriller"],
        consensus=(
            "Bolstered by powerful lead performances from Javier Bardem and Tommy "
            "Lee Jones, the Coen brothers craft a bleak, uncompromising neo-western "
            "masterpiece."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/no_country_for_old_men",
        rating_or_score="93% RT / 4 Academy Awards",
    ),
    # Romance / Dramedy / World Cinema
    AcclaimedFilm(
        title="Before Sunrise",
        year=1995,
        director="Richard Linklater",
        genres=["Romance", "Drama"],
        consensus=(
            "Thought-provoking and charismatic, Before Sunrise is an intelligent, "
            "delicately observed romance that floats on the chemistry of its leads."
        ),
        portal_name="Rotten Tomatoes",
        review_url="https://www.rottentomatoes.com/m/before_sunrise",
        rating_or_score="100% RT / Essential Romance",
    ),
    AcclaimedFilm(
        title="Past Lives",
        year=2023,
        director="Celine Song",
        genres=["Romance", "Drama"],
        consensus=(
            "A remarkable debut for writer-director Celine Song, Past Lives uses the "
            "bonds between its sensitive central characters to offer profound "
            "reflections on the human condition."
        ),
        portal_name="The Guardian",
        review_url="https://www.theguardian.com/film/2023/sep/08/past-lives-review",
        rating_or_score="96% RT / ★ 5.0 Guardian",
    ),
    # Malayalam Cinema Masterpieces Outside Diary
    AcclaimedFilm(
        title="Iratta",
        year=2023,
        director="Rohit M.G. Krishnan",
        genres=["Crime", "Drama", "Mystery", "Thriller"],
        consensus=(
            "Joju George delivers a powerhouse dual performance in a harrowing, "
            "unflinching Malayalam investigative drama with an unforgettable climax."
        ),
        portal_name="The Hindu",
        review_url="https://www.thehindu.com/entertainment/movies/iratta-movie-review/article66468494.ece",
        rating_or_score="Critically Acclaimed",
    ),
    AcclaimedFilm(
        title="Bramayugam",
        year=2024,
        director="Rahul Sadasivan",
        genres=["Horror", "Mystery", "Thriller"],
        consensus=(
            "Mammootty's sinister performance and Rahul Sadasivan's exquisite "
            "monochrome staging create a spellbinding period horror steeped in "
            "folklore."
        ),
        portal_name="Film Companion",
        review_url="https://www.filmcompanion.in/reviews/malayalam-review/bramayugam-movie-review",
        rating_or_score="Acclaimed Masterpiece",
    ),
    AcclaimedFilm(
        title="Manjummel Boys",
        year=2024,
        director="Chidambaram",
        genres=["Adventure", "Drama", "Thriller"],
        consensus=(
            "A breathtaking survival thriller that celebrates friendship, grit, "
            "and emotional resonance with sheer cinematic finesse."
        ),
        portal_name="The Hindu",
        review_url="https://www.thehindu.com/entertainment/movies/manjummel-boys-movie-review/article67876118.ece",
        rating_or_score="All-Time Industry Hit",
    ),
]


def find_acclaimed_wider_cinema(
    query: str,
    taste_profile: Any = None,
    limit: int = 2,
) -> list[MovieCitation]:
    """Finds acclaimed wider cinema films matching query, genre, and taste profile."""
    query_lower = query.lower()
    tokens = _tokenize(query)
    content_tokens = [t for t in tokens if t not in CINEMA_STOPWORDS]

    # Detect genres requested
    target_genres = [GENRE_SYNONYMS[t] for t in tokens if t in GENRE_SYNONYMS]

    scored_films: list[tuple[float, AcclaimedFilm]] = []

    for film in ACCLAIMED_WIDER_CINEMA:
        score = 0.0

        # Exact title match
        if film.title.lower() in query_lower:
            score += 20.0

        # Director match in query
        if film.director.lower() in query_lower:
            score += 10.0

        # Genre alignment
        if target_genres:
            if any(g in film.genres for g in target_genres):
                score += 8.0
            elif film.title.lower() not in query_lower:
                # Strictly avoid suggesting unrelated genres
                continue

        # Lexical content overlap
        text_content = (
            f"{film.title} {film.director} {' '.join(film.genres)} {film.consensus}"
        ).lower()
        for ct in content_tokens:
            if ct in text_content:
                score += 3.0

        # Incorporate taste profile
        if taste_profile is not None:
            liked_directors = getattr(taste_profile, "liked_directors", [])
            if any(ld.lower() in film.director.lower() for ld in liked_directors):
                score += 5.0

            liked_genres = getattr(taste_profile, "liked_genres", [])
            if any(lg in film.genres for lg in liked_genres):
                score += 2.5

            disliked_elements = getattr(taste_profile, "disliked_elements", [])
            if any(de.lower() in text_content for de in disliked_elements):
                score -= 15.0

        if score > 0:
            scored_films.append((score, film))

    scored_films.sort(key=lambda x: x[0], reverse=True)
    return [film.to_citation() for _, film in scored_films[:limit]]
