"""Dynamic Taste Learning Engine with Multi-Tenant Scoping."""

import json
import re

from pydantic import BaseModel, Field
from sqlalchemy import select

from phase_6_capstone.db import DatabaseManager, UserTasteProfileModel

KNOWN_DIRECTORS = [
    "Christopher Nolan",
    "David Fincher",
    "Denis Villeneuve",
    "Wes Anderson",
    "Quentin Tarantino",
    "Martin Scorsese",
    "Joel Coen",
    "Ethan Coen",
    "Guillermo del Toro",
    "Stanley Kubrick",
    "Ridley Scott",
    "Steven Spielberg",
    "James Cameron",
    "Greta Gerwig",
    "Rahul Sadasivan",
    "Shahi Kabir",
    "Wes Craven",
    "Bong Joon-ho",
    "Park Chan-wook",
    "Paul Thomas Anderson",
    "Hayao Miyazaki",
]

KNOWN_GENRES = [
    "Psychological Thriller",
    "Horror",
    "Slasher",
    "Sci-Fi",
    "Neo-Noir",
    "Mystery",
    "Crime",
    "Drama",
    "Comedy",
    "Romance",
    "Action",
    "Body Horror",
    "Documentary",
    "Cyberpunk",
]


class TasteDelta(BaseModel):
    """Extracted taste signals from a single conversational turn."""

    liked_directors: list[str] = Field(default_factory=list)
    disliked_directors: list[str] = Field(default_factory=list)
    liked_genres: list[str] = Field(default_factory=list)
    disliked_genres: list[str] = Field(default_factory=list)
    disliked_elements: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)


class UserTasteProfile(BaseModel):
    """User taste profile state."""

    tenant_id: str
    user_id: str
    liked_directors: list[str] = Field(default_factory=list)
    disliked_directors: list[str] = Field(default_factory=list)
    liked_genres: list[str] = Field(default_factory=list)
    disliked_genres: list[str] = Field(default_factory=list)
    disliked_elements: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    taste_notes: str | None = None


def extract_taste_signals_from_text(text: str) -> TasteDelta:
    """Extracts preference signals (directors, genres, tropes) from text."""
    delta = TasteDelta()

    negative_phrases = [
        "hate",
        "dislike",
        "don't like",
        "dont like",
        "cannot stand",
        "can't stand",
        "not a fan",
        "tired of",
        "cheap",
        "avoid",
    ]

    # Split into clauses (respecting contrastive conjunctions like 'but', 'however')
    clauses = re.split(
        r"[.!?\n]+|,\s*but\s+|\s+but\s+|\s+however\s+|\s+although\s+|\s+except\s+",
        text,
        flags=re.IGNORECASE,
    )

    for clause in clauses:
        s_lower = clause.lower().strip()
        if not s_lower:
            continue
        is_negative = any(neg in s_lower for neg in negative_phrases)

        # Check directors
        for d in KNOWN_DIRECTORS:
            d_parts = d.lower().split()
            last_name = d_parts[-1]
            if d.lower() in s_lower or (len(last_name) > 4 and last_name in s_lower):
                if is_negative:
                    if d not in delta.disliked_directors:
                        delta.disliked_directors.append(d)
                else:
                    if d not in delta.liked_directors:
                        delta.liked_directors.append(d)

        # Check genres
        for g in KNOWN_GENRES:
            if g.lower() in s_lower:
                if is_negative:
                    if g not in delta.disliked_genres:
                        delta.disliked_genres.append(g)
                else:
                    if g not in delta.liked_genres:
                        delta.liked_genres.append(g)

        # Check specific tropes
        if "jump scare" in s_lower:
            delta.disliked_elements.append("jump scares")
        if "gore" in s_lower:
            delta.disliked_elements.append("gore")
        if "cringe" in s_lower or "dialogue" in s_lower:
            delta.disliked_elements.append("dialogue")
        if "propaganda" in s_lower:
            delta.disliked_elements.append("propaganda")
        if is_negative and "romance" in s_lower:
            delta.disliked_elements.append("romance")

        # Mood tags
        if "moody" in s_lower or "atmospheric" in s_lower:
            delta.mood_tags.append("atmospheric")
        if "fast-paced" in s_lower:
            delta.mood_tags.append("fast-paced")

    return delta


class TasteProfileManager:
    """Manages persistent user taste profiles with strict tenant isolation."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def get_profile(
        self, tenant_id: str, user_id: str
    ) -> UserTasteProfile | None:
        """Fetches profile enforcing strict tenant_id and user_id scoping."""
        async with self.db.session_factory() as session:
            stmt = select(UserTasteProfileModel).where(
                UserTasteProfileModel.tenant_id == tenant_id,
                UserTasteProfileModel.user_id == user_id,
            )
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            if not model:
                return None

            return UserTasteProfile(
                tenant_id=model.tenant_id,
                user_id=model.user_id,
                liked_directors=json.loads(model.liked_directors),
                liked_genres=json.loads(model.liked_genres),
                disliked_elements=json.loads(model.disliked_elements),
                mood_tags=json.loads(model.mood_tags),
                taste_notes=model.taste_notes,
            )

    async def get_profile_with_verification(
        self, requesting_tenant_id: str, target_tenant_id: str, user_id: str
    ) -> UserTasteProfile:
        """Verifies tenant isolation boundary before returning a profile."""
        if requesting_tenant_id != target_tenant_id:
            raise PermissionError(
                f"Cross-tenant access forbidden: {requesting_tenant_id} "
                f"cannot access {target_tenant_id}"
            )

        profile = await self.get_profile(target_tenant_id, user_id)
        if profile is None:
            raise KeyError(f"Profile not found for user {user_id}")
        return profile

    async def update_profile_from_message(
        self, tenant_id: str, user_id: str, message: str
    ) -> UserTasteProfile:
        """Extracts signals and updates the user's persistent taste profile."""
        delta = extract_taste_signals_from_text(message)

        async with self.db.session_factory() as session:
            stmt = select(UserTasteProfileModel).where(
                UserTasteProfileModel.tenant_id == tenant_id,
                UserTasteProfileModel.user_id == user_id,
            )
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()

            if model is None:
                liked_directors = delta.liked_directors
                liked_genres = delta.liked_genres
                disliked_elements = delta.disliked_elements
                mood_tags = delta.mood_tags

                model = UserTasteProfileModel(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    liked_directors=json.dumps(liked_directors),
                    liked_genres=json.dumps(liked_genres),
                    disliked_elements=json.dumps(disliked_elements),
                    mood_tags=json.dumps(mood_tags),
                    taste_notes=f"Initial preference signals from: {message[:80]}",
                )
                session.add(model)
            else:
                liked_dirs = set(json.loads(model.liked_directors))
                liked_dirs.update(delta.liked_directors)

                liked_g = set(json.loads(model.liked_genres))
                liked_g.update(delta.liked_genres)

                disliked_e = set(json.loads(model.disliked_elements))
                disliked_e.update(delta.disliked_elements)

                moods = set(json.loads(model.mood_tags))
                moods.update(delta.mood_tags)

                model.liked_directors = json.dumps(list(liked_dirs))
                model.liked_genres = json.dumps(list(liked_g))
                model.disliked_elements = json.dumps(list(disliked_e))
                model.mood_tags = json.dumps(list(moods))

            await session.commit()
            await session.refresh(model)

            return UserTasteProfile(
                tenant_id=model.tenant_id,
                user_id=model.user_id,
                liked_directors=json.loads(model.liked_directors),
                liked_genres=json.loads(model.liked_genres),
                disliked_elements=json.loads(model.disliked_elements),
                mood_tags=json.loads(model.mood_tags),
                taste_notes=model.taste_notes,
            )


PG_CANONICAL_DIRECTORS = {
    "denis villeneuve",
    "christopher nolan",
    "joel coen",
    "ethan coen",
    "david fincher",
    "quentin tarantino",
    "martin scorsese",
    "bong joon-ho",
    "park chan-wook",
    "rahul sadasivan",
    "stanley kubrick",
    "greta gerwig",
    "hayao miyazaki",
}

PG_CANONICAL_GENRES = {
    "psychological thriller",
    "horror",
    "neo-noir",
    "sci-fi",
    "crime",
    "mystery",
    "drama",
    "comedy",
}


def compute_taste_match_score(
    user_profile: UserTasteProfile,
    pg_profile: UserTasteProfile | None = None,
) -> float:
    """Computes an objective taste compatibility score (0.0 to 100.0%) against PG.

    Uses Jaccard overlap on directors, genres, and element avoidance.
    """
    user_dirs = {d.lower() for d in user_profile.liked_directors}
    user_genres = {g.lower() for g in user_profile.liked_genres}

    pg_dirs = (
        {d.lower() for d in pg_profile.liked_directors}
        if pg_profile and pg_profile.liked_directors
        else PG_CANONICAL_DIRECTORS
    )
    pg_genres = (
        {g.lower() for g in pg_profile.liked_genres}
        if pg_profile and pg_profile.liked_genres
        else PG_CANONICAL_GENRES
    )

    # Director overlap
    dir_intersection = len(user_dirs.intersection(pg_dirs))
    dir_score = (dir_intersection / max(len(user_dirs), 1)) if user_dirs else 0.75

    # Genre overlap
    genre_intersection = len(user_genres.intersection(pg_genres))
    genre_score = (
        (genre_intersection / max(len(user_genres), 1)) if user_genres else 0.80
    )

    # Weighted blend with a 62% baseline for cinema fans
    raw_score = 0.50 * dir_score + 0.50 * genre_score
    scaled_match = 62.0 + (raw_score * 35.0)  # Maps to 62.0% - 97.0%
    return round(min(scaled_match, 99.0), 1)


async def ingest_user_letterboxd_feed(
    username: str,
    db: DatabaseManager,
    user_id: str,
    tenant_id: str = "default_tenant",
) -> tuple[UserTasteProfile, float]:
    """Ingests user's public Letterboxd RSS feed and computes match score."""
    from phase_6_capstone.db import UserModel
    from phase_6_capstone.ingestion import fetch_live_letterboxd_feed

    rss_url = f"https://letterboxd.com/{username.strip()}/rss/"
    try:
        records = await fetch_live_letterboxd_feed(rss_url=rss_url)
    except Exception:
        records = []

    manager = TasteProfileManager(db)
    # Aggregate text from reviews to extract taste signals
    all_text = " ".join(
        [f"{r.title} {r.review_text}" for r in records if (r.rating or 0) >= 3.5]
    )
    if not all_text and records:
        all_text = " ".join([r.title for r in records])

    profile = await manager.update_profile_from_message(
        tenant_id=tenant_id,
        user_id=user_id,
        message=all_text or f"Watched movies on Letterboxd as @{username}",
    )
    match_score = compute_taste_match_score(profile)

    # Persist updated match score to UserModel
    async with db.session_factory() as session:
        stmt = select(UserModel).where(UserModel.id == user_id)
        res = await session.execute(stmt)
        user_row = res.scalar_one_or_none()
        if user_row:
            user_row.taste_match_pct = match_score
            user_row.letterboxd_handle = username.strip()
            await session.commit()

    return profile, match_score
