"""Tests for Semantic Query Caching in PG Recommends."""

import pytest

from phase_6_capstone.cache import (
    SemanticQueryCache,
    compute_query_hash,
    normalize_query_for_cache,
)
from phase_6_capstone.db import DatabaseManager


@pytest.mark.asyncio
async def test_semantic_query_normalization():
    """Test punctuation and whitespace normalization produces stable hashes."""
    q1 = "Recommend me some psychological thrillers?"
    q2 = "recommend me some psychological thrillers"
    q3 = "  Recommend  ME  some   psychological thrillers!  "

    assert normalize_query_for_cache(q1) == "recommend me some psychological thrillers"
    assert compute_query_hash(q1) == compute_query_hash(q2)
    assert compute_query_hash(q1) == compute_query_hash(q3)


@pytest.mark.asyncio
async def test_semantic_cache_set_and_get():
    """Test that cached responses are stored and retrieved with hit count tracking."""
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db.init_models()

    cache = SemanticQueryCache(db=db, max_mem_entries=10)

    # Miss before set
    miss = await cache.get("What did PG think of Dune?")
    assert miss is None

    # Set entry
    sample_response = "PG thought Dune: Part Two was a masterclass in sound design."
    sample_citations = [
        {"title": "Dune: Part Two", "rating": 5.0, "source": "Letterboxd"}
    ]
    sample_critic_citations = [
        {"movie_title": "Dune: Part Two", "portal_name": "Variety"}
    ]

    await cache.set(
        query="What did PG think of Dune?",
        response=sample_response,
        citations=sample_citations,
        critic_citations=sample_critic_citations,
    )

    # Hit immediately from memory tier
    hit1 = await cache.get("what did pg think of dune?")
    assert hit1 is not None
    assert hit1["from_cache"] is True
    assert hit1["response"] == sample_response
    assert len(hit1["citations"]) == 1
    assert hit1["hit_count"] == 2

    # Clear memory cache to test fallback to DB persistent tier
    cache._mem_cache.clear()
    hit_db = await cache.get("What did PG think of Dune?")
    assert hit_db is not None
    assert hit_db["response"] == sample_response
    assert hit_db["hit_count"] >= 2
