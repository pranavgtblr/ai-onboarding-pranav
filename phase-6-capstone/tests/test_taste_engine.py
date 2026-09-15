"""Tests for Taste Profile Engine and Multi-Tenant Isolation."""

import pytest

from phase_6_capstone.db import DatabaseManager
from phase_6_capstone.taste_engine import (
    TasteProfileManager,
    extract_taste_signals_from_text,
)


@pytest.fixture
def test_db():
    db = DatabaseManager(database_url="sqlite+aiosqlite:///:memory:")
    return db


@pytest.mark.asyncio
async def test_taste_profile_signal_extraction():
    """Verify extracting preferences and dislikes from user dialogue."""
    utterance = (
        "I really love psychological thrillers and films by Denis Villeneuve, "
        "especially with moody cinematography. "
        "But I hate jump scares and cheap slasher gore."
    )

    delta = extract_taste_signals_from_text(utterance)

    assert any("thriller" in g.lower() for g in delta.liked_genres)
    assert any("villeneuve" in d.lower() for d in delta.liked_directors)
    assert any(
        "jump scare" in dis.lower() or "gore" in dis.lower()
        for dis in delta.disliked_elements
    )


@pytest.mark.asyncio
async def test_taste_profile_persistence(test_db: DatabaseManager):
    """Test creating and progressively updating a user taste profile in DB."""
    await test_db.init_models()
    manager = TasteProfileManager(db=test_db)

    # First turn
    turn_1 = "I love David Fincher and Christopher Nolan movies."
    profile_1 = await manager.update_profile_from_message(
        tenant_id="stream_tenant",
        user_id="user_101",
        message=turn_1,
    )
    assert "David Fincher" in profile_1.liked_directors or any(
        "fincher" in d.lower() for d in profile_1.liked_directors
    )

    # Second turn adds new likes without wiping previous ones
    turn_2 = "I also enjoy neo-noir and cyberpunk."
    profile_2 = await manager.update_profile_from_message(
        tenant_id="stream_tenant",
        user_id="user_101",
        message=turn_2,
    )
    assert any("noir" in g.lower() for g in profile_2.liked_genres)
    # Check that Fincher is still remembered!
    assert any("fincher" in d.lower() for d in profile_2.liked_directors)


@pytest.mark.asyncio
async def test_tenant_and_user_isolation(test_db: DatabaseManager):
    """Verify strict tenant and user isolation: Tenant A cannot read Tenant B."""
    await test_db.init_models()
    manager = TasteProfileManager(db=test_db)

    # Create profile for User 1 in Tenant A
    await manager.update_profile_from_message(
        tenant_id="tenant_A",
        user_id="user_alice",
        message="I love Wes Anderson and quirky comedies.",
    )

    # User Bob in Tenant B tries to fetch Alice's data or has his own
    bob_profile = await manager.get_profile(
        tenant_id="tenant_B",
        user_id="user_alice",  # Same user_id, different tenant
    )
    assert bob_profile is None

    # An unauthorized cross-tenant query returns None/Forbidden
    with pytest.raises(PermissionError):
        await manager.get_profile_with_verification(
            requesting_tenant_id="tenant_B",
            target_tenant_id="tenant_A",
            user_id="user_alice",
        )
