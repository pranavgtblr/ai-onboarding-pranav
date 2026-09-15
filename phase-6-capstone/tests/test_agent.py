"""Tests for the Stateful LangGraph Agent and Human Escalation."""

import pytest

from phase_6_capstone.agent import CapstoneAgent
from phase_6_capstone.db import DatabaseManager
from phase_6_capstone.models import MovieRecord
from phase_6_capstone.retrieval import HybridMovieRetriever
from phase_6_capstone.taste_engine import TasteProfileManager


@pytest.fixture
def agent_setup():
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    catalog = [
        MovieRecord(
            movie_id="mov_1",
            title="Scream",
            year=1996,
            letterboxd_url="https://boxd.it/2ePSaX",
            rating=4.0,
            review_text="Great meta slasher with clever foreshadowing.",
            genres=["Horror", "Mystery"],
            director="Wes Craven",
        ),
        MovieRecord(
            movie_id="mov_2",
            title="The Tragedy of Macbeth",
            year=2021,
            letterboxd_url="https://boxd.it/2ub67l",
            rating=5.0,
            review_text=(
                "Denzel Washington and gorgeous black-and-white cinematography."
            ),
            genres=["Drama", "Thriller"],
            director="Joel Coen",
        ),
        MovieRecord(
            movie_id="mov_3",
            title="Bhoothakaalam",
            year=2022,
            letterboxd_url="https://boxd.it/2vq7GF",
            rating=4.0,
            review_text="Chilling psychological horror inside an ordinary house.",
            genres=["Horror", "Mystery"],
            director="Rahul Sadasivan",
        ),
    ]
    retriever = HybridMovieRetriever(catalog)
    taste_manager = TasteProfileManager(db)
    agent = CapstoneAgent(
        retriever=retriever,
        taste_manager=taste_manager,
        db=db,
    )
    return agent, db


@pytest.mark.asyncio
async def test_agent_recommendation_flow_with_citations(agent_setup):
    """Test standard recommendation request generates response with citations."""
    agent, db = agent_setup
    await db.init_models()

    state = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_1",
        conversation_id="conv_101",
        message="Suggest an atmospheric Malayalam horror movie.",
    )

    assert state.escalation_status is None
    assert len(state.citations) >= 1
    assert state.citations[0].title == "Bhoothakaalam"
    assert "https://boxd.it/2vq7GF" in state.citations[0].letterboxd_url
    assert "Bhoothakaalam" in state.final_response
    assert "★" in state.final_response


@pytest.mark.asyncio
async def test_agent_taste_profile_learning(agent_setup):
    """Test that conversing with the agent updates the user's taste profile."""
    agent, db = agent_setup
    await db.init_models()

    await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_2",
        conversation_id="conv_102",
        message="I love Joel Coen and Denzel Washington dramas.",
    )

    profile = await agent.taste_manager.get_profile("tenant_alpha", "user_2")
    assert profile is not None
    assert any("coen" in d.lower() for d in profile.liked_directors)


@pytest.mark.asyncio
async def test_agent_human_escalation_flow(agent_setup):
    """Test that explicit request to speak with PG triggers human escalation."""
    agent, db = agent_setup
    await db.init_models()

    state = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_3",
        conversation_id="conv_103",
        message=(
            "I have an urgent request. Can I escalate to PG directly "
            "for festival curation?"
        ),
    )

    assert state.escalation_status == "escalated"
    assert state.escalation_ticket_id is not None
    assert state.escalation_ticket_id.startswith("TICK-")
    assert "escalat" in state.final_response.lower()
    assert state.escalation_ticket_id in state.final_response
