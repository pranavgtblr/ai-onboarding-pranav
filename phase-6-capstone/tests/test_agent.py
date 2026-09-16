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
            review_text=(
                "A proper modern horror movie in Malayalam! Chilling atmosphere "
                "inside an ordinary house."
            ),
            genres=["Horror", "Mystery"],
            director="Rahul Sadasivan",
        ),
        MovieRecord(
            movie_id="mov_4",
            title="Animal",
            year=2023,
            letterboxd_url="https://boxd.it/5Hbz5z",
            rating=0.5,
            review_text="Sandeep Reddy Vanga's Animal is regressive and toxic slop.",
            genres=["Action", "Crime", "Drama"],
            director="Sandeep Reddy Vanga",
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

    # Verify Multi-Source Reputed Critic Citations strictly from allowed portals
    assert len(state.critic_citations) >= 1
    assert any(
        "Independent" in c.portal_name or "Rotten Tomatoes" in c.portal_name
        for c in state.critic_citations
    )
    for c in state.critic_citations:
        assert c.portal_name in [
            "RogerEbert.com",
            "Variety",
            "The Independent",
            "The New York Times",
            "The Hollywood Reporter",
            "The Guardian",
            "Rotten Tomatoes",
            "Metacritic",
        ]


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


@pytest.mark.asyncio
async def test_agent_recommendations_go_beyond_csv_to_acclaimed_cinema(
    agent_setup,
):
    """Verify that agent recommendations span both diary and wider acclaimed cinema."""
    agent, db = agent_setup
    await db.init_models()

    state = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_4",
        conversation_id="conv_104",
        message=("Can you suggest an action movie? I want intense martial arts."),
    )

    assert state.escalation_status is None
    # Verify citations include acclaimed cinema outside PG's sample catalog
    has_acclaimed = any(c.source_type == "acclaimed_cinema" for c in state.citations)
    assert has_acclaimed
    acclaimed_titles = [
        c.title for c in state.citations if c.source_type == "acclaimed_cinema"
    ]
    assert any(
        t in ["The Raid", "John Wick: Chapter 4", "Hard Boiled"]
        for t in acclaimed_titles
    )

    # Verify conversation mentions action films and does not recommend romcoms
    assert "action" in state.final_response.lower()
    assert not any(
        rc in state.final_response.lower()
        for rc in ["love at first sight", "frances ha", "romcom"]
    )


@pytest.mark.asyncio
async def test_agent_handles_genre_correction_and_excludes_rejected_movie(
    agent_setup,
):
    """Verify that agent handles negative feedback and excludes the rejected movie."""
    agent, db = agent_setup
    await db.init_models()

    # Turn 1: user asks to explore action
    state_1 = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_5",
        conversation_id="conv_105",
        message="I want to explore action",
    )
    assert not any("marwencol" in c.title.lower() for c in state_1.citations)

    # Turn 2: user corrects with negative feedback
    state_2 = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_5",
        conversation_id="conv_105",
        message="marwencole is not an action movie",
    )
    # Ensure marwencol is strictly excluded
    assert not any("marwencol" in c.title.lower() for c in state_2.citations)
    # Ensure response acknowledges correction and does not repeat paragraph
    assert (
        "mistake" in state_2.final_response.lower()
        or "right" in state_2.final_response.lower()
    )
    assert state_2.final_response != state_1.final_response


@pytest.mark.asyncio
async def test_agent_handles_specific_movie_query_and_roasts_hated_film(agent_setup):
    """Verify asking about a specific hated movie roasts it without recommendations."""
    agent, db = agent_setup
    await db.init_models()

    state = await agent.run_turn(
        tenant_id="tenant_alpha",
        user_id="user_6",
        conversation_id="conv_106",
        message="What about animal (2023)?",
    )

    # Citations must ONLY be Animal, no other recommendations like Babe
    assert len(state.citations) == 1
    assert state.citations[0].title == "Animal"
    assert state.citations[0].rating == 0.5

    # Final response must express hatred/disdain, not recommend
    resp_low = state.final_response.lower()
    assert (
        "loathed" in resp_low
        or "half-star" in resp_low
        or "0.5" in resp_low
        or "sandeep" in resp_low
    )
    assert "definitely check out" not in resp_low
    assert "another standout" not in resp_low

    # Critic citations must be for Animal from allowed portals
    assert len(state.critic_citations) >= 1
    assert any(
        "guardian" in c.portal_name.lower() or "rotten" in c.portal_name.lower()
        for c in state.critic_citations
    )
    assert not any("babe" in c.movie_title.lower() for c in state.critic_citations)
