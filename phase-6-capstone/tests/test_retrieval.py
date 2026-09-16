"""Tests for the Hybrid Retrieval and Citation Engine in PG Recommends."""

import pytest

from phase_6_capstone.models import MovieRecord
from phase_6_capstone.retrieval import HybridMovieRetriever
from phase_6_capstone.web_search import search_cinema_web


@pytest.fixture
def sample_catalog() -> list[MovieRecord]:
    return [
        MovieRecord(
            movie_id="mov_1",
            title="Scream",
            year=1996,
            letterboxd_url="https://boxd.it/2ePSaX",
            rating=4.0,
            review_text=(
                "Scream was actually the first slasher movie that I had ever "
                "watched. It was a weird choice considering the fact that the "
                "whole movie was also a Satire making fun of the clichés of the "
                "horror/slasher genre. Great clues and foreshadowing."
            ),
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
                "Masterfully made and brilliantly acted. Each frame looks so "
                "meticulously crafted. The score was so haunting and Denzel!! "
                "Shakespearean dialogue at its finest."
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
                "inside a plain middle class house. Creepy psychological turmoil."
            ),
            genres=["Horror", "Mystery"],
            director="Rahul Sadasivan",
        ),
        MovieRecord(
            movie_id="mov_4",
            title="Red Notice",
            year=2021,
            letterboxd_url="https://boxd.it/2wXdTZ",
            rating=1.5,
            review_text=(
                "Shitty movies are shit, regardless of the stars, budget or language."
            ),
            genres=["Action", "Comedy"],
            director="Rawson Marshall Thurber",
        ),
        MovieRecord(
            movie_id="mov_5",
            title="Love at First Sight",
            year=2023,
            letterboxd_url="https://boxd.it/2eLove",
            rating=3.5,
            review_text="Delightful, sweet feel-good romance with genuine charm.",
            genres=["Comedy", "Romance"],
            director="Vanessa Caswill",
        ),
    ]


def test_hybrid_search_bm25_and_semantic(sample_catalog: list[MovieRecord]):
    """Test that searching returns the most relevant films."""
    retriever = HybridMovieRetriever(catalog=sample_catalog)

    # Search for slasher horror
    results = retriever.search("slasher horror meta satire clues", top_k=2)
    assert len(results) >= 1
    assert results[0].record.title == "Scream"

    # Search for Shakespeare Denzel
    results_macbeth = retriever.search("meticulous Shakespeare Denzel", top_k=2)
    assert len(results_macbeth) >= 1
    assert results_macbeth[0].record.title == "The Tragedy of Macbeth"


def test_retrieval_rating_filter(sample_catalog: list[MovieRecord]):
    """Verify that results can be filtered by minimum star rating."""
    retriever = HybridMovieRetriever(catalog=sample_catalog)

    # Filtering by min_rating=4.5 should exclude Scream (4.0) and Red Notice (1.5)
    results = retriever.search("movie", top_k=10, min_rating=4.5)
    assert len(results) == 1
    assert results[0].record.title == "The Tragedy of Macbeth"
    assert results[0].record.rating == 5.0


def test_citation_generation(sample_catalog: list[MovieRecord]):
    """Verify each retrieved item formats a valid, verifiable citation."""
    retriever = HybridMovieRetriever(catalog=sample_catalog)
    results = retriever.search("Malayalam modern horror chilling", top_k=1)
    assert len(results) == 1

    item = results[0]
    citation = item.to_citation()

    assert citation.title == "Bhoothakaalam"
    assert citation.year == 2022
    assert citation.rating == 4.0
    assert citation.letterboxd_url == "https://boxd.it/2vq7GF"
    assert "Malayalam" in citation.excerpt
    assert citation.citation_label.startswith("[PG Review: Bhoothakaalam")


def test_romcom_query_excludes_horror(sample_catalog: list[MovieRecord]):
    """Verify that asking for a romcom prioritizes romance/comedy and avoids horror."""
    retriever = HybridMovieRetriever(catalog=sample_catalog)
    results = retriever.search("Can you recommend a good romcom?", top_k=2)
    assert len(results) >= 1
    assert results[0].record.title == "Love at First Sight"
    assert not any(r.record.title in ["Scream", "Bhoothakaalam"] for r in results)


def test_live_internet_cinema_search():
    """Verify that cinema search queries the internet for film and director info."""
    results = search_cinema_web("Christopher Nolan director", limit=2)
    assert len(results) >= 1
    assert any("Nolan" in r["title"] or "Nolan" in r["snippet"] for r in results)
    assert "source_url" in results[0]


def test_reputed_critic_reviews_retrieval():
    """Verify multi-source retrieval of reputed critic reviews and consensus."""
    from phase_6_capstone.critic_reviews import fetch_reputed_critic_reviews

    # Test retrieval for a classic like La La Land
    reviews_lalaland = fetch_reputed_critic_reviews("La La Land", 2016)
    assert len(reviews_lalaland) >= 1
    portals = [r.portal_name for r in reviews_lalaland]
    assert any("Rotten Tomatoes" in p or "RogerEbert" in p for p in portals)
    assert reviews_lalaland[0].excerpt
    assert reviews_lalaland[0].review_url.startswith("http")

    # Test retrieval for Malayalam masterpiece Kumbalangi Nights
    reviews_kn = fetch_reputed_critic_reviews("Kumbalangi Nights", 2019)
    assert len(reviews_kn) >= 1
    kn_portals = [r.portal_name for r in reviews_kn]
    assert any("Guardian" in p or "Rotten Tomatoes" in p for p in kn_portals)


def test_action_query_excludes_romcom(sample_catalog: list[MovieRecord]):
    """Verify that asking for an action movie returns action and excludes romcoms."""
    retriever = HybridMovieRetriever(catalog=sample_catalog)
    results = retriever.search("suggest an action movie", top_k=3)
    assert len(results) >= 1
    assert results[0].record.title == "Red Notice"  # only action film in sample catalog
    assert not any(r.record.title in ["Love at First Sight", "Scream"] for r in results)


def test_retrieval_incorporates_taste_profile(
    sample_catalog: list[MovieRecord],
):
    """Verify that user taste profile boosts favored directors."""
    from phase_6_capstone.taste_engine import UserTasteProfile

    retriever = HybridMovieRetriever(catalog=sample_catalog)
    profile = UserTasteProfile(
        tenant_id="t1",
        user_id="u1",
        liked_directors=["Rahul Sadasivan"],
        disliked_elements=["slasher"],
    )

    results = retriever.search("horror movie", top_k=2, taste_profile=profile)
    assert len(results) >= 1
    assert results[0].record.title == "Bhoothakaalam"
    assert results[0].record.director == "Rahul Sadasivan"


def test_acclaimed_wider_cinema_retrieval():
    """Verify that wider acclaimed cinema retrieval surfaces world cinema picks."""
    from phase_6_capstone.acclaimed_cinema import find_acclaimed_wider_cinema

    picks = find_acclaimed_wider_cinema("suggest an action movie", limit=2)
    assert len(picks) >= 1
    assert any(p.title in ["The Raid", "Kill", "S.W.A.T."] for p in picks)
    assert picks[0].source_type == "acclaimed_cinema"
    assert picks[0].source_portal in [
        "Rotten Tomatoes",
        "RogerEbert.com",
        "The Guardian",
        "The Hollywood Reporter",
        "Variety",
        "The Independent",
        "The New York Times",
        "Metacritic",
    ]
