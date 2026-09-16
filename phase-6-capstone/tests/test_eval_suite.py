"""Automated Test for Phase 6 Capstone Comprehensive Evaluation Suite."""

from pathlib import Path

import pytest

from phase_6_capstone.db import DatabaseManager
from phase_6_capstone.eval import run_full_evaluation
from phase_6_capstone.models import MovieRecord


@pytest.fixture
def eval_catalog() -> list[MovieRecord]:
    return [
        MovieRecord(
            movie_id="mov_1",
            title="Scream",
            year=1996,
            rating=4.0,
            review_text=(
                "Scream was actually the first slasher movie that I watched... "
                "Knowing the killer's identity and noticing clues is awesome."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/scream-1996/",
        ),
        MovieRecord(
            movie_id="mov_2",
            title="Bhoothakaalam",
            year=2022,
            rating=4.0,
            review_text=(
                "A modern horror movie in Malayalam was long overdue, and this is it! "
                "Atmospheric psychological dread inside a middle class house."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/bhoothakaalam/",
        ),
        MovieRecord(
            movie_id="mov_3",
            title="The Tragedy of Macbeth",
            year=2021,
            rating=5.0,
            review_text=(
                "Masterfully made and brilliantly acted. Each frame looks crafted. "
                "The score was haunting. And Denzel!! Chills, literal chills."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/the-tragedy-of-macbeth-2021/",
        ),
        MovieRecord(
            movie_id="mov_4",
            title="Late Night",
            year=2019,
            rating=3.5,
            review_text=(
                "Nice, funny, entertaining, feel-good pleasure. Workplace comedy. "
                "Emma Thompson is brilliant."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/late-night-2019/",
        ),
        MovieRecord(
            movie_id="mov_5",
            title="Animal",
            year=2023,
            rating=0.5,
            review_text=(
                "Toxic, bloated, three-hour assault on common sense and posturing."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/animal-2023/",
        ),
        MovieRecord(
            movie_id="mov_6",
            title="Red Notice",
            year=2021,
            rating=1.5,
            review_text=(
                "Shitty movies are shit regardless of stars, budget, or language!"
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/red-notice/",
        ),
        MovieRecord(
            movie_id="mov_7",
            title="Hridayam",
            year=2022,
            rating=1.0,
            review_text=(
                "The number of songs and song montages are inversely proportional "
                "to dialogue quality."
            ),
            letterboxd_url="https://letterboxd.com/pranavg/film/hridayam/",
        ),
        MovieRecord(
            movie_id="mov_8",
            title="La La Land",
            year=2016,
            rating=5.0,
            review_text="Musical masterpiece with cinematography and romance.",
            letterboxd_url="https://letterboxd.com/pranavg/film/la-la-land/",
        ),
        MovieRecord(
            movie_id="mov_9",
            title="The Batman",
            year=2022,
            rating=4.5,
            review_text="Atmospheric neo-noir detective thriller with stunning score.",
            letterboxd_url="https://letterboxd.com/pranavg/film/the-batman/",
        ),
        MovieRecord(
            movie_id="mov_10",
            title="Kumbalangi Nights",
            year=2019,
            rating=5.0,
            review_text="Heartwarming family drama set in a coastal Kerala village.",
            letterboxd_url="https://letterboxd.com/pranavg/film/kumbalangi-nights/",
        ),
    ]


@pytest.mark.asyncio
async def test_full_evaluation_benchmarks(eval_catalog: list[MovieRecord]):
    """Execute evaluation suite, assert metric thresholds, and verify report."""
    test_db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    output_dir = Path(__file__).parent.parent / "reports"

    report = await run_full_evaluation(
        catalog=eval_catalog,
        db=test_db,
        output_dir=output_dir,
    )

    summary = report["benchmark_summary"]

    # 1. Retrieval Recall@5 >= 80%
    assert summary["recall_at_5"] >= 0.80, f"Recall@5 too low: {summary['recall_at_5']}"

    # 2. NDCG@5 >= 0.70
    assert summary["ndcg_at_5"] >= 0.70, f"NDCG@5 too low: {summary['ndcg_at_5']}"

    # 3. Citation Fidelity == 100%
    assert summary["citation_fidelity"] == 1.0, (
        f"Citation fidelity failed: {summary['citation_fidelity']}"
    )

    # 4. Taste Learning Agreement >= 85%
    assert summary["taste_learning_agreement"] >= 0.85, (
        f"Taste agreement too low: {summary['taste_learning_agreement']}"
    )

    # 5. Verify reports generated
    assert (output_dir / "EVALUATION_REPORT.md").exists()
    assert (output_dir / "evaluation_report.json").exists()
