"""Tests for Letterboxd CSV ingestion and live RSS incremental sync."""

from pathlib import Path

import pytest

from phase_6_capstone.ingestion import (
    MovieRecord,
    parse_letterboxd_rss_xml,
    parse_reviews_csv,
    sync_movies_to_store,
)

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:letterboxd="https://letterboxd.com">
  <channel>
    <title>Letterboxd - pranavg</title>
    <link>https://letterboxd.com/pranavg/</link>
    <item>
      <title>The Devil Wears Prada, 2006 - ★★★★</title>
      <link>https://letterboxd.com/pranavg/film/the-devil-wears-prada/1/</link>
      <guid isPermaLink="false">letterboxd-review-1437797457</guid>
      <pubDate>Fri, 7 Aug 2026 19:09:25 +1200</pubDate>
      <letterboxd:watchedDate>2026-08-05</letterboxd:watchedDate>
      <letterboxd:rewatch>Yes</letterboxd:rewatch>
      <letterboxd:filmTitle>The Devil Wears Prada</letterboxd:filmTitle>
      <letterboxd:filmYear>2006</letterboxd:filmYear>
      <letterboxd:memberRating>4.0</letterboxd:memberRating>
      <description><![CDATA[<p><img src="https://a.ltrbxd.com/poster.jpg"/></p>
<p>Meryl Streep is the best part of this movie...</p>]]></description>
    </item>
    <item>
      <title>Ela Veezha Poonchira, 2022 - ★★★★</title>
      <link>https://letterboxd.com/pranavg/film/ela-veezha-poonchira/</link>
      <guid isPermaLink="false">letterboxd-review-1432716745</guid>
      <pubDate>Tue, 4 Aug 2026 21:34:21 +1200</pubDate>
      <letterboxd:watchedDate>2026-08-02</letterboxd:watchedDate>
      <letterboxd:rewatch>No</letterboxd:rewatch>
      <letterboxd:filmTitle>Ela Veezha Poonchira</letterboxd:filmTitle>
      <letterboxd:filmYear>2022</letterboxd:filmYear>
      <letterboxd:memberRating>4.0</letterboxd:memberRating>
      <description><![CDATA[<p>The atmosphere was great...</p>]]></description>
    </item>
  </channel>
</rss>
"""


def test_parse_reviews_csv_sample():
    """Test parsing a sample reviews CSV format."""
    csv_content = (
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review,Tags,Watched Date\n"
        "2021-10-27,Scream,1996,https://boxd.it/2ePSaX,4,Yes,"
        '"Scream was great.",,2021-10-27\n'
        "2021-10-27,Dune,2021,https://boxd.it/2ePSRR,2.5,,Disappointing.,,2021-10-27\n"
    )

    records = parse_reviews_csv(csv_content=csv_content)
    assert len(records) == 2

    scream = records[0]
    assert scream.title == "Scream"
    assert scream.year == 1996
    assert scream.rating == 4.0
    assert scream.rewatch is True
    assert "Scream was great." in scream.review_text
    assert scream.letterboxd_url == "https://boxd.it/2ePSaX"

    dune = records[1]
    assert dune.title == "Dune"
    assert dune.rating == 2.5
    assert dune.rewatch is False


def test_parse_real_reviews_csv():
    """Verify parsing against the real reviews.csv file in phase-6-capstone."""
    csv_path = Path(__file__).parent.parent / "reviews.csv"
    if not csv_path.exists():
        pytest.skip("reviews.csv not found")

    records = parse_reviews_csv(file_path=csv_path, limit=50)
    assert len(records) == 50
    for rec in records:
        assert isinstance(rec, MovieRecord)
        assert rec.title
        assert rec.year > 1900
        assert rec.letterboxd_url.startswith("http")


def test_parse_rss_xml():
    """Test extracting structured records from Letterboxd RSS XML."""
    records = parse_letterboxd_rss_xml(SAMPLE_RSS_XML)
    assert len(records) == 2

    prada = records[0]
    assert prada.title == "The Devil Wears Prada"
    assert prada.year == 2006
    assert prada.rating == 4.0
    assert prada.rewatch is True
    assert "Meryl Streep" in prada.review_text
    assert prada.poster_url == "https://a.ltrbxd.com/poster.jpg"
    assert prada.guid == "letterboxd-review-1437797457"

    ela = records[1]
    assert ela.title == "Ela Veezha Poonchira"
    assert ela.year == 2022
    assert ela.rating == 4.0
    assert "The atmosphere" in ela.review_text


def test_incremental_sync_deduplication():
    """Verify that syncing updates existing records without duplication."""
    store: dict[str, MovieRecord] = {}

    initial_csv = (
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review,Tags,Watched Date\n"
        "2021-10-27,Scream,1996,https://boxd.it/2ePSaX,4,Yes,"
        "Initial review.,,2021-10-27\n"
    )
    records_1 = parse_reviews_csv(csv_content=initial_csv)
    stats_1 = sync_movies_to_store(records_1, store)
    assert stats_1["inserted"] == 1
    assert stats_1["updated"] == 0
    assert len(store) == 1
    assert store["https://boxd.it/2ePSaX"].review_text == "Initial review."

    # Now sync an updated version of the same review
    updated_csv = (
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review,Tags,Watched Date\n"
        "2021-10-27,Scream,1996,https://boxd.it/2ePSaX,4.5,Yes,"
        "Updated review.,,2021-10-27\n"
        "2021-10-29,No Time to Die,2021,https://boxd.it/2fasyB,4,,Good.,,2021-10-29\n"
    )
    records_2 = parse_reviews_csv(csv_content=updated_csv)

    stats_2 = sync_movies_to_store(records_2, store)
    assert stats_2["inserted"] == 1
    assert stats_2["updated"] == 1
    assert len(store) == 2
    assert store["https://boxd.it/2ePSaX"].rating == 4.5
    assert store["https://boxd.it/2ePSaX"].review_text == "Updated review."
