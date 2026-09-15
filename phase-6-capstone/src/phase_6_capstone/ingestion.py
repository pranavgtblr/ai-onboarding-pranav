"""Ingestion pipeline for PG Recommends: CSV parser and live Letterboxd RSS syncer."""

import csv
import html
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from phase_6_capstone.models import MovieRecord

LETTERBOXD_RSS_URL = "https://letterboxd.com/pranavg/rss/"
IMG_SRC_PATTERN = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

GENRE_KEYWORDS = {
    "Romance": ["romance", "romcom", "romantic", "love story", "chick flick"],
    "Comedy": [
        "comedy",
        "romcom",
        "hilarious",
        "humour",
        "humor",
        "funny",
        "satire",
    ],
    "Horror": [
        "horror",
        "slasher",
        "spooky",
        "scary",
        "chilling",
        "creepy",
        "blood",
        "monster",
    ],
    "Thriller": [
        "thriller",
        "suspense",
        "mystery",
        "neo-noir",
        "crime",
        "investigation",
    ],
    "Sci-Fi": [
        "sci-fi",
        "science fiction",
        "alien",
        "space",
        "cyberpunk",
        "futuristic",
    ],
    "Drama": ["drama", "biopic", "coming-of-age", "emotional", "character study"],
}


def _clean_html(text: str) -> str:
    """Removes HTML tags, unescapes entities (multi-pass), and cleans whitespace."""
    if not text:
        return ""
    unescaped = html.unescape(html.unescape(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", unescaped)
    cleaned = (
        cleaned.replace("&#039;", "'")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return " ".join(cleaned.split()).strip()


def _infer_genres(title: str, review: str, tags: list[str]) -> list[str]:
    """Infers genre tags from title, review text, and explicit Letterboxd tags."""
    combined = f"{title} {review} {' '.join(tags)}".lower()
    inferred = set(tags)
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            inferred.add(genre)
    return sorted(list(inferred))


def parse_reviews_csv(
    file_path: Path | str | None = None,
    csv_content: str | None = None,
    limit: int | None = None,
) -> list[MovieRecord]:
    """Parses exported Letterboxd reviews CSV into MovieRecord instances."""
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif csv_content:
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
    else:
        raise ValueError("Must provide either file_path or csv_content")

    records: list[MovieRecord] = []
    count = 0

    for row in rows:
        if limit and count >= limit:
            break

        raw_title = (row.get("Name") or "").strip()
        title = (
            html.unescape(html.unescape(raw_title))
            .replace("&#039;", "'")
            .replace("&#39;", "'")
        )
        if not title:
            continue

        year_val = row.get("Year", "").strip()
        try:
            year = int(year_val)
        except (ValueError, TypeError):
            year = 0

        rating_val = row.get("Rating", "").strip()
        try:
            rating = float(rating_val) if rating_val else None
        except (ValueError, TypeError):
            rating = None

        url = (row.get("Letterboxd URI") or "").strip()
        raw_review = (row.get("Review") or "").strip()
        review = (
            html.unescape(html.unescape(raw_review))
            .replace("&#039;", "'")
            .replace("&#39;", "'")
            .replace("&quot;", '"')
            .replace("&amp;", "&")
        )
        rewatch = (row.get("Rewatch") or "").strip().lower() in ("yes", "true", "1")
        watched_date = (row.get("Watched Date") or row.get("Date") or "").strip()
        tags_raw = (row.get("Tags") or "").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        genres = _infer_genres(title, review, tags)
        movie_id = f"mov_{abs(hash((title, year, url))) % 100000000}"

        records.append(
            MovieRecord(
                movie_id=movie_id,
                title=title,
                year=year,
                letterboxd_url=url,
                rating=rating,
                review_text=review,
                watched_date=watched_date or None,
                rewatch=rewatch,
                tags=tags,
                genres=genres,
            )
        )
        count += 1

    return records


def parse_letterboxd_rss_xml(xml_content: str) -> list[MovieRecord]:
    """Parses Letterboxd RSS syndication XML into structured MovieRecord list."""
    root = ET.fromstring(xml_content)
    channel = root.find("channel")
    if channel is None:
        return []

    records: list[MovieRecord] = []

    for item in channel.findall("item"):
        guid_elem = item.find("guid")
        guid = guid_elem.text if guid_elem is not None and guid_elem.text else None

        link_elem = item.find("link")
        link = (
            link_elem.text.strip() if link_elem is not None and link_elem.text else ""
        )

        title = ""
        year = 0
        rating = None
        rewatch = False
        watched_date = None

        for child in item:
            tag = child.tag
            text = child.text.strip() if child.text else ""
            if "filmTitle" in tag:
                title = text
            elif "filmYear" in tag:
                try:
                    year = int(text)
                except ValueError:
                    year = 0
            elif "memberRating" in tag:
                try:
                    rating = float(text)
                except ValueError:
                    rating = None
            elif "rewatch" in tag:
                rewatch = text.lower() in ("yes", "true", "1")
            elif "watchedDate" in tag:
                watched_date = text

        if not title:
            title_elem = item.find("title")
            title = (
                title_elem.text if title_elem is not None and title_elem.text else ""
            )

        title = (
            html.unescape(html.unescape(title))
            .replace("&#039;", "'")
            .replace("&#39;", "'")
        )

        desc_elem = item.find("description")
        desc_raw = desc_elem.text if desc_elem is not None and desc_elem.text else ""

        poster_url = None
        match = IMG_SRC_PATTERN.search(desc_raw)
        if match:
            poster_url = match.group(1)

        review_text = _clean_html(desc_raw)
        genres = _infer_genres(title, review_text, [])
        movie_id = f"mov_{abs(hash((title, year, link or guid))) % 100000000}"

        records.append(
            MovieRecord(
                movie_id=movie_id,
                title=title,
                year=year,
                letterboxd_url=link,
                rating=rating,
                review_text=review_text,
                watched_date=watched_date,
                rewatch=rewatch,
                poster_url=poster_url,
                guid=guid,
                genres=genres,
            )
        )

    return records


def sync_movies_to_store(
    records: list[MovieRecord], store: dict[str, MovieRecord]
) -> dict[str, int]:
    """Idempotently upserts movie records into a memory or database key-value store."""
    inserted = 0
    updated = 0

    for rec in records:
        key = rec.letterboxd_url or rec.guid or f"{rec.title}:{rec.year}"
        if key in store:
            store[key] = rec
            updated += 1
        else:
            store[key] = rec
            inserted += 1

    return {"inserted": inserted, "updated": updated, "total": len(store)}


async def fetch_live_letterboxd_feed(
    url: str = LETTERBOXD_RSS_URL,
) -> list[MovieRecord]:
    """Asynchronously fetches and parses the live Letterboxd RSS feed."""
    headers = {"User-Agent": "PGRecommends/1.0 (Mozilla/5.0)"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return parse_letterboxd_rss_xml(response.text)


def sync_cli() -> None:
    """CLI script entrypoint for manual or cron execution."""
    import asyncio

    async def _run() -> None:
        print(f"Fetching live Letterboxd feed from {LETTERBOXD_RSS_URL}...")
        records = await fetch_live_letterboxd_feed()
        print(f"Successfully parsed {len(records)} recent reviews from live RSS feed.")
        for r in records[:5]:
            print(f"- {r.title} ({r.year}) {r.star_display}: {r.review_text[:60]}...")

    asyncio.run(_run())
