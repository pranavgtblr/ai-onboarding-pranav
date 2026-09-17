"""Dynamic Movie Poster & Artwork Resolution Engine for PG Recommends.

Fetches real-time, high-resolution posters and backdrops dynamically via:
1. The Movie Database (TMDB) API search (when TMDB_API_KEY is configured)
2. Live Letterboxd OpenGraph CDN resolution (via movie URL or slug)
3. High-performance in-memory cache to eliminate redundant network fetches
4. Dynamic SVG cinematic artwork fallback
"""

import json
import logging
import os
import re
import urllib.parse
import urllib.request

from phase_6_capstone.config import settings

logger = logging.getLogger(__name__)

# In-memory cache for resolved posters (title_year -> (poster_url, backdrop_url))
_POSTER_CACHE: dict[str, tuple[str, str]] = {}


def fetch_tmdb_poster(
    title: str,
    year: int | None = None,
    api_key: str | None = None,
) -> tuple[str, str] | None:
    """Dynamically queries TMDB Search API for official movie poster and backdrop."""
    key = (
        api_key or getattr(settings, "tmdb_api_key", None) or os.getenv("TMDB_API_KEY")
    )
    if not key or key.startswith("your_"):
        return None

    query_str = urllib.parse.quote(title.strip())
    url = f"https://api.themoviedb.org/3/search/movie?api_key={key}&query={query_str}"
    if year:
        url += f"&year={year}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PGRecommends/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return None
            first = results[0]
            p_path = first.get("poster_path")
            b_path = first.get("backdrop_path")
            if not p_path:
                return None
            poster = f"https://image.tmdb.org/t/p/w500{p_path}"
            backdrop = f"https://image.tmdb.org/t/p/w1280{b_path}" if b_path else poster
            return poster, backdrop
    except Exception as e:
        logger.debug("TMDB search failed for %s: %s", title, e)
        return None


def fetch_letterboxd_poster(url: str) -> tuple[str, str] | None:
    """Extracts high-resolution og:image poster from a Letterboxd / boxd.it page."""
    if not url or ("letterboxd" not in url and "boxd.it" not in url):
        return None
    try:
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
            if m:
                img_url = m.group(1)
                return img_url, img_url
    except Exception as e:
        logger.debug("Letterboxd scrape failed for %s: %s", url, e)
        return None


def search_letterboxd_slug(
    title: str, year: int | None = None
) -> tuple[str, str] | None:
    """Generates candidate Letterboxd slugs and fetches official artwork."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        return None

    candidates = [f"https://letterboxd.com/film/{slug}/"]
    if year:
        candidates.append(f"https://letterboxd.com/film/{slug}-{year}/")

    for cand in candidates:
        res = fetch_letterboxd_poster(cand)
        if res:
            return res
    return None


def generate_cinematic_placeholder(
    title: str, year: int | None = None
) -> tuple[str, str]:
    """Dynamically generates modern cinematic placeholder cards for the given film."""
    encoded_title = urllib.parse.quote(title)
    year_str = str(year) if year else "Cinema"
    poster_url = f"https://placehold.co/400x600/141a20/00e054?text={encoded_title}+{year_str}&font=montserrat"
    backdrop_url = f"https://placehold.co/1280x720/0c0f12/40bcf4?text={encoded_title}&font=montserrat"
    return poster_url, backdrop_url


def resolve_movie_poster(
    title: str,
    year: int | None = None,
    letterboxd_url: str | None = None,
) -> tuple[str, str]:
    """Resolves movie poster and backdrop dynamically.

    Resolution Strategy:
    1. Check in-memory LRU cache.
    2. Query TMDB Search API (if TMDB_API_KEY is configured).
    3. Scrape Letterboxd og:image if letterboxd_url is available.
    4. Query Letterboxd public slug endpoint (/film/{slug}/).
    5. Fall back to styled cinematic placeholder.
    """
    clean_title = title.lower().strip()
    cache_key = f"{clean_title}_{year or ''}"
    if cache_key in _POSTER_CACHE:
        return _POSTER_CACHE[cache_key]

    # 1. Try TMDB Search API
    tmdb_res = fetch_tmdb_poster(title, year)
    if tmdb_res:
        _POSTER_CACHE[cache_key] = tmdb_res
        return tmdb_res

    # 2. Try provided Letterboxd URL
    if letterboxd_url:
        lb_res = fetch_letterboxd_poster(letterboxd_url)
        if lb_res:
            _POSTER_CACHE[cache_key] = lb_res
            return lb_res

    # 3. Try Letterboxd slug search
    slug_res = search_letterboxd_slug(title, year)
    if slug_res:
        _POSTER_CACHE[cache_key] = slug_res
        return slug_res

    # 4. Fallback placeholder
    fallback = generate_cinematic_placeholder(title, year)
    _POSTER_CACHE[cache_key] = fallback
    return fallback
