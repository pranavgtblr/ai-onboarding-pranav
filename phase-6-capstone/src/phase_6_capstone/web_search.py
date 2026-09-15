"""Live internet cinema search engine for PG Recommends."""

import re
from typing import Any

import httpx

HTML_TAG_CLEANER = re.compile(r"<[^>]+>")


def search_cinema_web(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Searches live internet sources (Wikipedia API / DuckDuckGo) for cinema knowledge.

    Returns structured entries with title, snippet, and reference link.
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    # 1. Query Wikipedia API for verified film, director, and industry data
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_q,
            "utf8": 1,
            "format": "json",
        }
        headers = {"User-Agent": "PGRecommends/1.0 (contact: pranav.g@toobler.com)"}
        with httpx.Client(timeout=6.0, headers=headers) as client:
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("query", {}).get("search", [])
                entries = []
                for item in results[:limit]:
                    snippet_raw = item.get("snippet", "")
                    clean_snippet = HTML_TAG_CLEANER.sub(" ", snippet_raw)
                    clean_snippet = " ".join(clean_snippet.split())
                    title = item.get("title", "")
                    link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    entries.append(
                        {
                            "title": title,
                            "snippet": clean_snippet,
                            "source_url": link,
                        }
                    )
                if entries:
                    return entries
    except Exception:
        pass

    # 2. Query DuckDuckGo Instant Answer API fallback
    try:
        ddg_url = "https://api.duckduckgo.com/"
        ddg_params = {"q": clean_q, "format": "json", "no_html": 1}
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(ddg_url, params=ddg_params)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Abstract"):
                    return [
                        {
                            "title": data.get("Heading", clean_q),
                            "snippet": data.get("Abstract", ""),
                            "source_url": data.get(
                                "AbstractURL", "https://duckduckgo.com"
                            ),
                        }
                    ]
    except Exception:
        pass

    return []
