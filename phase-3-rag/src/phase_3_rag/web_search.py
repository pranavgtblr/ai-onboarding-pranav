"""Web Search RAG module for Project D.

Provides:
1. Query rewriting: converts conversational user questions into keyword-dense
   search engine queries using Gemini or rule-based heuristic fallbacks.
2. Search provider abstraction: DuckDuckGoSearchProvider (zero-key web scraper)
   and MockSearchProvider (deterministic offline/test provider).
3. Resilient page content extraction: fetches full page text while handling
   dead links, paywalls, and junk pages without crashing.
4. Publication date extraction: extracts dates from Schema.org, meta tags, and
   time elements.
5. Grounded answer synthesis: produces answers citing source URLs and
   publication dates.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
from abc import ABC, abstractmethod
from enum import Enum

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class FetchStatus(str, Enum):
    """Classification status for web page content extraction."""

    SUCCESS = "SUCCESS"
    DEAD_LINK = "DEAD_LINK"
    PAYWALL = "PAYWALL"
    JUNK_PAGE = "JUNK_PAGE"
    TIMEOUT = "TIMEOUT"
    FETCH_ERROR = "FETCH_ERROR"
    SNIPPET_ONLY = "SNIPPET_ONLY"


class SourceCitation(BaseModel):
    """Citation metadata pairing source URL with publication date and title."""

    title: str = Field(..., description="Title of the cited source.")
    url: str = Field(..., description="Canonical URL of the source.")
    published_date: str | None = Field(
        default=None, description="Extracted publication or modified date."
    )
    domain: str = Field(default="", description="Hostname/domain of the source.")


class SearchResult(BaseModel):
    """A single retrieved web search result."""

    title: str = Field(..., description="Title of the web page.")
    url: str = Field(..., description="Full canonical destination URL.")
    snippet: str = Field(..., description="Search engine snippet/excerpt.")
    rank: int = Field(..., description="1-based rank in search results.")
    published_date: str | None = Field(
        default=None,
        description="Extracted publication or last modified date.",
    )
    page_content: str | None = Field(
        default=None,
        description="Full extracted readable body text if fetched.",
    )
    fetch_status: FetchStatus = Field(
        default=FetchStatus.SNIPPET_ONLY,
        description="Status of full page extraction.",
    )
    fetch_error: str | None = Field(
        default=None,
        description="Failure reason if page extraction failed.",
    )

    @property
    def effective_content(self) -> str:
        """Return full page content if successfully extracted, else snippet."""
        if self.fetch_status == FetchStatus.SUCCESS and self.page_content:
            return self.page_content
        return self.snippet


class QueryRewriteResult(BaseModel):
    """The result of rewriting a user's conversational question for search."""

    original_query: str = Field(..., description="The original user question.")
    search_query: str = Field(..., description="Primary keyword search query.")
    alternative_queries: list[str] = Field(
        default_factory=list,
        description="Alternative search queries for broader coverage.",
    )
    intent: str = Field(default="", description="Inferred information-seeking intent.")
    explanation: str = Field(
        default="", description="Rationale behind query transformation."
    )


class WebSearchRAGResponse(BaseModel):
    """Complete response from the Web Search RAG pipeline."""

    original_question: str
    rewritten_query: QueryRewriteResult
    search_results: list[SearchResult]
    answer: str
    citations: list[str] = Field(
        default_factory=list, description="Unique source URLs referenced."
    )
    source_citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Structured citations with URLs and publication dates.",
    )


# -----------------------------------------------------------------------------
# Query Rewriting
# -----------------------------------------------------------------------------

REWRITE_PROMPT_TEMPLATE = """You are an expert search engine query optimizer.
Your task is to transform conversational, ambiguous, or multi-part user questions
into high-precision keyword search queries optimized for search engine indexing.

Instructions:
1. Strip conversational filler (e.g. "Can you tell me", "I want to know", "please").
2. Resolve pronouns and retain all essential entities, names, technologies, and dates.
3. If the user asks about 'latest' or 'recent' events, append temporal
   keywords if relevant.
4. Generate 1 primary search query and 1-2 alternative queries.
5. Infer the search intent (e.g., informational, transactional, navigational).

User Question: {question}

Return ONLY valid JSON matching this exact structure:
{{
  "search_query": "primary keyword search query",
  "alternative_queries": ["alternative query 1", "alternative query 2"],
  "intent": "informational/navigational/troubleshooting",
  "explanation": "brief reason for rewriting"
}}
"""


def heuristic_rewrite_query(question: str) -> QueryRewriteResult:
    """Deterministic, rule-based fallback for query rewriting.

    Used when no LLM API key is present or when running offline.
    """
    cleaned = question.strip()
    patterns = [
        r"^(?:could|can)\s+you\s+(?:please\s+)?(?:tell|explain|show)\s+me\s+",
        r"^(?:i\s+(?:want|need|would\s+like)\s+to\s+know\s+)",
        r"^(?:what\s+is|what\s+are|who\s+is|where\s+is|how\s+to|why\s+is)\s+",
        r"^(?:please\s+)",
    ]
    transformed = cleaned
    for pat in patterns:
        transformed = re.sub(pat, "", transformed, flags=re.IGNORECASE)

    transformed = transformed.rstrip("?.! ")

    for fluff in [" recently", " right now", " please", " exactly", " basically"]:
        transformed = re.sub(re.escape(fluff), "", transformed, flags=re.IGNORECASE)

    search_query = transformed.strip() or cleaned
    alt_query = f"{search_query} documentation overview"

    return QueryRewriteResult(
        original_query=question,
        search_query=search_query,
        alternative_queries=[alt_query],
        intent="informational",
        explanation="Rule-based heuristic stripped conversational boilerplate.",
    )


def rewrite_search_query(
    question: str,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> QueryRewriteResult:
    """Rewrite a user question into a search query using Gemini or fallback."""
    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        return heuristic_rewrite_query(question)

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={active_key}"
    )

    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    close_client = False
    if client is None:
        client = httpx.Client(timeout=20.0)
        close_client = True

    try:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            logger.warning(
                "Gemini API returned status %s for query rewrite; using fallback",
                resp.status_code,
            )
            return heuristic_rewrite_query(question)

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        search_query = parsed.get("search_query", "").strip() or question
        alt_queries = [
            q.strip()
            for q in parsed.get("alternative_queries", [])
            if isinstance(q, str) and q.strip()
        ]
        intent = parsed.get("intent", "informational")
        explanation = parsed.get("explanation", "Rewritten with Gemini.")

        return QueryRewriteResult(
            original_query=question,
            search_query=search_query,
            alternative_queries=alt_queries,
            intent=intent,
            explanation=explanation,
        )
    except Exception as exc:
        logger.warning("Exception during LLM query rewrite (%s); falling back", exc)
        return heuristic_rewrite_query(question)
    finally:
        if close_client:
            client.close()


# -----------------------------------------------------------------------------
# Publication Date Extraction (Task 3.20)
# -----------------------------------------------------------------------------


def extract_publication_date(html: str = "", snippet: str = "") -> str | None:
    """Extract publication or last-modified date from HTML or search snippet.

    Checks:
    1. Schema.org JSON-LD scripts (datePublished, dateModified).
    2. HTML meta tags (article:published_time, og:published_time, pubdate, date).
    3. <time datetime="..."> elements.
    4. Text date patterns in snippet (e.g., 'Oct 7, 2024' or '2024-10-07').
    """
    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")

            # 1. Check Schema.org JSON-LD
            for script in soup.find_all("script", type="application/ld+json"):
                if script.string:
                    try:
                        data = json.loads(script.string)
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            if isinstance(item, dict):
                                date_val = (
                                    item.get("datePublished")
                                    or item.get("dateModified")
                                    or item.get("uploadDate")
                                )
                                if date_val and isinstance(date_val, str):
                                    return date_val.split("T")[0]
                    except Exception:
                        pass

            # 2. Check Meta tags
            # 2. Check HTML meta tags
            target_attrs = {
                ("property", "article:published_time"),
                ("property", "og:published_time"),
                ("name", "pubdate"),
                ("name", "publishdate"),
                ("name", "date"),
                ("name", "dc.date"),
                ("name", "dc.date.issued"),
                ("name", "article.published"),
                ("itemprop", "datePublished"),
            }
            for meta_tag in soup.find_all("meta"):
                for key, expected_val in target_attrs:
                    raw_val = meta_tag.get(key)
                    if (
                        isinstance(raw_val, str)
                        and raw_val.strip().lower() == expected_val.lower()
                    ):
                        content_val = meta_tag.get("content")
                        if isinstance(content_val, str) and content_val.strip():
                            return content_val.strip().split("T")[0]

            # 3. Check <time> tag
            time_tag = soup.find("time")
            if time_tag is not None:
                dt_val = time_tag.get("datetime")
                if isinstance(dt_val, str) and dt_val.strip():
                    return dt_val.strip().split("T")[0]
                text_val = time_tag.get_text(strip=True)
                if text_val and len(text_val) <= 30:
                    return text_val

        except Exception:
            pass

    # 4. Fallback to snippet timestamp pattern
    if snippet:
        # Match 'Oct 7, 2024' or 'October 7, 2024'
        match = re.search(
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            snippet,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

        # Match '2024-10-07'
        match_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", snippet)
        if match_iso:
            return match_iso.group(1)

        # Match relative '2 days ago', '3 weeks ago'
        match_rel = re.search(
            r"\b(\d{1,2}\s+(?:hours?|days?|weeks?|months?)\s+ago)\b",
            snippet,
            re.IGNORECASE,
        )
        if match_rel:
            return match_rel.group(1)

    return None


# -----------------------------------------------------------------------------
# Search Providers
# -----------------------------------------------------------------------------


class WebSearchProvider(ABC):
    """Abstract interface for web search retrieval."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute web search and return top search results."""
        ...


class DuckDuckGoSearchProvider(WebSearchProvider):
    """Zero-credential web search provider querying DuckDuckGo HTML Lite."""

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        *,
        timeout: float = 12.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self._external_client = client

    def _clean_url(self, raw_url: str) -> str:
        """Extract clean destination URL from DuckDuckGo redirect link."""
        if not raw_url:
            return ""
        if "duckduckgo.com/l/?" in raw_url:
            parsed = urllib.parse.urlparse(raw_url)
            params = urllib.parse.parse_qs(parsed.query)
            dest = params.get("uddg", [None])[0]
            if dest:
                return dest
        if raw_url.startswith("//"):
            return "https:" + raw_url
        return raw_url

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search DuckDuckGo HTML and parse result snippets."""
        close_client = False
        client = self._external_client
        if client is None:
            client = httpx.Client(timeout=self.timeout, follow_redirects=True)
            close_client = True

        try:
            resp = client.post(
                self.SEARCH_URL,
                data={"q": query},
                headers=self.DEFAULT_HEADERS,
            )
            if resp.status_code != 200:
                logger.error("DuckDuckGo search error: HTTP %s", resp.status_code)
                return []

            return self.parse_html_results(resp.text, num_results=num_results)
        except Exception as exc:
            logger.error("DuckDuckGo request failed: %s", exc)
            return []
        finally:
            if close_client:
                client.close()

    def parse_html_results(self, html: str, num_results: int = 5) -> list[SearchResult]:
        """Parse DuckDuckGo HTML output into SearchResult objects."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []

        result_bodies = soup.select(".result__body")
        rank = 1
        for body in result_bodies:
            if rank > num_results:
                break

            title_elem = body.select_one(".result__title")
            snippet_elem = body.select_one(".result__snippet")
            if not title_elem:
                continue

            title_text = title_elem.get_text(strip=True)
            snippet_text = snippet_elem.get_text(strip=True) if snippet_elem else ""

            # Extract destination link
            raw_href = ""
            a_tag = body.select_one("a.result__url") or body.select_one(
                "a.result__snippet"
            )
            if not a_tag:
                a_tag = title_elem.find("a")
            if a_tag and a_tag.has_attr("href"):
                raw_href = str(a_tag["href"])

            clean_link = self._clean_url(raw_href)
            if not clean_link or clean_link.startswith("javascript:"):
                continue

            # Exclude ads if present
            if "Viewing ads is privacy protected" in title_text:
                continue

            pub_date = extract_publication_date("", snippet=snippet_text)

            results.append(
                SearchResult(
                    title=title_text,
                    url=clean_link,
                    snippet=snippet_text,
                    rank=rank,
                    published_date=pub_date,
                )
            )
            rank += 1

        return results


class MockSearchProvider(WebSearchProvider):
    """Mock search provider for deterministic tests and offline development."""

    def __init__(
        self, canned_results: dict[str, list[SearchResult]] | None = None
    ) -> None:
        self.canned_results = canned_results or {}

    def add_canned_results(self, query: str, results: list[SearchResult]) -> None:
        """Register canned results for a specific query."""
        self.canned_results[query.lower().strip()] = results

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Return canned or synthesized mock search results."""
        key = query.lower().strip()
        if key in self.canned_results:
            return self.canned_results[key][:num_results]

        # Partial matching check
        for canned_key, results in self.canned_results.items():
            if canned_key in key or key in canned_key:
                return results[:num_results]

        # Synthesize fallback mock results based on query keywords
        return [
            SearchResult(
                title=f"Documentation & Reference for {query.title()}",
                url=f"https://example.org/docs/{urllib.parse.quote(query.lower())}",
                snippet=(
                    f"Official reference, guides, and comprehensive information "
                    f"regarding {query}. Includes architecture, specs, and usage."
                ),
                rank=1,
                published_date="2024-10-07",
            ),
            SearchResult(
                title=f"Latest Updates & Community News: {query.title()}",
                url=f"https://example.org/news/{urllib.parse.quote(query.lower())}",
                snippet=(
                    f"Recent announcements, breaking releases, and performance "
                    f"analysis discussing {query}."
                ),
                rank=2,
                published_date="2026-01-15",
            ),
        ][:num_results]


# -----------------------------------------------------------------------------
# Resilient Page Content Extraction (Task 3.18 & 3.20)
# -----------------------------------------------------------------------------


class PageContentExtractor:
    """Resilient fetcher and text extractor for search results."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    PAYWALL_INDICATORS = [
        "subscribe to continue reading",
        "subscriber-exclusive content",
        "you've reached your free article limit",
        "you have reached your limit of free articles",
        "create a free account to continue",
        "sign in to read the full story",
        "join now to read",
        "exclusive to subscribers",
        "this article is for subscribers only",
        "membership required to read",
        "unlock this story with a subscription",
    ]

    BOT_CHALLENGE_INDICATORS = [
        "cloudflare ray id",
        "verify you are human",
        "attention required! | cloudflare",
        "please complete the security check",
        "enable javascript and cookies to continue",
        "checking your browser before accessing",
        "ddos protection by cloudflare",
        "just a moment...",
        "access denied | www.",
        "shieldsquare captcha",
        "distil networks",
    ]

    def __init__(
        self,
        *,
        timeout: float = 6.0,
        max_bytes: int = 1_048_576,  # 1 MB
        max_chars: int = 12_000,
        client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self._external_client = client

    def extract_html_text(
        self, html: str, url: str = ""
    ) -> tuple[str | None, FetchStatus, str | None, str | None]:
        """Extract clean body text and publication date from HTML."""
        lower_html = html.lower()

        # 1. Check for bot / CAPTCHA challenge pages
        for indicator in self.BOT_CHALLENGE_INDICATORS:
            if indicator in lower_html:
                return (
                    None,
                    FetchStatus.JUNK_PAGE,
                    f"Bot challenge or CAPTCHA detected ('{indicator}').",
                    None,
                )

        # 2. Check for paywall indicators
        for indicator in self.PAYWALL_INDICATORS:
            if indicator in lower_html:
                return (
                    None,
                    FetchStatus.PAYWALL,
                    f"Paywall or subscription barrier detected ('{indicator}').",
                    None,
                )

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            return None, FetchStatus.JUNK_PAGE, f"HTML parse error: {exc}", None

        # Extract publication date from metadata
        pub_date = extract_publication_date(html)

        # 3. Strip boilerplate tags
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "header",
                "footer",
                "nav",
                "aside",
                "form",
            ]
        ):
            tag.decompose()

        boilerplate_pattern = re.compile(
            r"(cookie|banner|modal|popup|sidebar|advertisement|newsletter)",
            re.IGNORECASE,
        )
        for tag in soup.find_all(class_=boilerplate_pattern):
            tag.decompose()

        container = (
            soup.find("article")
            or soup.find("main")
            or soup.select_one('[role="main"]')
            or soup.body
            or soup
        )

        text = container.get_text(separator=" ", strip=True)
        clean_text = re.sub(r"\s+", " ", text).strip()

        # 4. Reject junk or nearly empty pages (< 120 characters)
        if len(clean_text) < 120:
            return (
                clean_text or None,
                FetchStatus.JUNK_PAGE,
                (
                    f"Extracted content too short ({len(clean_text)} chars); "
                    "page likely empty or JS-only."
                ),
                pub_date,
            )

        # 5. Truncate to maximum character budget at word boundary
        if len(clean_text) > self.max_chars:
            clean_text = clean_text[: self.max_chars].rsplit(" ", 1)[0] + "..."

        return clean_text, FetchStatus.SUCCESS, None, pub_date

    def fetch_and_extract(
        self, url: str, snippet: str = ""
    ) -> tuple[str | None, FetchStatus, str | None, str | None]:
        """Fetch URL with timeout/size guards and extract text and publication date."""
        close_client = False
        client = self._external_client
        if client is None:
            client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.DEFAULT_USER_AGENT},
            )
            close_client = True

        try:
            with client.stream("GET", url) as resp:
                status = resp.status_code

                if status in (401, 403):
                    return (
                        None,
                        FetchStatus.PAYWALL,
                        f"HTTP {status} Forbidden/Unauthorized (Access Gated)",
                        None,
                    )
                if status in (404, 410):
                    return None, FetchStatus.DEAD_LINK, f"HTTP {status} Not Found", None
                if status >= 500:
                    return (
                        None,
                        FetchStatus.DEAD_LINK,
                        f"HTTP {status} Server Error",
                        None,
                    )
                if status >= 400:
                    return (
                        None,
                        FetchStatus.FETCH_ERROR,
                        f"HTTP {status} Client Error",
                        None,
                    )

                content_type = resp.headers.get("Content-Type", "").lower()
                allowed_types = ("text/html", "text/plain", "application/xhtml")
                if not any(t in content_type for t in allowed_types):
                    return (
                        None,
                        FetchStatus.JUNK_PAGE,
                        (
                            f"Unsupported Content-Type: '{content_type}' "
                            "(non-text/binary payload)"
                        ),
                        None,
                    )

                content_bytes = bytearray()
                for chunk in resp.iter_bytes(chunk_size=8192):
                    content_bytes.extend(chunk)
                    if len(content_bytes) > self.max_bytes:
                        break

            encoding = resp.encoding or "utf-8"
            html_text = content_bytes.decode(encoding, errors="replace")
            text, st, err, pub_date = self.extract_html_text(html_text, url=url)
            if not pub_date and snippet:
                pub_date = extract_publication_date("", snippet=snippet)
            return text, st, err, pub_date

        except (httpx.TimeoutException, TimeoutError):
            return (
                None,
                FetchStatus.TIMEOUT,
                f"Timed out after {self.timeout}s",
                None,
            )
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            return (
                None,
                FetchStatus.DEAD_LINK,
                f"Connection/DNS failure: {exc}",
                None,
            )
        except Exception as exc:
            return None, FetchStatus.FETCH_ERROR, f"Fetch error: {exc}", None
        finally:
            if close_client:
                client.close()

    def enrich_results(
        self,
        results: list[SearchResult],
        *,
        fetch_pages: bool = True,
    ) -> list[SearchResult]:
        """Enrich search results with full page content and publication dates."""
        for res in results:
            if fetch_pages:
                content, status, err, pub_date = self.fetch_and_extract(
                    res.url, snippet=res.snippet
                )
                res.page_content = content
                res.fetch_status = status
                res.fetch_error = err
                if pub_date:
                    res.published_date = pub_date
            # Fallback to snippet timestamp if still not set
            if not res.published_date:
                res.published_date = extract_publication_date("", snippet=res.snippet)
        return results


# -----------------------------------------------------------------------------
# Answer Synthesis with URL and Date Citations (Task 3.20)
# -----------------------------------------------------------------------------

SYNTHESIS_PROMPT_TEMPLATE = """You are an accurate web search assistant.
Answer the user's question using ONLY the retrieved web search results,
extracted page contents, and publication dates below.

Retrieved Web Search Results:
{context}

User Question: {question}

Instructions:
1. Answer the question thoroughly, factually, and concisely using the provided
   page contents, snippets, and publication dates.
2. Every major claim or fact MUST cite its source using inline Markdown links
   in the format: [Source Name (Publication Date)](URL) or [Source Name](URL).
   Always include the publication date when available.
3. Do NOT make up information, URLs, or publication dates not present in the context.
4. If the retrieved sources do not contain enough information to fully answer,
   clearly state what is known and what cannot be answered from the sources.
"""


def synthesize_web_answer(
    question: str,
    search_results: list[SearchResult],
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, list[str], list[SourceCitation]]:
    """Synthesize an answer grounded in web search snippets with URL and date citations.

    Returns:
        (answer_text, list_of_cited_urls, list_of_source_citations)
    """
    if not search_results:
        return (
            "I could not find any relevant web search results to answer "
            f"your question: '{question}'.",
            [],
            [],
        )

    # Format context blocks with publication dates
    context_lines: list[str] = []
    available_urls: list[str] = []
    url_to_source: dict[str, SearchResult] = {r.url: r for r in search_results}

    for r in search_results:
        available_urls.append(r.url)
        content_body = r.effective_content
        date_str = r.published_date or "Date Unknown"
        if r.fetch_status == FetchStatus.SUCCESS and r.page_content:
            status_tag = f"Full Page Content | Published: {date_str}"
        else:
            status_tag = (
                f"Snippet Fallback ({r.fetch_status.value}) | Published: {date_str}"
            )

        context_lines.append(
            f"[{r.rank}] Title: {r.title}\n"
            f"    URL: {r.url}\n"
            f"    Publication Date: {date_str}\n"
            f"    Status: {status_tag}\n"
            f"    Content: {content_body}"
        )
    context_str = "\n\n".join(context_lines)

    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        summary_lines = [f"Based on retrieved web results for '{question}':\n"]
        for r in search_results:
            preview = r.effective_content[:250]
            if len(r.effective_content) > 250:
                preview += "..."
            date_info = (
                f" (Published: {r.published_date})"
                if r.published_date
                else " (Date: Unknown)"
            )
            summary_lines.append(
                f"- **[{r.title}]({r.url})**{date_info} [{r.fetch_status.value}]: "
                f"{preview}"
            )

        source_citations = [
            SourceCitation(
                title=r.title,
                url=r.url,
                published_date=r.published_date,
                domain=urllib.parse.urlparse(r.url).netloc,
            )
            for r in search_results
        ]
        return "\n".join(summary_lines), available_urls, source_citations

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={active_key}"
    )

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(context=context_str, question=question)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
    }

    close_client = False
    if client is None:
        client = httpx.Client(timeout=30.0)
        close_client = True

    try:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            logger.warning(
                "Gemini synthesis returned HTTP %s; falling back to snippets",
                resp.status_code,
            )
            fallback_text = (
                f"Retrieved {len(search_results)} search results for "
                f"'{question}'. Top source: [{search_results[0].title}]"
                f"({search_results[0].url})."
            )
            top_r = search_results[0]
            top_cit = SourceCitation(
                title=top_r.title,
                url=top_r.url,
                published_date=top_r.published_date,
                domain=urllib.parse.urlparse(top_r.url).netloc,
            )
            return fallback_text, [top_r.url], [top_cit]

        data = resp.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Extract cited URLs from markdown links
        cited_urls: list[str] = []
        for match in re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", answer):
            if match in available_urls and match not in cited_urls:
                cited_urls.append(match)

        if not cited_urls:
            cited_urls = available_urls

        # Build structured citations with URLs and publication dates
        source_citations: list[SourceCitation] = []
        for u in cited_urls:
            res_item = url_to_source.get(u)
            title = res_item.title if res_item else u
            pub_date = res_item.published_date if res_item else None
            domain = urllib.parse.urlparse(u).netloc
            source_citations.append(
                SourceCitation(
                    title=title,
                    url=u,
                    published_date=pub_date,
                    domain=domain,
                )
            )

        return answer, cited_urls, source_citations
    except Exception as exc:
        logger.error("Failed to synthesize answer via LLM: %s", exc)
        top_r = search_results[0]
        top_cit = SourceCitation(
            title=top_r.title,
            url=top_r.url,
            published_date=top_r.published_date,
            domain=urllib.parse.urlparse(top_r.url).netloc,
        )
        return (
            f"Error generating response: {exc}. Top result: "
            f"[{top_r.title}]({top_r.url})",
            [top_r.url],
            [top_cit],
        )
    finally:
        if close_client:
            client.close()


# -----------------------------------------------------------------------------
# End-to-End Pipeline
# -----------------------------------------------------------------------------


class WebSearchRAG:
    """Orchestrator for Web Search Retrieval-Augmented Generation."""

    def __init__(
        self,
        search_provider: WebSearchProvider | None = None,
        content_extractor: PageContentExtractor | None = None,
        *,
        client: httpx.Client | None = None,
        model: str | None = None,
        api_key: str | None = None,
        fetch_pages: bool = True,
    ) -> None:
        self.search_provider = search_provider or DuckDuckGoSearchProvider(
            client=client
        )
        self.content_extractor = content_extractor or PageContentExtractor(
            client=client
        )
        self.client = client
        self.model = model
        self.api_key = api_key
        self.fetch_pages = fetch_pages

    def query(
        self,
        question: str,
        num_results: int = 5,
        fetch_pages: bool | None = None,
    ) -> WebSearchRAGResponse:
        """Run complete pipeline: rewrite -> retrieve -> extract -> synthesize."""
        should_fetch = self.fetch_pages if fetch_pages is None else fetch_pages

        # 1. Rewrite user's conversational query
        rewritten = rewrite_search_query(
            question,
            client=self.client,
            model=self.model,
            api_key=self.api_key,
        )

        # 2. Retrieve web search results using rewritten query
        results = self.search_provider.search(
            rewritten.search_query, num_results=num_results
        )

        # If primary query returned 0 results and an alternative exists, try it
        if not results and rewritten.alternative_queries:
            alt_q = rewritten.alternative_queries[0]
            logger.info("Primary search yielded 0 results; trying alt: %s", alt_q)
            results = self.search_provider.search(alt_q, num_results=num_results)

        # 2.5. Fetch and extract page body content and dates (Task 3.18 & 3.20)
        if should_fetch and results:
            results = self.content_extractor.enrich_results(results, fetch_pages=True)

        # 3. Synthesize cited response
        answer, citations, source_citations = synthesize_web_answer(
            question,
            results,
            client=self.client,
            model=self.model,
            api_key=self.api_key,
        )

        return WebSearchRAGResponse(
            original_question=question,
            rewritten_query=rewritten,
            search_results=results,
            answer=answer,
            citations=citations,
            source_citations=source_citations,
        )


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI runner for Web Search RAG."""
    parser = argparse.ArgumentParser(
        description="Web Search RAG with URL & Date Citations (Task 3.20)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Question to ask the Web Search RAG pipeline.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search provider instead of live DuckDuckGo.",
    )
    parser.add_argument(
        "--num-results",
        "-n",
        type=int,
        default=5,
        help="Number of web search results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Disable full page fetching and use search snippets only.",
    )
    args = parser.parse_args()

    provider = MockSearchProvider() if args.mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(
        search_provider=provider,
        fetch_pages=not args.no_fetch,
    )

    if args.query:
        print("\n" + "=" * 60)
        print("WEB SEARCH RAG PIPELINE (TASK 3.20)")
        print("=" * 60)
        response = rag.query(args.query, num_results=args.num_results)

        print(f"\n[1] Original Question:\n    {response.original_question}")
        sq = response.rewritten_query.search_query
        print(f"\n[2] Rewritten Search Query:\n    '{sq}'")
        if response.rewritten_query.alternative_queries:
            print(f"    Alternatives: {response.rewritten_query.alternative_queries}")
        print(f"    Intent: {response.rewritten_query.intent}")

        print(f"\n[3] Retrieved Web Results ({len(response.search_results)} found):")
        for r in response.search_results:
            status_lbl = f"[{r.fetch_status.value}]"
            date_lbl = f"[Date: {r.published_date or 'Unknown'}]"
            print(f"    [{r.rank}] {status_lbl} {date_lbl} {r.title}")
            print(f"        URL: {r.url}")

        print("\n[4] Synthesized Answer:")
        print("-" * 60)
        print(response.answer)
        print("-" * 60)
        print("Cited Sources (URLs and Dates):")
        for cit in response.source_citations:
            date_info = f" (Date: {cit.published_date})" if cit.published_date else ""
            print(f" - {cit.title}{date_info}")
            print(f"   URL: {cit.url}")
        print()
    else:
        print("Interactive Web Search RAG (Type 'exit' to quit)\n")
        while True:
            try:
                user_q = input("Question> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_q or user_q.lower() in {"exit", "quit", "q"}:
                break

            resp = rag.query(user_q, num_results=args.num_results)
            print(f"\n-> Search Query: '{resp.rewritten_query.search_query}'")
            print(f"\nAnswer:\n{resp.answer}\n")
            print("Citations:")
            for cit in resp.source_citations:
                date_str = (
                    f" (Published: {cit.published_date})" if cit.published_date else ""
                )
                print(f" * {cit.title}{date_str} -> {cit.url}")
            print("-" * 50)


if __name__ == "__main__":
    main()
