"""Web Search RAG module for Project D.

Provides:
1. Query rewriting: converts conversational user questions into keyword-dense
   search engine queries using Gemini or rule-based heuristic fallbacks.
2. Search provider abstraction: DuckDuckGoSearchProvider (zero-key web scraper)
   and MockSearchProvider (deterministic offline/test provider).
3. Grounded answer synthesis: produces answers with explicit source URL citations.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
from abc import ABC, abstractmethod

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single retrieved web search result."""

    title: str = Field(..., description="Title of the web page.")
    url: str = Field(..., description="Full canonical destination URL.")
    snippet: str = Field(..., description="Search engine snippet/excerpt.")
    rank: int = Field(..., description="1-based rank in search results.")


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
    # Remove leading conversational filler
    patterns = [
        r"^(?:could|can)\s+you\s+(?:please\s+)?(?:tell|explain|show)\s+me\s+",
        r"^(?:i\s+(?:want|need|would\s+like)\s+to\s+know\s+)",
        r"^(?:what\s+is|what\s+are|who\s+is|where\s+is|how\s+to|why\s+is)\s+",
        r"^(?:please\s+)",
    ]
    transformed = cleaned
    for pat in patterns:
        transformed = re.sub(pat, "", transformed, flags=re.IGNORECASE)

    # Strip trailing punctuation
    transformed = transformed.rstrip("?.! ")

    # Strip common conversational qualifiers
    for fluff in [" recently", " right now", " please", " exactly", " basically"]:
        transformed = re.sub(re.escape(fluff), "", transformed, flags=re.IGNORECASE)

    search_query = transformed.strip() or cleaned

    # Generate an alternative query
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

            results.append(
                SearchResult(
                    title=title_text,
                    url=clean_link,
                    snippet=snippet_text,
                    rank=rank,
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
            ),
            SearchResult(
                title=f"Latest Updates & Community News: {query.title()}",
                url=f"https://example.org/news/{urllib.parse.quote(query.lower())}",
                snippet=(
                    f"Recent announcements, breaking releases, and performance "
                    f"analysis discussing {query}."
                ),
                rank=2,
            ),
        ][:num_results]


# -----------------------------------------------------------------------------
# Answer Synthesis with Citations
# -----------------------------------------------------------------------------

SYNTHESIS_PROMPT_TEMPLATE = """You are an accurate web search assistant.
Answer the user's question using ONLY the retrieved web search snippets below.

Retrieved Web Search Results:
{context}

User Question: {question}

Instructions:
1. Answer the question thoroughly, factually, and concisely.
2. Every major claim or fact MUST cite its source using inline Markdown links
   in the format: [Source Name](URL) or [Number](URL).
3. Do NOT make up information or URLs that are not in the retrieved snippets.
4. If the retrieved snippets do not contain enough information to fully answer,
   clearly state what is known and what cannot be answered from the sources.
"""


def synthesize_web_answer(
    question: str,
    search_results: list[SearchResult],
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, list[str]]:
    """Synthesize an answer grounded in web search snippets with URL citations.

    Returns:
        (answer_text, list_of_cited_urls)
    """
    if not search_results:
        return (
            "I could not find any relevant web search results to answer "
            f"your question: '{question}'.",
            [],
        )

    # Format context blocks
    context_lines: list[str] = []
    available_urls: list[str] = []
    for r in search_results:
        available_urls.append(r.url)
        context_lines.append(
            f"[{r.rank}] Title: {r.title}\n    URL: {r.url}\n    Snippet: {r.snippet}"
        )
    context_str = "\n\n".join(context_lines)

    settings = get_settings()
    active_key = api_key if api_key is not None else settings.gemini_api_key
    if not active_key:
        # Offline summary fallback
        summary_lines = [f"Based on retrieved web results for '{question}':\n"]
        for r in search_results:
            summary_lines.append(f"- **[{r.title}]({r.url})**: {r.snippet}")
        return "\n".join(summary_lines), available_urls

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
            return fallback_text, [search_results[0].url]

        data = resp.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Extract cited URLs from markdown links
        cited_urls: list[str] = []
        for match in re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", answer):
            if match in available_urls and match not in cited_urls:
                cited_urls.append(match)

        # If no explicit links were parsed, include all retrieved URLs
        if not cited_urls:
            cited_urls = available_urls

        return answer, cited_urls
    except Exception as exc:
        logger.error("Failed to synthesize answer via LLM: %s", exc)
        return (
            f"Error generating response: {exc}. Top result: "
            f"[{search_results[0].title}]({search_results[0].url})",
            [search_results[0].url],
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
        *,
        client: httpx.Client | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.search_provider = search_provider or DuckDuckGoSearchProvider(
            client=client
        )
        self.client = client
        self.model = model
        self.api_key = api_key

    def query(self, question: str, num_results: int = 5) -> WebSearchRAGResponse:
        """Run complete pipeline: rewrite -> retrieve -> synthesize."""
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

        # 3. Synthesize cited response
        answer, citations = synthesize_web_answer(
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
        )


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI runner for Web Search RAG."""
    parser = argparse.ArgumentParser(
        description="Web Search RAG with Query Rewriting (Task 3.17)"
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
    args = parser.parse_args()

    provider = MockSearchProvider() if args.mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(search_provider=provider)

    if args.query:
        print("\n" + "=" * 60)
        print("WEB SEARCH RAG PIPELINE")
        print("=" * 60)
        response = rag.query(args.query, num_results=args.num_results)

        print(f"\n[1] Original Question:\n    {response.original_question}")
        sq = response.rewritten_query.search_query
        print(f"\n[2] Rewritten Search Query:\n    '{sq}'")
        if response.rewritten_query.alternative_queries:
            print(f"    Alternatives: {response.rewritten_query.alternative_queries}")
        print(f"    Intent: {response.rewritten_query.intent}")
        print(f"    Explanation: {response.rewritten_query.explanation}")

        print(f"\n[3] Retrieved Web Results ({len(response.search_results)} found):")
        for r in response.search_results:
            print(f"    [{r.rank}] {r.title}")
            print(f"        URL: {r.url}")
            print(f"        Snippet: {r.snippet[:120]}...")

        print("\n[4] Synthesized Answer with Citations:")
        print("-" * 60)
        print(response.answer)
        print("-" * 60)
        print("Cited Sources:")
        for url in response.citations:
            print(f" - {url}")
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
            print(f"-> Hits Retrieved: {len(resp.search_results)}")
            print(f"\nAnswer:\n{resp.answer}\n")


if __name__ == "__main__":
    main()
