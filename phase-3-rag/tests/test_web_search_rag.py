"""Unit and integration tests for Web Search RAG (Task 3.17 & 3.18)."""

import json
from unittest.mock import MagicMock

import httpx

from phase_3_rag.web_search import (
    DuckDuckGoSearchProvider,
    FetchStatus,
    MockSearchProvider,
    PageContentExtractor,
    QueryRewriteResult,
    SearchResult,
    WebSearchRAG,
    heuristic_rewrite_query,
    rewrite_search_query,
    synthesize_web_answer,
)


def test_heuristic_rewrite_query_basic() -> None:
    question = "Can you tell me what is Python GIL right now?"
    result = heuristic_rewrite_query(question)

    assert isinstance(result, QueryRewriteResult)
    assert result.original_query == question
    # Conversational prefix and filler removed
    assert "Can you tell me" not in result.search_query
    assert "right now" not in result.search_query
    assert "Python GIL" in result.search_query
    assert len(result.alternative_queries) > 0
    assert result.intent == "informational"


def test_heuristic_rewrite_query_clean_fallback() -> None:
    question = "FastAPI async endpoints"
    result = heuristic_rewrite_query(question)
    assert result.search_query == "FastAPI async endpoints"


def test_rewrite_search_query_llm_mocked(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "search_query": "Python 3.13 free threaded GIL",
                                    "alternative_queries": [
                                        "PEP 703 Python free threading",
                                        "Python remove GIL status",
                                    ],
                                    "intent": "informational",
                                    "explanation": (
                                        "Extracted core keywords and PEP reference."
                                    ),
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    res = rewrite_search_query(
        "Can you explain if the GIL is finally removed in Python 3.13?",
        client=mock_client,
        api_key="test-fake-key",
    )

    assert res.search_query == "Python 3.13 free threaded GIL"
    assert len(res.alternative_queries) == 2
    assert "PEP 703" in res.alternative_queries[0]
    assert res.intent == "informational"


def test_rewrite_search_query_llm_fallback_on_error(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_client.post.return_value = mock_resp

    res = rewrite_search_query(
        "Can you tell me how to deploy Docker containers?",
        client=mock_client,
        api_key="test-fake-key",
    )
    # Falls back to heuristic rewriter
    assert "deploy Docker containers" in res.search_query
    assert res.intent == "informational"


def test_duckduckgo_clean_url() -> None:
    provider = DuckDuckGoSearchProvider()

    # Case 1: Standard DDG redirect URL
    redirect_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&rut=123"
    assert provider._clean_url(redirect_url) == "https://www.python.org/"

    # Case 2: Protocol-relative direct URL
    proto_url = "//en.wikipedia.org/wiki/Python"
    assert provider._clean_url(proto_url) == "https://en.wikipedia.org/wiki/Python"

    # Case 3: Clean standard URL
    clean_url = "https://docs.python.org/3/"
    assert provider._clean_url(clean_url) == "https://docs.python.org/3/"


def test_duckduckgo_parse_html() -> None:
    sample_html = """
    <html>
      <body>
        <div class="result__body">
          <h2 class="result__title">
            <a class="result__a"
               href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffastapi.tiangolo.com">
              FastAPI
            </a>
          </h2>
          <a class="result__snippet"
             href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffastapi.tiangolo.com">
            FastAPI framework, high performance, easy to learn, fast to code.
          </a>
        </div>
        <div class="result__body">
          <h2 class="result__title">
            <a class="result__a" href="https://github.com/tiangolo/fastapi">
              GitHub - tiangolo/fastapi
            </a>
          </h2>
          <div class="result__snippet">
            Source repository and issue tracker for FastAPI web framework.
          </div>
        </div>
      </body>
    </html>
    """
    provider = DuckDuckGoSearchProvider()
    results = provider.parse_html_results(sample_html, num_results=5)

    assert len(results) == 2
    assert results[0].title == "FastAPI"
    assert results[0].url == "https://fastapi.tiangolo.com"
    assert "high performance" in results[0].snippet
    assert results[0].rank == 1

    assert results[1].title == "GitHub - tiangolo/fastapi"
    assert results[1].url == "https://github.com/tiangolo/fastapi"
    assert results[1].rank == 2


def test_mock_search_provider_canned() -> None:
    provider = MockSearchProvider()
    canned = [
        SearchResult(
            title="AsyncIO Guide",
            url="https://docs.python.org/3/library/asyncio.html",
            snippet="Asynchronous I/O, event loop, and coroutines.",
            rank=1,
        )
    ]
    provider.add_canned_results("python asyncio", canned)

    hits = provider.search("python asyncio", num_results=5)
    assert len(hits) == 1
    assert hits[0].title == "AsyncIO Guide"

    # Test fallback query generation
    fallback_hits = provider.search("kubernetes cluster autoscaler", num_results=2)
    assert len(fallback_hits) == 2
    assert "kubernetes cluster autoscaler" in fallback_hits[0].title.lower()


def test_synthesize_web_answer_offline() -> None:
    results = [
        SearchResult(
            title="FastAPI Docs",
            url="https://fastapi.tiangolo.com",
            snippet="High performance Python web framework.",
            rank=1,
        )
    ]
    answer, citations = synthesize_web_answer("What is FastAPI?", results, api_key="")

    assert "FastAPI Docs" in answer
    assert "https://fastapi.tiangolo.com" in answer
    assert citations == ["https://fastapi.tiangolo.com"]


def test_synthesize_web_answer_llm_mocked() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "FastAPI is a modern Python web framework created by "
                                "[Sebastián Ramírez](https://fastapi.tiangolo.com). "
                                "It is built on Starlette and Pydantic."
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    results = [
        SearchResult(
            title="FastAPI",
            url="https://fastapi.tiangolo.com",
            snippet="High performance API framework.",
            rank=1,
        )
    ]
    answer, citations = synthesize_web_answer(
        "Who created FastAPI?", results, client=mock_client, api_key="test-fake-key"
    )

    assert "Sebastián Ramírez" in answer
    assert citations == ["https://fastapi.tiangolo.com"]


# -----------------------------------------------------------------------------
# Task 3.18 Resilient Page Content Extraction Tests
# -----------------------------------------------------------------------------


def test_extract_html_text_clean_article() -> None:
    html = """
    <html>
      <head><title>Python Multiprocessing</title></head>
      <body>
        <header><nav><a href="/">Home</a></nav></header>
        <div class="cookie-banner">Accept all cookies</div>
        <main>
          <article>
            <h1>Python Multiprocessing Architecture and Process Isolation</h1>
            <p>The multiprocessing package offers concurrency, effectively
            side-stepping the GIL by using sub-processes. This allows developers
            to fully leverage multiple CPU processors on modern computers.</p>
          </article>
        </main>
        <footer>Copyright 2026. All rights reserved.</footer>
      </body>
    </html>
    """
    extractor = PageContentExtractor()
    text, status, err = extractor.extract_html_text(html)

    assert status == FetchStatus.SUCCESS
    assert err is None
    assert text is not None
    assert "side-stepping the GIL" in text
    # Verify boilerplate was removed
    assert "Accept all cookies" not in text
    assert "Copyright 2026" not in text


def test_extract_html_text_detects_paywall() -> None:
    html = """
    <html>
      <body>
        <h1>Breaking News: Chip Manufacturing Shift</h1>
        <p>Semiconductor fabrication plants have reported significant changes...</p>
        <div class="paywall-overlay">
          <h3>Subscribe to continue reading</h3>
          <p>This article is for subscribers only. Join now to read.</p>
        </div>
      </body>
    </html>
    """
    extractor = PageContentExtractor()
    text, status, err = extractor.extract_html_text(html)

    assert status == FetchStatus.PAYWALL
    assert text is None
    assert "Paywall" in str(err)


def test_extract_html_text_detects_cloudflare_challenge() -> None:
    html = """
    <html>
      <head><title>Just a moment...</title></head>
      <body>
        <h1>Attention Required! | Cloudflare</h1>
        <p>Please complete the security check to access the website.</p>
        <div id="ray-id">Cloudflare Ray ID: 89f2a0134b</div>
      </body>
    </html>
    """
    extractor = PageContentExtractor()
    text, status, err = extractor.extract_html_text(html)

    assert status == FetchStatus.JUNK_PAGE
    assert text is None
    assert "Bot challenge" in str(err)


def test_extract_html_text_rejects_micro_content() -> None:
    html = "<html><body><p>Hello world.</p></body></html>"
    extractor = PageContentExtractor()
    text, status, err = extractor.extract_html_text(html)

    # Content length < 120 chars is marked JUNK_PAGE
    assert status == FetchStatus.JUNK_PAGE
    assert "too short" in str(err)


def test_fetch_and_extract_handles_http_404() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_client.stream.return_value.__enter__.return_value = mock_resp

    extractor = PageContentExtractor(client=mock_client)
    text, status, err = extractor.fetch_and_extract("https://example.com/dead-page")

    assert status == FetchStatus.DEAD_LINK
    assert "404" in str(err)
    assert text is None


def test_fetch_and_extract_handles_http_403_paywall() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_client.stream.return_value.__enter__.return_value = mock_resp

    extractor = PageContentExtractor(client=mock_client)
    text, status, err = extractor.fetch_and_extract("https://example.com/forbidden")

    assert status == FetchStatus.PAYWALL
    assert "403" in str(err)
    assert text is None


def test_fetch_and_extract_handles_timeout() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.side_effect = httpx.TimeoutException("Read timeout")

    extractor = PageContentExtractor(client=mock_client)
    text, status, err = extractor.fetch_and_extract("https://example.com/slow-hanging")

    assert status == FetchStatus.TIMEOUT
    assert "Timed out" in str(err)
    assert text is None


def test_fetch_and_extract_handles_connect_error() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.side_effect = httpx.ConnectError("DNS failed")

    extractor = PageContentExtractor(client=mock_client)
    text, status, err = extractor.fetch_and_extract("https://nonexistent-domain.xyz")

    assert status == FetchStatus.DEAD_LINK
    assert "DNS failure" in str(err)
    assert text is None


def test_fetch_and_extract_blocks_binary_content_type() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/pdf"}
    mock_client.stream.return_value.__enter__.return_value = mock_resp

    extractor = PageContentExtractor(client=mock_client)
    text, status, err = extractor.fetch_and_extract("https://example.com/document.pdf")

    assert status == FetchStatus.JUNK_PAGE
    assert "Unsupported Content-Type" in str(err)
    assert text is None


def test_enrich_results_mixed_resilience() -> None:
    class DummyExtractor(PageContentExtractor):
        def fetch_and_extract(
            self, url: str
        ) -> tuple[str | None, FetchStatus, str | None]:
            if "valid" in url:
                return (
                    "Extracted article text with substantial technical details.",
                    FetchStatus.SUCCESS,
                    None,
                )
            if "dead" in url:
                return None, FetchStatus.DEAD_LINK, "HTTP 404 Not Found"
            if "paywall" in url:
                return None, FetchStatus.PAYWALL, "Subscription barrier"
            return None, FetchStatus.JUNK_PAGE, "Bot challenge"

    extractor = DummyExtractor()
    items = [
        SearchResult(
            title="Valid Tech Doc",
            url="https://site.org/valid",
            snippet="Short snippet 1.",
            rank=1,
        ),
        SearchResult(
            title="Dead Link",
            url="https://site.org/dead",
            snippet="Short snippet 2.",
            rank=2,
        ),
        SearchResult(
            title="Paywalled Paper",
            url="https://site.org/paywall",
            snippet="Short snippet 3.",
            rank=3,
        ),
    ]

    enriched = extractor.enrich_results(items, fetch_pages=True)

    assert enriched[0].fetch_status == FetchStatus.SUCCESS
    assert enriched[0].effective_content.startswith("Extracted article text")

    assert enriched[1].fetch_status == FetchStatus.DEAD_LINK
    assert enriched[1].effective_content == "Short snippet 2."

    assert enriched[2].fetch_status == FetchStatus.PAYWALL
    assert enriched[2].effective_content == "Short snippet 3."


def test_web_search_rag_pipeline_end_to_end() -> None:
    provider = MockSearchProvider()
    provider.add_canned_results(
        "redis pubsub scaling",
        [
            SearchResult(
                title="Scaling Redis Pub/Sub",
                url="https://redis.io/topics/pubsub",
                snippet="Redis Pub/Sub implements high throughput messaging.",
                rank=1,
            )
        ],
    )

    rag = WebSearchRAG(search_provider=provider, fetch_pages=False)
    response = rag.query("redis pubsub scaling", num_results=3)

    assert response.original_question == "redis pubsub scaling"
    assert response.rewritten_query.search_query != ""
    assert len(response.search_results) >= 1
    assert "https://redis.io/topics/pubsub" in response.citations
