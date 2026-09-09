"""Unit and integration tests for Web Search RAG (Task 3.17)."""

import json
from unittest.mock import MagicMock

import httpx

from phase_3_rag.web_search import (
    DuckDuckGoSearchProvider,
    MockSearchProvider,
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


def test_synthesize_web_answer_offline(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")

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


def test_synthesize_web_answer_llm_mocked(monkeypatch) -> None:
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

    rag = WebSearchRAG(search_provider=provider)
    response = rag.query("redis pubsub scaling", num_results=3)

    assert response.original_question == "redis pubsub scaling"
    assert response.rewritten_query.search_query != ""
    assert len(response.search_results) >= 1
    assert "https://redis.io/topics/pubsub" in response.citations
