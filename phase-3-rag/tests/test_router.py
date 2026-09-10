"""Unit and integration tests for Adaptive Query Router (Task 3.19)."""

import json
from unittest.mock import MagicMock

import httpx

from phase_3_rag.router import (
    AdaptiveRAGRouter,
    RouteTarget,
    RoutingDecision,
    classify_route_heuristic,
    classify_route_llm,
)
from phase_3_rag.web_search import MockSearchProvider, SearchResult, WebSearchRAG


def test_heuristic_routing_direct_llm_greetings() -> None:
    greetings = [
        "Hello! How are you doing today?",
        "Hi there",
        "Hey, who are you?",
        "Good morning",
    ]
    for q in greetings:
        decision = classify_route_heuristic(q)
        assert decision.route == RouteTarget.DIRECT_LLM
        assert decision.needs_retrieval is False
        assert decision.confidence >= 0.90


def test_heuristic_routing_direct_llm_coding_and_math() -> None:
    code_and_math = [
        "Write a Python function to compute fibonacci numbers.",
        "Implement a binary search tree in JavaScript.",
        "Calculate 128 * 4 + 10.",
        "Explain what a closure is in programming.",
    ]
    for q in code_and_math:
        decision = classify_route_heuristic(q)
        assert decision.route == RouteTarget.DIRECT_LLM
        assert decision.needs_retrieval is False


def test_heuristic_routing_local_corpus() -> None:
    local_queries = [
        "What is the maximum operating voltage of the ECLSS power bus?",
        "What propellant mixture does the Mars MDAS propulsion system use?",
        "What is the nominal cabin pressure limit for the habitat?",
        "Show me the telemetry limits for cryogen storage.",
    ]
    for q in local_queries:
        decision = classify_route_heuristic(q)
        assert decision.route == RouteTarget.LOCAL_CORPUS
        assert decision.needs_retrieval is True
        assert len(decision.keywords) > 0


def test_heuristic_routing_web_search() -> None:
    web_queries = [
        "What is the latest release version of Python in 2026?",
        "What is the current weather forecast for Tokyo today?",
        "What happened in breaking tech news this week?",
        "Who won yesterday's football match?",
    ]
    for q in web_queries:
        decision = classify_route_heuristic(q)
        assert decision.route == RouteTarget.WEB_SEARCH
        assert decision.needs_retrieval is True


def test_classify_route_llm_mocked() -> None:
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
                                    "route": "LOCAL_CORPUS",
                                    "confidence": 0.98,
                                    "reasoning": (
                                        "Asks about Project Odyssey ECLSS parameters."
                                    ),
                                    "needs_retrieval": True,
                                    "keywords": ["ECLSS", "cabin pressure"],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    decision = classify_route_llm(
        "What is the emergency minimum cabin pressure for Odyssey ECLSS?",
        client=mock_client,
        api_key="test-fake-key",
    )

    assert isinstance(decision, RoutingDecision)
    assert decision.route == RouteTarget.LOCAL_CORPUS
    assert decision.confidence == 0.98
    assert decision.needs_retrieval is True
    assert "ECLSS" in decision.keywords


def test_classify_route_llm_fallback_on_error() -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_client.post.return_value = mock_resp

    decision = classify_route_llm(
        "Write a Python function to calculate area of a circle",
        client=mock_client,
        api_key="test-fake-key",
    )

    # Gracefully falls back to heuristic classifier
    assert decision.route == RouteTarget.DIRECT_LLM
    assert decision.needs_retrieval is False


def test_adaptive_router_dispatch_direct_llm() -> None:
    router = AdaptiveRAGRouter(api_key="")
    resp = router.route_and_execute("Write a Python function to reverse a list.")

    assert resp.decision.route == RouteTarget.DIRECT_LLM
    assert resp.decision.needs_retrieval is False
    assert resp.retrieval_time_ms == 0.0
    assert len(resp.sources) == 0
    assert len(resp.answer) > 0


def test_adaptive_router_dispatch_local_corpus() -> None:
    router = AdaptiveRAGRouter(api_key="")
    resp = router.route_and_execute(
        "What is the ECLSS cabin atmospheric pressure nominal limit?"
    )

    assert resp.decision.route == RouteTarget.LOCAL_CORPUS
    assert resp.decision.needs_retrieval is True
    assert len(resp.sources) > 0
    assert "101.3 kPa" in resp.answer or "doc_eclss_limits" in resp.sources


def test_heuristic_routing_structured_db() -> None:
    db_queries = [
        "Show me all customers with orders over $100.",
        "What is the total revenue by product category?",
        "How many appointments are scheduled for Dr. Smith?",
        "List the top 5 highest priced products in stock.",
    ]
    for q in db_queries:
        decision = classify_route_heuristic(q)
        assert decision.route == RouteTarget.STRUCTURED_DB
        assert decision.needs_retrieval is True
        assert len(decision.keywords) > 0


def test_classify_route_llm_structured_db() -> None:
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
                                    "route": "STRUCTURED_DB",
                                    "confidence": 0.99,
                                    "reasoning": (
                                        "Query requires aggregating orders from SQL DB."
                                    ),
                                    "needs_retrieval": True,
                                    "keywords": ["orders", "revenue"],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    decision = classify_route_llm(
        "What is the total revenue from customer orders?",
        client=mock_client,
        api_key="test-fake-key",
    )

    assert isinstance(decision, RoutingDecision)
    assert decision.route == RouteTarget.STRUCTURED_DB
    assert decision.confidence == 0.99
    assert decision.needs_retrieval is True


def test_adaptive_router_dispatch_structured_db() -> None:
    router = AdaptiveRAGRouter(api_key="")
    resp = router.route_and_execute("How many customers are in the database?")

    assert resp.decision.route == RouteTarget.STRUCTURED_DB
    assert resp.source_used == RouteTarget.STRUCTURED_DB
    assert resp.source_label == "Structured Relational Database (SQL)"
    assert resp.decision.needs_retrieval is True
    assert len(resp.citations) > 0
    assert any("SQL:" in c or "Table" in c for c in resp.citations)


def test_adaptive_router_dispatch_web_search() -> None:
    mock_provider = MockSearchProvider()
    mock_provider.add_canned_results(
        "python 3.14 latest release news",
        [
            SearchResult(
                title="Python 3.14 Release Schedule",
                url="https://python.org/dev/peps/pep-0745/",
                snippet="Python 3.14 alpha release features and planned dates.",
                rank=1,
            )
        ],
    )
    mock_web_rag = WebSearchRAG(search_provider=mock_provider, fetch_pages=False)
    router = AdaptiveRAGRouter(web_search_rag=mock_web_rag, api_key="")

    resp = router.route_and_execute(
        "What are the latest news updates on the new Python release?"
    )

    assert resp.decision.route == RouteTarget.WEB_SEARCH
    assert resp.decision.needs_retrieval is True
    assert len(resp.sources) > 0
