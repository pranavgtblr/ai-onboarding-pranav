"""Tests for Task 4.3 RAG Agent Tools and Autonomous Multi-Tool Selection.

Verifies:
1. Individual RAG tools (pdf_search, site_search, db_query, web_search).
2. Security enforcement on db_query (blocking destructive SQL).
3. Autonomous agent routing across the 4 tools vs direct response using LangGraph.
"""

from __future__ import annotations

from phase_4_agents.agent_cli import build_tool_agent, run_agent_query
from phase_4_agents.rag_tools import (
    db_query,
    get_all_tools,
    get_rag_tools,
    pdf_search,
    site_search,
    web_search,
)

# -----------------------------------------------------------------------------
# Unit Tests for Individual RAG Tools
# -----------------------------------------------------------------------------


def test_pdf_search_returns_citations_and_excerpts() -> None:
    """Verify pdf_search retrieves Mars specs and formats cited document IDs."""
    result = pdf_search.invoke({"query": "ECLSS cabin pressure limits"})
    assert "PDF Search Results for 'ECLSS cabin pressure limits':" in result
    assert "[" in result and "]" in result
    assert "relevance:" in result


def test_site_search_returns_chunks() -> None:
    """Verify site_search retrieves crawled website pages with title and URL."""
    result = site_search.invoke({"query": "digital innovation capabilities"})
    assert "Website Search Results" in result
    assert "Title:" in result
    assert "URL:" in result
    assert "Content:" in result


def test_db_query_natural_language() -> None:
    """Verify db_query handles natural language requests for customers."""
    result = db_query.invoke({"query": "find customers in San Francisco"})
    assert "Database Query Results:" in result
    assert "Executed SQL:" in result
    assert "SELECT" in result
    assert "Rows Returned:" in result


def test_db_query_raw_sql() -> None:
    """Verify db_query executes valid read-only SELECT SQL statements."""
    result = db_query.invoke(
        {"query": "SELECT product_id, name, price FROM products LIMIT 3;"}
    )
    assert "Database Query Results:" in result
    expected_sql = (
        "Executed SQL: `SELECT product_id, name, price FROM products LIMIT 3;`"
    )
    assert expected_sql in result
    assert "Rows Returned:" in result


def test_db_query_blocks_destructive_sql() -> None:
    """Verify db_query rejects DROP/DELETE/INSERT/UPDATE mutations."""
    result = db_query.invoke({"query": "DROP TABLE customers;"})
    assert "Security Error: Only read-only SELECT queries are allowed." in result

    result_delete = db_query.invoke({"query": "DELETE FROM orders WHERE order_id = 1;"})
    assert "Security Error: Only read-only SELECT queries are allowed." in result_delete


def test_web_search_returns_citations() -> None:
    """Verify web_search returns formatted external web sources and URLs."""
    result = web_search.invoke({"query": "LangGraph 1.0 release"})
    assert "Web Search Results for 'LangGraph 1.0 release':" in result
    assert "http" in result
    assert "Snippet:" in result


def test_tool_collections() -> None:
    """Verify get_rag_tools and get_all_tools return correct tool sets."""
    rag = get_rag_tools()
    all_tools = get_all_tools()

    assert len(rag) == 4
    assert {t.name for t in rag} == {
        "pdf_search",
        "site_search",
        "db_query",
        "web_search",
    }
    assert len(all_tools) == 6
    assert "calculator" in {t.name for t in all_tools}
    assert "get_weather" in {t.name for t in all_tools}


# -----------------------------------------------------------------------------
# Agent Integration Tests (Autonomous Tool Selection via LangGraph create_agent)
# -----------------------------------------------------------------------------


def test_agent_chooses_pdf_search() -> None:
    """Agent autonomously calls pdf_search for Mars Odyssey PDF questions."""
    agent = build_tool_agent(provider="mock", model_name="mock-agent")
    query = "What are the ECLSS pressure limits in our Mars PDF manuals?"
    result = run_agent_query(agent, query, provider="mock", model_name="mock-agent")

    assert result.total_steps >= 3
    tool_calls = [
        tc["name"] for s in result.steps if s.tool_calls for tc in s.tool_calls
    ]
    assert "pdf_search" in tool_calls
    assert any(s.actor == "Tool Execution" for s in result.steps)
    assert result.final_answer != ""


def test_agent_chooses_site_search() -> None:
    """Agent autonomously calls site_search for company website documentation."""
    agent = build_tool_agent(provider="mock", model_name="mock-agent")
    query = "Search the company website documentation for Toobler cloud capabilities"
    result = run_agent_query(agent, query, provider="mock", model_name="mock-agent")

    tool_calls = [
        tc["name"] for s in result.steps if s.tool_calls for tc in s.tool_calls
    ]
    assert "site_search" in tool_calls


def test_agent_chooses_db_query() -> None:
    """Agent autonomously calls db_query for customer/order SQL database requests."""
    agent = build_tool_agent(provider="mock", model_name="mock-agent")
    query = "Query the database for customers in San Francisco"
    result = run_agent_query(agent, query, provider="mock", model_name="mock-agent")

    tool_calls = [
        tc["name"] for s in result.steps if s.tool_calls for tc in s.tool_calls
    ]
    assert "db_query" in tool_calls


def test_agent_chooses_web_search() -> None:
    """Agent calls web_search for real-time 2026 news and external facts."""
    agent = build_tool_agent(provider="mock", model_name="mock-agent")
    query = "Search the web for the latest 2026 Artemis updates"
    result = run_agent_query(agent, query, provider="mock", model_name="mock-agent")

    tool_calls = [
        tc["name"] for s in result.steps if s.tool_calls for tc in s.tool_calls
    ]
    assert "web_search" in tool_calls


def test_agent_answers_greetings_directly_without_tools() -> None:
    """Agent answers conversational greetings directly without invoking any tools."""
    agent = build_tool_agent(provider="mock", model_name="mock-agent")
    query = "Hello, how are you today?"
    result = run_agent_query(agent, query, provider="mock", model_name="mock-agent")

    tool_calls = [
        tc["name"] for s in result.steps if s.tool_calls for tc in s.tool_calls
    ]
    assert len(tool_calls) == 0
    assert "Hello!" in result.final_answer
