"""Task 4.9: Automated Tests for Progressive Intermediate Step Streaming.

Verifies:
1. Synchronous stream_agent_steps yields discrete node events.
2. Asynchronous astream_agent_steps yields discrete node events.
3. FastAPI GET /stream streams real-time SSE events for agent queries.
4. FastAPI GET /stream backward compatibility with Phase 0 baseline (20 chunks).
5. FastAPI POST /stream endpoint handles JSON payloads.
6. Web UI endpoint GET / serves complete HTML dashboard.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from phase_4_agents.graph_agent import (
    AgentStreamEvent,
    astream_agent_steps,
    build_state_graph_agent,
    stream_agent_steps,
)
from phase_4_agents.server import app


def test_agent_stream_event_to_sse_format() -> None:
    """Verify AgentStreamEvent formats compliant Server-Sent Events."""
    event = AgentStreamEvent(
        event="tool_decision",
        node="call_model",
        step_number=2,
        data={"tools_requested": [{"name": "db_query", "args": {}}]},
    )
    sse = event.to_sse()
    assert sse.startswith("event: tool_decision\n")
    assert "data: " in sse
    assert sse.endswith("\n\n")

    # Parse JSON payload from data line
    data_line = [line for line in sse.split("\n") if line.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["event"] == "tool_decision"
    assert payload["node"] == "call_model"
    assert payload["step"] == 2


def test_stream_agent_steps_yields_intermediate_nodes() -> None:
    """Verify synchronous generator yields progress events as nodes execute."""
    agent = build_state_graph_agent(provider="mock")
    events = list(stream_agent_steps(agent, "Find customers living in Tokyo"))

    assert len(events) >= 5
    event_names = [e.event for e in events]
    assert event_names[0] == "step_start"
    assert "tool_decision" in event_names
    assert "tool_execution" in event_names
    assert "final_answer" in event_names
    assert event_names[-1] == "done"

    # Verify node names match LangGraph graph structure
    nodes = [e.node for e in events]
    assert "START" in nodes
    assert "call_model" in nodes
    assert "execute_tools" in nodes
    assert "END" in nodes


@pytest.mark.asyncio
async def test_astream_agent_steps_async_generator() -> None:
    """Verify asynchronous generator yields real-time events."""
    agent = build_state_graph_agent(provider="mock")
    events: list[AgentStreamEvent] = []

    async for event in astream_agent_steps(agent, "What is the weather in Tokyo?"):
        events.append(event)

    assert len(events) >= 4
    assert events[0].event == "step_start"
    assert any(e.event == "tool_decision" for e in events)
    assert any(e.event == "final_answer" for e in events)
    assert events[-1].event == "done"


def test_fastapi_stream_phase_0_backward_compatibility() -> None:
    """Verify GET /stream without query emits the 20 Phase 0 baseline chunks."""
    client = TestClient(app)
    response = client.get("/stream")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    content = response.text
    assert "chunk 1\n" in content
    assert "chunk 20\n" in content
    chunks = [c for c in content.split("\n") if c.strip()]
    assert len(chunks) == 20


def test_fastapi_stream_endpoint_agent_sse() -> None:
    """Verify GET /stream?query=... streams SSE events for an agent run."""
    client = TestClient(app)
    response = client.get(
        "/stream",
        params={"query": "Find customers in Tokyo", "provider": "mock"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "event: step_start" in body
    assert "event: tool_decision" in body
    assert "event: final_answer" in body
    assert "event: done" in body


def test_fastapi_post_stream_endpoint() -> None:
    """Verify POST /stream streams intermediate steps for JSON body."""
    client = TestClient(app)
    payload = {
        "query": "What are the ECLSS pressure limits in our Mars PDF manuals?",
        "provider": "mock",
        "thread_id": "test-post-stream",
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "event: step_start" in body
    assert "pdf_search" in body


def test_fastapi_ui_page_serves_html() -> None:
    """Verify GET / and GET /ui return the web UI dashboard."""
    client = TestClient(app)

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "Odyssey Agent Live Stream" in res_root.text
    assert "Task 4.9" in res_root.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert res_ui.text == res_root.text
