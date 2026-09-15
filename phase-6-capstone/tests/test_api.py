"""Tests for FastAPI API Gateway and SSE streaming in PG Recommends."""

import json

import pytest
from fastapi.testclient import TestClient

from phase_6_capstone.db import DatabaseManager
from phase_6_capstone.models import MovieRecord
from phase_6_capstone.server import create_app


@pytest.fixture
def client_setup():
    db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    sample_movies = [
        MovieRecord(
            movie_id="mov_1",
            title="Scream",
            year=1996,
            letterboxd_url="https://boxd.it/2ePSaX",
            rating=4.0,
            review_text="Clever meta slasher.",
            genres=["Horror"],
        ),
        MovieRecord(
            movie_id="mov_2",
            title="The Tragedy of Macbeth",
            year=2021,
            letterboxd_url="https://boxd.it/2ub67l",
            rating=5.0,
            review_text="Gorgeous cinematography and Denzel.",
            genres=["Drama"],
        ),
    ]
    app = create_app(db=db, initial_catalog=sample_movies)
    with TestClient(app) as client:
        yield client, db


def test_health_endpoint(client_setup):
    """Test health check returns healthy status."""
    client, _ = client_setup
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "PG Recommends" in data["app"]


def test_frontend_static_serving(client_setup):
    """Test that the root endpoint responds with 200 (HTML or API fallback)."""
    client, _ = client_setup
    response = client.get("/")
    assert response.status_code == 200
    assert "PG Recommends" in response.text


def test_movies_endpoint(client_setup):
    """Test fetching PG's reviewed movies list."""
    client, _ = client_setup
    response = client.get("/api/movies")
    assert response.status_code == 200
    data = response.json()
    assert len(data["movies"]) >= 2
    assert data["movies"][0]["title"] in ["Scream", "The Tragedy of Macbeth"]


def test_chat_stream_sse(client_setup):
    """Test POST /api/chat/stream returns valid Server-Sent Events."""
    client, _ = client_setup
    payload = {
        "tenant_id": "stream_tenant",
        "user_id": "test_user_1",
        "conversation_id": "test_conv_1",
        "message": "Recommend a meta slasher movie like Scream",
    }
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            content = line[6:].strip()
            if content:
                try:
                    events.append(json.loads(content))
                except json.JSONDecodeError:
                    pass

    # Should contain token events and completion
    token_events = [e for e in events if e.get("event") == "token"]
    assert len(token_events) > 0


def test_escalation_endpoint(client_setup):
    """Test creating an escalation ticket via direct API."""
    client, _ = client_setup
    payload = {
        "tenant_id": "stream_tenant",
        "user_id": "user_404",
        "conversation_id": "conv_99",
        "reason": "Need human advice for MIFF festival selection",
        "transcript_summary": "Looking for indie recommendations",
    }
    response = client.post("/api/escalate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "escalated"
    assert data["ticket_id"].startswith("TICK-")
