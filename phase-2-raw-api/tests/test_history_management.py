"""Unit and integration tests for Task 2.4: History Management (Drop vs Summarize)."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.history_management import (
    ManagedHistory,
    estimate_tokens,
    execute_managed_turn,
)


def test_estimate_tokens() -> None:
    """Verifies that token estimation produces sensible positive approximations."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") > 0
    sentence = "The quick brown fox jumps over the lazy dog."
    assert 8 <= estimate_tokens(sentence) <= 15


def test_managed_history_drop_strategy() -> None:
    """Verifies that oldest turns are dropped when token threshold is exceeded."""
    history = ManagedHistory(max_tokens=30, strategy="drop")

    # Add messages that will exceed 30 tokens
    history.add_user_message("This is turn one with quite a few words.")
    history.add_model_message("Here is the answer for turn one with many words.")
    assert len(history.messages) == 2

    # Adding turn 2 should cause turn 1 to be evicted
    history.add_user_message("Now adding turn two message.")
    assert len(history.eviction_events) > 0
    assert any("Dropped message" in ev for ev in history.eviction_events)


def test_managed_history_summarize_strategy_mocked() -> None:
    """Verifies that turns are summarized and summary is injected into payload."""
    history = ManagedHistory(max_tokens=25, strategy="summarize")

    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "User likes blue. Assistant confirmed."}]}}
        ]
    }
    mock_client.post.return_value = mock_response

    # Add messages to cross threshold
    history.add_user_message(
        "I really love the color blue more than anything else in the world.",
        client=mock_client,
        api_key="mock_key",
    )
    history.add_model_message(
        "Blue is a wonderful and calming color choice.",
        client=mock_client,
        api_key="mock_key",
    )
    history.add_user_message(
        "What other colors might match well with it?",
        client=mock_client,
        api_key="mock_key",
    )

    assert history.summary is not None
    assert "User likes blue" in history.summary

    # Verify that payload now has injected summary context
    payload = history.build_payload()
    first_part_text = payload["contents"][0]["parts"][0]["text"]
    assert "System Context Briefing" in first_part_text
    assert "User likes blue" in first_part_text


def test_execute_managed_turn_mocked() -> None:
    """Verifies execute_managed_turn with mocked API."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Reply to test"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 5,
        },
    }
    mock_client.post.return_value = mock_response

    history = ManagedHistory(max_tokens=100, strategy="drop")
    reply, prompt_tokens, candidate_tokens = execute_managed_turn(
        history, "Hello AI", api_key="mock_key", client=mock_client
    )

    assert reply == "Reply to test"
    assert prompt_tokens == 25
    assert candidate_tokens == 5
    assert len(history.messages) == 2


def test_live_summarize_strategy_integration() -> None:
    """Live integration test: verifies summary generation and factual retention."""
    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    is_placeholder = settings.gemini_api_key == "your_gemini_api_key_here"
    if not has_key or is_placeholder:
        pytest.skip("No valid GEMINI_API_KEY available for live integration test")

    history = ManagedHistory(max_tokens=70, strategy="summarize")

    # Turn 1: Introduce a specific fact
    execute_managed_turn(history, "My favorite planet is Saturn because of rings.")

    # Turn 2: Send a longer message to trigger summarization threshold
    execute_managed_turn(
        history, "Explain gravity in two sentences using simple words."
    )

    # Turn 3: Check if model still knows favorite planet via summary
    reply3, _, _ = execute_managed_turn(history, "Which planet is my favorite?")
    assert "Saturn" in reply3
