"""Unit and integration tests for Task 2.3: Multi-turn Chat Management."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.multi_turn_chat import (
    ChatMessage,
    ConversationManager,
    send_chat_turn,
)


def test_conversation_manager_message_appending() -> None:
    """Verifies that messages are appended with correct roles."""
    manager = ConversationManager()
    manager.add_user_message("Hello")
    manager.add_model_message("Hi there!")

    assert len(manager.messages) == 2
    assert manager.messages[0] == ChatMessage(role="user", content="Hello")
    assert manager.messages[1] == ChatMessage(role="model", content="Hi there!")


def test_conversation_manager_payload_structure() -> None:
    """Verifies format of the multi-turn contents payload sent to API."""
    manager = ConversationManager(system_instruction="Be helpful.")
    manager.add_user_message("First question")
    manager.add_model_message("First answer")

    payload = manager.build_payload()
    assert "contents" in payload
    assert len(payload["contents"]) == 2
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "First question"
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["text"] == "First answer"
    assert "systemInstruction" in payload
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be helpful."


def test_conversation_manager_pruning() -> None:
    """Verifies sliding window pruning when message limit is exceeded."""
    manager = ConversationManager(max_history_turns=2)  # Max 4 messages (2 turns)

    for i in range(4):
        manager.add_user_message(f"User {i}")
        manager.add_model_message(f"Model {i}")

    # Total 8 messages added, should be pruned to the last 4 messages
    assert len(manager.messages) == 4
    assert manager.messages[0].content == "User 2"
    assert manager.messages[1].content == "Model 2"
    assert manager.messages[2].content == "User 3"
    assert manager.messages[3].content == "Model 3"


def test_conversation_manager_token_accumulation() -> None:
    """Verifies running token ledger updates across multiple turns."""
    manager = ConversationManager()

    stats1 = manager.record_turn(prompt_tokens=10, candidate_tokens=15)
    assert stats1.turn_index == 1
    assert stats1.turn_total_tokens == 25
    assert stats1.session_total_tokens == 25

    stats2 = manager.record_turn(prompt_tokens=30, candidate_tokens=20)
    assert stats2.turn_index == 2
    assert stats2.turn_total_tokens == 50
    assert stats2.session_total_tokens == 75

    manager.reset()
    assert manager.cumulative_tokens == 0
    assert len(manager.messages) == 0
    assert len(manager.turn_history) == 0


def test_send_chat_turn_mocked() -> None:
    """Verifies sending a turn using a mocked HTTP client."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Paris is the capital of France."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 8,
            "totalTokenCount": 28,
        },
    }
    mock_client.post.return_value = mock_response

    manager = ConversationManager()
    reply, stats = send_chat_turn(
        manager,
        "What is the capital of France?",
        api_key="mock_key",
        client=mock_client,
    )

    assert "Paris" in reply
    assert stats.turn_index == 1
    assert stats.prompt_tokens == 20
    assert stats.candidate_tokens == 8
    assert len(manager.messages) == 2  # User + Model
    assert manager.messages[1].content == "Paris is the capital of France."


def test_live_multi_turn_context_growth() -> None:
    """Integration test verifying context memory and token growth across 2 turns."""
    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    is_placeholder = settings.gemini_api_key == "your_gemini_api_key_here"
    if not has_key or is_placeholder:
        pytest.skip("No valid GEMINI_API_KEY available for live integration test")

    manager = ConversationManager()

    # Turn 1: Establish unique context
    reply1, stats1 = send_chat_turn(manager, "My secret password is BLUE-FALCON.")
    assert len(reply1.strip()) > 0
    assert stats1.turn_index == 1

    # Turn 2: Ask model to recall the secret from conversation context
    reply2, stats2 = send_chat_turn(manager, "What is my secret password?")
    assert "BLUE-FALCON" in reply2.upper()
    assert stats2.turn_index == 2

    # Verify that Turn 2 prompt tokens grew because Turn 1 was resent in full
    assert stats2.prompt_tokens > stats1.prompt_tokens
    assert stats2.session_total_tokens > stats1.session_total_tokens
