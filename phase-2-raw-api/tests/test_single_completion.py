"""Unit and integration tests for Task 2.1: Single Raw HTTPX Completion."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.single_completion import (
    build_generate_content_payload,
    generate_single_completion,
    generate_single_completion_async,
    parse_gemini_response,
)

SAMPLE_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": (
                            "Inception represents recursion: a dream within a dream "
                            "where each level executes until the kick triggers."
                        )
                    }
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 16,
        "candidatesTokenCount": 24,
        "totalTokenCount": 40,
    },
    "modelVersion": "gemini-3.5-flash-lite",
}


def test_build_generate_content_payload() -> None:
    """Verifies that the raw HTTP JSON payload conforms to Gemini API schema."""
    prompt = "Hello AI"
    payload = build_generate_content_payload(prompt)
    assert "contents" in payload
    assert len(payload["contents"]) == 1
    assert payload["contents"][0]["parts"][0]["text"] == "Hello AI"


def test_parse_gemini_response() -> None:
    """Verifies content, stop reason, and token usage parsing."""
    result = parse_gemini_response(SAMPLE_GEMINI_RESPONSE)
    assert (
        result.content == "Inception represents recursion: a dream within a dream "
        "where each level executes until the kick triggers."
    )
    assert result.stop_reason == "STOP"
    assert result.input_tokens == 16
    assert result.output_tokens == 24
    assert result.total_tokens == 40
    assert result.model_version == "gemini-3.5-flash-lite"


def test_parse_gemini_response_empty_candidates() -> None:
    """Verifies ValueError when candidates list is empty."""
    with pytest.raises(ValueError, match="No candidates returned"):
        parse_gemini_response({"candidates": []})


def test_generate_single_completion_missing_key() -> None:
    """Verifies ValueError if API key is empty."""
    with pytest.raises(ValueError, match="Gemini API key is required"):
        generate_single_completion("test", api_key="")


def test_generate_single_completion_mocked_sync() -> None:
    """Verifies synchronous raw completion using a mocked httpx.Client."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_GEMINI_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response

    result = generate_single_completion(
        prompt="Tell me a joke",
        api_key="mock_key",
        client=mock_client,
    )

    assert result.stop_reason == "STOP"
    assert result.input_tokens == 16
    assert result.output_tokens == 24
    assert "Inception" in result.content

    # Ensure correct headers were sent
    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["x-goog-api-key"] == "mock_key"
    assert kwargs["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_generate_single_completion_mocked_async() -> None:
    """Verifies asynchronous raw completion using a mocked httpx.AsyncClient."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_GEMINI_RESPONSE
    mock_response.raise_for_status.return_value = None

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client.post = mock_post

    result = await generate_single_completion_async(
        prompt="Tell me a joke",
        api_key="mock_key",
        client=mock_client,
    )

    assert result.stop_reason == "STOP"
    assert result.input_tokens == 16
    assert result.output_tokens == 24


def test_generate_single_completion_http_error() -> None:
    """Verifies that HTTP errors are wrapped in RuntimeError with status information."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "API key not valid."
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response
    )
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="HTTP 401"):
        generate_single_completion("test", api_key="bad_key", client=mock_client)


def test_live_single_completion_integration() -> None:
    """Integration test against actual Gemini REST API if valid key is set."""
    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    is_placeholder = settings.gemini_api_key == "your_gemini_api_key_here"
    if not has_key or is_placeholder:
        pytest.skip("No valid GEMINI_API_KEY available for live integration test")

    result = generate_single_completion(
        prompt="Say 'Phase 2 Test OK' and nothing else."
    )
    assert len(result.content.strip()) > 0
    assert result.stop_reason in ("STOP", "MAX_TOKENS")
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.total_tokens >= (result.input_tokens + result.output_tokens)
