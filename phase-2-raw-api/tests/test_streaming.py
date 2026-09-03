"""Unit and integration tests for Task 2.2: Streaming Completion."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.streaming_completion import (
    parse_sse_line,
    stream_completion,
    stream_completion_async,
)


def test_parse_sse_line_empty_and_comments() -> None:
    """Verifies that non-data lines return None."""
    assert parse_sse_line("") is None
    assert parse_sse_line("   ") is None
    assert parse_sse_line(": keepalive comment") is None
    assert parse_sse_line("event: message") is None


def test_parse_sse_line_done_marker() -> None:
    """Verifies handling of [DONE] streaming sentinel."""
    chunk = parse_sse_line("data: [DONE]")
    assert chunk is not None
    assert chunk.is_final is True


def test_parse_sse_line_valid_chunk() -> None:
    """Verifies parsing of a standard Gemini SSE data line."""
    sample_line = (
        'data: {"candidates": [{"content": {"parts": [{"text": "Hello world"}]}}],'
        '"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}'
    )
    chunk = parse_sse_line(sample_line)
    assert chunk is not None
    assert chunk.text == "Hello world"
    assert chunk.input_tokens == 5
    assert chunk.output_tokens == 2
    assert chunk.finish_reason is None
    assert chunk.is_final is False


def test_parse_sse_line_final_chunk() -> None:
    """Verifies parsing of the concluding chunk containing finishReason."""
    sample_line = (
        'data: {"candidates": [{"content": {"parts": [{"text": "!"}]},'
        '"finishReason": "STOP"}],'
        '"usageMetadata": {"totalTokenCount": 10}}'
    )
    chunk = parse_sse_line(sample_line)
    assert chunk is not None
    assert chunk.text == "!"
    assert chunk.finish_reason == "STOP"
    assert chunk.is_final is True


def test_stream_completion_mocked_sync() -> None:
    """Verifies synchronous streaming using a mocked httpx context manager."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200

    lines = [
        'data: {"candidates": [{"content": {"parts": [{"text": "Token1 "}]}}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "Token2"}]},'
        ' "finishReason": "STOP"}]}',
    ]
    mock_response.iter_lines.return_value = iter(lines)
    mock_client.stream.return_value.__enter__.return_value = mock_response

    chunks = list(
        stream_completion("test prompt", api_key="test_key", client=mock_client)
    )

    assert len(chunks) == 2
    assert chunks[0].text == "Token1 "
    assert chunks[1].text == "Token2"
    assert chunks[1].finish_reason == "STOP"

    # Verify stream method was called with correct headers
    _, kwargs = mock_client.stream.call_args
    assert kwargs["headers"]["x-goog-api-key"] == "test_key"


@pytest.mark.asyncio
async def test_stream_completion_mocked_async() -> None:
    """Verifies asynchronous streaming generator using a mocked async client."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200

    lines = [
        'data: {"candidates": [{"content": {"parts": [{"text": "Async1 "}]}}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "Async2"}]},'
        ' "finishReason": "STOP"}]}',
    ]

    async def mock_aiter_lines():
        for line in lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    class AsyncStreamContext:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_client.stream.return_value = AsyncStreamContext()

    received_texts = []
    async for chunk in stream_completion_async(
        "test async prompt", api_key="test_key", client=mock_client
    ):
        received_texts.append(chunk.text)

    assert received_texts == ["Async1 ", "Async2"]


def test_stream_completion_http_error() -> None:
    """Verifies that non-200 responses raise RuntimeError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 403
    mock_response.read.return_value = b'{"error": "Forbidden"}'
    mock_client.stream.return_value.__enter__.return_value = mock_response

    with pytest.raises(RuntimeError, match="HTTP 403"):
        list(stream_completion("test", api_key="bad_key", client=mock_client))


def test_live_streaming_integration() -> None:
    """Live streaming integration test against actual Gemini SSE endpoint."""
    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    is_placeholder = settings.gemini_api_key == "your_gemini_api_key_here"
    if not has_key or is_placeholder:
        pytest.skip("No valid GEMINI_API_KEY available for live integration test")

    chunks = []
    for chunk in stream_completion("Count: 1, 2, 3."):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(c.text for c in chunks)
    assert len(full_text.strip()) > 0
    assert any(c.finish_reason == "STOP" for c in chunks)
