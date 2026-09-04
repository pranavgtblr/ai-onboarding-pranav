"""Unit and integration tests for Task 2.6 Hand-written Tool-Calling Loop."""

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.tool_calling import (
    execute_calculator,
    execute_weather,
    get_tool_declarations,
    run_tool_calling_loop,
)


def _make_gemini_candidate_response(
    parts: list[dict[str, Any]],
    prompt_tokens: int = 50,
    cand_tokens: int = 25,
) -> dict[str, Any]:
    """Helper to construct mock Gemini response payloads."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": parts,
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": cand_tokens,
            "totalTokenCount": prompt_tokens + cand_tokens,
        },
    }


def test_execute_calculator() -> None:
    """Verify calculator tool handles all arithmetic operations and edge cases."""
    assert execute_calculator("add", 10, 5) == {"result": 15}
    assert execute_calculator("subtract", 10, 5) == {"result": 5}
    assert execute_calculator("multiply", 4, 3.5) == {"result": 14}
    assert execute_calculator("divide", 10, 2) == {"result": 5}
    assert execute_calculator("divide", 7, 2) == {"result": 3.5}
    assert "error" in execute_calculator("divide", 10, 0)
    assert "error" in execute_calculator("power", 2, 3)


def test_execute_weather() -> None:
    """Verify weather lookup handles known cities and unknown fallback."""
    tokyo = execute_weather("Tokyo")
    assert tokyo["city"] == "Tokyo"
    assert tokyo["temperature_c"] == 18

    london = execute_weather("london ")
    assert london["city"] == "London"

    unknown = execute_weather("Atlantis")
    assert unknown["city"] == "Atlantis"
    assert "temperature_c" in unknown


def test_get_tool_declarations() -> None:
    """Verify tool declarations match Gemini API function declaration schema."""
    decls = get_tool_declarations()
    assert len(decls) == 1
    fns = decls[0]["function_declarations"]
    fn_names = [f["name"] for f in fns]
    assert "calculator" in fn_names
    assert "get_weather" in fn_names


def test_tool_calling_loop_no_tools_needed_mocked() -> None:
    """Verify loop exits in 1 iteration when model answers directly without tools."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = _make_gemini_candidate_response(
        parts=[{"text": "The capital of France is Paris."}]
    )
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    result = run_tool_calling_loop(
        prompt="What is the capital of France?",
        max_iterations=5,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.final_text == "The capital of France is Paris."
    assert result.total_iterations == 1
    assert result.hit_max_guard is False
    assert len(result.iteration_logs[0].tool_calls) == 0


def test_tool_calling_loop_single_tool_mocked() -> None:
    """Verify loop executes tool call on turn 1 and receives final answer on turn 2."""
    # Iteration 1: Model requests calculator(operation='multiply', a=6, b=7)
    resp1 = MagicMock(spec=httpx.Response)
    resp1.status_code = 200
    resp1.json.return_value = _make_gemini_candidate_response(
        parts=[
            {
                "functionCall": {
                    "name": "calculator",
                    "args": {"operation": "multiply", "a": 6, "b": 7},
                }
            }
        ]
    )
    resp1.raise_for_status = MagicMock()

    # Iteration 2: Model receives functionResponse and generates final text
    resp2 = MagicMock(spec=httpx.Response)
    resp2.status_code = 200
    resp2.json.return_value = _make_gemini_candidate_response(
        parts=[{"text": "6 multiplied by 7 is 42."}]
    )
    resp2.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = [resp1, resp2]

    result = run_tool_calling_loop(
        prompt="What is 6 * 7?",
        max_iterations=5,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.final_text == "6 multiplied by 7 is 42."
    assert result.total_iterations == 2
    assert result.hit_max_guard is False
    assert len(result.iteration_logs[0].tool_calls) == 1
    assert result.iteration_logs[0].tool_calls[0].output == {"result": 42}


def test_tool_calling_loop_max_iterations_guard_mocked() -> None:
    """Verify loop terminates and flags hit_max_guard when iterations exceed limit."""
    # Model endlessly requests calculator
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = _make_gemini_candidate_response(
        parts=[
            {
                "functionCall": {
                    "name": "calculator",
                    "args": {"operation": "add", "a": 1, "b": 1},
                }
            }
        ]
    )
    resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = resp

    result = run_tool_calling_loop(
        prompt="Loop forever",
        max_iterations=2,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.total_iterations == 2
    assert result.hit_max_guard is True
    assert mock_client.post.call_count == 2


def test_tool_calling_loop_unknown_tool_mocked() -> None:
    """Verify calling an unregistered tool returns an error dict without crashing."""
    resp1 = MagicMock(spec=httpx.Response)
    resp1.status_code = 200
    resp1.json.return_value = _make_gemini_candidate_response(
        parts=[
            {
                "functionCall": {
                    "name": "non_existent_tool",
                    "args": {"param": 123},
                }
            }
        ]
    )
    resp1.raise_for_status = MagicMock()

    resp2 = MagicMock(spec=httpx.Response)
    resp2.status_code = 200
    resp2.json.return_value = _make_gemini_candidate_response(
        parts=[{"text": "I could not use that tool."}]
    )
    resp2.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = [resp1, resp2]

    result = run_tool_calling_loop(
        prompt="Use unknown tool",
        max_iterations=3,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.total_iterations == 2
    assert "error" in result.iteration_logs[0].tool_calls[0].output


def test_live_tool_calling_integration() -> None:
    """Live API test verifying multi-tool execution and response synthesis."""
    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not configured; skipping live test")

    result = run_tool_calling_loop(
        prompt="What is 15 * 8, and what is the weather in London?",
        max_iterations=5,
    )

    assert result.hit_max_guard is False
    assert result.total_iterations >= 2
    assert len(result.final_text) > 10
    # Must contain 120 and London weather info
    assert "120" in result.final_text
    assert "London" in result.final_text or "london" in result.final_text
