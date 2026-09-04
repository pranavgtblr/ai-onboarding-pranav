"""Unit and integration tests for Task 2.5 Structured Output & Self-Healing Retry."""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, Field

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.structured_output import (
    HeroProfile,
    StructuredOutputError,
    build_schema_prompt,
    clean_json_text,
    format_validation_error,
    generate_structured_output,
)


class SampleUser(BaseModel):
    """Simple model for unit testing validation."""

    username: str
    age: int = Field(ge=18, le=120)
    is_admin: bool


def _make_mock_gemini_json_response(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """Helper to create a mock Gemini API JSON response dictionary."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(payload_dict)}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 30,
            "totalTokenCount": 80,
        },
    }


def test_clean_json_text() -> None:
    """Verify clean_json_text handles raw JSON and markdown code blocks."""
    raw = '{"name": "Bruce", "age": 35}'
    assert clean_json_text(raw) == raw

    fenced_json = '```json\n{"name": "Bruce", "age": 35}\n```'
    assert clean_json_text(fenced_json) == raw

    fenced_plain = '```\n{"name": "Bruce", "age": 35}\n```'
    assert clean_json_text(fenced_plain) == raw

    padded = '   \n```json\n{"name": "Bruce", "age": 35}\n``` \n  '
    assert clean_json_text(padded) == raw


def test_build_schema_prompt() -> None:
    """Verify prompt builder injects JSON schema into user instructions."""
    prompt = "Create user Alice"
    full_prompt = build_schema_prompt(prompt, SampleUser)

    assert "Create user Alice" in full_prompt
    assert "JSON Schema:" in full_prompt
    assert '"username"' in full_prompt
    assert '"age"' in full_prompt


def test_format_validation_error() -> None:
    """Verify validation error formatting extracts field names and reasons."""
    try:
        SampleUser.model_validate(
            {"username": "Bob", "age": 12, "is_admin": "not-bool"}
        )
    except Exception as exc:
        formatted = format_validation_error(exc)
        assert "age" in formatted
        assert "is_admin" in formatted


def test_generate_structured_output_mocked_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify successful validation on attempt 1 with mocked HTTP response."""
    valid_payload = {"username": "tony", "age": 45, "is_admin": True}
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = _make_mock_gemini_json_response(valid_payload)
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    result = generate_structured_output(
        prompt="Create Tony",
        model_cls=SampleUser,
        max_retries=1,
        force_bad_first_attempt=False,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.data.username == "tony"
    assert result.data.age == 45
    assert result.data.is_admin is True
    assert result.attempts == 1
    assert result.retried is False
    assert len(result.validation_errors) == 0


def test_generate_structured_output_mocked_retry_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify attempt 1 failure triggers retry and succeeds on attempt 2."""
    # Attempt 1 returns invalid age (under 18)
    bad_payload = {"username": "peter", "age": 15, "is_admin": False}
    # Attempt 2 returns valid age
    good_payload = {"username": "peter", "age": 19, "is_admin": False}

    resp1 = MagicMock(spec=httpx.Response)
    resp1.status_code = 200
    resp1.json.return_value = _make_mock_gemini_json_response(bad_payload)
    resp1.raise_for_status = MagicMock()

    resp2 = MagicMock(spec=httpx.Response)
    resp2.status_code = 200
    resp2.json.return_value = _make_mock_gemini_json_response(good_payload)
    resp2.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = [resp1, resp2]

    result = generate_structured_output(
        prompt="Create Peter",
        model_cls=SampleUser,
        max_retries=1,
        force_bad_first_attempt=False,
        api_key="mock-key",
        client=mock_client,
    )

    assert result.data.username == "peter"
    assert result.data.age == 19
    assert result.attempts == 2
    assert result.retried is True
    assert len(result.validation_errors) == 1
    assert "age" in result.validation_errors[0]
    assert mock_client.post.call_count == 2


def test_generate_structured_output_mocked_exhausted_retries() -> None:
    """Verify StructuredOutputError is raised when retry also fails validation."""
    bad_payload = {"username": "invalid", "age": 10, "is_admin": False}

    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = _make_mock_gemini_json_response(bad_payload)
    resp.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = resp

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_structured_output(
            prompt="Create invalid user",
            model_cls=SampleUser,
            max_retries=1,
            force_bad_first_attempt=False,
            api_key="mock-key",
            client=mock_client,
        )

    err = exc_info.value
    assert err.attempts == 2
    assert len(err.errors) == 2
    assert "age" in err.errors[0]


def test_force_bad_first_attempt_mocked() -> None:
    """Verify force_bad_first_attempt injects invalid output on attempt 1."""
    # Attempt 2 response with valid HeroProfile
    valid_hero = {
        "name": "Iron Man",
        "real_name": "Tony Stark",
        "role": "Tech Specialist",
        "power_level": 92,
        "abilities": ["Flight", "Repulsors"],
        "is_active": True,
        "summary": "Genius inventor in armor.",
    }

    resp1 = MagicMock(spec=httpx.Response)
    resp1.status_code = 200
    resp1.json.return_value = _make_mock_gemini_json_response(valid_hero)
    resp1.raise_for_status = MagicMock()

    resp2 = MagicMock(spec=httpx.Response)
    resp2.status_code = 200
    resp2.json.return_value = _make_mock_gemini_json_response(valid_hero)
    resp2.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = [resp1, resp2]

    result = generate_structured_output(
        prompt="Generate Iron Man",
        model_cls=HeroProfile,
        max_retries=1,
        force_bad_first_attempt=True,  # Deliberately triggers attempt 1 failure
        api_key="mock-key",
        client=mock_client,
    )

    assert result.attempts == 2
    assert result.retried is True
    assert result.data.name == "Iron Man"
    assert result.data.power_level == 92
    assert len(result.validation_errors) == 1
    # Check that the forced bad errors were caught
    assert "power_level" in result.validation_errors[0]


def test_live_structured_output_clean_integration() -> None:
    """Live API test verifying clean structured output on attempt 1."""
    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live test")

    result = generate_structured_output(
        prompt="Generate a profile for Thor Odinson",
        model_cls=HeroProfile,
        max_retries=1,
        force_bad_first_attempt=False,
    )

    assert isinstance(result.data, HeroProfile)
    assert result.attempts == 1
    assert result.retried is False
    assert 1 <= result.data.power_level <= 100
    assert len(result.data.abilities) >= 1
    assert len(result.data.summary) <= 200


def test_live_structured_output_force_bad_self_healing_integration() -> None:
    """Live API test verifying forced failure on attempt 1 heals on attempt 2."""
    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live test")

    result = generate_structured_output(
        prompt="Generate a profile for Doctor Strange (Stephen Strange)",
        model_cls=HeroProfile,
        max_retries=1,
        force_bad_first_attempt=True,  # Force bad attempt 1
    )

    assert isinstance(result.data, HeroProfile)
    assert result.attempts == 2
    assert result.retried is True
    assert len(result.validation_errors) == 1
    assert 1 <= result.data.power_level <= 100
    assert len(result.data.abilities) >= 1
    assert len(result.data.summary) <= 200
