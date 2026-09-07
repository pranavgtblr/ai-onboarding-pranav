"""Unit and integration tests for Task 2.7 Resilient Error Handling Layer."""

from unittest.mock import MagicMock

import httpx
import pytest

from phase_2_raw_api.config import get_settings
from phase_2_raw_api.error_handling import (
    BaseLLMProvider,
    GeminiProvider,
    SecondaryBackupProvider,
    calculate_backoff_delay,
    execute_resilient_completion,
)


class MockProvider(BaseLLMProvider):
    """Mock provider with controllable behavior for unit testing."""

    def __init__(
        self,
        name: str = "Mock-Primary",
        model: str = "mock-model",
        max_context_tokens: int = 100,
    ) -> None:
        self.name = name
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        client: httpx.Client,
        timeout: float,
    ) -> tuple[str, int]:
        self.call_count += 1
        return f"Mocked response for '{prompt[:20]}'", 42


def test_calculate_backoff_delay_exponential_growth() -> None:
    """Verify backoff delay doubles with each attempt."""
    d1 = calculate_backoff_delay(1, base_delay=1.0, jitter=0.0)
    d2 = calculate_backoff_delay(2, base_delay=1.0, jitter=0.0)
    d3 = calculate_backoff_delay(3, base_delay=1.0, jitter=0.0)

    assert d1 == 1.0
    assert d2 == 2.0
    assert d3 == 4.0


def test_calculate_backoff_delay_max_cap() -> None:
    """Verify backoff delay does not exceed max_delay cap."""
    delay = calculate_backoff_delay(10, base_delay=1.0, max_delay=15.0, jitter=0.0)
    assert delay == 15.0


def test_calculate_backoff_delay_retry_after() -> None:
    """Verify Retry-After header overrides calculated exponential delay."""
    delay = calculate_backoff_delay(1, base_delay=1.0, retry_after=12.5)
    assert delay == 12.5


def test_resilient_completion_clean_success() -> None:
    """Verify clean execution on first attempt without retries."""
    primary = MockProvider(name="Clean-Primary")
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="Hello world",
        primary_provider=primary,
        secondary_provider=secondary,
        max_retries=3,
    )

    assert result.provider == "Clean-Primary"
    assert result.attempts == 1
    assert result.retried is False
    assert result.failed_over is False
    assert primary.call_count == 1
    assert secondary.call_count == 0


def test_resilient_completion_429_retry_and_recovery() -> None:
    """Verify HTTP 429 rate limit triggers backoff and recovers."""
    mock_sleep = MagicMock()
    primary = MockProvider(name="Primary")
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="Test 429 recovery",
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_429_count=2,  # Fail twice, succeed on attempt 3
        max_retries=3,
        base_delay=0.1,
        sleep_fn=mock_sleep,
    )

    assert result.provider == "Primary"
    assert result.attempts == 3
    assert result.retried is True
    assert result.failed_over is False
    assert mock_sleep.call_count == 2
    assert primary.call_count == 1  # 2 simulated 429s, 1 real call


def test_resilient_completion_429_exhausted_failover() -> None:
    """Verify exhausted 429 rate limits fail over to secondary provider."""
    mock_sleep = MagicMock()
    primary = MockProvider(name="Primary")
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="Test 429 failover",
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_429_count=5,  # More than max_retries (3)
        max_retries=3,
        base_delay=0.01,
        sleep_fn=mock_sleep,
    )

    assert result.provider == "Backup"
    assert result.attempts == 4  # 3 on primary + 1 on secondary
    assert result.retried is True
    assert result.failed_over is True
    assert "rate limits exhausted" in result.failover_reason
    assert secondary.call_count == 1


def test_resilient_completion_timeout_retry_and_failover() -> None:
    """Verify timeout triggers retries and eventual failover."""
    mock_sleep = MagicMock()
    primary = MockProvider(name="Primary")
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="Test timeout failover",
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_timeout=True,
        max_retries=2,
        sleep_fn=mock_sleep,
    )

    assert result.provider == "Backup"
    assert result.attempts == 3  # 2 timeouts on primary + 1 on secondary
    assert result.failed_over is True
    assert "timed out" in result.failover_reason
    assert mock_sleep.call_count == 1


def test_resilient_completion_context_exceeded_truncation() -> None:
    """Verify context length exceeded triggers automatic prompt truncation."""
    primary = MockProvider(name="Primary", max_context_tokens=50)
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="A" * 500,
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_context_exceeded=True,
        auto_truncate_context=True,
    )

    assert result.provider == "Primary"
    assert result.attempts == 2
    assert result.retried is True
    assert result.failed_over is False


def test_resilient_completion_context_exceeded_route_to_secondary() -> None:
    """Verify context exceeded routes to secondary when truncation is disabled."""
    primary = MockProvider(name="Primary", max_context_tokens=50)
    secondary = MockProvider(name="High-Context-Backup")

    result = execute_resilient_completion(
        prompt="A" * 500,
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_context_exceeded=True,
        auto_truncate_context=False,
    )

    assert result.provider == "High-Context-Backup"
    assert result.failed_over is True
    assert "Context length exceeded" in result.failover_reason


def test_resilient_completion_outage_immediate_failover() -> None:
    """Verify 5xx server outage immediately triggers failover without retries."""
    primary = MockProvider(name="Primary")
    secondary = MockProvider(name="Backup")

    result = execute_resilient_completion(
        prompt="Test 503 outage",
        primary_provider=primary,
        secondary_provider=secondary,
        simulate_primary_failure=True,
        max_retries=3,
    )

    assert result.provider == "Backup"
    assert result.failed_over is True
    assert result.attempts == 2  # 1 on primary (aborted immediately) + 1 on secondary


def test_live_resilient_completion() -> None:
    """Live integration test against Gemini API."""
    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not configured; skipping live test")

    result = execute_resilient_completion(
        prompt="Say 'Hello Resilience!' in 2 words.",
        primary_provider=GeminiProvider(),
        secondary_provider=SecondaryBackupProvider(),
        max_retries=2,
    )

    assert result.failed_over is False
    assert result.provider == "Gemini-Primary"
    assert len(result.content) > 0
    assert result.tokens > 0
