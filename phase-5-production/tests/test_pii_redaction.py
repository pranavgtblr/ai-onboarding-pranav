"""Task 5.9: Automated Tests for PII Audit and Log Redaction.

Verifies:
1. Email address pattern matching and replacement.
2. International and US phone number redaction.
3. Social Security Number (SSN) redaction.
4. Credit Card (PAN) redaction with standard patterns and Luhn validation.
5. API Key, Bearer token, JWT, and URI password redaction.
6. IPv4 address redaction.
7. Recursive dictionary and list sanitization with sensitive key redaction.
8. Standard library `PIILoggingFilter` and `PIIFormatter` log interception.
9. `PIISanitizedTracer` telemetry preview and error sanitization.
10. PII auditing counter (`audit_text`).
"""

from __future__ import annotations

import io
import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from phase_5_production.pii_redaction import (
    PIIFormatter,
    PIILoggingFilter,
    PIIRedactor,
    is_valid_luhn,
    redact_pii,
)
from phase_5_production.telemetry import PIISanitizedTracer

# =============================================================================
# 1. PII Pattern & Redactor Unit Tests
# =============================================================================


def test_email_redaction() -> None:
    """Test scrubbing of various email addresses."""
    text = (
        "Contact primary user at alice.smith@example.com or support@sub.corp.io. "
        "Also test+filter@domain.org."
    )
    redacted = redact_pii(text)
    assert "alice.smith@example.com" not in redacted
    assert "support@sub.corp.io" not in redacted
    assert "test+filter@domain.org" not in redacted
    assert redacted.count("[REDACTED_EMAIL]") == 3


def test_phone_redaction() -> None:
    """Test scrubbing of international and US formatted phone numbers."""
    text = "Call me at +1-555-839-2001 or (555) 234-5678, or direct 555-888-9999."
    redacted = redact_pii(text)
    assert "555-839-2001" not in redacted
    assert "(555) 234-5678" not in redacted
    assert "555-888-9999" not in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_ssn_redaction() -> None:
    """Test scrubbing of Social Security Numbers."""
    text = "User SSN is 123-45-6789 on file."
    redacted = redact_pii(text)
    assert "123-45-6789" not in redacted
    assert redacted == "User SSN is [REDACTED_SSN] on file."


def test_credit_card_redaction_and_luhn() -> None:
    """Test credit card redaction and Luhn validator algorithm."""
    # Valid test Visa number (passes Luhn)
    valid_visa = "4532-0150-1234-5671"
    assert is_valid_luhn(valid_visa) is True

    # Invalid card number (fails Luhn)
    invalid_cc = "4532-0150-1234-5672"
    assert is_valid_luhn(invalid_cc) is False

    text = f"Charged card {valid_visa} successfully."
    redacted = redact_pii(text)
    assert valid_visa not in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted

    # Test redactor with Luhn validation enabled
    luhn_redactor = PIIRedactor(validate_luhn=True)
    mixed_text = f"Valid: {valid_visa} | Invalid: {invalid_cc}"
    luhn_redacted = luhn_redactor.redact_text(mixed_text)
    assert "[REDACTED_CREDIT_CARD]" in luhn_redacted
    assert invalid_cc in luhn_redacted  # Should not redact invalid CC when Luhn is on


def test_api_keys_and_secret_tokens() -> None:
    """Test scrubbing of OpenAI, Anthropic, Google keys, and Bearer tokens."""
    openai_key = "sk-proj-abc1234567890abcdef1234567890"
    anthropic_key = "sk-ant-live0123456789abcdef0123456789"
    google_key = "AIzaSyD-1234567890abcdefghijklmnopqrst"
    bearer_token = "Bearer test_generic_token_abcdef1234567890"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDc"

    text = (
        f"Keys: {openai_key}, {anthropic_key}, {google_key}, "
        f"{bearer_token}, and {jwt_token}"
    )
    redacted = redact_pii(text)
    assert openai_key not in redacted
    assert anthropic_key not in redacted
    assert google_key not in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "Bearer [REDACTED_TOKEN]" in redacted
    assert "[REDACTED_JWT_TOKEN]" in redacted


def test_database_uri_password_redaction() -> None:
    """Test scrubbing of embedded credentials in connection strings."""
    uri = "postgresql://db_admin:SuperSecretPass123!@db.internal:5432/customers"
    redacted = redact_pii(uri)
    assert "SuperSecretPass123!" not in redacted
    assert (
        redacted
        == "postgresql://db_admin:[REDACTED_PASSWORD]@db.internal:5432/customers"
    )


def test_ipv4_address_redaction() -> None:
    """Test scrubbing of client IP addresses."""
    text = "Client connected from 192.168.1.105 on port 8080."
    redacted = redact_pii(text)
    assert "192.168.1.105" not in redacted
    assert redacted == "Client connected from [REDACTED_IP] on port 8080."


def test_recursive_dict_and_list_redaction() -> None:
    """Test deep sanitization of nested JSON structures."""
    redactor = PIIRedactor()
    raw_payload = {
        "customer": {
            "name": "Alice",
            "email": "alice@company.com",
            "phone": "555-222-3333",
        },
        "order": {
            "card": "4111-2222-3333-4444",
            "api_key": "sk-secret12345678901234567890",
        },
        "tags": ["user@test.org", "ip: 10.0.0.1"],
        "password": "ClearTextPassword123!",
    }

    sanitized = redactor.redact_dict(raw_payload)
    assert sanitized["customer"]["email"] == "[REDACTED_EMAIL]"
    assert sanitized["customer"]["phone"] == "[REDACTED_PHONE]"
    assert sanitized["order"]["card"] == "[REDACTED_CREDIT_CARD]"
    assert sanitized["order"]["api_key"] == "[REDACTED_SECRET]"
    assert sanitized["password"] == "[REDACTED_SECRET]"
    assert sanitized["tags"][0] == "[REDACTED_EMAIL]"
    assert sanitized["tags"][1] == "ip: [REDACTED_IP]"


def test_pii_audit_counter() -> None:
    """Test detection counter accuracy across multiple PII categories."""
    redactor = PIIRedactor()
    sample = (
        "User bob@domain.com (phone 555-123-4567) registered with SSN 000-11-2222. "
        "OpenAI sk-abcdefghijklmnopqrstuvwxyz123 and IP 127.0.0.1."
    )
    counts = redactor.audit_text(sample)
    assert counts.get("emails") == 1
    assert counts.get("phones") == 1
    assert counts.get("ssns") == 1
    assert counts.get("openai_keys") == 1
    assert counts.get("ipv4_addresses") == 1


# =============================================================================
# 2. Python Logging Filter & Formatter Integration Tests
# =============================================================================


def test_logging_filter_and_formatter() -> None:
    """Verify PIILoggingFilter and PIIFormatter intercept records emitted by loggers."""
    test_logger = logging.getLogger("test_pii_audit_logger")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(PIIFormatter("[%(levelname)s] %(message)s"))
    test_logger.addHandler(handler)
    test_logger.addFilter(PIILoggingFilter())

    # Log messages with sensitive PII
    test_logger.info(
        "Found customer record: email=%s, ssn=%s", "jane@doe.com", "987-65-4321"
    )
    test_logger.error("Failed DB connect: postgresql://admin:MyDbSecret456@db:5432/app")

    output = log_stream.getvalue()
    assert "jane@doe.com" not in output
    assert "987-65-4321" not in output
    assert "MyDbSecret456" not in output
    assert "[REDACTED_EMAIL]" in output
    assert "[REDACTED_SSN]" in output
    assert "[REDACTED_PASSWORD]" in output


# =============================================================================
# 3. PIISanitizedTracer Telemetry Integration Tests
# =============================================================================


def test_pii_sanitized_tracer_redacts_previews_and_errors() -> None:
    """Verify PIISanitizedTracer scrubs prompt, output, tool, and error previews."""
    tracer = PIISanitizedTracer(log_to_console=False)

    # 1. Model start with PII in human prompt
    run_id = uuid4()
    msg = HumanMessage(
        content="Lookup account for john.doe@secure.org with SSN 111-22-3333."
    )
    tracer.on_chat_model_start(
        serialized={"name": "gemini-2.5-flash"},
        messages=[[msg]],
        run_id=run_id,
    )

    # 2. Model end with PII in response text
    ai_msg = AIMessage(
        content="Found phone 555-444-1212 and API key sk-test12345678901234567890."
    )
    res = LLMResult(
        generations=[[ChatGeneration(message=ai_msg)]],
        llm_output={"token_usage": {"prompt_tokens": 30, "completion_tokens": 20}},
    )
    tracer.on_llm_end(res, run_id=run_id)

    step = tracer.steps[0]
    assert "john.doe@secure.org" not in step.input_preview
    assert "111-22-3333" not in step.input_preview
    assert "[REDACTED_EMAIL]" in step.input_preview
    assert "[REDACTED_SSN]" in step.input_preview

    assert "555-444-1212" not in step.output_preview
    assert "sk-test12345678901234567890" not in step.output_preview
    assert "[REDACTED_PHONE]" in step.output_preview
    assert "[REDACTED_API_KEY]" in step.output_preview

    # 3. Tool error with sensitive connection string
    tool_run_id = uuid4()
    tracer.on_tool_start(
        serialized={"name": "db_query"},
        input_str="SELECT * FROM users WHERE email='alice@site.com'",
        run_id=tool_run_id,
    )
    db_error = ConnectionError(
        "Connection to postgresql://user:SecretDbPass77@localhost:5432 failed."
    )
    tracer.on_tool_error(db_error, run_id=tool_run_id)

    tool_step = tracer.steps[1]
    assert "alice@site.com" not in tool_step.input_preview
    assert "[REDACTED_EMAIL]" in tool_step.input_preview
    assert "SecretDbPass77" not in (tool_step.error_message or "")
    assert "[REDACTED_PASSWORD]" in (tool_step.error_message or "")
