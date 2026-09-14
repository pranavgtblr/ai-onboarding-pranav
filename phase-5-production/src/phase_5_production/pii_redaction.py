"""Task 5.9: PII Audit and Log Redaction Engine.

Provides:
1. `PIIRedactor`: High-performance regex and rule-based sanitizer redacting
   emails, phone numbers, credit cards, SSNs, API keys/bearer tokens,
   connection URI passwords, and IP addresses.
2. `PIILoggingFilter`: Standard library `logging.Filter` mutating log records
   in-flight before console or file emission.
3. `PIIFormatter`: Standard library `logging.Formatter` ensuring formatted log
   output and tracebacks never leak sensitive personal identifiers.
4. `install_pii_log_sanitizer`: Attaches the PII scrubber to Python loggers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

# -----------------------------------------------------------------------------
# 1. PII Regular Expression Definitions
# -----------------------------------------------------------------------------

# RFC 5322-compatible email pattern
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Standard international & US phone numbers
PHONE_PATTERN = re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")

# Social Security Numbers (SSN: 3-2-4 digits)
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit Card (PAN: Visa, MasterCard, Amex, Discover 13-19 digits)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b"
)

# API Keys & Auth Tokens (OpenAI, Anthropic, Google, generic Bearer, JWT)
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
ANTHROPIC_KEY_PATTERN = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")
GOOGLE_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z-_]{30,}\b")
BEARER_TOKEN_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-_.]{20,}")
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+\b")

# Connection URIs containing embedded passwords (e.g. postgres://user:pass@host)
URI_PASSWORD_PATTERN = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:\s]+:)(?!\[REDACTED)([^@\s]+)(@[^\s]+)"
)

# IPv4 Address pattern
IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)


# -----------------------------------------------------------------------------
# 2. Luhn Algorithm Validator for Credit Cards
# -----------------------------------------------------------------------------


def is_valid_luhn(card_number_str: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


# -----------------------------------------------------------------------------
# 3. Core PII Redactor
# -----------------------------------------------------------------------------


class PIIRedactor:
    """Production PII sanitization and redaction engine."""

    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ssns: bool = True,
        redact_credit_cards: bool = True,
        redact_api_keys: bool = True,
        redact_passwords: bool = True,
        redact_ips: bool = True,
        validate_luhn: bool = False,
    ) -> None:
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssns = redact_ssns
        self.redact_credit_cards = redact_credit_cards
        self.redact_api_keys = redact_api_keys
        self.redact_passwords = redact_passwords
        self.redact_ips = redact_ips
        self.validate_luhn = validate_luhn

    def redact_text(self, text: str) -> str:
        """Sanitize and redact sensitive PII patterns from text."""
        if not text:
            return text

        result = text

        # 1. API Keys & Secrets (execute first to avoid number collisions)
        if self.redact_api_keys:
            result = OPENAI_KEY_PATTERN.sub("[REDACTED_API_KEY]", result)
            result = ANTHROPIC_KEY_PATTERN.sub("[REDACTED_API_KEY]", result)
            result = GOOGLE_KEY_PATTERN.sub("[REDACTED_API_KEY]", result)
            result = BEARER_TOKEN_PATTERN.sub(r"\1[REDACTED_TOKEN]", result)
            result = JWT_PATTERN.sub("[REDACTED_JWT_TOKEN]", result)

        # 2. Database URI embedded credentials
        if self.redact_passwords:
            result = URI_PASSWORD_PATTERN.sub(r"\1[REDACTED_PASSWORD]\3", result)

        # 3. Email addresses
        if self.redact_emails:
            result = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)

        # 4. Social Security Numbers (SSN)
        if self.redact_ssns:
            result = SSN_PATTERN.sub("[REDACTED_SSN]", result)

        # 5. Credit Cards (PAN)
        if self.redact_credit_cards:
            if self.validate_luhn:
                # Only replace if Luhn passes
                def _replace_cc(match: re.Match[str]) -> str:
                    candidate = match.group(0)
                    return (
                        "[REDACTED_CREDIT_CARD]"
                        if is_valid_luhn(candidate)
                        else candidate
                    )

                result = CREDIT_CARD_PATTERN.sub(_replace_cc, result)
            else:
                result = CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", result)

        # 6. Phone numbers
        if self.redact_phones:
            result = PHONE_PATTERN.sub("[REDACTED_PHONE]", result)

        # 7. IPv4 addresses
        if self.redact_ips:
            result = IPV4_PATTERN.sub("[REDACTED_IP]", result)

        return result

    def redact_dict(self, data: Any) -> Any:
        """Recursively redact PII in strings within dictionaries or lists."""
        if isinstance(data, str):
            return self.redact_text(data)
        if isinstance(data, dict):
            sanitized: dict[str, Any] = {}
            for k, v in data.items():
                # Redact sensitive key names directly if they hold raw passwords
                key_lower = str(k).lower()
                if any(
                    secret_tag in key_lower
                    for secret_tag in ("password", "secret", "auth_token", "api_key")
                ):
                    sanitized[k] = "[REDACTED_SECRET]"
                else:
                    sanitized[k] = self.redact_dict(v)
            return sanitized
        if isinstance(data, list):
            return [self.redact_dict(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.redact_dict(item) for item in data)
        return data

    def audit_text(self, text: str) -> dict[str, int]:
        """Scan text and return counts of detected PII occurrences."""
        if not text:
            return {}

        counts: dict[str, int] = {
            "emails": len(EMAIL_PATTERN.findall(text)),
            "phones": len(PHONE_PATTERN.findall(text)),
            "ssns": len(SSN_PATTERN.findall(text)),
            "credit_cards": len(CREDIT_CARD_PATTERN.findall(text)),
            "openai_keys": len(OPENAI_KEY_PATTERN.findall(text)),
            "anthropic_keys": len(ANTHROPIC_KEY_PATTERN.findall(text)),
            "google_keys": len(GOOGLE_KEY_PATTERN.findall(text)),
            "jwt_tokens": len(JWT_PATTERN.findall(text)),
            "uri_passwords": len(URI_PASSWORD_PATTERN.findall(text)),
            "ipv4_addresses": len(IPV4_PATTERN.findall(text)),
        }
        return {k: v for k, v in counts.items() if v > 0}


# Default global redactor instance
DEFAULT_REDACTOR = PIIRedactor()


def redact_pii(text: str) -> str:
    """Convenience helper to redact PII from text using the default redactor."""
    return DEFAULT_REDACTOR.redact_text(text)


# -----------------------------------------------------------------------------
# 4. Standard Library Logging Filter & Formatter
# -----------------------------------------------------------------------------


class PIILoggingFilter(logging.Filter):
    """Logging filter that scrubs sensitive PII from record messages in-place."""

    def __init__(
        self,
        name: str = "",
        redactor: PIIRedactor | None = None,
    ) -> None:
        super().__init__(name)
        self.redactor = redactor or DEFAULT_REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        """Intercept and sanitize record message and argument attributes."""
        if isinstance(record.msg, str):
            record.msg = self.redactor.redact_text(record.msg)

        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    self.redactor.redact_text(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (self.redactor.redact_text(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }

        return True


class PIIFormatter(logging.Formatter):
    """Logging formatter ensuring output messages and tracebacks are scrubbed."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.redactor = redactor or DEFAULT_REDACTOR

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and execute a final PII redaction pass."""
        formatted = super().format(record)
        return self.redactor.redact_text(formatted)


# -----------------------------------------------------------------------------
# 5. Installer Helper
# -----------------------------------------------------------------------------


def install_pii_log_sanitizer(
    logger_name: str | None = None,
    redactor: PIIRedactor | None = None,
) -> None:
    """Attach the PII filter and formatter to target logger and its handlers."""
    target_logger = logging.getLogger(logger_name)
    pii_filter = PIILoggingFilter(redactor=redactor)
    target_logger.addFilter(pii_filter)

    for handler in target_logger.handlers:
        handler.addFilter(pii_filter)
        if handler.formatter:
            orig_fmt = handler.formatter._fmt
            orig_datefmt = handler.formatter.datefmt
            handler.setFormatter(
                PIIFormatter(fmt=orig_fmt, datefmt=orig_datefmt, redactor=redactor)
            )
