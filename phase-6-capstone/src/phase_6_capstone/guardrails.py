"""Production Safety Guardrails.

Features prompt injection defense, HTML escaping, and PII redaction.
"""

import html
import re

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
BEARER_TOKEN_PATTERN = re.compile(r"Bearer\s+([A-Za-z0-9-_.]+)", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)


def wrap_untrusted_review_context(content: str) -> str:
    """Quarantines untrusted review text inside an XML envelope with CDATA."""
    sanitized = content.replace("]]>", "]]&gt;")
    return f"<untrusted_context>\n<![CDATA[\n{sanitized}\n]]>\n</untrusted_context>"


def escape_web_output(content: str) -> str:
    """Contextually escapes HTML to prevent XSS in rendered web pages."""
    return html.escape(content, quote=True)


def redact_sensitive_log_message(message: str) -> str:
    """Redacts emails, bearer tokens, and phone numbers from telemetry logs."""
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", message)
    redacted = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED_TOKEN]", redacted)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return redacted
