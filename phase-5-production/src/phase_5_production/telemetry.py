"""Task 5.1 & Task 5.9: Production Telemetry, Tracing, and PII Sanitization.

Re-exports core telemetry models and helpers from `phase_4_agents.telemetry`
along with PII-redacted tracers (`PIISanitizedTracer`) and logging sanitizers.
"""

from __future__ import annotations

import time
from uuid import UUID

from langchain_core.outputs import LLMResult
from phase_4_agents.telemetry import (
    MODEL_RATES,
    StepTelemetry,
    StepTelemetryTracer,
    calculate_token_cost,
    configure_langsmith_tracing,
)

from phase_5_production.pii_redaction import (
    PIIFormatter,
    PIILoggingFilter,
    PIIRedactor,
    install_pii_log_sanitizer,
    redact_pii,
)

__all__ = [
    "MODEL_RATES",
    "PIIFormatter",
    "PIILoggingFilter",
    "PIIRedactor",
    "PIISanitizedTracer",
    "StepTelemetry",
    "StepTelemetryTracer",
    "calculate_token_cost",
    "configure_langsmith_tracing",
    "install_pii_log_sanitizer",
    "main",
    "redact_pii",
]


class PIISanitizedTracer(StepTelemetryTracer):
    """Production StepTelemetryTracer with automated PII scrubbing.

    Ensures that inputs, prompt previews, tool argument strings, outputs,
    and exception traces never leak emails, phone numbers, credit cards,
    SSNs, API keys, or URI passwords into telemetry storage or logs.
    """

    def __init__(
        self,
        default_model: str = "gemini-2.5-flash",
        log_to_console: bool = True,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self.redactor = redactor or PIIRedactor()
        super().__init__(
            default_model=default_model,
            log_to_console=log_to_console,
            sanitizer=self.redactor.redact_text,
        )


def main() -> None:
    """CLI runner demonstrating production step telemetry with PII protection."""
    print("Initializing production telemetry demo with PII sanitization...")
    tracer = PIISanitizedTracer(log_to_console=False)

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration

    # Simulate a step with sensitive user information
    fake_run_id = UUID("12345678-1234-5678-1234-567812345678")
    tracer.on_chat_model_start(
        serialized={"name": "gemini-2.5-flash"},
        messages=[
            [
                HumanMessage(
                    content=(
                        "Customer query: Contact alice.smith@example.com "
                        "at +1-555-839-2001 regarding SSN 000-12-3456."
                    )
                )
            ]
        ],
        run_id=fake_run_id,
    )
    time.sleep(0.02)

    ai_msg = AIMessage(
        content="Looked up account with key sk-abcdef1234567890abcdef1234567890.",
        usage_metadata={"input_tokens": 45, "output_tokens": 15, "total_tokens": 60},
    )
    res = LLMResult(
        generations=[[ChatGeneration(message=ai_msg)]],
        llm_output={"token_usage": {"prompt_tokens": 45, "completion_tokens": 15}},
    )
    tracer.on_llm_end(res, run_id=fake_run_id)

    print(tracer.format_summary_table())
    for s in tracer.steps:
        print(f"Sanitized Input Preview : {s.input_preview}")
        print(f"Sanitized Output Preview: {s.output_preview}")


if __name__ == "__main__":
    main()
