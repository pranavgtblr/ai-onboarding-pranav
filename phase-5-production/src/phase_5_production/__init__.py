"""Phase 5: Production AI Engineering.

Focuses on productionizing autonomous systems:
- Tracing (per-step tokens, cost, latency with LangSmith or equivalent)
- CI/CD regression suites for prompts
- LLM-as-a-judge evaluation & calibration
- Security against prompt injection & unsafe outputs
- PII masking and data retention compliance
- Cost control via caching and routing
- Provider fallback, idempotency, and reliability
"""

from phase_5_production.telemetry import (
    StepTelemetry,
    StepTelemetryTracer,
    configure_langsmith_tracing,
)

__all__ = [
    "StepTelemetry",
    "StepTelemetryTracer",
    "configure_langsmith_tracing",
]
