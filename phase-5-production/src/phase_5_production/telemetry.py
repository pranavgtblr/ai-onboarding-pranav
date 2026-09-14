"""Task 5.1: Production Telemetry and Tracing for AI Agents.

Re-exports core telemetry models and helpers from `phase_4_agents.telemetry`
and provides production-level CLI and dashboard hooks.
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

__all__ = [
    "MODEL_RATES",
    "StepTelemetry",
    "StepTelemetryTracer",
    "calculate_token_cost",
    "configure_langsmith_tracing",
    "main",
]


def main() -> None:
    """CLI runner demonstrating production step telemetry."""
    print("Initializing production telemetry demo...")
    tracer = StepTelemetryTracer(log_to_console=False)

    # Simulate a 2-step agent run for verification
    fake_run_id = UUID("12345678-1234-5678-1234-567812345678")
    tracer.on_chat_model_start(
        serialized={"name": "gemini-2.5-flash"},
        messages=[[type("Msg", (), {"content": "Search Mars rover specs"})()]],  # type: ignore
        run_id=fake_run_id,
    )
    time.sleep(0.05)  # 50ms simulated model call
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration

    ai_msg = AIMessage(
        content="Calling pdf_search",
        usage_metadata={"input_tokens": 45, "output_tokens": 15, "total_tokens": 60},
    )
    res = LLMResult(
        generations=[[ChatGeneration(message=ai_msg)]],
        llm_output={"token_usage": {"prompt_tokens": 45, "completion_tokens": 15}},
    )
    tracer.on_llm_end(res, run_id=fake_run_id)

    tool_run_id = UUID("87654321-4321-8765-4321-876543218765")
    tracer.on_tool_start(
        serialized={"name": "pdf_search"},
        input_str="Mars rover specs",
        run_id=tool_run_id,
    )
    time.sleep(0.02)  # 20ms simulated tool call
    tracer.on_tool_end("ECLSS specs returned 3 chunks.", run_id=tool_run_id)

    print(tracer.format_summary_table())


if __name__ == "__main__":
    main()
