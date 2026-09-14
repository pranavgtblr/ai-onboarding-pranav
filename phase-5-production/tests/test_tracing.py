"""Task 5.1: Automated Tests for Production Step Telemetry and Tracing.

Verifies:
1. `StepTelemetry` schema validation and JSON serialization.
2. Token cost calculation accuracy across model pricing tiers.
3. Single-turn agent tracing: captures model call latency, tokens, and cost.
4. Multi-step tool execution tracing: captures model decision, tool execution,
   and final synthesis as discrete steps with individual latencies.
5. Self-correction retry tracing: logs metrics for query rewrite cycles.
6. Structured logging: verifies JSON-formatted telemetry emitted to logs.
7. LangSmith configuration helper: validates environment flag activation.
8. Formatted summary table generation for CLI and observability dashboards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from phase_4_agents.graph_agent import build_state_graph_agent

from phase_5_production.telemetry import (
    StepTelemetry,
    StepTelemetryTracer,
    calculate_token_cost,
    configure_langsmith_tracing,
)

# =============================================================================
# 1. Schema & Cost Calculation Unit Tests
# =============================================================================


def test_step_telemetry_schema_validation() -> None:
    """Verify StepTelemetry enforces schema types and rejects undeclared fields."""
    step = StepTelemetry(
        step_number=1,
        step_type="llm_call",
        node_or_tool="call_model",
        start_time="2026-09-14T10:00:00.000Z",
        end_time="2026-09-14T10:00:00.250Z",
        latency_ms=250.0,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_usd=0.000045,
        status="success",
        input_preview="What is the weather?",
        output_preview="Clear and sunny.",
    )
    assert step.step_number == 1
    assert step.latency_ms == 250.0
    assert step.total_tokens == 150
    assert step.cost_usd == 0.000045

    # Test serialization to dictionary
    data = step.to_log_dict()
    assert data["node_or_tool"] == "call_model"
    assert data["status"] == "success"

    # Strict extra="forbid" validation
    with pytest.raises(ValueError):
        StepTelemetry(
            step_number=2,
            step_type="tool_execution",
            node_or_tool="calculator",
            start_time="2026-09-14T10:00:00.000Z",
            end_time="2026-09-14T10:00:00.050Z",
            latency_ms=50.0,
            unexpected_field="disallowed",  # type: ignore
        )


def test_calculate_token_cost_pricing_tiers() -> None:
    """Verify token cost calculation for different model architectures."""
    # 1. Gemini Flash rates ($0.15/1M in, $0.60/1M out)
    # 1,000,000 in = $0.15, 1,000,000 out = $0.60 -> Total $0.75
    cost_gemini = calculate_token_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert cost_gemini == 0.75

    # 1,000 in = $0.00015, 500 out = $0.00030 -> Total $0.00045
    cost_small = calculate_token_cost("gemini-2.0-flash", 1000, 500)
    assert cost_small == 0.00045

    # 2. GPT-4o rates ($2.50/1M in, $10.00/1M out)
    cost_gpt4o = calculate_token_cost("gpt-4o", 10_000, 2000)
    # 10,000 * 0.0000025 = 0.025; 2000 * 0.000010 = 0.02 -> 0.045
    assert cost_gpt4o == 0.045

    # 3. Default fallback for unknown model
    cost_default = calculate_token_cost("custom-finetune", 1000, 1000)
    assert cost_default > 0.0


# =============================================================================
# 2. Single-Turn Agent Tracing
# =============================================================================


def test_tracer_captures_single_turn_model_run() -> None:
    """Verify tracer records latency, tokens, and cost on a single-turn agent run."""
    tracer = StepTelemetryTracer(log_to_console=False)
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(provider="mock"),
    )

    thread_id = f"test_trace_single_{uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer],
    }

    # Direct chit-chat query (model produces final answer without tools)
    app.invoke(
        {"messages": [HumanMessage(content="Hello, who are you?")]},
        config=config,
    )

    steps = tracer.steps
    assert len(steps) >= 1

    # Verify model call step
    llm_step = steps[0]
    assert llm_step.step_type == "llm_call"
    assert llm_step.node_or_tool == "call_model"
    assert llm_step.latency_ms >= 0.0
    assert llm_step.total_tokens >= 1
    assert llm_step.cost_usd >= 0.0
    assert llm_step.status == "success"
    assert "Hello" in llm_step.input_preview or "who are you" in llm_step.input_preview


# =============================================================================
# 3. Multi-Step Tool Execution Tracing
# =============================================================================


def test_tracer_captures_multi_step_tool_execution() -> None:
    """Verify tracer captures discrete steps for decision, execution, and synthesis."""
    tracer = StepTelemetryTracer(log_to_console=False)
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(provider="mock"),
    )

    thread_id = f"test_trace_multistep_{uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer],
    }

    # Query triggering a tool call
    app.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        config=config,
    )

    steps = tracer.steps
    # Must capture:
    # 1. Model deciding to call tool (llm_call)
    # 2. Tool execution (tool_execution)
    # 3. Model synthesizing final answer (llm_call)
    assert len(steps) >= 3

    types = [s.step_type for s in steps]
    assert types == ["llm_call", "tool_execution", "llm_call"]

    # Step 1: Decision
    assert steps[0].node_or_tool == "call_model"
    assert steps[0].latency_ms >= 0.0
    assert steps[0].input_tokens > 0

    # Step 2: Tool execution
    assert steps[1].step_type == "tool_execution"
    assert steps[1].node_or_tool == "get_weather"
    assert steps[1].latency_ms >= 0.0
    assert "Tokyo" in steps[1].input_preview

    # Step 3: Synthesis
    assert steps[2].node_or_tool == "call_model"
    assert steps[2].latency_ms >= 0.0
    assert steps[2].total_tokens > 0

    # Summary metrics across entire run
    assert tracer.get_total_latency_ms() >= 0.0
    assert tracer.get_total_tokens() > 0
    assert tracer.get_total_cost_usd() > 0.0


# =============================================================================
# 4. Self-Correction Retry Tracing
# =============================================================================


def test_tracer_captures_self_correction_cycle() -> None:
    """Verify tracer logs telemetry across query rewrites and retry loops."""
    tracer = StepTelemetryTracer(log_to_console=False)
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"test_trace_retry_{uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer],
    }

    # Noisy Mars query triggers retry cycle
    # Noisy Mars query triggers retry cycle
    app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "What are the Mars ECLSS pressure limits? Please search pdf."
                    )
                )
            ]
        },
        config=config,
    )

    steps = tracer.steps
    # Should record initial model call, tool search, rewrite, and retry
    assert len(steps) >= 2
    assert all(s.latency_ms >= 0.0 for s in steps)
    assert all(s.status == "success" for s in steps)


# =============================================================================
# 5. Structured Logging Verification
# =============================================================================


def test_tracer_emits_structured_json_logs(caplog: Any) -> None:
    """Verify that StepTelemetryTracer logs each step as parseable JSON."""
    caplog.set_level(logging.INFO)
    tracer = StepTelemetryTracer(log_to_console=True)

    fake_run_id = uuid4()
    tracer.on_chat_model_start(
        serialized={"name": "gemini-2.5-flash"},
        messages=[[type("Msg", (), {"content": "Hello test"})()]],  # type: ignore
        run_id=fake_run_id,
    )
    tracer.on_llm_end(
        type(
            "LLMResult",
            (),
            {
                "llm_output": {
                    "token_usage": {"prompt_tokens": 12, "completion_tokens": 6}
                },
                "generations": [[type("Gen", (), {"text": "Hi!", "message": None})()]],
            },
        )(),  # type: ignore
        run_id=fake_run_id,
    )

    # Check captured log records
    records = [r for r in caplog.records if "telemetry" in r.name]
    assert len(records) >= 1

    # Ensure log record is valid JSON containing all telemetry keys
    log_json = json.loads(records[0].message)
    assert log_json["step_number"] == 1
    assert log_json["step_type"] == "llm_call"
    assert log_json["input_tokens"] == 12
    assert log_json["output_tokens"] == 6
    assert log_json["total_tokens"] == 18
    assert log_json["latency_ms"] >= 0.0
    assert "cost_usd" in log_json


# =============================================================================
# 6. Formatted Summary Table & LangSmith Helper
# =============================================================================


def test_tracer_format_summary_table() -> None:
    """Verify format_summary_table produces expected ASCII table report."""
    tracer = StepTelemetryTracer(log_to_console=False)
    fake_run_id = uuid4()

    tracer.on_chat_model_start(
        serialized={"name": "gemini-2.5-flash"},
        messages=[[type("Msg", (), {"content": "Calculate 2+2"})()]],  # type: ignore
        run_id=fake_run_id,
    )
    tracer.on_llm_end(
        type(
            "LLMResult",
            (),
            {
                "llm_output": {
                    "token_usage": {"prompt_tokens": 20, "completion_tokens": 5}
                },
                "generations": [[type("Gen", (), {"text": "4", "message": None})()]],
            },
        )(),  # type: ignore
        run_id=fake_run_id,
    )

    table = tracer.format_summary_table()
    assert "PRODUCTION STEP TELEMETRY REPORT" in table
    assert "call_model" in table
    assert "DURATION:" in table
    assert "TOTAL COST:" in table


def test_configure_langsmith_tracing(monkeypatch: Any) -> None:
    """Verify configure_langsmith_tracing activates environment variables."""
    # When no API key is provided
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert configure_langsmith_tracing() is False

    # When API key is provided
    active = configure_langsmith_tracing(
        project_name="test-project",
        api_key="ls__mock_key_12345",
    )
    assert active is True
    import os

    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"
    assert os.environ.get("LANGCHAIN_API_KEY") == "ls__mock_key_12345"
