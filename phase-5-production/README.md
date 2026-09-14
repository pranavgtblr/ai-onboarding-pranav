# Phase 5: Production AI Engineering

Phase 5 addresses the gap between a demo and software clients pay for. A demo needs to work once, on chosen data. A production system needs to work for thousands of strangers, at predictable cost, with observable traces, hardened against prompt injection, and backed by automated regression evaluation.

---

## Task 5.1: Per-Step Tracing (Tokens, Cost, Latency) with LangSmith & StepTelemetryTracer

> *"You cannot debug what you cannot see."*

Task 5.1 wires comprehensive observability into our agent runtime. Every single execution captures and logs **per-step tokens, cost, and latency** across every model call and tool execution.

### 1. Dual-Mode Tracing Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   PRODUCTION AGENT TRACING PIPELINE                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   [Agent Invocation: `agent.invoke()` or `stream_agent_steps()`]       │
│                                  │                                     │
│                     LangChain / LangGraph Callbacks                    │
│                                  │                                     │
│         ┌────────────────────────┴────────────────────────┐            │
│         ▼                                                 ▼            │
│  [Cloud Tracing: LangSmith]              [Local Telemetry: StepTracer] │
│  • Enabled via LANGSMITH_API_KEY         • Zero external dependencies  │
│  • Full tree visualization               • Microsecond wall-clock time │
│  • Project run tracking                  • Exact / estimated tokens    │
│                                          • Provider rate cost ($)      │
│                                          • Structured JSON log lines   │
│                                                           │            │
│                                                           ▼            │
│                                           [Observability Dashboard]   │
│                                           • Step Telemetry Table       │
│                                           • Per-node latency breakdown │
│                                           • Total conversation budget  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2. The Core Telemetry Fields

Each step records a validated `StepTelemetry` record:
- **`step_number`**: Monotonically increasing counter (1, 2, 3...).
- **`step_type`**: `llm_call`, `tool_execution`, or `node_update`.
- **`node_or_tool`**: Name of the executing node (`call_model`, `get_weather`, `pdf_search`).
- **`latency_ms`**: Precise wall-clock duration in milliseconds (`time.perf_counter()`).
- **`input_tokens` / `output_tokens` / `total_tokens`**: Token counts extracted from provider metadata or estimated.
- **`cost_usd`**: Calculated against live provider rates ($0.15/1M input, $0.60/1M output for Gemini Flash).
- **`status`**: `success`, `error`, or `paused`.
- **`input_preview` / `output_preview`**: Sanitized payloads for debugging.

### 3. Usage & CLI Demonstrations

#### Running with CLI `--trace` flag
```bash
uv run python -m phase_4_agents.agent_cli --provider mock --query "What is the weather in Tokyo?" --trace
```

#### Output:
```text
========================================================================================
 📊 PRODUCTION STEP TELEMETRY REPORT (TASK 5.1)
========================================================================================
Step  Type             Node/Tool              Latency      In Tok   Out Tok  Cost ($)    
----------------------------------------------------------------------------------------
1     llm_call         call_model             1.0 ms       471      1        $0.000071   
2     tool_execution   get_weather            0.5 ms       0        0        $0.000000   
3     llm_call         call_model             0.2 ms       492      30       $0.000092   
----------------------------------------------------------------------------------------
TOTAL STEPS: 3 | DURATION: 1.65 ms | TOKENS: 994 | TOTAL COST: $0.000163
========================================================================================
```

#### Enabling LangSmith Cloud Tracing
Add your credentials to `.env`:
```bash
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="production-agents"
```
Or call programmatically:
```python
from phase_5_production.telemetry import configure_langsmith_tracing

configure_langsmith_tracing(project_name="customer-support-agent")
```

### 4. Automated Tests

Run the dedicated test suite:
```bash
uv run pytest tests/test_tracing.py -v
```

Verifies:
- `StepTelemetry` schema enforcement and JSON export.
- Pricing engine calculations across Gemini, GPT-4o, and default tiers.
- Single-turn agent tracing (latency > 0, token count, calculated cost).
- Multi-step tool runs (model decision -> tool execution -> synthesis).
- Query self-correction retries.
- Structured JSON logging via standard logger.
- LangSmith environment configuration helper.
