"""Tests for Task 4.6 Postgres Checkpointer & Process Interruption/Resumption.

Verifies:
1. Postgres checkpointer connection & automatic schema setup.
2. Interruption mid-run (process killed): state checkpointed to Postgres.
3. Resumption in a completely separate process/connection using thread_id:
   Verifies next node inspection, message history preservation, and completion.
4. Thread isolation: multiple concurrent sessions maintain isolated checkpoints.
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.config import get_postgres_checkpointer
from phase_4_agents.graph_agent import AgentState, build_state_graph_agent


@pytest.fixture(scope="module")
def postgres_available() -> bool:
    """Verify if PostgreSQL container is running and accessible."""
    try:
        with get_postgres_checkpointer() as checkpointer:
            checkpointer.setup()
        return True
    except Exception:
        return False


def test_postgres_saver_connection_and_setup(postgres_available: bool) -> None:
    """Verify PostgresSaver connects to port 5433 and initializes tables."""
    if not postgres_available:
        pytest.skip("PostgreSQL container on port 5433 not accessible")

    with get_postgres_checkpointer() as checkpointer:
        checkpointer.setup()
        # Ensure checkpointer instance was returned and active
        assert checkpointer is not None


def test_kill_process_mid_run_and_resume_from_checkpoint(
    postgres_available: bool,
) -> None:
    """Verify that an interrupted execution can be cleanly resumed from Postgres.

    Process 1:
      - Interrupted before 'execute_tools'
      - Checkpointed to Postgres
      - Exits (simulating process crash / kill)

    Process 2:
      - Fresh connection & fresh graph compilation
      - Loads state from Postgres via thread_id
      - Resumes with None (no new user input)
      - Successfully finishes execution to END
    """
    if not postgres_available:
        pytest.skip("PostgreSQL container on port 5433 not accessible")

    import uuid

    thread_id = f"test_thread_kill_resume_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # =========================================================================
    # PROCESS 1: Run until interrupt (mid-run)
    # =========================================================================
    with get_postgres_checkpointer() as cp1:
        app1 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                checkpointer=cp1,
                interrupt_before=["execute_tools"],
            ),
        )

        initial_input: AgentState = {
            "messages": [HumanMessage(content="Find customers living in Tokyo")],
            "step_count": 0,
            "rewrite_count": 0,
        }

        # Invocation will halt before 'execute_tools'
        res1 = app1.invoke(initial_input, config=config)
        messages1 = list(res1["messages"])

        # Model made a tool decision, but execute_tools has NOT executed yet
        assert len(messages1) == 2
        assert isinstance(messages1[0], HumanMessage)
        assert isinstance(messages1[1], AIMessage)
        assert getattr(messages1[1], "tool_calls", None)

    # Process 1 has exited, context manager closed and disconnected.
    # The state is now persisted in Postgres.

    # =========================================================================
    # PROCESS 2: Fresh client, load checkpoint, and resume to completion
    # =========================================================================
    with get_postgres_checkpointer() as cp2:
        app2 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                checkpointer=cp2,
                # Process 2 doesn't interrupt before execute_tools so it can finish
            ),
        )

        # 1. Inspect checkpoint in Postgres
        saved_state = app2.get_state(config)
        assert saved_state is not None
        assert saved_state.next == ("execute_tools",)

        saved_messages = list(saved_state.values.get("messages", []))
        assert len(saved_messages) == 2
        assert saved_messages[0].content == "Find customers living in Tokyo"

        # 2. Resume execution from checkpoint!
        res2 = app2.invoke(None, config=config)
        messages2 = list(res2["messages"])

        # Verify resumed execution completed:
        # Initial HumanMessage -> Tool Decision -> Tool Execution -> Final Synthesis
        assert len(messages2) >= 4
        tool_messages = [m for m in messages2 if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1

        last_msg = messages2[-1]
        assert isinstance(last_msg, AIMessage)
        assert len(str(last_msg.content)) > 0


def test_checkpointer_thread_isolation(postgres_available: bool) -> None:
    """Verify distinct thread_ids maintain completely isolated states."""
    if not postgres_available:
        pytest.skip("PostgreSQL container on port 5433 not accessible")

    import uuid

    uid = uuid.uuid4().hex[:8]
    thread_a = f"thread_alpha_{uid}"
    thread_b = f"thread_beta_{uid}"

    with get_postgres_checkpointer() as cp:
        app = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                checkpointer=cp,
            ),
        )

        # Thread A: Chit-chat
        app.invoke(
            {"messages": [HumanMessage(content="Hello from Thread Alpha!")]},
            config={"configurable": {"thread_id": thread_a}},
        )

        # Thread B: Weather query
        app.invoke(
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
            config={"configurable": {"thread_id": thread_b}},
        )

        state_a = app.get_state({"configurable": {"thread_id": thread_a}})
        state_b = app.get_state({"configurable": {"thread_id": thread_b}})

        msg_a = state_a.values["messages"][0].content
        msg_b = state_b.values["messages"][0].content

        assert "Alpha" in msg_a
        assert "Alpha" not in msg_b
        assert "weather" in msg_b.lower()
