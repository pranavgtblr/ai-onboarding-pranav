"""Tests for Task 4.7 Human-in-the-Loop Approval Gate.

Verifies:
1. Safe read actions (db_query, pdf_search, calculator) run without human intervention.
2. Write actions touching client data (add_to_cart) pause at 'human_approval'.
3. Human approval resumes execution and persists the write to ecommerce.db.
4. Human rejection aborts the write, modifies state, and produces a cancellation report
   without mutating client database records.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.graph_agent import (
    AgentState,
    build_state_graph_agent,
    handle_human_approval,
    is_write_action,
    route_model_output,
)
from phase_4_agents.rag_tools import DEFAULT_DB_PATH

# -----------------------------------------------------------------------------
# 1. Routing & Classifier Unit Tests
# -----------------------------------------------------------------------------


def test_is_write_action_detector() -> None:
    """Verify is_write_action detects tools modifying client data."""
    # Read tool call -> False
    read_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "db_query",
                "args": {"query": "SELECT * FROM customers"},
                "id": "tc_1",
                "type": "tool_call",
            }
        ],
    )
    assert is_write_action({"messages": [read_msg], "step_count": 1}) is False

    # Write tool call -> True
    write_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add_to_cart",
                "args": {"product_name": "Titanium Drill Bit", "quantity": 2},
                "id": "tc_2",
                "type": "tool_call",
            }
        ],
    )
    assert is_write_action({"messages": [write_msg], "step_count": 1}) is True


def test_route_model_output_routes_writes_to_human_approval() -> None:
    """Verify router redirects write tools to human_approval node."""
    write_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add_to_cart",
                "args": {"product_name": "Mars Rover Sensor", "quantity": 1},
                "id": "tc_write",
                "type": "tool_call",
            }
        ],
    )
    state: AgentState = {"messages": [write_msg], "step_count": 1}
    assert route_model_output(state) == "human_approval"


# -----------------------------------------------------------------------------
# 2. Graph Pause & Resume (HITL) Execution Tests
# -----------------------------------------------------------------------------


def test_read_tools_execute_without_human_pause() -> None:
    """Verify read-only queries finish without pausing on human approval."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"test_read_hitl_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    res = app.invoke(
        {"messages": [HumanMessage(content="Find customers living in Tokyo")]},
        config=config,
    )

    # State should have finished to END, next is empty
    state = app.get_state(config)
    assert state.next == ()
    messages = list(res["messages"])
    assert any(isinstance(m, ToolMessage) for m in messages)


def test_write_tool_pauses_for_human_approval() -> None:
    """Verify add_to_cart pauses execution before executing the write."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"test_write_pause_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Model will propose add_to_cart
    app.invoke(
        {"messages": [HumanMessage(content="Add 2 Titanium Drill Bits to my cart")]},
        config=config,
    )

    # Graph MUST be paused right before human_approval node!
    state = app.get_state(config)
    assert state.next == ("human_approval",)

    last_msg = state.values["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert getattr(last_msg, "tool_calls", None)
    assert last_msg.tool_calls[0]["name"] == "add_to_cart"


def test_human_approval_executes_write_and_updates_db() -> None:
    """Verify approved write action executes against database and finishes."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"test_write_approve_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Step 1: Query that triggers write
    app.invoke(
        {
            "messages": [
                HumanMessage(content="Add 3 Oxygen Scrubber Cartridges to my cart")
            ]
        },
        config=config,
    )

    # Verify paused
    paused_state = app.get_state(config)
    assert paused_state.next == ("human_approval",)

    # 2. Step 2: Human Approves!
    res = handle_human_approval(app, config, approved=True)

    # Verify completed
    final_state = app.get_state(config)
    assert final_state.next == ()

    messages = list(res["messages"])
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    assert "Cart Updated" in str(tool_msgs[0].content)

    # Check sqlite DB record was created
    if DEFAULT_DB_PATH.exists():
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM cart_items WHERE product_name LIKE '%Oxygen%'"
        )
        count = cur.fetchone()[0]
        conn.close()
        assert count >= 1


def test_human_rejection_aborts_write_without_database_mutation() -> None:
    """Verify declined write action cancels tool execution.

    Mutates zero client data.
    """
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"test_write_reject_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Trigger write action
    app.invoke(
        {"messages": [HumanMessage(content="Buy 50 Mars Rover Sensors")]},
        config=config,
    )

    # Verify paused
    paused_state = app.get_state(config)
    assert paused_state.next == ("human_approval",)

    # Record initial count of cart items
    initial_count = 0
    if DEFAULT_DB_PATH.exists():
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cart_items WHERE quantity = 50")
        initial_count = cur.fetchone()[0]
        conn.close()

    # 2. Human REJECTS the action!
    res = handle_human_approval(
        app,
        config,
        approved=False,
        rejection_reason="Unauthorized purchase amount detected.",
    )

    # Verify graph halted and completed
    final_state = app.get_state(config)
    assert final_state.next == ()

    messages = list(res["messages"])
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    assert "Action Cancelled" in str(tool_msgs[0].content)
    assert "Unauthorized purchase amount" in str(tool_msgs[0].content)

    # Verify database was NOT mutated
    if DEFAULT_DB_PATH.exists():
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cart_items WHERE quantity = 50")
        final_count = cur.fetchone()[0]
        conn.close()
        assert final_count == initial_count
