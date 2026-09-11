"""Task 4.8: Automated Tests for Time-Travel Debugging in LangGraph.

Verifies:
1. Snapshot recording & history inspection across multi-step execution.
2. Rewinding to an arbitrary intermediate checkpoint and replaying deterministically.
3. Modifying state at an earlier checkpoint to fork execution along a new branch.
4. Correcting intermediate tool outputs to fix downstream LLM reasoning.
5. Time travel with persistent PostgreSQL checkpointer.
"""

from __future__ import annotations

import uuid
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.graph_agent import (
    build_state_graph_agent,
    get_checkpoint_snapshot,
    list_state_history,
    time_travel_replay,
)


def test_list_state_history_records_snapshots() -> None:
    """Verify list_state_history captures chronological snapshots and metadata."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(provider="mock", checkpointer=checkpointer),
    )

    thread_id = f"test_hist_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Execute a standard tool-calling query
    app.invoke(
        {"messages": [HumanMessage(content="Find customers living in Tokyo")]},
        config=config,
    )

    history = list_state_history(app, config)
    assert len(history) >= 4

    # Latest snapshot should be finished (next is empty tuple)
    assert history[0]["next"] == ()
    assert history[0]["checkpoint_id"] != ""

    # Checkpoints must have parent links showing the chain of execution
    assert history[0]["parent_checkpoint_id"] is not None

    # Verify earliest snapshot represents initial input
    earliest = history[-1]
    assert earliest["parent_checkpoint_id"] is None
    assert earliest["messages_count"] == 0 or earliest["messages_count"] == 1


def test_replay_from_intermediate_checkpoint() -> None:
    """Rewind to an intermediate checkpoint and replay downstream execution."""
    checkpointer = MemorySaver()
    # Pause before tools to easily capture an intermediate pause state
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            checkpointer=checkpointer,
            interrupt_before=["execute_tools"],
        ),
    )

    thread_id = f"test_replay_step_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Run until pause before tool execution
    app.invoke(
        {"messages": [HumanMessage(content="Find customers living in Tokyo")]},
        config=config,
    )

    history = list_state_history(app, config)
    # Find the checkpoint paused at execute_tools
    paused_snap = next((s for s in history if "execute_tools" in s["next"]), None)
    assert paused_snap is not None
    checkpoint_id = paused_snap["checkpoint_id"]

    # 2. Replay from this exact checkpoint without modifying state
    fork_cfg, result = time_travel_replay(app, config, checkpoint_id)
    assert fork_cfg is not None

    messages = result.get("messages", [])
    assert len(messages) >= 2
    # Downstream execution should have run the tool and synthesized an answer
    assert any(isinstance(m, ToolMessage) for m in messages)
    assert any(isinstance(m, AIMessage) for m in messages)


def test_time_travel_with_modified_state_forks_history() -> None:
    """Rewind to initial state, modify the prompt, and verify execution branches."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(provider="mock", checkpointer=checkpointer),
    )

    thread_id = f"test_fork_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Run original query for Tokyo
    app.invoke(
        {"messages": [HumanMessage(content="Find customers living in Tokyo")]},
        config=config,
    )

    orig_history = list_state_history(app, config)
    orig_count = len(orig_history)

    # Find the earliest snapshot right after initial input was accepted
    initial_snap = orig_history[-2]
    target_id = initial_snap["checkpoint_id"]

    # 2. Time-travel: fork from initial state with a DIFFERENT question (Paris)
    fork_cfg, res = time_travel_replay(
        app,
        config,
        target_id,
        state_update={
            "messages": [HumanMessage(content="Find customers living in Paris")]
        },
    )

    # 3. Verify fork configuration
    fork_id = fork_cfg.get("configurable", {}).get("checkpoint_id")
    assert fork_id != target_id
    assert fork_id is not None

    # 4. Verify the new branch executed for Paris
    messages = res.get("messages", [])
    assert any(
        isinstance(m, ToolMessage) and "paris" in str(m.content).lower()
        for m in messages
    )

    # 5. Verify original history still exists and branched history expanded
    new_history = list_state_history(app, config)
    assert len(new_history) > orig_count


def test_time_travel_corrects_tool_output_to_debug_llm() -> None:
    """Simulate time-travel debugging: edit an intermediate tool message.

    A developer discovers a faulty tool output, edits the tool message in
    the state checkpoint, and verifies the LLM synthesizes the corrected answer.
    """
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(provider="mock", checkpointer=checkpointer),
    )

    thread_id = f"test_debug_tool_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. Run original query
    app.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        config=config,
    )

    # 2. Locate snapshot right after tool execution (before final LLM answer)
    history = list_state_history(app, config)
    tool_exec_snap = None
    for item in history:
        snap = get_checkpoint_snapshot(app, config, item["checkpoint_id"])
        if snap and any(
            isinstance(m, ToolMessage) for m in snap.values.get("messages", [])
        ):
            # Check if this snapshot is waiting for call_model
            if "call_model" in snap.next:
                tool_exec_snap = snap
                break

    assert tool_exec_snap is not None
    target_id = tool_exec_snap.config.get("configurable", {}).get("checkpoint_id")
    assert target_id is not None

    # Find the tool message ID
    messages = list(tool_exec_snap.values.get("messages", []))
    orig_tool_msg = next(m for m in messages if isinstance(m, ToolMessage))

    # 3. Time travel: inject a corrected tool message
    corrected_tool_msg = ToolMessage(
        content="Simulated Correction: Tokyo Weather is 35°C, Blazing Sunshine.",
        tool_call_id=orig_tool_msg.tool_call_id,
    )

    fork_cfg, res = time_travel_replay(
        app,
        config,
        target_id,
        state_update={"messages": [corrected_tool_msg]},
        as_node="execute_tools",
    )

    final_msgs = res.get("messages", [])
    final_ai_msg = final_msgs[-1]
    assert isinstance(final_ai_msg, AIMessage)
    assert "Blazing Sunshine" in str(final_ai_msg.content)


def test_time_travel_with_postgres_saver() -> None:
    """Verify time-travel history inspection and replay with PostgreSQL checkpointer."""
    try:
        from phase_4_agents.config import get_postgres_checkpointer

        checkpointer_ctx = get_postgres_checkpointer()
        postgres_saver = checkpointer_ctx.__enter__()
    except Exception:
        # If Postgres is not reachable in local environment, gracefully skip
        return

    try:
        app = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                checkpointer=postgres_saver,
                interrupt_before=["execute_tools"],
            ),
        )

        thread_id = f"test_pg_tt_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 1. Run until paused
        app.invoke(
            {"messages": [HumanMessage(content="Find customers living in Tokyo")]},
            config=config,
        )

        # 2. Inspect Postgres history
        history = list_state_history(app, config)
        assert len(history) >= 2

        # 3. Replay with state update
        paused_id = history[0]["checkpoint_id"]
        fork_cfg, res = time_travel_replay(
            app,
            config,
            paused_id,
            state_update={
                "messages": [
                    HumanMessage(content="Find customers living in London instead")
                ]
            },
        )

        assert fork_cfg.get("configurable", {}).get("checkpoint_id") != paused_id
        assert len(res.get("messages", [])) >= 1
    finally:
        checkpointer_ctx.__exit__(None, None, None)
