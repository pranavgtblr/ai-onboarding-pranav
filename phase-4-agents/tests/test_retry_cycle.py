"""Tests for Task 4.5 LangGraph Corrective Retrieval Cycle.

Verifies:
1. Retrieval evaluator detection: recognizes empty/irrelevant results vs populated hits.
2. Query rewrite logic: correctly strips noise and appends guidance.
3. 1 Retry Cycle leading to success:
   Initial query fails -> rewrite query triggered -> revised tool call -> success.
4. Maximum loop termination guard:
   Queries with no possible matches retry up to exactly 2 times and gracefully end.
"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.graph_agent import (
    AgentState,
    build_state_graph_agent,
    is_retrieval_empty_or_irrelevant,
    rewrite_search_query,
    route_tool_output,
)

# -----------------------------------------------------------------------------
# 1. Evaluator & Rewriter Unit Tests
# -----------------------------------------------------------------------------


def test_evaluator_identifies_empty_or_irrelevant_retrievals() -> None:
    """Verify is_retrieval_empty_or_irrelevant accurately spots misses."""
    # Matches / Populated hits -> Not empty
    good_pdf = ToolMessage(
        content=(
            "Document Match 1: Section 4.2 Environmental Control "
            "and Life Support System."
        ),
        tool_call_id="tc_1",
    )
    assert is_retrieval_empty_or_irrelevant(good_pdf) is False

    good_db = ToolMessage(
        content="Results: [{'customer_id': 1, 'name': 'Alice'}]",
        tool_call_id="tc_2",
    )
    assert is_retrieval_empty_or_irrelevant(good_db) is False

    # Empty / Zero hits -> True
    empty_pdf = ToolMessage(
        content=(
            "No relevant PDF sections found for query 'atlantis'. "
            "Try broader search keywords."
        ),
        tool_call_id="tc_3",
    )
    assert is_retrieval_empty_or_irrelevant(empty_pdf) is True

    empty_db = ToolMessage(
        content="Query returned 0 rows.",
        tool_call_id="tc_4",
    )
    assert is_retrieval_empty_or_irrelevant(empty_db) is True

    empty_list = ToolMessage(
        content="Results: []",
        tool_call_id="tc_5",
    )
    assert is_retrieval_empty_or_irrelevant(empty_list) is True

    empty_site = ToolMessage(
        content=(
            "No relevant documentation pages found for query 'quantum'. "
            "Try general navigation."
        ),
        tool_call_id="tc_6",
    )
    assert is_retrieval_empty_or_irrelevant(empty_site) is True


def test_rewrite_search_query_strips_noise() -> None:
    """Verify rewrite_search_query cleans up stop words and noise."""
    query = "please find any customers living in Atlantis"
    rewritten = rewrite_search_query(query)
    assert "please" not in rewritten.lower()
    assert "find" not in rewritten.lower()
    assert len(rewritten) > 0


def test_route_tool_output_conditional_router() -> None:
    """Verify route_tool_output routes to rewrite_query or call_model."""
    # Case 1: Empty results and rewrite_count < 2 -> route to rewrite_query
    empty_tool_msg = ToolMessage(content="Query returned 0 rows.", tool_call_id="tc_1")
    state_retry: AgentState = {
        "messages": [empty_tool_msg],
        "step_count": 2,
        "rewrite_count": 0,
    }
    assert route_tool_output(state_retry) == "rewrite_query"

    # Case 2: Populated results -> route directly to call_model for synthesis
    success_tool_msg = ToolMessage(
        content="Results: [{'name': 'Bob'}]", tool_call_id="tc_2"
    )
    state_success: AgentState = {
        "messages": [success_tool_msg],
        "step_count": 2,
        "rewrite_count": 0,
    }
    assert route_tool_output(state_success) == "call_model"

    # Case 3: Empty results but rewrite_count >= 2 -> route to call_model fallback
    state_max_retries: AgentState = {
        "messages": [empty_tool_msg],
        "step_count": 6,
        "rewrite_count": 2,
    }
    assert route_tool_output(state_max_retries) == "call_model"


# -----------------------------------------------------------------------------
# 2. End-to-End StateGraph Cycle Tests
# -----------------------------------------------------------------------------


def test_agent_retry_cycle_succeeds_on_second_attempt() -> None:
    """Verify agent self-corrects: 1 retry cycle leading to success.

    Flow:
    1. START -> call_model -> issues db_query for Atlantis
    2. execute_tools -> ToolMessage: "Query returned 0 rows."
    3. route_tool_output detects empty retrieval -> routes to rewrite_query
    4. rewrite_query increments rewrite_count to 1 and posts HumanMessage guidance
    5. call_model -> issues revised tool call (MockToolChatModel detects retry guidance)
    6. execute_tools -> returns customer records
    7. route_tool_output detects populated data -> routes to call_model
    8. call_model -> synthesizes final answer -> route_model_output -> END.
    """

    app = cast(CompiledStateGraph, build_state_graph_agent(provider="mock"))
    initial_input: AgentState = {
        "messages": [HumanMessage(content="Find customers living in Atlantis")],
        "step_count": 0,
        "rewrite_count": 0,
    }

    final_state = app.invoke(initial_input)
    messages = list(final_state["messages"])

    # Verify rewrite occurred
    assert final_state.get("rewrite_count") == 1

    # Verify message sequence contains:
    # 1. Initial HumanMessage
    # 2. First AIMessage (tool call for Atlantis)
    # 3. First ToolMessage (0 rows)
    # 4. Self-correction HumanMessage
    # 5. Second AIMessage (revised tool call)
    # 6. Second ToolMessage (populated results)
    # 7. Final AIMessage (synthesized response)
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    assert len(tool_messages) >= 2
    assert any(
        "No matching records found" in str(m.content)
        or "Rows Returned: 0" in str(m.content)
        for m in tool_messages
    )
    assert any(
        "Customer" in str(m.content)
        or "Alice" in str(m.content)
        or "Tokyo" in str(m.content)
        for m in tool_messages
    )

    # Check that self-correction instruction message was injected
    assert any(
        "[Self-Correction Attempt 1/2]" in str(m.content) for m in human_messages
    )

    # Final response is non-empty synthesis
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage)
    assert len(str(last_msg.content)) > 0


def test_agent_terminates_after_max_two_retries() -> None:
    """Verify loop termination guard prevents infinite loops on impossible queries.

    An unmatchable query cycles at most 2 times (rewrite_count reaches 2)
    and then halts at call_model -> END with a transparent explanation.
    """
    app = cast(CompiledStateGraph, build_state_graph_agent(provider="mock"))
    initial_input: AgentState = {
        "messages": [
            HumanMessage(content="Unmatchable non-existent information query xyz123")
        ],
        "step_count": 0,
        "rewrite_count": 0,
    }

    final_state = app.invoke(initial_input)
    messages = list(final_state["messages"])

    # Verify rewrite_count reached exactly the ceiling of 2
    assert final_state.get("rewrite_count") == 2

    # Verify final message is a synthesized answer from the model (not stuck in a loop)
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage)
    assert len(str(last_msg.content)) > 0
