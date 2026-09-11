"""Tests for Task 4.4 LangGraph StateGraph Agent Engine.

Verifies:
1. Explicit StateGraph compilation with typed AgentState schema.
2. Direct conversational path (START -> call_model -> END).
3. Multi-step tool execution loop:
   START -> call_model -> execute_tools -> call_model -> END.
4. Reducer accumulation and state preservation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.agent_cli import build_tool_agent, run_agent_query
from phase_4_agents.graph_agent import (
    AgentState,
    build_state_graph_agent,
    route_model_output,
)
from phase_4_agents.rag_tools import pdf_search

# -----------------------------------------------------------------------------
# 1. State Schema & Router Unit Tests
# -----------------------------------------------------------------------------


def test_agent_state_schema_structure() -> None:
    """Verify AgentState typed schema contains messages and step_count."""
    assert "messages" in AgentState.__annotations__
    assert "step_count" in AgentState.__annotations__


def test_route_model_output_conditional_router() -> None:
    """Verify route_model_output routes to execute_tools or __end__."""
    # Case 1: Empty state
    empty_state: AgentState = {"messages": [], "step_count": 0}
    assert route_model_output(empty_state) == "__end__"

    # Case 2: Model produced plain text -> route to __end__
    text_msg = AIMessage(content="Hello there!")
    text_state: AgentState = {"messages": [text_msg], "step_count": 1}
    assert route_model_output(text_state) == "__end__"

    # Case 3: Model produced read tool calls -> route to execute_tools
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "pdf_search",
                "args": {"query": "eclss"},
                "id": "tc_1",
                "type": "tool_call",
            }
        ],
    )
    tool_state: AgentState = {"messages": [tool_call_msg], "step_count": 1}
    assert route_model_output(tool_state) == "execute_tools"

    # Case 4: Model produced client write tool call (add_to_cart)
    # -> route to human_approval
    write_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add_to_cart",
                "args": {"product_name": "Titanium Drill Bit", "quantity": 1},
                "id": "tc_write_1",
                "type": "tool_call",
            }
        ],
    )
    write_state: AgentState = {"messages": [write_call_msg], "step_count": 1}
    assert route_model_output(write_state) == "human_approval"


# -----------------------------------------------------------------------------
# 2. StateGraph Compilation & Graph Topology Tests
# -----------------------------------------------------------------------------


def test_state_graph_compilation() -> None:
    """Verify StateGraph compiles into a CompiledStateGraph with explicit nodes."""
    agent = build_state_graph_agent(provider="mock", model_name="mock-agent")
    assert isinstance(agent, CompiledStateGraph)
    assert "call_model" in agent.nodes
    assert "human_approval" in agent.nodes
    assert "execute_tools" in agent.nodes


# -----------------------------------------------------------------------------
# 3. Graph Execution & Transition Tests
# -----------------------------------------------------------------------------


def test_state_graph_direct_response_path() -> None:
    """Verify single-step path (START -> call_model -> END)."""
    agent = build_state_graph_agent(provider="mock", model_name="mock-agent")
    initial_input: AgentState = {
        "messages": [HumanMessage(content="Hello, how are you today?")]
    }

    output = agent.invoke(initial_input)
    messages = output["messages"]

    # Exactly 2 messages: user greeting and direct assistant reply
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert not getattr(messages[1], "tool_calls", None)
    assert "Hello!" in str(messages[1].content)
    assert output.get("step_count") == 1


def test_state_graph_tool_execution_loop() -> None:
    """Verify multi-step loop: call_model -> execute_tools -> call_model -> END."""
    agent = build_state_graph_agent(provider="mock", model_name="mock-agent")
    initial_input: AgentState = {
        "messages": [
            HumanMessage(
                content="What are the ECLSS pressure limits in our Mars PDF manuals?"
            )
        ]
    }

    output = agent.invoke(initial_input)
    messages = output["messages"]

    # 4 messages in loop: User -> AIMessage(tool) -> ToolMessage -> AIMessage(final)
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert getattr(messages[1], "tool_calls", None)
    assert messages[1].tool_calls[0]["name"] == "pdf_search"
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage)
    assert not getattr(messages[3], "tool_calls", None)
    assert output.get("step_count", 0) >= 2


def test_state_graph_custom_toolset() -> None:
    """Verify StateGraph can be built with a specific subset of tools."""
    agent = build_state_graph_agent(
        tools=[pdf_search],
        provider="mock",
        model_name="mock-agent",
    )
    assert isinstance(agent, CompiledStateGraph)
    input_state: AgentState = {
        "messages": [HumanMessage(content="What are the Mars PDF pressure limits?")]
    }
    output = agent.invoke(input_state)
    assert len(output["messages"]) == 4


def test_cli_integration_with_state_graph_engine() -> None:
    """Verify agent_cli executes successfully using --engine state_graph."""
    agent = build_tool_agent(
        engine="state_graph", provider="mock", model_name="mock-agent"
    )
    result = run_agent_query(
        agent,
        "Query the database for customers in San Francisco",
        provider="mock",
        model_name="mock-agent",
    )

    assert result.total_steps >= 3
    tool_names = [
        tc["name"] for step in result.steps if step.tool_calls for tc in step.tool_calls
    ]
    assert "db_query" in tool_names
    assert any(step.actor == "Tool Execution" for step in result.steps)
