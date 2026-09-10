"""Task 4.4: Explicit LangGraph StateGraph Multi-Tool Agent.

Rebuilds the high-level create_agent logic from Task 4.3 using an explicit:
1. AgentState typed schema with an add_messages reducer.
2. Nodes: 'call_model' and 'execute_tools'.
3. Edges: START -> call_model, conditional routing to execute_tools or END,
   and execute_tools -> call_model loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from phase_4_agents.config import get_chat_model
from phase_4_agents.rag_tools import get_all_tools

DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent multi-source reasoning assistant with access to "
    "specialized tools:\n"
    "1. pdf_search: Search Project Odyssey Mars engineering manuals & ECLSS specs.\n"
    "2. site_search: Search crawled website documentation and company capabilities.\n"
    "3. db_query: Query structured relational database tables (customers, orders).\n"
    "4. web_search: Search live internet sources for real-time news & releases.\n"
    "5. calculator: Perform arithmetic calculations.\n"
    "6. get_weather: Look up live city weather conditions.\n\n"
    "Analyze the user query carefully and autonomously decide which tool(s) to call, "
    "or call multiple tools if the query spans multiple domains. If the question can "
    "be answered from general knowledge (greetings, coding logic, basic chit-chat), "
    "answer directly without invoking tools."
)


# -----------------------------------------------------------------------------
# 1. Typed State Schema
# -----------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Explicitly typed state schema for the LangGraph agent.

    Attributes:
        messages: Annotated sequence of conversation messages. Uses the `add_messages`
            reducer to append new messages from nodes immutably.
        step_count: Number of reasoning iterations executed by the graph.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int


# -----------------------------------------------------------------------------
# 2. Conditional Router
# -----------------------------------------------------------------------------


def route_model_output(state: AgentState) -> Literal["execute_tools", "__end__"]:
    """Inspect the last message produced by the model node.

    If tool calls were requested -> route to 'execute_tools'.
    If plain conversational text -> route to '__end__'.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        return "execute_tools"

    return "__end__"


# -----------------------------------------------------------------------------
# 3. StateGraph Constructor
# -----------------------------------------------------------------------------


def build_state_graph_agent(
    *,
    model: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
):
    """Construct a compiled StateGraph with explicit nodes, edges, and state.

    Args:
        model: Optional pre-configured BaseChatModel instance.
        tools: Optional list of BaseTool tools (defaults to all 6 tools).
        system_prompt: Optional system prompt instructions.
        provider: Provider name ('google_genai', 'openai', 'mock').
        model_name: Model identifier.
        api_key: Provider API key override.
        temperature: Sampling temperature.

    Returns:
        CompiledStateGraph runnable.
    """
    llm = model or get_chat_model(
        provider=provider,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

    active_tools = tools if tools is not None else get_all_tools()
    active_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # Bind tools to the model
    bound_model = llm.bind_tools(active_tools)

    # Define Node 1: Call Model
    def call_model(state: AgentState) -> dict[str, Any]:
        """Invoke the LLM with current state messages and optional system prompt."""
        messages = list(state.get("messages", []))

        # Prepend system prompt if not already present
        if messages and not isinstance(messages[0], SystemMessage):
            full_messages = [SystemMessage(content=active_prompt), *messages]
        else:
            full_messages = messages

        response = bound_model.invoke(full_messages)
        curr_steps = state.get("step_count", 0) + 1
        return {
            "messages": [response],
            "step_count": curr_steps,
        }

    # Define Node 2: Execute Tools using LangGraph's prebuilt ToolNode
    tool_node = ToolNode(active_tools)

    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        """Execute requested tool calls and increment step count."""
        tool_output = tool_node.invoke(state)
        curr_steps = state.get("step_count", 0) + 1
        # ToolNode returns {"messages": [ToolMessage, ...]}
        return {
            "messages": tool_output.get("messages", []),
            "step_count": curr_steps,
        }

    # 4. Assemble the Explicit Graph
    workflow = StateGraph(AgentState)

    # Add explicit nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools_node)

    # Add explicit edges
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_model_output,
        {
            "execute_tools": "execute_tools",
            "__end__": END,
        },
    )
    workflow.add_edge("execute_tools", "call_model")

    return workflow.compile()
