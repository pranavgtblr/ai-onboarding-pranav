"""Task 4.4 & 4.5: Explicit LangGraph StateGraph Multi-Tool Agent with Retry Cycle.

Rebuilds the agent using an explicit:
1. AgentState typed schema with an add_messages reducer and rewrite_count.
2. Nodes: 'call_model', 'execute_tools', and 'rewrite_query'.
3. Edges:
   - START -> call_model
   - call_model -> route_model_output (execute_tools or END)
   - execute_tools -> route_tool_output (rewrite_query or call_model)
   - rewrite_query -> call_model (closing the self-correcting cycle)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
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
        rewrite_count: Number of query rewriting retry attempts executed (max 2).
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int
    rewrite_count: int


# -----------------------------------------------------------------------------
# 2. Relevance Evaluator & Query Rewriter
# -----------------------------------------------------------------------------


def is_retrieval_empty_or_irrelevant(content_or_message: str | BaseMessage) -> bool:
    """Check if tool output indicates no relevant data was found."""
    if isinstance(content_or_message, BaseMessage):
        content = str(content_or_message.content)
    else:
        content = str(content_or_message)

    lower = content.lower().strip()
    indicators = [
        "no matching records found",
        "rows returned: 0",
        "0 rows",
        "returned 0 rows",
        "no relevant pdf documents matched",
        "no relevant pdf sections found",
        "no matching documentation",
        "no relevant documentation pages found",
        "no relevant web results found",
        "0 matching records",
        "empty result",
        "results: []",
    ]
    return any(ind in lower for ind in indicators)


def rewrite_search_query(failed_query: str, attempt: int = 1) -> str:
    """Generate an alternative, broader, or reformatted search query."""
    q_clean = failed_query.strip()
    q_lower = q_clean.lower()

    # Domain-specific heuristics for query expansion
    if "customer" in q_lower or "client" in q_lower:
        return "customers"
    if "order" in q_lower or "sale" in q_lower:
        return "orders"
    if "product" in q_lower or "stock" in q_lower or "price" in q_lower:
        return "products"
    if "appointment" in q_lower or "doctor" in q_lower:
        return "appointments"
    if "eclss" in q_lower or "pressure" in q_lower or "mars" in q_lower:
        return "Mars ECLSS operating limits"
    if "toobler" in q_lower or "website" in q_lower or "site" in q_lower:
        return "Toobler digital innovation capabilities"
    if "artemis" in q_lower or "news" in q_lower or "nasa" in q_lower:
        return "NASA Artemis mission updates"

    # Generic fallback: extract keywords
    words = re.findall(r"\b\w+\b", q_clean)
    if len(words) > 2:
        return " ".join(words[:2])
    return q_clean


# -----------------------------------------------------------------------------
# 3. Conditional Routers
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


def route_tool_output(
    state: AgentState,
) -> Literal["rewrite_query", "call_model"]:
    """Evaluate tool execution results.

    If retrieval returned nothing relevant AND rewrite_count < 2:
        -> Route to 'rewrite_query' to retry.
    Otherwise (data found OR retries exhausted):
        -> Route to 'call_model' to synthesize response.
    """
    messages = state.get("messages", [])
    curr_retries = state.get("rewrite_count", 0)

    # Enforce maximum 2 retry attempts guard
    if curr_retries >= 2:
        return "call_model"

    # Inspect the most recent ToolMessage(s)
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    if not tool_messages:
        return "call_model"

    last_tool_msg = tool_messages[-1]
    tool_content = str(last_tool_msg.content)

    if is_retrieval_empty_or_irrelevant(tool_content):
        return "rewrite_query"

    return "call_model"


# -----------------------------------------------------------------------------
# 4. StateGraph Constructor
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
    """Construct a compiled StateGraph with explicit nodes, edges, and retry cycle.

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

    bound_model = llm.bind_tools(active_tools)

    # Node 1: Call Model
    def call_model(state: AgentState) -> dict[str, Any]:
        """Invoke the LLM with current state messages and optional system prompt."""
        messages = list(state.get("messages", []))

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

    # Node 2: Execute Tools
    tool_node = ToolNode(active_tools)

    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        """Execute requested tool calls and increment step count."""
        tool_output = tool_node.invoke(state)
        curr_steps = state.get("step_count", 0) + 1
        return {
            "messages": tool_output.get("messages", []),
            "step_count": curr_steps,
        }

    # Node 3: Rewrite Query (Task 4.5 Self-Correction Cycle)
    def rewrite_query_node(state: AgentState) -> dict[str, Any]:
        """Rewrite search query after empty retrieval and increment retry count."""
        curr_retries = state.get("rewrite_count", 0)
        messages = list(state.get("messages", []))

        # Identify the failed query from the last tool call
        failed_query = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    failed_query = str(tc.get("args", {}).get("query", ""))
                    if failed_query:
                        break
            if failed_query:
                break

        if not failed_query and messages:
            failed_query = str(messages[0].content)

        rewritten = rewrite_search_query(failed_query, curr_retries + 1)
        next_retries = curr_retries + 1

        self_correction_msg = HumanMessage(
            content=(
                f"[Self-Correction Attempt {next_retries}/2]: Previous search for "
                f"'{failed_query}' returned no relevant results. Retrying with "
                f"rewritten query: '{rewritten}'."
            )
        )
        return {
            "messages": [self_correction_msg],
            "rewrite_count": next_retries,
            "step_count": state.get("step_count", 0) + 1,
        }

    # 4. Assemble the Explicit Graph
    workflow = StateGraph(AgentState)

    # Add explicit nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("rewrite_query", rewrite_query_node)

    # Add explicit edges and cycle
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_model_output,
        {
            "execute_tools": "execute_tools",
            "__end__": END,
        },
    )
    workflow.add_conditional_edges(
        "execute_tools",
        route_tool_output,
        {
            "rewrite_query": "rewrite_query",
            "call_model": "call_model",
        },
    )
    # The cycle: loop back to call_model with the rewritten query!
    workflow.add_edge("rewrite_query", "call_model")

    return workflow.compile()
