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

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from phase_4_agents.rag_tools import CLIENT_DATA_WRITE_TOOLS, get_all_tools

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
# 0. Custom Exceptions (Task 4.10 Guardrails)
# -----------------------------------------------------------------------------


class AgentExecutionError(Exception):
    """Base exception for agent execution and guardrail failures."""


class AgentStuckError(AgentExecutionError):
    """Raised when the agent gets stuck in a loop calling identical tools."""


class AgentMaxIterationsExceededError(AgentExecutionError):
    """Raised when the agent reaches the configured maximum iteration limit."""


class AgentCostCapExceededError(AgentExecutionError):
    """Raised when the agent's estimated token cost exceeds the configured cost cap."""


# -----------------------------------------------------------------------------
# Stuck Loop Detection Helper (Task 4.10)
# -----------------------------------------------------------------------------


def _normalize_args(args: Any) -> str:
    """Deterministically serialize tool arguments for comparison."""
    if isinstance(args, dict):
        return json.dumps(args, sort_keys=True, default=str)
    return str(args)


def detect_stuck_agent(
    tool_call_history: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Detect if agent is stuck in repetitive tool-calling cycles or identical loops.

    Conditions checked:
    1. Two consecutive identical tool calls (same tool name and identical arguments).
    2. Same tool call repeated 3 or more times across total conversation history.

    Returns:
        (is_stuck, reason_message)
    """
    if not tool_call_history:
        return False, ""

    # 1. Immediate consecutive duplicate check
    if len(tool_call_history) >= 2:
        last_call = tool_call_history[-1]
        prev_call = tool_call_history[-2]
        if last_call.get("name") == prev_call.get("name") and _normalize_args(
            last_call.get("args")
        ) == _normalize_args(prev_call.get("args")):
            tool_name = last_call.get("name")
            args_str = _normalize_args(last_call.get("args"))
            return (
                True,
                f"Detected consecutive identical tool call: '{tool_name}' "
                f"with args {args_str} called twice in a row without state progress.",
            )

    # 2. Repeated cyclic tool call detection (same call appears >= 3 times in history)
    signatures = [
        f"{c.get('name')}:{_normalize_args(c.get('args'))}" for c in tool_call_history
    ]
    for sig in signatures:
        if signatures.count(sig) >= 3:
            return (
                True,
                f"Detected cyclic loop: Tool call '{sig}' executed 3 or more times. "
                "Halting execution to prevent infinite cycle.",
            )

    return False, ""


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
        total_tokens: Cumulative prompt + completion tokens used across calls.
        total_cost_usd: Estimated cumulative cost in USD.
        tool_call_history: List of past tool invocations for stuck-loop detection.
        error_status: Optional description of fatal guardrail failure.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int
    rewrite_count: int
    total_tokens: int
    total_cost_usd: float
    tool_call_history: list[dict[str, Any]]
    error_status: str | None


# -----------------------------------------------------------------------------
# 2. Relevance Evaluator, Query Rewriter & Stuck Detector
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
    if attempt >= 2 and words:
        return words[0]
    if len(words) > 2:
        return " ".join(words[:2])
    return q_clean


# -----------------------------------------------------------------------------
# 3. Conditional Routers & Human Approval Check
# -----------------------------------------------------------------------------


def is_write_action(state: AgentState) -> bool:
    """Check if the latest model response contains a write action on client data."""
    messages = state.get("messages", [])
    if not messages:
        return False
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        for tc in last_msg.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name in CLIENT_DATA_WRITE_TOOLS:
                return True
    return False


def route_model_output(
    state: AgentState,
) -> Literal["human_approval", "execute_tools", "__end__"]:
    """Inspect the last message produced by the model node.

    If error_status is set -> route to '__end__'.
    If tool calls were requested:
        - If any tool touches client data (write action) -> route to 'human_approval'.
        - If all tools are safe read actions -> route to 'execute_tools'.
    If plain conversational text -> route to '__end__'.
    """
    if state.get("error_status"):
        return "__end__"

    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        if is_write_action(state):
            return "human_approval"
        return "execute_tools"

    return "__end__"


def route_tool_output(
    state: AgentState,
) -> Literal["rewrite_query", "call_model"]:
    """Evaluate tool execution results.

    If error_status is set -> route to 'call_model' for termination.
    If retrieval returned nothing relevant AND rewrite_count < 2:
        -> Route to 'rewrite_query' to retry.
    Otherwise (data found OR retries exhausted):
        -> Route to 'call_model' to synthesize response.
    """
    if state.get("error_status"):
        return "call_model"

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
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    max_iterations: int = 15,
    cost_cap_usd: float = 0.05,
    fail_loudly: bool = True,
):
    """Construct a compiled StateGraph with nodes, edges, cycles, and guardrails.

    Args:
        model: Optional pre-configured BaseChatModel instance.
        tools: Optional list of BaseTool tools (defaults to all 6 tools).
        system_prompt: Optional system prompt instructions.
        provider: Provider name ('google_genai', 'openai', 'mock').
        model_name: Model identifier.
        api_key: Provider API key override.
        temperature: Sampling temperature.
        checkpointer: Optional checkpointer (e.g. PostgresSaver or MemorySaver).
        interrupt_before: Optional list of node names to interrupt before.
        max_iterations: Maximum reasoning steps permitted before halting (15).
        cost_cap_usd: Cumulative estimated dollar budget cap (default: $0.05).
        fail_loudly: Whether to raise typed exceptions on failure (default: True).

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
        """Invoke the LLM with current state messages, tracking costs and guardrails."""
        curr_steps = state.get("step_count", 0) + 1
        if curr_steps > max_iterations:
            msg = (
                "AGENT HALTED: Reached maximum iteration limit of "
                f"{max_iterations} steps. "
                "Failing loudly to prevent runaway infinite loop."
            )
            if fail_loudly:
                raise AgentMaxIterationsExceededError(msg)
            return {"step_count": curr_steps, "error_status": msg}

        messages = list(state.get("messages", []))

        if messages and not isinstance(messages[0], SystemMessage):
            full_messages = [SystemMessage(content=active_prompt), *messages]
        else:
            full_messages = messages

        response = bound_model.invoke(full_messages)

        # Token usage and pricing calculation
        usage = getattr(response, "usage_metadata", None) or {}
        in_tokens = int(usage.get("input_tokens") or 0)
        out_tokens = int(usage.get("output_tokens") or 0)
        if in_tokens == 0 and out_tokens == 0:
            char_count = sum(len(str(m.content)) for m in full_messages) + len(
                str(response.content)
            )
            in_tokens = max(1, char_count // 4)
            out_tokens = max(1, len(str(response.content)) // 4)

        # Rates: $0.15/1M input ($0.00000015/token), $0.60/1M output ($0.00000060/token)
        step_cost = (in_tokens * 0.00000015) + (out_tokens * 0.00000060)
        cum_tokens = state.get("total_tokens", 0) + in_tokens + out_tokens
        cum_cost = state.get("total_cost_usd", 0.0) + step_cost

        # Cost cap check
        if cum_cost > cost_cap_usd:
            msg = (
                f"AGENT HALTED: Estimated cost ${cum_cost:.6f} exceeded cost cap "
                f"of ${cost_cap_usd:.6f}. Total tokens used: {cum_tokens}. "
                "Failing loudly to prevent budget overrun."
            )
            if fail_loudly:
                raise AgentCostCapExceededError(msg)
            return {
                "messages": [response],
                "step_count": curr_steps,
                "total_tokens": cum_tokens,
                "total_cost_usd": cum_cost,
                "error_status": msg,
            }

        # Tool call history and stuck loop detector
        new_history = list(state.get("tool_call_history", []))
        if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                new_history.append(
                    {
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                        "step": curr_steps,
                    }
                )
            is_stuck, reason = detect_stuck_agent(new_history)
            if is_stuck:
                msg = f"AGENT HALTED: Stuck in loop! {reason}"
                if fail_loudly:
                    raise AgentStuckError(msg)
                return {
                    "messages": [response],
                    "step_count": curr_steps,
                    "total_tokens": cum_tokens,
                    "total_cost_usd": cum_cost,
                    "tool_call_history": new_history,
                    "error_status": msg,
                }

        return {
            "messages": [response],
            "step_count": curr_steps,
            "total_tokens": cum_tokens,
            "total_cost_usd": cum_cost,
            "tool_call_history": new_history,
        }

    # Node 2: Execute Tools
    tool_node = ToolNode(active_tools)

    def execute_tools_node(state: AgentState) -> dict[str, Any]:
        """Execute requested tool calls and increment step count."""
        curr_steps = state.get("step_count", 0) + 1
        if curr_steps > max_iterations:
            msg = (
                "AGENT HALTED: Reached maximum iteration limit of "
                f"{max_iterations} steps. "
                "Failing loudly to prevent runaway infinite loop."
            )
            if fail_loudly:
                raise AgentMaxIterationsExceededError(msg)
            return {"step_count": curr_steps, "error_status": msg}

        tool_output = tool_node.invoke(state)
        return {
            "messages": tool_output.get("messages", []),
            "step_count": curr_steps,
        }

    # Node 3: Rewrite Query (Task 4.5 Self-Correction Cycle)
    def rewrite_query_node(state: AgentState) -> dict[str, Any]:
        """Rewrite search query after empty retrieval and increment retry count."""
        curr_steps = state.get("step_count", 0) + 1
        if curr_steps > max_iterations:
            msg = (
                "AGENT HALTED: Reached maximum iteration limit of "
                f"{max_iterations} steps. "
                "Failing loudly to prevent runaway infinite loop."
            )
            if fail_loudly:
                raise AgentMaxIterationsExceededError(msg)
            return {"step_count": curr_steps, "error_status": msg}

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
            "step_count": curr_steps,
        }

    # Node 4: Human Approval Gate (Task 4.7)
    def human_approval_node(state: AgentState) -> dict[str, Any]:
        """Intermediate gate before write actions touching client data.

        Graph pauses here when compiled with interrupt_before=['human_approval'].
        """
        return {"step_count": state.get("step_count", 0) + 1}

    # 4. Assemble the Explicit Graph
    workflow = StateGraph(AgentState)

    # Add explicit nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("rewrite_query", rewrite_query_node)

    # Add explicit edges and cycle
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_model_output,
        {
            "human_approval": "human_approval",
            "execute_tools": "execute_tools",
            "__end__": END,
        },
    )
    workflow.add_edge("human_approval", "execute_tools")
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

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    # By default, always pause before human_approval if checkpointer is enabled!
    active_interrupts = list(interrupt_before or [])
    if checkpointer is not None and "human_approval" not in active_interrupts:
        active_interrupts.append("human_approval")

    if active_interrupts:
        compile_kwargs["interrupt_before"] = active_interrupts

    return workflow.compile(**compile_kwargs)


def handle_human_approval(
    agent: Any,
    config: Any,
    *,
    approved: bool,
    rejection_reason: str = "User declined the write action.",
) -> Any:
    """Process human approval or rejection of a paused write action.

    Args:
        agent: CompiledStateGraph instance.
        config: RunnableConfig containing the thread_id.
        approved: True to approve and resume execution, False to abort.
        rejection_reason: Explanation recorded if rejected.

    Returns:
        The updated state or final execution result.
    """
    state = agent.get_state(config)
    if not state or not state.values:
        raise ValueError("No state found for the specified thread_id.")

    if approved:
        # User approved: continue execution from the paused interrupt
        return agent.invoke(None, config=config)

    # User rejected: abort write action
    # Find pending tool calls from the last AIMessage
    messages = list(state.values.get("messages", []))
    pending_tool_calls: Sequence[Any] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            pending_tool_calls = msg.tool_calls
            break

    rejection_messages: list[BaseMessage] = []
    for tc in pending_tool_calls:
        tc_id = tc.get("id", "cancelled_tool_call")
        tc_name = tc.get("name", "write_tool")
        rejection_messages.append(
            ToolMessage(
                content=(
                    f"Action Cancelled: {rejection_reason} "
                    f"({tc_name} was not executed)."
                ),
                tool_call_id=tc_id,
            )
        )

    # If no specific tool call ID found, add a generic cancellation message
    if not rejection_messages:
        rejection_messages.append(
            HumanMessage(
                content=(
                    "[Human Rejection]: Write action was denied by user. "
                    f"Reason: {rejection_reason}"
                )
            )
        )

    # Update state with rejection ToolMessage as if execute_tools completed,
    # thereby bypassing actual tool execution and letting model synthesize
    # cancellation response.
    agent.update_state(
        config, {"messages": rejection_messages}, as_node="execute_tools"
    )
    return agent.invoke(None, config=config)


# -----------------------------------------------------------------------------
# 5. Task 4.8: Time-Travel Debugging Utilities
# -----------------------------------------------------------------------------


def list_state_history(agent: Any, config: Any) -> list[dict[str, Any]]:
    """Return chronological or reverse-chronological checkpoints for a thread.

    Args:
        agent: CompiledStateGraph instance with an attached checkpointer.
        config: RunnableConfig containing the thread_id.

    Returns:
        List of summarized checkpoint dictionaries.
    """
    history_snapshots = list(agent.get_state_history(config))
    summaries: list[dict[str, Any]] = []

    for idx, snap in enumerate(history_snapshots):
        cfg = snap.config.get("configurable", {})
        parent_cfg = (snap.parent_config or {}).get("configurable", {})
        msgs = snap.values.get("messages", [])
        last_msg = msgs[-1] if msgs else None

        summaries.append(
            {
                "index": idx,
                "checkpoint_id": cfg.get("checkpoint_id", ""),
                "parent_checkpoint_id": parent_cfg.get("checkpoint_id"),
                "thread_id": cfg.get("thread_id", ""),
                "next": tuple(snap.next),
                "step_count": snap.values.get("step_count", 0),
                "rewrite_count": snap.values.get("rewrite_count", 0),
                "messages_count": len(msgs),
                "last_message_type": (type(last_msg).__name__ if last_msg else None),
                "last_message_preview": (
                    str(last_msg.content)[:120] if last_msg else ""
                ),
                "created_at": getattr(snap, "created_at", None),
                "metadata": getattr(snap, "metadata", {}),
            }
        )

    return summaries


def get_checkpoint_snapshot(agent: Any, config: Any, checkpoint_id: str) -> Any:
    """Locate and return the exact StateSnapshot for a specific checkpoint ID.

    Args:
        agent: CompiledStateGraph instance.
        config: RunnableConfig containing the thread_id.
        checkpoint_id: Checkpoint UUID string to find.

    Returns:
        The matching StateSnapshot, or None if not found.
    """
    for snap in agent.get_state_history(config):
        cfg = snap.config.get("configurable", {})
        if cfg.get("checkpoint_id") == checkpoint_id:
            return snap
    return None


def time_travel_replay(
    agent: Any,
    config: Any,
    checkpoint_id: str,
    state_update: dict[str, Any] | None = None,
    as_node: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Replay execution from an arbitrary checkpoint with optional modified state.

    Rewinds to checkpoint_id, applies state_update (if provided) to branch
    a new checkpoint fork, and resumes downstream execution via agent.invoke().

    Args:
        agent: CompiledStateGraph instance.
        config: RunnableConfig containing thread_id.
        checkpoint_id: Target checkpoint UUID to rewind to.
        state_update: Optional dictionary of state keys/messages to update.
        as_node: Optional node name to attribute the update to.

    Returns:
        A tuple of (fork_config, execution_result).
    """
    target_snap = get_checkpoint_snapshot(agent, config, checkpoint_id)
    if not target_snap:
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found in thread history.")

    target_config = target_snap.config

    if state_update is not None:
        # Fork the checkpoint history with updated state
        update_kwargs: dict[str, Any] = {}
        if as_node is not None:
            update_kwargs["as_node"] = as_node
        fork_config = agent.update_state(target_config, state_update, **update_kwargs)
    else:
        fork_config = target_config

    result = agent.invoke(None, config=fork_config)
    return fork_config, result


# -----------------------------------------------------------------------------
# 6. Task 4.9: Progressive Intermediate Step Streaming
# -----------------------------------------------------------------------------


@dataclass
class AgentStreamEvent:
    """Discrete intermediate execution event emitted during graph streaming."""

    event: str
    node: str
    step_number: int
    data: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_sse(self) -> str:
        """Serialize event to standard Server-Sent Event (SSE) format."""
        payload = {
            "event": self.event,
            "node": self.node,
            "step": self.step_number,
            "timestamp": self.timestamp,
            **self.data,
        }
        return f"event: {self.event}\ndata: {json.dumps(payload)}\n\n"


def _parse_stream_update_to_event(
    node_name: str,
    state_chunk: dict[str, Any],
    step_counter: int,
) -> AgentStreamEvent:
    """Convert a LangGraph stream update chunk into an AgentStreamEvent."""
    messages = state_chunk.get("messages", [])
    last_msg = messages[-1] if messages else None

    if node_name == "call_model":
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            tools = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                }
                for tc in last_msg.tool_calls
            ]
            return AgentStreamEvent(
                event="tool_decision",
                node=node_name,
                step_number=step_counter,
                data={
                    "tools_requested": tools,
                    "thought": str(last_msg.content),
                },
            )
        return AgentStreamEvent(
            event="final_answer",
            node=node_name,
            step_number=step_counter,
            data={
                "content": str(last_msg.content) if last_msg else "",
            },
        )

    if node_name == "execute_tools":
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results.append(
                    {
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                        "content": str(msg.content),
                    }
                )
        return AgentStreamEvent(
            event="tool_execution",
            node=node_name,
            step_number=step_counter,
            data={"results": tool_results},
        )

    if node_name == "rewrite_query":
        return AgentStreamEvent(
            event="self_correction",
            node=node_name,
            step_number=step_counter,
            data={
                "rewrite_count": state_chunk.get("rewrite_count", 1),
                "message": str(last_msg.content) if last_msg else "",
            },
        )

    if node_name == "human_approval":
        return AgentStreamEvent(
            event="approval_paused",
            node=node_name,
            step_number=step_counter,
            data={"status": "waiting_for_approval"},
        )

    return AgentStreamEvent(
        event="node_update",
        node=node_name,
        step_number=step_counter,
        data={"messages_count": len(messages)},
    )


def stream_agent_steps(
    agent: Any,
    query: str,
    config: Any | None = None,
) -> Iterator[AgentStreamEvent]:
    """Synchronous generator yielding progressive AgentStreamEvents."""
    input_payload = {"messages": [HumanMessage(content=query)]}
    step_counter = 1

    yield AgentStreamEvent(
        event="step_start",
        node="START",
        step_number=step_counter,
        data={"query": query},
    )

    try:
        for chunk in agent.stream(input_payload, config=config, stream_mode="updates"):
            for node_name, state_chunk in chunk.items():
                step_counter += 1
                yield _parse_stream_update_to_event(
                    node_name, state_chunk, step_counter
                )
    except AgentExecutionError as exc:
        step_counter += 1
        yield AgentStreamEvent(
            event="error",
            node="guardrail",
            step_number=step_counter,
            data={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "status": "halted",
            },
        )
        raise

    yield AgentStreamEvent(
        event="done",
        node="END",
        step_number=step_counter + 1,
        data={"status": "completed"},
    )


async def astream_agent_steps(
    agent: Any,
    query: str,
    config: Any | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    """Asynchronous generator yielding progressive AgentStreamEvents."""
    queue: asyncio.Queue[AgentStreamEvent | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _worker() -> None:
        try:
            for ev in stream_agent_steps(agent, query, config=config):
                asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    loop.run_in_executor(None, _worker)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
