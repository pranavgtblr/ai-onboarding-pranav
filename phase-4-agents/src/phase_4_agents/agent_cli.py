"""Task 4.1: Rebuilt Tool-Calling CLI using LangChain/LangGraph create_agent.

Executes calculator and weather tools via modern create_agent, intercepts the
LangGraph execution trace, and maps every step back to the hand-written loop
from Task 2.6.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from phase_4_agents.config import get_chat_model, get_settings
from phase_4_agents.rag_tools import get_all_tools, get_rag_tools
from phase_4_agents.tools import ALL_TOOLS as BASIC_TOOLS

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


@dataclass
class AgentStepTrace:
    """Detailed trace of an intermediate step in the agent loop."""

    step_number: int
    actor: str  # "User", "Model", "Tool"
    message_type: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    phase_2_6_mapping: str = ""


@dataclass
class AgentExecutionResult:
    """Full execution result containing trace steps and final synthesized answer."""

    query: str
    steps: list[AgentStepTrace]
    final_answer: str
    total_steps: int
    provider: str = "google_genai"
    model: str = "gemini-3.5-flash-lite"


def build_tool_agent(
    *,
    tools: list[Any] | None = None,
    system_prompt: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
):
    """Construct a tool-calling agent using provider-agnostic get_chat_model."""
    llm = get_chat_model(
        provider=provider,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

    active_tools = tools if tools is not None else get_all_tools()
    active_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # create_agent creates a CompiledStateGraph with a tool calling loop
    return create_agent(
        model=llm,
        tools=active_tools,
        system_prompt=active_prompt,
    )


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from string or structured content list."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
        return " ".join(texts).strip()
    return str(content)


def run_agent_query(
    agent: Any,
    query: str,
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> AgentExecutionResult:
    """Run a query through the agent, capturing the step-by-step trace."""
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages: list[BaseMessage] = response.get("messages", [])

    step_traces: list[AgentStepTrace] = []
    final_answer = ""
    step_counter = 1

    for msg in messages:
        msg_type = type(msg).__name__

        if isinstance(msg, HumanMessage):
            trace = AgentStepTrace(
                step_number=step_counter,
                actor="User",
                message_type=msg_type,
                content=extract_text_from_content(msg.content),
                phase_2_6_mapping=(
                    "Task 2.6: Initial conversation state initialized: "
                    'contents = [{"role": "user", "parts": [{"text": prompt}]}]'
                ),
            )
            step_traces.append(trace)
            step_counter += 1

        elif isinstance(msg, AIMessage):
            content_text = extract_text_from_content(msg.content)
            raw_tool_calls = getattr(msg, "tool_calls", []) or []

            if raw_tool_calls:
                trace = AgentStepTrace(
                    step_number=step_counter,
                    actor="Model (Tool Decision)",
                    message_type=msg_type,
                    content=content_text,
                    tool_calls=raw_tool_calls,
                    phase_2_6_mapping=(
                        "Task 2.6: Model returned candidate with functionCall; "
                        "tool loop identified tool request and queued execution."
                    ),
                )
            else:
                final_answer = content_text
                trace = AgentStepTrace(
                    step_number=step_counter,
                    actor="Model (Final Answer)",
                    message_type=msg_type,
                    content=content_text,
                    phase_2_6_mapping=(
                        "Task 2.6: Model returned text without functionCall; "
                        "if not function_calls: break (loop terminated successfully)."
                    ),
                )
            step_traces.append(trace)
            step_counter += 1

        elif isinstance(msg, ToolMessage):
            trace = AgentStepTrace(
                step_number=step_counter,
                actor="Tool Execution",
                message_type=msg_type,
                content=extract_text_from_content(msg.content),
                phase_2_6_mapping=(
                    "Task 2.6: Local function executed (execute_calculator / "
                    "execute_weather) and result appended to conversation as "
                    '{"role": "function", "parts": [{"functionResponse": ...}]}'
                ),
            )
            step_traces.append(trace)
            step_counter += 1

    settings = get_settings()
    active_provider = provider or settings.model_provider
    active_model = model_name or settings.model_name
    return AgentExecutionResult(
        query=query,
        steps=step_traces,
        final_answer=final_answer,
        total_steps=len(step_traces),
        provider=active_provider,
        model=active_model,
    )


def print_trace_report(result: AgentExecutionResult) -> None:
    """Print an execution trace report mapped to Task 2.6."""
    print("\n" + "=" * 78)
    print(" 🤖 LANGCHAIN / LANGGRAPH CREATE_AGENT EXECUTION TRACE (TASK 4.2)")
    print("=" * 78)
    print(f"QUESTION        : {result.query}")
    print(f"ACTIVE PROVIDER : {result.provider}")
    print(f"ACTIVE MODEL    : {result.model}")
    print(f"TOTAL STEPS     : {result.total_steps}")
    print("-" * 78)

    for step in result.steps:
        print(f"\n[Step {step.step_number}] Actor: {step.actor} ({step.message_type})")
        if step.tool_calls:
            print("  Tools Requested:")
            for tc in step.tool_calls:
                print(f"    • {tc.get('name')}({tc.get('args')}) [ID: {tc.get('id')}]")
        if step.content:
            preview = (
                step.content if len(step.content) < 300 else step.content[:300] + "..."
            )
            print(f"  Content: {preview}")
        print(f"  -> Task 2.6 Hand-Rolled Mapping: {step.phase_2_6_mapping}")

    print("\n" + "=" * 78)
    print("FINAL SYNTHESIZED ANSWER:")
    print(result.final_answer or "(No text output)")
    print("=" * 78 + "\n")


def run_interactive(agent: Any) -> None:
    """Interactive CLI REPL for testing the agent."""
    print("\n" + "=" * 78)
    print(" 🪐 Tool-Calling Agent CLI (LangChain create_agent + LangGraph runtime)")
    print("=" * 78)
    print("Available Tools:")
    print(" • pdf_search(query): Internal Mars Odyssey PDF specs & ECLSS limits")
    print(" • site_search(query): Crawled website documentation & capabilities")
    print(" • db_query(query): Structured SQL database (customers, orders, products)")
    print(" • web_search(query): Live internet search (breaking news, 2026 updates)")
    print(" • calculator(operation, a, b): Arithmetic calculations")
    print(" • get_weather(city): Live city weather conditions")
    print("-" * 78)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Ask Agent > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting tool-calling agent CLI.")
                break

            result = run_agent_query(agent, query)
            print_trace_report(result)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting tool-calling agent CLI.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"\n[!] Error during agent execution: {exc}\n")


def main() -> None:
    """CLI entrypoint for tool-calling agent."""
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Tool RAG Agent CLI (Task 4.3)"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Single question to answer with trace."
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Launch interactive CLI loop."
    )
    parser.add_argument(
        "--tools",
        "-t",
        type=str,
        choices=["all", "rag", "basic"],
        default="all",
        help="Toolset to equip the agent with (default: all).",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="Model provider (google_genai, openai, mock).",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model identifier."
    )
    args = parser.parse_args()

    toolset_map = {
        "all": get_all_tools(),
        "rag": get_rag_tools(),
        "basic": BASIC_TOOLS,
    }
    selected_tools = toolset_map[args.tools]

    agent = build_tool_agent(
        tools=selected_tools,
        provider=args.provider,
        model_name=args.model,
    )

    if args.query:
        result = run_agent_query(
            agent,
            args.query,
            provider=args.provider,
            model_name=args.model,
        )
        print_trace_report(result)
    elif args.interactive or len(sys.argv) == 1:
        run_interactive(agent)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
