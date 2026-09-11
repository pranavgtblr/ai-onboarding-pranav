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
from langchain_core.runnables import RunnableConfig

from phase_4_agents.config import get_chat_model, get_settings
from phase_4_agents.graph_agent import (
    build_state_graph_agent,
    handle_human_approval,
    list_state_history,
    time_travel_replay,
)
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

    query: str | None
    steps: list[AgentStepTrace]
    final_answer: str
    total_steps: int
    provider: str = "google_genai"
    model: str = "gemini-3.5-flash-lite"


def build_tool_agent(
    *,
    engine: str = "state_graph",
    tools: list[Any] | None = None,
    system_prompt: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
):
    """Construct a tool-calling agent using StateGraph (4.4) or create_agent (4.1)."""
    if engine.lower() == "state_graph":
        return build_state_graph_agent(
            tools=tools,
            system_prompt=system_prompt,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
        )

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
    query: str | None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    config: RunnableConfig | None = None,
) -> AgentExecutionResult:
    """Run a query through the agent, capturing the step-by-step trace."""
    invoke_config = config or {}
    if query is not None:
        invoke_input: Any = {"messages": [{"role": "user", "content": query}]}
    else:
        # Resuming an interrupted run without new input
        invoke_input = None

    response = agent.invoke(invoke_input, config=invoke_config)
    messages: list[BaseMessage] = response.get("messages", []) if response else []

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
        "--engine",
        "-e",
        type=str,
        choices=["state_graph", "create_agent"],
        default="state_graph",
        help="Agent execution engine: 'state_graph' (Task 4.4) or 'create_agent'.",
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
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default-session",
        help="Session thread identifier for persistent checkpointing.",
    )
    parser.add_argument(
        "--checkpointer",
        choices=["postgres", "memory", "none"],
        default="postgres",
        help="Persistence checkpointer (default: postgres).",
    )
    parser.add_argument(
        "--simulate-kill",
        action="store_true",
        help="Simulate process termination mid-run (interrupts before tool execution).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution of an interrupted thread from its checkpoint.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="List checkpoint history for thread (Task 4.8 Time Travel).",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Target checkpoint ID for time travel inspection or replay.",
    )
    parser.add_argument(
        "--modify-query",
        type=str,
        default=None,
        help="Modified query text to inject when replaying from a checkpoint.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay execution from --checkpoint-id with optional --modify-query.",
    )
    args = parser.parse_args()

    toolset_map = {
        "all": get_all_tools(),
        "rag": get_rag_tools(),
        "basic": BASIC_TOOLS,
    }
    selected_tools = toolset_map[args.tools]

    # Configure checkpointer
    checkpointer_ctx: Any = None
    active_checkpointer: Any = None
    if args.checkpointer == "postgres":
        try:
            from phase_4_agents.config import get_postgres_checkpointer

            checkpointer_ctx = get_postgres_checkpointer()
            active_checkpointer = checkpointer_ctx.__enter__()
        except Exception as exc:
            print(
                f"[Checkpointer Warning] Could not connect to Postgres ({exc}). "
                "Falling back to MemorySaver."
            )
            from langgraph.checkpoint.memory import MemorySaver

            active_checkpointer = MemorySaver()
    elif args.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        active_checkpointer = MemorySaver()

    interrupt_nodes = ["execute_tools"] if args.simulate_kill else None

    try:
        agent = build_tool_agent(
            engine=args.engine,
            tools=selected_tools,
            provider=args.provider,
            model_name=args.model,
            checkpointer=active_checkpointer,
            interrupt_before=interrupt_nodes,
        )

        run_config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}

        if args.simulate_kill:
            if not args.query:
                print("Error: --simulate-kill requires --query to start a run.")
                sys.exit(1)
            print(f"[Process 1] Starting run on thread '{args.thread_id}'...")
            print(
                "[Process 1] Executing until interrupt point (before tool execution)..."
            )
            result = run_agent_query(
                agent,
                args.query,
                provider=args.provider,
                model_name=args.model,
                config=run_config,
            )
            print_trace_report(result)
            print("\n=================================================================")
            msg = (
                f"💥 [PROCESS KILLED / INTERRUPTED] Thread '{args.thread_id}' "
                f"saved to {args.checkpointer.upper()}!"
            )
            print(msg)
            print("To resume this exact execution in a new process, run:")
            print(
                f"  uv run python src/phase_4_agents/agent_cli.py "
                f"--thread-id {args.thread_id} --resume"
            )
            print("=================================================================")
            return

        if args.resume:
            print(
                f"[Process 2] Connecting to {args.checkpointer.upper()} checkpointer..."
            )
            print(f"[Process 2] Loading checkpoint for thread '{args.thread_id}'...")
            saved_state = agent.get_state(run_config)
            if not saved_state or not saved_state.values:
                print(f"Error: No checkpoint found for thread '{args.thread_id}'.")
                sys.exit(1)

            print(f"[Process 2] Found checkpoint! Next node to run: {saved_state.next}")
            print("[Process 2] Resuming execution from checkpoint...")
            result = run_agent_query(
                agent,
                None,  # Resume with no new input
                provider=args.provider,
                model_name=args.model,
                config=run_config,
            )
            print_trace_report(result)
            done_msg = (
                f"\n✅ [RUN RESUMED & COMPLETED] Finished execution for thread "
                f"'{args.thread_id}'!"
            )
            print(done_msg)
            return

        if args.history:
            print(f"\n⏳ [CHECKPOINT HISTORY] Thread '{args.thread_id}':")
            history = list_state_history(agent, run_config)
            if not history:
                print(f"No checkpoint history found for thread '{args.thread_id}'.")
            else:
                hdr = (
                    f"{'Idx':<4} {'Checkpoint ID':<38} {'Next Node':<18} "
                    f"{'Msgs':<6} {'Last Message'}"
                )
                print(hdr)
                print("-" * 88)
                for item in history:
                    cid = item["checkpoint_id"]
                    nxt = str(item["next"])
                    mc = item["messages_count"]
                    prev = item["last_message_preview"].replace("\n", " ")[:24]
                    idx = item["index"]
                    print(f"{idx:<4} {cid:<38} {nxt:<18} {mc:<6} {prev}")
            return

        if args.replay:
            if not args.checkpoint_id:
                print("Error: --replay requires --checkpoint-id <UUID>.")
                sys.exit(1)

            print(
                f"\n⏳ [TIME TRAVEL] Replaying thread '{args.thread_id}' "
                f"from checkpoint '{args.checkpoint_id}'..."
            )
            state_update = None
            if args.modify_query:
                print(f"🔄 Injecting modified query: '{args.modify_query}'")
                state_update = {"messages": [HumanMessage(content=args.modify_query)]}

            fork_cfg, raw_result = time_travel_replay(
                agent,
                run_config,
                args.checkpoint_id,
                state_update=state_update,
            )
            fork_cid = fork_cfg.get("configurable", {}).get("checkpoint_id", "")
            print(f"🌱 Created new checkpoint fork: {fork_cid}")

            messages = raw_result.get("messages", [])
            step_traces: list[AgentStepTrace] = []
            final_ans = ""
            for s_idx, msg in enumerate(messages, start=1):
                m_type = type(msg).__name__
                txt = extract_text_from_content(msg.content)
                if isinstance(msg, HumanMessage):
                    step_traces.append(
                        AgentStepTrace(
                            step_number=s_idx,
                            actor="User",
                            message_type=m_type,
                            content=txt,
                        )
                    )
                elif isinstance(msg, AIMessage):
                    t_calls = getattr(msg, "tool_calls", [])
                    if t_calls:
                        step_traces.append(
                            AgentStepTrace(
                                step_number=s_idx,
                                actor="Model (Tool Decision)",
                                message_type=m_type,
                                content=txt,
                                tool_calls=list(t_calls),
                            )
                        )
                    else:
                        final_ans = txt
                        step_traces.append(
                            AgentStepTrace(
                                step_number=s_idx,
                                actor="Model (Final Answer)",
                                message_type=m_type,
                                content=txt,
                            )
                        )
                elif isinstance(msg, ToolMessage):
                    step_traces.append(
                        AgentStepTrace(
                            step_number=s_idx,
                            actor="Tool Execution",
                            message_type=m_type,
                            content=txt,
                        )
                    )

            replay_result = AgentExecutionResult(
                query=args.modify_query or "(Replayed from checkpoint)",
                steps=step_traces,
                final_answer=final_ans,
                total_steps=len(step_traces),
                provider=args.provider or "default",
                model=args.model or "default",
            )
            print_trace_report(replay_result)
            return

        if args.query:
            result = run_agent_query(
                agent,
                args.query,
                provider=args.provider,
                model_name=args.model,
                config=run_config,
            )
            print_trace_report(result)

            # Check if execution paused on human_approval gate
            current_state = agent.get_state(run_config)
            if current_state and "human_approval" in current_state.next:
                print(
                    "\n================================================================="
                )
                print("🛑 [HUMAN APPROVAL REQUIRED FOR CLIENT WRITE ACTION]")
                print(
                    "================================================================="
                )
                pending_tools: list[str] = []
                messages = current_state.values.get("messages", [])
                if messages and isinstance(messages[-1], AIMessage):
                    for tc in getattr(messages[-1], "tool_calls", []):
                        pending_tools.append(f"{tc.get('name')}({tc.get('args')})")
                print("Pending write action(s) touching client data:")
                for pt in pending_tools:
                    print(f"  • {pt}")

                try:
                    prompt_str = "\nApprove this write action? [y/N]: "
                    choice = input(prompt_str).strip().lower()
                except EOFError:
                    choice = "y"

                approved = choice in ("y", "yes")
                if approved:
                    print("✅ Action Approved! Resuming execution...")
                    final_res = run_agent_query(
                        agent,
                        None,
                        provider=args.provider,
                        model_name=args.model,
                        config=run_config,
                    )
                    print_trace_report(final_res)
                else:
                    print("❌ Action Denied! Aborting write operation...")
                    handle_human_approval(
                        agent,
                        run_config,
                        approved=False,
                        rejection_reason="Declined by user in CLI.",
                    )
                    final_res = run_agent_query(
                        agent,
                        None,
                        provider=args.provider,
                        model_name=args.model,
                        config=run_config,
                    )
                    print_trace_report(final_res)

        elif args.interactive or len(sys.argv) == 1:
            run_interactive(agent)
        else:
            parser.print_help()
    finally:
        if checkpointer_ctx is not None:
            checkpointer_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
