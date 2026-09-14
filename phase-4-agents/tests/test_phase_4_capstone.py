"""Phase 4 Capstone Integration Tests.

Validates the full Phase 4 Done criteria:
1. Agent answers over a Phase 3 knowledge base (via Model Context Protocol).
2. Takes an approval-gated write action (add_to_cart paused before execution).
3. Survives a process restart (persisted checkpoint reloaded in a fresh process,
   approval granted, write executed against database).
4. Streams its progress (Server-Sent Events emitting intermediate events).
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from phase_4_agents.config import get_postgres_checkpointer
from phase_4_agents.graph_agent import (
    build_state_graph_agent,
    handle_human_approval,
    stream_agent_steps,
)
from phase_4_agents.mcp_client import convert_mcp_to_langchain_tools
from phase_4_agents.mcp_server import create_ecommerce_mcp_server


@pytest.fixture(scope="module")
def postgres_available() -> bool:
    """Verify if PostgreSQL container is running and accessible."""
    try:
        with get_postgres_checkpointer() as checkpointer:
            checkpointer.setup()
        return True
    except Exception:
        return False


@pytest.fixture
def capstone_db(tmp_path: Any) -> str:
    """Create a temporary SQLite database with Phase 3 schema and sample data."""
    db_file = tmp_path / "capstone_ecommerce.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            city TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE cart_items (
            cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total_price REAL NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        INSERT INTO customers (name, email, city) VALUES
            ('Alice Smith', 'alice@example.com', 'San Francisco'),
            ('Bob Jones', 'bob@example.com', 'New York');

        INSERT INTO products (name, category, price, stock_quantity) VALUES
            ('MacBook Pro 16', 'Laptops', 2499.00, 10),
            ('Keychron K2 Keyboard', 'Accessories', 89.00, 50),
            ('Sony WH-1000XM5 Headphones', 'Audio', 399.00, 0);

        INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES
            (1, '2026-03-01', 'completed', 2499.00);
        """
    )
    conn.commit()
    conn.close()
    return str(db_file)


@pytest.fixture
def mcp_tools(capstone_db: str) -> list[Any]:
    """Provide LangChain tools converted from the Phase 3 Ecommerce MCP Server."""
    server = create_ecommerce_mcp_server(db_path=capstone_db)
    return convert_mcp_to_langchain_tools(server)


# =============================================================================
# 1. Phase 3 Knowledge Base Answering (Read-Only)
# =============================================================================


def test_capstone_answers_over_phase_3_knowledge_base(
    mcp_tools: list[Any],
) -> None:
    """Requirement 1: Agent answers questions over Phase 3 knowledge base via MCP."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            tools=mcp_tools,
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"capstone_kb_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    query = "What accessories or keyboard products do you have in the catalog?"
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    messages = list(result["messages"])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    # 1. Verify tool call was made to MCP tool
    assert len(tool_messages) >= 1
    assert "Keychron K2 Keyboard" in str(tool_messages[0].content)

    # 2. Verify synthesized final answer answers over the knowledge base
    final_ai = ai_messages[-1]
    assert "Keychron K2 Keyboard" in str(final_ai.content)

    # 3. Read query did not pause on human approval
    state = app.get_state(config)
    assert state.next == ()


# =============================================================================
# 2. Approval-Gated Write Action (add_to_cart)
# =============================================================================


def test_capstone_takes_approval_gated_write_action(
    mcp_tools: list[Any], capstone_db: str
) -> None:
    """Requirement 2: Write actions touching client data pause at human_approval."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            tools=mcp_tools,
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"capstone_write_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    query = "Add 2 Keychron K2 Keyboards to my cart"
    app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    # Graph MUST pause immediately before human_approval node
    state = app.get_state(config)
    assert state.next == ("human_approval",)

    # Verify write tool was planned
    last_msg = state.values["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert getattr(last_msg, "tool_calls", None)
    tool_call = last_msg.tool_calls[0]
    assert tool_call["name"] == "add_to_cart"
    assert tool_call["args"]["quantity"] == 2

    # Verify write has NOT executed yet (database is untouched before approval)
    conn = sqlite3.connect(capstone_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cart_items")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0


# =============================================================================
# 3. Process Interruption & Resumption (Survives Restart)
# =============================================================================


def test_capstone_survives_process_restart(
    mcp_tools: list[Any],
    capstone_db: str,
    postgres_available: bool,
) -> None:
    """Requirement 3: State persists across restart and resumes upon approval."""
    if not postgres_available:
        pytest.skip("PostgreSQL container on port 5433 not accessible")

    thread_id = f"capstone_restart_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # =========================================================================
    # PROCESS 1: Start write request, pause at approval gate, and shutdown
    # =========================================================================
    with get_postgres_checkpointer() as cp1:
        app1 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                tools=mcp_tools,
                checkpointer=cp1,
            ),
        )

        app1.invoke(
            {
                "messages": [
                    HumanMessage(content="Add 2 Keychron K2 Keyboards to my cart")
                ]
            },
            config=config,
        )

        state1 = app1.get_state(config)
        assert state1.next == ("human_approval",)

    # Process 1 context has exited. Connection closed. Process killed.

    # =========================================================================
    # PROCESS 2: Fresh instance, connect to checkpointer, reload state, approve
    # =========================================================================
    with get_postgres_checkpointer() as cp2:
        app2 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                tools=mcp_tools,
                checkpointer=cp2,
            ),
        )

        # 1. Verify state survived process restart intact
        reloaded_state = app2.get_state(config)
        assert reloaded_state is not None
        assert reloaded_state.next == ("human_approval",)

        # 2. Grant human approval to execute the pending write action
        res = handle_human_approval(app2, config, approved=True)

        # 3. Verify graph ran to completion
        final_state = app2.get_state(config)
        assert final_state.next == ()

        # 4. Verify write action was executed against the database
        conn = sqlite3.connect(capstone_db)
        cur = conn.cursor()
        cur.execute("SELECT quantity, unit_price FROM cart_items WHERE customer_id = 1")
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 2
        assert row[1] == 89.00

        # 5. Verify final AI response confirmed the action
        messages = list(res["messages"])
        assert any(
            isinstance(m, ToolMessage) and "Successfully added" in str(m.content)
            for m in messages
        )


# =============================================================================
# 4. Real-Time Streaming of Execution Steps
# =============================================================================


def test_capstone_streams_progress(mcp_tools: list[Any]) -> None:
    """Requirement 4: Agent streams intermediate execution events in real time."""
    checkpointer = MemorySaver()
    app = cast(
        CompiledStateGraph,
        build_state_graph_agent(
            provider="mock",
            tools=mcp_tools,
            checkpointer=checkpointer,
        ),
    )

    thread_id = f"capstone_stream_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    events = list(
        stream_agent_steps(
            app,
            "What accessories products do you have?",
            config=config,
        )
    )

    event_types = [e.event for e in events]

    # Verify essential streaming sequence
    assert "step_start" in event_types
    assert "tool_decision" in event_types
    assert "tool_execution" in event_types
    assert "final_answer" in event_types
    assert "done" in event_types

    # Verify monotonically non-decreasing steps
    step_nums = [e.step_number for e in events]
    assert step_nums == sorted(step_nums)

    # Verify payload content
    final_event = next(e for e in events if e.event == "final_answer")
    assert final_event.data.get("content")


# =============================================================================
# 5. End-to-End Capstone Orchestration (All 4 Criteria in One Workflow)
# =============================================================================


def test_capstone_end_to_end_orchestration(
    mcp_tools: list[Any],
    capstone_db: str,
    postgres_available: bool,
) -> None:
    """Unified Capstone Test:
    1. Answers over Phase 3 KB while streaming progress.
    2. Issues approval-gated write action (add_to_cart).
    3. Stream detects approval pause and halts.
    4. Process terminates and restarts in fresh connection.
    5. State reloads from Postgres checkpointer, human approves, and write executes.
    """
    if not postgres_available:
        pytest.skip("PostgreSQL container on port 5433 not accessible")

    thread_id = f"capstone_e2e_{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # =========================================================================
    # STAGE 1: Process 1 streams read query over Phase 3 knowledge base
    # =========================================================================
    with get_postgres_checkpointer() as cp1:
        app1 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                tools=mcp_tools,
                checkpointer=cp1,
            ),
        )

        read_events = list(
            stream_agent_steps(
                app1,
                "What keyboards and accessories are available in the catalog?",
                config=config,
            )
        )
        assert any(e.event == "final_answer" for e in read_events)

        # STAGE 2: Request approval-gated write action (add_to_cart)
        write_events = list(
            stream_agent_steps(
                app1,
                "Add 2 Keychron K2 Keyboards to my cart",
                config=config,
            )
        )
        # Stream must signal that execution is paused for approval
        write_types = [e.event for e in write_events]
        assert "approval_paused" in write_types

        # Verify state is paused in checkpointer
        state1 = app1.get_state(config)
        assert state1.next == ("human_approval",)

    # Process 1 shuts down.

    # =========================================================================
    # STAGE 3: Process 2 boots, connects to Postgres, approves, and executes write
    # =========================================================================
    with get_postgres_checkpointer() as cp2:
        app2 = cast(
            CompiledStateGraph,
            build_state_graph_agent(
                provider="mock",
                tools=mcp_tools,
                checkpointer=cp2,
            ),
        )

        # Reload state across process restart
        reloaded_state = app2.get_state(config)
        assert reloaded_state.next == ("human_approval",)

        # Human approves write
        resume_res = handle_human_approval(app2, config, approved=True)

        # Finished to END
        final_state = app2.get_state(config)
        assert final_state.next == ()

        # Verify database mutation
        conn = sqlite3.connect(capstone_db)
        cur = conn.cursor()
        cur.execute(
            "SELECT quantity, unit_price, product_name "
            "FROM cart_items WHERE customer_id = 1"
        )
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 2
        assert row[1] == 89.00
        assert row[2] == "Keychron K2 Keyboard"

        # Verify final confirmation
        messages = list(resume_res["messages"])
        assert any(
            isinstance(m, ToolMessage) and "Successfully added" in str(m.content)
            for m in messages
        )
