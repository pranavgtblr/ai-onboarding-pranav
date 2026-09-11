"""Tests for Task 4.10: Tool Design Pass, Actionable Errors, and Guardrails.

Verifies:
1. Tight Pydantic schemas (type safety, forbidden extras, boundary validation).
2. Actionable error messages written specifically for the model to self-correct.
3. Hard iteration limits that fail loudly instead of running indefinitely.
4. Token budget & cost caps that fail loudly upon budget overrun.
5. Stuck loop detection preventing ping-ponging and repeated identical calls.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from phase_4_agents.graph_agent import (
    AgentCostCapExceededError,
    AgentMaxIterationsExceededError,
    AgentState,
    AgentStuckError,
    build_state_graph_agent,
    detect_stuck_agent,
)
from phase_4_agents.rag_tools import (
    add_to_cart,
    db_query,
    pdf_search,
)
from phase_4_agents.schemas import (
    AddToCartInput,
    CalculatorInput,
    DatabaseQueryInput,
    PDFSearchInput,
    SiteSearchInput,
    WeatherInput,
    WebSearchInput,
)
from phase_4_agents.tools import calculator

# =============================================================================
# 1. Tight Pydantic Schemas Validation
# =============================================================================


def test_schema_valid_inputs_pass() -> None:
    """Verify standard valid inputs instantiate correctly across all schemas."""
    pdf_in = PDFSearchInput(query="Mars ECLSS operating limits")
    assert pdf_in.query == "Mars ECLSS operating limits"

    site_in = SiteSearchInput(query="Toobler cloud capabilities")
    assert site_in.query == "Toobler cloud capabilities"

    db_in = DatabaseQueryInput(query="SELECT * FROM customers LIMIT 5")
    assert "customers" in db_in.query

    cart_in = AddToCartInput(
        product_name="Titanium Drill Bit", quantity=2, customer_id=1
    )
    assert cart_in.quantity == 2

    web_in = WebSearchInput(query="Artemis 2026 roadmap")
    assert "Artemis" in web_in.query

    calc_in = CalculatorInput(operation="multiply", a=12.5, b=4.0)
    assert calc_in.operation == "multiply"

    weather_in = WeatherInput(city="Tokyo")
    assert weather_in.city == "Tokyo"


def test_schema_rejects_extra_fields() -> None:
    """Verify schemas enforce extra='forbid' to prevent LLM argument hallucination."""
    with pytest.raises(ValidationError):
        AddToCartInput.model_validate(
            {
                "product_name": "Titanium Drill Bit",
                "quantity": 1,
                "hallucinated_param": True,
            }
        )

    with pytest.raises(ValidationError):
        DatabaseQueryInput.model_validate(
            {"query": "SELECT * FROM orders", "format": "markdown"}
        )


def test_schema_enforces_bounds_and_constraints() -> None:
    """Verify boundary checks: quantities, string lengths, and enum operations."""
    # AddToCart: quantity must be >= 1 and <= 100
    with pytest.raises(ValidationError):
        AddToCartInput(product_name="Mars Rover Sensor", quantity=0)

    with pytest.raises(ValidationError):
        AddToCartInput(product_name="Mars Rover Sensor", quantity=101)

    # Calculator: operation must have at least 1 character
    with pytest.raises(ValidationError):
        CalculatorInput.model_validate({"operation": "", "a": 2.0, "b": 3.0})

    # PDFSearch: query must be at least 2 characters
    with pytest.raises(ValidationError):
        PDFSearchInput(query="a")


# =============================================================================
# 2. Actionable Error Messages Written for the Model
# =============================================================================


def test_db_query_actionable_error_on_forbidden_sql() -> None:
    """Verify db_query returns guidance directing model away from mutations."""
    res = db_query.invoke({"query": "DROP TABLE customers;"})
    assert "Only read-only SELECT queries are allowed" in res
    assert "Write operation 'DROP' is strictly forbidden" in res
    assert "add_to_cart" in res


def test_db_query_actionable_error_on_missing_table() -> None:
    """Verify db_query returns valid table names and column hints upon error."""
    res = db_query.invoke({"query": "SELECT * FROM non_existent_inventory_table;"})
    assert "Database Query Error" in res
    assert "Valid tables in this database are 'customers', 'orders'" in res
    assert "customers(customer_id, name, email, phone, city)" in res


def test_add_to_cart_actionable_error_on_missing_product() -> None:
    """Verify add_to_cart returns available products when product is not found."""
    res = add_to_cart.invoke({"product_name": "Antimatter Battery", "quantity": 1})
    assert (
        "Catalog Error: Product 'Antimatter Battery' not found in catalog" in res
    )
    assert "Available products are: 'Mars Rover Sensor' ($450.00)" in res
    assert "'Titanium Drill Bit' ($120.00)" in res


def test_calculator_actionable_error_on_division_by_zero() -> None:
    """Verify calculator returns clear guidance when dividing by zero."""
    res = calculator.invoke({"operation": "divide", "a": 100, "b": 0})
    assert "Calculator Error: Division by zero is mathematically undefined" in res
    assert "Action for model" in res


def test_pdf_search_actionable_miss_message() -> None:
    """Verify pdf_search suggests alternative keywords and boundaries on misses."""
    res = pdf_search.invoke({"query": "nonexistent alien biotechnology specs"})
    assert "No relevant PDF documents matched the query" in res
    assert "Action for model: Check spelling or try broader keywords" in res
    assert "DO NOT query company capabilities or live news here" in res


# =============================================================================
# 3. Stuck Loop Detection
# =============================================================================


def test_detect_stuck_agent_consecutive_identical_calls() -> None:
    """Verify detect_stuck_agent catches identical back-to-back tool calls."""
    history = [
        {"name": "db_query", "args": {"query": "SELECT * FROM users"}, "step": 1},
        {"name": "db_query", "args": {"query": "SELECT * FROM users"}, "step": 2},
    ]
    is_stuck, reason = detect_stuck_agent(history)
    assert is_stuck is True
    assert "Detected consecutive identical tool call: 'db_query'" in reason


def test_detect_stuck_agent_cyclic_repeated_calls() -> None:
    """Verify detect_stuck_agent catches cyclic repetition appearing >= 3 times."""
    history = [
        {"name": "db_query", "args": {"query": "customers"}, "step": 1},
        {"name": "web_search", "args": {"query": "news"}, "step": 2},
        {"name": "db_query", "args": {"query": "customers"}, "step": 3},
        {"name": "web_search", "args": {"query": "news"}, "step": 4},
        {"name": "db_query", "args": {"query": "customers"}, "step": 5},
    ]
    is_stuck, reason = detect_stuck_agent(history)
    assert is_stuck is True
    assert "Detected cyclic loop" in reason


def test_detect_stuck_agent_progressing_calls_not_stuck() -> None:
    """Verify changing arguments do not trigger stuck detector."""
    history = [
        {
            "name": "db_query",
            "args": {"query": "Find customers living in Atlantis"},
            "step": 1,
        },
        {"name": "db_query", "args": {"query": "customers"}, "step": 2},
    ]
    is_stuck, reason = detect_stuck_agent(history)
    assert is_stuck is False
    assert reason == ""


# =============================================================================
# 4. End-to-End StateGraph Guardrails (Fail Loudly)
# =============================================================================


def test_agent_iteration_limit_fails_loudly() -> None:
    """Verify agent halts when step count exceeds iteration limit."""
    # Build agent with aggressive iteration limit of 2 steps
    agent = build_state_graph_agent(
        provider="mock",
        max_iterations=2,
        fail_loudly=True,
    )
    # A query requiring tool call + execution + synthesis normally takes >= 3 steps
    initial_state: AgentState = {
        "messages": [HumanMessage(content="What is 25 multiplied by 4?")],
        "step_count": 0,
    }

    with pytest.raises(AgentMaxIterationsExceededError) as exc_info:
        agent.invoke(initial_state)

    assert "Reached maximum iteration limit of 2 steps" in str(exc_info.value)
    assert "Failing loudly" in str(exc_info.value)


def test_agent_cost_cap_fails_loudly() -> None:
    """Verify agent halts when token usage exceeds cost budget cap."""
    # Build agent with near-zero dollar budget cap ($0.000000001)
    agent = build_state_graph_agent(
        provider="mock",
        cost_cap_usd=0.000000001,
        fail_loudly=True,
    )
    initial_state: AgentState = {
        "messages": [HumanMessage(content="Hello there! Can you assist me?")],
        "step_count": 0,
    }

    with pytest.raises(AgentCostCapExceededError) as exc_info:
        agent.invoke(initial_state)

    assert "exceeded cost cap" in str(exc_info.value)
    assert "Failing loudly" in str(exc_info.value)


def test_agent_stuck_loop_fails_loudly() -> None:
    """Verify agent halts with AgentStuckError when looping on tool calls."""
    # Simulate a stuck loop by pre-populating tool_call_history with an identical call
    # and invoking on a query that reproduces the same tool call
    agent = build_state_graph_agent(
        provider="mock",
        fail_loudly=True,
    )
    initial_state: AgentState = {
        "messages": [HumanMessage(content="What is the weather in Tokyo?")],
        "step_count": 1,
        "tool_call_history": [
            {"name": "get_weather", "args": {"city": "Tokyo"}, "step": 1},
        ],
    }

    with pytest.raises(AgentStuckError) as exc_info:
        agent.invoke(initial_state)

    assert "Stuck in loop" in str(exc_info.value)
    assert "get_weather" in str(exc_info.value)
