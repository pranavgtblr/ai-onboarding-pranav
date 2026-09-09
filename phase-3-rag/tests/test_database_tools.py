"""Unit and integration tests for Task 3.15: Structured Database Tool Calling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from phase_3_rag.database import init_database
from phase_3_rag.database_tools import (
    execute_database_tool,
    get_appointments,
    get_customer,
    get_database_tool_declarations,
    get_order_details,
    get_orders,
    run_database_tool_loop,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Initialize a dedicated SQLite test database."""
    db_file = tmp_path / "test_ecommerce.db"
    return init_database(db_file)


# ---------------------------------------------------------------------------
# Unit Tests for Typed Query Functions
# ---------------------------------------------------------------------------


def test_get_customer_lookup(test_db: Path):
    """Verify customer lookup by ID, partial name, and email."""
    # Lookup by name substring
    res_name = get_customer(name="Alice", db_path=test_db)
    assert res_name["count"] == 1
    assert res_name["customers"][0]["name"] == "Alice Johnson"
    customer_id = res_name["customers"][0]["customer_id"]

    # Lookup by exact ID
    res_id = get_customer(customer_id=customer_id, db_path=test_db)
    assert res_id["count"] == 1
    assert res_id["customers"][0]["email"] == "alice@example.com"

    # Lookup by email
    res_email = get_customer(email="bob@example.com", db_path=test_db)
    assert res_email["count"] == 1
    assert res_email["customers"][0]["name"] == "Bob Smith"

    # Lookup without parameters returns helpful error
    res_empty = get_customer(db_path=test_db)
    assert "error" in res_empty


def test_get_appointments_filters(test_db: Path):
    """Verify appointment retrieval by customer ID, date ranges, and status."""
    # All appointments for customer 1
    all_alice = get_appointments(customer_id=1, db_path=test_db)
    assert all_alice["count"] == 2
    services = [a["service_type"] for a in all_alice["appointments"]]
    assert "Hardware Diagnostics" in services
    assert "Cloud Architecture Review" in services

    # Date range filter (excluding later appointment on 2026-09-20)
    early_alice = get_appointments(
        customer_id=1,
        from_date="2026-09-01",
        to_date="2026-09-15",
        db_path=test_db,
    )
    assert early_alice["count"] == 1
    assert early_alice["appointments"][0]["service_type"] == "Hardware Diagnostics"

    # Status filter
    confirmed_alice = get_appointments(
        customer_id=1, status="confirmed", db_path=test_db
    )
    assert confirmed_alice["count"] == 1
    assert confirmed_alice["appointments"][0]["status"] == "confirmed"

    scheduled_alice = get_appointments(
        customer_id=1, status="scheduled", db_path=test_db
    )
    assert scheduled_alice["count"] == 1
    assert scheduled_alice["appointments"][0]["status"] == "scheduled"


def test_get_orders_and_details(test_db: Path):
    """Verify order retrieval and deep line-item detail lookup."""
    # Customer 1 has 2 orders
    orders_res = get_orders(customer_id=1, db_path=test_db)
    assert orders_res["count"] == 2

    # Status filter
    shipped_orders = get_orders(customer_id=1, status="shipped", db_path=test_db)
    assert shipped_orders["count"] == 1
    assert shipped_orders["orders"][0]["total_amount"] == 89.00

    # Order details lookup
    order_id = orders_res["orders"][0]["order_id"]
    details = get_order_details(order_id=order_id, db_path=test_db)
    assert details["customer_name"] == "Alice Johnson"
    assert "items" in details
    assert len(details["items"]) > 0

    # Non-existent order
    bad_order = get_order_details(order_id=99999, db_path=test_db)
    assert "error" in bad_order


# ---------------------------------------------------------------------------
# Security & Parameterization Tests
# ---------------------------------------------------------------------------


def test_parameterization_prevents_sql_injection(test_db: Path):
    """Verify malicious SQL payloads in arguments are treated as raw literals."""
    injection_payload = "' OR '1'='1"

    # Attack on get_customer name parameter
    res = get_customer(name=injection_payload, db_path=test_db)
    assert res["count"] == 0  # Should NOT return all customers!

    # Attack on get_appointments status parameter
    bad_status = "confirmed' OR 1=1 --"
    res_appt = get_appointments(customer_id=1, status=bad_status, db_path=test_db)
    assert res_appt["count"] == 0


def test_execute_database_tool_dispatch(test_db: Path):
    """Verify dispatcher safely runs known tools and catches unknown/bad calls."""
    # Valid dispatch
    res = execute_database_tool("get_customer", {"name": "Bob"}, db_path=test_db)
    assert res["count"] == 1
    assert res["customers"][0]["name"] == "Bob Smith"

    # Unknown tool
    unknown_res = execute_database_tool("drop_tables", {}, db_path=test_db)
    assert "error" in unknown_res
    assert "not recognized" in unknown_res["error"]

    # Invalid arguments (missing required parameter or bad type)
    bad_args_res = execute_database_tool(
        "get_appointments", {"invalid_param": 123}, db_path=test_db
    )
    assert "error" in bad_args_res


def test_tool_declarations_schema():
    """Verify Gemini tool declarations contain all required function schemas."""
    declarations = get_database_tool_declarations()
    assert len(declarations) == 1
    funcs = declarations[0]["function_declarations"]
    func_names = [f["name"] for f in funcs]

    assert "get_customer" in func_names
    assert "get_appointments" in func_names
    assert "get_orders" in func_names
    assert "get_order_details" in func_names


# ---------------------------------------------------------------------------
# Multi-Turn Tool Calling Loop Integration Tests
# ---------------------------------------------------------------------------


def test_tool_calling_loop_mocked_execution(test_db: Path):
    """Verify multi-turn tool calling loop: calls get_customer, then appointments."""
    # Turn 1: Model asks for get_customer(name="Alice")
    step1_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_customer",
                                "args": {"name": "Alice"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    # Turn 2: Model receives customer info and asks for get_appointments(customer_id=1)
    step2_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_appointments",
                                "args": {"customer_id": 1, "status": "confirmed"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    # Turn 3: Model produces final conversational answer
    step3_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Alice Johnson has a confirmed appointment for "
                                "Hardware Diagnostics on 2026-09-12 at 11:00 AM."
                            )
                        }
                    ]
                }
            }
        ]
    }

    call_count = 0

    def mock_post_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=step1_response)
        elif call_count == 2:
            return httpx.Response(200, json=step2_response)
        else:
            return httpx.Response(200, json=step3_response)

    client = httpx.Client(transport=httpx.MockTransport(mock_post_handler))

    with patch("phase_3_rag.database_tools.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="mock_key",
            gemini_model="gemini-3.5-flash-lite",
            timeout_seconds=30.0,
        )
        result = run_database_tool_loop(
            "What is Alice's next appointment?",
            client=client,
            db_path=test_db,
        )

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "get_customer"
    assert result.tool_calls[1].tool_name == "get_appointments"
    assert "Hardware Diagnostics" in result.final_answer
    assert not result.hit_max_guard


def test_tool_calling_loop_direct_answer(test_db: Path):
    """Verify pipeline handles cases where model produces direct text."""
    direct_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Hello! How can I assist you with your account today?"
                            )
                        }
                    ]
                }
            }
        ]
    }

    def direct_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=direct_response)

    client = httpx.Client(transport=httpx.MockTransport(direct_handler))

    with patch("phase_3_rag.database_tools.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="mock_key",
            gemini_model="gemini-3.5-flash-lite",
            timeout_seconds=30.0,
        )
        result = run_database_tool_loop(
            "Hello there!",
            client=client,
            db_path=test_db,
        )

    assert len(result.tool_calls) == 0
    assert "Hello! How can I assist you" in result.final_answer
    assert not result.hit_max_guard


def test_tool_calling_loop_max_guard(test_db: Path):
    """Verify tool loop halts when reaching max_iterations without terminating."""
    looping_tool_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_customer",
                                "args": {"name": "Alice"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=looping_tool_response)
        )
    )

    with patch("phase_3_rag.database_tools.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="mock_key",
            gemini_model="gemini-3.5-flash-lite",
            timeout_seconds=30.0,
        )
        result = run_database_tool_loop(
            "Loop forever please",
            client=client,
            db_path=test_db,
            max_iterations=3,
        )

    assert result.hit_max_guard is True
    assert "maximum tool execution iterations" in result.final_answer
