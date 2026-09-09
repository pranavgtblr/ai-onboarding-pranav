"""Comprehensive penetration tests for Task 3.16: Row-Level Access Control (RLS).

Simulates User A (Alice Johnson, customer_id=1) attempting to access User B's
(Bob Smith, customer_id=2) records across 5 distinct attack vectors:
1. Direct Request
2. Role-Play / Administrator Impersonation Jailbreak
3. Translation / Multilingual Obfuscation
4. Indirect Reference (city, timestamp, or order inference)
5. Enumeration across Joined Tables (order_items, order IDs)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from phase_3_rag.database import init_database
from phase_3_rag.database_tools import (
    execute_database_tool,
    get_appointments,
    get_customer,
    get_order_details,
    get_orders,
)
from phase_3_rag.text_to_sql import (
    SecurityValidationError,
    execute_readonly_sql,
    run_text_to_sql_pipeline,
    validate_and_sanitize_sql,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Initialize a dedicated SQLite test database."""
    db_file = tmp_path / "test_rls_ecommerce.db"
    return init_database(db_file)


# ---------------------------------------------------------------------------
# Vector 1: Direct Request Attacks
# ---------------------------------------------------------------------------


def test_vector1_direct_request_structured_tools(test_db: Path):
    """User A directly asks for User B's appointments, orders, or profile."""
    # Alice (session=1) attempts to query Bob's customer record (ID 2)
    res_cust_id = get_customer(session_customer_id=1, customer_id=2, db_path=test_db)
    assert "Access denied" in res_cust_id["error"]

    # Alice attempts to query Bob's profile by name
    res_cust_name = get_customer(session_customer_id=1, name="Bob", db_path=test_db)
    assert "Access denied" in res_cust_name["error"]

    # Alice attempts to query Bob's appointments directly
    res_appt = get_appointments(session_customer_id=1, customer_id=2, db_path=test_db)
    assert "Access denied" in res_appt["error"]

    # Alice attempts to query Bob's orders directly
    res_orders = get_orders(session_customer_id=1, customer_id=2, db_path=test_db)
    assert "Access denied" in res_orders["error"]


def test_vector1_direct_request_text_to_sql(test_db: Path):
    """User A uses Text-to-SQL with an explicit target of customer_id 2."""
    # Direct equality targeting Bob
    with pytest.raises(SecurityValidationError) as exc1:
        validate_and_sanitize_sql(
            "SELECT * FROM orders WHERE customer_id = 2",
            session_customer_id=1,
        )
    assert "Row-level security violation" in str(exc1.value)

    # Direct IN targeting Bob
    with pytest.raises(SecurityValidationError) as exc2:
        validate_and_sanitize_sql(
            "SELECT * FROM appointments WHERE customer_id IN (2, 3)",
            session_customer_id=1,
        )
    assert "Row-level security violation" in str(exc2.value)


# ---------------------------------------------------------------------------
# Vector 2: Role-Play / Admin Impersonation Jailbreak
# ---------------------------------------------------------------------------


def test_vector2_roleplay_jailbreak_text_to_sql(test_db: Path):
    """User A pretends to be admin requesting an unrestricted SELECT orders.

    Even if the LLM writes an unqualified query without WHERE clause, the
    tenant-isolated connection shadows the 'orders' table to only customer 1.
    """
    unrestricted_sql = "SELECT order_id, customer_id, total_amount FROM orders"

    # Execute as Alice (session_customer_id=1)
    _, rows = execute_readonly_sql(
        unrestricted_sql,
        session_customer_id=1,
        db_path=test_db,
    )

    # Alice should ONLY see her 2 orders, never Bob's or Charlie's
    assert len(rows) == 2
    for r in rows:
        assert r["customer_id"] == 1


def test_vector2_roleplay_jailbreak_structured_tools(test_db: Path):
    """User A role-plays as admin to list all appointments."""
    res = get_appointments(session_customer_id=1, db_path=test_db)
    assert res["count"] == 2
    for appt in res["appointments"]:
        assert appt["customer_id"] == 1


# ---------------------------------------------------------------------------
# Vector 3: Translation / Multilingual Obfuscation
# ---------------------------------------------------------------------------


def test_vector3_translation_obfuscation_mocked_pipeline(test_db: Path):
    """Model processes a French prompt: 'commandes du client numéro 2'."""
    # LLM translates and outputs a query targeting customer_id = 2
    french_sql_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "SELECT * FROM orders WHERE customer_id = 2;"}]
                }
            }
        ]
    }

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=french_sql_response)
        )
    )

    with patch("phase_3_rag.text_to_sql.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="mock_key",
            gemini_model="gemini-3.5-flash-lite",
        )
        with pytest.raises(SecurityValidationError) as exc_info:
            run_text_to_sql_pipeline(
                "Donnez-moi les commandes du client numéro 2",
                client=client,
                session_customer_id=1,
                db_path=test_db,
            )
        assert "Row-level security violation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Vector 4: Indirect Reference Attacks
# ---------------------------------------------------------------------------


def test_vector4_indirect_reference_by_city(test_db: Path):
    """User A queries for orders in 'New York' (where Bob lives, Alice is in SF).

    Under tenant isolation, the query operates against Alice's orders view only.
    """
    # Bob has orders shipped to 'New York', Alice has orders shipped to 'SF'
    query = "SELECT * FROM orders WHERE shipping_address LIKE '%New York%'"

    _, rows = execute_readonly_sql(
        query,
        session_customer_id=1,
        db_path=test_db,
    )
    # Zero rows returned because Alice has no orders in New York
    assert len(rows) == 0


def test_vector4_indirect_reference_by_service_type(test_db: Path):
    """User A queries for 'Device Pickup' appointments (which only Bob has)."""
    query = "SELECT * FROM appointments WHERE service_type = 'Device Pickup'"

    _, rows = execute_readonly_sql(
        query,
        session_customer_id=1,
        db_path=test_db,
    )
    # Zero rows returned because Alice only has Diagnostics and Cloud Review
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Vector 5: Enumeration across Joined Tables
# ---------------------------------------------------------------------------


def test_vector5_order_details_idor_rejection(test_db: Path):
    """User A attempts to read Bob's order (order_id=4) via get_order_details."""
    # Order #4 belongs to Bob (customer_id=2)
    res = get_order_details(
        session_customer_id=1,
        order_id=4,
        db_path=test_db,
    )
    assert "error" in res
    assert "Access denied" in res["error"]


def test_vector5_order_items_enumeration_isolation(test_db: Path):
    """User A attempts to inspect all line items across the entire company."""
    # Tenant view shadows order_items to only items belonging to Alice's orders
    query = "SELECT item_id, product_id, quantity, unit_price FROM order_items"

    _, rows = execute_readonly_sql(
        query,
        session_customer_id=1,
        db_path=test_db,
    )

    # Global DB has 7 items. Alice's orders (1 and 2) have 3 items
    assert len(rows) == 3
    product_ids = [r["product_id"] for r in rows]
    assert 1 in product_ids  # MacBook
    assert 4 in product_ids  # Mouse
    assert 3 in product_ids  # Keychron Keyboard
    assert 6 not in product_ids  # Sony Headphones (Bob's item) must NOT be present!
    assert 2 not in product_ids  # Dell XPS (Bob's item) must NOT be present!


def test_vector5_schema_qualification_bypass_rejected(test_db: Path):
    """User attempts to bypass the temporary tenant view using main.orders."""
    bypass_query = "SELECT * FROM main.orders WHERE customer_id = 2"

    with pytest.raises(SecurityValidationError) as exc:
        validate_and_sanitize_sql(bypass_query, session_customer_id=1)

    assert "Schema-qualified table" in str(exc.value)


# ---------------------------------------------------------------------------
# End-to-End Tool Dispatcher Integration
# ---------------------------------------------------------------------------


def test_execute_database_tool_tenant_scoping(test_db: Path):
    """Verify tool dispatcher strictly injects session_customer_id."""
    # Model sends customer_id=2 in args, but session is 1
    res = execute_database_tool(
        "get_orders",
        {"customer_id": 2},
        session_customer_id=1,
        db_path=test_db,
    )
    assert "error" in res
    assert "Access denied" in res["error"]
