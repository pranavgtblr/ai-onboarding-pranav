"""Unit and integration tests for Task 3.14: Secure Text-to-SQL Engine."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from phase_3_rag.database import get_readonly_connection, init_database
from phase_3_rag.text_to_sql import (
    DEFAULT_MAX_LIMIT,
    SecurityValidationError,
    execute_readonly_sql,
    run_text_to_sql_pipeline,
    validate_and_sanitize_sql,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Initialize a dedicated SQLite test database."""
    db_file = tmp_path / "test_ecommerce.db"
    return init_database(db_file)


def test_schema_initialization_and_seeding(test_db: Path):
    """Verify database creates tables and populates sample transactional data."""
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM customers")
    assert cursor.fetchone()[0] == 4

    cursor.execute("SELECT COUNT(*) FROM products")
    assert cursor.fetchone()[0] == 6

    cursor.execute("SELECT COUNT(*) FROM orders")
    assert cursor.fetchone()[0] == 5

    cursor.execute("SELECT COUNT(*) FROM appointments")
    assert cursor.fetchone()[0] == 5

    cursor.execute("SELECT COUNT(*) FROM sensitive_admin_logs")
    assert cursor.fetchone()[0] == 2
    conn.close()


def test_validate_sql_allows_valid_select():
    """Verify standard SELECT queries on allowlisted tables pass validation."""
    query = "SELECT name, email FROM customers WHERE city = 'San Francisco'"
    sanitized = validate_and_sanitize_sql(query)
    assert "SELECT" in sanitized
    assert "customers" in sanitized
    assert f"LIMIT {DEFAULT_MAX_LIMIT}" in sanitized


def test_validate_sql_injects_mandatory_limit():
    """Verify queries missing a LIMIT clause have LIMIT injected automatically."""
    query = "SELECT order_id, total_amount FROM orders WHERE status = 'delivered'"
    sanitized = validate_and_sanitize_sql(query, max_limit=25)
    assert "LIMIT 25" in sanitized


def test_validate_sql_clamps_excessive_limit():
    """Verify queries requesting more than MAX_LIMIT are clamped down."""
    query = "SELECT * FROM products LIMIT 500"
    sanitized = validate_and_sanitize_sql(query, max_limit=50)
    assert "LIMIT 50" in sanitized
    assert "500" not in sanitized


def test_validate_sql_allows_reasonable_limit():
    """Verify queries requesting a smaller limit retain their limit."""
    query = "SELECT * FROM products LIMIT 5"
    sanitized = validate_and_sanitize_sql(query, max_limit=50)
    assert "LIMIT 5" in sanitized


@pytest.mark.parametrize(
    "forbidden_query",
    [
        "INSERT INTO customers (name, email) VALUES ('Hacker', 'h@bad.com')",
        "UPDATE orders SET status = 'delivered' WHERE order_id = 1",
        "DELETE FROM appointments WHERE appointment_id = 1",
        "DROP TABLE customers",
        "ALTER TABLE orders ADD COLUMN hack TEXT",
        "CREATE TABLE backdoor (id INT)",
        "PRAGMA table_info(customers)",
    ],
)
def test_validate_sql_blocks_write_and_schema_modifications(forbidden_query: str):
    """Verify all mutating statements and PRAGMA commands are strictly rejected."""
    with pytest.raises(SecurityValidationError) as exc_info:
        validate_and_sanitize_sql(forbidden_query)
    assert "Only SELECT queries are permitted" in str(
        exc_info.value
    ) or "Prohibited SQL expression detected" in str(exc_info.value)


def test_validate_sql_blocks_non_allowlisted_tables():
    """Verify queries targeting non-allowlisted or system tables are rejected."""
    with pytest.raises(SecurityValidationError) as exc_info1:
        validate_and_sanitize_sql("SELECT * FROM sensitive_admin_logs")
    assert "non-allowlisted table" in str(exc_info1.value)

    with pytest.raises(SecurityValidationError) as exc_info2:
        validate_and_sanitize_sql("SELECT name FROM sqlite_master WHERE type='table'")
    assert "non-allowlisted table" in str(exc_info2.value)


def test_validate_sql_blocks_multi_statement_injection():
    """Verify semicolon statement chaining is detected and rejected."""
    attack = "SELECT * FROM orders; DROP TABLE customers;"
    with pytest.raises(SecurityValidationError) as exc_info:
        validate_and_sanitize_sql(attack)
    assert "Multi-statement execution rejected" in str(exc_info.value)


def test_readonly_connection_blocks_modifications_at_sqlite_engine_level(
    test_db: Path,
):
    """Verify SQLite engine blocks writes in mode=ro if validation was bypassed."""
    conn = get_readonly_connection(test_db)
    cursor = conn.cursor()

    # Attempt raw insert on read-only connection
    with pytest.raises(sqlite3.OperationalError) as exc_info:
        cursor.execute(
            "INSERT INTO customers (name, email, phone, city) "
            "VALUES ('X', 'x@test.com', '123', 'SF')"
        )
    assert "readonly" in str(exc_info.value).lower()
    conn.close()


def test_execute_readonly_sql_retrieves_live_records(test_db: Path):
    """Verify read-only execution extracts correct records and formats rows."""
    query = """
    SELECT c.name, a.service_type, a.scheduled_time, a.status
    FROM customers c
    JOIN appointments a ON c.customer_id = a.customer_id
    WHERE c.name LIKE '%Alice%'
    ORDER BY a.scheduled_time ASC
    """
    sanitized, rows = execute_readonly_sql(query, db_path=test_db)

    assert "LIMIT" in sanitized
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice Johnson"
    assert rows[0]["service_type"] == "Hardware Diagnostics"
    assert rows[0]["status"] == "confirmed"


def test_end_to_end_mocked_llm_pipeline(test_db: Path):
    """Verify end-to-end pipeline: LLM generates SQL, query executes, answer."""
    mock_sql_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "```sql\n"
                                "SELECT scheduled_time, service_type, status "
                                "FROM appointments "
                                "JOIN customers ON appointments.customer_id = "
                                "customers.customer_id "
                                "WHERE customers.name LIKE '%Alice%' "
                                "ORDER BY scheduled_time ASC LIMIT 1;\n"
                                "```"
                            )
                        }
                    ]
                }
            }
        ]
    }

    mock_answer_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Alice Johnson's next appointment is for "
                                "Hardware Diagnostics scheduled on 2026-09-12 "
                                "at 11:00:00 (Status: Confirmed)."
                            )
                        }
                    ]
                }
            }
        ]
    }

    def mock_post_handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        if "Generate the exact SQLite SQL query" in body:
            return httpx.Response(200, json=mock_sql_response)
        return httpx.Response(200, json=mock_answer_response)

    client = httpx.Client(transport=httpx.MockTransport(mock_post_handler))

    with patch("phase_3_rag.text_to_sql.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(gemini_api_key="mock_key_123")
        result = run_text_to_sql_pipeline(
            "What is Alice's next appointment?", client=client, db_path=test_db
        )

    assert result.row_count == 1
    assert "Hardware Diagnostics" in result.answer
    assert "LIMIT 1" in result.sanitized_sql
