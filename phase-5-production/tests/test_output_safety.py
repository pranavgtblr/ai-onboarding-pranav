"""Tests for Task 5.6: Output safety (HTML escaping & SQL parameterization)."""

from __future__ import annotations

import sqlite3

import pytest
from phase_4_agents.rag_tools import db_query

from phase_5_production.output_safety import (
    SafeQueryExecutor,
    UnparameterizedQueryError,
    escape_html_output,
    render_model_output_page,
    verify_html_safety,
)

# =====================================================================
# PART 1: HTML Escaping & XSS Neutralization Tests
# =====================================================================


def test_escape_html_output_neutralizes_special_characters() -> None:
    """Special characters (&, <, >, ", ') must be converted to HTML entities."""
    raw = '<script>alert("test & verify")</script>'
    escaped = escape_html_output(raw)

    assert "<script>" not in escaped
    assert "</script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&quot;test &amp; verify&quot;" in escaped


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert('XSS')</script>",
        '<img src="x" onerror="alert(document.cookie)">',
        "<svg onload=alert(document.domain)>",
        '<iframe src="javascript:alert(1)"></iframe>',
        '<a href="javascript:alert(1)">Click Me</a>',
        '"><script>fetch("http://evil.com")</script>',
        '<object data="javascript:alert(1)"></object>',
    ],
)
def test_render_model_output_page_escapes_xss_vectors(payload: str) -> None:
    """Model output rendered in safe mode must pass HTML safety verification."""
    safe_page = render_model_output_page(payload, is_safe=True)

    # 1. Statically verify raw script tags do not exist
    assert payload not in safe_page

    # 2. Verify HTML safety parser detects 0 executable tags or handlers
    is_safe, violations = verify_html_safety(safe_page)
    assert is_safe is True
    assert violations == []

    # 3. Contrast with unsafe rendering: unsafe mode MUST detect violations
    vuln_page = render_model_output_page(payload, is_safe=False)
    is_vuln_safe, vuln_violations = verify_html_safety(vuln_page)
    assert is_vuln_safe is False
    assert len(vuln_violations) > 0


def test_render_page_contains_csp_header() -> None:
    """Safe page must enforce Content Security Policy prohibiting inline scripts."""
    page = render_model_output_page("Safe model output.", is_safe=True)
    assert "Content-Security-Policy" in page
    assert "script-src 'none'" in page


# =====================================================================
# PART 2: SQL Parameterization & Injection Barrier Tests
# =====================================================================


@pytest.fixture
def mock_db() -> sqlite3.Connection:
    """Creates an in-memory SQLite database populated with test records."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT NOT NULL
        );
        """
    )
    cursor.executemany(
        "INSERT INTO customers (name, email, city) VALUES (?, ?, ?);",
        [
            ("Alice Smith", "alice@example.com", "London"),
            ("Bob Jones", "bob@example.com", "Paris"),
            ("Charlie Brown", "charlie@example.com", "Berlin"),
        ],
    )
    conn.commit()
    return conn


def test_safe_query_executor_parameterized_lookup(
    mock_db: sqlite3.Connection,
) -> None:
    """Parameterized queries return matching rows without syntax tampering."""
    executor = SafeQueryExecutor(mock_db)
    results = executor.execute_parameterized(
        "SELECT * FROM customers WHERE city = ?;",
        ("London",),
    )
    assert len(results) == 1
    assert results[0]["name"] == "Alice Smith"


def test_sql_injection_tautology_neutralized(
    mock_db: sqlite3.Connection,
) -> None:
    """Tautology payloads (' OR '1'='1') must be bound literally, not evaluated."""
    executor = SafeQueryExecutor(mock_db)
    injection_payload = "London' OR '1'='1' --"

    # In safe parameterized execution: payload treated as literal city name
    safe_results = executor.execute_parameterized(
        "SELECT * FROM customers WHERE city = ?;",
        (injection_payload,),
    )
    assert len(safe_results) == 0  # No customer lives in city "London' OR '1'='1' --"

    # Contrast with raw unparameterized f-string query:
    unsafe_sql = f"SELECT * FROM customers WHERE city = '{injection_payload}';"
    unsafe_results = executor.execute_raw_unsafe(unsafe_sql)
    assert len(unsafe_results) == 3  # Raw injection leaked all 3 customer rows!


def test_sql_injection_drop_table_neutralized(
    mock_db: sqlite3.Connection,
) -> None:
    """Stacked statement payloads ('; DROP TABLE...') cannot alter or drop tables."""
    executor = SafeQueryExecutor(mock_db)
    malicious_input = "London'; DROP TABLE customers; --"

    # Safe parameterized query treats the entire input as a single parameter literal
    safe_results = executor.execute_parameterized(
        "SELECT * FROM customers WHERE city = ?;",
        (malicious_input,),
    )
    assert len(safe_results) == 0

    # Verify table remains intact
    cursor = mock_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers;")
    count = cursor.fetchone()[0]
    assert count == 3


def test_safe_query_executor_blocks_multi_statement(
    mock_db: sqlite3.Connection,
) -> None:
    """Attempting multi-statement execution via template must raise an error."""
    executor = SafeQueryExecutor(mock_db)
    with pytest.raises(UnparameterizedQueryError, match="Multi-statement"):
        executor.execute_parameterized(
            "SELECT * FROM customers; DROP TABLE customers;",
            (),
        )


def test_safe_query_executor_blocks_destructive_ddl(
    mock_db: sqlite3.Connection,
) -> None:
    """Attempting destructive DDL through query executor must be blocked."""
    executor = SafeQueryExecutor(mock_db)
    with pytest.raises(UnparameterizedQueryError, match="Destructive DDL"):
        executor.execute_parameterized("DROP TABLE customers;", ())


def test_rag_tools_db_query_resists_injection_payload() -> None:
    """Verify db_query tool safely parameterizes model-derived city input."""
    injection_prompt = "find customers in London' OR '1'='1"
    output = db_query.invoke({"query": injection_prompt})

    # The query must execute parameterized and find 0 records
    assert "Rows Returned: 0" in output
    assert "No matching records found" in output
    assert "Security Error" not in output
